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

"""Tests for composable shaping-flap packing templates."""

import copy
import functools
import json
import time
from typing import Any
from unittest import mock

from absl.testing import absltest
import bp_pipeline as bp
from bp_pipeline.packing import config as packing_config_lib
from bp_pipeline.packing import scoring
from bp_pipeline.packing import shaping_flap_templates as sft
from bp_pipeline.packing import types as packing_types
from bp_pipeline.packing.debug import debug_types
from bp_pipeline.shaping import recipes
from bp_pipeline.shaping import state as state_lib
from bp_pipeline.shaping import toolbox
from bp_pipeline.solving import hinges
from bp_pipeline.stick_figures import graph_checks
from bp_pipeline.stick_figures import stick_figure as sf_lib
from bp_pipeline.stick_figures.examples import example_trees


def _tipped_tail_animal_dict() -> dict[str, Any]:
  """Symmetric animal with an ordinary neck chain + central `TailTemplate`."""
  return {
      'name': 'tipped_tail_animal',
      'category': 'animal',
      'complexity': 2,
      'root': 'body',
      'children': [
          {
              'name': 'body',
              'start': 'j_shoulder',
              'end': 'j_hip',
              'length': 2,
              'width': 1,
              'azimuth_angle': 180.0,
              'elevation_angle': 0.0,
              'shaping_flap': False,
          },
          {
              'name': 'neck_base',
              'start': 'j_shoulder',
              'end': 'j_neck_mid',
              'length': 1,
              'width': 1,
              'azimuth_angle': 0.0,
              'elevation_angle': 25.0,
              'shaping_flap': False,
          },
          {
              'name': 'head',
              'start': 'j_neck_mid',
              'end': 'j_head',
              'length': 1,
              'width': 1,
              'azimuth_angle': 0.0,
              'elevation_angle': 35.0,
              'shaping_flap': False,
          },
          {
              'name': 'arm_l',
              'start': 'j_shoulder',
              'end': 'j_al',
              'length': 2,
              'width': 1,
              'azimuth_angle': 75.0,
              'elevation_angle': -55.0,
              'shaping_flap': False,
          },
          {
              'name': 'arm_r',
              'start': 'j_shoulder',
              'end': 'j_ar',
              'length': 2,
              'width': 1,
              'azimuth_angle': -75.0,
              'elevation_angle': -55.0,
              'shaping_flap': False,
          },
          {
              'name': 'leg_l',
              'start': 'j_hip',
              'end': 'j_ll',
              'length': 2,
              'width': 1,
              'azimuth_angle': 105.0,
              'elevation_angle': -60.0,
              'shaping_flap': False,
          },
          {
              'name': 'leg_r',
              'start': 'j_hip',
              'end': 'j_lr',
              'length': 2,
              'width': 1,
              'azimuth_angle': -105.0,
              'elevation_angle': -60.0,
              'shaping_flap': False,
          },
          {
              'name': 'tail_r1',
              'start': 'j_hip',
              'end': 'j_t1',
              'length': 1,
              'width': 1,
              'azimuth_angle': 180.0,
              'elevation_angle': 15.0,
              'shaping_flap': False,
          },
          {
              'name': 'tail_sf1',
              'start': 'j_t1',
              'end': 'j_sf1',
              'length': 1,
              'width': 1,
              'azimuth_angle': 180.0,
              'elevation_angle': 75.0,
              'shaping_flap': True,
          },
          {
              'name': 'tail_r0',
              'start': 'j_t1',
              'end': 'j_t0',
              'length': 1,
              'width': 1,
              'azimuth_angle': 180.0,
              'elevation_angle': 25.0,
              'shaping_flap': False,
          },
          {
              'name': 'tail_sf0',
              'start': 'j_t0',
              'end': 'j_sf0',
              'length': 1,
              'width': 1,
              'azimuth_angle': 180.0,
              'elevation_angle': 75.0,
              'shaping_flap': True,
          },
          {
              'name': 'tail_tip',
              'start': 'j_t0',
              'end': 'j_tip',
              'length': 2,
              'width': 1,
              'azimuth_angle': 180.0,
              'elevation_angle': 30.0,
              'shaping_flap': False,
          },
      ],
  }


def _composable_knee_and_tail_dict() -> dict[str, Any]:
  """Symmetric creature combining bilateral `KneeTemplate`s + `TailTemplate`."""
  return {
      'name': 'composable_knee_tail',
      'category': 'animal',
      'complexity': 3,
      'root': 'body',
      'children': [
          {
              'name': 'body',
              'start': 'j_back',
              'end': 'j_front',
              'length': 2,
              'width': 1,
              'azimuth_angle': 0.0,
              'elevation_angle': 10.0,
              'shaping_flap': False,
          },
          {
              'name': 'neck',
              'start': 'j_front',
              'end': 'j_neck',
              'length': 2,
              'width': 1,
              'azimuth_angle': 0.0,
              'elevation_angle': 20.0,
              'shaping_flap': False,
          },
          {
              'name': 'head',
              'start': 'j_neck',
              'end': 'j_head',
              'length': 2,
              'width': 1,
              'azimuth_angle': 0.0,
              'elevation_angle': 25.0,
              'shaping_flap': False,
          },
          {
              'name': 'thigh_l',
              'start': 'j_front',
              'end': 'j_kl',
              'length': 2,
              'width': 1,
              'azimuth_angle': 65.0,
              'elevation_angle': -35.0,
              'shaping_flap': False,
          },
          {
              'name': 'shin_l',
              'start': 'j_kl',
              'end': 'j_fl',
              'length': 2,
              'width': 1,
              'azimuth_angle': 55.0,
              'elevation_angle': -75.0,
              'shaping_flap': False,
          },
          {
              'name': 'knee_cap_l',
              'start': 'j_kl',
              'end': 'j_kcl',
              'length': 1,
              'width': 1,
              'azimuth_angle': 45.0,
              'elevation_angle': -15.0,
              'shaping_flap': True,
          },
          {
              'name': 'thigh_r',
              'start': 'j_front',
              'end': 'j_kr',
              'length': 2,
              'width': 1,
              'azimuth_angle': -65.0,
              'elevation_angle': -35.0,
              'shaping_flap': False,
          },
          {
              'name': 'shin_r',
              'start': 'j_kr',
              'end': 'j_fr',
              'length': 2,
              'width': 1,
              'azimuth_angle': -55.0,
              'elevation_angle': -75.0,
              'shaping_flap': False,
          },
          {
              'name': 'knee_cap_r',
              'start': 'j_kr',
              'end': 'j_kcr',
              'length': 1,
              'width': 1,
              'azimuth_angle': -45.0,
              'elevation_angle': -15.0,
              'shaping_flap': True,
          },
          {
              'name': 'leg_l',
              'start': 'j_back',
              'end': 'j_ll',
              'length': 4,
              'width': 1,
              'azimuth_angle': 115.0,
              'elevation_angle': -65.0,
              'shaping_flap': False,
          },
          {
              'name': 'leg_r',
              'start': 'j_back',
              'end': 'j_lr',
              'length': 4,
              'width': 1,
              'azimuth_angle': -115.0,
              'elevation_angle': -65.0,
              'shaping_flap': False,
          },
          {
              'name': 'tail_r1',
              'start': 'j_back',
              'end': 'j_t1',
              'length': 2,
              'width': 1,
              'azimuth_angle': 180.0,
              'elevation_angle': 15.0,
              'shaping_flap': False,
          },
          {
              'name': 'tail_sf1',
              'start': 'j_t1',
              'end': 'j_sf1',
              'length': 2,
              'width': 1,
              'azimuth_angle': 180.0,
              'elevation_angle': 60.0,
              'shaping_flap': True,
          },
          {
              'name': 'tail_r0',
              'start': 'j_t1',
              'end': 'j_t0',
              'length': 2,
              'width': 1,
              'azimuth_angle': 180.0,
              'elevation_angle': 35.0,
              'shaping_flap': False,
          },
          {
              'name': 'tail_sf0',
              'start': 'j_t0',
              'end': 'j_sf0',
              'length': 2,
              'width': 1,
              'azimuth_angle': 180.0,
              'elevation_angle': 70.0,
              'shaping_flap': True,
          },
      ],
  }


@functools.lru_cache(maxsize=1)
def _solved_composable_knee_and_tail() -> bp.SolvedCreasePattern:
  design = bp.prepare(_composable_knee_and_tail_dict())
  packed = bp.pack(
      design,
      bp.PackingTaskConfig(
          packer='builtin',
          max_retries=6,
          max_cps_per_grid_size=2,
          max_number_of_cps=2,
      ),
  )
  return bp.solve_best(
      packed,
      bp.SolverConfig(
          check_global_flat_foldability=True,
          max_number_of_cps=1,
          max_time_limit=20.0,
      ),
  )


def _assert_globally_flat_foldable(
    testcase: absltest.TestCase, solved: bp.SolvedCreasePattern
) -> None:
  testcase.assertTrue(
      hinges.valid_crease_pattern(
          solved.creases,
          solved.grid_size,
          local_flat_foldability=True,
          global_flat_foldability=True,
      )
  )
  folded_dict = solved.flat_fold()
  testcase.assertNotEmpty(folded_dict.get('faces_vertices', ()))


class ShapingFlapTemplatesTest(absltest.TestCase):

  def test_merge_linear_chains_merges_ordinary_chain_and_preserves_template(
      self,
  ):
    raw = sf_lib.StickFigure.from_dict(_composable_knee_and_tail_dict())
    merged, merged_map = raw.merge_linear_chains()
    merged_names = {s.name for s in merged.children}
    # Ordinary degree-2 chain `neck` + `head` merges into `neck_head`, whereas
    # degree-2 template joint `j_t0` (`tail_r0` + `tail_sf0`) stays unmerged.
    self.assertIn('neck_head', merged_names)
    self.assertEqual(merged_map['neck_head'], ['neck', 'head'])
    self.assertIn('tail_r0', merged_names)
    self.assertIn('tail_sf0', merged_names)
    self.assertNotIn('tail_r0_tail_sf0', merged_names)

    design = bp.prepare(_composable_knee_and_tail_dict())
    self.assertEqual({s.name for s in design.merged.children}, merged_names)

    # Explicit opt-out (`ignore_shaping_flaps=False`) merges across `j_t0` too.
    collapsed, _ = raw.merge_linear_chains(ignore_shaping_flaps=False)
    self.assertIn(
        'tail_r0_tail_sf0', {s.name for s in collapsed.children}
    )

  def test_prevalidate_and_packer_reject_invalid_or_excess_templates(self):
    valid = _composable_knee_and_tail_dict()
    self.assertEmpty(
        graph_checks.prevalidate(
            valid, check_symmetry=True, use_shaping_flaps=True
        )
    )

    orphan = _tipped_tail_animal_dict()
    orphan['children'].append({
        'name': 'bad_hub_sf',
        'start': 'j_shoulder',
        'end': 'j_bad',
        'length': 1,
        'width': 1,
        'azimuth_angle': 0.0,
        'elevation_angle': 80.0,
        'shaping_flap': True,
    })
    errs = graph_checks.prevalidate(
        orphan, check_symmetry=True, use_shaping_flaps=True
    )
    self.assertTrue(any('bad_hub_sf' in e for e in errs), msg=str(errs))
    with self.assertRaisesRegex(bp.PackingFailedError, 'bad_hub_sf'):
      bp.pack(bp.prepare(orphan), packer='builtin')

    # Exceeding 2 symmetric KneeTemplate pairs raises a clear limit error.
    three_knee_pairs = _composable_knee_and_tail_dict()
    for side, sign in (('l', 1.0), ('r', -1.0)):
      for idx in (2, 3):
        three_knee_pairs['children'].extend([
            {
                'name': f'extra_arm{idx}_{side}',
                'start': 'j_back',
                'end': f'j_ek{idx}_{side}',
                'length': 1,
                'width': 1,
                'azimuth_angle': sign * (80.0 + 10.0 * idx),
                'elevation_angle': -20.0,
                'shaping_flap': False,
            },
            {
                'name': f'extra_hand{idx}_{side}',
                'start': f'j_ek{idx}_{side}',
                'end': f'j_eh{idx}_{side}',
                'length': 1,
                'width': 1,
                'azimuth_angle': sign * (80.0 + 10.0 * idx),
                'elevation_angle': -50.0,
                'shaping_flap': False,
            },
            {
                'name': f'extra_cap{idx}_{side}',
                'start': f'j_ek{idx}_{side}',
                'end': f'j_ec{idx}_{side}',
                'length': 1,
                'width': 1,
                'azimuth_angle': sign * (70.0 + 10.0 * idx),
                'elevation_angle': 0.0,
                'shaping_flap': True,
            },
        ])
    limit_errs = graph_checks.prevalidate(
        three_knee_pairs, check_symmetry=True, use_shaping_flaps=True
    )
    self.assertTrue(
        any('At most 2 mirrored KneeTemplate pairs' in e for e in limit_errs),
        msg=str(limit_errs),
    )
    with self.assertRaisesRegex(
        bp.PackingFailedError, 'At most 2 mirrored KneeTemplate pairs'
    ):
      bp.pack(bp.prepare(three_knee_pairs), packer='builtin')

    # easy_mode=True is rejected explicitly rather than falling back silently.
    with self.assertRaisesRegex(bp.PackingFailedError, 'easy_mode'):
      bp.pack(
          bp.prepare(_tipped_tail_animal_dict()),
          bp.PackingTaskConfig(
              packer='builtin',
              packing_config=packing_config_lib.PackingConfig(
                  easy_mode=True,
                  start_with_hugging_river=True,
                  traversal_mode=packing_types.TraversalMode.PROXIMITY,
              ),
          ),
      )

  def test_dual_tails_asymmetric_limits_and_solver_guards(self):
    # Dual TailTemplates (4x4 each) + 2 mirrored KneeTemplate pairs (2x4 each
    # in Tall orientation) tile 8x8 exactly with zero remaining rivers/flaps.
    dual_tails: dict[str, Any] = {
        'name': 'dual_tail_beetle',
        'category': 'animal',
        'complexity': 2,
        'root': 'tail_r0',
        'children': [],
    }
    for prefix, az_base in (('k1', 60.0), ('k2', 120.0)):
      for side, sign in (('l', 1.0), ('r', -1.0)):
        dual_tails['children'].extend([
            {
                'name': f'{prefix}_arm_{side}',
                'start': 'j_hub',
                'end': f'j_{prefix}_{side}',
                'length': 1,
                'width': 1,
                'azimuth_angle': sign * az_base,
                'elevation_angle': -20.0,
                'shaping_flap': False,
            },
            {
                'name': f'{prefix}_tip_{side}',
                'start': f'j_{prefix}_{side}',
                'end': f'j_{prefix}_tip_{side}',
                'length': 1,
                'width': 1,
                'azimuth_angle': sign * az_base,
                'elevation_angle': -50.0,
                'shaping_flap': False,
            },
            {
                'name': f'{prefix}_cap_{side}',
                'start': f'j_{prefix}_{side}',
                'end': f'j_{prefix}_cap_{side}',
                'length': 1,
                'width': 1,
                'azimuth_angle': sign * (az_base - 10.0),
                'elevation_angle': 30.0,
                'shaping_flap': True,
            },
        ])
    dual_tails['children'].extend([
        {
            'name': 'tail_r0',
            'start': 'j_hub',
            'end': 'j_t0',
            'length': 1,
            'width': 1,
            'azimuth_angle': 180.0,
            'elevation_angle': 15.0,
            'shaping_flap': False,
        },
        {
            'name': 'tail_tip',
            'start': 'j_t0',
            'end': 'j_ttip',
            'length': 1,
            'width': 1,
            'azimuth_angle': 180.0,
            'elevation_angle': 5.0,
            'shaping_flap': False,
        },
        {
            'name': 'tail_sf0',
            'start': 'j_t0',
            'end': 'j_tsf0',
            'length': 1,
            'width': 1,
            'azimuth_angle': 180.0,
            'elevation_angle': 75.0,
            'shaping_flap': True,
        },
        {
            'name': 'crest_r0',
            'start': 'j_hub',
            'end': 'j_cr0',
            'length': 1,
            'width': 1,
            'azimuth_angle': 0.0,
            'elevation_angle': 25.0,
            'shaping_flap': False,
        },
        {
            'name': 'crest_tip',
            'start': 'j_cr0',
            'end': 'j_ctip',
            'length': 1,
            'width': 1,
            'azimuth_angle': 0.0,
            'elevation_angle': 10.0,
            'shaping_flap': False,
        },
        {
            'name': 'crest_sf0',
            'start': 'j_cr0',
            'end': 'j_csf0',
            'length': 1,
            'width': 1,
            'azimuth_angle': 0.0,
            'elevation_angle': 80.0,
            'shaping_flap': True,
        },
    ])
    dual_design = bp.prepare(dual_tails)
    tails, claimed = sft.find_tail_templates(dual_design.source)
    knees = sft.find_knee_templates(dual_design.source, exclude_sticks=claimed)
    self.assertLen(tails, 2)
    self.assertLen(knees, 4)
    for orient in ('Wide', 'Tall'):
      inner_w, inner_h = knees[0].inner_size(orient)
      total_w, total_h = knees[0].total_size(orient)
      expected_river_cells = total_w * total_h - inner_w * inner_h
      for corner in ('BottomLeft', 'BottomRight', 'TopLeft', 'TopRight'):
        _, corner_river = knees[0].pack_polygons(orient, corner, grid_size=8)
        river_cells = sum(
            rect.x_len * rect.y_len for rect in corner_river.rectangles
        )
        self.assertEqual(
            river_cells, expected_river_cells, msg=f'{orient}/{corner}'
        )
    dual_trackers = sft.solve_trackers_with_shaping_templates(
        dual_design.merged,
        dual_design.merged,
        grid_size=8,
        is_symmetric=True,
        symmetry_axes=None,
        pockets_info=None,
        default_traversal_plan=[],
        debug=debug_types.DebugConfig(),
        scoring_config=scoring.ScoringConfig(),
        max_solutions=1,
        config=packing_config_lib.PackingConfig(),
    )
    self.assertNotEmpty(dual_trackers)
    self.assertIn('tail_r0', dual_trackers[0].packed_rivers)
    self.assertIn('crest_r0', dual_trackers[0].packed_rivers)

    # Odd grid_size on symmetric figure returns empty tracker list cleanly, and
    # a tree without templates raises ValueError in solve_trackers_with_...
    odd_res = sft.solve_trackers_with_shaping_templates(
        dual_design.merged,
        dual_design.merged,
        grid_size=15,
        is_symmetric=True,
        symmetry_axes=None,
        pockets_info=None,
        default_traversal_plan=[],
        debug=debug_types.DebugConfig(),
        scoring_config=scoring.ScoringConfig(),
        max_solutions=1,
        config=packing_config_lib.PackingConfig(),
    )
    self.assertEmpty(odd_res)

    plain = bp.prepare(bp.load_example('cat'))
    with self.assertRaisesRegex(ValueError, 'do not form any valid'):
      sft.solve_trackers_with_shaping_templates(
          plain.merged,
          plain.merged,
          grid_size=16,
          is_symmetric=True,
          symmetry_axes=None,
          pockets_info=None,
          default_traversal_plan=[],
          debug=debug_types.DebugConfig(),
          scoring_config=scoring.ScoringConfig(),
          max_solutions=1,
          config=packing_config_lib.PackingConfig(),
      )

    # Exceeding 2 TailTemplates or 4 asymmetric KneeTemplates fails validation.
    three_tails = _tipped_tail_animal_dict()
    for idx in (2, 3):
      three_tails['children'].extend([
          {
              'name': f't{idx}_r1',
              'start': 'j_shoulder',
              'end': f'j_t{idx}_1',
              'length': 1,
              'width': 1,
              'azimuth_angle': 0.0,
              'elevation_angle': 10.0 * idx,
              'shaping_flap': False,
          },
          {
              'name': f't{idx}_sf1',
              'start': f'j_t{idx}_1',
              'end': f'j_t{idx}_sf1',
              'length': 1,
              'width': 1,
              'azimuth_angle': 0.0,
              'elevation_angle': 70.0,
              'shaping_flap': True,
          },
          {
              'name': f't{idx}_r0',
              'start': f'j_t{idx}_1',
              'end': f'j_t{idx}_0',
              'length': 1,
              'width': 1,
              'azimuth_angle': 0.0,
              'elevation_angle': 20.0 * idx,
              'shaping_flap': False,
          },
          {
              'name': f't{idx}_sf0',
              'start': f'j_t{idx}_0',
              'end': f'j_t{idx}_sf0',
              'length': 1,
              'width': 1,
              'azimuth_angle': 0.0,
              'elevation_angle': 75.0,
              'shaping_flap': True,
          },
      ])
    tail_errs = graph_checks.prevalidate(three_tails, use_shaping_flaps=True)
    self.assertTrue(
        any('At most 2 TailTemplates' in e for e in tail_errs),
        msg=str(tail_errs),
    )

    asym_five_knees: dict[str, Any] = {
        'name': 'asym_five_knees',
        'category': 'animal',
        'complexity': 2,
        'root': 'hub_arm_0',
        'children': [],
    }
    for idx in range(5):
      asym_five_knees['children'].extend([
          {
              'name': f'hub_arm_{idx}',
              'start': 'j_hub',
              'end': f'j_k{idx}',
              'length': idx + 1,
              'width': 1,
              'azimuth_angle': float(17 * idx),
              'elevation_angle': 10.0,
              'shaping_flap': False,
          },
          {
              'name': f'tip_{idx}',
              'start': f'j_k{idx}',
              'end': f'j_tip{idx}',
              'length': 1,
              'width': 1,
              'azimuth_angle': float(17 * idx),
              'elevation_angle': -30.0,
              'shaping_flap': False,
          },
          {
              'name': f'cap_{idx}',
              'start': f'j_k{idx}',
              'end': f'j_cap{idx}',
              'length': 1,
              'width': 1,
              'azimuth_angle': float(17 * idx + 5),
              'elevation_angle': 45.0,
              'shaping_flap': True,
          },
      ])
    asym_errs = graph_checks.prevalidate(
        asym_five_knees, use_shaping_flaps=True
    )
    self.assertTrue(
        any('At most 4 corner KneeTemplates' in e for e in asym_errs),
        msg=str(asym_errs),
    )

  def test_shipped_prompting_examples_pack_and_solve_and_unfit_fails_fast(self):
    examples = example_trees.get_examples_for_prompting(
        use_shaping_flaps=True, symmetry=True
    )
    self.assertLen(examples, 3)
    for ex_tree in examples:
      design = bp.prepare(copy.deepcopy(ex_tree))
      packed = bp.pack(design, packer='builtin')
      solved = bp.solve_best(
          packed,
          bp.SolverConfig(
              check_global_flat_foldability=True,
              max_number_of_cps=1,
              max_time_limit=15.0,
          ),
      )
      _assert_globally_flat_foldable(self, solved)

    # An un-packable template budget (where TailTemplate fills 16x16 leaving no
    # cells for the 4 body_top flaps) fails fast and names TailTemplate.
    tail_design = bp.prepare(copy.deepcopy(example_trees.TAIL_SIMPLE))
    t0 = time.perf_counter()
    with self.assertRaisesRegex(bp.PackingFailedError, 'TailTemplate'):
      bp.pack(
          tail_design,
          bp.PackingTaskConfig(
              packer='builtin',
              max_cps_per_grid_size=1,
              max_number_of_cps=1,
          ),
          packer_options=bp.PackerOptions(grid_size=16),
      )
    self.assertLess(time.perf_counter() - t0, 5.0)

  def test_estimate_template_min_grid_size_raises_starting_grid(self):
    # Deep 3-stage TailTemplate whose stacked height exceeds raw tree floor.
    deep_tail = _tipped_tail_animal_dict()
    deep_tail['children'] = [
        c for c in deep_tail['children'] if c['name'] != 'tail_tip'
    ]
    for s in deep_tail['children']:
      if s['name'] in ('tail_sf0', 'tail_sf1'):
        s['length'] = 4
    design = bp.prepare(deep_tail)
    raw_floor = design.merged.get_packing_gridsize()
    est = sft.estimate_template_min_grid_size(
        design.merged, design.is_symmetric
    )
    self.assertGreater(est, raw_floor)
    packed = bp.pack(
        design,
        bp.PackingTaskConfig(
            packer='builtin',
            max_retries=4,
            max_cps_per_grid_size=1,
            max_number_of_cps=1,
        ),
    )
    self.assertGreaterEqual(packed.grid_size, est)

  def test_tipped_tail_template_packs_and_solves_globally_flat_foldable(self):
    design = bp.prepare(_tipped_tail_animal_dict())
    tails, _ = sft.find_tail_templates(design.source)
    self.assertLen(tails, 1)
    self.assertIsNotNone(tails[0].tip_flap)
    assert tails[0].tip_flap is not None
    self.assertEqual(tails[0].tip_flap.name, 'tail_tip')

    packed = bp.pack(
        design,
        bp.PackingTaskConfig(
            packer='builtin',
            max_retries=6,
            max_cps_per_grid_size=1,
            max_number_of_cps=1,
        ),
    )
    solved = bp.solve_best(
        packed,
        bp.SolverConfig(
            check_global_flat_foldability=True,
            remove_central_crease=True,
            max_number_of_cps=1,
            max_time_limit=15.0,
        ),
    )
    self.assertTrue(solved.central_crease_removed)
    _assert_globally_flat_foldable(self, solved)

  def test_composable_knee_and_tail_templates_solve_globally_flat_foldable(
      self,
  ):
    design = bp.prepare(_composable_knee_and_tail_dict())
    tails, claimed = sft.find_tail_templates(design.source)
    knees = sft.find_knee_templates(design.source, exclude_sticks=claimed)
    self.assertLen(tails, 1)
    self.assertLen(knees, 2)
    self.assertEqual(knees[0].inner_size('Wide'), (4, 2))
    self.assertEqual(knees[0].total_size('Wide'), (6, 4))

    solved = _solved_composable_knee_and_tail()
    _assert_globally_flat_foldable(self, solved)

  def test_shape_flap_template_selective_armor_and_probe(self):
    state = bp.ShapingState.from_solved(_solved_composable_knee_and_tail())

    # Probe advertises shape_flap_template only on template shaping_flap sticks
    # (not ordinary sticks or orphan shaping_flap sticks).
    probed = bp.probe_shaping(state, ['knee_cap_l', 'tail_sf1', 'shin_l'])
    self.assertIn(
        'shape_flap_template', probed['knee_cap_l']['available_tools']
    )
    self.assertIn('shape_flap_template', probed['tail_sf1']['available_tools'])
    self.assertNotIn('shape_flap_template', probed['shin_l']['available_tools'])

    orphan_tree_dict = state.tree.to_dict()
    for st in orphan_tree_dict['children']:
      if st['name'] in ('leg_l', 'leg_r'):
        st['shaping_flap'] = True
    orphan_state = state.replace(
        tree=sf_lib.StickFigure.from_dict(orphan_tree_dict)
    )
    orphan_probed = bp.probe_shaping(orphan_state, ['leg_l', 'knee_cap_l'])
    self.assertNotIn(
        'shape_flap_template', orphan_probed['leg_l']['available_tools']
    )
    self.assertIn(
        'shape_flap_template', orphan_probed['knee_cap_l']['available_tools']
    )

    # Selective targeting on knee_cap_l mirrors to knee_cap_r, skips tail_sf*,
    # keeps `arguments` pure input kwargs, and leaves non-shaping limbs intact.
    shin_before = state.element('shin_l').creases
    sel_state, sel_res = bp.apply_shaping_tool(
        state,
        'shape_flap_template',
        {'element_name': 'knee_cap_l', 'style': 'armor', 'open_percent': 0.4},
    )
    self.assertTrue(sel_res.ok, msg=sel_res.message)
    self.assertEqual(
        set(sel_res.arguments),
        {'element_name', 'style', 'open_percent', 'verify_fold'},
    )
    self.assertIn('knee_cap_l=open_flap', sel_res.message)
    self.assertIn('knee_cap_r=open_flap', sel_res.message)
    self.assertIn('skipped: tail_sf0, tail_sf1', sel_res.message)
    self.assertEqual(sel_state.element('shin_l').creases, shin_before)

    # Group selection on a tail connecting river with `style='armor'` opens
    # every tail shaping flap while skipping knee templates.
    _, armor_tail_res = bp.apply_shaping_tool(
        state,
        'shape_flap_template',
        {
            'element_name': 'tail_r1',
            'style': 'armor',
            'open_percent': 0.5,
            'verify_fold': False,
        },
    )
    self.assertTrue(armor_tail_res.ok, msg=armor_tail_res.message)
    self.assertIn('tail_sf1=open_flap', armor_tail_res.message)
    self.assertIn('tail_sf0=open_flap', armor_tail_res.message)
    self.assertIn('skipped: knee_cap_l, knee_cap_r', armor_tail_res.message)

  def test_shape_flap_template_tuck_replay_and_pose_exclusions(self):
    state = bp.ShapingState.from_solved(_solved_composable_knee_and_tail())
    shin_before = state.element('shin_l').creases

    tuck_state, tuck_res = bp.apply_shaping_tool(
        state, 'shape_flap_template', {'style': 'tuck'}
    )
    self.assertTrue(tuck_res.ok, msg=tuck_res.message)
    for sf_name in ('tail_sf1', 'tail_sf0', 'knee_cap_l', 'knee_cap_r'):
      self.assertIn(f'{sf_name}=simple_fold', tuck_res.message)
    self.assertNotIn('skipped:', tuck_res.message)
    self.assertEqual(tuck_state.element('shin_l').creases, shin_before)
    self.assertLen(tuck_state.history, 1)

    replayed_state, replayed_results = bp.apply_shaping_tools(
        state, json.loads(toolbox.history_json(tuck_state))
    )
    self.assertTrue(all(r.ok for r in replayed_results))
    self.assertEqual(
        state_lib.crease_multiset(replayed_state.pattern),
        state_lib.crease_multiset(tuck_state.pattern),
    )

    # Default automatic `bp.pose` never invokes `shape_flap_template`, and
    # `measured_directions` excludes shaping flaps from `side_crossings` and
    # `direction_error_deg` while keeping raw `directions`.
    posed, _ = recipes.pose_state(
        state,
        bp.PoseConfig(mode='figure', feedback_rounds=0),
    )
    used_tools = [entry.get('tool') for entry in posed.history]
    self.assertNotIn('shape_flap_template', used_tools)
    rep, dirs = toolbox.measured_directions(posed)
    sf_names = {'knee_cap_l', 'knee_cap_r', 'tail_sf0', 'tail_sf1'}
    self.assertTrue(sf_names.issubset(dirs))
    self.assertEmpty(sf_names & set(rep['direction_error_deg']))
    crossing_names = {c['stick'] for c in rep.get('side_crossings', ())}
    self.assertEmpty(sf_names & crossing_names)

  def test_shape_flap_template_taper_and_validation_errors(self):
    state = bp.ShapingState.from_solved(_solved_composable_knee_and_tail())

    # `taper` on an open central spine uses narrow_central_flap when the
    # central blade spans `x = 0.5` with `length >= 2`, and falls back to
    # simple_fold(narrow=True) on length-1, split-spine, or off-axis blades.
    tip_dict = _tipped_tail_animal_dict()
    for st in tip_dict['children']:
      if st['name'] == 'tail_sf1':
        st['length'] = 2
    tip_design = bp.prepare(tip_dict)
    tip_packed = bp.pack(
        tip_design,
        bp.PackingTaskConfig(
            packer='builtin',
            max_retries=6,
            max_cps_per_grid_size=1,
            max_number_of_cps=1,
        ),
    )
    tip_solved = bp.solve_best(
        tip_packed,
        bp.SolverConfig(
            check_global_flat_foldability=True,
            remove_central_crease=True,
            max_number_of_cps=1,
            max_time_limit=15.0,
        ),
    )
    tip_state = bp.ShapingState.from_solved(tip_solved)
    _, taper_res = bp.apply_shaping_tool(
        tip_state, 'shape_flap_template', {'style': 'taper'}
    )
    self.assertTrue(taper_res.ok, msg=taper_res.message)
    self.assertIn('tail_sf1=narrow_central_flap', taper_res.message)
    self.assertIn('tail_sf0=simple_fold', taper_res.message)

    _, split_taper_res = bp.apply_shaping_tool(
        state,
        'shape_flap_template',
        {'element_name': 'tail_sf1', 'style': 'taper', 'verify_fold': False},
    )
    self.assertTrue(split_taper_res.ok, msg=split_taper_res.message)
    self.assertIn('tail_sf1=simple_fold', split_taper_res.message)

    with mock.patch.object(state_lib, 'element_spans_axis', return_value=False):
      _, off_axis_res = bp.apply_shaping_tool(
          tip_state,
          'shape_flap_template',
          {'element_name': 'tail_sf1', 'style': 'taper', 'verify_fold': False},
      )
    self.assertFalse(off_axis_res.ok)
    self.assertIn('simple_fold', off_axis_res.message)

    # Invalid inputs and `apply_symmetric(tool='shape_flap_template')` fail.
    _, bad_style = bp.apply_shaping_tool(
        state, 'shape_flap_template', {'style': 'unknown'}
    )
    self.assertFalse(bad_style.ok)
    _, bad_pct = bp.apply_shaping_tool(
        state, 'shape_flap_template', {'open_percent': 1.0}
    )
    self.assertFalse(bad_pct.ok)
    _, bad_elem = bp.apply_shaping_tool(
        state, 'shape_flap_template', {'element_name': 'head'}
    )
    self.assertFalse(bad_elem.ok)
    _, bad_kwarg = bp.apply_shaping_tool(
        state, 'shape_flap_template', {'applied_operations': {'bogus': 'x'}}
    )
    self.assertFalse(bad_kwarg.ok)
    _, bad_sym = bp.apply_shaping_tool(
        state,
        'apply_symmetric',
        {
            'tool': 'shape_flap_template',
            'element_name': 'knee_cap_l',
            'arguments': {'style': 'tuck', 'verify_fold': False},
        },
    )
    self.assertFalse(bad_sym.ok)

    # Mirrored KneeTemplate failure surfaces the inner `verify_3d_fold` reason
    # and names both symmetric shaping flaps.
    with mock.patch.object(
        state_lib,
        'verify_3d_fold',
        return_value=(False, 'simulated 3D strain 1.8%', None),
    ):
      _, sym_fail_res = bp.apply_shaping_tool(
          state,
          'shape_flap_template',
          {'element_name': 'knee_cap_l', 'style': 'tuck', 'verify_fold': True},
      )
    self.assertFalse(sym_fail_res.ok)
    self.assertIn('knee_cap_l', sym_fail_res.message)
    self.assertIn('knee_cap_r', sym_fail_res.message)
    self.assertIn('1.8%', sym_fail_res.message)

    # Asymmetric single KneeTemplate (`len(group) == 1`) sculpts cleanly and
    # surfaces single-flap failure reasons cleanly.
    asym_knee = {
        'name': 'asym_single_knee',
        'category': 'animal',
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
    }
    asym_packed = bp.pack(
        bp.prepare(asym_knee),
        bp.PackingTaskConfig(
            packer='builtin',
            max_retries=4,
            max_cps_per_grid_size=1,
            max_number_of_cps=1,
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
    asym_state = bp.ShapingState.from_solved(asym_solved)
    _, asym_ok = bp.apply_shaping_tool(
        asym_state, 'shape_flap_template', {'style': 'tuck'}
    )
    self.assertTrue(asym_ok.ok, msg=asym_ok.message)
    self.assertIn('cap=simple_fold', asym_ok.message)
    with mock.patch.object(
        state_lib,
        'verify_3d_fold',
        return_value=(False, 'simulated 3D strain 2.1%', None),
    ):
      _, asym_fail = bp.apply_shaping_tool(
          asym_state,
          'shape_flap_template',
          {'style': 'tuck', 'verify_fold': True},
      )
    self.assertFalse(asym_fail.ok)
    self.assertIn('cap', asym_fail.message)
    self.assertIn('2.1%', asym_fail.message)


if __name__ == '__main__':
  absltest.main()
