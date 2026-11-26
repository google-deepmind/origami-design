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

"""Tests for StickFigure grid-size lower bounds and packing defaults."""

import dataclasses
from typing import Any
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import bp_pipeline as bp
from bp_pipeline import pipeline
from bp_pipeline.packing import config as packing_config_lib
from bp_pipeline.packing import grid_size as grid_size_lib
from bp_pipeline.packing import utils_packing_flaps
from bp_pipeline.packing import utils_solver
from bp_pipeline.stick_figures import stick as stick_lib
from bp_pipeline.stick_figures import stick_figure
from bp_pipeline.stick_figures.examples import example_trees


def _make_figure(
    name: str,
    children_specs: tuple[tuple[str, str, str, int, int], ...],
) -> stick_figure.StickFigure:
  """Builds an arbitrary StickFigure from `(name, start, end, length, width)`."""
  children = [
      stick_lib.Stick(
          name=stick_name,
          start=start,
          end=end,
          azimuth_angle=(idx * 37) % 360,
          length=length,
          width=width,
      )
      for idx, (stick_name, start, end, length, width) in enumerate(
          children_specs
      )
  ]
  return stick_figure.StickFigure(
      name=name,
      category='test',
      complexity=2,
      root=children[0].name,
      children=children,
  )


class StickFigureGridSizeTest(parameterized.TestCase):

  def test_default_get_packing_gridsize_stays_v2(self):
    tree = example_trees.CAT_SIMPLE
    self.assertEqual(tree.get_packing_gridsize(), 10)
    self.assertEqual(
        tree.get_packing_gridsize(mode='v2'),
        stick_figure.calculate_grid_size(tree, mode='v2'),
    )

  def test_wide_limbs_align_with_utils_packing_flaps_get_flap_dim(self):
    # `utils_packing_flaps.get_flap_dim` returns `2 * stick.length` when
    # `use_level_shifter=False` and adds `level_shifter_pocket_depth` (+2 for
    # `width == 2`) when `use_level_shifter=True`.
    for tree, expected_no_ls, expected_with_ls in (
        (example_trees.CAT_LEGS_WIDTH_2, 12, 14),
        (example_trees.HUMAN_LIMBS_WIDTH_2, 12, 16),
    ):
      bound_no_ls = grid_size_lib.calculate_v4_grid_size(
          tree,
          use_level_shifter=False,
          flap_dim_fn=utils_packing_flaps.get_flap_dim,
      )
      bound_with_ls = grid_size_lib.calculate_v4_grid_size(
          tree,
          use_level_shifter=True,
          flap_dim_fn=utils_packing_flaps.get_flap_dim,
      )
      self.assertEqual(bound_no_ls, expected_no_ls)
      self.assertEqual(bound_with_ls, expected_with_ls)
      self.assertEqual(
          pipeline.get_base_grid_size(
              tree,
              pipeline.PackingTaskConfig(
                  packing_config=packing_config_lib.PackingConfig(
                      use_level_shifter=True
                  )
              ),
          ),
          expected_with_ls,
      )
      self.assertEqual(
          tree.get_packing_gridsize(use_level_shifter=True, mode='v4'),
          expected_with_ls,
      )
      self.assertEqual(
          stick_figure.calculate_grid_size(
              tree, mode='v4', use_level_shifter=True
          ),
          expected_with_ls,
      )
      packed_ls_grid = None
      for candidate_grid in range(expected_no_ls, 36, 2):
        ls_outputs = utils_solver.find_tiling_and_generate_crease_pattern(
            tree,
            grid_size=candidate_grid,
            config=packing_config_lib.PackingConfig(
                use_level_shifter=True,
                enforce_proximity_constraints=False,
            ),
        )
        if ls_outputs and ls_outputs[0].success:
          packed_ls_grid = ls_outputs[0].details['grid_size']
          break
      self.assertIsNotNone(packed_ls_grid)
      self.assertGreaterEqual(packed_ls_grid, bound_with_ls)

  def test_custom_flap_dim_fn_margin_enforces_ls_inset_bounds(self):
    star = stick_figure.StickFigure(
        name='five_star_wide_ls',
        category='test',
        complexity=1,
        root='leaf0',
        children=[
            stick_lib.Stick(
                name=f'leaf{i}',
                start='center',
                end=f'tip{i}',
                azimuth_angle=angle,
                length=3,
                width=2,
            )
            for i, angle in enumerate((0, 65, 135, 210, 290))
        ],
    )
    self.assertEqual(
        grid_size_lib.calculate_v4_grid_size(star, use_level_shifter=False), 12
    )

    def _custom_flap_dim(stick: stick_lib.Stick, use_ls: bool) -> int:
      margin = (
          max(0, int(stick.width) - 1)
          if use_ls and stick.requires_level_shifter
          else 0
      )
      return int(stick.length) * 2 + 2 * margin

    self.assertEqual(
        grid_size_lib.calculate_v4_grid_size(
            star, use_level_shifter=True, flap_dim_fn=_custom_flap_dim
        ),
        18,
    )

  def test_v4_pigeonhole_star_and_symmetric_left_half(self):
    asym_star = stick_figure.StickFigure(
        name='asym_five_star',
        category='test',
        complexity=1,
        root='f0',
        children=[
            stick_lib.Stick(
                name='f0', start='c', end='t0', azimuth_angle=0, length=2
            ),
            stick_lib.Stick(
                name='f1', start='c', end='t1', azimuth_angle=35, length=2
            ),
            stick_lib.Stick(
                name='f2', start='c', end='t2', azimuth_angle=90, length=2
            ),
            stick_lib.Stick(
                name='f3', start='c', end='t3', azimuth_angle=170, length=2
            ),
            stick_lib.Stick(
                name='f4', start='c', end='t4', azimuth_angle=280, length=2
            ),
        ],
    )
    self.assertFalse(asym_star.is_symmetric_tree)
    self.assertEqual(asym_star.get_packing_gridsize(mode='v2'), 5)
    self.assertEqual(asym_star.get_packing_gridsize(mode='v4'), 8)

    sym_wings = stick_figure.StickFigure(
        name='sym_eight_wings',
        category='test',
        complexity=1,
        root='l1',
        children=[
            stick_lib.Stick(
                name='ax', start='c', end='a_tip', azimuth_angle=0, length=1
            ),
            stick_lib.Stick(
                name='l1', start='c', end='l1t', azimuth_angle=30, length=2
            ),
            stick_lib.Stick(
                name='r1', start='c', end='r1t', azimuth_angle=330, length=2
            ),
            stick_lib.Stick(
                name='l2', start='c', end='l2t', azimuth_angle=60, length=2
            ),
            stick_lib.Stick(
                name='r2', start='c', end='r2t', azimuth_angle=300, length=2
            ),
            stick_lib.Stick(
                name='l3', start='c', end='l3t', azimuth_angle=90, length=2
            ),
            stick_lib.Stick(
                name='r3', start='c', end='r3t', azimuth_angle=270, length=2
            ),
            stick_lib.Stick(
                name='l4', start='c', end='l4t', azimuth_angle=120, length=2
            ),
            stick_lib.Stick(
                name='r4', start='c', end='r4t', azimuth_angle=240, length=2
            ),
        ],
    )
    self.assertTrue(sym_wings.is_symmetric_tree)
    self.assertEqual(sym_wings.get_packing_gridsize(mode='v4'), 12)

  def test_v4_handles_zero_length_mirrored_leaf_pairs(self):
    figure = stick_figure.StickFigure(
        name='zero_len_sym_leaves',
        category='test',
        complexity=1,
        root='axis',
        children=[
            stick_lib.Stick(
                name='axis', start='root', end='hub', azimuth_angle=0, length=1
            ),
            stick_lib.Stick(
                name='l1', start='hub', end='lt1', azimuth_angle=45, length=0
            ),
            stick_lib.Stick(
                name='r1', start='hub', end='rt1', azimuth_angle=315, length=0
            ),
            stick_lib.Stick(
                name='l2', start='hub', end='lt2', azimuth_angle=90, length=0
            ),
            stick_lib.Stick(
                name='r2', start='hub', end='rt2', azimuth_angle=270, length=0
            ),
        ],
    )
    self.assertTrue(figure.is_symmetric_tree)
    for mode in ('v2', 'v4'):
      self.assertEqual(figure.get_packing_gridsize(mode=mode), 2)

  def test_v4_respects_is_symmetric_override_and_unmerged_asymmetry(self):
    sym_eight_wings = stick_figure.StickFigure(
        name='sym_eight_wings',
        category='test',
        complexity=1,
        root='l1',
        children=[
            stick_lib.Stick(
                name='ax', start='c', end='a_tip', azimuth_angle=0, length=1
            ),
            stick_lib.Stick(
                name='l1', start='c', end='l1t', azimuth_angle=30, length=2
            ),
            stick_lib.Stick(
                name='r1', start='c', end='r1t', azimuth_angle=330, length=2
            ),
            stick_lib.Stick(
                name='l2', start='c', end='l2t', azimuth_angle=60, length=2
            ),
            stick_lib.Stick(
                name='r2', start='c', end='r2t', azimuth_angle=300, length=2
            ),
            stick_lib.Stick(
                name='l3', start='c', end='l3t', azimuth_angle=90, length=2
            ),
            stick_lib.Stick(
                name='r3', start='c', end='r3t', azimuth_angle=270, length=2
            ),
            stick_lib.Stick(
                name='l4', start='c', end='l4t', azimuth_angle=120, length=2
            ),
            stick_lib.Stick(
                name='r4', start='c', end='r4t', azimuth_angle=240, length=2
            ),
        ],
    )
    for fig in (sym_eight_wings, example_trees.SPIKE):
      self.assertEqual(
          grid_size_lib.calculate_v4_grid_size(fig, is_symmetric=False), 8
      )
      outputs = utils_solver.find_tiling_and_generate_crease_pattern(
          fig,
          is_symmetric=False,
          config=packing_config_lib.PackingConfig(
              enforce_proximity_constraints=False
          ),
      )
      self.assertTrue(outputs and outputs[0].success)
      self.assertEqual(outputs[0].details['grid_size'], 8)

    unmerged_split = example_trees.HUMAN_LIMBS_WIDTH_2_SPLIT
    merged_split, _ = unmerged_split.merge_linear_chains()
    self.assertFalse(unmerged_split.is_symmetric_tree)
    self.assertTrue(merged_split.is_symmetric_tree)
    self.assertEqual(unmerged_split.get_packing_gridsize(mode='v4'), 11)
    self.assertEqual(merged_split.get_packing_gridsize(mode='v4'), 12)
    split_outputs = utils_solver.find_tiling_and_generate_crease_pattern(
        unmerged_split,
        config=packing_config_lib.PackingConfig(
            enforce_proximity_constraints=False
        ),
    )
    self.assertTrue(split_outputs and split_outputs[0].success)
    self.assertEqual(split_outputs[0].details['grid_size'], 11)

  @parameterized.named_parameters(
      dict(
          testcase_name='two_bend_river_packs_at_6',
          children_specs=(
              ('river', 'hub_a', 'hub_b', 1, 1),
              ('a_long', 'hub_a', 'a1', 3, 1),
              ('a_short', 'hub_a', 'a2', 1, 1),
              ('b_0', 'hub_b', 'b0', 1, 1),
              ('b_1', 'hub_b', 'b1', 2, 1),
              ('b_2', 'hub_b', 'b2', 2, 1),
              ('b_3', 'hub_b', 'b3', 1, 1),
          ),
          max_valid_grid=6,
      ),
      dict(
          testcase_name='single_river_u_b6_packs_at_16',
          children_specs=(
              ('river', 'hub_a', 'hub_b', 1, 1),
              ('a_0', 'hub_a', 'a0', 1, 1),
              ('a_1', 'hub_a', 'a1', 1, 1),
              ('b_0', 'hub_b', 'b0', 6, 1),
              ('b_1', 'hub_b', 'b1', 6, 1),
              ('b_2', 'hub_b', 'b2', 6, 1),
              ('b_3', 'hub_b', 'b3', 6, 1),
          ),
          max_valid_grid=16,
      ),
      dict(
          testcase_name='single_river_u_b10_packs_at_24',
          children_specs=(
              ('river', 'hub_a', 'hub_b', 1, 1),
              ('a_0', 'hub_a', 'a0', 1, 1),
              ('a_1', 'hub_a', 'a1', 1, 1),
              ('b_0', 'hub_b', 'b0', 10, 1),
              ('b_1', 'hub_b', 'b1', 10, 1),
              ('b_2', 'hub_b', 'b2', 10, 1),
              ('b_3', 'hub_b', 'b3', 10, 1),
          ),
          max_valid_grid=24,
      ),
      dict(
          testcase_name='two_closed_ring_rivers_pack_at_16',
          children_specs=(
              ('m0', 'hub_m', 'mt0', 4, 1),
              ('m1', 'hub_m', 'mt1', 4, 1),
              ('m2', 'hub_m', 'mt2', 4, 1),
              ('m3', 'hub_m', 'mt3', 4, 1),
              ('m4', 'hub_m', 'mt4', 4, 1),
              ('m5', 'hub_m', 'mt5', 4, 1),
              ('m6', 'hub_m', 'mt6', 4, 1),
              ('m7', 'hub_m', 'mt7', 4, 1),
              ('r1', 'hub_m', 'hub_r1', 1, 1),
              ('r1_0', 'hub_r1', 'r1t0', 1, 1),
              ('r1_1', 'hub_r1', 'r1t1', 1, 1),
              ('r1_2', 'hub_r1', 'r1t2', 1, 1),
              ('r2', 'hub_m', 'hub_r2', 1, 1),
              ('r2_0', 'hub_r2', 'r2t0', 1, 1),
              ('r2_1', 'hub_r2', 'r2t1', 1, 1),
              ('r2_2', 'hub_r2', 'r2t2', 1, 1),
          ),
          max_valid_grid=16,
      ),
  )
  def test_v4_never_exceeds_river_cut_and_ring_counterexamples(
      self,
      children_specs: tuple[tuple[str, str, str, int, int], ...],
      max_valid_grid: int,
  ):
    figure = _make_figure('counterexample', children_specs)
    self.assertFalse(figure.is_symmetric_tree)
    self.assertLessEqual(
        figure.get_packing_gridsize(use_level_shifter=False, mode='v4'),
        max_valid_grid,
    )

  @parameterized.named_parameters(
      dict(testcase_name='cat_simple', figure=example_trees.CAT_SIMPLE),
      dict(testcase_name='human_simple', figure=example_trees.HUMAN_SIMPLE),
      dict(testcase_name='spider_simple', figure=example_trees.SPIDER_SIMPLE),
      dict(
          testcase_name='palm_tree_simple',
          figure=example_trees.PALM_TREE_SIMPLE,
      ),
      dict(
          testcase_name='cat_legs_width_2',
          figure=example_trees.CAT_LEGS_WIDTH_2,
      ),
      dict(
          testcase_name='human_limbs_width_2',
          figure=example_trees.HUMAN_LIMBS_WIDTH_2,
      ),
  )
  def test_v4_does_not_exceed_actual_solver_packing(
      self, figure: stick_figure.StickFigure
  ):
    step = 2 if figure.is_symmetric_tree else 1
    start_grid = figure.get_packing_gridsize(mode='v2')
    if figure.is_symmetric_tree and start_grid % 2 == 1:
      start_grid += 1
    packed_grid = None
    for candidate_grid in range(start_grid, 36, step):
      outputs = utils_solver.find_tiling_and_generate_crease_pattern(
          figure,
          grid_size=candidate_grid,
          config=packing_config_lib.PackingConfig(
              enforce_proximity_constraints=False
          ),
      )
      if outputs and outputs[0].success:
        packed_grid = outputs[0].details['grid_size']
        break
    self.assertIsNotNone(packed_grid)
    v4_bound = grid_size_lib.calculate_v4_grid_size(
        figure,
        use_level_shifter=False,
        flap_dim_fn=utils_packing_flaps.get_flap_dim,
    )
    self.assertLessEqual(v4_bound, packed_grid)

  def test_v4_no_overestimate_on_known_tight_examples(self):
    self.assertLessEqual(
        example_trees.GREY_WINGED_TRUMPETER.get_packing_gridsize(mode='v4'),
        12,
    )
    self.assertLessEqual(
        example_trees.PACKING_DANCER.get_packing_gridsize(mode='v4'),
        32,
    )
    self.assertLessEqual(
        example_trees.FORAGING_SNIPE.get_packing_gridsize(mode='v4'),
        14,
    )
    self.assertLessEqual(
        example_trees.RIVER_STAIRCASE.get_packing_gridsize(mode='v4'),
        16,
    )
    self.assertLessEqual(
        example_trees.DIVER_BACKFLIP.get_packing_gridsize(mode='v4'),
        16,
    )

  def test_v4_chain_merge_invariance_and_subset(self):
    unmerged = example_trees.HUMAN_WITH_BICEPS
    merged, _ = unmerged.merge_linear_chains()
    self.assertEqual(
        unmerged.get_packing_gridsize(use_level_shifter=True, mode='v4'),
        merged.get_packing_gridsize(use_level_shifter=True, mode='v4'),
    )
    tree = example_trees.CAT_SIMPLE
    self.assertEqual(
        stick_figure.calculate_grid_size(
            tree,
            sticks=tree.joints['shoulders'].sticks,
            mode='v4',
            use_longest_path=False,
        ),
        6,
    )

  def test_pipeline_uses_v4_while_preserving_min_grid_size(self):
    self.assertEqual(pipeline.PackingTaskConfig.default().grid_mode, 'v4')
    self.assertEqual(pipeline.PackingTaskConfig.reference().grid_mode, 'v2')

    expected_cat_start = example_trees.CAT_SIMPLE.get_packing_gridsize(
        mode='v4'
    )
    cat_result = pipeline.pack_crease_pattern(
        example_trees.CAT_SIMPLE,
        config=pipeline.PackingTaskConfig(
            max_number_of_cps=1,
            max_cps_per_grid_size=1,
            max_retries=3,
            sweep_timeout_seconds=20.0,
            packing_config=packing_config_lib.PackingConfig(
                enforce_proximity_constraints=False,
            ),
        ),
    )
    self.assertEqual(cat_result.attempted_grid_sizes[0], expected_cat_start)
    self.assertNotEmpty(cat_result.packing_outputs)

    human_design = bp.prepare('human')
    self.assertEqual(human_design.min_grid_size, 12)
    human_packed_12 = bp.pack(
        human_design,
        config=bp.PackingTaskConfig(
            packing_config=bp.PackingConfig(
                enforce_proximity_constraints=False
            ),
            sweep_timeout_seconds=20.0,
        ),
        packer_options=bp.PackerOptions(grid_size=12),
    )
    self.assertEqual(human_packed_12.grid_size, 12)

  def test_get_base_grid_size_symmetry_override_and_even_alignment(self):
    sym_wings = stick_figure.StickFigure(
        name='sym_eight_wings',
        category='test',
        complexity=1,
        root='l1',
        children=[
            stick_lib.Stick(
                name='ax', start='c', end='a_tip', azimuth_angle=0, length=1
            ),
            stick_lib.Stick(
                name='l1', start='c', end='l1t', azimuth_angle=30, length=2
            ),
            stick_lib.Stick(
                name='r1', start='c', end='r1t', azimuth_angle=330, length=2
            ),
            stick_lib.Stick(
                name='l2', start='c', end='l2t', azimuth_angle=60, length=2
            ),
            stick_lib.Stick(
                name='r2', start='c', end='r2t', azimuth_angle=300, length=2
            ),
            stick_lib.Stick(
                name='l3', start='c', end='l3t', azimuth_angle=90, length=2
            ),
            stick_lib.Stick(
                name='r3', start='c', end='r3t', azimuth_angle=270, length=2
            ),
            stick_lib.Stick(
                name='l4', start='c', end='l4t', azimuth_angle=120, length=2
            ),
            stick_lib.Stick(
                name='r4', start='c', end='r4t', azimuth_angle=240, length=2
            ),
        ],
    )
    self.assertEqual(
        pipeline.get_base_grid_size(sym_wings, pipeline.PackingTaskConfig()),
        12,
    )
    self.assertEqual(
        pipeline.get_base_grid_size(
            sym_wings, pipeline.PackingTaskConfig(), is_symmetric=False
        ),
        8,
    )
    odd_v2_sym = stick_figure.StickFigure(
        name='odd_v2_sym',
        category='test',
        complexity=1,
        root='ax',
        children=[
            stick_lib.Stick(
                name='ax', start='c', end='a_tip', azimuth_angle=0, length=2
            ),
            stick_lib.Stick(
                name='l1', start='c', end='l1t', azimuth_angle=45, length=2
            ),
            stick_lib.Stick(
                name='r1', start='c', end='r1t', azimuth_angle=315, length=2
            ),
            stick_lib.Stick(
                name='l2', start='c', end='l2t', azimuth_angle=135, length=2
            ),
            stick_lib.Stick(
                name='r2', start='c', end='r2t', azimuth_angle=225, length=2
            ),
        ],
    )
    self.assertEqual(
        pipeline.get_base_grid_size(
            odd_v2_sym, pipeline.PackingTaskConfig.reference()
        ),
        6,
    )
    self.assertEqual(
        pipeline.get_base_grid_size(
            odd_v2_sym,
            pipeline.PackingTaskConfig.reference(),
            is_symmetric=False,
        ),
        5,
    )

  def test_v4_symmetric_offaxis_river_mirror_twin_separation(self):
    children = [
        stick_lib.Stick(
            name='ax', start='hub', end='ax_tip', azimuth_angle=0, length=1
        ),
        stick_lib.Stick(
            name='l_arm',
            start='hub',
            end='l_elbow',
            azimuth_angle=90,
            length=3,
        ),
        stick_lib.Stick(
            name='r_arm',
            start='hub',
            end='r_elbow',
            azimuth_angle=270,
            length=3,
        ),
    ]
    for i, angle in enumerate((20, 50, 80, 100, 130, 160)):
      children.append(
          stick_lib.Stick(
              name=f'l_f{i}',
              start='l_elbow',
              end=f'l_t{i}',
              azimuth_angle=angle,
              length=1,
          )
      )
      children.append(
          stick_lib.Stick(
              name=f'r_f{i}',
              start='r_elbow',
              end=f'r_t{i}',
              azimuth_angle=360 - angle,
              length=1,
          )
      )
    sym_offaxis_river = stick_figure.StickFigure(
        name='sym_offaxis_river',
        category='test',
        complexity=2,
        root='ax',
        children=children,
    )
    self.assertTrue(sym_offaxis_river.is_symmetric_tree)
    self.assertEqual(sym_offaxis_river.get_packing_gridsize(mode='v4'), 10)
    self.assertEqual(
        grid_size_lib.calculate_v4_grid_size(
            sym_offaxis_river, is_symmetric=False
        ),
        8,
    )
    failed_8 = utils_solver.find_tiling_and_generate_crease_pattern(
        sym_offaxis_river,
        grid_size=8,
        config=packing_config_lib.PackingConfig(
            enforce_proximity_constraints=False
        ),
    )
    self.assertFalse(failed_8 and failed_8[0].success)
    packed_grid = None
    for candidate_grid in range(10, 18, 2):
      packed = utils_solver.find_tiling_and_generate_crease_pattern(
          sym_offaxis_river,
          grid_size=candidate_grid,
          config=packing_config_lib.PackingConfig(
              enforce_proximity_constraints=False
          ),
      )
      if packed and packed[0].success:
        packed_grid = packed[0].details['grid_size']
        break
    self.assertIsNotNone(packed_grid)
    self.assertGreaterEqual(packed_grid, 10)

  def test_longest_path_in_joints_empty_and_v4_unpaired_symmetric_override(
      self,
  ):
    @dataclasses.dataclass(frozen=True)
    class _EmptyJoint:
      degree: int = 0
      sticks: tuple[Any, ...] = ()

    self.assertEqual(grid_size_lib.longest_path_in_joints({}), 0)
    self.assertEqual(
        grid_size_lib.longest_path_in_joints({'j0': _EmptyJoint()}), 0
    )

    asym_odd = _make_figure(
        'asym_odd_diameter',
        (
            ('s0', 'hub', 't0', 3, 1),
            ('s1', 'hub', 't1', 4, 1),
            ('s2', 'hub', 't2', 1, 1),
        ),
    )
    self.assertFalse(asym_odd.is_symmetric_tree)
    self.assertEqual(
        grid_size_lib.calculate_v4_grid_size(asym_odd, is_symmetric=False), 7
    )
    self.assertEqual(
        grid_size_lib.calculate_v4_grid_size(asym_odd, is_symmetric=True), 8
    )

  def test_v4_level_shifter_rejects_unsupported_width_gt_2(self):
    wide_3 = _make_figure(
        'wide_3_flap',
        (
            ('a', 'c', 't0', 4, 3),
            ('b', 'c', 't1', 2, 1),
        ),
    )
    with self.assertRaisesRegex(ValueError, 'support width == 2'):
      wide_3.get_packing_gridsize(use_level_shifter=True, mode='v4')

  def test_v4_single_stick_wide_flap_does_not_overestimate(self):
    single_wide = _make_figure(
        'single_wide_flap',
        (('flap', 'root', 'tip', 4, 2),),
    )
    self.assertEqual(
        grid_size_lib.calculate_v4_grid_size(
            single_wide,
            use_level_shifter=True,
            is_symmetric=False,
            flap_dim_fn=utils_packing_flaps.get_flap_dim,
        ),
        6,
    )
    self.assertEqual(
        single_wide.get_packing_gridsize(
            use_level_shifter=True, mode='v4', is_symmetric=False
        ),
        6,
    )

  def test_allow_paired_axis_placement_relaxes_v4_and_preserves_even_grid(self):
    sym_wings = stick_figure.StickFigure(
        name='sym_eight_wings',
        category='test',
        complexity=1,
        root='l1',
        children=[
            stick_lib.Stick(
                name='ax', start='c', end='a_tip', azimuth_angle=0, length=1
            ),
            stick_lib.Stick(
                name='l1', start='c', end='l1t', azimuth_angle=30, length=2
            ),
            stick_lib.Stick(
                name='r1', start='c', end='r1t', azimuth_angle=330, length=2
            ),
            stick_lib.Stick(
                name='l2', start='c', end='l2t', azimuth_angle=60, length=2
            ),
            stick_lib.Stick(
                name='r2', start='c', end='r2t', azimuth_angle=300, length=2
            ),
            stick_lib.Stick(
                name='l3', start='c', end='l3t', azimuth_angle=90, length=2
            ),
            stick_lib.Stick(
                name='r3', start='c', end='r3t', azimuth_angle=270, length=2
            ),
            stick_lib.Stick(
                name='l4', start='c', end='l4t', azimuth_angle=120, length=2
            ),
            stick_lib.Stick(
                name='r4', start='c', end='r4t', azimuth_angle=240, length=2
            ),
        ],
    )
    self.assertEqual(
        pipeline.get_base_grid_size(
            sym_wings,
            pipeline.PackingTaskConfig(allow_paired_axis_placement=True),
        ),
        8,
    )

    sym_odd_paired = stick_figure.StickFigure(
        name='sym_odd_paired',
        category='test',
        complexity=1,
        root='ax',
        children=[
            stick_lib.Stick(
                name='ax', start='c', end='a_tip', azimuth_angle=0, length=5
            ),
            stick_lib.Stick(
                name='l1', start='c', end='l1t', azimuth_angle=45, length=2
            ),
            stick_lib.Stick(
                name='r1', start='c', end='r1t', azimuth_angle=315, length=2
            ),
        ],
    )
    self.assertTrue(sym_odd_paired.is_symmetric_tree)
    self.assertEqual(
        grid_size_lib.calculate_v4_grid_size(
            sym_odd_paired, is_symmetric=False
        ),
        7,
    )
    self.assertEqual(
        pipeline.get_base_grid_size(
            sym_odd_paired,
            pipeline.PackingTaskConfig(allow_paired_axis_placement=True),
        ),
        8,
    )
    with mock.patch.object(
        grid_size_lib,
        'calculate_v4_grid_size',
        wraps=grid_size_lib.calculate_v4_grid_size,
    ) as spy_v4:
      paired_outputs = utils_solver.find_tiling_and_generate_crease_pattern(
          example_trees.CAT_SIMPLE,
          allow_paired_axis_placement=True,
          config=packing_config_lib.PackingConfig(
              enforce_proximity_constraints=False
          ),
      )
    self.assertFalse(spy_v4.call_args.kwargs['is_symmetric'])
    self.assertTrue(paired_outputs and paired_outputs[0].success)

  def test_v4_disconnected_symmetric_graph_skips_unreachable_pairs(self):
    @dataclasses.dataclass(frozen=True)
    class _MockStick:
      name: str
      start: str
      end: str
      length: int
      width: int = 1
      is_flap: bool = True
      is_river: bool = False
      is_on_symmetry_axis: bool = False
      symmetric_stick_name: str | None = None
      requires_level_shifter: bool = False

    @dataclasses.dataclass(frozen=True)
    class _MockJoint:
      sticks: tuple[_MockStick, ...]

      @property
      def degree(self) -> int:
        return len(self.sticks)

    class _MockDisconnectedFigure:

      def __init__(self) -> None:
        self.symmetry_axis = ['ax1', 'ax2']
        self.is_symmetric_tree = True
        self.children = [
            _MockStick(
                name='ax1',
                start='c1',
                end='a1',
                length=1,
                is_on_symmetry_axis=True,
            ),
            _MockStick(
                name='l1',
                start='c1',
                end='l1t',
                length=2,
                symmetric_stick_name='r1',
            ),
            _MockStick(
                name='r1',
                start='c1',
                end='r1t',
                length=2,
                symmetric_stick_name='l1',
            ),
            _MockStick(
                name='ax2',
                start='c2',
                end='a2',
                length=1,
                is_on_symmetry_axis=True,
            ),
            _MockStick(
                name='l2',
                start='c2',
                end='l2t',
                length=2,
                symmetric_stick_name='r2',
            ),
            _MockStick(
                name='r2',
                start='c2',
                end='r2t',
                length=2,
                symmetric_stick_name='l2',
            ),
        ]
        by_name = {s.name: s for s in self.children}
        self._by_name = by_name
        joints: dict[str, list[_MockStick]] = {}
        for s in self.children:
          joints.setdefault(s.start, []).append(s)
          joints.setdefault(s.end, []).append(s)
        self.joints = {k: _MockJoint(tuple(v)) for k, v in joints.items()}

      def __getitem__(self, key: str) -> _MockStick:
        return self._by_name[key]

      def merge_linear_chains(
          self,
      ) -> tuple['_MockDisconnectedFigure', dict[str, Any]]:
        return self, {}

    self.assertEqual(
        grid_size_lib.calculate_v4_grid_size(_MockDisconnectedFigure()), 4
    )

  def test_v4_subset_sticks_symmetry_axis_and_longest_path(self):
    sym_axis_chain = stick_figure.StickFigure(
        name='sym_axis_chain',
        category='test',
        complexity=1,
        root='ax1',
        children=[
            stick_lib.Stick(
                name='ax1',
                start='root_j',
                end='mid_j',
                azimuth_angle=0,
                elevation_angle=90.0,
                length=3,
            ),
            stick_lib.Stick(
                name='ax2',
                start='mid_j',
                end='hub_j',
                azimuth_angle=0,
                elevation_angle=90.0,
                length=4,
            ),
            stick_lib.Stick(
                name='l1',
                start='hub_j',
                end='l1t',
                azimuth_angle=45,
                length=2,
            ),
            stick_lib.Stick(
                name='r1',
                start='hub_j',
                end='r1t',
                azimuth_angle=315,
                length=2,
            ),
        ],
    )
    self.assertTrue(sym_axis_chain.is_symmetric_tree)
    self.assertIn('ax1', sym_axis_chain.symmetry_axis)
    self.assertEqual(sym_axis_chain.joints['mid_j'].degree, 2)
    ax1_only = [sym_axis_chain['ax1']]
    self.assertEqual(
        grid_size_lib.calculate_v4_grid_size(
            sym_axis_chain, sticks=ax1_only, use_longest_path=False
        ),
        4,
    )
    self.assertEqual(
        grid_size_lib.calculate_v4_grid_size(
            sym_axis_chain,
            sticks=ax1_only,
            is_symmetric=False,
            use_longest_path=False,
        ),
        3,
    )
    self.assertEqual(
        grid_size_lib.calculate_v4_grid_size(
            sym_axis_chain,
            sticks=ax1_only,
            is_symmetric=False,
            use_longest_path=True,
        ),
        9,
    )
    self.assertEqual(
        grid_size_lib.calculate_v4_grid_size(
            sym_axis_chain, sticks=ax1_only, use_longest_path=True
        ),
        10,
    )

  def test_v4_sym_rect_pigeonhole_tight_half_boundary(self):
    sym_ribs = stick_figure.StickFigure(
        name='sym_four_ribs',
        category='test',
        complexity=1,
        root='ax12',
        children=[
            stick_lib.Stick(
                name='ax12',
                start='h1',
                end='h2',
                azimuth_angle=0,
                elevation_angle=90.0,
                length=1,
            ),
            stick_lib.Stick(
                name='ax23',
                start='h2',
                end='h3',
                azimuth_angle=0,
                elevation_angle=90.0,
                length=1,
            ),
            stick_lib.Stick(
                name='ax34',
                start='h3',
                end='h4',
                azimuth_angle=0,
                elevation_angle=90.0,
                length=1,
            ),
            stick_lib.Stick(
                name='l1', start='h1', end='l1t', azimuth_angle=45, length=1
            ),
            stick_lib.Stick(
                name='r1', start='h1', end='r1t', azimuth_angle=315, length=1
            ),
            stick_lib.Stick(
                name='l2', start='h2', end='l2t', azimuth_angle=45, length=1
            ),
            stick_lib.Stick(
                name='r2', start='h2', end='r2t', azimuth_angle=315, length=1
            ),
            stick_lib.Stick(
                name='l3', start='h3', end='l3t', azimuth_angle=45, length=1
            ),
            stick_lib.Stick(
                name='r3', start='h3', end='r3t', azimuth_angle=315, length=1
            ),
            stick_lib.Stick(
                name='l4', start='h4', end='l4t', azimuth_angle=45, length=1
            ),
            stick_lib.Stick(
                name='r4', start='h4', end='r4t', azimuth_angle=315, length=1
            ),
        ],
    )
    self.assertTrue(sym_ribs.is_symmetric_tree)
    self.assertEqual(grid_size_lib.calculate_v4_grid_size(sym_ribs), 8)
    packed_6 = utils_solver.find_tiling_and_generate_crease_pattern(
        sym_ribs,
        grid_size=6,
        config=packing_config_lib.PackingConfig(
            enforce_proximity_constraints=False
        ),
    )
    self.assertFalse(packed_6 and packed_6[0].success)
    packed_8 = utils_solver.find_tiling_and_generate_crease_pattern(
        sym_ribs,
        grid_size=8,
        config=packing_config_lib.PackingConfig(
            enforce_proximity_constraints=False
        ),
    )
    self.assertTrue(packed_8 and packed_8[0].success)


if __name__ == '__main__':
  absltest.main()
