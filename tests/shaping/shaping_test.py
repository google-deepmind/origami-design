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

"""Unit and end-to-end tests for the box-pleating shaping package."""

import copy
import logging
import math
import pathlib
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import bp_pipeline as bp
from bp_pipeline import pipeline
from bp_pipeline import rectangle as rect_utils
from bp_pipeline.crease_pattern import crease as crease_utils
from bp_pipeline.crease_pattern import point as point_utils
from bp_pipeline.packing import types as packing_types
from bp_pipeline.packing import utils_solver
from bp_pipeline.shaping import bus_adapter
from bp_pipeline.shaping import geometric_fold_3d_heuristics
from bp_pipeline.shaping import geometric_fold_3d_heuristics_utils
from bp_pipeline.shaping import geometric_simple_fold
from bp_pipeline.shaping import geometric_utils
from bp_pipeline.shaping import heuristics
from bp_pipeline.shaping import heuristics_3d
from bp_pipeline.shaping import heuristics_utils
from bp_pipeline.shaping import shaping_elements
from bp_pipeline.shaping import shaping_utils
from bp_pipeline.shaping import vector
from bp_pipeline.solving import hinges as hinge_utils
from bp_pipeline.stick_figures.examples import example_trees
import cp_test_fixtures as examples
import numpy as np


class ShapingTest(parameterized.TestCase):

  def test_default_and_strategy_sweep_configs(self):
    default_cfg = bp.ShapingTaskConfig()
    self.assertTrue(default_cfg.apply_simple_fold_shaping)
    self.assertTrue(default_cfg.shape_with_unmerged_tree)
    self.assertTrue(default_cfg.filter_by_local_flat_foldability)
    self.assertEqual(default_cfg.max_number_of_cps, 8)
    self.assertAlmostEqual(default_cfg.max_slope_thresh, 10.0)
    self.assertAlmostEqual(default_cfg.clip_slope_thresh, 3.0)
    self.assertEqual(default_cfg.fold_strategy_3d, bp.FoldStrategy3D.DEFAULT)

    ref_cfg = bp.ShapingTaskConfig.default()
    self.assertEqual(ref_cfg.max_number_of_cps, 8)
    self.assertEqual(ref_cfg.timeout_seconds, 3600)
    self.assertAlmostEqual(ref_cfg.cp_tolerance, 0.001)
    self.assertFalse(ref_cfg.enable_strategy_fallback)
    self.assertTrue(ref_cfg.apply_simple_fold_shaping)
    self.assertTrue(ref_cfg.shape_with_unmerged_tree)

    sweep_cfg = bp.ShapingTaskConfig.with_strategy_sweep()
    swept = list(sweep_cfg.config_iter())
    self.assertLen(swept, 3)
    self.assertEqual(
        [c.fold_strategy_3d for c in swept],
        [
            bp.FoldStrategy3D.DEFAULT,
            bp.FoldStrategy3D.NO_FLIP,
            bp.FoldStrategy3D.SHALLOW_FOLDS,
        ],
    )

  @parameterized.named_parameters(
      dict(
          testcase_name='geometric_3d_default',
          strategy=bp.FoldStrategy3D.DEFAULT,
          expected_strategy=bp.ShapingStrategy.DEFAULT,
      ),
      dict(
          testcase_name='geometric_3d_no_flip',
          strategy=bp.FoldStrategy3D.NO_FLIP,
          expected_strategy=bp.ShapingStrategy.ROTATE_BEND,
      ),
      dict(
          testcase_name='geometric_3d_shallow_folds',
          strategy=bp.FoldStrategy3D.SHALLOW_FOLDS,
          expected_strategy=bp.ShapingStrategy.SHALLOW_FOLDS,
      ),
  )
  def test_shape_strategies_and_modes(
      self,
      strategy: bp.FoldStrategy3D,
      expected_strategy: bp.ShapingStrategy,
  ):
    fold_res = bp.fold('cat', preset='demo')
    cfg = bp.ShapingTaskConfig(
        fold_strategy_3d=strategy,
        filter_by_local_flat_foldability=True,
        enable_strategy_fallback=False,
    )
    shaped = bp.shape(fold_res.solved, cfg)
    self.assertIsInstance(shaped, bp.ShapedCreasePattern)
    self.assertIsInstance(shaped.strategy, bp.ShapingStrategy)
    self.assertEqual(shaped.strategy, expected_strategy)
    self.assertTrue(shaped.is_locally_flat_foldable)
    self.assertTrue(shaped.creases.local_flat_foldability())
    self.assertGreater(
        len(shaped.creases.creases), len(fold_res.solved.creases.creases)
    )
    self.assertIn(
        '<svg',
        shaped.to_svg(
            stroke_width=1.5,
            dashed_valley=True,
            include_grid=False,
        ),
    )

  def test_codec_round_trip_shaped_and_deserialized_solved(self):
    fold_res = bp.fold('human', preset='demo')
    shaped = bp.shape(fold_res.solved)
    serialized = bp.dumps(shaped)
    decoded_shaped = bp.loads(serialized)
    self.assertIsInstance(decoded_shaped, bp.ShapedCreasePattern)
    self.assertEqual(decoded_shaped.strategy, shaped.strategy)
    self.assertEqual(bp.to_dict(decoded_shaped), bp.to_dict(shaped))
    self.assertEqual(
        decoded_shaped.creases.hash_value,
        shaped.creases.hash_value,
    )

    # Verify that a SolvedCreasePattern round-tripped through JSON takes the
    # exact same code path and produces identical shaped output.
    decoded_solved = bp.loads(bp.dumps(fold_res.solved))
    shaped_from_decoded = bp.shape(decoded_solved)
    self.assertTrue(shaped_from_decoded.is_locally_flat_foldable)
    self.assertEqual(shaped_from_decoded.strategy, shaped.strategy)
    self.assertEqual(
        shaped_from_decoded.creases.hash_value,
        shaped.creases.hash_value,
    )

  def test_fold_with_shape_and_export_entrypoints(self):
    res = bp.fold('cat', preset='demo', shape=True)
    self.assertIsNotNone(res.shaped)
    self.assertNotEmpty(res.shaped_candidates)
    self.assertTrue(res.shaped.is_locally_flat_foldable)
    self.assertIsInstance(res.shaped.strategy, bp.ShapingStrategy)
    self.assertIn('strategy', res.shaped.diagnostics)

    self.assertIn('<svg', bp.to_svg(res.shaped))
    self.assertIn('vertices_coords', bp.to_fold(res.shaped))
    self.assertIn('design', bp.to_bp_studio(res.shaped))
    self.assertEqual(bp.to_svg(res), bp.to_svg(res.shaped))

    # Sweeping strategies via ShapingTaskConfig.with_strategy_sweep().
    sweep_res = bp.fold(
        'cat',
        preset='demo',
        shape=True,
        shaping_config=bp.ShapingTaskConfig.with_strategy_sweep(
            max_number_of_cps=1
        ),
    )
    self.assertGreaterEqual(len(sweep_res.shaped_candidates), 1)

  def test_locked_example_reference_parity(self):
    tree = copy.deepcopy(example_trees.HUMAN_SIMPLE)
    grid_size = 12
    packing_outputs = utils_solver.find_tiling_and_generate_crease_pattern(
        tree, grid_size, is_symmetric=True
    )
    self.assertNotEmpty(packing_outputs)
    packing_output = packing_outputs[0]
    solver_results = hinge_utils.cp_from_packing(
        cp=packing_output.packing_cp,
        grid_size=grid_size,
        check_global_flat_foldability=False,
        tie_break_by_hinge_symmetry=True,
        remove_central_crease=False,
        disallow_trapping_edge_flaps=True,
        packing_information=packing_output,
    )
    self.assertNotEmpty(solver_results)

    design = bp.DesignTree.from_stick_figure(tree)
    packed = bp.PackedCreasePattern.from_packing_output(
        packing_output, design=design
    )
    solved = bp.SolvedCreasePattern.from_solver_result(
        solver_results[0], packed=packed
    )
    ref_cfg = bp.ShapingTaskConfig.default()
    shaped = bp.shape(solved, ref_cfg)
    self.assertEqual(shaped.strategy, bp.ShapingStrategy.DEFAULT)
    self.assertTrue(shaped.is_locally_flat_foldable)
    self.assertTrue(shaped.creases.local_flat_foldability())
    self.assertEqual(shaped.grid_size, 12)
    self.assertGreater(len(shaped.creases.creases), len(solved.creases.creases))

    # Also verify direct pipeline parity on the same (packing_output,
    # solver_result) vs bus-adapted solved CP.
    direct_res = pipeline.shape_crease_pattern(
        pipeline.ShapingTaskInput(
            name='locked_example',
            shaping_instances=(
                pipeline.ShapingInstance(
                    tree=tree,
                    packing=packing_output,
                    solving=solver_results[0],
                ),
            ),
        ),
        config=ref_cfg,
        stick_figure_input=tree,
    )
    self.assertNotEmpty(direct_res.shaped_cps)
    self.assertEqual(
        shaped.creases.hash_value,
        direct_res.shaped_cps[0].hash_value,
    )
    self.assertLen(
        shaped.creases.creases, len(direct_res.shaped_cps[0].creases)
    )

  def test_shaping_does_not_mutate_input_solved_or_design_tree(self):
    fold_res = bp.fold('cat', preset='demo')
    solved = fold_res.solved
    dumps_before = bp.codec.dumps(solved)
    design_source_before = copy.deepcopy(solved.packed.design.source)
    shaped = bp.shape(solved)
    self.assertIsNotNone(shaped)
    self.assertEqual(bp.codec.dumps(solved), dumps_before)
    self.assertEqual(solved.packed.design.source, design_source_before)

  @parameterized.parameters('left_arm', 'right_arm')
  def test_symmetric_pair_rollback_when_single_side_fails(
      self, failing_limb: str
  ):
    fold_res = bp.fold('human', preset='demo')
    solved = fold_res.solved

    orig_get_creases = (
        geometric_fold_3d_heuristics.get_shaping_creases_with_3d_geometric_folding_heuristics
    )
    captured_results = []

    def _spy_get_creases(*args, **kwargs):
      res = orig_get_creases(*args, **kwargs)
      captured_results.append(res)
      return res

    with mock.patch.object(
        geometric_fold_3d_heuristics,
        'get_shaping_creases_with_3d_geometric_folding_heuristics',
        side_effect=_spy_get_creases,
    ):
      full_shaped = bp.shape(solved)
    self.assertTrue(full_shaped.is_locally_flat_foldable)
    full_info = captured_results[-1]['detailed_shaping_info']
    self.assertIn('left_arm', full_info)
    self.assertIn('right_arm', full_info)

    orig_shape_elem = (
        geometric_fold_3d_heuristics.shape_element_with_3d_geometric_folding_heuristics
    )
    bad_crease = crease_utils.Crease(
        point_utils.Point(x=0.23, y=0.37),
        point_utils.Point(x=0.61, y=0.79),
        fold=crease_utils.Fold.MOUNTAIN,
    )

    def _inject_single_side_ff_violation(*args, **kwargs):
      element = kwargs.get('element') or args[0]
      res = orig_shape_elem(*args, **kwargs)
      if element.name == failing_limb:
        return {
            **res,
            'shaping_creases': list(res['shaping_creases']) + [bad_crease],
        }
      return res

    captured_results.clear()
    with (
        mock.patch.object(
            geometric_fold_3d_heuristics,
            'get_shaping_creases_with_3d_geometric_folding_heuristics',
            side_effect=_spy_get_creases,
        ),
        mock.patch.object(
            geometric_fold_3d_heuristics,
            'shape_element_with_3d_geometric_folding_heuristics',
            side_effect=_inject_single_side_ff_violation,
        ),
    ):
      rolled_back_shaped = bp.shape(solved)

    self.assertTrue(rolled_back_shaped.is_locally_flat_foldable)
    rollback_info = captured_results[-1]['detailed_shaping_info']
    self.assertNotIn('left_arm', rollback_info)
    self.assertNotIn('right_arm', rollback_info)
    self.assertLess(
        len(rolled_back_shaped.creases.creases),
        len(full_shaped.creases.creases),
    )


class VectorTest(parameterized.TestCase):

  def test_vector_add(self):
    v1 = vector.Vector(x=1, y=2)
    v2 = vector.Vector(x=3, y=4)
    pred_v = vector.Vector(x=4, y=6)
    self.assertEqual(v1 + v2, pred_v)

  def test_vector_sub(self):
    v1 = vector.Vector(x=1, y=2)
    v2 = vector.Vector(x=3, y=4)
    pred_v = vector.Vector(x=-2, y=-2)
    self.assertEqual(v1 - v2, pred_v)

  def test_vector_dot(self):
    v1 = vector.Vector(x=1, y=2)
    v2 = vector.Vector(x=3, y=4)
    self.assertEqual(v1.dot(v2), 11)

  def test_vector_point_add(self):
    v = vector.Vector(x=0.1, y=0.2)
    p = point_utils.Point(x=0.3, y=0.4)
    pred_p = point_utils.Point(x=0.4, y=0.6)
    self.assertTrue(pred_p.isclose(v.point_add(p)))

  def test_vector_scalar_mul(self):
    v = vector.Vector(x=1, y=2)
    pred_v = vector.Vector(x=2, y=4)
    self.assertEqual(2 * v, pred_v)
    self.assertEqual(v * 2, pred_v)

  def test_vector_norm(self):
    v = vector.Vector(x=1, y=2)
    self.assertAlmostEqual(v.norm, math.sqrt(5), places=3)

  def test_vector_slope(self):
    v = vector.Vector(x=1, y=2)
    self.assertEqual(v.slope, 2)
    v = vector.Vector(x=0, y=2)
    self.assertEqual(v.slope, float('inf'))

  def test_vector_normalize(self):
    v = vector.Vector(x=1, y=2)
    pred_v = vector.Vector(x=1 / math.sqrt(5), y=2 / math.sqrt(5))
    self.assertEqual(v.normalize, pred_v)

  def test_vector_inf_norm(self):
    v = vector.Vector(x=1, y=2)
    self.assertAlmostEqual(v.inf_norm, 2, places=3)

  def test_vector_inf_normalize(self):
    v = vector.Vector(x=1, y=2)
    pred_v = vector.Vector(x=1 / 2, y=1)
    self.assertEqual(v.inf_normalize, pred_v)


class ShapingUtilsTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='crease_inside',
          crease_to_check=crease_utils.Crease(
              point_utils.Point(x=0.2, y=0.2),
              point_utils.Point(x=0.4, y=0.4),
              fold=crease_utils.Fold.MOUNTAIN,
          ),
          strict=False,
          expected=True,
      ),
      dict(
          testcase_name='crease_inside_strict',
          crease_to_check=crease_utils.Crease(
              point_utils.Point(x=0.2, y=0.2),
              point_utils.Point(x=0.4, y=0.4),
              fold=crease_utils.Fold.MOUNTAIN,
          ),
          strict=True,
          expected=True,
      ),
      dict(
          testcase_name='crease_on_border',
          crease_to_check=crease_utils.Crease(
              point_utils.Point(x=0.1, y=0.1),
              point_utils.Point(x=0.5, y=0.5),
              fold=crease_utils.Fold.MOUNTAIN,
          ),
          strict=False,
          expected=True,
      ),
      dict(
          testcase_name='crease_on_border_strict',
          crease_to_check=crease_utils.Crease(
              point_utils.Point(x=0.1, y=0.1),
              point_utils.Point(x=0.5, y=0.5),
              fold=crease_utils.Fold.MOUNTAIN,
          ),
          strict=True,
          expected=False,
      ),
      dict(
          testcase_name='crease_outside',
          crease_to_check=crease_utils.Crease(
              point_utils.Point(x=0.6, y=0.6),
              point_utils.Point(x=0.7, y=0.7),
              fold=crease_utils.Fold.MOUNTAIN,
          ),
          strict=False,
          expected=False,
      ),
      dict(
          testcase_name='crease_partially_inside',
          crease_to_check=crease_utils.Crease(
              point_utils.Point(x=0.4, y=0.4),
              point_utils.Point(x=0.6, y=0.6),
              fold=crease_utils.Fold.MOUNTAIN,
          ),
          strict=False,
          expected=False,
      ),
  )
  def test_crease_is_within(self, crease_to_check, strict, expected):
    rect = shaping_utils.Rectangle(
        name='test_rect', x_start=0.1, y_start=0.1, x_len=0.4, y_len=0.4
    )
    self.assertEqual(rect.crease_is_within(crease_to_check, strict), expected)

  @parameterized.named_parameters(
      dict(
          testcase_name='strictly_contained',
          crease_1=crease_utils.Crease(
              point_utils.Point(x=0.1, y=0.1),
              point_utils.Point(x=0.5, y=0.5),
              fold=crease_utils.Fold.MOUNTAIN,
          ),
          crease_2=crease_utils.Crease(
              point_utils.Point(x=0.2, y=0.2),
              point_utils.Point(x=0.4, y=0.4),
              fold=crease_utils.Fold.MOUNTAIN,
          ),
          expected=crease_utils.Crease(
              point_utils.Point(x=0.1, y=0.1),
              point_utils.Point(x=0.5, y=0.5),
              fold=crease_utils.Fold.MOUNTAIN,
          ),
      ),
      dict(
          testcase_name='overlapping',
          crease_1=crease_utils.Crease(
              point_utils.Point(x=0.1, y=0.1),
              point_utils.Point(x=0.3, y=0.3),
              fold=crease_utils.Fold.MOUNTAIN,
          ),
          crease_2=crease_utils.Crease(
              point_utils.Point(x=0.2, y=0.2),
              point_utils.Point(x=0.5, y=0.5),
              fold=crease_utils.Fold.MOUNTAIN,
          ),
          expected=crease_utils.Crease(
              point_utils.Point(x=0.1, y=0.1),
              point_utils.Point(x=0.5, y=0.5),
              fold=crease_utils.Fold.MOUNTAIN,
          ),
      ),
      dict(
          testcase_name='touching_at_tip',
          crease_1=crease_utils.Crease(
              point_utils.Point(x=0.1, y=0.1),
              point_utils.Point(x=0.3, y=0.3),
              fold=crease_utils.Fold.MOUNTAIN,
          ),
          crease_2=crease_utils.Crease(
              point_utils.Point(x=0.3, y=0.3),
              point_utils.Point(x=0.5, y=0.5),
              fold=crease_utils.Fold.MOUNTAIN,
          ),
          expected=crease_utils.Crease(
              point_utils.Point(x=0.1, y=0.1),
              point_utils.Point(x=0.5, y=0.5),
              fold=crease_utils.Fold.MOUNTAIN,
          ),
      ),
      dict(
          testcase_name='identical',
          crease_1=crease_utils.Crease(
              point_utils.Point(x=0.1, y=0.1),
              point_utils.Point(x=0.3, y=0.3),
              fold=crease_utils.Fold.MOUNTAIN,
          ),
          crease_2=crease_utils.Crease(
              point_utils.Point(x=0.1, y=0.1),
              point_utils.Point(x=0.3, y=0.3),
              fold=crease_utils.Fold.MOUNTAIN,
          ),
          expected=crease_utils.Crease(
              point_utils.Point(x=0.1, y=0.1),
              point_utils.Point(x=0.3, y=0.3),
              fold=crease_utils.Fold.MOUNTAIN,
          ),
      ),
  )
  def test_merge_overlapping_creases(self, crease_1, crease_2, expected):
    result = shaping_utils._merge_overlapping_creases(crease_1, crease_2)
    self.assertTrue(result.isclose(expected))

  @parameterized.parameters([
      (1, 1, 2, 2, True),
      (0, 0, 2, 2, True),
      (8, 0, 2, 2, True),
      (0, 8, 2, 2, True),
      (8, 8, 2, 2, True),
      (0, 0, 10, 10, True),
      (-1, 1, 2, 2, True),
      (9, 1, 2, 2, True),
      (1, -1, 2, 2, True),
      (1, 9, 2, 2, True),
      (-3, 1, 2, 2, False),
      (11, 1, 2, 2, False),
      (1, -3, 2, 2, False),
      (1, 11, 2, 2, False),
      (-1, -1, 12, 12, False),
      (5, 5, 0, 0, True),
      (0, 5, 0, 0, True),
      (10, 10, 0, 0, True),
      (-1, 5, 0, 0, False),
      (1, 1, 8, 0, True),
      (1, 1, 0, 8, True),
      (-2, 1, 4, 0, True),
      (1, -2, 0, 4, True),
      (11, 1, 2, 0, False),
      (1, 11, 0, 2, False),
      (3, 1, -2, 2, True),
      (1, 3, 2, -2, True),
      (-1, -1, -2, -2, False),
  ])
  def test_is_corner_inside_or_on_border_10x10(
      self, x_start, y_start, x_len, y_len, expected
  ):
    rect = shaping_utils.Rectangle(
        name='test_rect',
        x_start=x_start,
        y_start=y_start,
        x_len=x_len,
        y_len=y_len,
    )
    self.assertEqual(expected, rect.is_corner_inside_or_on_border(10, 10))

  def test_normalize_rectangle_and_errors(self):
    rect = shaping_utils.Rectangle(
        name='test_rect',
        x_start=0,
        y_start=0,
        x_len=10,
        y_len=10,
        circle_centers=[
            point_utils.Point(x=5, y=5, require_on_paper=False),
        ],
    )
    normalized = rect.normalize_rectangle(10)
    expected = shaping_utils.Rectangle(
        name='test_rect',
        x_start=0,
        y_start=0,
        x_len=1,
        y_len=1,
        circle_centers=[
            point_utils.Point(x=0.5, y=0.5, require_on_paper=False),
        ],
    )
    self.assertEqual(expected, normalized)
    with self.assertRaises(ValueError):
      rect.normalize_rectangle(grid_size=-1)

  def test_from_shaping_rect(self):
    shaping_rect = rect_utils.Rectangle(
        name='test_rect', x_start=0.1, y_start=0.2, x_len=0.3, y_len=0.4
    )
    rect = shaping_utils.Rectangle.from_shaping_rect(shaping_rect)
    self.assertEqual(rect.name, 'test_rect')
    self.assertAlmostEqual(rect.x_start, 0.1)
    self.assertAlmostEqual(rect.y_start, 0.2)
    self.assertAlmostEqual(rect.x_len, 0.3)
    self.assertAlmostEqual(rect.y_len, 0.4)

  @parameterized.named_parameters(
      dict(
          testcase_name='internal',
          x_start=1,
          y_start=1,
          x_len=1,
          y_len=1,
          grid_size=10,
          expected=shaping_utils.FlapType.INTERNAL,
      ),
      dict(
          testcase_name='edge',
          x_start=0,
          y_start=1,
          x_len=1,
          y_len=1,
          grid_size=10,
          expected=shaping_utils.FlapType.EDGE,
      ),
      dict(
          testcase_name='corner',
          x_start=0,
          y_start=0,
          x_len=1,
          y_len=1,
          grid_size=10,
          expected=shaping_utils.FlapType.CORNER,
      ),
  )
  def test_flap_type(self, x_start, y_start, x_len, y_len, grid_size, expected):
    rect = shaping_utils.Rectangle(
        name='test_rect',
        x_start=x_start,
        y_start=y_start,
        x_len=x_len,
        y_len=y_len,
    )
    self.assertEqual(rect.flap_type(grid_size), expected)

  @parameterized.parameters(
      (math.pi / 24,),
      (math.pi / 12,),
      (math.pi / 6,),
      (math.pi / 3,),
      (math.pi / 2,),
      (math.pi * 5 / 6,),
      (math.pi * 2 / 3,),
      (math.pi * 7 / 12,),
      (math.pi * 5 / 12,),
      (math.pi * 11 / 24,),
  )
  def test_fold_angle_from_depth(self, angle):
    depth = shaping_utils.zigzag_depth_from_fold_angle(angle)
    self.assertAlmostEqual(
        angle, shaping_utils.fold_angle_from_zigzag_depth(depth)
    )

  def test_restrict_crease_to_rectangles(self):
    rects_overlapping = [
        shaping_utils.Rectangle(
            name='r1', x_start=0, y_start=0, x_len=2, y_len=2
        ),
        shaping_utils.Rectangle(
            name='r2', x_start=1, y_start=0, x_len=2, y_len=2
        ),
    ]
    crease = crease_utils.Crease(
        point_utils.Point(x=0.5, y=1, require_on_paper=False),
        point_utils.Point(x=2.5, y=1, require_on_paper=False),
        fold=crease_utils.Fold.MOUNTAIN,
    )
    result = shaping_utils.restrict_crease_to_rectangles(
        rects_overlapping, crease
    )
    self.assertLen(result, 1)
    self.assertTrue(
        result[0].p1.isclose(
            point_utils.Point(x=0.5, y=1, require_on_paper=False)
        )
    )
    self.assertTrue(
        result[0].p2.isclose(
            point_utils.Point(x=2.5, y=1, require_on_paper=False)
        )
    )

    rects_separate = [
        shaping_utils.Rectangle(
            name='r1', x_start=0, y_start=0, x_len=1, y_len=2
        ),
        shaping_utils.Rectangle(
            name='r2', x_start=2, y_start=0, x_len=1, y_len=2
        ),
    ]
    result_sep = shaping_utils.restrict_crease_to_rectangles(
        rects_separate, crease
    )
    self.assertLen(result_sep, 2)

  def test_flap_hierarchy_nested_and_disjoint(self):
    r0 = shaping_utils.Rectangle(
        name='r0', x_start=0, y_start=0, x_len=10, y_len=10
    )
    r1 = shaping_utils.Rectangle(
        name='r1', x_start=1, y_start=1, x_len=8, y_len=8
    )
    r2 = shaping_utils.Rectangle(
        name='r2', x_start=2, y_start=2, x_len=6, y_len=6
    )
    r3 = shaping_utils.Rectangle(
        name='r3', x_start=20, y_start=20, x_len=1, y_len=1
    )
    hierarchy = shaping_utils.FlapHierarchy([r0, r1, r2, r3])
    self.assertEqual(hierarchy.parent_map['r0'], r0)
    self.assertEqual(hierarchy.parent_map['r1'], r0)
    self.assertEqual(hierarchy.parent_map['r2'], r0)
    self.assertEqual(hierarchy.parent_map['r3'], r3)
    res0 = hierarchy.get_root_flap_and_distance('r0')
    self.assertIsNotNone(res0)
    self.assertEqual(res0[0], 'r0')
    self.assertAlmostEqual(res0[1], 0.0)
    res1 = hierarchy.get_root_flap_and_distance('r1')
    self.assertIsNotNone(res1)
    self.assertEqual(res1[0], 'r0')
    self.assertAlmostEqual(res1[1], 1.0)
    res2 = hierarchy.get_root_flap_and_distance('r2')
    self.assertIsNotNone(res2)
    self.assertEqual(res2[0], 'r0')
    self.assertAlmostEqual(res2[1], 2.0)
    res3 = hierarchy.get_root_flap_and_distance('r3')
    self.assertIsNotNone(res3)
    self.assertEqual(res3[0], 'r3')
    self.assertAlmostEqual(res3[1], 0.0)


class ShapingElementsTest(parameterized.TestCase):

  @parameterized.named_parameters([
      dict(
          testcase_name='square_flap',
          x_start=0,
          y_start=0,
          x_len=1,
          y_len=1,
          circle_centers=[point_utils.Point(x=0.5, y=0.5)],
          grid_size=8,
          expected=True,
      ),
      dict(
          testcase_name='wide_flap',
          x_start=0,
          y_start=0,
          x_len=1,
          y_len=0.5,
          circle_centers=[
              point_utils.Point(x=0.25, y=0.25),
              point_utils.Point(x=0.75, y=0.25),
          ],
          grid_size=8,
          expected=True,
      ),
      dict(
          testcase_name='bad_square',
          x_start=0,
          y_start=0,
          x_len=1,
          y_len=1,
          circle_centers=[point_utils.Point(x=0.25, y=0.25)],
          grid_size=8,
          expected=False,
      ),
  ])
  def test_shaping_flap_init_and_shrink(
      self, x_start, y_start, x_len, y_len, circle_centers, grid_size, expected
  ):
    if expected:
      flap = shaping_elements.ShapingFlap(
          name='test_flap',
          x_start=x_start,
          y_start=y_start,
          x_len=x_len,
          y_len=y_len,
          _circle_centers=circle_centers,
          grid_size=grid_size,
      )
      flap_copy = copy.deepcopy(flap)
      self.assertEqual(flap_copy, flap)
    else:
      with self.assertRaises(ValueError):
        shaping_elements.ShapingFlap(
            name='test_flap',
            x_start=x_start,
            y_start=y_start,
            x_len=x_len,
            y_len=y_len,
            _circle_centers=circle_centers,
            grid_size=grid_size,
        )

  def test_shaping_river_init_and_ridges(self):
    river = shaping_elements.ShapingRiver(
        name='test_river',
        bank_1=[
            crease_utils.Crease(
                point_utils.Point(x=0, y=4 / 32),
                point_utils.Point(x=12 / 32, y=4 / 32),
                fold=crease_utils.Fold.HINGE,
            ),
            crease_utils.Crease(
                point_utils.Point(x=12 / 32, y=4 / 32),
                point_utils.Point(x=12 / 32, y=0),
                fold=crease_utils.Fold.HINGE,
            ),
        ],
        bank_2=[
            crease_utils.Crease(
                point_utils.Point(x=0, y=7 / 32),
                point_utils.Point(x=15 / 32, y=7 / 32),
                fold=crease_utils.Fold.HINGE,
            ),
            crease_utils.Crease(
                point_utils.Point(x=15 / 32, y=7 / 32),
                point_utils.Point(x=15 / 32, y=0),
                fold=crease_utils.Fold.HINGE,
            ),
        ],
        width=4,
        grid_size=32,
    )
    expected_ridges = [
        crease_utils.Crease(
            point_utils.Point(x=12 / 32, y=4 / 32),
            point_utils.Point(x=15 / 32, y=7 / 32),
            fold=crease_utils.Fold.MOUNTAIN,
        )
    ]
    self.assertEqual(river.ridges, expected_ridges)
    river_copy = copy.deepcopy(river)
    self.assertEqual(river_copy.name, river.name)
    self.assertEqual(river_copy.ridges, expected_ridges)

  def test_open_shaping_boundary_edge_index_and_walk(self):
    grid_size = 4
    single_edge_river = shaping_elements.ShapingRiver(
        name='horizontal_river',
        bank_1=[
            crease_utils.Crease(
                point_utils.Point(x=0.0, y=0.5),
                point_utils.Point(x=1.0, y=0.5),
                fold=crease_utils.Fold.HINGE,
            )
        ],
        bank_2=[
            crease_utils.Crease(
                point_utils.Point(x=0.0, y=0.75),
                point_utils.Point(x=1.0, y=0.75),
                fold=crease_utils.Fold.HINGE,
            )
        ],
        width=1,
        grid_size=grid_size,
    )
    boundary = shaping_elements.get_shaping_boundary(single_edge_river, 'FIRST')
    self.assertFalse(boundary.closed)

    start = point_utils.Point(x=0.0, y=0.5)
    middle = point_utils.Point(x=0.5, y=0.5)
    end = point_utils.Point(x=1.0, y=0.5)
    off = point_utils.Point(x=0.5, y=0.25)

    self.assertEqual(boundary.get_edge_index(start), 0)
    self.assertEqual(boundary.get_edge_index(middle), 0)
    self.assertEqual(boundary.get_edge_index(end), 0)
    self.assertIsNone(boundary.get_edge_index(off))

    # Walking directly from the terminal tip of an open boundary must fail
    # immediately rather than taking a zero-length step.
    end_edge_index = boundary.get_edge_index(end)
    assert end_edge_index is not None
    with self.assertRaises(ValueError):
      boundary.walk(end, end_edge_index, grid_size)

    # Full walk from start reaches end at step 4 and terminates on step 5.
    cur_p = start
    cur_edge = boundary.get_edge_index(start)
    assert cur_edge is not None
    visited = [cur_p]
    for _ in range(grid_size):
      cur_p, cur_edge = boundary.walk(cur_p, cur_edge, grid_size)
      visited.append(cur_p)
    self.assertTrue(visited[-1].isclose(end))
    self.assertEqual(cur_edge, 1)
    with self.assertRaises(ValueError):
      boundary.walk(cur_p, cur_edge, grid_size)

    # Multi-edge L-bend open boundary maps its terminal tip to the last edge.
    l_bend_river = shaping_elements.ShapingRiver(
        name='l_bend_river',
        bank_1=[
            crease_utils.Crease(
                point_utils.Point(x=0.0, y=0.5),
                point_utils.Point(x=0.5, y=0.5),
                fold=crease_utils.Fold.HINGE,
            ),
            crease_utils.Crease(
                point_utils.Point(x=0.5, y=0.5),
                point_utils.Point(x=0.5, y=0.0),
                fold=crease_utils.Fold.HINGE,
            ),
        ],
        bank_2=[
            crease_utils.Crease(
                point_utils.Point(x=0.0, y=0.75),
                point_utils.Point(x=0.75, y=0.75),
                fold=crease_utils.Fold.HINGE,
            ),
            crease_utils.Crease(
                point_utils.Point(x=0.75, y=0.75),
                point_utils.Point(x=0.75, y=0.0),
                fold=crease_utils.Fold.HINGE,
            ),
        ],
        width=1,
        grid_size=grid_size,
    )
    l_boundary = shaping_elements.get_shaping_boundary(l_bend_river, 'FIRST')
    self.assertEqual(
        l_boundary.get_edge_index(point_utils.Point(x=0.5, y=0.5)), 1
    )
    self.assertEqual(
        l_boundary.get_edge_index(point_utils.Point(x=0.5, y=0.0)), 1
    )

  def test_inscribe_cut_line_open_river_and_border_clipped_flap(self):
    river = shaping_elements.ShapingRiver(
        name='body_river',
        bank_1=[
            crease_utils.Crease(
                point_utils.Point(x=0.0, y=0.5),
                point_utils.Point(x=1.0, y=0.5),
                fold=crease_utils.Fold.HINGE,
            )
        ],
        bank_2=[
            crease_utils.Crease(
                point_utils.Point(x=0.0, y=0.75),
                point_utils.Point(x=1.0, y=0.75),
                fold=crease_utils.Fold.HINGE,
            )
        ],
        width=2,
        grid_size=8,
        _outer_bank_index='FIRST',
    )

    left_anchor, left_shrunk, left_inscribed = (
        geometric_fold_3d_heuristics_utils.inscribe_cut_line(
            point_utils.Point(x=0.0, y=0.5), river, inscribe_depth=0.1
        )
    )
    right_anchor, right_shrunk, right_inscribed = (
        geometric_fold_3d_heuristics_utils.inscribe_cut_line(
            point_utils.Point(x=1.0, y=0.5), river, inscribe_depth=0.1
        )
    )

    self.assertTrue(left_inscribed)
    self.assertTrue(right_inscribed)
    self.assertTrue(left_anchor.on_paper)
    self.assertTrue(right_anchor.on_paper)
    self.assertAlmostEqual(left_anchor.x, 0.25)
    self.assertAlmostEqual(left_anchor.y, 0.5125)
    self.assertAlmostEqual(right_anchor.x, 0.75)
    self.assertAlmostEqual(right_anchor.y, 0.5125)
    self.assertIsInstance(left_shrunk, shaping_elements.ShapingRiver)
    self.assertIsInstance(right_shrunk, shaping_elements.ShapingRiver)

    # A 1-grid-unit terminal segment stays strictly inside the segment rather
    # than overshooting past the bend corner.
    short_stub_river = shaping_elements.ShapingRiver(
        name='stub_river',
        bank_1=[
            crease_utils.Crease(
                point_utils.Point(x=7 / 8, y=0.0),
                point_utils.Point(x=7 / 8, y=1 / 8),
                fold=crease_utils.Fold.HINGE,
            ),
            crease_utils.Crease(
                point_utils.Point(x=7 / 8, y=1 / 8),
                point_utils.Point(x=1.0, y=1 / 8),
                fold=crease_utils.Fold.HINGE,
            ),
        ],
        bank_2=[
            crease_utils.Crease(
                point_utils.Point(x=5 / 8, y=0.0),
                point_utils.Point(x=5 / 8, y=3 / 8),
                fold=crease_utils.Fold.HINGE,
            ),
            crease_utils.Crease(
                point_utils.Point(x=5 / 8, y=3 / 8),
                point_utils.Point(x=1.0, y=3 / 8),
                fold=crease_utils.Fold.HINGE,
            ),
        ],
        width=2,
        grid_size=8,
        _outer_bank_index='FIRST',
    )
    stub_anchor, _, stub_inscribed = (
        geometric_fold_3d_heuristics_utils.inscribe_cut_line(
            point_utils.Point(x=1.0, y=1 / 8),
            short_stub_river,
            inscribe_depth=0.1,
        )
    )
    self.assertTrue(stub_inscribed)
    self.assertTrue(stub_anchor.on_paper)
    self.assertAlmostEqual(stub_anchor.x, 7.5 / 8)
    self.assertAlmostEqual(stub_anchor.y, 1.1 / 8)

    # A radius-1 border flap whose corner sits 1 grid unit from x=1.0 must keep
    # its inscribed anchor strictly on-paper inside the shrunk flap rather than
    # shifting +2 grid units off the right edge of the paper.
    border_flap = shaping_elements.ShapingFlap(
        name='right_ear',
        x_start=21 / 22,
        y_start=18 / 22,
        x_len=2 / 22,
        y_len=4 / 22,
        _circle_centers=[
            point_utils.Point(x=1.0, y=19 / 22, require_on_paper=False),
            point_utils.Point(x=1.0, y=21 / 22, require_on_paper=False),
        ],
        grid_size=22,
    )
    flap_anchor, shrunk_flap, flap_inscribed = (
        geometric_fold_3d_heuristics_utils.inscribe_cut_line(
            point_utils.Point(x=21 / 22, y=18 / 22),
            border_flap,
            inscribe_depth=0.1,
        )
    )
    self.assertTrue(flap_inscribed)
    self.assertTrue(flap_anchor.on_paper)
    self.assertTrue(shrunk_flap.is_internal_point(flap_anchor))
    self.assertAlmostEqual(flap_anchor.x, 21.55 / 22)
    self.assertAlmostEqual(flap_anchor.y, 18.1 / 22)

  def test_shape_multi_river_open_boundary_with_elevation_bend(self):
    # Regression test for CF-3(a): a multi-river figure whose inner horizontal
    # river spans x=0..1 and bends in elevation relative to its downstream child
    # exercises `inscribe_cut_line` at the open river's terminal endpoint x=1.0.
    figure = bp.StickFigure.from_dict({
        'name': 'stag_deer_repro',
        'category': 'animals',
        'complexity': 3,
        'root': 'body',
        'children': [
            {
                'name': 'body',
                'start': 'hips',
                'end': 'shoulders',
                'length': 4,
                'azimuth_angle': 0.0,
                'elevation_angle': 0.0,
            },
            {
                'name': 'neck',
                'start': 'shoulders',
                'end': 'crown',
                'length': 2,
                'azimuth_angle': 0.0,
                'elevation_angle': 55.0,
            },
            {
                'name': 'snout',
                'start': 'crown',
                'end': 'nose',
                'length': 2,
                'azimuth_angle': 0.0,
                'elevation_angle': 10.0,
            },
            {
                'name': 'left_ear',
                'start': 'crown',
                'end': 'left_ear_tip',
                'length': 1,
                'azimuth_angle': 85.0,
                'elevation_angle': 35.0,
            },
            {
                'name': 'right_ear',
                'start': 'crown',
                'end': 'right_ear_tip',
                'length': 1,
                'azimuth_angle': -85.0,
                'elevation_angle': 35.0,
            },
            {
                'name': 'left_antler_base',
                'start': 'crown',
                'end': 'left_antler_mid',
                'length': 2,
                'azimuth_angle': 65.0,
                'elevation_angle': 60.0,
            },
            {
                'name': 'left_antler_tip',
                'start': 'left_antler_mid',
                'end': 'left_antler_end',
                'length': 2,
                'azimuth_angle': 35.0,
                'elevation_angle': 80.0,
            },
            {
                'name': 'right_antler_base',
                'start': 'crown',
                'end': 'right_antler_mid',
                'length': 2,
                'azimuth_angle': -65.0,
                'elevation_angle': 60.0,
            },
            {
                'name': 'right_antler_tip',
                'start': 'right_antler_mid',
                'end': 'right_antler_end',
                'length': 2,
                'azimuth_angle': -35.0,
                'elevation_angle': 80.0,
            },
            {
                'name': 'front_left_leg',
                'start': 'shoulders',
                'end': 'front_left_hoof',
                'length': 4,
                'azimuth_angle': 90.0,
                'elevation_angle': -75.0,
            },
            {
                'name': 'front_right_leg',
                'start': 'shoulders',
                'end': 'front_right_hoof',
                'length': 4,
                'azimuth_angle': -90.0,
                'elevation_angle': -75.0,
            },
            {
                'name': 'rear_left_leg',
                'start': 'hips',
                'end': 'rear_left_hoof',
                'length': 4,
                'azimuth_angle': 90.0,
                'elevation_angle': -75.0,
            },
            {
                'name': 'rear_right_leg',
                'start': 'hips',
                'end': 'rear_right_hoof',
                'length': 4,
                'azimuth_angle': -90.0,
                'elevation_angle': -75.0,
            },
            {
                'name': 'tail',
                'start': 'hips',
                'end': 'tail_tip',
                'length': 2,
                'azimuth_angle': 180.0,
                'elevation_angle': 35.0,
            },
        ],
    })
    design = bp.DesignTree.from_stick_figure(figure)
    grid_size = 22
    packing_outputs = utils_solver.find_tiling_and_generate_crease_pattern(
        design.merged, grid_size, is_symmetric=True
    )
    self.assertNotEmpty(packing_outputs)
    packing_output = packing_outputs[0]
    solver_results = hinge_utils.cp_from_packing(
        cp=packing_output.packing_cp,
        max_number_of_cps=1,
        grid_size=grid_size,
        check_global_flat_foldability=False,
        tie_break_by_hinge_symmetry=True,
        packing_information=packing_output,
    )
    self.assertNotEmpty(solver_results)
    packed = bp.PackedCreasePattern.from_packing_output(
        packing_output, design=design
    )
    solved = bp.SolvedCreasePattern.from_solver_result(
        solver_results[0], packed=packed
    )
    shaped = bp.shape(solved, bp.ShapingTaskConfig.default(max_number_of_cps=2))
    self.assertIsInstance(shaped, bp.ShapedCreasePattern)
    self.assertTrue(shaped.is_locally_flat_foldable)


class HeuristicsUnitTest(parameterized.TestCase):

  def test_human_simple_shaping_angles(self):
    figure = example_trees.HUMAN_SIMPLE
    parent_stick, relative_angles = heuristics.shaping_angles(figure)
    self.assertEqual(parent_stick.name, 'torso')
    self.assertIn('torso', relative_angles)
    self.assertEqual(relative_angles.pop('torso'), (None, 0.0, 0.0))
    expected_angles_degrees = {
        'head': ('torso', -180.0, 180.0),
        'right_arm': ('torso', 15.0, 70.0),
        'left_arm': ('torso', -15.0, 70.0),
        'right_leg': ('torso', -90.0, -150.0),
        'left_leg': ('torso', 90.0, -150.0),
    }
    self.assertLen(relative_angles, len(expected_angles_degrees))
    for k, (p, az, el) in expected_angles_degrees.items():
      self.assertEqual(relative_angles[k][0], p)
      self.assertAlmostEqual(relative_angles[k][1], math.radians(az))
      self.assertAlmostEqual(relative_angles[k][2], math.radians(el))

  @parameterized.named_parameters(
      dict(
          testcase_name='human_simple',
          tree=example_trees.HUMAN_SIMPLE,
          expected='torso',
      ),
      dict(
          testcase_name='scorpion_example',
          tree=example_trees.SCORPION_EXAMPLE,
          expected='tail',
      ),
      dict(
          testcase_name='cat_simple',
          tree=example_trees.CAT_SIMPLE,
          expected='body',
      ),
      dict(
          testcase_name='francesco_example',
          tree=example_trees.FRANCESCO_EXAMPLE,
          expected='11a14a',
      ),
  )
  def test_get_first_stick_for_shaping(self, tree, expected):
    got = heuristics.get_first_stick_for_shaping(tree=tree)
    self.assertEqual(expected, got.name)

  def test_shaping_angles_human_example_with_hinge_info(self):
    stick_figure = example_trees.HUMAN_SIMPLE
    hinge_info_list = [
        packing_types.HingeInfo(
            coordinate=None, assigned=False, flap_names=('left_arm', 'head')
        ),
        packing_types.HingeInfo(
            coordinate=None, assigned=False, flap_names=('right_arm', 'head')
        ),
        packing_types.HingeInfo(
            coordinate=None, assigned=True, flap_names=('right_arm', 'torso')
        ),
        packing_types.HingeInfo(
            coordinate=None, assigned=False, flap_names=('head', 'torso')
        ),
        packing_types.HingeInfo(
            coordinate=None, assigned=True, flap_names=('left_arm', 'torso')
        ),
        packing_types.HingeInfo(
            coordinate=None, assigned=False, flap_names=('left_leg', 'torso')
        ),
        packing_types.HingeInfo(
            coordinate=None, assigned=False, flap_names=('right_leg', 'torso')
        ),
        packing_types.HingeInfo(
            coordinate=None, assigned=True, flap_names=('left_leg', 'right_leg')
        ),
    ]
    expected_angles = {
        'torso': None,
        'left_leg': -1.570796326,
        'right_leg': 1.570796326,
        'head': 0.0,
        'left_arm': -0.2617993877991494,
        'right_arm': 0.2617993877991494,
    }
    expected_directions = {
        'torso': heuristics.AbsFlapDirection.REFERENCE,
        'left_leg': heuristics.AbsFlapDirection.DOWN,
        'right_leg': heuristics.AbsFlapDirection.DOWN,
        'head': heuristics.AbsFlapDirection.UP,
        'left_arm': heuristics.AbsFlapDirection.DOWN,
        'right_arm': heuristics.AbsFlapDirection.DOWN,
    }
    flap_dict = heuristics.get_flap_info_dict_with_hinge_info(
        stick_figure,
        hinge_info_list,
        root_direction_follow_stick_figure_root=True,
    )
    for stick_name, expected_angle in expected_angles.items():
      self.assertAlmostEqual(
          flap_dict[stick_name].shape_angle, expected_angle, places=8
      )
    for stick_name, expected_dir in expected_directions.items():
      self.assertEqual(flap_dict[stick_name].absolute_direction, expected_dir)

  def test_unmerged_stick_figure_on_vaulted_canopy(self):
    unmerged_stick_figure = examples.VALUTED_CANOPY.unmerged_stick_figure
    self.assertIsNotNone(unmerged_stick_figure)
    merged_stick_figure, merged_stick_name_dict = (
        unmerged_stick_figure.merge_linear_chains()
    )
    grid_size = examples.VALUTED_CANOPY.grid_size
    packing_outputs = utils_solver.find_tiling_and_generate_crease_pattern(
        merged_stick_figure, grid_size, is_symmetric=True
    )
    self.assertNotEmpty(packing_outputs)
    packing_output = packing_outputs[0]
    solver_results = hinge_utils.cp_from_packing(
        cp=packing_output.packing_cp,
        max_number_of_cps=1,
        grid_size=grid_size,
        check_global_flat_foldability=False,
        tie_break_by_hinge_symmetry=True,
    )
    flap_info = heuristics.get_flap_info_dict(
        merged_stick_figure.align_root_stick_to_target_azimuth_elevation(
            target_azimuth=0, target_elevation=0
        ),
        packing_output,
        solver_results[0],
        root_direction_follow_stick_figure_root=True,
    )
    hinge_info_list = heuristics.copy_hinge_info_list(packing_output.hinges)
    hinge_info_list = heuristics.update_hinge_info_from_solver_result(
        hinge_info_list, solver_results[0]
    )
    unmerged_flap_info = (
        heuristics.get_flap_info_dict_for_unmerged_stick_figure(
            unmerged_stick_figure.align_root_stick_to_target_azimuth_elevation(
                target_azimuth=0, target_elevation=0
            ),
            merged_stick_name_dict,
            hinge_info_list,
            flap_info,
            True,
        )
    )
    self.assertEqual(
        unmerged_flap_info['central_apex_post'].fold_state,
        heuristics.FoldState.REFERENCE,
    )
    self.assertAlmostEqual(
        unmerged_flap_info['vault_rib_rear_left'].shape_angle,
        -2.3879825837347166,
        places=8,
    )
    self.assertAlmostEqual(
        unmerged_flap_info['pillar_front_right'].shape_angle,
        0.7536100698550765,
        places=8,
    )

  def test_apply_small_angle_flip_and_geometric_helpers(self):
    split_angles = heuristics_utils.apply_small_angle_flip(
        heuristics_utils.MIN_ANGLE / 2
    )
    self.assertLen(split_angles, 2)
    self.assertAlmostEqual(
        split_angles[0],
        heuristics_utils.LARGE_ANGLE,
    )
    self.assertEqual(
        heuristics_utils.apply_small_angle_flip(math.pi / 2),
        [math.pi / 2],
    )
    self.assertAlmostEqual(
        geometric_fold_3d_heuristics._clip_by_min_inscribe_depth(0.0, 48),
        1e-3 * 48 + 1e-5,
    )
    self.assertEqual(
        geometric_simple_fold.interpolate(
            point1=(0, 0), point2=(10, 10), t=0.5
        ),
        (5.0, 5.0),
    )
    new_creases = geometric_simple_fold.create_new_creases_from_cuts(
        selected_face_indices=[0, 1],
        face_cuts={
            0: ((0.1, 0.1), (0.2, 0.2)),
            1: ((0.3, 0.3), (0.4, 0.4)),
        },
        face_orientations=[False, True],
        fold_assignment=crease_utils.Fold.MOUNTAIN,
    )
    self.assertLen(new_creases, 2)
    self.assertEqual(new_creases[0].fold, crease_utils.Fold.MOUNTAIN)
    self.assertEqual(new_creases[1].fold, crease_utils.Fold.VALLEY)

  def test_check_validity_allows_split_sticks_of_undetermined_merged_stick(
      self,
  ):
    def _make_node(
        name: str,
        *,
        fold_state: heuristics.FoldState,
        absolute_direction: heuristics.AbsFlapDirection,
        is_root: bool = False,
    ) -> heuristics.Node:
      return heuristics.Node(
          name=name,
          length=1,
          is_root=is_root,
          relative_azimuth_angle=0.0,
          relative_elevation_angle=0.0,
          hinges=[],
          root_direction_follow_stick_figure_root=True,
          fold_state=fold_state,
          absolute_direction=absolute_direction,
      )

    node_dict = {
        'm': _make_node(
            'm',
            fold_state=heuristics.FoldState.FOLD_IN,
            absolute_direction=heuristics.AbsFlapDirection.NOT_DETERMINED,
        )
    }
    unmerged_node_dict = {
        's1': _make_node(
            's1',
            fold_state=heuristics.FoldState.FOLD_IN,
            absolute_direction=heuristics.AbsFlapDirection.NOT_DETERMINED,
        ),
        's2': _make_node(
            's2',
            fold_state=heuristics.FoldState.FOLD_OUT,
            absolute_direction=heuristics.AbsFlapDirection.NOT_DETERMINED,
        ),
    }
    heuristics.check_validity(
        node_dict, unmerged_node_dict, {'m': ['s1', 's2']}
    )

    determined_node_dict = {
        'm': _make_node(
            'm',
            fold_state=heuristics.FoldState.FOLD_IN,
            absolute_direction=heuristics.AbsFlapDirection.UP,
        )
    }
    with self.assertRaisesRegex(ValueError, 'NOT_DETERMINED'):
      heuristics.check_validity(
          determined_node_dict, unmerged_node_dict, {'m': ['s1', 's2']}
      )

    bad_fold_state_dict = {
        's1': _make_node(
            's1',
            fold_state=heuristics.FoldState.NOT_DETERMINED,
            absolute_direction=heuristics.AbsFlapDirection.NOT_DETERMINED,
        )
    }
    with self.assertRaisesRegex(ValueError, 'fold_state NOT_DETERMINED'):
      heuristics.check_validity(node_dict, bad_fold_state_dict, {'m': ['s1']})

  def test_get_face_intersections_and_folded_pos_skip_degenerate_edges(self):
    # Quad whose folded corners include a sub-tolerance edge (0.5, 0.25) ->
    # (0.5, 0.2504) produced during sub-face narrowing.
    intersections = (
        geometric_simple_fold.get_face_intersections_in_unfolded_plane(
            face_vertex_indices=[0, 1, 2, 3],
            folded_vertices=[
                (0.0, 0.0),
                (0.5, 0.25),
                (0.5, 0.2504),
                (0.0, 0.5),
            ],
            unfolded_vertices=[
                (0.0, 0.0),
                (0.5, 0.25),
                (0.5, 0.26),
                (0.0, 0.5),
            ],
            cut_line_start=(0.25, -1.0),
            cut_line_end=(0.25, 1.0),
        )
    )
    self.assertLen(intersections, 2)

    folded_pos = geometric_utils.get_folded_pos(
        point=point_utils.Point(x=0.25, y=0.25),
        unfolded_coords=[
            (0.0, 0.0),
            (0.0, 0.0004),
            (1.0, 0.0),
            (1.0, 1.0),
            (0.0, 1.0),
        ],
        folded_coords=[
            (0.0, 0.0),
            (0.0, 0.0004),
            (1.0, 0.0),
            (1.0, 1.0),
            (0.0, 1.0),
        ],
        edge_vertices=[],
        face_vertices=[[0, 1, 2, 3, 4]],
    )
    self.assertAlmostEqual(folded_pos[0], 0.25, places=3)
    self.assertAlmostEqual(folded_pos[1], 0.25, places=3)

  def test_root_reference_orientation_priority_and_base_signs(self):
    fold_res = bp.fold('human', preset='demo')
    solved = fold_res.solved
    packing_out, solver_res = bus_adapter.legacy_inputs(solved)
    flaps, rivers = (
        shaping_elements.convert_packing_elements_to_shaping_elements(
            packing_rectangles=packing_out.flap_rectangles,
            packing_rivers=packing_out.packing_rivers,
            packing_grid_size=packing_out.packing_cp.grid_size,
            solving_grid_size=solved.creases.grid_size,
            base_cp=solved.creases,
        )
    )
    elements = {e.name: e for e in (*flaps, *rivers)}
    stick_fig = solved.packed.design.source
    root_el = elements[stick_fig.root]
    if isinstance(root_el, shaping_elements.ShapingRiver):
      outer_bank = root_el.get_outer_bank_index(
          stick_fig, stick_fig.root, list(flaps), list(rivers)
      )
      root_el = root_el.update_outer_bank_index(outer_bank)
    ref_pt = geometric_fold_3d_heuristics.root_reference_point(root_el)
    self.assertIsNotNone(ref_pt)
    self.assertTrue(
        root_el.is_internal_point(
            point_utils.Point(x=ref_pt[0], y=ref_pt[1], require_on_paper=False),
            strict=True,
        )
    )

    # Planar Z-rotation reflects x before rotating when root_flipped is False
    # and the folded root points upward (uy < 0).
    graph_data = geometric_simple_fold.prepare_graph_data(solved.creases)
    folded_verts = graph_data['folded_vertices']
    d_unflipped = geometric_fold_3d_heuristics._rotate_simple_fold_direction_by_model_orientation(
        np.array([1.0, 0.5, 0.0]),
        root_el,
        solved.packed.design.source,
        graph_data,
        folded_verts,
        root_flipped=False,
    )
    d_flipped = geometric_fold_3d_heuristics._rotate_simple_fold_direction_by_model_orientation(
        np.array([1.0, 0.5, 0.0]),
        root_el,
        solved.packed.design.source,
        graph_data,
        folded_verts,
        root_flipped=True,
    )
    self.assertAlmostEqual(
        float(np.linalg.norm(d_unflipped)),
        float(np.linalg.norm(d_flipped)),
        places=6,
    )
    self.assertFalse(np.allclose(d_unflipped, d_flipped))

    # Empty graph_data raises ValueError cleanly.
    empty_graph = {
        'vertices_coords': [],
        'edges_vertices': [],
        'edge_vertices': [],
        'edges_assignment': [],
        'faces_vertices': [],
        'face_vertices': [],
        'face_orientations': [],
    }
    with self.assertRaises(ValueError):
      geometric_fold_3d_heuristics._get_root_element_orientation(
          root_el, solved.packed.design.source, empty_graph, []
      )

    # Custom element shaping priority + measure_base_directions + FF log check.
    hinges = heuristics.update_hinge_info_from_solver_result(
        packing_out.hinges, solver_res
    )
    hparams = geometric_fold_3d_heuristics.GeometricFolding3DHParams(
        priority=('right_arm', 'left_arm'),
        measure_base_directions=True,
        check_ff_violations=True,
    )
    res = geometric_fold_3d_heuristics.get_shaping_creases_with_3d_geometric_folding_heuristics(
        base_cp=solved.creases,
        stick_figure=solved.packed.design.source,
        shaping_flaps=flaps,
        shaping_rivers=rivers,
        hinge_info_list=hinges,
        central_crease_removed=solved.central_crease_removed,
        shape_with_unmerged_tree=False,
        hparams=hparams,
    )
    self.assertNotEmpty(res['shaping_creases'])

    # base_signs with sign=0.0 must be a no-op (must not collapse direction).
    shaping_data = heuristics_utils.get_heuristics_shaping_data(
        stick_figure=solved.packed.design.source,
        shaping_flaps=list(flaps),
        shaping_rivers=list(rivers),
        hinge_info_list=list(hinges),
        central_crease_removed=solved.central_crease_removed,
        shape_with_unmerged_tree=False,
    )
    posts_default = heuristics_3d.get_flap_post_dict(
        solved.packed.design.source, shaping_data['flap_info']
    )
    posts_zero_sign = heuristics_3d.get_flap_post_dict(
        solved.packed.design.source,
        shaping_data['flap_info'],
        base_signs={'left_arm': 0.0, 'right_arm': -1.0},
        unrotated={'head'},
    )
    self.assertEqual(
        len(posts_default['left_arm']), len(posts_zero_sign['left_arm'])
    )

    # Flat-foldability discard warning matches CLI log capture regex.
    bad_crease = crease_utils.Crease(
        point_utils.Point(x=0.23, y=0.37),
        point_utils.Point(x=0.61, y=0.79),
        fold=crease_utils.Fold.MOUNTAIN,
    )
    orig_shape = (
        geometric_fold_3d_heuristics.shape_element_with_3d_geometric_folding_heuristics
    )

    def _inject_bad(*args, **kwargs):
      out = orig_shape(*args, **kwargs)
      elem = kwargs.get('element') or args[0]
      if elem.name == 'left_arm':
        return {**out, 'shaping_creases': [*out['shaping_creases'], bad_crease]}
      return out

    with (
        mock.patch.object(
            geometric_fold_3d_heuristics,
            'shape_element_with_3d_geometric_folding_heuristics',
            side_effect=_inject_bad,
        ),
        self.assertLogs(level=logging.WARNING) as captured_logs,
    ):
      bp.shape(solved)
    self.assertTrue(
        any(
            'Flat foldability violation detected after shaping element:'
            ' left_arm. Discarding'
            in msg
            for msg in captured_logs.output
        )
    )

  @parameterized.named_parameters(
      dict(
          testcase_name='unflipped_root_up',
          root_flipped=False,
          root_orientation=(0.0, -1.0, 0.0),
          expected=(1.0, -0.5, 0.0),
      ),
      dict(
          testcase_name='unflipped_root_down',
          root_flipped=False,
          root_orientation=(0.0, 1.0, 0.0),
          expected=(1.0, 0.5, 0.0),
      ),
      dict(
          testcase_name='flipped_root_up',
          root_flipped=True,
          root_orientation=(0.0, -1.0, 0.0),
          expected=(-1.0, -0.5, 0.0),
      ),
      dict(
          testcase_name='flipped_root_down',
          root_flipped=True,
          root_orientation=(0.0, 1.0, 0.0),
          expected=(1.0, 0.5, 0.0),
      ),
      dict(
          testcase_name='unflipped_root_horizontal',
          root_flipped=False,
          root_orientation=(1.0, 0.0, 0.0),
          expected=(0.5, -1.0, 0.0),
      ),
  )
  def test_rotate_simple_fold_direction_four_quadrants(
      self,
      root_flipped: bool,
      root_orientation: tuple[float, float, float],
      expected: tuple[float, float, float],
  ):
    with mock.patch.object(
        geometric_fold_3d_heuristics,
        '_get_root_element_orientation',
        return_value=np.array(root_orientation, dtype=float),
    ):
      actual = geometric_fold_3d_heuristics._rotate_simple_fold_direction_by_model_orientation(
          np.array([1.0, 0.5, 0.0]),
          root_element=mock.Mock(),
          stick_figure=mock.Mock(),
          graph_data={},  # type: ignore[arg-type]
          folded_verts=[],
          root_flipped=root_flipped,
      )
    np.testing.assert_allclose(actual, np.array(expected), atol=1e-9)

  @parameterized.named_parameters(
      dict(
          testcase_name='human_unflipped_uy_neg',
          fixture_name='human',
          expected_root_flipped=False,
          expected_uy_sign=-1,
      ),
      dict(
          testcase_name='horned_owl_unflipped_uy_pos',
          fixture_name='horned_owl',
          expected_root_flipped=False,
          expected_uy_sign=1,
      ),
      dict(
          testcase_name='level_shifter_2_rivers_flipped_uy_pos',
          fixture_name='level_shifter_2_rivers',
          expected_root_flipped=True,
          expected_uy_sign=1,
      ),
      dict(
          testcase_name='spike_flipped_uy_neg',
          fixture_name='spike',
          expected_root_flipped=True,
          expected_uy_sign=-1,
      ),
  )
  def test_bp_shape_preserves_bilateral_limb_sides_and_pose_gate(
      self,
      fixture_name: str,
      expected_root_flipped: bool,
      expected_uy_sign: int,
  ):
    fixture_path = (
        pathlib.Path(__file__).parent
        / 'testdata'
        / f'{fixture_name}_solved.json'
    )
    solved = bp.loads(fixture_path.read_text(encoding='utf-8'))
    self.assertIsInstance(solved, bp.SolvedCreasePattern)

    # Verify the fixture exercises the expected (root_flipped, sign(uy)) case.
    stick_fig = solved.packed.design.source
    packing_out, _ = bus_adapter.legacy_inputs(solved)
    flaps, rivers = (
        shaping_elements.convert_packing_elements_to_shaping_elements(
            packing_rectangles=packing_out.flap_rectangles,
            packing_rivers=packing_out.packing_rivers,
            packing_grid_size=packing_out.packing_cp.grid_size,
            solving_grid_size=solved.creases.grid_size,
            base_cp=solved.creases,
        )
    )
    merged_fig, _ = stick_fig.merge_linear_chains()
    root_stick = heuristics.get_first_stick_for_shaping(merged_fig)
    root_el = shaping_elements.get_element_by_name(
        root_stick.name, flaps, rivers
    )
    if isinstance(root_el, shaping_elements.ShapingRiver):
      outer_bank = root_el.get_outer_bank_index(
          merged_fig, root_stick.name, flaps, rivers
      )
      root_el = root_el.update_outer_bank_index(outer_bank)
    graph_data = geometric_simple_fold.prepare_graph_data(solved.creases)
    root_flipped = geometric_fold_3d_heuristics._is_root_reference_flipped(
        root_el, graph_data
    )
    root_orient = geometric_fold_3d_heuristics._get_root_element_orientation(
        root_el, merged_fig, graph_data, graph_data['folded_vertices']
    )
    self.assertEqual(root_flipped, expected_root_flipped)
    self.assertGreater(root_orient[1] * expected_uy_sign, 0.0)

    shaped = bp.shape(solved)
    report = bp.evaluate_fidelity(shaped)
    self.assertTrue(
        report.pose_gate.passed,
        msg=(
            f'{fixture_name}: pose_gate={report.pose_gate.to_dict()}'
            f' (fidelity={report.fidelity:.2f})'
        ),
    )
    self.assertEmpty(report.pose_gate.crossing)


if __name__ == '__main__':
  absltest.main()
