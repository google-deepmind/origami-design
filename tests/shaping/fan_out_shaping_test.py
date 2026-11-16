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

"""Tests for fan out shaping."""

import dataclasses
from bp_pipeline.crease_pattern import crease as crease_utils
from bp_pipeline.crease_pattern import crease_pattern as cp_utils
from bp_pipeline.crease_pattern import point as point_utils
from shaping import shaping_test_fixtures as shaping_examples
from bp_pipeline.shaping import fan_out_shaping
from bp_pipeline.shaping import shaping_utils
from bp_pipeline.crease_pattern.parser import svg as parser
from absl.testing import absltest as googletest
from absl.testing import parameterized


class FanOutShapingTest(parameterized.TestCase):

  @parameterized.parameters([
      shaping_examples.PHOENIX_FAN_OUT_RIGHT_TAIL,
      shaping_examples.PHOENIX_FAN_OUT_LEFT_TAIL,
      shaping_examples.PHOENIX_FAN_OUT_BOTH_TAILS,
      # shaping_examples.VERTICAL_FAN_OUT_SHAPING_TEST_TOP_HALF,
      shaping_examples.VERTICAL_FAN_OUT_SHAPING_TEST_BOTH_HALVES,
      # shaping_examples.ZZ_PLANT_FAN_OUT,
  ])
  def test_fan_out_shaping(
      self, example: shaping_examples.BoxPleatingShapingExample
  ):
    cp = parser.parse_svg(
        example.base_example.base_svg, grid_size=example.base_example.grid_size,
        merge_trivial=True,
    )

    shaped_cp = fan_out_shaping.shape_cp_with_fan_out(
        cp, shaping_params=example.fan_out_shaping_params
    )

    self.assertIsNotNone(shaped_cp)
    self.assertTrue(shaped_cp.local_flat_foldability())
    self.assertTrue(example.check_fan_out_shaping(shaped_cp))


class GetFanOutCandidatesTest(parameterized.TestCase):

  def _get_phoenix_cp(self):
    example = shaping_examples.PHOENIX_FAN_OUT_RIGHT_TAIL
    return (
        parser.parse_svg(
            example.base_example.base_svg,
            grid_size=example.base_example.grid_size,
            merge_trivial=True,
        ),
        example,
    )

  def test_returns_valid_result(self):
    cp, example = self._get_phoenix_cp()
    candidates = fan_out_shaping._get_fan_out_candidates(
        cp, example.fan_out_shaping_params, example.base_example.grid_size
    )
    self.assertIsNotNone(candidates)
    self.assertNotEmpty(candidates.modified_creases)
    self.assertNotEmpty(candidates.unmodified_creases)
    self.assertNotEmpty(candidates.modified_rectangles)
    self.assertGreater(candidates.avg_modified_crease_length, 0)

  def test_returns_none_when_no_valid_creases(self):
    cp, example = self._get_phoenix_cp()
    params = [
        fan_out_shaping.FanOutShapingParams(
            valid_fold_types=[crease_utils.Fold.MOUNTAIN],
            valid_orientations=[fan_out_shaping.Orientation.HORIZONTAL],
            rectangles=[
                shaping_utils.Rectangle(
                    name='empty',
                    x_start=0.99,
                    y_start=0.99,
                    x_len=0.001,
                    y_len=0.001,
                )
            ],
            move_delta=-1.0,
        ),
    ]
    candidates = fan_out_shaping._get_fan_out_candidates(
        cp, params, example.base_example.grid_size
    )
    self.assertIsNone(candidates)

  def test_avg_length_computed_correctly(self):
    cp, example = self._get_phoenix_cp()
    candidates = fan_out_shaping._get_fan_out_candidates(
        cp, example.fan_out_shaping_params, example.base_example.grid_size
    )
    self.assertIsNotNone(candidates)
    expected_avg = sum(c.length for c in candidates.modified_creases) / len(
        candidates.modified_creases
    )
    self.assertAlmostEqual(candidates.avg_modified_crease_length, expected_avg)


class ShapeCpWithFanOutHeuristicTest(parameterized.TestCase):

  def _get_phoenix_cp_and_heuristic_params(self):
    example = shaping_examples.PHOENIX_FAN_OUT_RIGHT_TAIL
    cp = parser.parse_svg(
        example.base_example.base_svg,
        grid_size=example.base_example.grid_size,
        merge_trivial=True,
    )
    heuristic_params = [
        fan_out_shaping.FanOutShapingParams(
            valid_fold_types=[],
            valid_orientations=[],
            rectangles=param.rectangles,
            move_delta=param.move_delta,
        )
        for param in example.fan_out_shaping_params
    ]
    return cp, heuristic_params

  def test_heuristic_rejects_non_empty_fold_types(self):
    cp, _ = self._get_phoenix_cp_and_heuristic_params()
    params = [
        fan_out_shaping.FanOutShapingParams(
            valid_fold_types=[crease_utils.Fold.MOUNTAIN],
            valid_orientations=[],
            rectangles=[
                shaping_utils.Rectangle(
                    name='test',
                    x_start=0,
                    y_start=0,
                    x_len=1,
                    y_len=1,
                )
            ],
            move_delta=-1.0,
        ),
    ]
    with self.assertRaises(ValueError):
      fan_out_shaping.shape_cp_with_fan_out_heuristic(cp, shaping_params=params)

  def test_heuristic_rejects_non_empty_orientations(self):
    cp, _ = self._get_phoenix_cp_and_heuristic_params()
    params = [
        fan_out_shaping.FanOutShapingParams(
            valid_fold_types=[],
            valid_orientations=[fan_out_shaping.Orientation.HORIZONTAL],
            rectangles=[
                shaping_utils.Rectangle(
                    name='test',
                    x_start=0,
                    y_start=0,
                    x_len=1,
                    y_len=1,
                )
            ],
            move_delta=-1.0,
        ),
    ]
    with self.assertRaises(ValueError):
      fan_out_shaping.shape_cp_with_fan_out_heuristic(cp, shaping_params=params)

  def test_heuristic_success(self):
    cp, heuristic_params = self._get_phoenix_cp_and_heuristic_params()
    shaped_cp = fan_out_shaping.shape_cp_with_fan_out_heuristic(
        cp, shaping_params=heuristic_params
    )
    self.assertIsNotNone(shaped_cp)
    self.assertTrue(shaped_cp.local_flat_foldability())

  def test_heuristic_raises_for_impossible_params(self):
    cp, _ = self._get_phoenix_cp_and_heuristic_params()
    params = [
        fan_out_shaping.FanOutShapingParams(
            valid_fold_types=[],
            valid_orientations=[],
            rectangles=[
                shaping_utils.Rectangle(
                    name='empty',
                    x_start=0.99,
                    y_start=0.99,
                    x_len=0.001,
                    y_len=0.001,
                )
            ],
            move_delta=-1.0,
        ),
    ]
    with self.assertRaises(ValueError):
      fan_out_shaping.shape_cp_with_fan_out_heuristic(cp, shaping_params=params)

  def test_handles_rectangle_with_list_circle_centers(self):
    example = shaping_examples.PHOENIX_FAN_OUT_RIGHT_TAIL
    cp = parser.parse_svg(
        example.base_example.base_svg,
        grid_size=example.base_example.grid_size,
        merge_trivial=True,
    )
    list_rects = []
    for param in example.fan_out_shaping_params:
      for r in param.rectangles:
        list_rects.append(
            dataclasses.replace(
                r, circle_centers=[point_utils.Point(x=0.5, y=0.5)]
            )
        )
    params = [
        fan_out_shaping.FanOutShapingParams(
            valid_fold_types=[],
            valid_orientations=[],
            rectangles=list_rects,
            move_delta=-1.0,
        )
    ]
    shaped_cp = fan_out_shaping.shape_cp_with_fan_out_heuristic(
        cp, shaping_params=params
    )
    self.assertTrue(shaped_cp.local_flat_foldability())

  def test_splits_merged_border_pleat_in_light_mode(self):
    p = lambda x, y: point_utils.Point(x=x, y=y)
    borders = [
        crease_utils.Crease(p(0, 0), p(1, 0), fold=crease_utils.Fold.BORDER),
        crease_utils.Crease(p(1, 0), p(1, 1), fold=crease_utils.Fold.BORDER),
        crease_utils.Crease(p(1, 1), p(0, 1), fold=crease_utils.Fold.BORDER),
        crease_utils.Crease(p(0, 1), p(0, 0), fold=crease_utils.Fold.BORDER),
    ]
    # Vertical pleat merged from interior (0.5, 0.25) to bottom border (0.5, 1)
    # through vertex (0.5, 0.5).
    creases = borders + [
        crease_utils.Crease(
            p(0.5, 0.0), p(0.5, 0.25), fold=crease_utils.Fold.VALLEY
        ),
        crease_utils.Crease(
            p(0.0, 0.25), p(0.5, 0.25), fold=crease_utils.Fold.MOUNTAIN
        ),
        crease_utils.Crease(
            p(0.5, 0.25), p(1.0, 0.25), fold=crease_utils.Fold.MOUNTAIN
        ),
        crease_utils.Crease(
            p(0.5, 0.25), p(0.5, 1.0), fold=crease_utils.Fold.MOUNTAIN
        ),
        crease_utils.Crease(
            p(0.0, 0.5), p(0.5, 0.5), fold=crease_utils.Fold.MOUNTAIN
        ),
        crease_utils.Crease(
            p(0.5, 0.5), p(1.0, 0.5), fold=crease_utils.Fold.VALLEY
        ),
    ]
    cp = cp_utils.CreasePattern(creases=creases, light_mode=True, grid_size=4)
    bottom_half = shaping_utils.Rectangle(
        name='bottom', x_start=0.25, y_start=0.5, x_len=0.5, y_len=0.5
    )
    shaped = fan_out_shaping.shape_cp_with_fan_out(
        cp,
        shaping_params=[
            fan_out_shaping.FanOutShapingParams(
                valid_fold_types=[crease_utils.Fold.MOUNTAIN],
                valid_orientations=[fan_out_shaping.Orientation.VERTICAL],
                rectangles=[bottom_half],
                move_delta=1.0,
            )
        ],
        grid_size=4,
    )
    self.assertTrue(shaped.local_flat_foldability())


if __name__ == '__main__':
  googletest.main()
