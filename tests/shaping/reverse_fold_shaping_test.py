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

"""Testing functions for reverse fold shaping in box pleating."""

from bp_pipeline.crease_pattern import crease as crease_utils
from bp_pipeline.crease_pattern import crease_pattern as cp_utils
from bp_pipeline.crease_pattern import point as point_utils
from shaping import shaping_test_fixtures as shaping_examples
from bp_pipeline.shaping import reverse_fold_shaping
from bp_pipeline.shaping import shaping_elements
from bp_pipeline.crease_pattern.parser import svg as parser
from absl.testing import absltest as googletest
from absl.testing import parameterized


class ReverseFoldShapingTest(parameterized.TestCase):

  @parameterized.named_parameters(
      # dict(
      #     testcase_name='basic_reverse_fold_vertical_top',
      #     test_example=shaping_examples.BASIC_REVERSE_FOLD_VERTICAL_TOP,
      # ),
      dict(
          testcase_name='basic_reverse_fold_vertical_bottom',
          test_example=shaping_examples.BASIC_REVERSE_FOLD_VERTICAL_BOTTOM,
      ),
      dict(
          testcase_name='basic_reverse_fold_horizontal_left',
          test_example=shaping_examples.BASIC_REVERSE_FOLD_HORIZONTAL_LEFT,
      ),
      dict(
          testcase_name='basic_reverse_fold_horizontal_right',
          test_example=shaping_examples.BASIC_REVERSE_FOLD_HORIZONTAL_RIGHT,
      ),
  )
  def test_reverse_fold_shaping(self, test_example):
    shaped_cp = reverse_fold_shaping.shape_cp_with_reverse_fold(
        cp=parser.parse_svg(
            test_example.base_example.base_svg,
        ),
        grid_size=test_example.base_example.grid_size,
        rectangles=test_example.rectangles,
        element_shaping_params=test_example.reverse_fold_shaping_params,
    )

    expected_cp = parser.parse_svg(
        test_example.reverse_fold_shaping_svg,
    )

    self.assertCountEqual(
        parser.to_svg(
            shaped_cp,
            ndigits=3,
            pretty=True,
            replace_border=True,
            grid=test_example.base_example.grid_size,
        )
        .strip()
        .split('\n'),
        parser.to_svg(
            expected_cp,
            ndigits=3,
            pretty=True,
            replace_border=True,
            grid=test_example.base_example.grid_size,
        )
        .strip()
        .split('\n'),
    )

  def test_river_reverse_fold(self):
    grid_size = 16
    river_name = 'river_1'
    river = shaping_elements.make_dummy_river(
        name=river_name, grid_size=grid_size
    )

    # Create a dummy CP
    cp = cp_utils.CreasePattern(
        creases=[
            crease_utils.Crease(
                point_utils.Point(x=0, y=0),
                point_utils.Point(x=1, y=0),
                fold=crease_utils.Fold.BORDER,
            ),
            crease_utils.Crease(
                point_utils.Point(x=1, y=0),
                point_utils.Point(x=1, y=1),
                fold=crease_utils.Fold.BORDER,
            ),
            crease_utils.Crease(
                point_utils.Point(x=1, y=1),
                point_utils.Point(x=0, y=1),
                fold=crease_utils.Fold.BORDER,
            ),
            crease_utils.Crease(
                point_utils.Point(x=0, y=1),
                point_utils.Point(x=0, y=0),
                fold=crease_utils.Fold.BORDER,
            ),
        ]
    )

    # Define shaping params for the river
    shaping_params = reverse_fold_shaping.ReverseFoldShapingParams(
        flap_name=river_name,
        reverse_fold_params=[
            reverse_fold_shaping.ReverseFoldParams(
                flap_angle=0.5,  # Arbitrary angle
                fold_order=[
                    crease_utils.Fold.VALLEY,
                    crease_utils.Fold.MOUNTAIN,
                ],
                even_phase=True,
                bank_index='FIRST',
            )
        ],
        orientation='vertical',
    )

    # Call get_reverse_fold_creases
    _, zigzag_instructions = reverse_fold_shaping.get_reverse_fold_creases(
        cp=cp,
        grid_size=grid_size,
        rectangles=[],  # No flaps
        rivers=[river],
        element_shaping_params=[shaping_params],
    )

    # Verification
    self.assertNotEmpty(zigzag_instructions)

    instruction = zigzag_instructions[0]
    # Should be False for rivers as per implementation
    self.assertFalse(instruction.is_beak)
    self.assertIsInstance(instruction.crease, crease_utils.Crease)


if __name__ == '__main__':
  googletest.main()
