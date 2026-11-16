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

"""Testing functions for crimp fold shaping in box pleating."""

from bp_pipeline.crease_pattern import crease as crease_utils
from bp_pipeline.crease_pattern import point as point_utils
from shaping import shaping_test_fixtures as shaping_examples
from bp_pipeline.shaping import crimp_fold_shaping
from bp_pipeline.shaping import shaping_elements
from bp_pipeline.crease_pattern.parser import svg as parser
from absl.testing import absltest as googletest
from absl.testing import parameterized


class CrimpFoldShapingTest(parameterized.TestCase):

  def test_crimp_fold_shaping(self):
    test_example = shaping_examples.INSECT_LEG
    shaped_cp = crimp_fold_shaping.shape_cp_with_crimp_fold(
        cp=parser.parse_svg(
            test_example.narrowed_svg,
        ),
        grid_size=test_example.base_example.grid_size,
        flaps=[
            shaping_elements.ShapingFlap.from_legacy_rectangle_direct(
                rect, test_example.base_example.grid_size
            )
            for rect in test_example.rectangles
        ],
        element_shaping_params=test_example.crimp_fold_shaping_params,
    )

    expected_cp = parser.parse_svg(
        test_example.crimp_fold_shaping_svg,
    )

    self.assertCountEqual(
        parser.to_svg(shaped_cp, ndigits=3, pretty=True).strip().split('\n'),
        parser.to_svg(expected_cp, ndigits=3, pretty=True).strip().split('\n'),
    )

  def test_crimp_fold_shaping_with_rivers(self):
    test_example = shaping_examples.RIVERONEFLAP
    grid_size = test_example.grid_size
    # Create a dummy river.
    y1 = 0.5
    y2 = 0.5 + (2.0 / grid_size)
    p1_start = point_utils.Point(x=0.0, y=y1)
    p1_end = point_utils.Point(x=1.0, y=y1)
    p2_start = point_utils.Point(x=0.0, y=y2)
    p2_end = point_utils.Point(x=1.0, y=y2)

    river = shaping_elements.ShapingRiver(
        name='river1',
        bank_1=[
            crease_utils.Crease(p1_start, p1_end, fold=crease_utils.Fold.HINGE)
        ],
        bank_2=[
            crease_utils.Crease(p2_start, p2_end, fold=crease_utils.Fold.HINGE)
        ],
        width=2,  # 2 grid units
        grid_size=grid_size,
    )

    # Create params to shape this river.
    params = [
        crimp_fold_shaping.CrimpFoldShapingParams(
            element_name='river1',
            crimp_fold_params=[
                crimp_fold_shaping.CrimpFoldParams(
                    depth=1,
                    fold_order=[
                        crease_utils.Fold.MOUNTAIN,
                        crease_utils.Fold.VALLEY,
                    ],
                    fold_percent=1.0,
                    zigzag_flap_angle=0.0,
                    zigzag_fold_order=[],
                    zigzag_even_phase=True,
                    bank_index='FIRST',
                )
            ],
        )
    ]

    shaped_cp, _ = crimp_fold_shaping.get_crimp_fold_creases(
        cp=test_example.packing_cp,
        grid_size=grid_size,
        flaps=[],
        element_shaping_params=params,
        rivers=[river],
    )

    # Expect some creases to be generated for the river.
    self.assertNotEmpty(shaped_cp)


if __name__ == '__main__':
  googletest.main()
