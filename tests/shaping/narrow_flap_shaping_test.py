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

"""Tests for narrow flap shaping."""

from bp_pipeline.crease_pattern import crease as crease_utils
from bp_pipeline.crease_pattern import point as point_utils
from shaping import shaping_test_fixtures as shaping_examples
from bp_pipeline.shaping import narrow_flap_shaping
from bp_pipeline.crease_pattern.parser import svg as parser
from absl.testing import absltest as googletest
from absl.testing import parameterized


class NarrowFlapShapingTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='basic_narrow_45_zigzag_top',
          example=shaping_examples.BASIC_NARROW_45_ZIGZAG_TOP,
      ),
      dict(
          testcase_name='basic_narrow_45_zigzag_bottom',
          example=shaping_examples.BASIC_NARROW_45_ZIGZAG_BOTTOM,
      ),
      dict(
          testcase_name='basic_narrow_22p5_zigzag_top',
          example=shaping_examples.BASIC_NARROW_22P5_ZIGZAG_TOP,
          ndigits=5,
      ),
      dict(
          testcase_name='basic_narrow_26p56_zigzag_top',
          example=shaping_examples.BASIC_NARROW_26P56_ZIGZAG_TOP,
          ndigits=3,
      ),
      dict(
          testcase_name='basic_narrow_56p31_zigzag_top',
          example=shaping_examples.BASIC_NARROW_56P31_ZIGZAG_TOP,
      ),
      dict(
          testcase_name='basic_narrow_56p31_zigzag_bottom',
          example=shaping_examples.BASIC_NARROW_56P31_ZIGZAG_BOTTOM,
      ),
      dict(
          testcase_name='crimp_leg',
          example=shaping_examples.CRIMP_NARROWING,
          ndigits=3,
      ),
      dict(
          testcase_name='narrow_edge_case1',
          example=shaping_examples.NARROW_EDGE_CASE1,
          check_flat_foldability=False,
          ndigits=3,
      ),
      dict(
          testcase_name='narrow_edge_case2',
          example=shaping_examples.NARROW_EDGE_CASE2,
          check_flat_foldability=False,
          ndigits=3,
      ),
  )
  def test_narrow_flap_shaping(
      self,
      example: shaping_examples.BoxPleatingShapingExample,
      check_flat_foldability: bool = True,
      ndigits: int = 4,
  ):
    cp = parser.parse_svg(example.base_example.base_svg)

    shaped_cp = narrow_flap_shaping.narrow(
        cp,
        example.base_example.zigzag_creases,
        example.rectangles,
        grid_size=example.base_example.grid_size,
        raise_error_on_ff_violation=check_flat_foldability,
    )

    self.assertTrue(
        example.check_narrow_flap_shaping(shaped_cp, ndigits=ndigits)
    )

  def test_orientation_canonical_order(self):
    canonical_order = [
        narrow_flap_shaping.Orientation.TOP,
        narrow_flap_shaping.Orientation.RIGHT,
        narrow_flap_shaping.Orientation.BOTTOM,
        narrow_flap_shaping.Orientation.LEFT,
    ]
    self.assertEqual(
        list(narrow_flap_shaping.Orientation),
        canonical_order,
    )

  def test_draw_narrowing_crease_does_not_rotate_at_paper_border(self):
    example = shaping_examples.BASIC_NARROW_45_ZIGZAG_TOP
    cp = parser.parse_svg(example.base_example.base_svg)
    assert example.rectangles is not None
    rect = example.rectangles[0]
    start = point_utils.Point(x=0.05, y=0.05)
    # 45-degree ray exiting the paper border at (0.0, 0.0) when
    # allow_paper_border_rotation=False or prev_clockwise_angle_delta is set.
    end = point_utils.Point(x=0.0, y=0.0)
    res = narrow_flap_shaping.draw_narrowing_crease(
        start_point=start,
        end_point=end,
        rectangle_border_points={
            point_utils.Point(x=0.0, y=0.0),
            point_utils.Point(x=0.5, y=0.0),
        },
        rectangle=rect,
        cp=cp,
        non_border_zigzag_creases=[
            crease_utils.Crease(
                point_utils.Point(x=0.05, y=0.05),
                point_utils.Point(x=0.5, y=0.5),
                fold=crease_utils.Fold.MOUNTAIN,
            )
        ],
        prev_clockwise_angle_delta=10.0,
        allow_paper_border_rotation=False,
    )
    self.assertIsNotNone(res)
    assert res is not None
    intersection_pt, _, zigzag_replacements, angle_delta = res
    self.assertTrue(intersection_pt.is_border)
    self.assertEmpty(zigzag_replacements)
    self.assertIsNone(angle_delta)


if __name__ == '__main__':
  googletest.main()
