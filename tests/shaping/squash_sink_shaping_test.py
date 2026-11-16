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

"""Tests for squash sink shaping."""

from bp_pipeline.crease_pattern import crease as crease_utils
from bp_pipeline.crease_pattern import crease_pattern as cp_utils
from bp_pipeline.crease_pattern import point as point_utils
from shaping import shaping_test_fixtures as shaping_examples
from bp_pipeline.shaping import shaping_utils
from bp_pipeline.shaping import squash_sink_shaping
from bp_pipeline.crease_pattern.parser import svg as parser
from absl.testing import absltest as googletest
from absl.testing import parameterized


class SquashSinkShapingTest(parameterized.TestCase):

  def test_get_bisector_point_easy_case(self):
    p1 = point_utils.Point(x=0, y=0.5)
    p2 = point_utils.Point(x=0.5, y=0.25)
    p3 = point_utils.Point(x=0.5, y=0.75)
    bisector_point = squash_sink_shaping.get_bisector_point(p1, p2, p3)
    self.assertTrue(bisector_point.isclose(point_utils.Point(x=0.5, y=0.5)))

  def test_get_bisector_point_hard_case(self):
    p1 = point_utils.Point(x=0.5, y=0.5)
    p2 = point_utils.Point(x=0.2, y=0.3)
    p3 = point_utils.Point(x=0.5, y=0.2)
    bisector_point = squash_sink_shaping.get_bisector_point(p1, p2, p3)
    self.assertTrue(
        bisector_point.isclose(
            point_utils.Point(x=0.3637509630206024, y=0.2454163456597992)
        )
    )

  def test_get_side_creases(self):
    point = point_utils.Point(x=0.5, y=0.5)
    side_crease_1 = crease_utils.Crease(
        point_utils.Point(x=0.5, y=0.5),
        point_utils.Point(x=0.5, y=0),
        fold=crease_utils.Fold.VALLEY,
    )
    side_crease_2 = crease_utils.Crease(
        point_utils.Point(x=0.5, y=0.5),
        point_utils.Point(x=0, y=0.5),
        fold=crease_utils.Fold.VALLEY,
    )
    ridge_crease_1 = crease_utils.Crease(
        point_utils.Point(x=0.5, y=0.5),
        point_utils.Point(x=0, y=0),
        fold=crease_utils.Fold.MOUNTAIN,
    )
    ridge_crease_2 = crease_utils.Crease(
        point_utils.Point(x=0.5, y=0.5),
        point_utils.Point(x=1, y=1),
        fold=crease_utils.Fold.VALLEY,
    )
    other_crease = crease_utils.Crease(
        point_utils.Point(x=0, y=0.75),
        point_utils.Point(x=0.25, y=1),
        fold=crease_utils.Fold.MOUNTAIN,
    )
    cp = cp_utils.CreasePattern(
        creases=[
            side_crease_1,
            side_crease_2,
            ridge_crease_1,
            ridge_crease_2,
            other_crease,
        ],
        grid_size=4,
    )
    side_creases = squash_sink_shaping.get_side_creases(
        point, cp, ridge_crease_2
    )
    self.assertLen(side_creases, 2)
    self.assertIn(side_crease_1, side_creases)
    self.assertIn(side_crease_2, side_creases)

  def test_get_adjacent_side_creases_at_flap_tip(self):
    # At a flap tip all ridges meet: keep the side creases angularly next to
    # the ridge direction (towards `toward`), one on each side.
    tip = point_utils.Point(x=0.5, y=0.5)

    def crease_to(x, y):
      return crease_utils.Crease(
          tip, point_utils.Point(x=x, y=y), fold=crease_utils.Fold.VALLEY
      )

    right, down, left, up = (
        crease_to(1.0, 0.5),
        crease_to(0.5, 1.0),
        crease_to(0.0, 0.5),
        crease_to(0.5, 0.0),
    )
    other_ridge = crease_to(1.0, 0.0)
    adjacent = squash_sink_shaping.get_adjacent_side_creases(
        tip,
        [left, up, right, other_ridge, down],
        toward=point_utils.Point(x=0.75, y=0.75),
    )
    self.assertCountEqual(adjacent, [right, down])

  def test_get_adjacent_side_creases_keeps_two_or_fewer(self):
    tip = point_utils.Point(x=0.5, y=0.5)
    creases = [
        crease_utils.Crease(
            tip, point_utils.Point(x=1.0, y=0.5), fold=crease_utils.Fold.VALLEY
        )
    ]
    self.assertEqual(
        squash_sink_shaping.get_adjacent_side_creases(
            tip, creases, toward=point_utils.Point(x=0.75, y=0.75)
        ),
        creases,
    )

  def test_get_oriented_ordered_crease_segments_on_ridge(self):
    ridge_crease_1 = crease_utils.Crease(
        point_utils.Point(x=0.0, y=0.0),
        point_utils.Point(x=0.25, y=0.25),
        fold=crease_utils.Fold.MOUNTAIN,
    )
    ridge_crease_2 = crease_utils.Crease(
        point_utils.Point(x=0.5, y=0.5),
        point_utils.Point(x=0.25, y=0.25),
        fold=crease_utils.Fold.VALLEY,
    )
    ridge_crease_3 = crease_utils.Crease(
        point_utils.Point(x=0.75, y=0.75),
        point_utils.Point(x=0.5, y=0.5),
        fold=crease_utils.Fold.MOUNTAIN,
    )
    ridge_crease_4 = crease_utils.Crease(
        point_utils.Point(x=0.75, y=0.75),
        point_utils.Point(x=1.0, y=1.0),
        fold=crease_utils.Fold.VALLEY,
    )
    cp = cp_utils.CreasePattern(
        creases=[
            ridge_crease_1,
            ridge_crease_4,
            ridge_crease_3,
            ridge_crease_2,
        ],
        grid_size=4,
    )
    ridge = crease_utils.Crease(
        point_utils.Point(x=0, y=0),
        point_utils.Point(x=1, y=1),
        fold=crease_utils.Fold.MOUNTAIN,
    )

    ridge_start_point = point_utils.Point(x=1, y=1)
    ridge_segments = (
        squash_sink_shaping.get_oriented_ordered_crease_segments_on_ridge(
            cp, ridge, ridge_start_point
        )
    )

    ridge_crease_1_oriented = crease_utils.Crease(
        point_utils.Point(x=1.0, y=1.0),
        point_utils.Point(x=0.75, y=0.75),
        fold=crease_utils.Fold.VALLEY,
    )
    ridge_crease_2_oriented = crease_utils.Crease(
        point_utils.Point(x=0.75, y=0.75),
        point_utils.Point(x=0.5, y=0.5),
        fold=crease_utils.Fold.MOUNTAIN,
    )
    ridge_crease_3_oriented = crease_utils.Crease(
        point_utils.Point(x=0.5, y=0.5),
        point_utils.Point(x=0.25, y=0.25),
        fold=crease_utils.Fold.VALLEY,
    )
    ridge_crease_4_oriented = crease_utils.Crease(
        point_utils.Point(x=0.25, y=0.25),
        point_utils.Point(x=0.0, y=0.0),
        fold=crease_utils.Fold.MOUNTAIN,
    )
    self.assertLen(ridge_segments, 4)
    self.assertEqual(ridge_segments[0].p1_init, ridge_crease_1_oriented.p1_init)
    self.assertEqual(ridge_segments[1].p1_init, ridge_crease_2_oriented.p1_init)
    self.assertEqual(ridge_segments[2].p1_init, ridge_crease_3_oriented.p1_init)
    self.assertEqual(ridge_segments[3].p1_init, ridge_crease_4_oriented.p1_init)

  def test_get_full_ridge(self):
    ridge = crease_utils.Crease(
        point_utils.Point(x=0.5, y=0.5),
        point_utils.Point(x=0.75, y=0.75),
        fold=crease_utils.Fold.MOUNTAIN,
    )
    full_ridge_1 = crease_utils.Crease(
        point_utils.Point(x=0, y=0),
        point_utils.Point(x=1, y=1),
        fold=crease_utils.Fold.MOUNTAIN,
    )
    full_ridge_2 = crease_utils.Crease(
        point_utils.Point(x=0.75, y=0),
        point_utils.Point(x=1, y=0.25),
        fold=crease_utils.Fold.MOUNTAIN,
    )
    full_ridge = squash_sink_shaping.get_full_ridge(
        ridge, [full_ridge_1, full_ridge_2]
    )
    self.assertEqual(full_ridge, full_ridge_1)

  @parameterized.parameters([
      shaping_examples.SNAKE_SQUASH_SINK_SHAPING_TEST,
  ])
  def test_get_start_side_of_ridge(
      self, example: shaping_examples.BoxPleatingShapingExample
  ):
    rectangles = example.rectangles
    full_ridge = crease_utils.Crease(
        point_utils.Point(x=0.0, y=0),
        point_utils.Point(x=1, y=1),
        fold=crease_utils.Fold.MOUNTAIN,
    )
    # the point closer to top left is closer to the rectangle corner.
    start_point = point_utils.Point(x=0.125, y=0.125)
    end_point = point_utils.Point(x=0.75, y=0.75)
    ridge = crease_utils.Crease(
        start_point,
        end_point,
        fold=crease_utils.Fold.MOUNTAIN,
    )
    start_side_of_ridge = squash_sink_shaping.get_start_side_of_ridge(
        ridge, full_ridge, rectangles
    )
    self.assertTrue(start_side_of_ridge.isclose(start_point))

  @parameterized.parameters([
      shaping_examples.SNAKE_SQUASH_SINK_SHAPING_TEST,
  ])
  def test_get_all_ridges_from_square_rectangles(
      self, example: shaping_examples.BoxPleatingShapingExample
  ):
    rectangles = example.rectangles
    ridges = squash_sink_shaping.get_all_ridges_from_rectangles(rectangles)
    self.assertLen(ridges, 4)
    ridge_1 = crease_utils.Crease(
        point_utils.Point(x=0.0, y=0.0),
        point_utils.Point(x=1, y=1),
        fold=crease_utils.Fold.MOUNTAIN,
    )
    ridge_2 = crease_utils.Crease(
        point_utils.Point(x=1.0, y=1.0),
        point_utils.Point(x=2, y=2, require_on_paper=False),
        fold=crease_utils.Fold.MOUNTAIN,
    )
    ridge_3 = crease_utils.Crease(
        point_utils.Point(x=1.0, y=1.0),
        point_utils.Point(x=0, y=2, require_on_paper=False),
        fold=crease_utils.Fold.MOUNTAIN,
    )
    ridge_4 = crease_utils.Crease(
        point_utils.Point(x=1.0, y=1.0),
        point_utils.Point(x=2, y=0, require_on_paper=False),
        fold=crease_utils.Fold.MOUNTAIN,
    )
    self.assertIn(ridge_1, ridges)
    self.assertIn(ridge_2, ridges)
    self.assertIn(ridge_3, ridges)
    self.assertIn(ridge_4, ridges)

  def test_get_all_ridges_from_long_rectangles(self):
    long_rectangle = shaping_utils.Rectangle(
        name="long_rectangle",
        x_start=0,
        y_start=0,
        x_len=0.5,
        y_len=1,
        circle_centers=[
            point_utils.Point(x=0.25, y=0.25),
            point_utils.Point(x=0.25, y=0.75),
        ],
    )
    ridges = squash_sink_shaping.get_all_ridges_from_rectangles(
        [long_rectangle]
    )
    self.assertLen(ridges, 4)
    ridge_1 = crease_utils.Crease(
        point_utils.Point(x=0.0, y=0.0),
        point_utils.Point(x=0.25, y=0.25),
        fold=crease_utils.Fold.MOUNTAIN,
    )
    ridge_2 = crease_utils.Crease(
        point_utils.Point(x=0.25, y=0.25),
        point_utils.Point(x=0.5, y=0, require_on_paper=False),
        fold=crease_utils.Fold.MOUNTAIN,
    )
    ridge_3 = crease_utils.Crease(
        point_utils.Point(x=0.25, y=0.75),
        point_utils.Point(x=0, y=1, require_on_paper=False),
        fold=crease_utils.Fold.MOUNTAIN,
    )
    ridge_4 = crease_utils.Crease(
        point_utils.Point(x=0.25, y=0.75),
        point_utils.Point(x=0.5, y=1, require_on_paper=False),
        fold=crease_utils.Fold.MOUNTAIN,
    )
    self.assertIn(ridge_1, ridges)
    self.assertIn(ridge_2, ridges)
    self.assertIn(ridge_3, ridges)
    self.assertIn(ridge_4, ridges)

  def test_apply_squash_sink_to_segment(self):
    # Rectangles are only required for ridge detection.
    # the creasing itself doesn't need them.
    point_a = point_utils.Point(x=0.5, y=0)
    point_b = point_utils.Point(x=0.5, y=0.25)
    point_c = point_utils.Point(x=0.5, y=0.5)
    b_side_crease_point_1 = point_utils.Point(x=0.25, y=0)
    b_side_crease_point_2 = point_utils.Point(x=0.75, y=0)
    c_side_crease_point_1 = point_utils.Point(x=0, y=0)
    c_side_crease_point_2 = point_utils.Point(x=1, y=0)
    b_side_crease_1 = crease_utils.Crease(
        b_side_crease_point_1,
        point_b,
        fold=crease_utils.Fold.MOUNTAIN,
    )
    b_side_crease_2 = crease_utils.Crease(
        point_b,
        b_side_crease_point_2,
        fold=crease_utils.Fold.MOUNTAIN,
    )
    c_side_crease_1 = crease_utils.Crease(
        point_c,
        c_side_crease_point_2,
        fold=crease_utils.Fold.VALLEY,
    )
    c_side_crease_2 = crease_utils.Crease(
        c_side_crease_point_1,
        point_c,
        fold=crease_utils.Fold.VALLEY,
    )
    crease_ab = crease_utils.Crease(
        point_a, point_b, fold=crease_utils.Fold.VALLEY
    )
    crease_bc = crease_utils.Crease(
        point_b, point_c, fold=crease_utils.Fold.MOUNTAIN
    )
    cp = cp_utils.CreasePattern(
        creases=[
            crease_ab,
            crease_bc,
            b_side_crease_1,
            b_side_crease_2,
            c_side_crease_1,
            c_side_crease_2,
        ],
        grid_size=4,
    )
    ridge = crease_utils.Crease(
        point_a, point_c, fold=crease_utils.Fold.MOUNTAIN
    )
    new_creases, creases_to_remove = (
        squash_sink_shaping.apply_squash_sink_to_segment(
            crease_ab, crease_bc, cp, [ridge]
        )
    )

    # --- Calculated/Remaining Unique Points ---
    point_d = point_utils.Point(x=0.5, y=0.07322330470336309)
    point_e = point_utils.Point(x=0.3232233047033631, y=0.07322330470336309)
    point_f = point_utils.Point(x=0.676776695296637, y=0.07322330470336304)

    # --- New creases ---
    crease_0 = crease_utils.Crease(
        point_e,
        point_c,
        fold=crease_utils.Fold.MOUNTAIN
    )
    crease_1 = crease_utils.Crease(
        point_f,
        point_c,
        fold=crease_utils.Fold.MOUNTAIN
    )
    crease_2 = crease_utils.Crease(
        point_e,
        point_d,
        fold=crease_utils.Fold.MOUNTAIN
    )
    crease_3 = crease_utils.Crease(
        point_d,
        point_f,
        fold=crease_utils.Fold.MOUNTAIN
    )
    crease_4 = crease_utils.Crease(
        point_a,
        point_d,
        fold=crease_utils.Fold.MOUNTAIN
    )
    crease_5 = crease_utils.Crease(
        point_e,
        point_a,
        fold=crease_utils.Fold.VALLEY
    )
    crease_6 = crease_utils.Crease(
        point_a,
        point_f,
        fold=crease_utils.Fold.VALLEY
    )
    crease_7 = crease_utils.Crease(
        point_d,
        point_c,
        fold=crease_utils.Fold.VALLEY
    )
    crease_8 = crease_utils.Crease(
        b_side_crease_point_1,
        point_e,
        fold=crease_utils.Fold.MOUNTAIN
    )
    crease_9 = crease_utils.Crease(
        point_f,
        b_side_crease_point_2,
        fold=crease_utils.Fold.MOUNTAIN
    )

    # --- Removed Creases ---
    crease_10 = crease_utils.Crease(
        point_a,
        point_b,
        fold=crease_utils.Fold.VALLEY
    )
    crease_11 = crease_utils.Crease(
        point_b,
        point_c,
        fold=crease_utils.Fold.MOUNTAIN
    )
    crease_12 = crease_utils.Crease(
        b_side_crease_point_1,
        point_b,
        fold=crease_utils.Fold.MOUNTAIN
    )
    crease_13 = crease_utils.Crease(
        point_b,
        b_side_crease_point_2,
        fold=crease_utils.Fold.MOUNTAIN
    )

    new_creases_expected = [
        crease_0,
        crease_1,
        crease_2,
        crease_3,
        crease_4,
        crease_5,
        crease_6,
        crease_7,
        crease_8,
        crease_9,
    ]
    creases_to_remove_expected = [
        crease_10,
        crease_11,
        crease_12,
        crease_13,
    ]
    self.assertLen(new_creases, len(new_creases_expected))
    self.assertLen(creases_to_remove, len(creases_to_remove_expected))
    for new_crease in new_creases:
      self.assertIn(new_crease, new_creases_expected)
    for crease_to_remove in creases_to_remove:
      self.assertIn(crease_to_remove, creases_to_remove_expected)

  @parameterized.parameters([
      shaping_examples.SNAKE_SQUASH_SINK_SHAPING_TEST,
  ])
  def test_squash_sink_shaping(
      self, example: shaping_examples.BoxPleatingShapingExample
  ):
    # Load base crease pattern and parameters from the example
    snake_cp = parser.parse_svg(
        example.base_example.base_svg, grid_size=example.base_example.grid_size
    )

    # a ridge that runs from top left to bottom right.
    ridge = crease_utils.Crease(
        point_utils.Point(x=0, y=0, require_on_paper=False),
        point_utils.Point(x=0.5, y=0.5, require_on_paper=False),
        fold=crease_utils.Fold.MOUNTAIN_ZEBRA,
    )

    # Call the shaping function
    shaped_cp = squash_sink_shaping.shape_cp_with_squash_sink(
        cp=snake_cp,
        rectangles=example.rectangles,
        ridge=ridge,
    )

    # convert shaped_cp to svg
    self.assertTrue(shaped_cp.local_flat_foldability())
    self.assertTrue(example.check_squash_sink_shaping(shaped_cp))

  def test_ridge_segments_found_despite_svg_rounding_jitter(self):
    # Exact ridge from (27/44, 41/44) to (29/44, 43/44), while segment endpoint
    # was rounded to 8 decimal places by SVG serialization.
    p = lambda x, y: point_utils.Point(x=x, y=y)
    ridge = crease_utils.Crease(
        p(27 / 44, 41 / 44),
        p(29 / 44, 43 / 44),
        fold=crease_utils.Fold.MOUNTAIN,
    )
    seg_1 = crease_utils.Crease(
        p(round(27 / 44, 8), round(41 / 44, 8)),
        p(round(28 / 44, 8), round(42 / 44, 8)),
        fold=crease_utils.Fold.MOUNTAIN,
    )
    seg_2 = crease_utils.Crease(
        p(round(28 / 44, 8), round(42 / 44, 8)),
        p(round(29 / 44, 8), round(43 / 44, 8)),
        fold=crease_utils.Fold.VALLEY,
    )
    cp = cp_utils.CreasePattern(creases=[seg_1, seg_2], light_mode=True)
    segments = (
        squash_sink_shaping.get_oriented_ordered_crease_segments_on_ridge(
            cp, ridge, p(29 / 44, 43 / 44)
        )
    )
    self.assertLen(segments, 2)

  def test_border_triangle_squash_is_flat_foldable(self):
    # Border-clipped 1-segment ridge from B=(0.5, 1.0) on bottom border to
    # internal C=(0.25, 0.75) with side creases to (1.0, 0.75) and (0.25, 1.0)
    # and ridge continuation to (0.0, 0.5).
    p = lambda x, y: point_utils.Point(x=x, y=y)
    borders = [
        crease_utils.Crease(p(0, 0), p(1, 0), fold=crease_utils.Fold.BORDER),
        crease_utils.Crease(p(1, 0), p(1, 1), fold=crease_utils.Fold.BORDER),
        crease_utils.Crease(p(1, 1), p(0, 1), fold=crease_utils.Fold.BORDER),
        crease_utils.Crease(p(0, 1), p(0, 0), fold=crease_utils.Fold.BORDER),
    ]
    creases = borders + [
        crease_utils.Crease(
            p(0.5, 1.0), p(0.25, 0.75), fold=crease_utils.Fold.VALLEY
        ),
        crease_utils.Crease(
            p(0.25, 0.75), p(1.0, 0.75), fold=crease_utils.Fold.MOUNTAIN
        ),
        crease_utils.Crease(
            p(0.25, 0.75), p(0.25, 1.0), fold=crease_utils.Fold.MOUNTAIN
        ),
        crease_utils.Crease(
            p(0.25, 0.75), p(0.0, 0.5), fold=crease_utils.Fold.MOUNTAIN
        ),
    ]
    cp = cp_utils.CreasePattern(creases=creases, light_mode=True)
    rect = shaping_utils.Rectangle(
        name="corner_flap",
        x_start=-0.25,
        y_start=0.25,
        x_len=1.0,
        y_len=1.0,
        circle_centers=(p(0.25, 0.75),),
    )
    ridge = crease_utils.Crease(
        p(0.25, 0.75),
        point_utils.Point(x=0.75, y=1.25, require_on_paper=False),
        fold=crease_utils.Fold.VALLEY,
    )
    shaped = squash_sink_shaping.shape_cp_with_squash_sink(cp, [rect], ridge)
    self.assertGreater(len(shaped.creases), len(cp.creases))
    self.assertTrue(shaped.local_flat_foldability())


if __name__ == "__main__":
  googletest.main()
