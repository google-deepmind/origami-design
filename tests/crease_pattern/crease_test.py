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

"""Tests for Crease object."""

import dataclasses
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from bp_pipeline.crease_pattern import crease
from bp_pipeline.crease_pattern import point


@dataclasses.dataclass(frozen=True)
class CreaseTestExample:
  """Contains a single variant of a problem."""

  first_point: point.Point
  second_point: point.Point
  fold: crease.Fold
  fold_percent: float = 1.0


_VALID_CREASE_EXAMPLES = [
    CreaseTestExample(
        first_point=point.Point(x=0, y=0),
        second_point=point.Point(x=1, y=1),
        fold=crease.Fold.MOUNTAIN,
    ),
    CreaseTestExample(
        first_point=point.Point(x=1, y=1),
        second_point=point.Point(x=0, y=0),
        fold=crease.Fold.BORDER,
        fold_percent=0.5,
    ),
    CreaseTestExample(
        first_point=point.Point.from_viewbox(
            x=1.00001, y=1, viewbox=(0, 0, 1, 1)
        ),
        second_point=point.Point.from_viewbox(
            x=0, y=-0.00001, viewbox=(0, 0, 1, 1)
        ),
        fold=crease.Fold.BORDER,
    ),
]

_INVALID_CREASE_EXAMPLES = [
    # negative fold percent
    CreaseTestExample(
        first_point=point.Point(x=0, y=0),
        second_point=point.Point(x=1, y=1),
        fold=crease.Fold.MOUNTAIN,
        fold_percent=-0.1,
    ),
    CreaseTestExample(
        first_point=point.Point(x=0, y=0),
        second_point=point.Point(x=1, y=1),
        fold=crease.Fold.MOUNTAIN,
        fold_percent=1.1,
    ),
    CreaseTestExample(
        first_point=point.Point(x=0.1, y=0.1),
        second_point=point.Point(x=0.1, y=0.1),
        fold=crease.Fold.MOUNTAIN,
        fold_percent=-0.1,
    ),
]


class CreaseTest(parameterized.TestCase):

  @parameterized.parameters(_VALID_CREASE_EXAMPLES)
  def test_valid_crease(self, example: CreaseTestExample):
    c = crease.Crease(
        example.first_point,
        example.second_point,
        fold=example.fold,
        fold_percent=example.fold_percent,
    )
    self.assertIsInstance(c, crease.Crease)

  @parameterized.parameters(_INVALID_CREASE_EXAMPLES)
  def test_invalid_creases(self, example: CreaseTestExample):
    with self.assertRaises(ValueError):
      crease.Crease(
          example.first_point,
          example.second_point,
          fold=example.fold,
          fold_percent=example.fold_percent,
      )

  @parameterized.parameters(_VALID_CREASE_EXAMPLES)
  def test_same_crease_isclose(self, example: CreaseTestExample):
    crease1 = crease.Crease(
        example.first_point,
        example.second_point,
        fold=example.fold,
        fold_percent=example.fold_percent,
    )
    crease2 = crease.Crease(
        example.first_point,
        example.second_point,
        fold=example.fold,
        fold_percent=example.fold_percent,
    )
    self.assertTrue(crease1.isclose(crease2))

  def test_is_parallel(self):
    crease1 = crease.Crease(
        point.Point(x=0.5, y=0),
        point.Point(x=1, y=1),
        fold=crease.Fold.MOUNTAIN,
        fold_percent=0.1,
    )
    crease2 = crease.Crease(
        point.Point(x=0.0, y=0),
        point.Point(x=0.5, y=1),
        fold=crease.Fold.MOUNTAIN,
        fold_percent=0.1,
    )
    crease3 = crease.Crease(
        point.Point(x=0, y=0),
        point.Point(x=0, y=1),
        fold=crease.Fold.MOUNTAIN,
        fold_percent=0.1,
    )
    crease4 = crease.Crease(
        point.Point(x=1, y=0.4),
        point.Point(x=1, y=1),
        fold=crease.Fold.MOUNTAIN,
        fold_percent=0.1,
    )
    crease5 = crease.Crease(
        point.Point(x=1, y=0.4),
        point.Point(x=1, y=1),
        fold=crease.Fold.MOUNTAIN,
        fold_percent=0.1,
    )
    crease6 = crease.Crease(
        point.Point(x=0.08333761034273847, y=0.08349331076593083),
        point.Point(x=0.08753948319996023, y=0.08523401614911437),
        fold=crease.Fold.MOUNTAIN,
        fold_percent=0.1,
    )
    crease7 = crease.Crease(
        point.Point(x=0.08333761034273847, y=0.08349331076593083),
        point.Point(x=0.08735689349121037, y=0.08428316386096116),
        fold=crease.Fold.VALLEY,
        fold_percent=0.1,
    )
    with self.subTest(name='Parallel_not_colinear'):
      self.assertTrue(crease1.is_parallel(crease2))
    with self.subTest(name='Parallel_not_colinear_vertical'):
      self.assertTrue(crease3.is_parallel(crease4))
    with self.subTest(name='Same_crease'):
      self.assertTrue(crease5.is_parallel(crease4))
    with self.subTest(name='Not_parallel'):
      self.assertFalse(crease1.is_parallel(crease3))
      self.assertFalse(crease1.is_parallel(crease4))
    with self.subTest(name='Numerical_close_but_not_parallel'):
      self.assertFalse(crease6.is_parallel(crease7))

  def test_is_perpendicular(self):
    crease1 = crease.Crease(
        point.Point(x=0.0, y=0),
        point.Point(x=0.0, y=1.0),
        fold=crease.Fold.MOUNTAIN,
        fold_percent=0.1,
    )
    crease2 = crease.Crease(
        point.Point(x=0.0, y=0),
        point.Point(x=1.0, y=0.0),
        fold=crease.Fold.MOUNTAIN,
        fold_percent=0.1,
    )
    crease3 = crease.Crease(
        point.Point(x=0, y=0),
        point.Point(x=1, y=1),
        fold=crease.Fold.MOUNTAIN,
        fold_percent=0.1,
    )
    crease4 = crease.Crease(
        point.Point(x=0, y=1),
        point.Point(x=1, y=0),
        fold=crease.Fold.MOUNTAIN,
        fold_percent=0.1,
    )
    with self.subTest(name='Perpendicular'):

      self.assertTrue(crease1.is_perpendicular(crease2))
      self.assertTrue(crease3.is_perpendicular(crease4))

    with self.subTest(name='Not_perpendicular'):

      self.assertFalse(crease1.is_perpendicular(crease3))
      self.assertFalse(crease1.is_perpendicular(crease4))
      self.assertFalse(crease2.is_perpendicular(crease3))
      self.assertFalse(crease2.is_perpendicular(crease4))

  def test_length(self):
    crease1 = crease.Crease(
        point.Point(x=0.0, y=0),
        point.Point(x=0.0, y=1.0),
        fold=crease.Fold.MOUNTAIN,
        fold_percent=0.1,
    )
    crease2 = crease.Crease(
        point.Point(x=0, y=0),
        point.Point(x=1, y=1),
        fold=crease.Fold.MOUNTAIN,
        fold_percent=0.1,
    )

    self.assertAlmostEqual(crease1.length, 1, places=5)
    self.assertAlmostEqual(crease2.length, 1.4142135623730951, places=5)

  def test_is_overlapping(self):
    crease1 = crease.Crease(
        point.Point(x=0.0, y=0),
        point.Point(x=0.5, y=0.5),
        fold=crease.Fold.MOUNTAIN,
        fold_percent=0.1,
    )
    crease2 = crease.Crease(
        point.Point(x=0.3, y=0.3),
        point.Point(x=0.7, y=0.7),
        fold=crease.Fold.MOUNTAIN,
        fold_percent=0.1,
    )
    crease3 = crease.Crease(
        point.Point(x=0.5, y=0.5),
        point.Point(x=0.7, y=0.7),
        fold=crease.Fold.MOUNTAIN,
        fold_percent=0.1,
    )
    crease4 = crease.Crease(
        point.Point(x=0.7, y=0.7),
        point.Point(x=1, y=1),
        fold=crease.Fold.MOUNTAIN,
        fold_percent=0.1,
    )
    crease9 = crease.Crease(
        point.Point(x=0.1, y=0.1),
        point.Point(x=0.9, y=0.9),
        fold=crease.Fold.MOUNTAIN,
        fold_percent=0.1,
    )
    crease10 = crease.Crease(
        point.Point(x=0.1, y=0.1),
        point.Point(x=0.3, y=0.3),
        fold=crease.Fold.MOUNTAIN,
        fold_percent=0.1,
    )
    crease5 = crease.Crease(
        point.Point(x=0.0, y=0),
        point.Point(x=0.0, y=0.5),
        fold=crease.Fold.MOUNTAIN,
        fold_percent=0.1,
    )
    crease6 = crease.Crease(
        point.Point(x=0.0, y=0.3),
        point.Point(x=0.0, y=0.7),
        fold=crease.Fold.MOUNTAIN,
        fold_percent=0.1,
    )
    crease7 = crease.Crease(
        point.Point(x=0.0, y=0.5),
        point.Point(x=0.0, y=0.7),
        fold=crease.Fold.MOUNTAIN,
        fold_percent=0.1,
    )
    crease8 = crease.Crease(
        point.Point(x=0.0, y=0.7),
        point.Point(x=0.0, y=1),
        fold=crease.Fold.MOUNTAIN,
        fold_percent=0.1,
    )
    crease11 = crease.Crease(
        point.Point(x=0.0, y=0.0),
        point.Point(x=0.0, y=1.0),
        fold=crease.Fold.MOUNTAIN,
        fold_percent=0.1,
    )
    crease12 = crease.Crease(
        point.Point(x=0.0, y=0.0),
        point.Point(x=1.0, y=0.0),
        fold=crease.Fold.MOUNTAIN,
        fold_percent=0.1,
    )
    crease13 = crease.Crease(
        point.Point(x=0.0, y=0.5),
        point.Point(x=0.5, y=0.5),
        fold=crease.Fold.MOUNTAIN,
        fold_percent=0.1,
    )

    with self.subTest(name='Overlapping_no_shared_tip'):
      self.assertTrue(crease1.is_overlapping(crease2))
      self.assertTrue(crease2.is_overlapping(crease1))
      self.assertTrue(crease2.is_overlapping(crease9))
      self.assertTrue(crease9.is_overlapping(crease2))
    with self.subTest(name='Overlapping_shared_tip'):
      self.assertTrue(crease9.is_overlapping(crease10))
      self.assertTrue(crease10.is_overlapping(crease9))
    with self.subTest(name='Not_overlapping_shared_tip'):
      self.assertFalse(crease1.is_overlapping(crease3))
    with self.subTest(name='Not_overlapping_no_shared_tip'):
      self.assertFalse(crease1.is_overlapping(crease4))
    with self.subTest(name='Vertical_overlapping_no_shared_tip'):
      self.assertTrue(crease5.is_overlapping(crease6))
      self.assertTrue(crease6.is_overlapping(crease5))
    with self.subTest(name='Vertical_overlapping_no_shared_tip'):
      self.assertFalse(crease5.is_overlapping(crease7))
    with self.subTest(name='Vertical_not_overlapping'):
      self.assertFalse(crease5.is_overlapping(crease8))
    with self.subTest(name='Not_overlapping_shared_tip'):
      self.assertFalse(crease12.is_overlapping(crease11))
      self.assertFalse(crease13.is_overlapping(crease11))

  @parameterized.parameters(_VALID_CREASE_EXAMPLES)
  def test_crease_dunder_equal(self, example: CreaseTestExample):
    c1 = crease.Crease(
        example.first_point,
        example.second_point,
        fold=example.fold,
        fold_percent=example.fold_percent,
    )

    # Equivalent creases
    with self.subTest(name='Same_crease'):
      self.assertEqual(c1, c1)

    identical_crease = dataclasses.replace(c1)
    with self.subTest(name='Identical_crease'):
      self.assertEqual(c1, identical_crease)

    epsilon_different_first_point = dataclasses.replace(
        c1, p1_init=c1.p1_init - point.Point(x=1e-4, y=1e-4)
    )
    with self.subTest(name='Epsilon_different_first_point'):
      self.assertEqual(c1, epsilon_different_first_point)

    epsilon_different_second_point = dataclasses.replace(
        c1, p2_init=c1.p2_init - point.Point(x=-1e-4, y=1e-4)
    )
    with self.subTest(name='Epsilon_different_second_point'):
      self.assertEqual(c1, epsilon_different_second_point)

    with self.subTest(name='Mock'):
      self.assertEqual(c1, mock.ANY)

    # Different Creases
    other_fold_type = (
        crease.Fold.VALLEY
        if example.fold == crease.Fold.MOUNTAIN
        else crease.Fold.MOUNTAIN
    )
    different_fold_crease = dataclasses.replace(c1, fold=other_fold_type)
    with self.subTest(name='Different_fold'):
      self.assertNotEqual(c1, different_fold_crease)

    different_fold_percent = dataclasses.replace(
        c1, fold_percent=c1.fold_percent - 0.1
    )
    with self.subTest(name='Different_fold_percent'):
      self.assertNotEqual(c1, different_fold_percent)

    with self.subTest(name='different first_point'):
      self.assertNotEqual(
          c1,
          dataclasses.replace(
              c1,
              p1_init=point.Point(x=(c1.p1_init.x + 0.1) / 2, y=c1.p1_init.y),
          ),
      )

    with self.subTest(name='different second_point'):
      self.assertNotEqual(
          c1,
          dataclasses.replace(
              c1,
              p2_init=point.Point(x=c1.p1_init.x, y=(c1.p2_init.y + 0.1) / 2),
          ),
      )

    with self.subTest(name='Different_type'):
      self.assertNotEqual(c1, 'string')

  def test_is_collinear(self):
    """Test that collinearity is symmetric in numerical sensitive case."""
    c1 = crease.Crease(
        point.Point(x=0.101, y=0.343),
        point.Point(x=0.172, y=0.172),
        fold=crease.Fold.MOUNTAIN,
        fold_percent=0.1,
    )
    c2 = crease.Crease(
        point.Point(x=0.136, y=0.257),
        point.Point(x=0.142, y=0.243),
        fold=crease.Fold.MOUNTAIN,
        fold_percent=0.1,
    )
    c3 = crease.Crease(
        point.Point(x=0.992, y=1.0),
        point.Point(x=1.0, y=1.0),
        fold=crease.Fold.BORDER,
    )
    c4 = crease.Crease(
        point.Point(x=1.0, y=0.0),
        point.Point(x=1.0, y=0.585),
        fold=crease.Fold.BORDER,
        fold_percent=1.0,
    )

    self.assertFalse(c1.is_collinear(c2))
    self.assertFalse(c2.is_collinear(c1))
    self.assertFalse(point.is_collinear(c1.p1, c1.p2, c2.p1))
    self.assertFalse(point.is_collinear(c1.p1, c1.p2, c2.p2))
    self.assertFalse(point.is_collinear(c2.p1, c2.p2, c1.p1))
    self.assertFalse(point.is_collinear(c2.p1, c2.p2, c1.p2))

    self.assertFalse(c3.is_collinear(c4))
    self.assertFalse(c4.is_collinear(c3))
    self.assertFalse(point.is_collinear(c3.p1, c3.p2, c4.p1))
    self.assertFalse(point.is_collinear(c3.p1, c3.p2, c4.p2))
    self.assertFalse(point.is_collinear(c4.p1, c4.p2, c3.p1))
    self.assertTrue(point.is_collinear(c4.p1, c4.p2, c3.p2))

  def test_split_to_creases(self):
    c1 = crease.Crease(
        point.Point(x=0.0, y=0.0),
        point.Point(x=1.0, y=1.0),
        fold=crease.Fold.MOUNTAIN,
        fold_percent=0.1,
    )
    c2 = crease.Crease(
        point.Point(x=0.0, y=1.0),
        point.Point(x=1.0, y=0.0),
        fold=crease.Fold.VALLEY,
        fold_percent=0.1,
    )
    c1_1 = crease.Crease(
        point.Point(x=0.0, y=0.0),
        point.Point(x=0.5, y=0.5),
        fold=crease.Fold.MOUNTAIN,
        fold_percent=0.1,
    )
    c1_2 = crease.Crease(
        point.Point(x=0.5, y=0.5),
        point.Point(x=1.0, y=1.0),
        fold=crease.Fold.MOUNTAIN,
        fold_percent=0.1,
    )
    c2_1 = crease.Crease(
        point.Point(x=0.0, y=1.0),
        point.Point(x=0.5, y=0.5),
        fold=crease.Fold.MOUNTAIN,
        fold_percent=0.1,
    )
    c2_2 = crease.Crease(
        point.Point(x=0.5, y=0.5),
        point.Point(x=1.0, y=0.0),
        fold=crease.Fold.MOUNTAIN,
        fold_percent=0.1,
    )
    c3 = crease.Crease(
        point.Point(x=0.0, y=0.0),
        point.Point(x=1.0, y=0.0),
        fold=crease.Fold.MOUNTAIN,
        fold_percent=0.1,
    )
    with self.subTest(name='segments_are_correct'):
      c1_segments = c1.split_to_segments([c2])
      self.assertLen(c1_segments, 2)
      self.assertTrue(c1_segments[0].isclose(c1_1))
      self.assertTrue(c1_segments[1].isclose(c1_2))
      c2_segments = c2.split_to_segments([c1])
      self.assertLen(c2_segments, 2)
      self.assertTrue(c2_segments[0].isclose(c2_1))
      self.assertTrue(c2_segments[1].isclose(c2_2))
    with self.subTest(name='try_to_keep_original_order'):
      c3_segments = c3.split_to_segments([c1])
      self.assertLen(c3_segments, 1)
      self.assertTrue(c3_segments[0].p1_init.isclose(c3.p1_init))
      self.assertTrue(c3_segments[0].p2_init.isclose(c3.p2_init))

  def test_border_split_to_segments(self):
    c1 = crease.Crease(
        point.Point(x=0.0, y=0.0),
        point.Point(x=1.0, y=0.0),
        fold=crease.Fold.BORDER,
        fold_percent=0.1,
    )
    c2 = crease.Crease(
        point.Point(x=0.5, y=0.5),
        point.Point(x=0.5, y=0.0),
        fold=crease.Fold.MOUNTAIN,
        fold_percent=0.1,
    )
    border_segments = c1.split_to_segments([c2])
    self.assertLen(border_segments, 2)

    s1, s2 = border_segments
    self.assertTrue(s1.p1_init.isclose(c1.p1_init))
    self.assertTrue(s2.p2_init.isclose(c1.p2_init))

    self.assertTrue(point.is_collinear(s1.p1_init, s1.p2_init, s2.p1_init))
    self.assertTrue(point.is_collinear(s1.p1_init, s1.p2_init, s2.p2_init))

  def test_split_to_segments_near_endpoint_numerical_stability(self):
    """Tests numerical stability of split_to_segments when near endpoints."""
    # Crease to be split, from (0,0) to (0.5, 0.5).
    crease_to_split = crease.Crease(
        point.Point(x=0.0, y=0.0),
        point.Point(x=0.5, y=0.5),
        fold=crease.Fold.MOUNTAIN,
    )

    intersecting_crease = crease.Crease(
        point.Point(x=0.1, y=0.902),
        point.Point(x=0.9, y=0.102),
        fold=crease.Fold.VALLEY,
    )

    segments = crease_to_split.split_to_segments([intersecting_crease])

    # The intersection is outside, so the crease should not be split.
    self.assertLen(segments, 1)
    self.assertEqual(segments[0], crease_to_split)

  @parameterized.named_parameters(
      dict(
          testcase_name='no_intersection',
          c1=crease.Crease(
              point.Point(x=0.0, y=0.0),
              point.Point(x=1.0, y=1.0),
              fold=crease.Fold.MOUNTAIN,
              fold_percent=0.1,
          ),
          c2=crease.Crease(
              point.Point(x=0.0, y=0.1),
              point.Point(x=0.0, y=0.8),
              fold=crease.Fold.MOUNTAIN,
              fold_percent=0.1,
          ),
          expected_intersection=None,
      ),
      dict(
          testcase_name='parallel_no_intersection',
          c1=crease.Crease(
              point.Point(x=0.0, y=0.0),
              point.Point(x=1.0, y=1.0),
              fold=crease.Fold.MOUNTAIN,
              fold_percent=0.1,
          ),
          c2=crease.Crease(
              point.Point(x=0.2, y=0.1),
              point.Point(x=0.3, y=0.2),
              fold=crease.Fold.MOUNTAIN,
              fold_percent=0.1,
          ),
          expected_intersection=None,
      ),
      dict(
          testcase_name='intersection_on_tip',
          c1=crease.Crease(
              point.Point(x=0.0, y=0.0),
              point.Point(x=1.0, y=1.0),
              fold=crease.Fold.MOUNTAIN,
              fold_percent=0.1,
          ),
          c2=crease.Crease(
              point.Point(x=0.0, y=0.0),
              point.Point(x=1.0, y=0.0),
              fold=crease.Fold.VALLEY,
              fold_percent=0.1,
          ),
          expected_intersection=point.Point(x=0.0, y=0.0),
      ),
      dict(
          testcase_name='intersection_not_on_tip',
          c1=crease.Crease(
              point.Point(x=0.0, y=0.0),
              point.Point(x=1.0, y=1.0),
              fold=crease.Fold.MOUNTAIN,
              fold_percent=0.1,
          ),
          c2=crease.Crease(
              point.Point(x=0.0, y=1.0),
              point.Point(x=1.0, y=0.0),
              fold=crease.Fold.VALLEY,
              fold_percent=0.1,
          ),
          expected_intersection=point.Point(x=0.5, y=0.5),
      ),
      dict(
          testcase_name='intersection_outside_or_range',
          c1=crease.Crease(
              point.Point(x=0.0, y=0.0),
              point.Point(x=0.25, y=1.0),
              fold=crease.Fold.VALLEY,
              fold_percent=0.1,
          ),
          c2=crease.Crease(
              point.Point(x=0.75, y=1.0),
              point.Point(x=1.0, y=0.0),
              fold=crease.Fold.VALLEY,
              fold_percent=0.1,
          ),
          expected_intersection=None,
      ),
      dict(
          testcase_name='intersection_not_on_creases',
          c1=crease.Crease(
              point.Point(x=0.0, y=0.0),
              point.Point(x=1.0, y=1.0),
              fold=crease.Fold.MOUNTAIN,
              fold_percent=0.1,
          ),
          c2=crease.Crease(
              point.Point(x=0.0, y=0.1),
              point.Point(x=0.0, y=0.8),
              fold=crease.Fold.MOUNTAIN,
              fold_percent=0.1,
          ),
          expected_intersection=(0, 0),
      ),
      dict(
          testcase_name='intersection_outside_unit_square',
          c1=crease.Crease(
              point.Point(x=0.0, y=0.0),
              point.Point(x=0.25, y=1.0),
              fold=crease.Fold.VALLEY,
              fold_percent=0.1,
          ),
          c2=crease.Crease(
              point.Point(x=0.75, y=1.0),
              point.Point(x=1.0, y=0.0),
              fold=crease.Fold.VALLEY,
              fold_percent=0.1,
          ),
          expected_intersection=(0.5, 2.0),
      ),
  )
  def test_vertex_with_crease_1(self, c1, c2, expected_intersection):
    if expected_intersection is None:
      self.assertIsNone(c1.vertex_with_crease(c2))
    elif isinstance(expected_intersection, tuple):
      self.assertEqual(
          c1.vertex_with_crease_external(c2), expected_intersection
      )
    else:
      self.assertEqual(c1.vertex_with_crease(c2), expected_intersection)

  @parameterized.named_parameters(
      dict(
          testcase_name='segments_inside_extended_intersection_outside',
          c1=crease.Crease(
              point.Point(x=0.0, y=0.0),
              point.Point(x=0.1, y=0.0),
              fold=crease.Fold.MOUNTAIN,
          ),
          c2=crease.Crease(
              point.Point(x=0.0, y=0.1),
              point.Point(x=0.1, y=0.2),
              fold=crease.Fold.MOUNTAIN,
          ),
          expected_intersections=[
              point.Point(x=-0.1, y=0.0, require_on_paper=False),
              None,
              None,
              None,
          ],
      ),
      dict(
          testcase_name='segments_inside_outside_intersection_outside',
          c1=crease.Crease(
              point.Point(x=-0.2, y=0.0, require_on_paper=False),
              point.Point(x=0.1, y=0.0),
              fold=crease.Fold.MOUNTAIN,
          ),
          c2=crease.Crease(
              point.Point(x=-0.2, y=-0.1, require_on_paper=False),
              point.Point(x=0.1, y=0.2),
              fold=crease.Fold.MOUNTAIN,
          ),
          expected_intersections=[
              point.Point(x=-0.1, y=0.0, require_on_paper=False),
              None,
              point.Point(x=-0.1, y=0.0, require_on_paper=False),
              None,
          ],
      ),
      dict(
          testcase_name='segments_inside_outside_intersection_inside',
          c1=crease.Crease(
              point.Point(x=-1, y=-1, require_on_paper=False),
              point.Point(x=1, y=1, require_on_paper=False),
              fold=crease.Fold.MOUNTAIN,
          ),
          c2=crease.Crease(
              point.Point(x=-1, y=1, require_on_paper=False),
              point.Point(x=1, y=-1, require_on_paper=False),
              fold=crease.Fold.MOUNTAIN,
          ),
          expected_intersections=[
              point.Point(x=0.0, y=0.0, require_on_paper=False),
              point.Point(x=0, y=0),
              point.Point(x=0.0, y=0.0, require_on_paper=False),
              point.Point(x=0, y=0),
          ],
      ),
      dict(
          testcase_name='segments_outside_intersection_outside',
          c1=crease.Crease(
              point.Point(x=1, y=1, require_on_paper=False),
              point.Point(x=2, y=1, require_on_paper=False),
              fold=crease.Fold.MOUNTAIN,
          ),
          c2=crease.Crease(
              point.Point(x=1.5, y=0.5, require_on_paper=False),
              point.Point(x=1.5, y=1.5, require_on_paper=False),
              fold=crease.Fold.MOUNTAIN,
          ),
          expected_intersections=[
              point.Point(x=1.5, y=1, require_on_paper=False),
              None,
              point.Point(x=1.5, y=1, require_on_paper=False),
              None,
          ],
      ),
      dict(
          testcase_name='intersecting_tips',
          c1=crease.Crease(
              point.Point(x=0, y=0),
              point.Point(x=0.5, y=0.5),
              fold=crease.Fold.MOUNTAIN,
          ),
          c2=crease.Crease(
              point.Point(x=0.5, y=0.5),
              point.Point(x=1.0, y=1.0),
              fold=crease.Fold.MOUNTAIN,
          ),
          expected_intersections=[
              point.Point(x=0.5, y=0.5, require_on_paper=False),
              point.Point(x=0.5, y=0.5, require_on_paper=True),
              point.Point(x=0.5, y=0.5, require_on_paper=False),
              point.Point(x=0.5, y=0.5, require_on_paper=True),
          ],
      ),
  )
  def test_vertex_with_crease_2(self, c1, c2, expected_intersections):
    params = [
        {'extend_lines': True, 'include_external': True},
        {'extend_lines': True, 'include_external': False},
        {'extend_lines': False, 'include_external': True},
        {'extend_lines': False, 'include_external': False},
    ]
    for expected_intersection, param in zip(expected_intersections, params):
      if expected_intersection is None:
        self.assertIsNone(c1.vertex_with_crease(c2, **param))
      else:
        self.assertEqual(
            c1.vertex_with_crease(c2, **param), expected_intersection
        )

  def test_is_valid(self):
    c1 = crease.Crease(
        point.Point(x=0.75, y=1.0),
        point.Point(x=1.0, y=0.0),
        fold=crease.Fold.BORDER,
        fold_percent=0.1,
    )
    c2 = crease.Crease(
        point.Point(x=0.75, y=1.0),
        point.Point(x=1.0, y=0.0),
        fold=crease.Fold.VALLEY,
        fold_percent=0.1,
    )
    c3 = crease.Crease(
        point.Point(x=0.75, y=0.9),
        point.Point(x=0.9, y=0.9),
        fold=crease.Fold.VALLEY,
        fold_percent=0.1,
    )
    c4 = crease.Crease(
        point.Point(x=0.75, y=0.9),
        point.Point(x=0.9, y=0.9),
        fold=crease.Fold.BORDER,
        fold_percent=0.1,
    )
    c5 = crease.Crease(
        point.Point(x=0.0, y=1.0),
        point.Point(x=1.0, y=0.0),
        fold=crease.Fold.BORDER,
        fold_percent=0.1,
    )
    c6 = crease.Crease(
        point.Point(x=0.0, y=1.0),
        point.Point(x=0.5, y=1),
        fold=crease.Fold.BORDER,
        fold_percent=0.1,
    )
    c7 = crease.Crease(
        point.Point(x=0.0, y=0.0),
        point.Point(x=0.0, y=0.5),
        fold=crease.Fold.MOUNTAIN,
    )
    with self.subTest(name='not border_crease'):
      self.assertTrue(c2.is_valid)
      self.assertTrue(c3.is_valid)
    with self.subTest(name='border_crease_on_border'):
      self.assertTrue(c6.is_valid)
    with self.subTest(name='border_crease_not_on_border'):
      self.assertFalse(c4.is_valid)
      self.assertFalse(c5.is_valid)
      self.assertFalse(c1.is_valid)
    with self.subTest('crease on border is invalid'):
      self.assertFalse(c7.is_valid)

  def test_merge_with_segments(self):
    c1 = crease.Crease(
        point.Point(x=0.3, y=0.3),
        point.Point(x=0.4, y=0.4),
        fold=crease.Fold.MOUNTAIN,
        fold_percent=0.1,
    )
    c2 = crease.Crease(
        point.Point(x=0.4, y=0.4),
        point.Point(x=0.5, y=0.5),
        fold=crease.Fold.MOUNTAIN,
        fold_percent=0.1,
    )
    c3 = crease.Crease(
        point.Point(x=0.3, y=0.3),
        point.Point(x=0.2, y=0.2),
        fold=crease.Fold.MOUNTAIN,
        fold_percent=0.1,
    )
    c4 = crease.Crease(
        point.Point(x=0.8, y=0.8),
        point.Point(x=0.9, y=0.9),
        fold=crease.Fold.MOUNTAIN,
        fold_percent=0.1,
    )
    c5 = crease.Crease(
        point.Point(x=0.9, y=0.9),
        point.Point(x=1.0, y=1.0),
        fold=crease.Fold.MOUNTAIN,
        fold_percent=0.1,
    )
    c6 = crease.Crease(
        point.Point(x=0.2, y=0.2),
        point.Point(x=0.1, y=0.1),
        fold=crease.Fold.VALLEY,
        fold_percent=0.1,
    )
    c7 = crease.Crease(
        point.Point(x=0.6, y=0.6),
        point.Point(x=0.7, y=0.7),
        fold=crease.Fold.VALLEY,
        fold_percent=0.2,
    )
    merged = crease.Crease(
        point.Point(x=0.2, y=0.2),
        point.Point(x=0.5, y=0.5),
        fold=crease.Fold.MOUNTAIN,
        fold_percent=0.1,
    )
    c1_vertical = crease.Crease(
        point.Point(x=0.0, y=0.3),
        point.Point(x=0.0, y=0.4),
        fold=crease.Fold.MOUNTAIN,
        fold_percent=0.1,
    )
    c2_vertical = crease.Crease(
        point.Point(x=0.0, y=0.4),
        point.Point(x=0.0, y=0.5),
        fold=crease.Fold.MOUNTAIN,
        fold_percent=0.1,
    )
    merged_vertical = crease.Crease(
        point.Point(x=0.0, y=0.3),
        point.Point(x=0.0, y=0.5),
        fold=crease.Fold.MOUNTAIN,
        fold_percent=0.1,
    )
    c1_trivial = crease.Crease(
        point.Point(x=0.0, y=0.4),
        point.Point(x=0.1, y=0.5),
        fold=crease.Fold.MOUNTAIN,
        fold_percent=0.1,
    )

    with self.subTest(name='merge_with_segments'):
      merged_ = c1.merge_with_segments([c1, c2, c3, c4, c5, c6, c7])

      self.assertTrue(merged_.isclose(merged))
      self.assertEqual(merged_.fold, merged.fold)
      self.assertEqual(merged_.fold_percent, merged.fold_percent)

    with self.subTest(name='merge_vertical_segments'):
      merged_vertical_ = c1_vertical.merge_with_segments(
          [c1_vertical, c2_vertical]
      )

      self.assertTrue(merged_vertical_.isclose(merged_vertical))
      self.assertEqual(merged_vertical_.fold, merged_vertical.fold)
      self.assertEqual(
          merged_vertical_.fold_percent, merged_vertical.fold_percent
      )

    with self.subTest(name='merge_trivial_segments'):
      merged_trivial_ = c1_vertical.merge_with_segments(
          [c1_vertical, c2_vertical], only_trivial=True
      )
      merged_trivial_false = c1_vertical.merge_with_segments(
          [c1_vertical, c2_vertical, c1_trivial], only_trivial=True
      )

      self.assertTrue(merged_trivial_.isclose(merged_vertical))
      self.assertEqual(merged_trivial_.fold, merged_vertical.fold)
      self.assertEqual(
          merged_trivial_.fold_percent, merged_vertical.fold_percent
      )
      self.assertFalse(merged_trivial_false.isclose(merged_vertical))

  @parameterized.named_parameters(
      dict(
          testcase_name='is_within_other',
          other_crease=crease.Crease(
              point.Point(x=0, y=0),
              point.Point(x=0.3, y=0.3),
              fold=crease.Fold.MOUNTAIN,
          ),
          match_fold=True,
          expected=True,
      ),
      dict(
          testcase_name='is_within_other_match_fold_false',
          other_crease=crease.Crease(
              point.Point(x=0, y=0),
              point.Point(x=0.3, y=0.3),
              fold=crease.Fold.VALLEY,
          ),
          match_fold=False,
          expected=True,
      ),
      dict(
          testcase_name='is_not_within_other',
          other_crease=crease.Crease(
              point.Point(x=0.4, y=0.4),
              point.Point(x=0.5, y=0.5),
              fold=crease.Fold.MOUNTAIN,
          ),
          match_fold=True,
          expected=False,
      ),
      dict(
          testcase_name='is_not_within_other_match_fold_true',
          other_crease=crease.Crease(
              point.Point(x=0.1, y=0.1),
              point.Point(x=0.2, y=0.2),
              fold=crease.Fold.VALLEY,
          ),
          match_fold=True,
          expected=False,
      ),
  )
  def test_is_within(self, other_crease, match_fold, expected):
    """Tests for is_within method."""
    sub_crease = crease.Crease(
        point.Point(x=0.1, y=0.1),
        point.Point(x=0.2, y=0.2),
        fold=crease.Fold.MOUNTAIN,
    )
    result = sub_crease.is_within(other_crease, match_fold=match_fold)
    self.assertEqual(result, expected)

  def test_is_within_self(self):
    """Tests for is_within method with self."""
    c = crease.Crease(
        point.Point(x=0.1, y=0.1),
        point.Point(x=0.2, y=0.2),
        fold=crease.Fold.MOUNTAIN,
    )
    result_match = c.is_within(c, match_fold=True)
    self.assertTrue(result_match)

    result = c.is_within(c, match_fold=False)
    self.assertTrue(result)

  def test_hash(self):
    c1 = crease.Crease(
        point.Point(x=0.1, y=0.1),
        point.Point(x=0.2, y=0.2),
        fold=crease.Fold.MOUNTAIN,
    )
    c2 = crease.Crease(
        point.Point(x=0.1, y=0.1),
        point.Point(x=0.2, y=0.2),
        fold=crease.Fold.MOUNTAIN,
    )
    c3 = crease.Crease(
        point.Point(x=0.1, y=0.2),
        point.Point(x=0.2, y=0.2),
        fold=crease.Fold.BORDER,
    )

    self.assertEqual(c1.hash_value, c2.hash_value)
    self.assertNotEqual(c1.hash_value, c3.hash_value)

  def test_hash_creases(self):
    c1 = crease.Crease(
        point.Point(x=0.1, y=0.1),
        point.Point(x=0.2, y=0.2),
        fold=crease.Fold.MOUNTAIN,
    )
    c2 = crease.Crease(
        point.Point(x=0.1, y=0.1),
        point.Point(x=0.2, y=0.3),
        fold=crease.Fold.MOUNTAIN,
    )
    c1_ = crease.Crease(
        point.Point(x=0.1, y=0.1),
        point.Point(x=0.2, y=0.2),
        fold=crease.Fold.MOUNTAIN,
    )
    c3 = crease.Crease(
        point.Point(x=0.1, y=0.1),
        point.Point(x=0.3, y=0.3),
        fold=crease.Fold.MOUNTAIN,
    )
    self.assertEqual(
        hash(crease.Creases.create_from_creases(creases=[c1, c2])),
        hash(crease.Creases.create_from_creases(creases=[c2, c1])),
    )
    self.assertEqual(
        hash(crease.Creases.create_from_creases(creases=[c1_, c2, c1])),
        hash(crease.Creases.create_from_creases(creases=[c2, c1])),
    )
    self.assertEqual(
        hash(crease.Creases.create_from_creases(creases=[c1, c2, c3])),
        hash(crease.Creases.create_from_creases(creases=[c1, c3, c2])),
    )
    self.assertNotEqual(
        hash(crease.Creases.create_from_creases(creases=[c1, c2])),
        hash(crease.Creases.create_from_creases(creases=[c1, c3])),
    )

  def test_intersects_border(self):
    c1 = crease.Crease(
        point.Point(x=0.0, y=0.1),
        point.Point(x=0.2, y=0.2),
        fold=crease.Fold.MOUNTAIN,
    )
    c2 = crease.Crease(
        point.Point(x=0.1, y=0.1),
        point.Point(x=0.2, y=0.2),
        fold=crease.Fold.MOUNTAIN,
    )
    self.assertTrue(c1.intersects_border)
    self.assertFalse(c2.intersects_border)

  def test_is_tip(self):
    c1 = crease.Crease(
        point.Point(x=0.1, y=0.1),
        point.Point(x=0.2, y=0.2),
        fold=crease.Fold.MOUNTAIN,
    )
    self.assertTrue(c1.is_tip(point.Point(x=0.1, y=0.1)))
    self.assertTrue(c1.is_tip(point.Point(x=0.2, y=0.2)))
    self.assertFalse(c1.is_tip(point.Point(x=0.3, y=0.3)))

  def test_intersect_at_tip(self):
    c1 = crease.Crease(
        point.Point(x=0.1, y=0.1),
        point.Point(x=0.2, y=0.2),
        fold=crease.Fold.MOUNTAIN,
    )
    c2 = crease.Crease(
        point.Point(x=0.1, y=0.1),
        point.Point(x=0.3, y=0.3),
        fold=crease.Fold.MOUNTAIN,
    )
    c3 = crease.Crease(
        point.Point(x=0.3, y=0.3),
        point.Point(x=0.4, y=0.4),
        fold=crease.Fold.MOUNTAIN,
    )
    self.assertTrue(c1.intersect_at_tip(c2))
    self.assertFalse(c1.intersect_at_tip(c3))

  def test_mountain_to_valley(self):
    c1 = crease.Crease(
        point.Point(x=0.1, y=0.1),
        point.Point(x=0.2, y=0.2),
        fold=crease.Fold.MOUNTAIN,
    )
    c2 = crease.Crease(
        point.Point(x=0.1, y=0.1),
        point.Point(x=0.2, y=0.2),
        fold=crease.Fold.VALLEY,
    )
    self.assertEqual(c1.mountain_to_valley(), crease.Fold.VALLEY)
    self.assertEqual(c2.mountain_to_valley(), crease.Fold.MOUNTAIN)
    with self.assertRaises(ValueError):
      crease.Crease(
          point.Point(x=0.1, y=0.1),
          point.Point(x=0.2, y=0.2),
          fold=crease.Fold.BORDER,
      ).mountain_to_valley()

  def test_flip(self):
    fold = crease.Fold.MOUNTAIN
    self.assertEqual(fold.flip(), crease.Fold.VALLEY)
    fold = crease.Fold.VALLEY
    self.assertEqual(fold.flip(), crease.Fold.MOUNTAIN)
    with self.assertRaises(ValueError):
      crease.Fold.BORDER.flip()

  def test_is_overlapping_border(self):
    p1 = point.Point(x=0.5, y=0.5)
    p2 = point.Point(x=0.75, y=0.75)
    c = crease.Crease(p1, p2, fold=crease.Fold.MOUNTAIN)
    self.assertFalse(c.is_overlapping_border)

    p1 = point.Point(x=0, y=0)
    p2 = point.Point(x=0, y=1)
    c = crease.Crease(p1, p2, fold=crease.Fold.MOUNTAIN)
    self.assertTrue(c.is_overlapping_border)

    p1 = point.Point(x=1, y=0)
    p2 = point.Point(x=0, y=0)
    c = crease.Crease(p1, p2, fold=crease.Fold.MOUNTAIN)
    self.assertTrue(c.is_overlapping_border)

    p1 = point.Point(x=1, y=0.2)
    p2 = point.Point(x=1, y=0.8)
    c = crease.Crease(p1, p2, fold=crease.Fold.BORDER)
    self.assertTrue(
        c.is_overlapping_border, 'Partial right border crease failed.'
    )

    p1 = point.Point(x=0, y=0)
    p2 = point.Point(x=1, y=1)
    c = crease.Crease(p1, p2, fold=crease.Fold.MOUNTAIN)
    self.assertFalse(
        c.is_overlapping_border, 'Diagonal corner-to-corner crease failed.'
    )

    p1 = point.Point(x=0, y=0.5)
    p2 = point.Point(x=0.5, y=0)
    c = crease.Crease(p1, p2, fold=crease.Fold.MOUNTAIN)
    self.assertFalse(
        c.is_overlapping_border, 'Diagonal edge-to-edge crease failed.'
    )

    p1 = point.Point(x=0.5, y=0)
    p2 = point.Point(x=0.5, y=1)
    c = crease.Crease(p1, p2, fold=crease.Fold.MOUNTAIN)
    self.assertFalse(
        c.is_overlapping_border, 'Internal vertical crease failed.'
    )

    p1 = point.Point(x=0, y=0.5)
    p2 = point.Point(x=1, y=0.5)
    c = crease.Crease(p1, p2, fold=crease.Fold.MOUNTAIN)
    self.assertFalse(
        c.is_overlapping_border, 'Internal horizontal crease failed.'
    )

    p1 = point.Point(x=0.5, y=0.5)
    p2 = point.Point(x=1, y=0.5)
    c = crease.Crease(p1, p2, fold=crease.Fold.MOUNTAIN)
    self.assertFalse(
        c.is_overlapping_border, 'Crease with one endpoint on border failed.'
    )

  @parameterized.parameters(
      (
          crease.Crease(
              point.Point(x=0.1, y=0.1),
              point.Point(x=0.2, y=0.2),
              fold=crease.Fold.MOUNTAIN,
          ),
          True,
      ),
      (
          crease.Crease(
              point.Point(x=0.1000001, y=0.1),
              point.Point(x=0.2, y=0.2),
              fold=crease.Fold.MOUNTAIN,
          ),
          False,
      ),
      (
          crease.CreaseWithExactMatch(
              crease.Crease(
                  point.Point(x=0.1, y=0.1),
                  point.Point(x=0.2, y=0.2),
                  fold=crease.Fold.MOUNTAIN,
              )
          ),
          True,
      ),
      (
          crease.CreaseWithExactMatch(
              crease.Crease(
                  point.Point(x=0.1000001, y=0.1),
                  point.Point(x=0.2, y=0.2),
                  fold=crease.Fold.MOUNTAIN,
              )
          ),
          False,
      ),
      (
          'not a crease',
          False,
      ),
  )
  def test_creases_contains(self, item_to_check, expected_in):
    creases_in_set = [
        crease.Crease(
            point.Point(x=0.1, y=0.1),
            point.Point(x=0.2, y=0.2),
            fold=crease.Fold.MOUNTAIN,
        ),
    ]
    creases_set = crease.Creases.create_from_creases(creases_in_set)
    self.assertEqual(item_to_check in creases_set, expected_in)

  @parameterized.parameters(
      (
          crease.Crease(
              point.Point(x=0.1, y=0.1),
              point.Point(x=0.2, y=0.2),
              fold=crease.Fold.MOUNTAIN,
          ),
          crease.Crease(
              point.Point(x=0.1, y=0.1),
              point.Point(x=0.2, y=0.2),
              fold=crease.Fold.MOUNTAIN,
          ),
          crease.Crease(
              point.Point(x=0.1, y=0.1),
              point.Point(x=0.2, y=0.2),
              fold=crease.Fold.MOUNTAIN,
          ),
      ),
      (
          crease.Crease(
              point.Point(x=0.1, y=0.1),
              point.Point(x=0.2, y=0.2),
              fold=crease.Fold.MOUNTAIN,
          ),
          crease.Crease(
              point.Point(x=0.15, y=0.15),
              point.Point(x=0.25, y=0.25),
              fold=crease.Fold.MOUNTAIN,
          ),
          crease.Crease(
              point.Point(x=0.15, y=0.15),
              point.Point(x=0.2, y=0.2),
              fold=crease.Fold.MOUNTAIN,
          ),
      ),
      (
          crease.Crease(
              point.Point(x=0.1, y=0.1),
              point.Point(x=0.2, y=0.2),
              fold=crease.Fold.MOUNTAIN,
          ),
          crease.Crease(
              point.Point(x=0.2, y=0.2),
              point.Point(x=0.3, y=0.3),
              fold=crease.Fold.MOUNTAIN,
          ),
          None,
      ),
      (
          crease.Crease(
              point.Point(x=0.1, y=0.1),
              point.Point(x=0.2, y=0.2),
              fold=crease.Fold.MOUNTAIN,
          ),
          crease.Crease(
              point.Point(x=0.0, y=0.0),
              point.Point(x=0.3, y=0.3),
              fold=crease.Fold.MOUNTAIN,
          ),
          crease.Crease(
              point.Point(x=0.1, y=0.1),
              point.Point(x=0.2, y=0.2),
              fold=crease.Fold.MOUNTAIN,
          ),
      ),
      (
          crease.Crease(
              point.Point(x=0.1, y=0.1),
              point.Point(x=0.2, y=0.2),
              fold=crease.Fold.MOUNTAIN,
          ),
          crease.Crease(
              point.Point(x=0.2, y=0.2),
              point.Point(x=0.3, y=0.3),
              fold=crease.Fold.MOUNTAIN,
          ),
          None,
      ),
      (
          crease.Crease(
              point.Point(x=0.1, y=0.1),
              point.Point(x=0.2, y=0.2),
              fold=crease.Fold.MOUNTAIN,
          ),
          crease.Crease(
              point.Point(x=0.1, y=0.1),
              point.Point(x=0.3, y=0.3),
              fold=crease.Fold.MOUNTAIN,
          ),
          crease.Crease(
              point.Point(x=0.1, y=0.1),
              point.Point(x=0.2, y=0.2),
              fold=crease.Fold.MOUNTAIN,
          ),
      ),
  )
  def test_get_overlap(self, c1, c2, expected_overlap):
    self.assertEqual(c1.get_overlap(c2), expected_overlap)

  @parameterized.named_parameters(
      dict(
          testcase_name='close_to_nothing',
          p1=(0.1, 0.1),
          p2=(0.9, 0.9),
          expected_left=False,
          expected_right=False,
          expected_top=False,
          expected_bottom=False,
      ),
      dict(
          testcase_name='close_to_left',
          p1=(1e-4, 0.1),
          p2=(0.9, 0.9),
          expected_left=True,
          expected_right=False,
          expected_top=False,
          expected_bottom=False,
      ),
      dict(
          testcase_name='close_to_right',
          p1=(0.1, 0.1),
          p2=(1 - 1e-4, 0.9),
          expected_left=False,
          expected_right=True,
          expected_top=False,
          expected_bottom=False,
      ),
      dict(
          testcase_name='close_to_top',
          p1=(0.1, 1e-4),
          p2=(0.9, 0.9),
          expected_left=False,
          expected_right=False,
          expected_top=True,
          expected_bottom=False,
      ),
      dict(
          testcase_name='close_to_bottom',
          p1=(0.1, 0.1),
          p2=(0.9, 1 - 1e-4),
          expected_left=False,
          expected_right=False,
          expected_top=False,
          expected_bottom=True,
      ),
      dict(
          testcase_name='close_to_top_left',
          p1=(1e-4, 1e-4),
          p2=(0.9, 0.9),
          expected_left=True,
          expected_right=False,
          expected_top=True,
          expected_bottom=False,
      ),
      dict(
          testcase_name='vertical_close_to_left',
          p1=(1e-4, 0.1),
          p2=(1e-4, 0.9),
          expected_left=True,
          expected_right=False,
          expected_top=False,
          expected_bottom=False,
      ),
      dict(
          testcase_name='horizontal_close_to_top',
          p1=(0.1, 1e-4),
          p2=(0.9, 1e-4),
          expected_left=False,
          expected_right=False,
          expected_top=True,
          expected_bottom=False,
      ),
  )
  def test_close_to_border(
      self, p1, p2, expected_left, expected_right, expected_top, expected_bottom
  ):
    c = crease.Crease(
        point.Point(x=p1[0], y=p1[1]),
        point.Point(x=p2[0], y=p2[1]),
        fold=crease.Fold.MOUNTAIN,
    )
    self.assertEqual(c.close_to_left_border, expected_left)
    self.assertEqual(c.close_to_right_border, expected_right)
    self.assertEqual(c.close_to_top_border, expected_top)
    self.assertEqual(c.close_to_bottom_border, expected_bottom)

  @parameterized.parameters(
      (
          crease.Crease(
              point.Point(x=0.1, y=0.1),
              point.Point(x=0.1, y=0.2),
              fold=crease.Fold.MOUNTAIN,
          ),
          crease.Crease(
              point.Point(x=0.1000001, y=0.1),
              point.Point(x=0.1000001, y=0.15),
              fold=crease.Fold.MOUNTAIN,
          ),
          3,
          crease.Crease(
              point.Point(x=0.100, y=0.1),
              point.Point(x=0.100, y=0.15),
              fold=crease.Fold.MOUNTAIN,
          ),
      ),
      (
          crease.Crease(
              point.Point(x=0.1, y=0.1),
              point.Point(x=0.1, y=0.2),
              fold=crease.Fold.MOUNTAIN,
          ),
          crease.Crease(
              point.Point(x=0.1000001, y=0.1),
              point.Point(x=0.1000001, y=0.15),
              fold=crease.Fold.MOUNTAIN,
          ),
          None,
          None,
      ),
  )
  def test_rounded_overlap(self, crease_1, crease_2, ndigits, expected_overlap):
    self.assertEqual(
        crease_1.get_overlap(crease_2, ndigits=ndigits), expected_overlap
    )

  def test_split_to_segments_steep_crease_tip_snapping_order(self):
    # Steep non-vertical crease (`dx = 0.002 > DEFAULT_TOLERANCE`) whose
    # interior intersection snaps to a tip with `x < p1.x`. Sorting by `.x`
    # previously placed the snapped interior tip before `p1` and crashed.
    steep = crease.Crease(
        point.Point(x=0.500, y=0.1),
        point.Point(x=0.502, y=0.9),
        fold=crease.Fold.MOUNTAIN,
    )
    crosser = crease.Crease(
        point.Point(x=0.2, y=0.5),
        point.Point(x=0.8, y=0.5),
        fold=crease.Fold.VALLEY,
    )
    snapped_tip = point.Point(x=0.5004, y=0.5)
    segments = steep.split_to_segments([crosser], tips=[snapped_tip])
    self.assertLen(segments, 2)
    self.assertTrue(segments[0].is_tip(point.Point(x=0.500, y=0.1)))
    self.assertTrue(segments[0].is_tip(snapped_tip))
    self.assertTrue(segments[1].is_tip(snapped_tip))
    self.assertTrue(segments[1].is_tip(point.Point(x=0.502, y=0.9)))

  def test_split_to_segments_sub_tolerance_distinct_intersections(self):
    target = crease.Crease(
        point.Point(x=0.1, y=0.5),
        point.Point(x=0.9, y=0.5),
        fold=crease.Fold.MOUNTAIN,
    )
    c_a = crease.Crease(
        point.Point(x=0.5000, y=0.2),
        point.Point(x=0.5000, y=0.8),
        fold=crease.Fold.VALLEY,
    )
    c_b = crease.Crease(
        point.Point(x=0.5003, y=0.2),
        point.Point(x=0.5003, y=0.8),
        fold=crease.Fold.VALLEY,
    )
    segments = target.split_to_segments([c_a, c_b])
    self.assertLen(segments, 2)

  def test_split_to_segments_zebra_when_p1_has_larger_x(self):
    # norm(0.8, 0.0) = 0.8 < norm(0.0, 0.9) = 0.9, so `p1 = (0.8, 0.0)` and
    # `p2 = (0.0, 0.9)` (`p1.x > p2.x`). `MOUNTAIN_ZEBRA` specifies MOUNTAIN
    # on the segment with smallest x coordinate.
    zebra = crease.Crease(
        point.Point(x=0.8, y=0.0),
        point.Point(x=0.0, y=0.9),
        fold=crease.Fold.MOUNTAIN_ZEBRA,
    )
    crosser = crease.Crease(
        point.Point(x=0.0, y=0.45),
        point.Point(x=1.0, y=0.45),
        fold=crease.Fold.VALLEY,
    )
    segments = zebra.split_to_segments([crosser])
    self.assertLen(segments, 2)
    by_min_x = sorted(segments, key=lambda c: min(c.p1.x, c.p2.x))
    self.assertEqual(by_min_x[0].fold, crease.Fold.MOUNTAIN)
    self.assertEqual(by_min_x[1].fold, crease.Fold.VALLEY)


class FastMergeCreasesTest(parameterized.TestCase):

  def _make_padding_creases(self, count: int = 210) -> list[crease.Crease]:
    out = []
    for i in range(count):
      y = 0.1 + 0.003 * i
      out.append(
          crease.Crease(
              point.Point(x=0.7, y=y),
              point.Point(x=0.75, y=y + 0.0015),
              fold=crease.Fold.BORDER,
          )
      )
    return out

  def test_merge_creases_spatial_index_collinear_and_wrap(self):
    pad = self._make_padding_creases(210)
    # Near-zero and near-pi angles meeting along the same horizontal line.
    seg1 = crease.Crease(
        point.Point(x=0.1, y=0.05),
        point.Point(x=0.2, y=0.05005),
        fold=crease.Fold.MOUNTAIN,
    )
    seg2 = crease.Crease(
        point.Point(x=0.2, y=0.05005),
        point.Point(x=0.3, y=0.05),
        fold=crease.Fold.MOUNTAIN,
    )
    merged = crease.merge_creases([seg1, seg2, *pad])
    self.assertLen(merged, 1 + len(pad))
    self.assertTrue(
        any(
            c.isclose(
                crease.Crease(
                    point.Point(x=0.1, y=0.05),
                    point.Point(x=0.3, y=0.05),
                    fold=crease.Fold.MOUNTAIN,
                )
            )
            for c in merged
        )
    )

  @parameterized.parameters(
      (False, False),
      (True, False),
      (False, True),
  )
  def test_merge_creases_indexed_matches_small_path(
      self, only_trivial: bool, merge_zebra: bool
  ):
    odd_fold = crease.Fold.VALLEY if merge_zebra else crease.Fold.MOUNTAIN
    chain = [
        crease.Crease(
            point.Point(x=0.05 + 0.05 * k, y=0.2),
            point.Point(x=0.05 + 0.05 * (k + 1), y=0.2),
            fold=crease.Fold.MOUNTAIN if k % 2 == 0 else odd_fold,
        )
        for k in range(5)
    ]
    # Branching crease at x=0.15 blocks trivial merge across that vertex.
    branch = crease.Crease(
        point.Point(x=0.15, y=0.2),
        point.Point(x=0.15, y=0.3),
        fold=crease.Fold.VALLEY,
    )
    small = [*chain, branch]
    pad = self._make_padding_creases(205)
    expected_small = crease.merge_creases(
        small, only_trivial=only_trivial, merge_zebra=merge_zebra
    )
    merged_large = crease.merge_creases(
        [*small, *pad], only_trivial=only_trivial, merge_zebra=merge_zebra
    )
    kept_non_pad = merged_large[: len(merged_large) - len(pad)]
    self.assertLen(kept_non_pad, len(expected_small))
    for got, exp in zip(kept_non_pad, expected_small, strict=True):
      self.assertEqual(got, exp)

  def test_merge_creases_boundary_isclose_and_collinear_regressions(self):
    pad = self._make_padding_creases(210)

    # Short near-identical creases where `c1.is_overlapping(c2)` is True via
    # `isclose` despite swinging canonical segment angle across many bins.
    short_a = crease.Crease(
        point.Point(x=0.5, y=0.5),
        point.Point(x=0.502, y=0.5),
        fold=crease.Fold.MOUNTAIN,
    )
    short_b = crease.Crease(
        point.Point(x=0.5, y=0.5009),
        point.Point(x=0.502, y=0.5001),
        fold=crease.Fold.MOUNTAIN,
    )
    self.assertTrue(short_a.is_overlapping(short_b))
    expected_short = crease.merge_creases([short_a, short_b])
    got_short = crease.merge_creases([short_a, short_b, *pad])
    self.assertLen(got_short[: len(got_short) - len(pad)], len(expected_short))

    # Distant collinear segments whose small angular delta shifts uncentered
    # Hesse distance, plus grid-128 tilted segments, sub-grid slivers, and seam
    # angles near 0 / pi.
    far_a = crease.Crease(
        point.Point(x=0.0, y=0.5),
        point.Point(x=0.05, y=0.5),
        fold=crease.Fold.MOUNTAIN,
    )
    far_b = crease.Crease(
        point.Point(x=0.90, y=0.4995),
        point.Point(x=0.95, y=0.5005),
        fold=crease.Fold.MOUNTAIN,
    )
    far_tilted = crease.Crease(
        point.Point(x=0.90, y=0.4992),
        point.Point(x=0.907, y=0.5008),
        fold=crease.Fold.MOUNTAIN,
    )
    sliver = crease.Crease(
        point.Point(x=0.80, y=0.4994),
        point.Point(x=0.804, y=0.5006),
        fold=crease.Fold.MOUNTAIN,
    )
    seam_hi = crease.Crease(
        point.Point(x=0.1, y=0.2008),
        point.Point(x=0.2, y=0.1992),
        fold=crease.Fold.VALLEY,
    )
    seam_lo = crease.Crease(
        point.Point(x=0.2, y=0.1992),
        point.Point(x=0.3, y=0.19925),
        fold=crease.Fold.VALLEY,
    )
    adversarial = [
        far_a,
        far_b,
        far_tilted,
        sliver,
        seam_hi,
        seam_lo,
        short_a,
        short_b,
        *pad,
    ]
    self.assertTrue(far_a.is_collinear(far_b))
    self.assertTrue(far_a.is_collinear(far_tilted))
    self.assertTrue(far_a.is_collinear(sliver))

    line_index = crease._LineSpatialIndex(adversarial)  # pylint: disable=protected-access
    for query in adversarial[:8]:
      got_ids = {id(c) for c in line_index.query_collinear(query)}
      expected_ids = {id(c) for c in adversarial if query.is_collinear(c)}
      self.assertEqual(got_ids, expected_ids)

    expected_merged = crease.merge_creases([seam_hi, seam_lo])
    got_merged = crease.merge_creases([seam_hi, seam_lo, *pad])
    kept = got_merged[: len(got_merged) - len(pad)]
    self.assertLen(kept, len(expected_merged))
    for got, exp in zip(kept, expected_merged, strict=True):
      self.assertEqual(got, exp)

  def test_crease_bbox_index_query_bbox_deduplicates_and_respects_pad(self):
    multi_cell = crease.Crease(
        point.Point(x=0.1, y=0.1),
        point.Point(x=0.4, y=0.4),
        fold=crease.Fold.MOUNTAIN,
    )
    near_edge = crease.Crease(
        point.Point(x=0.5015, y=0.2),
        point.Point(x=0.5015, y=0.3),
        fold=crease.Fold.VALLEY,
    )
    far_away = crease.Crease(
        point.Point(x=0.8, y=0.8),
        point.Point(x=0.9, y=0.9),
        fold=crease.Fold.MOUNTAIN,
    )
    index = crease.CreaseBBoxIndex()
    for c in (multi_cell, near_edge, far_away):
      index.add(c)

    default_hits = index.query_bbox(0.2, 0.2, 0.5, 0.5)
    self.assertEqual([id(c) for c in default_hits], [id(multi_cell)])

    padded_hits = index.query_bbox(0.2, 0.2, 0.5, 0.5, pad=0.002)
    self.assertEqual(
        [id(c) for c in padded_hits], [id(multi_cell), id(near_edge)]
    )


if __name__ == '__main__':
  absltest.main()
