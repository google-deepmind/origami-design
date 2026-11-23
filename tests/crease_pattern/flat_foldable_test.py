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

"""Tests for the local flat-foldability repair solver."""

from absl.testing import absltest
from absl.testing import parameterized
from bp_pipeline import rectangle as rectangle_lib
from bp_pipeline.crease_pattern import crease as crease_utils
from bp_pipeline.crease_pattern import crease_pattern as cp_utils
from bp_pipeline.crease_pattern import flat_foldable
from bp_pipeline.crease_pattern import point as point_utils
from bp_pipeline.crease_pattern.parser import svg
import cp_test_fixtures as shaping_examples


def _pt(x: float, y: float) -> point_utils.Point:
  return point_utils.Point(x=x, y=y)


def _unit_borders() -> list[crease_utils.Crease]:
  b = crease_utils.Fold.BORDER
  return [
      crease_utils.Crease(_pt(0, 0), _pt(1, 0), fold=b),
      crease_utils.Crease(_pt(1, 0), _pt(1, 1), fold=b),
      crease_utils.Crease(_pt(1, 1), _pt(0, 1), fold=b),
      crease_utils.Crease(_pt(0, 1), _pt(0, 0), fold=b),
  ]


class FlatFoldableTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='one_kawasaki_violation',
          example=shaping_examples.ONE_KAWASAKI_VIOLATION,
      ),
      dict(
          testcase_name='four_kawasaki_violations',
          example=shaping_examples.FOUR_KAWASAKI_VIOLATIONS,
      ),
  )
  def test_get_flat_foldable(
      self, example: shaping_examples.BoxPleatingCreasePatternExample
  ):
    base_cp = svg.parse_svg(example.base_svg, grid_size=example.grid_size)
    self.assertFalse(base_cp.local_flat_foldability())
    fixed_cp = flat_foldable.get_flat_foldable(base_cp)
    self.assertIsNotNone(fixed_cp)
    self.assertTrue(fixed_cp.local_flat_foldability())

  def test_max_candidates_limits_search(self):
    example = shaping_examples.FOUR_KAWASAKI_VIOLATIONS
    base_cp = svg.parse_svg(example.base_svg, grid_size=example.grid_size)
    self.assertFalse(base_cp.local_flat_foldability())
    self.assertIsNone(
        flat_foldable.get_flat_foldable(base_cp, max_candidates=1)
    )

  def test_max_crease_modifications_limits_search(self):
    example = shaping_examples.FOUR_KAWASAKI_VIOLATIONS
    base_cp = svg.parse_svg(example.base_svg, grid_size=example.grid_size)
    self.assertFalse(base_cp.local_flat_foldability())
    self.assertIsNone(
        flat_foldable.get_flat_foldable(base_cp, max_crease_modifications=0)
    )

  def test_preserves_light_mode_and_splits_merged_border(self):
    m = crease_utils.Fold.MOUNTAIN
    v = crease_utils.Fold.VALLEY
    fanned = crease_utils.Crease(_pt(0.5, 0.5), _pt(0.75, 1.0), fold=m)
    cp = cp_utils.CreasePattern(
        creases=[
            *_unit_borders(),
            fanned,
            crease_utils.Crease(_pt(0.5, 0.5), _pt(0.5, 0.0), fold=m),
            crease_utils.Crease(_pt(0.5, 0.5), _pt(0.0, 0.5), fold=m),
            crease_utils.Crease(_pt(0.5, 0.5), _pt(1.0, 0.5), fold=v),
        ],
        light_mode=True,
    )
    fixed = flat_foldable.get_flat_foldable(cp, unmodifiable_creases=[fanned])
    self.assertIsNotNone(fixed)
    self.assertTrue(fixed.light_mode)
    self.assertTrue(fixed.local_flat_foldability())
    # Split border replaces the 4 original borders with subdivided segments at
    # the interior-crease border endpoints.
    border_creases = [
        c for c in fixed.creases if c.fold == crease_utils.Fold.BORDER
    ]
    self.assertGreater(len(border_creases), 4)
    # The fanned crease remains untouched.
    self.assertIn(fanned, fixed.creases)

  def test_light_mode_rejects_rotation_crossing_interior_crease(self):
    m = crease_utils.Fold.MOUNTAIN
    v = crease_utils.Fold.VALLEY
    fanned = crease_utils.Crease(_pt(0.5, 0.5), _pt(0.75, 1.0), fold=m)
    blocker = crease_utils.Crease(_pt(0.1, 0.25), _pt(0.9, 0.25), fold=v)
    cp = cp_utils.CreasePattern(
        creases=[
            *_unit_borders(),
            fanned,
            crease_utils.Crease(_pt(0.5, 0.5), _pt(0.5, 0.0), fold=m),
            crease_utils.Crease(_pt(0.5, 0.5), _pt(0.0, 0.5), fold=m),
            crease_utils.Crease(_pt(0.5, 0.5), _pt(1.0, 0.5), fold=v),
            blocker,
        ],
        light_mode=True,
    )
    normalized = cp_utils.CreasePattern(
        flat_foldable.split_border_creases_at_tips(cp.creases), light_mode=True
    )
    cand = flat_foldable.CandidateCreasePattern.from_cp(
        normalized,
        unmodifiable_crease_hashes=frozenset(),
        num_crease_modifications=0,
        candidate_num=0,
    )
    center_v = next(
        vt for vt in cand.cp.vertices if vt.point.isclose(_pt(0.5, 0.5))
    )
    top_c = next(
        c
        for c in center_v.creases
        if c.p1.isclose(_pt(0.5, 0.0)) or c.p2.isclose(_pt(0.5, 0.0))
    )
    rotated_cp, modified = flat_foldable.get_new_cp_with_rotated_crease(
        cand.cp, top_c, center_v, cand.crease_hash_to_vertex_mapping
    )
    self.assertIsNone(rotated_cp)
    self.assertEmpty(modified)

  def test_flips_border_crease_to_resolve_maekawa_violation(self):
    m = crease_utils.Fold.MOUNTAIN
    v = crease_utils.Fold.VALLEY
    cp = cp_utils.CreasePattern(
        creases=[
            *_unit_borders(),
            crease_utils.Crease(_pt(0.5, 0.5), _pt(1.0, 0.5), fold=m),
            crease_utils.Crease(_pt(0.5, 0.5), _pt(0.5, 1.0), fold=m),
            crease_utils.Crease(_pt(0.5, 0.5), _pt(0.0, 0.5), fold=m),
            crease_utils.Crease(_pt(0.5, 0.5), _pt(0.5, 0.0), fold=m),
        ]
    )
    self.assertFalse(cp.local_flat_foldability())
    fixed = flat_foldable.get_flat_foldable(cp)
    self.assertIsNotNone(fixed)
    self.assertTrue(fixed.local_flat_foldability())
    center_v = next(
        vt for vt in fixed.vertices if vt.point.isclose(_pt(0.5, 0.5))
    )
    m_count = sum(1 for c in center_v.creases if c.fold == m)
    v_count = sum(1 for c in center_v.creases if c.fold == v)
    self.assertEqual((m_count, v_count), (3, 1))

  def test_rotates_boundary_crossing_crease_for_vertex_in_rectangle(self):
    example = shaping_examples.ONE_KAWASAKI_VIOLATION
    base_cp = svg.parse_svg(example.base_svg, grid_size=example.grid_size)
    _, _, bad_vertices = base_cp.vertex_flat_foldability_percentage()
    v_pt = bad_vertices[0].point
    tiny_rect = rectangle_lib.Rectangle(
        name='vertex_box',
        x_start=max(0.0, v_pt.x - 0.01),
        y_start=max(0.0, v_pt.y - 0.01),
        x_len=0.02,
        y_len=0.02,
    )
    fixed = flat_foldable.get_flat_foldable(base_cp, rectangles=[tiny_rect])
    self.assertIsNotNone(fixed)
    self.assertTrue(fixed.local_flat_foldability())
    _, _, remaining_bad = fixed.vertex_flat_foldability_percentage()
    self.assertEmpty(remaining_bad)

  def test_handles_non_mv_hinge_crease_at_violating_vertex(self):
    m = crease_utils.Fold.MOUNTAIN
    v = crease_utils.Fold.VALLEY
    f = crease_utils.Fold.HINGE
    c_right = crease_utils.Crease(_pt(0.5, 0.5), _pt(1.0, 0.5), fold=v)
    fanned = crease_utils.Crease(_pt(0.5, 0.5), _pt(0.7, 1.0), fold=m)
    c_left = crease_utils.Crease(_pt(0.5, 0.5), _pt(0.0, 0.5), fold=m)
    c_hinge = crease_utils.Crease(_pt(0.5, 0.5), _pt(0.0, 0.0), fold=f)
    c_bottom = crease_utils.Crease(_pt(0.5, 0.5), _pt(0.5, 0.0), fold=m)
    cp = cp_utils.CreasePattern(
        creases=[
            *_unit_borders(),
            c_hinge,
            fanned,
            c_bottom,
            c_left,
            c_right,
        ],
        light_mode=True,
    )
    # Locking the first three M/V creases forces rotation of `c_bottom`, whose
    # index in `violating_vertex.creases` exceeds `len(sector_angles) - 1`
    # because `c_hinge` precedes it.
    fixed = flat_foldable.get_flat_foldable(
        cp, unmodifiable_creases=[c_right, fanned, c_left]
    )
    self.assertIsNotNone(fixed)
    self.assertTrue(fixed.local_flat_foldability())
    self.assertIn(c_hinge, fixed.creases)
    # `c_bottom` must rotate on y=0 to match `fanned` (x=0.7 on y=1 -> y=0).
    rotated_bottom = [
        c
        for c in fixed.creases
        if c.fold == m
        and (c.p1.isclose(_pt(0.5, 0.5)) or c.p2.isclose(_pt(0.5, 0.5)))
        and (c.p1.y < 0.01 or c.p2.y < 0.01)
    ]
    self.assertLen(rotated_bottom, 1)
    tip = (
        rotated_bottom[0].p1
        if rotated_bottom[0].p1.y < 0.01
        else rotated_bottom[0].p2
    )
    self.assertAlmostEqual(tip.x, 0.7, places=5)
    self.assertAlmostEqual(tip.y, 0.0, places=5)


if __name__ == '__main__':
  absltest.main()
