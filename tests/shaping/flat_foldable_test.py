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

"""Tests for flat foldability solver."""

from bp_pipeline.shaping import flat_foldable
from shaping import shaping_test_fixtures as shaping_examples
from bp_pipeline.crease_pattern.parser import svg
from absl.testing import absltest as googletest
from absl.testing import parameterized

cp_utils = flat_foldable.cp_utils
crease_utils = flat_foldable.crease_utils
point_utils = flat_foldable.point_utils
shaping_utils = flat_foldable.shaping_utils


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
    result = flat_foldable.get_flat_foldable(base_cp, max_candidates=1)
    self.assertIsNone(result)

  def test_max_crease_modifications_limits_search(self):
    example = shaping_examples.FOUR_KAWASAKI_VIOLATIONS
    base_cp = svg.parse_svg(example.base_svg, grid_size=example.grid_size)
    self.assertFalse(base_cp.local_flat_foldability())
    result = flat_foldable.get_flat_foldable(
        base_cp, max_crease_modifications=0
    )
    self.assertIsNone(result)

  def test_preserves_light_mode_and_splits_merged_border(self):
    # A degree-4 vertex at (0.5, 0.5) whose bottom crease moved along a single
    # merged bottom border edge (0,1)->(1,1) in light_mode=True.
    p = lambda x, y: point_utils.Point(x=x, y=y)
    borders = [
        crease_utils.Crease(p(0, 0), p(1, 0), fold=crease_utils.Fold.BORDER),
        crease_utils.Crease(p(1, 0), p(1, 1), fold=crease_utils.Fold.BORDER),
        crease_utils.Crease(p(1, 1), p(0, 1), fold=crease_utils.Fold.BORDER),
        crease_utils.Crease(p(0, 1), p(0, 0), fold=crease_utils.Fold.BORDER),
    ]
    fanned = crease_utils.Crease(
        p(0.5, 0.5), p(0.75, 1.0), fold=crease_utils.Fold.MOUNTAIN
    )
    creases = borders + [
        fanned,
        crease_utils.Crease(
            p(0.5, 0.5), p(0.5, 0.0), fold=crease_utils.Fold.MOUNTAIN
        ),
        crease_utils.Crease(
            p(0.5, 0.5), p(0.0, 0.5), fold=crease_utils.Fold.MOUNTAIN
        ),
        crease_utils.Crease(
            p(0.5, 0.5), p(1.0, 0.5), fold=crease_utils.Fold.VALLEY
        ),
    ]
    cp = cp_utils.CreasePattern(creases=creases, light_mode=True)
    fixed = flat_foldable.get_flat_foldable(cp, unmodifiable_creases=[fanned])
    self.assertIsNotNone(fixed)
    self.assertTrue(fixed.light_mode)
    self.assertTrue(fixed.local_flat_foldability())

  def test_flips_border_crease_to_resolve_maekawa_and_crimp_violation(self):
    # Vertex at (0.5, 0.5) satisfies Kawasaki (angles 90, 90, 90, 90 ->
    # 90+90 == 90+90 == 180) but violates Maekawa (all 4 MOUNTAIN).
    p = lambda x, y: point_utils.Point(x=x, y=y)
    borders = [
        crease_utils.Crease(p(0, 0), p(1, 0), fold=crease_utils.Fold.BORDER),
        crease_utils.Crease(p(1, 0), p(1, 1), fold=crease_utils.Fold.BORDER),
        crease_utils.Crease(p(1, 1), p(0, 1), fold=crease_utils.Fold.BORDER),
        crease_utils.Crease(p(0, 1), p(0, 0), fold=crease_utils.Fold.BORDER),
    ]
    creases = borders + [
        crease_utils.Crease(
            p(0.5, 0.5), p(1.0, 0.5), fold=crease_utils.Fold.MOUNTAIN
        ),
        crease_utils.Crease(
            p(0.5, 0.5), p(0.5, 1.0), fold=crease_utils.Fold.MOUNTAIN
        ),
        crease_utils.Crease(
            p(0.5, 0.5), p(0.0, 0.5), fold=crease_utils.Fold.MOUNTAIN
        ),
        crease_utils.Crease(
            p(0.5, 0.5), p(0.5, 0.0), fold=crease_utils.Fold.MOUNTAIN
        ),
    ]
    cp = cp_utils.CreasePattern(creases=creases)
    self.assertFalse(cp.local_flat_foldability())
    fixed = flat_foldable.get_flat_foldable(cp)
    self.assertIsNotNone(fixed)
    self.assertTrue(fixed.local_flat_foldability())

  def test_allows_boundary_crossing_crease_for_vertex_in_rectangle(self):
    example = shaping_examples.ONE_KAWASAKI_VIOLATION
    base_cp = svg.parse_svg(example.base_svg, grid_size=example.grid_size)
    _, _, bad_vertices = base_cp.vertex_flat_foldability_percentage()
    v_pt = bad_vertices[0].point
    tiny_rect = shaping_utils.Rectangle(
        name='vertex_box',
        x_start=max(0.0, v_pt.x - 0.01),
        y_start=max(0.0, v_pt.y - 0.01),
        x_len=0.02,
        y_len=0.02,
    )
    fixed = flat_foldable.get_flat_foldable(base_cp, rectangles=[tiny_rect])
    self.assertIsNotNone(fixed)
    self.assertTrue(fixed.local_flat_foldability())


if __name__ == '__main__':
  googletest.main()
