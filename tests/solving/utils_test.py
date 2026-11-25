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

"""Tests for Box Pleating utils object."""

import math

from absl.testing import absltest
from absl.testing import parameterized
from bp_pipeline.crease_pattern import crease
from bp_pipeline.crease_pattern import point
from bp_pipeline.crease_pattern.parser import svg
from bp_pipeline.solving import utils as solving_utils


class UtilsTest(parameterized.TestCase):

  def test_orientation_after_quantization(self):
    xml = """
    <svg xmlns="http://www.w3.org/2000/svg" version="1.1" xmlns:xlink="http://www.w3.org/1999/xlink" xmlns:svgjs="http://svgjs.dev/svgjs" viewBox="-0.5 -0.5 33 33" width="33cm" height="33cm"><g class="crease"><line x1="32" y1="32" x2="24" y2="24" class="M" stroke-opacity="1" data-index="0" stroke-width="0.1" stroke="#ff0000"></line><line x1="0" y1="0" x2="8" y2="8" class="M" stroke-opacity="1" data-index="1" stroke-width="0.1" stroke="#ff0000"></line><line x1="8" y1="8" x2="0" y2="16" class="M" stroke-opacity="1" data-index="2" stroke-width="0.1" stroke="#ff0000"></line><line x1="24" y1="24" x2="32" y2="16" class="M" stroke-opacity="1" data-index="3" stroke-width="0.1" stroke="#ff0000"></line><line x1="32" y1="16" x2="24" y2="8" class="M" stroke-opacity="1" data-index="4" stroke-width="0.1" stroke="#ff0000"></line><line x1="24" y1="8" x2="32" y2="0" class="M" stroke-opacity="1" data-index="5" stroke-width="0.1" stroke="#ff0000"></line><line x1="8" y1="6" x2="14" y2="0" class="M" stroke-opacity="1" data-index="6" stroke-width="0.1" stroke="#ff0000"></line><line x1="24" y1="6" x2="18" y2="0" class="M" stroke-opacity="1" data-index="7" stroke-width="0.1" stroke="#ff0000"></line><line x1="0" y1="16" x2="8" y2="24" class="M" stroke-opacity="1" data-index="8" stroke-width="0.1" stroke="#ff0000"></line><line x1="8" y1="24" x2="0" y2="32" class="M" stroke-opacity="1" data-index="9" stroke-width="0.1" stroke="#ff0000"></line><line x1="8" y1="26" x2="14" y2="32" class="M" stroke-opacity="1" data-index="10" stroke-width="0.1" stroke="#ff0000"></line><line x1="24" y1="26" x2="18" y2="32" class="M" stroke-opacity="1" data-index="11" stroke-width="0.1" stroke="#ff0000"></line><line x1="8" y1="6" x2="12" y2="10" class="M" stroke-opacity="1" data-index="12" stroke-width="0.1" stroke="#ff0000"></line><line x1="16" y1="6" x2="12" y2="10" class="M" stroke-opacity="1" data-index="13" stroke-width="0.1" stroke="#ff0000"></line><line x1="12" y1="10" x2="8" y2="14" class="M" stroke-opacity="1" data-index="14" stroke-width="0.1" stroke="#ff0000"></line><line x1="12" y1="10" x2="16" y2="14" class="M" stroke-opacity="1" data-index="15" stroke-width="0.1" stroke="#ff0000"></line><line x1="16" y1="6" x2="20" y2="10" class="M" stroke-opacity="1" data-index="16" stroke-width="0.1" stroke="#ff0000"></line><line x1="24" y1="6" x2="20" y2="10" class="M" stroke-opacity="1" data-index="17" stroke-width="0.1" stroke="#ff0000"></line><line x1="20" y1="10" x2="16" y2="14" class="M" stroke-opacity="1" data-index="18" stroke-width="0.1" stroke="#ff0000"></line><line x1="20" y1="10" x2="24" y2="14" class="M" stroke-opacity="1" data-index="19" stroke-width="0.1" stroke="#ff0000"></line><line x1="8" y1="26" x2="14" y2="20" class="M" stroke-opacity="1" data-index="20" stroke-width="0.1" stroke="#ff0000"></line><line x1="14" y1="20" x2="8" y2="14" class="M" stroke-opacity="1" data-index="21" stroke-width="0.1" stroke="#ff0000"></line><line x1="24" y1="14" x2="18" y2="20" class="M" stroke-opacity="1" data-index="22" stroke-width="0.1" stroke="#ff0000"></line><line x1="18" y1="20" x2="24" y2="26" class="M" stroke-opacity="1" data-index="23" stroke-width="0.1" stroke="#ff0000"></line><line x1="14" y1="20" x2="18" y2="20" class="M" stroke-opacity="1" data-index="24" stroke-width="0.1" stroke="#ff0000"></line><line x1="0" y1="8" x2="8" y2="8" class="C" stroke-opacity="1" data-index="25" stroke-width="0.1" stroke="#00ff00"></line><line x1="8" y1="8" x2="8" y2="6" class="C" stroke-opacity="1" data-index="26" stroke-width="0.1" stroke="#00ff00"></line><line x1="8" y1="6" x2="8" y2="0" class="C" stroke-opacity="1" data-index="27" stroke-width="0.1" stroke="#00ff00"></line><line x1="8" y1="6" x2="16" y2="6" class="C" stroke-opacity="1" data-index="28" stroke-width="0.1" stroke="#00ff00"></line><line x1="16" y1="6" x2="24" y2="6" class="C" stroke-opacity="1" data-index="29" stroke-width="0.1" stroke="#00ff00"></line><line x1="24" y1="6" x2="24" y2="0" class="C" stroke-opacity="1" data-index="30" stroke-width="0.1" stroke="#00ff00"></line><line x1="24" y1="8" x2="24" y2="6" class="C" stroke-opacity="1" data-index="31" stroke-width="0.1" stroke="#00ff00"></line><line x1="24" y1="8" x2="32" y2="8" class="C" stroke-opacity="1" data-index="32" stroke-width="0.1" stroke="#00ff00"></line><line x1="24" y1="8" x2="24" y2="14" class="C" stroke-opacity="1" data-index="33" stroke-width="0.1" stroke="#00ff00"></line><line x1="24" y1="24" x2="24" y2="26" class="C" stroke-opacity="1" data-index="34" stroke-width="0.1" stroke="#00ff00"></line><line x1="24" y1="14" x2="24" y2="24" class="C" stroke-opacity="1" data-index="35" stroke-width="0.1" stroke="#00ff00"></line><line x1="24" y1="26" x2="24" y2="32" class="C" stroke-opacity="1" data-index="36" stroke-width="0.1" stroke="#00ff00"></line><line x1="24" y1="24" x2="32" y2="24" class="C" stroke-opacity="1" data-index="37" stroke-width="0.1" stroke="#00ff00"></line><line x1="8" y1="8" x2="8" y2="14" class="C" stroke-opacity="1" data-index="38" stroke-width="0.1" stroke="#00ff00"></line><line x1="8" y1="24" x2="8" y2="26" class="C" stroke-opacity="1" data-index="39" stroke-width="0.1" stroke="#00ff00"></line><line x1="8" y1="14" x2="8" y2="24" class="C" stroke-opacity="1" data-index="40" stroke-width="0.1" stroke="#00ff00"></line><line x1="8" y1="26" x2="8" y2="32" class="C" stroke-opacity="1" data-index="41" stroke-width="0.1" stroke="#00ff00"></line><line x1="8" y1="24" x2="0" y2="24" class="C" stroke-opacity="1" data-index="42" stroke-width="0.1" stroke="#00ff00"></line><line x1="8" y1="26" x2="24" y2="26" class="C" stroke-opacity="1" data-index="43" stroke-width="0.1" stroke="#00ff00"></line><line x1="8" y1="14" x2="16" y2="14" class="C" stroke-opacity="1" data-index="44" stroke-width="0.1" stroke="#00ff00"></line><line x1="16" y1="14" x2="24" y2="14" class="C" stroke-opacity="1" data-index="45" stroke-width="0.1" stroke="#00ff00"></line><line x1="16" y1="14" x2="16" y2="6" class="C" stroke-opacity="1" data-index="46" stroke-width="0.1" stroke="#00ff00"></line><line x1="0" y1="0" x2="0" y2="8" class="B" stroke-opacity="1" data-index="47" stroke="#000000" stroke-width="0.1"></line><line x1="0" y1="16" x2="0" y2="24" class="B" stroke-opacity="1" data-index="48" stroke="#000000" stroke-width="0.1"></line><line x1="0" y1="8" x2="0" y2="16" class="B" stroke-opacity="1" data-index="49" stroke="#000000" stroke-width="0.1"></line><line x1="0" y1="24" x2="0" y2="32" class="B" stroke-opacity="1" data-index="50" stroke="#000000" stroke-width="0.1"></line><line x1="0" y1="32" x2="8" y2="32" class="B" stroke-opacity="1" data-index="51" stroke="#000000" stroke-width="0.1"></line><line x1="14" y1="32" x2="18" y2="32" class="B" stroke-opacity="1" data-index="52" stroke="#000000" stroke-width="0.1"></line><line x1="18" y1="32" x2="24" y2="32" class="B" stroke-opacity="1" data-index="53" stroke="#000000" stroke-width="0.1"></line><line x1="8" y1="32" x2="14" y2="32" class="B" stroke-opacity="1" data-index="54" stroke="#000000" stroke-width="0.1"></line><line x1="24" y1="32" x2="32" y2="32" class="B" stroke-opacity="1" data-index="55" stroke="#000000" stroke-width="0.1"></line><line x1="32" y1="32" x2="32" y2="24" class="B" stroke-opacity="1" data-index="56" stroke="#000000" stroke-width="0.1"></line><line x1="32" y1="16" x2="32" y2="8" class="B" stroke-opacity="1" data-index="57" stroke="#000000" stroke-width="0.1"></line><line x1="32" y1="24" x2="32" y2="16" class="B" stroke-opacity="1" data-index="58" stroke="#000000" stroke-width="0.1"></line><line x1="32" y1="8" x2="32" y2="0" class="B" stroke-opacity="1" data-index="59" stroke="#000000" stroke-width="0.1"></line><line x1="32" y1="0" x2="24" y2="0" class="B" stroke-opacity="1" data-index="60" stroke="#000000" stroke-width="0.1"></line><line x1="14" y1="0" x2="8" y2="0" class="B" stroke-opacity="1" data-index="61" stroke="#000000" stroke-width="0.1"></line><line x1="18" y1="0" x2="14" y2="0" class="B" stroke-opacity="1" data-index="62" stroke="#000000" stroke-width="0.1"></line><line x1="24" y1="0" x2="18" y2="0" class="B" stroke-opacity="1" data-index="63" stroke="#000000" stroke-width="0.1"></line><line x1="8" y1="0" x2="0" y2="0" class="B" stroke-opacity="1" data-index="64" stroke="#000000" stroke-width="0.1"></line></g><style>line { stroke-linecap: round; }</style></svg>"""
    cp = svg.parse_svg(xml)
    ridges = [
        c
        for c in cp.creases
        if c.fold == crease.Fold.MOUNTAIN and not c.is_horizontal
    ]
    for a in ridges:
      self.assertTrue(
          all(a.is_perpendicular(c) or a.is_parallel(c) for c in ridges)
      )

    cp_cycle = svg.parse_svg(svg.to_svg(cp))
    ridges_cycle = [
        c
        for c in cp_cycle.creases
        if c.fold == crease.Fold.MOUNTAIN and not c.is_horizontal
    ]
    for a in ridges_cycle:
      self.assertFalse(
          all(a.is_perpendicular(c) or a.is_parallel(c) for c in ridges_cycle)
      )

    cp_cycle_grid = svg.parse_svg(svg.to_svg(cp), grid_size=32)
    ridges_cycle_grid = [
        c
        for c in cp_cycle_grid.creases
        if c.fold == crease.Fold.MOUNTAIN and not c.is_horizontal
    ]
    for a in ridges_cycle_grid:
      self.assertTrue(
          all(
              a.is_perpendicular(c) or a.is_parallel(c)
              for c in ridges_cycle_grid
          )
      )
    self.assertLen(cp.creases, len(cp_cycle_grid.creases))
    self.assertLen(cp.vertices, len(cp_cycle_grid.vertices))
    for v1, v2 in zip(cp.vertices, cp_cycle_grid.vertices, strict=True):
      self.assertAlmostEqual(v1.point.x, v2.point.x, places=3)
      self.assertAlmostEqual(v1.point.y, v2.point.y, places=3)

    remaining = list(cp_cycle_grid.creases)
    for c in cp.creases:
      match_idx = next(
          (
              i
              for i, cand in enumerate(remaining)
              if c.fold == cand.fold
              and (
                  (
                      math.isclose(c.p1.x, cand.p1.x, abs_tol=1e-3)
                      and math.isclose(c.p1.y, cand.p1.y, abs_tol=1e-3)
                      and math.isclose(c.p2.x, cand.p2.x, abs_tol=1e-3)
                      and math.isclose(c.p2.y, cand.p2.y, abs_tol=1e-3)
                  )
                  or (
                      math.isclose(c.p1.x, cand.p2.x, abs_tol=1e-3)
                      and math.isclose(c.p1.y, cand.p2.y, abs_tol=1e-3)
                      and math.isclose(c.p2.x, cand.p1.x, abs_tol=1e-3)
                      and math.isclose(c.p2.y, cand.p1.y, abs_tol=1e-3)
                  )
              )
          ),
          None,
      )
      self.assertIsNotNone(match_idx, msg=f'Unmatched crease: {c}')
      remaining.pop(match_idx)

  def test_deduplicate_and_creases_to_paths_preserve_input_order(self):
    c1 = crease.Crease(
        point.Point(x=0.0, y=0.25),
        point.Point(x=0.5, y=0.25),
        fold=crease.Fold.MOUNTAIN,
    )
    c2 = crease.Crease(
        point.Point(x=0.25, y=0.0),
        point.Point(x=0.25, y=0.5),
        fold=crease.Fold.MOUNTAIN,
    )
    c3 = crease.Crease(
        point.Point(x=0.5, y=0.25),
        point.Point(x=1.0, y=0.25),
        fold=crease.Fold.MOUNTAIN,
    )
    deduped = solving_utils.deduplicate([c1, c2, c1, c3])
    self.assertEqual([c.exact_match for c in deduped], [
        c1.exact_match,
        c2.exact_match,
        c3.exact_match,
    ])
    paths = solving_utils.creases_to_paths(deduped)
    self.assertLen(paths, 2)
    self.assertEqual(
        [c.exact_match for c in paths[0]], [c1.exact_match, c3.exact_match]
    )
    self.assertEqual([c.exact_match for c in paths[1]], [c2.exact_match])


if __name__ == '__main__':
  absltest.main()
