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

from bp_pipeline.shaping.clip_pattern import drawing_machine
from shaping import shaping_test_fixtures as examples
from bp_pipeline.shaping.clip_pattern import clip_pattern_shaping_adapter
from bp_pipeline.shaping import shaping_utils
from bp_pipeline.shaping.clip_pattern import triangle_geometry
from absl.testing import absltest as googletest
from absl.testing import parameterized

CellTriangle = triangle_geometry.CellTriangle


class ClipPatternShapingAdapterTest(parameterized.TestCase):

  def test_clip_pattern_shaping_adapter(self):
    adapter = clip_pattern_shaping_adapter.ClipPatternShapingAdapter.from_crease_patterns(
        base_cp=examples.HUMAN_WITH_CENTER.base_cp,
        packing_cp=examples.HUMAN_WITH_CENTER.packing_cp,
    )
    self.assertIsNotNone(adapter.base_grid)
    self.assertIsNotNone(adapter.hinges_grid)
    self.assertEqual(adapter.grid_size, 16)
    self.assertEqual(adapter.base_grid.num_rows, adapter.grid_size)
    self.assertEqual(adapter.base_grid.num_cols, adapter.grid_size)
    self.assertEqual(adapter.hinges_grid.num_rows, adapter.grid_size)
    self.assertEqual(adapter.hinges_grid.num_cols, adapter.grid_size)
    self.assertLen(adapter.align_triangles, 6)
    self.assertCountEqual(
        adapter.align_triangles,
        [
            (5, 5, CellTriangle.RIGHT),
            (5, 8, CellTriangle.BOTTOM),
            (5, 10, CellTriangle.LEFT),
            (7, 8, CellTriangle.BOTTOM),
            (8, 7, CellTriangle.TOP),
            (8, 8, CellTriangle.TOP),
        ],
    )

  def test_get_align_triangle_for_rect(self):
    adapter = clip_pattern_shaping_adapter.ClipPatternShapingAdapter.from_crease_patterns(
        base_cp=examples.HUMAN_WITH_CENTER.base_cp,
        packing_cp=examples.HUMAN_WITH_CENTER.packing_cp,
    )
    expected_align_triangles = dict({
        'left_arm': (5, 5, CellTriangle.RIGHT),
        'head': (5, 8, CellTriangle.BOTTOM),
        'right_arm': (5, 10, CellTriangle.LEFT),
        'body': (7, 8, CellTriangle.BOTTOM),
        'left_leg': (8, 7, CellTriangle.TOP),
        'right_leg': (8, 8, CellTriangle.TOP),
    })
    rectangles = [
        shaping_utils.Rectangle(
            name='left_arm',
            x_start=0,
            y_start=0,
            x_len=0.375,
            y_len=0.375,
        ),
        shaping_utils.Rectangle(
            name='head',
            x_start=0.375,
            y_start=0,
            x_len=0.25,
            y_len=0.375,
        ),
        shaping_utils.Rectangle(
            name='right_arm',
            x_start=0.375 + 0.25,
            y_start=0,
            x_len=0.375,
            y_len=0.375,
        ),
        shaping_utils.Rectangle(
            name='left_leg',
            x_start=0,
            y_start=0.5,
            x_len=0.5,
            y_len=0.5,
        ),
        shaping_utils.Rectangle(
            name='right_leg',
            x_start=0.5,
            y_start=0.5,
            x_len=0.5,
            y_len=0.5,
        ),
        shaping_utils.Rectangle(
            name='body',
            x_start=0,
            y_start=0.375,
            x_len=1,
            y_len=0.125,
        ),
    ]
    self.assertLen(adapter.align_triangles, len(rectangles))
    for rect in rectangles:
      align_triangle = adapter.get_align_triangle_for_rect(rect)
      self.assertIsNotNone(align_triangle)
      self.assertIn(rect.name, expected_align_triangles)
      self.assertEqual(align_triangle, expected_align_triangles[rect.name])

  @parameterized.named_parameters(
      dict(
          testcase_name='step_1_bend_45_rotate_90_narrow_True',
          step=1,
          bend_angle=45,
          rotate_angle=90,
          enable_narrow=True,
          narrow_symmetric=False,
          narrow_length=None,
          expected_lines_count=1 + 1 + 7,
          expected_step=1,
      ),
      dict(
          testcase_name='step_1_bend_45_rotate_90_narrow_True_narrow_length_5',
          step=1,
          bend_angle=45,
          rotate_angle=90,
          enable_narrow=True,
          narrow_symmetric=False,
          narrow_length=5,
          expected_lines_count=1 + 1 + 7 + 6,
          expected_step=1,
      ),
      dict(
          testcase_name='step_1_bend_0_rotate_0_narrow_True_narrow_length_3',
          step=1,
          bend_angle=0,
          rotate_angle=0,
          enable_narrow=True,
          narrow_symmetric=False,
          narrow_length=3,
          expected_lines_count=6 + 6,
          expected_step=1,
      ),
      dict(
          testcase_name='step_0_bend_0_rotate_0_narrow_True_narrow_length_2_99',
          step=1,
          bend_angle=0,
          rotate_angle=0,
          enable_narrow=True,
          narrow_symmetric=False,
          narrow_length=2.99,
          expected_lines_count=0,
          expected_step=1,
      ),
      dict(
          testcase_name='step_0_bend_0_rotate_0_narrow_False',
          step=0,
          bend_angle=0,
          rotate_angle=0,
          enable_narrow=False,
          narrow_symmetric=False,
          narrow_length=None,
          expected_lines_count=0,
          expected_step=0,
      ),
      dict(
          testcase_name='step_2_bend_0_rotate_90_narrow_False',
          step=2,
          bend_angle=0,
          rotate_angle=90,
          enable_narrow=False,
          narrow_symmetric=False,
          narrow_length=None,
          expected_lines_count=1,
          expected_step=2,
      ),
      dict(
          testcase_name='step_2_bend_-45_rotate_0_narrow_False',
          step=2,
          bend_angle=-45,
          rotate_angle=0,
          enable_narrow=False,
          narrow_symmetric=False,
          narrow_length=None,
          expected_lines_count=1,
          expected_step=2,
      ),
      dict(
          testcase_name='step_0_bend_-45_rotate_0_narrow_False',
          step=0,
          bend_angle=-45,
          rotate_angle=0,
          enable_narrow=False,
          narrow_symmetric=False,
          narrow_length=None,
          expected_lines_count=1,
          expected_step=0.05,  # When step is 0, bend adds a small step.
      ),
      dict(
          testcase_name='step_2_bend_-45_rotate_180_narrow_False',
          step=2,
          bend_angle=-45,
          rotate_angle=180,
          enable_narrow=False,
          narrow_symmetric=False,
          narrow_length=None,
          expected_lines_count=1 + 1,
          expected_step=2,
      ),
  )
  def test_get_clip_pattern_fold_draw_pattern(
      self,
      step,
      bend_angle,
      rotate_angle,
      enable_narrow,
      narrow_symmetric,
      narrow_length,
      expected_lines_count,
      expected_step,
  ):
    draw_pattern = (
        clip_pattern_shaping_adapter.get_clip_pattern_fold_draw_pattern(
            margin=0.05,
            step=step,
            bend_angle=bend_angle,
            rotate_angle=rotate_angle,
            enable_narrow=enable_narrow,
            narrow_symmetric=narrow_symmetric,
            narrow_length=narrow_length,
        )
    )
    machine = drawing_machine.CreasePatternDrawingMachine()
    draw_pattern(machine)
    self.assertLen(machine.lines, expected_lines_count)
    if expected_lines_count > 0:
      coords = []
      for line in machine.lines:
        coords.append(line.p1[1])
        coords.append(line.p2[1])
      self.assertAlmostEqual(min(coords), expected_step)

  def test_pleat_to_hide_custom_pleat_distance_and_multi_rect_fallback(self):
    draw_pattern = clip_pattern_shaping_adapter.get_clip_pattern_template_draw_pattern(
        template_name=clip_pattern_shaping_adapter.TemplateName.PLEAT_TO_HIDE,
        margin=0.0,
        step=1.0,
        length=3.0,
        pleat_distance=0.5,
    )
    machine = drawing_machine.CreasePatternDrawingMachine()
    draw_pattern(machine)
    # Effective length = 3.0 - 1.0 = 2.0 at distance 0.5 -> 4 lines.
    self.assertLen(machine.lines, 4)

    adapter = clip_pattern_shaping_adapter.ClipPatternShapingAdapter.from_crease_patterns(
        base_cp=examples.HUMAN_WITH_CENTER.base_cp,
        packing_cp=examples.HUMAN_WITH_CENTER.packing_cp,
    )
    empty_rect = shaping_utils.Rectangle(
        name='empty_subrect', x_start=0.0, y_start=0.375, x_len=0.01, y_len=0.01
    )
    body_rect = shaping_utils.Rectangle(
        name='body', x_start=0.0, y_start=0.375, x_len=1.0, y_len=0.125
    )
    lines = adapter.get_clip_pattern_shaping_lines(
        [empty_rect, body_rect], draw_pattern
    )
    self.assertNotEmpty(lines)


if __name__ == '__main__':
  googletest.main()
