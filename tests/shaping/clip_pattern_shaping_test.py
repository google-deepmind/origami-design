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

import math

from absl.testing import absltest
from absl.testing import parameterized
import bp_pipeline as bp
from bp_pipeline.crease_pattern import crease
from bp_pipeline.shaping import bus_adapter
from bp_pipeline.shaping import clip_pattern
from bp_pipeline.shaping import shaping_utils

_M = crease.Fold.MOUNTAIN
_V = crease.Fold.VALLEY

# Compact 7x7 U-shaped river base and hinge edges (exact unit-grid segments).
_RIVER_7X7_HINGES: tuple[tuple[int, int, int, int], ...] = (
    (0, 3, 2, 3),
    (2, 3, 2, 1),
    (2, 1, 5, 1),
    (5, 1, 5, 3),
    (5, 3, 7, 3),
    (0, 4, 3, 4),
    (3, 4, 3, 2),
    (3, 2, 4, 2),
    (4, 2, 4, 4),
    (4, 4, 7, 4),
)

_RIVER_7X7_BASE: tuple[tuple[int, int, int, int], ...] = (
    *_RIVER_7X7_HINGES,
    (2, 3, 3, 3),
    (4, 3, 5, 3),
    (1, 3, 1, 4),
    (2, 3, 2, 4),
    (5, 3, 5, 4),
    (6, 3, 6, 4),
    (2, 2, 3, 2),
    (4, 2, 5, 2),
    (3, 1, 3, 2),
    (4, 1, 4, 2),
    # 45-degree corner turn creases along the U-river.
    (2, 3, 3, 4),
    (2, 1, 3, 2),
    (4, 2, 5, 1),
    (5, 3, 4, 4),
)

# Expected clipped creases in 28x28 scaled grid units: `(p1, p2, fold, alpha)`.
_EXPECTED_RIVER_CREASES: tuple[
    tuple[tuple[float, float], tuple[float, float], crease.Fold, float], ...
] = (
    ((11.0, 7.0), (12.0, 6.0), _V, 0.5),
    ((10.0, 8.0), (11.0, 7.0), _M, 0.5),
    ((12.0, 6.0), (14.0, 8.0), _M, 0.5),
    ((10.0, 8.0), (12.0, 10.0), _V, 0.5),
    ((16.0, 10.0), (18.0, 12.0), _V, 0.5),
    ((2.0, 16.0), (4.0, 14.0), _V, 0.5),
    ((4.0, 14.0), (6.0, 16.0), _M, 0.5),
    ((11.0, 15.0), (12.0, 14.0), _M, 0.5),
    ((10.0, 16.0), (11.0, 15.0), _V, 0.5),
    ((18.0, 16.0), (20.0, 14.0), _V, 0.5),
    ((16.0, 14.0), (18.0, 12.0), _M, 0.5),
    ((20.0, 14.0), (22.0, 16.0), _M, 0.5),
    ((26.0, 16.0), (28.0, 14.0), _V, 0.5),
)


def _canonical_key(
    p1: tuple[float, float],
    p2: tuple[float, float],
    fold: crease.Fold,
    fold_percent: float,
) -> tuple[tuple[float, float], tuple[float, float], crease.Fold, float]:
  a = (round(p1[0], 6), round(p1[1], 6))
  b = (round(p2[0], 6), round(p2[1], 6))
  return (min(a, b), max(a, b), fold, round(fold_percent, 6))


def _reflect_vec_across_segment(
    vec: tuple[float, float], seg: clip_pattern.PatternSegment
) -> tuple[float, float]:
  dx = seg.p2[0] - seg.p1[0]
  dy = seg.p2[1] - seg.p1[1]
  norm = math.hypot(dx, dy)
  ux, uy = dx / norm, dy / norm
  dot = vec[0] * ux + vec[1] * uy
  return (2.0 * dot * ux - vec[0], 2.0 * dot * uy - vec[1])


class TemplatesTest(parameterized.TestCase):

  def test_rotate_realizes_exact_flap_rotation_angle(self):
    for theta_deg in range(-180, 181, 5):
      segs = clip_pattern.rotate(theta_deg)
      # Composing 2D reflections across each 180-degree crease maps the flap
      # direction (0, 1) to (-sin(theta), cos(theta)) (clockwise in SVG coords).
      vx, vy = 0.0, 1.0
      for seg in segs:
        vx, vy = _reflect_vec_across_segment((vx, vy), seg)
      rad = math.radians(theta_deg)
      self.assertAlmostEqual(vx, -math.sin(rad), places=6)
      self.assertAlmostEqual(vy, math.cos(rad), places=6)

  def test_pleat_to_hide_even_centered_lines(self):
    segs = clip_pattern.pleat_to_hide(length=1.0, distance=0.25)
    self.assertLen(segs, 4)
    for seg, y in zip(segs, (0.125, 0.375, 0.625, 0.875)):
      self.assertAlmostEqual(seg.p1[1], y)
      self.assertAlmostEqual(seg.p2[1], y)

    odd_candidate = clip_pattern.pleat_to_hide(length=1.1, distance=0.25)
    self.assertLen(odd_candidate, 4)
    self.assertEqual([s.fold for s in odd_candidate], [_M, _V, _M, _V])

  def test_narrow_turn_and_bend_templates(self):
    self.assertNotEmpty(
        clip_pattern.narrow(length=2.0, symmetric=True, n_times=2)
    )
    self.assertEmpty(clip_pattern.narrow(length=1.5, symmetric=True, n_times=2))
    self.assertLen(clip_pattern.turn_and_narrow(length=2.0), 30)
    bent_pos = clip_pattern.bend_when_turn_and_narrow(45.0)
    bent_neg = clip_pattern.bend_when_turn_and_narrow(-45.0)
    self.assertLen(bent_pos, 6)
    self.assertEqual(
        [s.fold for s in bent_neg], [s.fold.flip() for s in bent_pos]
    )

  @parameterized.named_parameters(
      dict(
          testcase_name='bend_rotate_narrow_flap',
          step=1.0,
          bend_angle=45.0,
          rotate_angle=90.0,
          enable_narrow=True,
          narrow_length=None,
          expected_count=9,
          expected_min_y=1.0,
      ),
      dict(
          testcase_name='bend_rotate_narrow_river',
          step=1.0,
          bend_angle=45.0,
          rotate_angle=90.0,
          enable_narrow=True,
          narrow_length=5.0,
          expected_count=15,
          expected_min_y=1.0,
      ),
      dict(
          testcase_name='narrow_exact_length_two',
          step=1.0,
          bend_angle=0.0,
          rotate_angle=0.0,
          enable_narrow=True,
          narrow_length=3.0,
          expected_count=12,
          expected_min_y=1.0,
      ),
      dict(
          testcase_name='narrow_too_short_emits_empty',
          step=1.0,
          bend_angle=0.0,
          rotate_angle=0.0,
          enable_narrow=True,
          narrow_length=2.99,
          expected_count=0,
          expected_min_y=0.0,
      ),
      dict(
          testcase_name='zero_step_zero_angles_emits_empty',
          step=0.0,
          bend_angle=0.0,
          rotate_angle=0.0,
          enable_narrow=False,
          narrow_length=None,
          expected_count=0,
          expected_min_y=0.0,
      ),
      dict(
          testcase_name='step_2_rotate_90_only',
          step=2.0,
          bend_angle=0.0,
          rotate_angle=90.0,
          enable_narrow=False,
          narrow_length=None,
          expected_count=1,
          expected_min_y=2.0,
      ),
      dict(
          testcase_name='step_2_bend_neg45_only',
          step=2.0,
          bend_angle=-45.0,
          rotate_angle=0.0,
          enable_narrow=False,
          narrow_length=None,
          expected_count=1,
          expected_min_y=2.0,
      ),
      dict(
          testcase_name='zero_step_bend_adds_margin',
          step=0.0,
          bend_angle=-45.0,
          rotate_angle=0.0,
          enable_narrow=False,
          narrow_length=None,
          expected_count=1,
          expected_min_y=0.05,
      ),
      dict(
          testcase_name='bend_and_180_rotate',
          step=2.0,
          bend_angle=-45.0,
          rotate_angle=180.0,
          enable_narrow=False,
          narrow_length=None,
          expected_count=2,
          expected_min_y=2.0,
      ),
  )
  def test_build_fold_pattern(
      self,
      step: float,
      bend_angle: float,
      rotate_angle: float,
      enable_narrow: bool,
      narrow_length: float | None,
      expected_count: int,
      expected_min_y: float,
  ):
    segs = clip_pattern.build_fold_pattern(
        margin=0.05,
        step=step,
        bend_angle=bend_angle,
        rotate_angle=rotate_angle,
        enable_narrow=enable_narrow,
        narrow_symmetric=False,
        narrow_length=narrow_length,
    )
    self.assertLen(segs, expected_count)
    if segs:
      ys = []
      for s in segs:
        ys.extend((s.p1[1], s.p2[1]))
      self.assertAlmostEqual(min(ys), expected_min_y)


class GridShaperTest(absltest.TestCase):

  def test_shapes_and_clips_u_river_with_exact_reflections(self):
    base_grid = clip_pattern.FoldGrid.from_grid_segments(7, _RIVER_7X7_BASE)
    hinges_grid = clip_pattern.FoldGrid.from_grid_segments(7, _RIVER_7X7_HINGES)
    pattern = (
        # Include a degenerate micro-segment to verify tolerance filtering.
        clip_pattern.PatternSegment(
            (0.5, 0.5), (0.5 + 1e-9, 0.5 + 1e-9), _V, 90.0
        ),
        clip_pattern.PatternSegment((0.5, 0.0), (1.5, 1.0), _V, 90.0),
    )
    creases = clip_pattern.shape_clip_pattern(
        base_grid=base_grid,
        hinges_grid=hinges_grid,
        align_triangle=(3, 0, clip_pattern.CellTriangle.BOTTOM),
        pattern=pattern,
    )
    actual = [
        _canonical_key(
            (c.p1_init.x * 28.0, c.p1_init.y * 28.0),
            (c.p2_init.x * 28.0, c.p2_init.y * 28.0),
            c.fold,
            c.fold_percent,
        )
        for c in creases
    ]
    expected = [_canonical_key(*entry) for entry in _EXPECTED_RIVER_CREASES]
    self.assertCountEqual(actual, expected)

  def test_human_packing_layout_anchors_and_shaping(self):
    tri = clip_pattern.CellTriangle
    rect_fn = lambda name, x, y, w, h: shaping_utils.Rectangle(
        name=name, x_start=x, y_start=y, x_len=w, y_len=h
    )
    flaps = [
        rect_fn('left_arm', 0.0, 0.0, 0.375, 0.375),
        rect_fn('head', 0.375, 0.0, 0.25, 0.375),
        rect_fn('right_arm', 0.625, 0.0, 0.375, 0.375),
        rect_fn('left_leg', 0.0, 0.5, 0.5, 0.5),
        rect_fn('right_leg', 0.5, 0.5, 0.5, 0.5),
    ]
    body = rect_fn('body', 0.0, 0.375, 1.0, 0.125)
    hinges = clip_pattern.FoldGrid.from_packing_layout(flaps, [[body]], 16)
    base = clip_pattern.FoldGrid.from_grid_segments(16, ())
    shaper = clip_pattern.ClipPatternShaper.from_grids(base, hinges)
    expected = {
        'left_arm': (5, 5, tri.RIGHT),
        'head': (5, 8, tri.BOTTOM),
        'right_arm': (5, 10, tri.LEFT),
        'body': (7, 8, tri.BOTTOM),
        'left_leg': (8, 7, tri.TOP),
        'right_leg': (8, 8, tri.TOP),
    }
    self.assertCountEqual(shaper.align_triangles, expected.values())
    for rect in [*flaps, body]:
      self.assertEqual(
          shaper.get_align_triangle_for_rect(rect), expected[rect.name]
      )

    pleat = clip_pattern.build_template_pattern(
        template_name=clip_pattern.TemplateName.PLEAT_TO_HIDE,
        margin=0.0,
        step=1.0,
        length=3.0,
        pleat_distance=0.5,
    )
    self.assertLen(pleat, 4)
    tiny_miss = rect_fn('empty', 0.0, 0.375, 0.01, 0.01)
    creases = shaper.shape_rectangles([tiny_miss, body], pleat)
    self.assertLen(creases, 2)
    self.assertEqual(sum(1 for c in creases if c.fold == _M), 1)
    self.assertEqual(sum(1 for c in creases if c.fold == _V), 1)

  def test_from_crease_patterns_on_solved_model(self):
    result = bp.fold('human', preset='demo', shape=False)
    packing_out, _ = bus_adapter.legacy_inputs(result.solved)
    shaper = clip_pattern.ClipPatternShaper.from_crease_patterns(
        base_cp=result.solved.creases,
        packing_cp=packing_out.packing_cp,
    )
    grid_size = result.solved.creases.grid_size
    self.assertEqual(shaper.grid_size, 12)
    self.assertLen(shaper.align_triangles, 9)
    first_flap = next(iter(result.solved.packed.flaps.values()))
    self.assertEqual(first_flap.id, 'left_arm')
    clipped = shaping_utils.Rectangle(
        name=first_flap.id,
        x_start=first_flap.rect.x / grid_size,
        y_start=first_flap.rect.y / grid_size,
        x_len=first_flap.rect.width / grid_size,
        y_len=first_flap.rect.height / grid_size,
    ).clip(xmin=0, ymin=0, xmax=1, ymax=1)
    self.assertIsNotNone(clipped)
    pattern = clip_pattern.build_fold_pattern(
        margin=0.05, step=0.5, bend_angle=45.0
    )
    creases = shaper.shape_rectangles([clipped], pattern)
    expected = [
        _canonical_key((0.0, 5 / 24), (1 / 12, 5 / 24), _V, 0.25),
        _canonical_key((1 / 12, 5 / 24), (2 / 12, 5 / 24), _M, 0.25),
        _canonical_key((2 / 12, 5 / 24), (3 / 12, 5 / 24), _V, 0.25),
        _canonical_key((3 / 12, 5 / 24), (4 / 12, 5 / 24), _M, 0.25),
        _canonical_key((4 / 12, 5 / 24), (9 / 24, 5 / 24), _V, 0.25),
        _canonical_key((9 / 24, 0.0), (9 / 24, 1 / 12), _M, 0.25),
        _canonical_key((9 / 24, 1 / 12), (9 / 24, 2 / 12), _V, 0.25),
        _canonical_key((9 / 24, 2 / 12), (9 / 24, 5 / 24), _M, 0.25),
    ]
    actual = [
        _canonical_key(
            (c.p1_init.x, c.p1_init.y),
            (c.p2_init.x, c.p2_init.y),
            c.fold,
            c.fold_percent,
        )
        for c in creases
    ]
    self.assertCountEqual(actual, expected)


if __name__ == '__main__':
  absltest.main()
