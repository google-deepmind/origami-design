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
"""Tests for fan-out shaping of pleated flaps."""

import dataclasses
import itertools

from absl.testing import absltest
from absl.testing import parameterized
import bp_pipeline as bp
from bp_pipeline.crease_pattern import crease as crease_utils
from bp_pipeline.crease_pattern import crease_pattern as cp_utils
from bp_pipeline.crease_pattern import point as point_utils
from bp_pipeline.shaping import fan_out_shaping
from bp_pipeline.shaping import shaping_utils

_M = crease_utils.Fold.MOUNTAIN
_V = crease_utils.Fold.VALLEY


def _pt(x: float, y: float) -> point_utils.Point:
  return point_utils.Point(x=x, y=y)


def _border_creases() -> list[crease_utils.Crease]:
  b = crease_utils.Fold.BORDER
  return [
      crease_utils.Crease(_pt(0, 0), _pt(1, 0), fold=b),
      crease_utils.Crease(_pt(1, 0), _pt(1, 1), fold=b),
      crease_utils.Crease(_pt(1, 1), _pt(0, 1), fold=b),
      crease_utils.Crease(_pt(0, 1), _pt(0, 0), fold=b),
  ]


def _make_horizontal_tail_cp(grid_size: int = 8) -> cp_utils.CreasePattern:
  """Flat-foldable CP with left and right pleated tails at T-junctions."""
  # Vertical spine pleats at x = 0.25 (M), 0.5 (V), 0.75 (M), plus horizontal
  # tail pleats in x in [0, 0.25] (left tail) and x in [0.75, 1.0] (right tail)
  # with flat-foldable degree-4 T-vertices (3 of one fold, 1 of the opposite).
  creases = _border_creases()
  ys = [0.25, 0.5, 0.75]
  stops = [0.0, *ys, 1.0]
  for seg_idx, (a, b) in enumerate(itertools.pairwise(stops)):
    spine_fold = _V if seg_idx % 2 == 0 else _M
    creases.append(
        crease_utils.Crease(_pt(0.5, a), _pt(0.5, b), fold=spine_fold)
    )
    creases.append(crease_utils.Crease(_pt(0.25, a), _pt(0.25, b), fold=_M))
    creases.append(crease_utils.Crease(_pt(0.75, a), _pt(0.75, b), fold=_M))
  for idx, y in enumerate(ys):
    outer = _M if idx % 2 == 0 else _V
    inner = _V if outer == _M else _M
    # Left tail border crease + compensating interior pleat to x=0.5.
    creases.append(crease_utils.Crease(_pt(0.0, y), _pt(0.25, y), fold=outer))
    creases.append(crease_utils.Crease(_pt(0.25, y), _pt(0.5, y), fold=inner))
    # Right tail border crease + compensating interior pleat to x=0.5.
    creases.append(crease_utils.Crease(_pt(0.75, y), _pt(1.0, y), fold=outer))
    creases.append(crease_utils.Crease(_pt(0.5, y), _pt(0.75, y), fold=inner))
  return cp_utils.CreasePattern(creases=creases, grid_size=grid_size)


def _make_vertical_tail_cp(grid_size: int = 8) -> cp_utils.CreasePattern:
  """90-degree transpose of `_make_horizontal_tail_cp` with top/bottom tails."""
  horiz = _make_horizontal_tail_cp(grid_size)
  swapped = [
      crease_utils.Crease(
          _pt(c.p1_init.y, c.p1_init.x),
          _pt(c.p2_init.y, c.p2_init.x),
          fold=c.fold,
      )
      for c in horiz.creases
  ]
  return cp_utils.CreasePattern(creases=swapped, grid_size=grid_size)


_LEFT_TAIL = shaping_utils.Rectangle(
    name='left_tail', x_start=0.0, y_start=0.0, x_len=0.25, y_len=1.0
)
_RIGHT_TAIL = shaping_utils.Rectangle(
    name='right_tail', x_start=0.75, y_start=0.0, x_len=0.25, y_len=1.0
)
_TOP_HALF = shaping_utils.Rectangle(
    name='top_half', x_start=0.0, y_start=0.0, x_len=1.0, y_len=0.25
)
_BOTTOM_HALF = shaping_utils.Rectangle(
    name='bottom_half', x_start=0.0, y_start=0.75, x_len=1.0, y_len=0.25
)


class FanOutShapingTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='right_tail_horizontal',
          vertical=False,
          rects=(_RIGHT_TAIL,),
          orientation=fan_out_shaping.Orientation.HORIZONTAL,
          delta=-1.0,
          expected_tilted=(
              ((0.625, 0.0), (0.75, 0.25), _M),
              ((0.75, 0.25), (1.0, 0.125), _M),
              ((0.75, 0.75), (0.875, 1.0), _V),
              ((0.75, 0.75), (1.0, 0.625), _V),
          ),
          forbidden_shifted=(
              ((0.75, 0.25), (1.0, 0.375), _M),
              ((0.75, 0.75), (1.0, 0.875), _M),
              ((0.75, 0.75), (1.0, 0.875), _V),
          ),
      ),
      dict(
          testcase_name='left_tail_horizontal',
          vertical=False,
          rects=(_LEFT_TAIL,),
          orientation=fan_out_shaping.Orientation.HORIZONTAL,
          delta=1.0,
          expected_tilted=(
              ((0.0, 0.375), (0.25, 0.25), _V),
              ((0.0, 0.875), (0.25, 0.75), _M),
              ((0.125, 0.0), (0.25, 0.25), _V),
              ((0.25, 0.75), (0.375, 1.0), _M),
          ),
          forbidden_shifted=(
              ((0.0, 0.125), (0.25, 0.25), _M),
              ((0.0, 0.125), (0.25, 0.25), _V),
              ((0.0, 0.625), (0.25, 0.75), _M),
          ),
      ),
      dict(
          testcase_name='both_tails_horizontal',
          vertical=False,
          rects=(_LEFT_TAIL, _RIGHT_TAIL),
          orientation=fan_out_shaping.Orientation.HORIZONTAL,
          delta=-1.0,
          expected_tilted=(
              ((0.0, 0.125), (0.25, 0.25), _M),
              ((0.0, 0.625), (0.25, 0.75), _M),
              ((0.25, 0.25), (0.5, 0.125), _V),
              ((0.25, 0.75), (0.5, 0.625), _V),
              ((0.5, 0.125), (0.75, 0.25), _V),
              ((0.5, 0.625), (0.75, 0.75), _V),
              ((0.75, 0.25), (1.0, 0.125), _M),
              ((0.75, 0.75), (1.0, 0.625), _M),
          ),
          forbidden_shifted=(
              ((0.0, 0.375), (0.25, 0.25), _M),
              ((0.75, 0.25), (1.0, 0.375), _M),
          ),
      ),
      dict(
          testcase_name='both_halves_vertical',
          vertical=True,
          rects=(_TOP_HALF, _BOTTOM_HALF),
          orientation=fan_out_shaping.Orientation.VERTICAL,
          delta=1.0,
          expected_tilted=(
              ((0.25, 0.25), (0.375, 0.0), _M),
              ((0.25, 0.25), (0.5, 0.375), _M),
              ((0.5, 0.375), (0.75, 0.25), _M),
              ((0.75, 0.25), (1.0, 0.375), _M),
          ),
          forbidden_shifted=(
              ((0.25, 0.25), (0.125, 0.0), _M),
              ((0.25, 0.75), (0.125, 1.0), _M),
          ),
      ),
  )
  def test_fan_out_shaping_preserves_flat_foldability(
      self,
      vertical: bool,
      rects: tuple[shaping_utils.Rectangle, ...],
      orientation: fan_out_shaping.Orientation,
      delta: float,
      expected_tilted: tuple[
          tuple[tuple[float, float], tuple[float, float], crease_utils.Fold],
          ...,
      ],
      forbidden_shifted: tuple[
          tuple[tuple[float, float], tuple[float, float], crease_utils.Fold],
          ...,
      ],
  ):
    cp = _make_vertical_tail_cp(8) if vertical else _make_horizontal_tail_cp(8)
    self.assertTrue(cp.local_flat_foldability())
    params = [
        fan_out_shaping.FanOutShapingParams(
            valid_fold_types=[_M],
            valid_orientations=[orientation],
            rectangles=list(rects),
            move_delta=delta,
        )
    ]
    shaped = fan_out_shaping.shape_cp_with_fan_out(
        cp, shaping_params=params, grid_size=8
    )
    self.assertTrue(shaped.local_flat_foldability())
    # Verify exact tilted crease set, direction of `move_delta`, and M/V parity.
    tilted = [
        c
        for c in shaped.creases
        if abs(c.p1_init.x - c.p2_init.x) > 1e-6
        and abs(c.p1_init.y - c.p2_init.y) > 1e-6
    ]
    tilted_keys = {
        (
            min(
                (round(float(c.p1.x), 5), round(float(c.p1.y), 5)),
                (round(float(c.p2.x), 5), round(float(c.p2.y), 5)),
            ),
            max(
                (round(float(c.p1.x), 5), round(float(c.p1.y), 5)),
                (round(float(c.p2.x), 5), round(float(c.p2.y), 5)),
            ),
            c.fold,
        )
        for c in tilted
    }
    self.assertEqual(
        tilted_keys,
        {(min(p1, p2), max(p1, p2), fold) for p1, p2, fold in expected_tilted},
    )
    for p1, p2, fold in forbidden_shifted:
      key = (min(p1, p2), max(p1, p2), fold)
      self.assertNotIn(key, tilted_keys)
    self.assertLess(bp.fold_3d(shaped).vertex_error, 1e-4)


class FanOutPlanTest(parameterized.TestCase):

  def test_find_and_resolve_plan_modes(self):
    cp = _make_horizontal_tail_cp(8)
    params = [
        fan_out_shaping.FanOutShapingParams(
            valid_fold_types=[_M],
            valid_orientations=[fan_out_shaping.Orientation.HORIZONTAL],
            rectangles=[_RIGHT_TAIL],
            move_delta=-1.0,
        )
    ]
    plan = fan_out_shaping.find_fan_out_plan(cp, params, grid_size=8)
    self.assertIsNotNone(plan)
    self.assertLen(plan.modified_creases, 2)
    self.assertNotEmpty(plan.unmodified_creases)
    self.assertEqual(plan.modified_rectangles, frozenset({_RIGHT_TAIL}))
    expected_avg = sum(c.length for c in plan.modified_creases) / len(
        plan.modified_creases
    )
    self.assertAlmostEqual(plan.avg_modified_crease_length, expected_avg)

    for mode in (
        fan_out_shaping.CandidateMode.STRICT,
        fan_out_shaping.CandidateMode.SPLIT,
        fan_out_shaping.CandidateMode.BORDER_TIP,
    ):
      mode_plan = fan_out_shaping.find_fan_out_plan(
          cp, params, grid_size=8, mode=mode
      )
      self.assertIsNotNone(mode_plan)
      self.assertLen(mode_plan.modified_creases, 2)

    resolved = fan_out_shaping.resolve_fan_out_plan(cp, plan)
    self.assertIsNotNone(resolved)
    self.assertTrue(resolved.local_flat_foldability())

  def test_returns_none_when_no_valid_creases(self):
    cp = _make_horizontal_tail_cp(8)
    empty = shaping_utils.Rectangle(
        name='empty', x_start=0.99, y_start=0.99, x_len=0.001, y_len=0.001
    )
    params = [
        fan_out_shaping.FanOutShapingParams(
            valid_fold_types=[_M],
            valid_orientations=[fan_out_shaping.Orientation.HORIZONTAL],
            rectangles=[empty],
            move_delta=-1.0,
        )
    ]
    self.assertIsNone(fan_out_shaping.find_fan_out_plan(cp, params, 8))


class FanOutHeuristicTest(parameterized.TestCase):

  def test_rejects_prepopulated_fold_types_or_orientations(self):
    cp = _make_horizontal_tail_cp(8)
    with self.assertRaises(ValueError):
      fan_out_shaping.shape_cp_with_fan_out_heuristic(
          cp,
          shaping_params=[
              fan_out_shaping.FanOutShapingParams(
                  valid_fold_types=[_M],
                  valid_orientations=[],
                  rectangles=[_RIGHT_TAIL],
                  move_delta=-1.0,
              )
          ],
      )
    with self.assertRaises(ValueError):
      fan_out_shaping.shape_cp_with_fan_out_heuristic(
          cp,
          shaping_params=[
              fan_out_shaping.FanOutShapingParams(
                  valid_fold_types=[],
                  valid_orientations=[fan_out_shaping.Orientation.HORIZONTAL],
                  rectangles=[_RIGHT_TAIL],
                  move_delta=-1.0,
              )
          ],
      )

  def test_heuristic_success_and_list_circle_centers(self):
    cp = _make_horizontal_tail_cp(8)
    rect_with_list = dataclasses.replace(
        _RIGHT_TAIL, circle_centers=[_pt(0.5, 0.5)]
    )
    params = [
        fan_out_shaping.FanOutShapingParams(
            valid_fold_types=[],
            valid_orientations=[],
            rectangles=[rect_with_list],
            move_delta=-1.0,
        )
    ]
    shaped = fan_out_shaping.shape_cp_with_fan_out_heuristic(
        cp, shaping_params=params
    )
    self.assertTrue(shaped.local_flat_foldability())
    # With move_delta=-1.0, right-border endpoints shift downward by 1/8.
    tilted = [
        c
        for c in shaped.creases
        if abs(c.p1.x - c.p2.x) > 1e-6 and abs(c.p1.y - c.p2.y) > 1e-6
    ]
    self.assertNotEmpty(tilted)
    border_ys: list[float] = []
    for c in tilted:
      for pt in (c.p1, c.p2):
        if abs(pt.x - 1.0) < 1e-5:
          border_ys.append(round(pt.y, 5))
    border_ys.sort()
    self.assertTrue(
        all(y in (0.125, 0.375, 0.625) for y in border_ys),
        msg=f'Unexpected shifted right-border y coords: {border_ys}',
    )

  def test_heuristic_raises_for_impossible_params(self):
    cp = _make_horizontal_tail_cp(8)
    empty = shaping_utils.Rectangle(
        name='empty', x_start=0.99, y_start=0.99, x_len=0.001, y_len=0.001
    )
    with self.assertRaises(ValueError):
      fan_out_shaping.shape_cp_with_fan_out_heuristic(
          cp,
          shaping_params=[
              fan_out_shaping.FanOutShapingParams(
                  valid_fold_types=[],
                  valid_orientations=[],
                  rectangles=[empty],
                  move_delta=-1.0,
              )
          ],
      )

  def test_splits_merged_border_pleat_in_light_mode(self):
    creases = _border_creases() + [
        crease_utils.Crease(_pt(0.5, 0.0), _pt(0.5, 0.25), fold=_V),
        crease_utils.Crease(_pt(0.0, 0.25), _pt(0.5, 0.25), fold=_M),
        crease_utils.Crease(_pt(0.5, 0.25), _pt(1.0, 0.25), fold=_M),
        crease_utils.Crease(_pt(0.5, 0.25), _pt(0.5, 1.0), fold=_M),
        crease_utils.Crease(_pt(0.0, 0.5), _pt(0.5, 0.5), fold=_M),
        crease_utils.Crease(_pt(0.5, 0.5), _pt(1.0, 0.5), fold=_V),
    ]
    cp = cp_utils.CreasePattern(creases=creases, light_mode=True, grid_size=4)
    bottom_half = shaping_utils.Rectangle(
        name='bottom', x_start=0.25, y_start=0.5, x_len=0.5, y_len=0.5
    )
    shaped = fan_out_shaping.shape_cp_with_fan_out(
        cp,
        shaping_params=[
            fan_out_shaping.FanOutShapingParams(
                valid_fold_types=[_M],
                valid_orientations=[fan_out_shaping.Orientation.VERTICAL],
                rectangles=[bottom_half],
                move_delta=1.0,
            )
        ],
        grid_size=4,
    )
    self.assertTrue(shaped.local_flat_foldability())
    self.assertTrue(shaped.light_mode)
    # The merged vertical crease (0.5, 0.25)->(0.5, 1.0) splits at (0.5, 0.5)
    # and shifts its bottom border tip right by +1/4 to (0.75, 1.0).
    tilted_m = [
        c
        for c in shaped.creases
        if c.fold == _M
        and abs(c.p1.x - c.p2.x) > 1e-6
        and abs(c.p1.y - c.p2.y) > 1e-6
    ]
    self.assertTrue(
        any(
            (c.p1.isclose(_pt(0.5, 0.5)) and c.p2.isclose(_pt(0.75, 1.0)))
            or (c.p2.isclose(_pt(0.5, 0.5)) and c.p1.isclose(_pt(0.75, 1.0)))
            for c in tilted_m
        ),
        msg=f'Expected shifted (0.5, 0.5)->(0.75, 1.0) crease, got {tilted_m}',
    )


if __name__ == '__main__':
  absltest.main()
