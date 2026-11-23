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
"""Tests for squash-sink ridge shaping."""

from absl.testing import absltest
from absl.testing import parameterized
import bp_pipeline as bp
from bp_pipeline.crease_pattern import crease as crease_utils
from bp_pipeline.crease_pattern import crease_pattern as cp_utils
from bp_pipeline.crease_pattern import point as point_utils
from bp_pipeline.shaping import shaping_utils
from bp_pipeline.shaping import squash_sink_shaping

_M = crease_utils.Fold.MOUNTAIN
_V = crease_utils.Fold.VALLEY


def _pt(x: float, y: float, *, on_paper: bool = True) -> point_utils.Point:
  return point_utils.Point(x=x, y=y, require_on_paper=on_paper)


def _make_diagonal_ridge_cp(
    grid_size: int = 8,
) -> tuple[cp_utils.CreasePattern, list[shaping_utils.Rectangle]]:
  """Flat-foldable accordion flap at `(1, 1)` with an alternating diagonal."""
  b = crease_utils.Fold.BORDER
  creases = [
      crease_utils.Crease(_pt(0, 0), _pt(1, 0), fold=b),
      crease_utils.Crease(_pt(1, 0), _pt(1, 1), fold=b),
      crease_utils.Crease(_pt(1, 1), _pt(0, 1), fold=b),
      crease_utils.Crease(_pt(0, 1), _pt(0, 0), fold=b),
  ]
  for i in range(grid_size):
    t0 = i / grid_size
    t1 = (i + 1) / grid_size
    diag_fold = _V if i % 2 == 0 else _M
    creases.append(
        crease_utils.Crease(_pt(t0, t0), _pt(t1, t1), fold=diag_fold)
    )
    if i > 0:
      side_fold = _M if i % 2 == 1 else _V
      creases.append(
          crease_utils.Crease(_pt(t0, 0.0), _pt(t0, t0), fold=side_fold)
      )
      creases.append(
          crease_utils.Crease(_pt(0.0, t0), _pt(t0, t0), fold=side_fold)
      )
  cp = cp_utils.CreasePattern(creases=creases, grid_size=grid_size)
  rect = shaping_utils.Rectangle(
      name='snake_tail',
      x_start=0.0,
      y_start=0.0,
      x_len=2.0,
      y_len=2.0,
      circle_centers=(_pt(1.0, 1.0),),
  )
  return cp, [rect]


class SquashSinkShapingTest(parameterized.TestCase):

  def test_get_bisector_point_easy_and_asymmetric(self):
    easy = squash_sink_shaping.get_bisector_point(
        _pt(0.0, 0.5), _pt(0.5, 0.25), _pt(0.5, 0.75)
    )
    self.assertTrue(easy.isclose(_pt(0.5, 0.5)))

    asym = squash_sink_shaping.get_bisector_point(
        _pt(0.5, 0.5), _pt(0.2, 0.3), _pt(0.5, 0.2)
    )
    self.assertTrue(asym.isclose(_pt(0.3637509630206024, 0.2454163456597992)))

  def test_get_side_creases(self):
    center = _pt(0.5, 0.5)
    side_1 = crease_utils.Crease(center, _pt(0.5, 0.0), fold=_V)
    side_2 = crease_utils.Crease(center, _pt(0.0, 0.5), fold=_V)
    ridge_1 = crease_utils.Crease(center, _pt(0.0, 0.0), fold=_M)
    ridge_2 = crease_utils.Crease(center, _pt(1.0, 1.0), fold=_V)
    other = crease_utils.Crease(_pt(0.0, 0.75), _pt(0.25, 1.0), fold=_M)
    cp = cp_utils.CreasePattern(
        creases=[side_1, side_2, ridge_1, ridge_2, other], grid_size=4
    )
    self.assertCountEqual(
        squash_sink_shaping.get_side_creases(center, cp, ridge_2),
        [side_1, side_2],
    )

  def test_get_adjacent_side_creases_at_flap_tip(self):
    tip = _pt(0.5, 0.5)
    right = crease_utils.Crease(tip, _pt(1.0, 0.5), fold=_V)
    down = crease_utils.Crease(tip, _pt(0.5, 1.0), fold=_V)
    left = crease_utils.Crease(tip, _pt(0.0, 0.5), fold=_V)
    up = crease_utils.Crease(tip, _pt(0.5, 0.0), fold=_V)
    other_ridge = crease_utils.Crease(tip, _pt(1.0, 0.0), fold=_V)
    adjacent = squash_sink_shaping.get_adjacent_side_creases(
        tip, [left, up, right, other_ridge, down], toward=_pt(0.75, 0.75)
    )
    self.assertCountEqual(adjacent, [right, down])
    self.assertEqual(
        squash_sink_shaping.get_adjacent_side_creases(
            tip, [right], toward=_pt(0.75, 0.75)
        ),
        [right],
    )

  def test_get_oriented_ordered_crease_segments_on_ridge(self):
    c1 = crease_utils.Crease(_pt(0.0, 0.0), _pt(0.25, 0.25), fold=_M)
    c2 = crease_utils.Crease(_pt(0.5, 0.5), _pt(0.25, 0.25), fold=_V)
    c3 = crease_utils.Crease(_pt(0.75, 0.75), _pt(0.5, 0.5), fold=_M)
    c4 = crease_utils.Crease(_pt(0.75, 0.75), _pt(1.0, 1.0), fold=_V)
    cp = cp_utils.CreasePattern(creases=[c1, c4, c3, c2], grid_size=4)
    ridge = crease_utils.Crease(_pt(0.0, 0.0), _pt(1.0, 1.0), fold=_M)
    segments = (
        squash_sink_shaping.get_oriented_ordered_crease_segments_on_ridge(
            cp, ridge, _pt(1.0, 1.0)
        )
    )
    self.assertEqual(
        [s.p1_init for s in segments],
        [_pt(1.0, 1.0), _pt(0.75, 0.75), _pt(0.5, 0.5), _pt(0.25, 0.25)],
    )

  def test_get_full_ridge_and_start_side(self):
    ridge = crease_utils.Crease(_pt(0.125, 0.125), _pt(0.75, 0.75), fold=_M)
    full_1 = crease_utils.Crease(_pt(0.0, 0.0), _pt(1.0, 1.0), fold=_M)
    full_2 = crease_utils.Crease(_pt(0.75, 0.0), _pt(1.0, 0.25), fold=_M)
    self.assertEqual(
        squash_sink_shaping.get_full_ridge(ridge, [full_1, full_2]), full_1
    )
    _, rects = _make_diagonal_ridge_cp(8)
    start = squash_sink_shaping.get_start_side_of_ridge(ridge, full_1, rects)
    self.assertTrue(start.isclose(_pt(0.125, 0.125)))

  def test_get_all_ridges_from_square_and_long_rectangles(self):
    _, square_rects = _make_diagonal_ridge_cp(8)
    sq_ridges = squash_sink_shaping.get_all_ridges_from_rectangles(square_rects)
    self.assertCountEqual(
        sq_ridges,
        [
            crease_utils.Crease(_pt(0, 0), _pt(1, 1), fold=_M),
            crease_utils.Crease(_pt(1, 1), _pt(2, 2, on_paper=False), fold=_M),
            crease_utils.Crease(_pt(1, 1), _pt(0, 2, on_paper=False), fold=_M),
            crease_utils.Crease(_pt(1, 1), _pt(2, 0, on_paper=False), fold=_M),
        ],
    )

    long_rect = shaping_utils.Rectangle(
        name='long',
        x_start=0.0,
        y_start=0.0,
        x_len=0.5,
        y_len=1.0,
        circle_centers=(_pt(0.25, 0.25), _pt(0.25, 0.75)),
    )
    long_ridges = squash_sink_shaping.get_all_ridges_from_rectangles(
        [long_rect]
    )
    self.assertCountEqual(
        long_ridges,
        [
            crease_utils.Crease(_pt(0.0, 0.0), _pt(0.25, 0.25), fold=_M),
            crease_utils.Crease(_pt(0.25, 0.25), _pt(0.5, 0.0), fold=_M),
            crease_utils.Crease(_pt(0.25, 0.75), _pt(0.0, 1.0), fold=_M),
            crease_utils.Crease(_pt(0.25, 0.75), _pt(0.5, 1.0), fold=_M),
        ],
    )

    # Rectangle without circle_centers is safely skipped instead of crashing.
    no_centers = shaping_utils.Rectangle(
        name='bare', x_start=0.0, y_start=0.0, x_len=0.5, y_len=0.5
    )
    self.assertEmpty(
        squash_sink_shaping.get_all_ridges_from_rectangles([no_centers])
    )

  def test_apply_squash_sink_to_segment(self):
    pa, pb, pc = _pt(0.5, 0.0), _pt(0.5, 0.25), _pt(0.5, 0.5)
    bs1, bs2 = _pt(0.25, 0.0), _pt(0.75, 0.0)
    cs1, cs2 = _pt(0.0, 0.0), _pt(1.0, 0.0)
    cab = crease_utils.Crease(pa, pb, fold=_V)
    cbc = crease_utils.Crease(pb, pc, fold=_M)
    b1 = crease_utils.Crease(bs1, pb, fold=_M)
    b2 = crease_utils.Crease(pb, bs2, fold=_M)
    c1 = crease_utils.Crease(pc, cs2, fold=_V)
    c2 = crease_utils.Crease(cs1, pc, fold=_V)
    cp = cp_utils.CreasePattern(creases=[cab, cbc, b1, b2, c1, c2], grid_size=4)
    ridge = crease_utils.Crease(pa, pc, fold=_M)

    new_creases, removed = squash_sink_shaping.apply_squash_sink_to_segment(
        cab, cbc, cp, [ridge]
    )
    pd = _pt(0.5, 0.07322330470336309)
    pe = _pt(0.3232233047033631, 0.07322330470336309)
    pf = _pt(0.676776695296637, 0.07322330470336304)
    expected_new = [
        crease_utils.Crease(pe, pc, fold=_M),
        crease_utils.Crease(pf, pc, fold=_M),
        crease_utils.Crease(pe, pd, fold=_M),
        crease_utils.Crease(pd, pf, fold=_M),
        crease_utils.Crease(pa, pd, fold=_M),
        crease_utils.Crease(pe, pa, fold=_V),
        crease_utils.Crease(pa, pf, fold=_V),
        crease_utils.Crease(pd, pc, fold=_V),
        crease_utils.Crease(bs1, pe, fold=_M),
        crease_utils.Crease(pf, bs2, fold=_M),
    ]
    self.assertCountEqual(new_creases, expected_new)
    self.assertCountEqual(removed, [cab, cbc, b1, b2])

  def test_squash_sink_shaping_end_to_end(self):
    cp, rects = _make_diagonal_ridge_cp(8)
    self.assertTrue(cp.local_flat_foldability())
    ridge = crease_utils.Crease(
        _pt(0.0, 0.0), _pt(0.5, 0.5), fold=crease_utils.Fold.MOUNTAIN_ZEBRA
    )
    shaped = squash_sink_shaping.shape_cp_with_squash_sink(
        cp=cp, rectangles=rects, ridge=ridge
    )
    self.assertTrue(shaped.local_flat_foldability())
    self.assertEqual(
        (
            len(shaped.creases),
            sum(1 for c in shaped.creases if c.fold == _M),
            sum(1 for c in shaped.creases if c.fold == _V),
            sum(
                1 for c in shaped.creases if c.fold == crease_utils.Fold.BORDER
            ),
            sum(
                1
                for c in shaped.creases
                if abs(c.p1.x - c.p2.x) > 1e-6
                and abs(c.p1.y - c.p2.y) > 1e-6
                and abs(abs(c.p1.x - c.p2.x) - abs(c.p1.y - c.p2.y)) > 1e-6
            ),
        ),
        (46, 16, 12, 18, 4),
    )
    crease_keys = {
        (
            min(
                (round(float(c.p1.x), 6), round(float(c.p1.y), 6)),
                (round(float(c.p2.x), 6), round(float(c.p2.y), 6)),
            ),
            max(
                (round(float(c.p1.x), 6), round(float(c.p1.y), 6)),
                (round(float(c.p2.x), 6), round(float(c.p2.y), 6)),
            ),
            c.fold,
        )
        for c in shaped.creases
    }
    # Outer untouched ridge segments along (0.5, 0.5) -> (1.0, 1.0) remain.
    for i in range(4, 8):
      t0, t1 = round(i / 8, 6), round((i + 1) / 8, 6)
      expected_fold = _V if i % 2 == 0 else _M
      self.assertIn(((t0, t0), (t1, t1), expected_fold), crease_keys)
    # The squashed pair (0.25, 0.25) -> (0.375, 0.375) -> (0.5, 0.5) was
    # replaced by the 22.5-deg bisector kite centered at (0.286612, 0.286612).
    self.assertNotIn(((0.25, 0.25), (0.375, 0.375), _V), crease_keys)
    self.assertNotIn(((0.375, 0.375), (0.5, 0.5), _M), crease_keys)
    d = round(0.25 + 0.125 * (1.0 - 1.0 / (2.0**0.5)), 6)
    e = round(0.375 - 0.125 * (2.0**0.5), 6)
    for expected_kite in (
        ((0.25, 0.25), (d, d), _M),
        ((d, d), (0.5, 0.5), _V),
        ((e, 0.375), (0.25, 0.25), _V),
        ((0.25, 0.25), (0.375, e), _V),
        ((e, 0.375), (0.5, 0.5), _M),
        ((0.375, e), (0.5, 0.5), _M),
        ((e, 0.375), (d, d), _M),
        ((d, d), (0.375, e), _M),
    ):
      self.assertIn(expected_kite, crease_keys)
    self.assertLess(bp.fold_3d(shaped).vertex_error, 1e-4)

  def test_ridge_segments_found_despite_svg_rounding_jitter(self):
    ridge = crease_utils.Crease(
        _pt(27 / 44, 41 / 44), _pt(29 / 44, 43 / 44), fold=_M
    )
    seg_1 = crease_utils.Crease(
        _pt(round(27 / 44, 8), round(41 / 44, 8)),
        _pt(round(28 / 44, 8), round(42 / 44, 8)),
        fold=_M,
    )
    seg_2 = crease_utils.Crease(
        _pt(round(28 / 44, 8), round(42 / 44, 8)),
        _pt(round(29 / 44, 8), round(43 / 44, 8)),
        fold=_V,
    )
    cp = cp_utils.CreasePattern(creases=[seg_1, seg_2], light_mode=True)
    segments = (
        squash_sink_shaping.get_oriented_ordered_crease_segments_on_ridge(
            cp, ridge, _pt(29 / 44, 43 / 44)
        )
    )
    self.assertLen(segments, 2)

  def test_border_triangle_squash_is_flat_foldable(self):
    b = crease_utils.Fold.BORDER
    creases = [
        crease_utils.Crease(_pt(0, 0), _pt(1, 0), fold=b),
        crease_utils.Crease(_pt(1, 0), _pt(1, 1), fold=b),
        crease_utils.Crease(_pt(1, 1), _pt(0, 1), fold=b),
        crease_utils.Crease(_pt(0, 1), _pt(0, 0), fold=b),
        crease_utils.Crease(_pt(0.5, 1.0), _pt(0.25, 0.75), fold=_V),
        crease_utils.Crease(_pt(0.25, 0.75), _pt(1.0, 0.75), fold=_M),
        crease_utils.Crease(_pt(0.25, 0.75), _pt(0.25, 1.0), fold=_M),
        crease_utils.Crease(_pt(0.25, 0.75), _pt(0.0, 0.5), fold=_M),
    ]
    cp = cp_utils.CreasePattern(creases=creases, light_mode=True)
    rect = shaping_utils.Rectangle(
        name='corner_flap',
        x_start=-0.25,
        y_start=0.25,
        x_len=1.0,
        y_len=1.0,
        circle_centers=(_pt(0.25, 0.75),),
    )
    ridge = crease_utils.Crease(
        _pt(0.25, 0.75), _pt(0.75, 1.25, on_paper=False), fold=_V
    )
    shaped = squash_sink_shaping.shape_cp_with_squash_sink(cp, [rect], ridge)
    self.assertTrue(shaped.local_flat_foldability())
    self.assertLess(bp.fold_3d(shaped).vertex_error, 1e-4)
    # Exact 22.5-deg bisectors land at x = 0.25*sqrt(2) and 0.5 + 0.25*sqrt(2).
    x_bisect_left = round(0.25 * (2.0**0.5), 6)
    x_bisect_right = round(0.5 + 0.25 * (2.0**0.5), 6)
    actual_keys = {
        (
            min(
                (round(float(c.p1.x), 6), round(float(c.p1.y), 6)),
                (round(float(c.p2.x), 6), round(float(c.p2.y), 6)),
            ),
            max(
                (round(float(c.p1.x), 6), round(float(c.p1.y), 6)),
                (round(float(c.p2.x), 6), round(float(c.p2.y), 6)),
            ),
            c.fold,
        )
        for c in shaped.creases
    }
    expected_keys = {
        ((0.0, 0.0), (0.0, 0.5), b),
        ((0.0, 0.0), (1.0, 0.0), b),
        ((0.0, 0.5), (0.0, 1.0), b),
        ((0.0, 0.5), (0.25, 0.75), _M),
        ((0.0, 1.0), (0.25, 1.0), b),
        ((0.25, 0.75), (0.25, 1.0), _M),
        ((0.25, 0.75), (x_bisect_left, 1.0), _V),
        ((0.25, 0.75), (0.5, 1.0), _M),
        ((0.25, 0.75), (x_bisect_right, 1.0), _V),
        ((0.25, 0.75), (1.0, 0.75), _M),
        ((0.25, 1.0), (x_bisect_left, 1.0), b),
        ((x_bisect_left, 1.0), (0.5, 1.0), b),
        ((0.5, 1.0), (x_bisect_right, 1.0), b),
        ((x_bisect_right, 1.0), (1.0, 1.0), b),
        ((1.0, 0.0), (1.0, 0.75), b),
        ((1.0, 0.75), (1.0, 1.0), b),
    }
    self.assertLen(shaped.creases, 16)
    self.assertEqual(actual_keys, expected_keys)


if __name__ == '__main__':
  absltest.main()
