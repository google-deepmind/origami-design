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
"""Tests for flap-narrowing crease pattern shaping."""

import math

from absl.testing import absltest
from absl.testing import parameterized
import bp_pipeline as bp
from bp_pipeline.crease_pattern import crease as crease_utils
from bp_pipeline.crease_pattern import crease_pattern as cp_utils
from bp_pipeline.crease_pattern import point as point_utils
from bp_pipeline.shaping import narrow_flap_shaping
from bp_pipeline.shaping import shaping_elements
from bp_pipeline.shaping import shaping_utils
from bp_pipeline.shaping import zigzag_shaping
import numpy as np

_M = crease_utils.Fold.MOUNTAIN
_V = crease_utils.Fold.VALLEY


def _pt(x: float, y: float, *, on_paper: bool = True) -> point_utils.Point:
  return point_utils.Point(x=x, y=y, require_on_paper=on_paper)


def _make_edge_flap_cp(
    grid_size: int = 8,
    *,
    bottom: bool = False,
) -> tuple[cp_utils.CreasePattern, shaping_utils.Rectangle]:
  """Builds a flat-foldable edge flap of radius `grid_size // 2` at x=0.5."""
  b = crease_utils.Fold.BORDER
  creases = [
      crease_utils.Crease(_pt(0, 0), _pt(1, 0), fold=b),
      crease_utils.Crease(_pt(1, 0), _pt(1, 1), fold=b),
      crease_utils.Crease(_pt(1, 1), _pt(0, 1), fold=b),
      crease_utils.Crease(_pt(0, 1), _pt(0, 0), fold=b),
  ]
  y_tip = 1.0 if bottom else 0.0
  y_opp = 0.0 if bottom else 1.0
  # Center spine pleat.
  creases.append(crease_utils.Crease(_pt(0.5, y_tip), _pt(0.5, y_opp), fold=_V))
  # Alternating 45-degree ridges from (0.5, y_tip) to (0.0, 0.5) and (1.0, 0.5)
  # so every degree-4 ridge vertex satisfies Kawasaki, Maekawa, and crimp BLB.
  half = grid_size // 2
  for i in range(half):
    d0 = i / grid_size
    d1 = (i + 1) / grid_size
    y0 = (1.0 - d0) if bottom else d0
    y1 = (1.0 - d1) if bottom else d1
    ridge_fold = _V if i % 2 == 0 else _M
    creases.append(
        crease_utils.Crease(
            _pt(0.5 - d0, y0), _pt(0.5 - d1, y1), fold=ridge_fold
        )
    )
    creases.append(
        crease_utils.Crease(
            _pt(0.5 + d0, y0), _pt(0.5 + d1, y1), fold=ridge_fold
        )
    )
  for i in range(1, half):
    d = i / grid_size
    pleat_fold = _V if i % 2 == 1 else _M
    yr = (1.0 - d) if bottom else d
    # Left & right vertical pleats into the lower/upper body.
    creases.append(
        crease_utils.Crease(
            _pt(0.5 - d, yr), _pt(0.5 - d, y_opp), fold=pleat_fold
        )
    )
    creases.append(
        crease_utils.Crease(
            _pt(0.5 + d, yr), _pt(0.5 + d, y_opp), fold=pleat_fold
        )
    )
    # Horizontal pleats outward from the left/right ridges to the side borders.
    creases.append(
        crease_utils.Crease(_pt(0.0, yr), _pt(0.5 - d, yr), fold=pleat_fold)
    )
    creases.append(
        crease_utils.Crease(_pt(0.5 + d, yr), _pt(1.0, yr), fold=pleat_fold)
    )

  cp = cp_utils.CreasePattern(creases=creases, grid_size=grid_size)
  y_start = 0.5 if bottom else -0.5
  rect = shaping_utils.Rectangle(
      name='edge_flap',
      x_start=0.0,
      y_start=y_start,
      x_len=1.0,
      y_len=1.0,
      circle_centers=(_pt(0.5, y_tip, on_paper=False),),
  )
  return cp, rect


def _make_corner_flap_cp(
    grid_size: int = 8,
) -> tuple[cp_utils.CreasePattern, shaping_utils.Rectangle]:
  """Builds a flat-foldable corner flap centered at `(0, 0)` of radius 0.5."""
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
    ridge_fold = _M if i % 2 == 0 else _V
    creases.append(
        crease_utils.Crease(_pt(t0, t0), _pt(t1, t1), fold=ridge_fold)
    )
  for i in range(1, grid_size):
    t = i / grid_size
    fold = _V if i % 2 == 1 else _M
    creases.append(crease_utils.Crease(_pt(t, 0.0), _pt(t, t), fold=fold))
    creases.append(crease_utils.Crease(_pt(0.0, t), _pt(t, t), fold=fold))
  cp = cp_utils.CreasePattern(creases=creases, grid_size=grid_size)
  rect = shaping_utils.Rectangle(
      name='corner_flap',
      x_start=-0.5,
      y_start=-0.5,
      x_len=1.0,
      y_len=1.0,
      circle_centers=(_pt(0.0, 0.0, on_paper=False),),
  )
  return cp, rect


class NarrowFlapShapingTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='45_deg_zigzag_top',
          zigzag_depth=1.0,
          bottom=False,
          even_phase=True,
          expected_counts=(116, 40, 56, 20, 40),
      ),
      dict(
          testcase_name='45_deg_zigzag_bottom',
          zigzag_depth=1.0,
          bottom=True,
          even_phase=False,
          expected_counts=(119, 46, 49, 24, 44),
      ),
      dict(
          testcase_name='26p56_deg_zigzag_top',
          zigzag_depth=0.5,
          bottom=False,
          even_phase=True,
          expected_counts=(136, 46, 70, 20, 64),
      ),
      dict(
          testcase_name='22p5_deg_zigzag_top',
          zigzag_depth=math.tan(math.pi / 8.0),
          bottom=False,
          even_phase=True,
          expected_counts=(136, 46, 70, 20, 64),
      ),
      dict(
          testcase_name='56p31_deg_zigzag_top',
          zigzag_depth=1.5,
          bottom=False,
          even_phase=True,
          expected_counts=(118, 40, 58, 20, 58),
      ),
      dict(
          testcase_name='56p31_deg_zigzag_bottom',
          zigzag_depth=1.5,
          bottom=True,
          even_phase=False,
          expected_counts=(119, 46, 49, 24, 56),
      ),
  )
  def test_narrow_edge_flap_across_zigzag_slopes(
      self,
      zigzag_depth: float,
      bottom: bool,
      even_phase: bool,
      expected_counts: tuple[int, int, int, int, int],
  ):
    grid_size = 8
    base_cp, rect = _make_edge_flap_cp(grid_size, bottom=bottom)
    self.assertTrue(base_cp.local_flat_foldability())
    flap = shaping_elements.ShapingFlap.from_legacy_rectangle_direct(
        rect, grid_size
    )
    fold_order = (_V, _M) if even_phase else (_M, _V)
    zz_creases = zigzag_shaping.shape_flap_with_zigzags(
        rect=flap,
        grid_size=grid_size,
        target_flap_angle=None,
        target_zigzag_depth=zigzag_depth,
        fold_order=fold_order,
        even_phase=even_phase,
        fold_percent=1.0,
        snap_to_grid=False,
    )
    cp_with_zz = base_cp.update_with_shaping_creases(zz_creases)
    self.assertTrue(cp_with_zz.local_flat_foldability())

    narrowed = narrow_flap_shaping.narrow(
        cp_with_zz,
        zz_creases,
        [rect],
        grid_size=grid_size,
        raise_error_on_ff_violation=True,
    )
    self.assertTrue(narrowed.local_flat_foldability())
    counts = (
        len(narrowed.creases),
        sum(1 for c in narrowed.creases if c.fold == _M),
        sum(1 for c in narrowed.creases if c.fold == _V),
        sum(1 for c in narrowed.creases if c.fold == crease_utils.Fold.BORDER),
        sum(
            1
            for c in narrowed.creases
            if abs(c.p1.x - c.p2.x) > 1e-6 and abs(c.p1.y - c.p2.y) > 1e-6
        ),
    )
    self.assertEqual(counts, expected_counts)
    # Verify exact rigid 3D folding and strict reduction of folded flap width.
    model_zz = bp.fold_3d(cp_with_zz)
    model_nar = bp.fold_3d(narrowed)
    self.assertLess(model_zz.vertex_error, 1e-4)
    self.assertLess(model_nar.vertex_error, 1e-4)
    xs = np.linspace(0.05, 0.95, 37)
    ys = np.full_like(xs, 0.85 if bottom else 0.15)
    pts_zz = model_zz.locate(np.column_stack([xs, ys]))
    pts_nar = model_nar.locate(np.column_stack([xs, ys]))
    span_zz = float(
        np.max(np.linalg.norm(pts_zz[:, None, :] - pts_zz[None, :, :], axis=-1))
    )
    span_nar = float(
        np.max(
            np.linalg.norm(pts_nar[:, None, :] - pts_nar[None, :, :], axis=-1)
        )
    )
    self.assertLess(span_nar, span_zz - 0.01)

  def test_narrow_multi_depth_zigzags_on_flap(self):
    grid_size = 16
    base_cp, rect = _make_edge_flap_cp(grid_size, bottom=False)
    self.assertTrue(base_cp.local_flat_foldability())
    flap = shaping_elements.ShapingFlap.from_legacy_rectangle_direct(
        rect, grid_size
    )
    inscribed_rect = shaping_utils.Rectangle(
        name='edge_flap_inner',
        x_start=rect.x_start + 2 / grid_size,
        y_start=rect.y_start + 2 / grid_size,
        x_len=rect.x_len - 4 / grid_size,
        y_len=rect.y_len - 4 / grid_size,
        circle_centers=rect.circle_centers,
    )
    inner_flap = shaping_elements.ShapingFlap.from_legacy_rectangle_direct(
        inscribed_rect, grid_size
    )
    zz_outer = zigzag_shaping.shape_flap_with_zigzags(
        rect=flap,
        grid_size=grid_size,
        target_flap_angle=math.pi / 2.0,
        target_zigzag_depth=None,
        fold_order=(_V, _M),
        even_phase=True,
        fold_percent=1.0,
        snap_to_grid=False,
    )
    zz_inner = zigzag_shaping.shape_flap_with_zigzags(
        rect=inner_flap,
        grid_size=grid_size,
        target_flap_angle=math.pi / 2.0,
        target_zigzag_depth=None,
        fold_order=(_M, _V),
        even_phase=True,
        fold_percent=1.0,
        snap_to_grid=False,
    )
    all_zz = [*zz_outer, *zz_inner]
    cp_with_zz = base_cp.update_with_shaping_creases(all_zz)
    self.assertTrue(cp_with_zz.local_flat_foldability())
    narrowed = narrow_flap_shaping.narrow(
        cp_with_zz,
        all_zz,
        [rect],
        grid_size=grid_size,
        raise_error_on_ff_violation=True,
    )
    self.assertTrue(narrowed.local_flat_foldability())
    self.assertLen(cp_with_zz.creases, 167)
    self.assertEqual(
        (
            len(narrowed.creases),
            sum(1 for c in narrowed.creases if c.fold == _M),
            sum(1 for c in narrowed.creases if c.fold == _V),
            sum(
                1
                for c in narrowed.creases
                if c.fold == crease_utils.Fold.BORDER
            ),
            sum(
                1
                for c in narrowed.creases
                if abs(c.p1.x - c.p2.x) > 1e-6 and abs(c.p1.y - c.p2.y) > 1e-6
            ),
        ),
        (328, 134, 154, 40, 118),
    )
    self.assertLess(bp.fold_3d(narrowed).vertex_error, 1e-4)

  def test_raises_on_flat_foldability_violation_when_requested(self):
    grid_size = 8
    base_cp, rect = _make_corner_flap_cp(grid_size)
    flap = shaping_elements.ShapingFlap.from_legacy_rectangle_direct(
        rect, grid_size
    )
    # Deliberately inverted M/V assignment violates Kawasaki/Maekawa.
    bad_zz = zigzag_shaping.shape_flap_with_zigzags(
        rect=flap,
        grid_size=grid_size,
        target_flap_angle=math.pi / 2.0,
        target_zigzag_depth=None,
        fold_order=(_M, _V),
        even_phase=True,
        fold_percent=1.0,
        snap_to_grid=False,
    )
    cp_with_bad = base_cp.update_with_shaping_creases(bad_zz)
    with self.assertRaises(ValueError):
      narrow_flap_shaping.narrow(
          cp_with_bad,
          bad_zz,
          [rect],
          grid_size=grid_size,
          raise_error_on_ff_violation=True,
      )
    relaxed = narrow_flap_shaping.narrow(
        cp_with_bad,
        bad_zz,
        [rect],
        grid_size=grid_size,
        raise_error_on_ff_violation=False,
    )
    self.assertFalse(relaxed.local_flat_foldability())
    self.assertEqual(
        (
            len(relaxed.creases),
            sum(1 for c in relaxed.creases if c.fold == _M),
            sum(1 for c in relaxed.creases if c.fold == _V),
            sum(
                1 for c in relaxed.creases if c.fold == crease_utils.Fold.BORDER
            ),
        ),
        (94, 35, 41, 18),
    )

  def test_rejects_fully_off_paper_rectangle_with_value_error(self):
    base_cp, _ = _make_corner_flap_cp(8)
    off_paper = shaping_utils.Rectangle(
        name='off',
        x_start=-2.0,
        y_start=-2.0,
        x_len=0.5,
        y_len=0.5,
        circle_centers=(_pt(-1.75, -1.75, on_paper=False),),
    )
    with self.assertRaisesRegex(ValueError, 'outside the unit paper'):
      narrow_flap_shaping.narrow(
          base_cp,
          [],
          [off_paper],
          grid_size=8,
          raise_error_on_ff_violation=True,
      )

  def test_orientation_canonical_order(self):
    self.assertEqual(
        list(narrow_flap_shaping.Orientation),
        [
            narrow_flap_shaping.Orientation.TOP,
            narrow_flap_shaping.Orientation.RIGHT,
            narrow_flap_shaping.Orientation.BOTTOM,
            narrow_flap_shaping.Orientation.LEFT,
        ],
    )

  def test_draw_narrowing_crease_does_not_rotate_at_paper_border(self):
    base_cp, rect = _make_corner_flap_cp(10)
    clipped = rect.clip(xmin=0, ymin=0, xmax=1, ymax=1)
    self.assertIsNotNone(clipped)
    res = narrow_flap_shaping.draw_narrowing_crease(
        start_point=_pt(0.05, 0.05),
        end_point=_pt(0.0, 0.0),
        rectangle_border_points={_pt(0.0, 0.0), _pt(0.5, 0.0)},
        rectangle=clipped,
        cp=base_cp,
        non_border_zigzag_creases=[
            crease_utils.Crease(_pt(0.05, 0.05), _pt(0.5, 0.5), fold=_M)
        ],
        prev_clockwise_angle_delta=10.0,
        allow_paper_border_rotation=False,
    )
    self.assertIsNotNone(res)
    intersection_pt, _, zigzag_replacements, angle_delta = res
    self.assertTrue(intersection_pt.is_border)
    self.assertEmpty(zigzag_replacements)
    self.assertIsNone(angle_delta)


if __name__ == '__main__':
  absltest.main()
