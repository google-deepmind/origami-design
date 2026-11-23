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
"""Tests for simple-fold, reverse-fold, crimp-fold, and combined heuristics."""

import copy
import math
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import bp_pipeline as bp
from bp_pipeline.crease_pattern import crease as crease_utils
from bp_pipeline.crease_pattern import crease_pattern as cp_utils
from bp_pipeline.crease_pattern import point as point_utils
from bp_pipeline.shaping import bus_adapter
from bp_pipeline.shaping import combined_heuristics
from bp_pipeline.shaping import fold_shaping
from bp_pipeline.shaping import geometric_fold_3d_heuristics
from bp_pipeline.shaping import heuristics
from bp_pipeline.shaping import heuristics_3d
from bp_pipeline.shaping import heuristics_utils
from bp_pipeline.shaping import narrow_flap_shaping
from bp_pipeline.shaping import shaping_elements
from bp_pipeline.shaping import shaping_utils
from bp_pipeline.shaping import types as shaping_types
from bp_pipeline.shaping import zigzag_shaping
import numpy as np

_M = crease_utils.Fold.MOUNTAIN
_V = crease_utils.Fold.VALLEY
_MV = (_M, _V)
_VM = (_V, _M)


def _pt(x: float, y: float, *, on_paper: bool = True) -> point_utils.Point:
  return point_utils.Point(x=x, y=y, require_on_paper=on_paper)


def _border_creases() -> list[crease_utils.Crease]:
  b = crease_utils.Fold.BORDER
  return [
      crease_utils.Crease(_pt(0, 0), _pt(1, 0), fold=b),
      crease_utils.Crease(_pt(1, 0), _pt(1, 1), fold=b),
      crease_utils.Crease(_pt(1, 1), _pt(0, 1), fold=b),
      crease_utils.Crease(_pt(0, 1), _pt(0, 0), fold=b),
  ]


def _make_grid_flap_cp(
    grid_size: int = 8,
    *,
    vertical_pleats: bool = True,
    diagonal_from_origin: bool = False,
) -> cp_utils.CreasePattern:
  """Constructs an alternating-pleat flat-foldable grid CP on `[0, 1]^2`."""
  creases = _border_creases()
  for i in range(1, grid_size):
    t = i / grid_size
    fold = _M if i % 2 == 1 else _V
    if vertical_pleats:
      creases.append(crease_utils.Crease(_pt(t, 0.0), _pt(t, 1.0), fold=fold))
    else:
      creases.append(crease_utils.Crease(_pt(0.0, t), _pt(1.0, t), fold=fold))
  if diagonal_from_origin:
    creases.append(crease_utils.Crease(_pt(0.0, 0.0), _pt(0.5, 0.5), fold=_M))
  return cp_utils.CreasePattern(creases=creases, grid_size=grid_size)


def _make_corner_flap(
    name: str = 'flap',
    *,
    x_center: float = 0.0,
    y_center: float = 0.0,
    radius_units: int = 4,
    grid_size: int = 8,
) -> shaping_elements.ShapingFlap:
  r = radius_units / grid_size
  return shaping_elements.ShapingFlap(
      name=name,
      x_start=x_center - r,
      y_start=y_center - r,
      x_len=2.0 * r,
      y_len=2.0 * r,
      _circle_centers=[_pt(x_center, y_center, on_paper=False)],
      grid_size=grid_size,
  )


def _make_horizontal_river(
    name: str = 'river',
    *,
    y_top: float = 0.5,
    width_units: int = 2,
    grid_size: int = 8,
) -> shaping_elements.ShapingRiver:
  y_bot = y_top + width_units / grid_size
  return shaping_elements.ShapingRiver(
      name=name,
      bank_1=[
          crease_utils.Crease(
              _pt(0.0, y_top), _pt(1.0, y_top), fold=crease_utils.Fold.HINGE
          )
      ],
      bank_2=[
          crease_utils.Crease(
              _pt(0.0, y_bot), _pt(1.0, y_bot), fold=crease_utils.Fold.HINGE
          )
      ],
      width=width_units,
      grid_size=grid_size,
      _outer_bank_index='FIRST',
  )


class SimpleFoldShapingTest(parameterized.TestCase):

  def test_dispatches_zigzag_and_pi_inscribed_requests(self):
    grid_size = 8
    cp = _make_grid_flap_cp(grid_size)
    flap = _make_corner_flap('arm', radius_units=4, grid_size=grid_size)
    river = _make_horizontal_river(
        'waist', y_top=0.5, width_units=2, grid_size=grid_size
    )
    requests = [
        shaping_types.ElementShapingParams(
            element_name='arm',
            params=shaping_types.ZigzagParams(
                target_flap_angle=math.pi,
                depth=1,
                fold_order=_MV,
                even_phase=True,
                fold_percent=0.5,
            ),
        ),
        shaping_types.ElementShapingParams(
            element_name='arm',
            params=shaping_types.ZigzagParams(
                target_flap_angle=math.pi / 2.0,
                depth=2,
                fold_order=_MV,
                even_phase=True,
                fold_percent=1.0,
            ),
        ),
        shaping_types.ElementShapingParams(
            element_name='waist',
            params=shaping_types.ZigzagParams(
                target_flap_angle=math.pi,
                depth=1,
                fold_order=_VM,
                even_phase=True,
                fold_percent=0.5,
                bank_index='FIRST',
            ),
        ),
    ]
    shaped = fold_shaping.shape_cp_with_simple_folds(
        cp=cp,
        grid_size=grid_size,
        flaps=[flap],
        rivers=[river],
        element_shaping_params=requests,
        central_crease_removed=False,
        light_mode=False,
    )
    self.assertEqual(
        (
            len(shaped.creases),
            sum(1 for c in shaped.creases if c.fold == _M),
            sum(1 for c in shaped.creases if c.fold == _V),
            sum(
                1 for c in shaped.creases if c.fold == crease_utils.Fold.BORDER
            ),
            sum(1 for c in shaped.creases if math.isclose(c.fold_percent, 0.5)),
            sum(1 for c in shaped.creases if math.isclose(c.fold_percent, 1.0)),
            sum(
                1
                for c in shaped.creases
                if max(c.p1.x, c.p2.x) <= 0.5 + 1e-6
                and max(c.p1.y, c.p2.y) <= 0.5 + 1e-6
                and c.fold != crease_utils.Fold.BORDER
            ),
            sum(
                1
                for c in shaped.creases
                if min(c.p1.y, c.p2.y) >= 0.5 - 1e-6
                and max(c.p1.y, c.p2.y) <= 0.75 + 1e-6
                and c.fold != crease_utils.Fold.BORDER
            ),
        ),
        (53, 17, 14, 22, 11, 42, 9, 8),
    )

  def test_simple_fold_with_narrowing_produces_flat_foldable_cp(self):
    grid_size = 8
    half = grid_size // 2
    creases = [
        *_border_creases(),
        crease_utils.Crease(_pt(0.5, 0.0), _pt(0.5, 1.0), fold=_V),
    ]
    for i in range(half):
      d0 = i / grid_size
      d1 = (i + 1) / grid_size
      ridge_fold = _V if i % 2 == 0 else _M
      creases.append(
          crease_utils.Crease(
              _pt(0.5 - d0, d0), _pt(0.5 - d1, d1), fold=ridge_fold
          )
      )
      creases.append(
          crease_utils.Crease(
              _pt(0.5 + d0, d0), _pt(0.5 + d1, d1), fold=ridge_fold
          )
      )
    for i in range(1, half):
      d = i / grid_size
      pleat_fold = _V if i % 2 == 1 else _M
      creases.append(
          crease_utils.Crease(
              _pt(0.5 - d, d), _pt(0.5 - d, 1.0), fold=pleat_fold
          )
      )
      creases.append(
          crease_utils.Crease(
              _pt(0.5 + d, d), _pt(0.5 + d, 1.0), fold=pleat_fold
          )
      )
      creases.append(
          crease_utils.Crease(_pt(0.0, d), _pt(0.5 - d, d), fold=pleat_fold)
      )
      creases.append(
          crease_utils.Crease(_pt(0.5 + d, d), _pt(1.0, d), fold=pleat_fold)
      )
    cp = cp_utils.CreasePattern(creases=creases, grid_size=grid_size)
    self.assertTrue(cp.local_flat_foldability())
    flap = shaping_elements.ShapingFlap.from_legacy_rectangle_direct(
        shaping_utils.Rectangle(
            name='top_flap',
            x_start=0.0,
            y_start=-0.5,
            x_len=1.0,
            y_len=1.0,
            circle_centers=(_pt(0.5, 0.0, on_paper=False),),
        ),
        grid_size=grid_size,
    )
    req = shaping_types.ElementShapingParams(
        element_name='top_flap',
        params=shaping_types.ZigzagParams(
            target_flap_angle=math.pi / 2.0,
            depth=0,
            fold_order=_VM,
            even_phase=True,
            fold_percent=1.0,
        ),
    )
    unnarrowed = fold_shaping.shape_cp_with_simple_folds(
        cp=cp,
        grid_size=grid_size,
        flaps=[flap],
        rivers=[],
        element_shaping_params=[req],
        central_crease_removed=False,
        light_mode=False,
        narrow=False,
    )
    narrowed = fold_shaping.shape_cp_with_simple_folds(
        cp=cp,
        grid_size=grid_size,
        flaps=[flap],
        rivers=[],
        element_shaping_params=[req],
        central_crease_removed=False,
        light_mode=False,
        narrow=True,
    )
    self.assertTrue(unnarrowed.local_flat_foldability())
    self.assertTrue(narrowed.local_flat_foldability())
    self.assertEqual(
        (
            len(unnarrowed.creases),
            sum(1 for c in unnarrowed.creases if c.fold == _M),
            sum(1 for c in unnarrowed.creases if c.fold == _V),
            sum(
                1
                for c in unnarrowed.creases
                if c.fold == crease_utils.Fold.BORDER
            ),
        ),
        (60, 16, 24, 20),
    )
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
        ),
        (116, 40, 56, 20),
    )
    model_un = bp.fold_3d(unnarrowed)
    model_nar = bp.fold_3d(narrowed)
    self.assertLess(model_un.vertex_error, 1e-4)
    self.assertLess(model_nar.vertex_error, 1e-4)
    xs = np.linspace(0.05, 0.95, 37)
    ys = np.full_like(xs, 0.15)
    pts_un = model_un.locate(np.column_stack([xs, ys]))
    pts_nar = model_nar.locate(np.column_stack([xs, ys]))
    span_un = float(
        np.max(np.linalg.norm(pts_un[:, None, :] - pts_un[None, :, :], axis=-1))
    )
    span_nar = float(
        np.max(
            np.linalg.norm(pts_nar[:, None, :] - pts_nar[None, :, :], axis=-1)
        )
    )
    self.assertLess(span_nar, span_un - 0.01)

  def test_infer_fold_order_phase_flap_river_pi_and_negative(self):
    grid_size = 16
    flap = shaping_elements.ShapingFlap.from_centers(
        name='test_flap',
        circle_centers=[_pt(0.25, 0.25, on_paper=False)],
        radius=math.sqrt(2.0) * 0.25,
        grid_size=grid_size,
    )
    angle = math.radians(45.0)
    flap_creases = zigzag_shaping.shape_flap_with_zigzags(
        rect=flap,
        grid_size=grid_size,
        target_flap_angle=angle,
        target_zigzag_depth=None,
        fold_order=_MV,
        even_phase=True,
        fold_percent=1.0,
        snap_to_grid=False,
    )
    inferred_flap = fold_shaping.infer_fold_order_phase_from_creases(
        target_creases=flap_creases,
        element=flap,
        target_flap_angle=angle,
        inscribe_depth=0.0,
        fold_percent=1.0,
        central_crease_removed=False,
    )
    self.assertIsNotNone(inferred_flap)
    self.assertTrue(inferred_flap.params.even_phase)
    self.assertEqual(inferred_flap.params.fold_order, _MV)

    # 180-degree inscribed rectangle inference.
    ins_creases, _ = zigzag_shaping.shape_flap_with_inscribed_rectangles(
        flap=flap,
        grid_size=grid_size,
        shaping_params=[
            shaping_types.InscribedParams(
                depth=1, fold_order=_VM, fold_percent=0.5
            )
        ],
        central_crease_removed=False,
    )
    inferred_ins = fold_shaping.infer_fold_order_phase_from_creases(
        target_creases=ins_creases,
        element=flap,
        target_flap_angle=math.pi,
        inscribe_depth=1.0,
        fold_percent=0.5,
        central_crease_removed=False,
    )
    self.assertIsNotNone(inferred_ins)
    self.assertEqual(inferred_ins.params.fold_order, _VM)

    # River zigzag inference.
    river = _make_horizontal_river(
        'test_river', y_top=0.0, width_units=8, grid_size=grid_size
    )
    river_creases = zigzag_shaping.shape_river_with_zigzags(
        river=river,
        grid_size=grid_size,
        target_flap_angle=angle,
        target_zigzag_depth=None,
        fold_order=_MV,
        even_phase=True,
        fold_percent=1.0,
        bank_index='FIRST',
        snap_to_grid=False,
    )
    inferred_river = fold_shaping.infer_fold_order_phase_from_creases(
        target_creases=river_creases,
        element=river,
        target_flap_angle=angle,
        inscribe_depth=0.0,
        fold_percent=1.0,
        central_crease_removed=False,
    )
    self.assertIsNotNone(inferred_river)
    self.assertTrue(inferred_river.params.even_phase)
    self.assertEqual(inferred_river.params.fold_order, _MV)
    self.assertEqual(inferred_river.params.bank_index, 'FIRST')

    # Non-matching creases return None.
    bogus = [crease_utils.Crease(_pt(0.25, 0.25), _pt(0.5, 0.5), fold=_M)]
    self.assertIsNone(
        fold_shaping.infer_fold_order_phase_from_creases(
            target_creases=bogus,
            element=flap,
            target_flap_angle=angle,
            inscribe_depth=0.0,
            fold_percent=1.0,
            central_crease_removed=False,
        )
    )

  def test_apply_heuristics_to_shaping_params_splits_small_angle(self):
    grid_size = 16
    flap = _make_corner_flap('tip', radius_units=6, grid_size=grid_size)
    hinge_seg = [
        crease_utils.Crease(_pt(0.0, 6 / 16), _pt(1 / 16, 6 / 16), fold=_M),
        crease_utils.Crease(_pt(1 / 16, 6 / 16), _pt(2 / 16, 6 / 16), fold=_V),
    ]
    raw = [
        shaping_types.ElementShapingParams(
            element_name='tip',
            params=shaping_types.ZigzagParams(
                target_flap_angle=heuristics_utils.MIN_ANGLE / 2.0,
                depth=0,
                fold_order=_MV,
                even_phase=True,
                fold_percent=1.0,
            ),
        )
    ]
    resolved = fold_shaping.apply_heuristics_to_shaping_params(
        grid_size=grid_size,
        flaps=[flap],
        rivers=[],
        element_orientations={
            'tip': heuristics.AbsFlapDirection.DOWN,
        },
        hinge_segments=hinge_seg,
        element_shaping_params=raw,
        use_heuristics_for_fold_order=True,
        use_heuristics_for_depth=True,
        use_heuristics_for_phase=False,
        convert_small_angles=True,
    )
    self.assertLen(resolved, 2)
    self.assertAlmostEqual(
        resolved[0].params.target_flap_angle, heuristics_utils.LARGE_ANGLE
    )
    self.assertNotEqual(
        resolved[0].params.fold_order, resolved[1].params.fold_order
    )


class ReverseAndCrimpFoldTest(parameterized.TestCase):

  def test_make_reverse_fold_params_accepts_mutable_list_and_caches(self):
    p1 = fold_shaping.make_reverse_fold_params(
        flap_angle=0.5, even_phase=True, fold_order=[_M, _V]
    )
    p2 = fold_shaping.make_reverse_fold_params(
        flap_angle=0.5, even_phase=True, fold_order=(_M, _V)
    )
    self.assertEqual(p1, p2)
    self.assertEqual(p1.fold_order, _MV)

  @parameterized.named_parameters(
      dict(
          testcase_name='vertical_bottom_flap',
          x_center=0.5,
          y_center=0.0,
          vertical_pleats=True,
          orientation='vertical',
          fold_type=_M,
          expected_counts=(48, 18, 10, 20),
          expected_oblique=(
              ((0.0, 0.0), (0.125, 0.125), _V),
              ((0.0, 0.25), (0.125, 0.125), _V),
              ((0.0, 0.25), (0.125, 0.375), _M),
              ((0.125, 0.375), (0.25, 0.5), _V),
              ((0.25, 0.5), (0.375, 0.375), _M),
              ((0.375, 0.375), (0.5, 0.5), _M),
              ((0.5, 0.5), (0.625, 0.375), _M),
              ((0.625, 0.375), (0.75, 0.5), _V),
              ((0.75, 0.5), (0.875, 0.375), _M),
              ((0.875, 0.125), (1.0, 0.0), _M),
              ((0.875, 0.125), (1.0, 0.25), _M),
              ((0.875, 0.375), (1.0, 0.25), _V),
          ),
      ),
      dict(
          testcase_name='horizontal_right_flap',
          x_center=0.0,
          y_center=0.5,
          vertical_pleats=False,
          orientation='horizontal',
          fold_type=_V,
          expected_counts=(48, 16, 12, 20),
          expected_oblique=(
              ((0.0, 0.0), (0.125, 0.125), _M),
              ((0.0, 1.0), (0.125, 0.875), _V),
              ((0.125, 0.125), (0.25, 0.0), _V),
              ((0.125, 0.875), (0.25, 1.0), _M),
              ((0.25, 0.0), (0.375, 0.125), _M),
              ((0.25, 1.0), (0.375, 0.875), _V),
              ((0.375, 0.125), (0.5, 0.25), _V),
              ((0.375, 0.375), (0.5, 0.25), _M),
              ((0.375, 0.375), (0.5, 0.5), _V),
              ((0.375, 0.625), (0.5, 0.5), _V),
              ((0.375, 0.625), (0.5, 0.75), _V),
              ((0.375, 0.875), (0.5, 0.75), _M),
          ),
      ),
      dict(
          testcase_name='horizontal_left_flap',
          x_center=1.0,
          y_center=0.5,
          vertical_pleats=False,
          orientation='horizontal',
          fold_type=_M,
          expected_counts=(48, 18, 10, 20),
          expected_oblique=(
              ((0.5, 0.25), (0.625, 0.125), _M),
              ((0.5, 0.25), (0.625, 0.375), _V),
              ((0.5, 0.5), (0.625, 0.375), _M),
              ((0.5, 0.5), (0.625, 0.625), _M),
              ((0.5, 0.75), (0.625, 0.625), _M),
              ((0.5, 0.75), (0.625, 0.875), _V),
              ((0.625, 0.125), (0.75, 0.0), _V),
              ((0.625, 0.875), (0.75, 1.0), _M),
              ((0.75, 0.0), (0.875, 0.125), _M),
              ((0.75, 1.0), (0.875, 0.875), _V),
              ((0.875, 0.125), (1.0, 0.0), _V),
              ((0.875, 0.875), (1.0, 1.0), _M),
          ),
      ),
  )
  def test_flap_reverse_fold_orientations(
      self,
      x_center: float,
      y_center: float,
      vertical_pleats: bool,
      orientation: str,
      fold_type: crease_utils.Fold,
      expected_counts: tuple[int, int, int, int],
      expected_oblique: tuple[
          tuple[tuple[float, float], tuple[float, float], crease_utils.Fold],
          ...,
      ],
  ):
    grid_size = 8
    cp = _make_grid_flap_cp(grid_size, vertical_pleats=vertical_pleats)
    flap = _make_corner_flap(
        'beak',
        x_center=x_center,
        y_center=y_center,
        radius_units=4,
        grid_size=grid_size,
    )
    params = [
        fold_shaping.ReverseFoldShapingParams(
            element_name='beak',
            reverse_fold_params=[
                fold_shaping.ReverseFoldParams(
                    flap_angle=math.pi / 2.0,
                    fold_order=_MV,
                    even_phase=True,
                )
            ],
            orientation=orientation,
            fold_type=fold_type,
        )
    ]
    shaped = fold_shaping.shape_cp_with_reverse_fold(
        cp=cp,
        grid_size=grid_size,
        rectangles=[flap],
        element_shaping_params=params,
    )
    self.assertEqual(
        (
            len(shaped.creases),
            sum(1 for c in shaped.creases if c.fold == _M),
            sum(1 for c in shaped.creases if c.fold == _V),
            sum(
                1 for c in shaped.creases if c.fold == crease_utils.Fold.BORDER
            ),
        ),
        expected_counts,
    )
    oblique = {
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
        for c in shaped.creases
        if abs(c.p1.x - c.p2.x) > 1e-6 and abs(c.p1.y - c.p2.y) > 1e-6
    }
    self.assertEqual(oblique, set(expected_oblique))

  def test_multi_element_reverse_fold_preserves_per_element_fold_type(self):
    grid_size = 8
    cp = _make_grid_flap_cp(grid_size, vertical_pleats=True)
    left_flap = _make_corner_flap(
        'left', x_center=0.25, y_center=0.0, radius_units=2, grid_size=grid_size
    )
    right_flap = _make_corner_flap(
        'right',
        x_center=0.75,
        y_center=0.0,
        radius_units=2,
        grid_size=grid_size,
    )
    params = [
        fold_shaping.ReverseFoldShapingParams(
            element_name='left',
            reverse_fold_params=[
                fold_shaping.ReverseFoldParams(
                    flap_angle=math.pi / 2.0, fold_order=_MV, even_phase=True
                )
            ],
            orientation='vertical',
            fold_type=_M,
        ),
        fold_shaping.ReverseFoldShapingParams(
            element_name='right',
            reverse_fold_params=[
                fold_shaping.ReverseFoldParams(
                    flap_angle=math.pi / 2.0, fold_order=_MV, even_phase=True
                )
            ],
            orientation='vertical',
            fold_type=_V,
        ),
    ]
    _, instructions = fold_shaping.get_reverse_fold_creases(
        cp=cp,
        grid_size=grid_size,
        rectangles=[left_flap, right_flap],
        element_shaping_params=params,
    )
    left_types = {
        inst.fold_type
        for inst in instructions
        if inst.flap_rectangle is not None
        and inst.flap_rectangle.name == 'left'
    }
    right_types = {
        inst.fold_type
        for inst in instructions
        if inst.flap_rectangle is not None
        and inst.flap_rectangle.name == 'right'
    }
    self.assertEqual(left_types, {_M})
    self.assertEqual(right_types, {_V})

    # Combined shaping must assign each element's own fold_type on its beaks.
    assigned = fold_shaping.assign_reverse_fold_orders(instructions)
    beak_folds = [
        c.fold
        for c, inst in zip(assigned, instructions, strict=True)
        if inst.is_beak
    ]
    self.assertEqual(beak_folds, [_M, _M, _V, _V])

  def test_river_reverse_fold(self):
    grid_size = 16
    river = shaping_elements.make_dummy_river(
        name='river_1', grid_size=grid_size
    )
    cp = cp_utils.CreasePattern(creases=_border_creases(), grid_size=grid_size)
    params = fold_shaping.ReverseFoldShapingParams(
        element_name='river_1',
        reverse_fold_params=[
            fold_shaping.ReverseFoldParams(
                flap_angle=0.5,
                fold_order=_VM,
                even_phase=True,
                bank_index='FIRST',
            )
        ],
        orientation='vertical',
    )
    _, instructions = fold_shaping.get_reverse_fold_creases(
        cp=cp,
        grid_size=grid_size,
        rectangles=[],
        rivers=[river],
        element_shaping_params=[params],
    )
    self.assertNotEmpty(instructions)
    self.assertTrue(all(not inst.is_beak for inst in instructions))

  def test_crimp_fold_on_flap_and_zero_zigzag_river(self):
    grid_size = 8
    cp = _make_grid_flap_cp(grid_size, vertical_pleats=True)
    flap = _make_corner_flap(
        'leg', x_center=0.5, y_center=0.0, radius_units=4, grid_size=grid_size
    )
    flap_params = [
        fold_shaping.CrimpFoldShapingParams(
            element_name='leg',
            crimp_fold_params=[
                fold_shaping.CrimpFoldParams(
                    depth=1,
                    fold_order=_MV,
                    fold_percent=1.0,
                    zigzag_flap_angle=math.pi / 2.0,
                    zigzag_fold_order=_VM,
                    zigzag_even_phase=True,
                )
            ],
        )
    ]
    shaped = fold_shaping.shape_cp_with_crimp_fold(
        cp=cp,
        grid_size=grid_size,
        flaps=[flap],
        element_shaping_params=flap_params,
        central_crease_removed=False,
    )
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
                if abs(c.p1.x - c.p2.x) > 1e-6 and abs(c.p1.y - c.p2.y) > 1e-6
            ),
        ),
        (56, 21, 17, 18, 12),
    )

    # River crimp with zero zigzag angle accepts an empty zigzag_fold_order.
    river = _make_horizontal_river(
        'river1', y_top=0.5, width_units=2, grid_size=grid_size
    )
    river_params = [
        fold_shaping.CrimpFoldShapingParams(
            element_name='river1',
            crimp_fold_params=[
                fold_shaping.CrimpFoldParams(
                    depth=1,
                    fold_order=_MV,
                    fold_percent=1.0,
                    zigzag_flap_angle=0.0,
                    zigzag_fold_order=(),
                    zigzag_even_phase=True,
                    bank_index='FIRST',
                )
            ],
        )
    ]
    ins_creases, zz_creases = fold_shaping.get_crimp_fold_creases(
        cp=cp,
        grid_size=grid_size,
        flaps=[],
        rivers=[river],
        element_shaping_params=river_params,
        central_crease_removed=False,
    )
    self.assertEmpty(zz_creases)
    expected_river_ins = [
        ((i / 8, 0.625), ((i + 1) / 8, 0.625), _M if i % 2 == 0 else _V)
        for i in range(8)
    ]
    actual_river_ins = [
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
        for c in ins_creases
    ]
    self.assertCountEqual(actual_river_ins, expected_river_ins)


class CombinedHeuristicsTest(parameterized.TestCase):

  def test_compute_narrowed_flaps_and_fallback_without_duplicate_seeds(self):
    fold_res = bp.fold('human', preset='demo')
    solved = fold_res.solved
    packing_out, solver_res = bus_adapter.legacy_inputs(solved)
    stick_fig = copy.deepcopy(solved.packed.design.source)
    for child in stick_fig.children:
      if child.name == 'left_arm':
        child.width = 0.5

    shaping_flaps, shaping_rivers = (
        shaping_elements.convert_packing_elements_to_shaping_elements(
            packing_rectangles=packing_out.flap_rectangles,
            packing_rivers=packing_out.packing_rivers,
            packing_grid_size=packing_out.packing_cp.grid_size,
            solving_grid_size=solved.creases.grid_size,
            base_cp=solved.creases,
        )
    )
    hinges = heuristics.update_hinge_info_from_solver_result(
        packing_out.hinges, solver_res
    )
    shaping_data = heuristics_utils.get_heuristics_shaping_data(
        stick_figure=stick_fig,
        shaping_flaps=list(shaping_flaps),
        shaping_rivers=list(shaping_rivers),
        hinge_info_list=list(hinges),
        central_crease_removed=solved.central_crease_removed,
        shape_with_unmerged_tree=False,
    )
    parents, children, strategies = combined_heuristics.compute_narrowed_flaps(
        stick_figure=stick_fig,
        shaping_data=shaping_data,
        use_narrowing=True,
    )
    self.assertEqual(parents, {'left_arm'})
    self.assertEmpty(children)
    self.assertEqual(
        strategies, {'left_arm': heuristics_3d.FoldStrategy3D.ROTATE_BEND}
    )

    off_parents, off_children, off_strats = (
        combined_heuristics.compute_narrowed_flaps(
            stick_figure=stick_fig,
            shaping_data=shaping_data,
            use_narrowing=False,
        )
    )
    self.assertEmpty(off_parents)
    self.assertEmpty(off_children)
    self.assertEmpty(off_strats)

    # Exercise end-to-end combined shaping, including graceful fallback when
    # narrow() raises ValueError (must not re-insert seed creases).
    with mock.patch.object(
        narrow_flap_shaping,
        'narrow',
        side_effect=ValueError('forced narrowing failure'),
    ):
      shaped = combined_heuristics.shape_cp_with_combined_heuristics(
          base_cp=solved.creases,
          stick_figure=stick_fig,
          packing_output=packing_out,
          solving_output=solver_res,
          solving_grid_size=solved.creases.grid_size,
          central_crease_removed=solved.central_crease_removed,
          shape_with_unmerged_tree=False,
          hparams=combined_heuristics.CombinedHeuristicsHParams(
              geometric_fold_hparams=(
                  geometric_fold_3d_heuristics.GeometricFolding3DHParams(
                      check_ff_violations=True
                  )
              )
          ),
      )
    self.assertTrue(shaped.local_flat_foldability())
    unique_keys = {
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
        for c in shaped.creases
    }
    self.assertLen(solved.creases.creases, 103)
    self.assertLen(unique_keys, len(shaped.creases))
    self.assertEqual(
        (
            len(shaped.creases),
            sum(1 for c in shaped.creases if c.fold == _M),
            sum(1 for c in shaped.creases if c.fold == _V),
            sum(
                1 for c in shaped.creases if c.fold == crease_utils.Fold.BORDER
            ),
        ),
        (163, 73, 60, 30),
    )


if __name__ == '__main__':
  absltest.main()
