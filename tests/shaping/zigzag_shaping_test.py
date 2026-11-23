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

"""Tests for zigzag shaping, inscribed elements, types, and 2D heuristics."""

import copy
import math

from absl.testing import absltest
from absl.testing import parameterized
import bp_pipeline as bp
from bp_pipeline.crease_pattern import crease as crease_utils
from bp_pipeline.crease_pattern import crease_pattern as cp_utils
from bp_pipeline.crease_pattern import point as point_utils
from bp_pipeline.shaping import bus_adapter
from bp_pipeline.shaping import heuristics
from bp_pipeline.shaping import heuristics_utils
from bp_pipeline.shaping import shaping_elements
from bp_pipeline.shaping import shaping_utils
from bp_pipeline.shaping import types as shaping_types
from bp_pipeline.shaping import zigzag_shaping
from bp_pipeline.stick_figures.examples import example_trees

_MV = shaping_types.MOUNTAIN_TO_VALLEY
_VM = shaping_types.VALLEY_TO_MOUNTAIN


def _border_cp(grid_size: int = 8) -> cp_utils.CreasePattern:
  b = crease_utils.Fold.BORDER
  p = lambda x, y: point_utils.Point(x=x, y=y)
  return cp_utils.CreasePattern(
      creases=[
          crease_utils.Crease(p(0.0, 0.0), p(1.0, 0.0), fold=b),
          crease_utils.Crease(p(1.0, 0.0), p(1.0, 1.0), fold=b),
          crease_utils.Crease(p(1.0, 1.0), p(0.0, 1.0), fold=b),
          crease_utils.Crease(p(0.0, 1.0), p(0.0, 0.0), fold=b),
      ],
      grid_size=grid_size,
  )


class ShapingTypesTest(parameterized.TestCase):

  def test_zigzag_and_inscribed_params_validation(self):
    zz = shaping_types.ZigzagParams(
        target_flap_angle=math.pi / 4,
        fold_order=[crease_utils.Fold.MOUNTAIN, crease_utils.Fold.VALLEY],
        even_phase=True,
    )
    self.assertEqual(zz.fold_order, _MV)

    with self.assertRaises(ValueError):
      shaping_types.ZigzagParams(
          fold_order=_MV,
          even_phase=True,
      )
    with self.assertRaises(ValueError):
      shaping_types.ZigzagParams(
          target_flap_angle=math.pi / 4,
          target_zigzag_depth=1.0,
          fold_order=_MV,
          even_phase=True,
      )
    with self.assertRaises(ValueError):
      shaping_types.ZigzagParams(
          target_flap_angle=math.pi / 4,
          fold_order=(),
          even_phase=True,
      )

    ins = shaping_types.InscribedParams(
        depth=1.0,
        fold_order=[crease_utils.Fold.VALLEY, crease_utils.Fold.MOUNTAIN],
        bank_index='FIRST',
    )
    self.assertEqual(ins.fold_order, _VM)
    self.assertEqual(ins.bank_index, 'FIRST')
    with self.assertRaises(ValueError):
      shaping_types.InscribedParams(depth=1.0, fold_order=())

    self.assertEqual(shaping_types.flip_fold_order(_MV), _VM)
    with self.assertRaises(ValueError):
      shaping_types.flip_fold_order((crease_utils.Fold.MOUNTAIN,))


class ZigzagAndInscribedShapingTest(parameterized.TestCase):

  def test_flap_and_river_zigzag_shaping_and_inscribe(self):
    grid_size = 8
    flap = shaping_elements.ShapingFlap.from_legacy_rectangle_direct(
        shaping_utils.Rectangle(
            name='corner_flap',
            x_start=-0.5,
            y_start=-0.5,
            x_len=1.0,
            y_len=1.0,
            circle_centers=(
                point_utils.Point(x=0.0, y=0.0, require_on_paper=False),
            ),
        ),
        grid_size=grid_size,
    )
    river = shaping_elements.ShapingRiver(
        name='mid_river',
        bank_1=[
            crease_utils.Crease(
                point_utils.Point(x=0.0, y=0.5),
                point_utils.Point(x=1.0, y=0.5),
                fold=crease_utils.Fold.HINGE,
            )
        ],
        bank_2=[
            crease_utils.Crease(
                point_utils.Point(x=0.0, y=0.75),
                point_utils.Point(x=1.0, y=0.75),
                fold=crease_utils.Fold.HINGE,
            )
        ],
        width=2,
        grid_size=grid_size,
        _outer_bank_index='FIRST',
    )
    params = [
        shaping_types.ElementShapingParams(
            element_name='corner_flap',
            params=shaping_types.ZigzagParams(
                target_flap_angle=math.pi / 4,
                fold_order=_MV,
                even_phase=True,
                depth=1.0,
            ),
        ),
        shaping_types.ElementShapingParams(
            element_name='mid_river',
            params=shaping_types.ZigzagParams(
                target_zigzag_depth=1.0,
                fold_order=_VM,
                even_phase=False,
                bank_index='FIRST',
            ),
        ),
    ]
    creases = zigzag_shaping.get_zigzag_shaping_creases(
        grid_size=grid_size,
        rectangles=[flap],
        rivers=[river],
        element_shaping_params=params,
        snap_to_grid=True,
    )
    m_fold = crease_utils.Fold.MOUNTAIN
    v_fold = crease_utils.Fold.VALLEY
    expected_creases = {
        ((0.0, 0.375), (0.10983, 0.10983), v_fold),
        ((0.0, 0.625), (0.125, 0.5), m_fold),
        ((0.10983, 0.10983), (0.375, 0.0), m_fold),
        ((0.125, 0.5), (0.25, 0.625), v_fold),
        ((0.16161, 0.16161), (0.25, 0.375), m_fold),
        ((0.16161, 0.16161), (0.375, 0.25), v_fold),
        ((0.25, 0.375), (0.28661, 0.28661), v_fold),
        ((0.25, 0.625), (0.375, 0.5), m_fold),
        ((0.28661, 0.28661), (0.375, 0.25), m_fold),
        ((0.375, 0.5), (0.5, 0.625), v_fold),
        ((0.5, 0.625), (0.625, 0.5), m_fold),
        ((0.625, 0.5), (0.75, 0.625), v_fold),
        ((0.75, 0.625), (0.875, 0.5), m_fold),
        ((0.875, 0.5), (1.0, 0.625), v_fold),
    }
    actual_creases = {
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
        for c in creases
    }
    self.assertLen(creases, 14)
    self.assertEqual(actual_creases, expected_creases)

    shaped = zigzag_shaping.shape_cp_with_zigzags(
        cp=_border_cp(grid_size),
        grid_size=grid_size,
        rectangles=[flap],
        rivers=[river],
        element_shaping_params=params,
    )
    self.assertEqual(
        (
            len(shaped.creases),
            sum(1 for c in shaped.creases if c.fold == m_fold),
            sum(1 for c in shaped.creases if c.fold == v_fold),
            sum(
                1 for c in shaped.creases if c.fold == crease_utils.Fold.BORDER
            ),
        ),
        (22, 7, 7, 8),
    )

  @parameterized.parameters([
      dict(
          rect=shaping_utils.Rectangle(
              name='center_flap',
              x_start=0.0,
              y_start=0.0,
              x_len=1.0,
              y_len=1.0,
              circle_centers=(point_utils.Point(x=0.5, y=0.5),),
          ),
          depths=(0, 1, 3, 4),
          targets=(
              (0.0, 0.0, 1.0, 1.0),
              (0.125, 0.125, 0.75, 0.75),
              (0.375, 0.375, 0.25, 0.25),
              None,
          ),
      ),
      dict(
          rect=shaping_utils.Rectangle(
              name='corner_tl',
              x_start=0.0,
              y_start=0.0,
              x_len=0.5,
              y_len=0.5,
              circle_centers=(point_utils.Point(x=0.0, y=0.0),),
          ),
          depths=(2,),
          targets=((0.0, 0.0, 0.25, 0.25),),
      ),
      dict(
          rect=shaping_utils.Rectangle(
              name='corner_tr',
              x_start=0.5,
              y_start=0.0,
              x_len=0.5,
              y_len=0.5,
              circle_centers=(point_utils.Point(x=1.0, y=0.0),),
          ),
          depths=(2,),
          targets=((0.75, 0.0, 0.25, 0.25),),
      ),
      dict(
          rect=shaping_utils.Rectangle(
              name='corner_br',
              x_start=0.5,
              y_start=0.5,
              x_len=0.5,
              y_len=0.5,
              circle_centers=(point_utils.Point(x=1.0, y=1.0),),
          ),
          depths=(2,),
          targets=((0.75, 0.75, 0.25, 0.25),),
      ),
      dict(
          rect=shaping_utils.Rectangle(
              name='border_top',
              x_start=0.25,
              y_start=0.0,
              x_len=0.5,
              y_len=0.25,
              circle_centers=(point_utils.Point(x=0.5, y=0.0),),
          ),
          depths=(1,),
          targets=((0.375, 0.0, 0.25, 0.125),),
      ),
      dict(
          rect=shaping_utils.Rectangle(
              name='border_right',
              x_start=0.75,
              y_start=0.25,
              x_len=0.25,
              y_len=0.5,
              circle_centers=(point_utils.Point(x=1.0, y=0.5),),
          ),
          depths=(1,),
          targets=((0.875, 0.375, 0.125, 0.25),),
      ),
      dict(
          rect=shaping_utils.Rectangle(
              name='tall_flap_on_paper',
              x_start=0.25,
              y_start=0.0,
              x_len=0.5,
              y_len=0.75,
              circle_centers=(
                  point_utils.Point(x=0.5, y=0.25),
                  point_utils.Point(x=0.5, y=0.5),
              ),
          ),
          depths=(0, 1, 2),
          targets=(
              (0.25, 0.0, 0.5, 0.75),
              (0.375, 0.125, 0.25, 0.5),
              None,
          ),
      ),
      dict(
          rect=shaping_utils.Rectangle(
              name='wide_flap_on_paper',
              x_start=0.0,
              y_start=0.0,
              x_len=0.75,
              y_len=0.5,
              circle_centers=(
                  point_utils.Point(x=0.25, y=0.25),
                  point_utils.Point(x=0.5, y=0.25),
              ),
          ),
          depths=(0, 1, 2),
          targets=(
              (0.0, 0.0, 0.75, 0.5),
              (0.125, 0.125, 0.5, 0.25),
              None,
          ),
      ),
      dict(
          rect=shaping_utils.Rectangle(
              name='tall_off_paper',
              x_start=0.25,
              y_start=-0.375,
              x_len=0.5,
              y_len=0.75,
              circle_centers=(
                  point_utils.Point(x=0.5, y=-0.125, require_on_paper=False),
                  point_utils.Point(x=0.5, y=0.125),
              ),
          ),
          depths=(0, 1, 2),
          targets=(
              (0.25, -0.375, 0.5, 0.75),
              (0.375, -0.25, 0.25, 0.5),
              None,
          ),
      ),
  ])
  def test_get_inscribed_rectangle(self, rect, depths, targets):
    for depth, target in zip(depths, targets, strict=True):
      if target is None:
        with self.assertRaises(ValueError):
          zigzag_shaping.get_inscribed_rectangle(
              rect.corners, rect.circle_centers, 8, depth
          )
      else:
        expected = shaping_utils.Rectangle(
            name='',
            x_start=target[0],
            y_start=target[1],
            x_len=target[2],
            y_len=target[3],
        )
        actual = zigzag_shaping.get_inscribed_rectangle(
            rect.corners, rect.circle_centers, 8, depth
        )
        self.assertEqual(actual.corners, expected.corners)

  @parameterized.parameters([
      (0.25, 0.25, 0),
      (0.0, 0.0, 1),
      (0.5, 0.0, 2),
      (0.5, 0.5, 3),
      (0.0, 0.5, 0),
      (0.25, 0.0, 1),
      (0.5, 0.25, 2),
      (0.25, 0.5, 3),
      (0.0, 0.25, 0),
  ])
  def test_find_inscribed_start_corner(self, x_start, y_start, expected):
    rect = shaping_utils.Rectangle(
        name='r', x_start=x_start, y_start=y_start, x_len=0.5, y_len=0.5
    )
    corners, edges, _, _ = shaping_utils.preprocess_rect(rect, 8, None, 1.0)
    self.assertEqual(
        zigzag_shaping.find_inscribed_start_corner(corners, edges), expected
    )

  def test_inscribed_centerline_bridge_three_centers_and_river_fold_step(self):
    grid_size = 8
    # Vertical edge directly on x = 0.5 stays axis-aligned into 8 segments.
    right_rect = shaping_utils.Rectangle(
        name='right', x_start=0.5, y_start=0.0, x_len=0.5, y_len=0.5
    )
    corners, edges, _, _ = shaping_utils.preprocess_rect(
        right_rect, grid_size, None, 1.0
    )
    start_idx = zigzag_shaping.find_inscribed_start_corner(corners, edges)
    edge_creases = zigzag_shaping.get_inscribed_rectangle_creases(
        grid_size=grid_size,
        start_index=start_idx,
        corners=corners,
        fold_order=_MV,
        fold_percent=0.5,
        central_crease_removed=True,
        fold_step=3,
    )
    self.assertLen(edge_creases, 8)
    self.assertTrue(all(c.is_horizontal or c.is_vertical for c in edge_creases))
    self.assertTrue(all(c.fold_step == 3 for c in edge_creases))

    # Horizontal edge crossing x = 0.5 with central crease removed bridges
    # across (0.375, 0.25) -> (0.625, 0.25) as a single 2-grid segment.
    straddle = shaping_utils.Rectangle(
        name='straddle',
        x_start=0.25,
        y_start=0.0,
        x_len=0.5,
        y_len=0.25,
        circle_centers=(point_utils.Point(x=0.5, y=0.0),),
    )
    bridged, rects = zigzag_shaping.shape_flap_with_inscribed_rectangles(
        flap=straddle,
        grid_size=grid_size,
        shaping_params=[
            shaping_types.InscribedParams(
                depth=0.0, fold_order=_MV, fold_percent=1.0
            )
        ],
        central_crease_removed=True,
    )
    self.assertLen(rects, 1)
    self.assertTrue(
        any(
            math.isclose(c.p1.y, 0.25)
            and math.isclose(c.p2.y, 0.25)
            and math.isclose(abs(c.p2.x - c.p1.x), 0.25)
            for c in bridged
        )
    )

    # Three collinear circle centers contract cleanly along their outer pair.
    three_rect = shaping_utils.Rectangle(
        name='three', x_start=0.0, y_start=0.0, x_len=0.75, y_len=0.5
    )
    three_centers = (
        point_utils.Point(x=2 / 8, y=2 / 8),
        point_utils.Point(x=3 / 8, y=2 / 8),
        point_utils.Point(x=4 / 8, y=2 / 8),
    )
    inscribed = zigzag_shaping.get_inscribed_rectangle(
        three_rect.corners, three_centers, grid_size, inscribe_depth=1
    )
    self.assertAlmostEqual(inscribed.x_start, 1 / 8)
    self.assertAlmostEqual(inscribed.y_start, 1 / 8)

    # Inscribed river propagates fold_step and shapes onto CP.
    river = shaping_elements.ShapingRiver(
        name='r1',
        bank_1=[
            crease_utils.Crease(
                point_utils.Point(x=0.0, y=0.5),
                point_utils.Point(x=1.0, y=0.5),
                fold=crease_utils.Fold.HINGE,
            )
        ],
        bank_2=[
            crease_utils.Crease(
                point_utils.Point(x=0.0, y=0.75),
                point_utils.Point(x=1.0, y=0.75),
                fold=crease_utils.Fold.HINGE,
            )
        ],
        width=2,
        grid_size=grid_size,
    )
    river_creases = zigzag_shaping.get_inscribed_shaping_creases(
        grid_size=grid_size,
        rectangles=[straddle],
        rivers=[river],
        element_shaping_params=[
            shaping_types.ElementShapingParams(
                element_name='r1',
                params=(
                    shaping_types.InscribedParams(
                        depth=1.0,
                        fold_order=_MV,
                        fold_percent=0.75,
                        bank_index='FIRST',
                    ),
                ),
            )
        ],
        fold_step=5,
    )
    m_fold = crease_utils.Fold.MOUNTAIN
    v_fold = crease_utils.Fold.VALLEY
    expected_river = [
        ((0.0, 0.625), (0.125, 0.625), m_fold, 0.75),
        ((0.125, 0.625), (0.25, 0.625), v_fold, 0.75),
        ((0.25, 0.625), (0.375, 0.625), m_fold, 0.75),
        ((0.375, 0.625), (0.625, 0.625), v_fold, 0.75),
        ((0.625, 0.625), (0.75, 0.625), m_fold, 0.75),
        ((0.75, 0.625), (0.875, 0.625), v_fold, 0.75),
        ((0.875, 0.625), (1.0, 0.625), m_fold, 0.75),
    ]
    actual_river = [
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
            round(c.fold_percent, 4),
        )
        for c in river_creases
    ]
    self.assertCountEqual(actual_river, expected_river)
    self.assertTrue(all(c.fold_step == 5 for c in river_creases))
    shaped = zigzag_shaping.shape_cp_with_inscribed_elements(
        cp=_border_cp(grid_size),
        grid_size=grid_size,
        rectangles=[straddle],
        rivers=[river],
        element_shaping_params=[
            shaping_types.ElementShapingParams(
                element_name='straddle',
                params=(
                    shaping_types.InscribedParams(
                        depth=1.0, fold_order=_MV, fold_percent=0.5
                    ),
                ),
            )
        ],
    )
    self.assertEqual(
        (
            len(shaped.creases),
            sum(1 for c in shaped.creases if c.fold == m_fold),
            sum(1 for c in shaped.creases if c.fold == v_fold),
            sum(
                1 for c in shaped.creases if c.fold == crease_utils.Fold.BORDER
            ),
        ),
        (9, 2, 1, 6),
    )


class HeuristicsUtilsTest(parameterized.TestCase):

  def test_angle_helpers_and_2d_heuristics_end_to_end(self):
    self.assertAlmostEqual(
        heuristics_utils.clip_angle_by_min_angle(0.01),
        heuristics_utils.MIN_ANGLE,
    )
    self.assertAlmostEqual(
        heuristics_utils.clip_angle_by_min_angle(-0.01),
        -heuristics_utils.MIN_ANGLE,
    )
    flipped_angles = heuristics_utils.apply_small_angle_flip(math.radians(30.0))
    self.assertLen(flipped_angles, 2)
    self.assertAlmostEqual(flipped_angles[0], heuristics_utils.LARGE_ANGLE)

    acute = shaping_types.ElementShapingParams(
        element_name='arm',
        params=shaping_types.ZigzagParams(
            target_flap_angle=math.radians(40.0),
            fold_order=_MV,
            even_phase=True,
        ),
    )
    split = heuristics_utils.split_acute_zigzag_params(acute)
    self.assertLen(split, 2)
    self.assertAlmostEqual(
        split[0].params.target_flap_angle, heuristics_utils.LARGE_ANGLE
    )
    self.assertEqual(split[1].params.fold_order, _VM)

    shifted = heuristics_utils.shift_zigzag_depth(acute, delta=1.5)
    self.assertAlmostEqual(shifted.params.depth, 1.5)

    # End-to-end 2D heuristic shaping and boundary-walk phase conversion on a
    # solved model (also verifies input elements are not mutated).
    res = bp.fold('human', preset='demo', shape=False)
    tree = copy.deepcopy(example_trees.HUMAN_SIMPLE)
    packing, solving = bus_adapter.legacy_inputs(res.solved)
    flaps, rivers = (
        shaping_elements.convert_packing_elements_to_shaping_elements(
            packing_rectangles=packing.flap_rectangles,
            packing_rivers=packing.packing_rivers,
            packing_grid_size=packing.packing_cp.grid_size,
            solving_grid_size=res.solved.grid_size,
            base_cp=res.solved.creases,
        )
    )
    phase = heuristics_utils.convert_flap_angle_to_global_phase(
        theta=math.pi / 3,
        absolute_direction=heuristics.AbsFlapDirection.DOWN,
        flap=flaps[0],
    )
    self.assertFalse(phase)
    flaps_before = copy.deepcopy(flaps)
    rivers_before = copy.deepcopy(rivers)
    shaped = heuristics_utils.shape_cp_with_heuristics(
        cp=res.solved.creases,
        stick_figure=tree,
        grid_size=res.solved.grid_size,
        shaping_flaps=list(flaps),
        shaping_rivers=list(rivers),
        central_crease_removed=res.solved.central_crease_removed,
        packing_output=packing,
        solving_output=solving,
    )
    self.assertEqual(flaps, flaps_before)
    self.assertEqual(rivers, rivers_before)
    self.assertLen(res.solved.creases.creases, 103)
    self.assertTrue(shaped.local_flat_foldability())
    self.assertEqual(
        (
            len(shaped.creases),
            sum(
                1
                for c in shaped.creases
                if c.fold == crease_utils.Fold.MOUNTAIN
            ),
            sum(
                1 for c in shaped.creases if c.fold == crease_utils.Fold.VALLEY
            ),
            sum(
                1 for c in shaped.creases if c.fold == crease_utils.Fold.BORDER
            ),
        ),
        (183, 75, 70, 38),
    )


if __name__ == '__main__':
  absltest.main()
