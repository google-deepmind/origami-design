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
from typing import cast

from bp_pipeline.crease_pattern import crease as crease_utils
from bp_pipeline.crease_pattern import crease_pattern as cp_utils
from bp_pipeline.crease_pattern import point as point_utils
from shaping import shaping_test_fixtures as shaping_examples
from bp_pipeline.shaping import inscribed_rectangle_shaping
from bp_pipeline.shaping import shaping_elements
from bp_pipeline.shaping import shaping_utils
from bp_pipeline.shaping import simple_fold_shaping
from bp_pipeline.shaping import types as shaping_types
from bp_pipeline.shaping import zigzag_shaping
from absl.testing import absltest as googletest
from absl.testing import parameterized


class SimpleFoldTest(parameterized.TestCase):

  @parameterized.parameters([
      shaping_examples.GOALIE,
      shaping_examples.JAGUAR_HEURISTICS_TEST,
  ])
  def test_inscribed_rectangle_shaping(
      self,
      example: shaping_examples.BoxPleatingShapingExample,
  ):
    requests = []
    requests.extend(example.zigzag_shaping_params)
    if example.inscribed_rectangle_shaping_params:
      for request in example.inscribed_rectangle_shaping_params:
        request: inscribed_rectangle_shaping.InscribedRectangleFlapShapingParams
        for params in request.inscribed_rectangles_params:
          requests.append(
              zigzag_shaping.ZigzagShapingParamsForElement(
                  element_name=request.flap_name,
                  zigzag_params=shaping_types.ZigzagParams(
                      target_flap_angle=math.pi,
                      depth=params.depth,
                      fold_order=params.fold_order,
                      even_phase=True,  # Default value.
                      fold_percent=params.fold_percent,
                  ),
              )
          )

    flaps = []
    for rect in example.rectangles:
      if isinstance(rect, shaping_elements.ShapingFlap):
        flaps.append(rect)
      elif isinstance(rect, shaping_utils.Rectangle):
        flaps.append(
            shaping_elements.ShapingFlap.from_legacy_rectangle_direct(
                rect, example.base_example.grid_size
            )
        )
    rivers = example.rivers
    if rivers is None:
      rivers = []

    shaped_cp = simple_fold_shaping.shape_cp_with_simple_folds(
        cp=example.base_example.base_cp,
        flaps=flaps,
        rivers=rivers,
        grid_size=example.base_example.grid_size,
        element_shaping_params=requests,
        central_crease_removed=example.central_crease_removed,
    )
    base_cp = example.base_example.base_cp
    assert isinstance(base_cp, cp_utils.CreasePattern)
    self.assertGreater(len(shaped_cp.creases), len(base_cp.creases))

  @parameterized.parameters([
      shaping_examples.GOALIE,
  ])
  def test_narrow_shaping(
      self, example: shaping_examples.BoxPleatingShapingExample
  ):
    requests = []
    requests.extend(example.zigzag_shaping_params)

    flaps = []
    for rect in example.rectangles:
      if isinstance(rect, shaping_elements.ShapingFlap):
        flaps.append(rect)
      elif isinstance(rect, shaping_utils.Rectangle):
        flaps.append(
            shaping_elements.ShapingFlap.from_legacy_rectangle_direct(
                rect, example.base_example.grid_size
            )
        )
    rivers = example.rivers
    if rivers is None:
      rivers = []

    shaped_cp = simple_fold_shaping.shape_cp_with_simple_folds(
        cp=example.base_example.base_cp,
        flaps=flaps,
        rivers=rivers,
        grid_size=example.base_example.grid_size,
        element_shaping_params=requests,
        central_crease_removed=example.central_crease_removed,
        narrow=True,
    )

    base_cp = example.base_example.base_cp
    assert isinstance(base_cp, cp_utils.CreasePattern)
    self.assertGreater(len(shaped_cp.creases), len(base_cp.creases))

    self.assertTrue(example.check_narrow_flap_shaping(shaped_cp))

  def test_infer_fold_order_phase_from_creases(self):
    grid_size = 16
    flap = shaping_elements.ShapingFlap.from_centers(
        name='test_flap',
        circle_centers=[
            point_utils.Point(x=0.25, y=0.25, require_on_paper=False)
        ],
        radius=math.sqrt(2) * 0.25,
        grid_size=grid_size,
    )

    target_flap_angle = math.radians(45)
    fold_percent = 1.0
    inscribe_depth = 0.0

    # Generate target creases with known parameters
    target_creases = zigzag_shaping.shape_flap_with_zigzags(
        rect=flap,
        grid_size=grid_size,
        target_flap_angle=target_flap_angle,
        target_zigzag_depth=None,
        fold_order=[crease_utils.Fold.MOUNTAIN, crease_utils.Fold.VALLEY],
        even_phase=True,
        fold_percent=fold_percent,
        snap_to_grid=False,
    )

    # Test inference
    result = simple_fold_shaping.infer_fold_order_phase_from_creases(
        target_creases=target_creases,
        element=flap,
        target_flap_angle=target_flap_angle,
        inscribe_depth=inscribe_depth,
        fold_percent=fold_percent,
        central_crease_removed=False,
    )

    self.assertIsNotNone(result)
    self.assertTrue(result.zigzag_params.even_phase)
    self.assertEqual(
        result.zigzag_params.fold_order,
        [crease_utils.Fold.MOUNTAIN, crease_utils.Fold.VALLEY],
    )

  def test_infer_fold_order_phase_from_creases_river(self):
    grid_size = 16
    river = shaping_elements.ShapingRiver(
        name='test_river',
        bank_1=[
            crease_utils.Crease(
                point_utils.Point(x=0, y=0, require_on_paper=False),
                point_utils.Point(x=1.0, y=0, require_on_paper=False),
                fold=crease_utils.Fold.HINGE,
            )
        ],
        bank_2=[
            crease_utils.Crease(
                point_utils.Point(x=0, y=0.5, require_on_paper=False),
                point_utils.Point(x=1.0, y=0.5, require_on_paper=False),
                fold=crease_utils.Fold.HINGE,
            )
        ],
        width=8,
        grid_size=grid_size,
        _outer_bank_index='FIRST',
    )

    target_flap_angle = math.radians(45)
    fold_percent = 1.0
    inscribe_depth = 0.0

    # Generate target creases with known parameters
    target_creases = zigzag_shaping.shape_river_with_zigzags(
        river=river,
        grid_size=grid_size,
        target_flap_angle=target_flap_angle,
        target_zigzag_depth=None,
        fold_order=[crease_utils.Fold.MOUNTAIN, crease_utils.Fold.VALLEY],
        even_phase=True,
        fold_percent=fold_percent,
        bank_index='FIRST',
        snap_to_grid=False,
    )

    # Test inference
    result = simple_fold_shaping.infer_fold_order_phase_from_creases(
        target_creases=target_creases,
        element=river,
        target_flap_angle=target_flap_angle,
        inscribe_depth=inscribe_depth,
        fold_percent=fold_percent,
        central_crease_removed=False,
    )

    self.assertIsNotNone(result)
    self.assertTrue(result.zigzag_params.even_phase)
    self.assertEqual(
        result.zigzag_params.fold_order,
        [crease_utils.Fold.MOUNTAIN, crease_utils.Fold.VALLEY],
    )
    self.assertIsInstance(result.zigzag_params, shaping_types.RiverZigzagParams)
    zigzag_params = cast(shaping_types.RiverZigzagParams, result.zigzag_params)
    self.assertEqual(zigzag_params.bank_index, 'FIRST')

  def test_infer_fold_order_phase_from_creases_negative(self):
    grid_size = 16
    flap = shaping_elements.ShapingFlap.from_centers(
        name='test_flap',
        circle_centers=[
            point_utils.Point(x=0.25, y=0.25, require_on_paper=False)
        ],
        radius=math.sqrt(2) * 0.25,
        grid_size=grid_size,
    )

    target_flap_angle = math.radians(45)
    fold_percent = 1.0
    inscribe_depth = 0.0

    # Generate target creases with known parameters
    # Note: Using different target angle/depth so no match will be found.
    # Actually, easier to manually construct incorrect creases.
    target_creases = [
        crease_utils.Crease(
            point_utils.Point(x=0.25, y=0.25, require_on_paper=False),
            point_utils.Point(x=0.5, y=0.5, require_on_paper=False),
            fold=crease_utils.Fold.MOUNTAIN,
        )
    ]

    # Test inference
    result = simple_fold_shaping.infer_fold_order_phase_from_creases(
        target_creases=target_creases,
        element=flap,
        target_flap_angle=target_flap_angle,
        inscribe_depth=inscribe_depth,
        fold_percent=fold_percent,
        central_crease_removed=False,
    )

    self.assertIsNone(result)


if __name__ == '__main__':
  googletest.main()
