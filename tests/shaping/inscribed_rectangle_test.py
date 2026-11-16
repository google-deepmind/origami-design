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

"""Test inscribed rectangle shaping utilities."""

from bp_pipeline.crease_pattern import crease as crease_utils
from bp_pipeline.crease_pattern import point as point_utils
from shaping import shaping_test_fixtures as shaping_examples
from bp_pipeline.shaping import inscribed_rectangle_shaping
from bp_pipeline.shaping import shaping_utils
from absl.testing import absltest as googletest
from absl.testing import parameterized


class InscribedRectangleShapingTest(parameterized.TestCase):

  @parameterized.parameters([
      shaping_examples.SCORPION_INSCRIBED_TEST_EXAMPLE,
  ])
  def test_inscribed_rectangle_shaping(
      self, example: shaping_examples.BoxPleatingShapingExample
  ):
    shaped_cp = inscribed_rectangle_shaping.shape_cp_with_inscribed_elements(
        cp=example.base_example.base_cp,
        grid_size=example.base_example.grid_size,
        rectangles=example.rectangles,
        rivers=example.rivers,
        element_shaping_params=example.inscribed_rectangle_shaping_params,
        central_crease_removed=example.central_crease_removed,
    )
    self.assertTrue(example.check_inscribed_rectangle_shaping(shaped_cp))

  def test_inscribed_rectangle_shaping_creases_and_rectangles(self):
    test_example = shaping_examples.SCORPION
    assert test_example.zigzag_shaping_cp is not None
    creases, rectangles = (
        inscribed_rectangle_shaping.get_inscribed_rectangle_shaping_creases_and_rectangles(
            test_example.grid_size,
            test_example.rectangles,
            test_example.inscribed_rectangle_shaping_params,
            fold_step=test_example.zigzag_shaping_cp.num_fold_steps,
            central_crease_removed=test_example.central_crease_removed,
        )
    )

    self.assertIsInstance(creases, list)
    self.assertNotEmpty(creases)
    self.assertTrue(all(isinstance(c, crease_utils.Crease) for c in creases))
    self.assertIsInstance(rectangles, list)
    self.assertNotEmpty(rectangles)
    self.assertTrue(
        all(isinstance(r, shaping_utils.Rectangle) for r in rectangles)
    )

  @parameterized.parameters([
      {
          "rect": shaping_utils.Rectangle(
              name="center_flap",
              x_start=0,
              y_start=0,
              x_len=1,
              y_len=1,
              circle_centers=[point_utils.Point(x=0.5, y=0.5)],
          ),
          "inscribe_depths": [0, 1, 3, 4],
          "targets": [
              shaping_utils.Rectangle(
                  name="",
                  x_start=0,
                  y_start=0,
                  x_len=1,
                  y_len=1,
              ),
              shaping_utils.Rectangle(
                  name="",
                  x_start=0.125,
                  y_start=0.125,
                  x_len=1 - 0.25,
                  y_len=1 - 0.25,
              ),
              shaping_utils.Rectangle(
                  name="",
                  x_start=3 * 0.125,
                  y_start=3 * 0.125,
                  x_len=1 - 3 * 0.25,
                  y_len=1 - 3 * 0.25,
              ),
              "ValueError",
          ],
      },
      {
          "rect": shaping_utils.Rectangle(
              name="corner_flap",
              x_start=0,
              y_start=0,
              x_len=0.5,
              y_len=0.5,
              circle_centers=[point_utils.Point(x=0.0, y=0.0)],
          ),
          "inscribe_depths": [2],
          "targets": [
              shaping_utils.Rectangle(
                  name="",
                  x_start=0,
                  y_start=0,
                  x_len=0.25,
                  y_len=0.25,
              ),
          ],
      },
      {
          "rect": shaping_utils.Rectangle(
              name="corner_flap",
              x_start=0.5,
              y_start=0,
              x_len=0.5,
              y_len=0.5,
              circle_centers=[point_utils.Point(x=1.0, y=0.0)],
          ),
          "inscribe_depths": [2],
          "targets": [
              shaping_utils.Rectangle(
                  name="center_flap",
                  x_start=0.75,
                  y_start=0,
                  x_len=0.25,
                  y_len=0.25,
              ),
          ],
      },
      {
          "rect": shaping_utils.Rectangle(
              name="corner_flap",
              x_start=0.5,
              y_start=0.5,
              x_len=0.5,
              y_len=0.5,
              circle_centers=[point_utils.Point(x=1.0, y=1.0)],
          ),
          "inscribe_depths": [2],
          "targets": [
              shaping_utils.Rectangle(
                  name="center_flap",
                  x_start=0.75,
                  y_start=0.75,
                  x_len=0.25,
                  y_len=0.25,
              ),
          ],
      },
      {
          "rect": shaping_utils.Rectangle(
              name="border_flap",
              x_start=0.25,
              y_start=0,
              x_len=0.5,
              y_len=0.25,
              circle_centers=[point_utils.Point(x=0.5, y=0)],
          ),
          "inscribe_depths": [1],
          "targets": [
              shaping_utils.Rectangle(
                  name="center_flap",
                  x_start=0.25 + 1 / 8,
                  y_start=0,
                  x_len=0.5 - 2 / 8,
                  y_len=0.25 - 1 / 8,
              ),
          ],
      },
      {
          "rect": shaping_utils.Rectangle(
              name="border_flap",
              x_start=0.75,
              y_start=0.25,
              x_len=0.25,
              y_len=0.5,
              circle_centers=[point_utils.Point(x=1, y=0.5)],
          ),
          "inscribe_depths": [1],
          "targets": [
              shaping_utils.Rectangle(
                  name="center_flap",
                  x_start=0.75 + 1 / 8,
                  y_start=0.25 + 1 / 8,
                  x_len=0.25 - 1 / 8,
                  y_len=0.5 - 2 / 8,
              ),
          ],
      },
      {
          "rect": shaping_utils.Rectangle(
              name="tall_flap_on_paper",
              x_start=2 / 8,
              y_start=0,
              x_len=4 / 8,
              y_len=6 / 8,
              circle_centers=[
                  point_utils.Point(x=4 / 8, y=2 / 8),
                  point_utils.Point(x=4 / 8, y=4 / 8),
              ],
          ),
          "inscribe_depths": [0, 1, 2],
          "targets": [
              shaping_utils.Rectangle(
                  name="",
                  x_start=2 / 8,
                  y_start=0,
                  x_len=4 / 8,
                  y_len=6 / 8,
              ),
              shaping_utils.Rectangle(
                  name="",
                  x_start=3 / 8,
                  y_start=1 / 8,
                  x_len=2 / 8,
                  y_len=4 / 8,
              ),
              "ValueError",
          ],
      },
      {
          "rect": shaping_utils.Rectangle(
              name="wide_flap_on_paper",
              x_start=0,
              y_start=0,
              x_len=6 / 8,
              y_len=4 / 8,
              circle_centers=[
                  point_utils.Point(x=2 / 8, y=2 / 8),
                  point_utils.Point(x=4 / 8, y=2 / 8),
              ],
          ),
          "inscribe_depths": [0, 1, 2],
          "targets": [
              shaping_utils.Rectangle(
                  name="",
                  x_start=0,
                  y_start=0,
                  x_len=6 / 8,
                  y_len=4 / 8,
              ),
              shaping_utils.Rectangle(
                  name="",
                  x_start=1 / 8,
                  y_start=1 / 8,
                  x_len=4 / 8,
                  y_len=2 / 8,
              ),
              "ValueError",
          ],
      },
      {
          "rect": shaping_utils.Rectangle(
              name="tall_flap_off_paper",
              x_start=2 / 8,
              y_start=-3 / 8,
              x_len=4 / 8,
              y_len=6 / 8,
              circle_centers=[
                  point_utils.Point(x=4 / 8, y=-1 / 8, require_on_paper=False),
                  point_utils.Point(x=4 / 8, y=1 / 8),
              ],
          ),
          "inscribe_depths": [0, 1, 2],
          "targets": [
              shaping_utils.Rectangle(
                  name="",
                  x_start=2 / 8,
                  y_start=-3 / 8,
                  x_len=4 / 8,
                  y_len=6 / 8,
              ),
              shaping_utils.Rectangle(
                  name="",
                  x_start=3 / 8,
                  y_start=-2 / 8,
                  x_len=2 / 8,
                  y_len=4 / 8,
              ),
              "ValueError",
          ],
      },
  ])
  def test_get_inscribed_rectangle(
      self,
      rect,
      inscribe_depths,
      targets,
  ):
    grid_size = 8
    corners = rect.corners
    for inscribe_depth, target in zip(inscribe_depths, targets):
      if isinstance(target, shaping_utils.Rectangle):
        pred = inscribed_rectangle_shaping.get_inscribed_rectangle(
            corners,
            rect.circle_centers,
            grid_size,
            inscribe_depth,
        )
        self.assertEqual(pred.corners, target.corners)
      elif target == "ValueError":
        with self.assertRaises(ValueError):
          _ = inscribed_rectangle_shaping.get_inscribed_rectangle(
              corners,
              rect.circle_centers,
              grid_size,
              inscribe_depth,
          )
      else:
        raise ValueError(f"Unknown target: {target}")

  def test_find_start_point(self):
    grid_size = 8
    rect = shaping_utils.Rectangle(
        name="test_rect",
        x_start=0.25,
        y_start=0.25,
        x_len=0.5,
        y_len=0.5,
    )
    corners, edges, _, _ = shaping_utils.preprocess_rect(
        rect, grid_size, None, 1.0
    )
    start_index = inscribed_rectangle_shaping.find_start_point(corners, edges)
    self.assertEqual(start_index, 0)

    rect = shaping_utils.Rectangle(
        name="test_rect",
        x_start=0,
        y_start=0,
        x_len=0.5,
        y_len=0.5,
    )
    corners, edges, _, _ = shaping_utils.preprocess_rect(
        rect, grid_size, None, 1.0
    )
    start_index = inscribed_rectangle_shaping.find_start_point(corners, edges)
    self.assertEqual(start_index, 1)

    rect = shaping_utils.Rectangle(
        name="test_rect",
        x_start=0.5,
        y_start=0,
        x_len=0.5,
        y_len=0.5,
    )
    corners, edges, _, _ = shaping_utils.preprocess_rect(
        rect, grid_size, None, 1.0
    )
    start_index = inscribed_rectangle_shaping.find_start_point(corners, edges)
    self.assertEqual(start_index, 2)

    rect = shaping_utils.Rectangle(
        name="test_rect",
        x_start=0.5,
        y_start=0.5,
        x_len=0.5,
        y_len=0.5,
    )
    corners, edges, _, _ = shaping_utils.preprocess_rect(
        rect, grid_size, None, 1.0
    )
    start_index = inscribed_rectangle_shaping.find_start_point(corners, edges)
    self.assertEqual(start_index, 3)

    rect = shaping_utils.Rectangle(
        name="test_rect",
        x_start=0,
        y_start=0.5,
        x_len=0.5,
        y_len=0.5,
    )
    corners, edges, _, _ = shaping_utils.preprocess_rect(
        rect, grid_size, None, 1.0
    )
    start_index = inscribed_rectangle_shaping.find_start_point(corners, edges)
    self.assertEqual(start_index, 0)

    rect = shaping_utils.Rectangle(
        name="test_rect",
        x_start=0.25,
        y_start=0,
        x_len=0.5,
        y_len=0.5,
    )
    corners, edges, _, _ = shaping_utils.preprocess_rect(
        rect, grid_size, None, 1.0
    )
    start_index = inscribed_rectangle_shaping.find_start_point(corners, edges)
    self.assertEqual(start_index, 1)

    rect = shaping_utils.Rectangle(
        name="test_rect",
        x_start=0.5,
        y_start=0.25,
        x_len=0.5,
        y_len=0.5,
    )
    corners, edges, _, _ = shaping_utils.preprocess_rect(
        rect, grid_size, None, 1.0
    )
    start_index = inscribed_rectangle_shaping.find_start_point(corners, edges)
    self.assertEqual(start_index, 2)

    rect = shaping_utils.Rectangle(
        name="test_rect",
        x_start=0.25,
        y_start=0.5,
        x_len=0.5,
        y_len=0.5,
    )
    corners, edges, _, _ = shaping_utils.preprocess_rect(
        rect, grid_size, None, 1.0
    )
    start_index = inscribed_rectangle_shaping.find_start_point(corners, edges)
    self.assertEqual(start_index, 3)

    rect = shaping_utils.Rectangle(
        name="test_rect",
        x_start=0,
        y_start=0.25,
        x_len=0.5,
        y_len=0.5,
    )
    corners, edges, _, _ = shaping_utils.preprocess_rect(
        rect, grid_size, None, 1.0
    )
    start_index = inscribed_rectangle_shaping.find_start_point(corners, edges)
    self.assertEqual(start_index, 0)

  def test_get_inscribed_rectangle_creases_vertical_edge_on_central_line(self):
    grid_size = 8
    rect = shaping_utils.Rectangle(
        name="right_of_center",
        x_start=0.5,
        y_start=0.0,
        x_len=0.5,
        y_len=0.5,
    )
    corners, edges, _, _ = shaping_utils.preprocess_rect(
        rect, grid_size, None, 1.0
    )
    start_index = inscribed_rectangle_shaping.find_start_point(corners, edges)
    creases = inscribed_rectangle_shaping.get_inscribed_rectangle_creases(
        grid_size=grid_size,
        start_index=start_index,
        corners=corners,
        fold_order=(crease_utils.Fold.MOUNTAIN, crease_utils.Fold.VALLEY),
        fold_percent=0.5,
        central_crease_removed=True,
        fold_step=0,
    )
    # 4 steps along y=0.5 (x=1.0->0.5) + 4 steps along x=0.5 (y=0.5->0.0) = 8.
    self.assertLen(creases, 8)
    self.assertTrue(
        all(c.is_horizontal or c.is_vertical for c in creases),
        msg=f"Expected axis-aligned creases, got: {creases}",
    )

  def test_get_inscribed_rectangle_three_circle_centers(self):
    grid_size = 8
    rect = shaping_utils.Rectangle(
        name="two_centers",
        x_start=0.0,
        y_start=0.0,
        x_len=0.75,
        y_len=0.5,
    )
    three_centers = [
        point_utils.Point(x=2 / 8, y=2 / 8),
        point_utils.Point(x=3 / 8, y=2 / 8),
        point_utils.Point(x=4 / 8, y=2 / 8),
    ]
    inscribed = inscribed_rectangle_shaping.get_inscribed_rectangle(
        rect.corners, three_centers, grid_size, inscribe_depth=1
    )
    self.assertAlmostEqual(inscribed.x_start, 1 / 8)
    self.assertAlmostEqual(inscribed.y_start, 1 / 8)


if __name__ == "__main__":
  googletest.main()
