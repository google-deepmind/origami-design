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

import numpy as np
from bp_pipeline.shaping.clip_pattern import drawing_machine
from bp_pipeline.shaping.clip_pattern import clip_patterns
from bp_pipeline.shaping.clip_pattern import world_space
from absl.testing import absltest as googletest
from absl.testing import parameterized


class ClipPatternsTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='_flap_sym',
          length=None,
          symmetric=True,
          narrow_n_times=2,
      ),
      dict(
          testcase_name='_flap_asym',
          length=None,
          symmetric=False,
          narrow_n_times=2,
      ),
      dict(
          testcase_name='_river_sym',
          length=4,
          symmetric=True,
          narrow_n_times=2,
      ),
      dict(
          testcase_name='_river_asym',
          length=4,
          symmetric=False,
          narrow_n_times=2,
      ),
  )
  def test_narrow_leaves_canvas_untouched(
      self, length: float | None, symmetric: bool, narrow_n_times: int
  ):
    machine = drawing_machine.CreasePatternDrawingMachine()
    machine.canvas.transform(translate=(1, 2, 3))
    initial_transform = machine.canvas.peek_transform().copy()
    clip_patterns.narrow(
        machine, length=length, symmetric=symmetric, n_times=narrow_n_times
    )
    final_transform = machine.canvas.peek_transform()
    np.testing.assert_allclose(final_transform, initial_transform)

  def test_rotate_leaves_canvas_untouched(self):
    machine = drawing_machine.CreasePatternDrawingMachine()
    machine.canvas.transform(translate=(1, 2, 3))
    initial_transform = machine.canvas.peek_transform().copy()
    clip_patterns.rotate(machine, 45)
    final_transform = machine.canvas.peek_transform()
    np.testing.assert_allclose(final_transform, initial_transform)

  def test_bend_leaves_canvas_untouched(self):
    machine = drawing_machine.CreasePatternDrawingMachine()
    machine.canvas.transform(translate=(1, 2, 3))
    initial_transform = machine.canvas.peek_transform().copy()
    clip_patterns.bend(machine, 45)
    final_transform = machine.canvas.peek_transform()
    np.testing.assert_allclose(final_transform, initial_transform)

  def test_pleat_to_hide_leaves_canvas_untouched(self):
    machine = drawing_machine.CreasePatternDrawingMachine()
    machine.canvas.transform(translate=(1, 2, 3))
    initial_transform = machine.canvas.peek_transform().copy()
    clip_patterns.pleat_to_hide(machine, length=1.0, distance=0.25)
    final_transform = machine.canvas.peek_transform()
    np.testing.assert_allclose(final_transform, initial_transform)

  def test_turn_and_narrow_leaves_canvas_untouched(self):
    machine = drawing_machine.CreasePatternDrawingMachine()
    machine.canvas.transform(translate=(1, 2, 3))
    initial_transform = machine.canvas.peek_transform().copy()
    clip_patterns.turn_and_narrow(machine)
    final_transform = machine.canvas.peek_transform()
    np.testing.assert_allclose(final_transform, initial_transform)

  def test_narrow_river_symmetric_length_2(self):
    machine = drawing_machine.CreasePatternDrawingMachine()
    machine.canvas.transform(translate=(1, 2, 3))
    initial_transform = machine.canvas.peek_transform().copy()
    clip_patterns.narrow(machine, length=2.0, symmetric=True, n_times=2)
    final_transform = machine.canvas.peek_transform()
    np.testing.assert_allclose(final_transform, initial_transform)
    self.assertNotEmpty(machine.lines)

  def test_pleat_to_hide_draws_correct_number_of_lines(self):
    machine = drawing_machine.CreasePatternDrawingMachine()
    clip_patterns.pleat_to_hide(machine, length=1.0, distance=0.25)
    # length=1.0, distance=0.25 should draw lines at y=0.0, 0.25, 0.5, 0.75
    # Total 4 lines.
    self.assertLen(machine.lines, 4)

  def test_pleat_to_hide_draws_even_number_of_lines(self):
    machine = drawing_machine.CreasePatternDrawingMachine()
    # length=1.1, distance=0.25 would normally draw 5 lines, but we expect 4.
    clip_patterns.pleat_to_hide(machine, length=1.1, distance=0.25)
    self.assertLen(machine.lines, 4)

  def test_pleat_to_hide_shifts_to_avoid_borders(self):
    machine = drawing_machine.CreasePatternDrawingMachine()
    clip_patterns.pleat_to_hide(machine, length=1.0, distance=0.25)
    # We expect lines at 0.125, 0.375, 0.625, 0.875
    expected_y = [0.125, 0.375, 0.625, 0.875]
    self.assertLen(machine.lines, 4)
    for line, y in zip(machine.lines, expected_y):
      self.assertAlmostEqual(line.p1[1], y)
      self.assertAlmostEqual(line.p2[1], y)

  def test_rotate_matches_angle(self):
    for theta_deg in range(0, 181, 5):
      machine = drawing_machine.CreasePatternDrawingMachine()
      clip_patterns.rotate(machine, theta_deg)
      world = world_space.World()
      for line in machine.lines:
        rotate_axis = line.p1[:3] - line.p2[:3]
        world.transform(rotate_axis=rotate_axis, rotate_deg=180)
      world.transform(translate=(0, 1, 0))
      final_axis = world.peek_point_at_origin()[:3]
      # Angle between (0,1,0) and same vector after rotations over creases is
      # expected to be theta_deg by the crease construction.
      angle_between = np.degrees(
          np.arccos(
              np.dot(final_axis, (0, 1, 0))
              / np.linalg.norm(final_axis)
              / np.linalg.norm((0, 1, 0))
          )
      )
      self.assertAlmostEqual(angle_between, theta_deg)


if __name__ == '__main__':
  googletest.main()
