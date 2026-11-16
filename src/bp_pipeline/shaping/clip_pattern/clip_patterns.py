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

"""Clip patterns."""

import math

import numpy as np

from ...crease_pattern import crease
from . import drawing_machine
from . import clip_patterns_utils

RotationMode = clip_patterns_utils.RotationMode


def noop_step(
    machine: drawing_machine.CreasePatternDrawingMachine,
    step: float,
):
  machine.canvas.transform(translate=(0, step, 0))


def bend(
    machine: drawing_machine.CreasePatternDrawingMachine, angle: float
) -> None:
  """Draws a horizontal crease to bend a flap by angle."""
  theta_deg = clip_patterns_utils.normalize_angle(angle)
  if np.isclose(theta_deg, 0):
    return
  fold_kind = crease.Fold.MOUNTAIN if theta_deg > 0 else crease.Fold.VALLEY
  machine.draw_line((0, 0), (1, 0), fold_kind, abs(theta_deg))


def rotate(
    machine: drawing_machine.CreasePatternDrawingMachine, angle: float
) -> None:
  """Draws one or two creases to rotate a flap by angle.

  To understand the context, imagine a rectangular strip of paper directed
  upwards. Left bottom corner is at (0, 0) and the right corner is at (1, 0). We
  call it a folded flap or just flap in this context.

  Angle 0 means flap is unchanged. >0 it moves clockwise, <0 it moves
  counter-clockwise.

  A single crease at an angle `alpha/2` results in a rotation of `alpha`.

  Example: To rotate a flap by 90 degrees, a single crease is drawn at 45
  degrees. This crease runs from (0, 0) to (1, 1). For -90 it will be from (1,0)
  to (0, 1).

  This particular algorithm is designed to use crease angles that are
  less than or equal to 45 degrees, ensuring the creases take up a small,
  predictable space (1x1 square). To achieve that the algorithm uses one or two
  creases depending on the angle.
  For instance, to achieve a rotation of 30 degrees, the implementation will
  first apply a 90-degree rotation using a 45-degree crease, and then apply a
  60-degree rotation back using a 30-degree crease. The y coordinate will be
  computed like tan((90 - 30)/2) or tan(45 - alpha/2).

  Args:
    machine: The drawing machine to draw the creases with.
    angle: The angle in degrees to rotate the flap.
  """

  mode, theta_val = clip_patterns_utils.rotation_mode(angle)
  theta = np.radians(theta_val / 2.0)
  fk = crease.Fold.VALLEY
  fa = 180

  match mode:
    case RotationMode.BETWEEN_0_AND_90:
      machine.draw_line((1, 0), (0, 1), fk.flip(), fa)
      machine.draw_line((1, 0), (0, np.tan(np.radians(45) - theta)), fk, fa)
    case RotationMode.BETWEEN_90_AND_180:
      machine.draw_line((1, 0), (0, np.tan(theta)), fk, fa)
    case RotationMode.BETWEEN_NEG_90_AND_0:
      machine.draw_line((0, 0), (1, 1), fk.flip(), fa)
      machine.draw_line((0, 0), (1, np.tan(np.radians(45) - theta)), fk, fa)
    case RotationMode.BETWEEN_180_AND_NEG_90:
      machine.draw_line((0, 0), (1, np.tan(theta)), fk, fa)
    case RotationMode.FIXED_180:
      machine.draw_line((0.0, 0.05), (1.0, 0.05), fk, fa)
    case RotationMode.FIXED_0:
      pass
    case RotationMode.FIXED_90:
      machine.draw_line((1, 0), (0, 1), fk, fa)
    case RotationMode.FIXED_NEG_90:
      machine.draw_line((0, 0), (1, 1), fk, fa)


def _narrow_adapter(machine):
  machine.draw_line((0, 0), (0.5, 0.5), crease.Fold.VALLEY, 180)
  machine.draw_line((0, 1), (0.5, 0.5), crease.Fold.VALLEY, 180)
  machine.draw_line((0.5, 0.5), (1, 0.5), crease.Fold.VALLEY, 180)
  machine.draw_line((0, 1), (0.5, 1), crease.Fold.MOUNTAIN, 180)
  machine.draw_line((0.5, 1), (1, 1), crease.Fold.MOUNTAIN, 180)
  machine.draw_line((0.5, 1), (0.5, 0.5), crease.Fold.MOUNTAIN, 180)


def _narrow_flap(machine: drawing_machine.CreasePatternDrawingMachine):
  _narrow_adapter(machine)
  machine.draw_line((0.5, 100), (0.5, 1), crease.Fold.VALLEY, 180)


def _narrow_river(
    machine: drawing_machine.CreasePatternDrawingMachine, length: float
):
  _narrow_adapter(machine)
  if not np.isclose(length, 2):
    machine.draw_line((0.5, length - 1), (0.5, 1), crease.Fold.VALLEY, 180)
  machine.canvas.transform(translate=(0, length, 0))
  machine.canvas.transform(scale=(1, -1, 1))
  _narrow_adapter(machine)
  machine.canvas.pop_transformation()
  machine.canvas.pop_transformation()


def narrow(
    machine: drawing_machine.CreasePatternDrawingMachine,
    length: float | None,
    symmetric: bool = True,
    n_times: int = 2,
):
  """Draws creases to narrow a flap or river.

  For the river, pattern starts as it ends with "adapter" pattern.
  For the flap, length is automatically set to a large value with the
  expectation that the excess part will be clipped by the shaping algorithm.
  Will do nothing if length is too short (i.e. 2 or less).

  Args:
    machine: The drawing machine to draw the creases with.
    length: The length of the river. For flaps set to None.
    symmetric: Whether to draw a symmetric narrow pattern or asymmetric.
    n_times: The number of times to narrow the flap, must be integer >= 2. Width
      will be 1/n_times of the original width.
  """

  if symmetric:
    _narrow_symmetric(machine, length=length, n_times=n_times)
  else:
    if length is None:
      _narrow_flap(machine)
    else:
      if length < 2:
        return
      _narrow_river(machine, length)


def _narrow_symmetric(
    machine: drawing_machine.CreasePatternDrawingMachine,
    length: float | None,
    n_times: int,
):
  """Draws a symmetric narrow pattern.

  This function draws the creases for a symmetric narrowing pattern,
  which reduces the width of a flap or river. The pattern is composed of
  a base half and main lines, mirrored and scaled to form the full pattern.

  Args:
    machine: The drawing machine to draw the creases with.
    length: The length of the river. If None, a large default length is used for
      a flap.
    n_times: The number of times to narrow the flap, must be integer >= 2.
  """

  def draw_narrow_symmetric_base_half():
    machine.draw_line((0, 0), (1, 1), crease.Fold.MOUNTAIN, 180)
    machine.draw_line((0, 1), (1, 1), crease.Fold.VALLEY, 180)
    machine.draw_line((1, 1), (0.5, 1.5), crease.Fold.MOUNTAIN, 180)
    machine.draw_line((0.5, 1.5), (0, 2), crease.Fold.MOUNTAIN, 180)
    machine.draw_line((0, 2), (0.5, 2), crease.Fold.VALLEY, 180)
    machine.draw_line((0.5, 2), (1, 2), crease.Fold.VALLEY, 180)

  def draw_narrow_symmetric_main_lines_half(length: float, n_times: int):
    k = 1 / n_times
    for i in range(n_times - 1):
      t = k * (i + 1)
      machine.draw_line((0, 2 - t), (t, 2 - t), crease.Fold.MOUNTAIN, 180)
      machine.draw_line((t, 2 - t), (t, 2), crease.Fold.VALLEY, 180)
      machine.draw_line((t, 2), (t, 2 + length), crease.Fold.MOUNTAIN, 180)
      machine.draw_line(
          (t, 2 + length), (t, 2 + length + t), crease.Fold.VALLEY, 180
      )
      machine.draw_line(
          (t, 2 + length + t), (0, 2 + length + t), crease.Fold.MOUNTAIN, 180
      )

  if length is None:
    length = 200
  if length < 2.0 - 1e-6:
    return

  for translate_x, scale_x, scale_y in [(0, 0.5, 0.5), (1, -0.5, 0.5)]:
    machine.canvas.transform(translate=(translate_x, 0, 0))
    machine.canvas.transform(scale=(scale_x, scale_y, 1))
    draw_narrow_symmetric_base_half()
    main_lines_length = (length - 2) * 2
    draw_narrow_symmetric_main_lines_half(main_lines_length, n_times)
    machine.canvas.pop_transformation()
    machine.canvas.pop_transformation()

  machine.canvas.transform(translate=(0, length, 0))
  machine.canvas.transform(scale=(1, -1, 1))
  for translate_x, scale_x, scale_y in [(0, 0.5, 0.5), (1, -0.5, 0.5)]:
    machine.canvas.transform(translate=(translate_x, 0, 0))
    machine.canvas.transform(scale=(scale_x, scale_y, 1))
    draw_narrow_symmetric_base_half()
    machine.canvas.pop_transformation()
    machine.canvas.pop_transformation()
  machine.canvas.pop_transformation()
  machine.canvas.pop_transformation()


def turn_and_narrow(
    machine: drawing_machine.CreasePatternDrawingMachine,
    length: float | None = None,
) -> None:
  """Draws creases to turn and narrow a flap.

  Visualization of the pattern
    - http://gpaste/5346375106691072 (SVG)

  Args:
    machine: The drawing machine to draw the creases with.
    length: Optional longitudinal length in grid units. If None, defaults to 600
      (global length).

  Returns:
    None
  """

  machine.canvas.transform(scale=(1 / 6, 1 / 6, 1))
  machine.draw_line((0, 0), (3, 3), crease.Fold.MOUNTAIN, 180)
  machine.draw_line((3, 3), (6, 0), crease.Fold.MOUNTAIN, 180)
  machine.draw_line((2, 4), (3, 3), crease.Fold.MOUNTAIN, 180)
  machine.draw_line((1, 5), (2, 4), crease.Fold.MOUNTAIN, 180)
  machine.draw_line((0, 6), (1, 5), crease.Fold.MOUNTAIN, 180)
  machine.draw_line((3, 3), (4, 4), crease.Fold.MOUNTAIN, 180)
  machine.draw_line((4, 4), (5, 5), crease.Fold.MOUNTAIN, 180)
  machine.draw_line((5, 5), (6, 6), crease.Fold.MOUNTAIN, 180)
  machine.draw_line((5, 5), (6, 5), crease.Fold.MOUNTAIN, 180)
  machine.draw_line((4, 4), (4, 6), crease.Fold.MOUNTAIN, 180)
  machine.draw_line((2, 4), (2, 6), crease.Fold.MOUNTAIN, 180)
  machine.draw_line((0, 5), (1, 5), crease.Fold.MOUNTAIN, 180)
  machine.draw_line((1, 5), (1, 6), crease.Fold.VALLEY, 180)
  machine.draw_line((5, 6), (5, 5), crease.Fold.VALLEY, 180)
  machine.draw_line((4, 4), (6, 4), crease.Fold.VALLEY, 180)
  machine.draw_line((0, 4), (2, 4), crease.Fold.VALLEY, 180)
  machine.draw_line((0, 3), (3, 3), crease.Fold.VALLEY, 90)
  machine.draw_line((3, 3), (6, 3), crease.Fold.VALLEY, 90)
  machine.draw_line((0, 6), (1, 6), crease.Fold.VALLEY, 180)
  machine.draw_line((1, 6), (2, 6), crease.Fold.VALLEY, 180)
  machine.draw_line((3, 3), (3, 6), crease.Fold.VALLEY, 180)
  machine.draw_line((2, 6), (3, 6), crease.Fold.VALLEY, 180)
  machine.draw_line((3, 6), (4, 6), crease.Fold.VALLEY, 180)
  machine.draw_line((4, 6), (5, 6), crease.Fold.VALLEY, 180)
  machine.draw_line((5, 6), (6, 6), crease.Fold.VALLEY, 180)
  end_y = 600 if length is None else max(6, int(round(6 + length * 6)))
  machine.draw_line((2, 6), (2, end_y), crease.Fold.VALLEY, 180)
  machine.draw_line((4, 6), (4, end_y), crease.Fold.VALLEY, 180)
  machine.draw_line((1, 6), (1, end_y), crease.Fold.MOUNTAIN, 180)
  machine.draw_line((5, 6), (5, end_y), crease.Fold.MOUNTAIN, 180)
  machine.draw_line((3, 6), (3, end_y), crease.Fold.MOUNTAIN, 180)
  machine.canvas.pop_transformation()


def bend_when_turn_and_narrow(
    machine: drawing_machine.CreasePatternDrawingMachine,
    bend_angle: float,
):
  """Draws creases to bend already narrowed flap with turn_and_narrow."""
  if np.isclose(bend_angle, 0):
    return
  abs_bend_angle = abs(bend_angle)
  machine.canvas.transform(scale=(1 / 6, 1 / 6, 1))
  machine.canvas.transform(scale=(1, 1, np.sign(bend_angle)))
  machine.draw_line((0, 0), (1, 0), crease.Fold.MOUNTAIN, abs_bend_angle)
  machine.draw_line((2, 0), (3, 0), crease.Fold.MOUNTAIN, abs_bend_angle)
  machine.draw_line((4, 0), (5, 0), crease.Fold.MOUNTAIN, abs_bend_angle)
  machine.draw_line((1, 0), (2, 0), crease.Fold.VALLEY, abs_bend_angle)
  machine.draw_line((3, 0), (4, 0), crease.Fold.VALLEY, abs_bend_angle)
  machine.draw_line((5, 0), (6, 0), crease.Fold.VALLEY, abs_bend_angle)
  machine.canvas.pop_transformation()
  machine.canvas.pop_transformation()


def _pleat(
    machine: drawing_machine.CreasePatternDrawingMachine,
    num_lines: int,
    start_y: float,
    distance: float,
) -> None:
  """Draws a series of horizontal lines at small equal distance.

  Args:
    machine: The drawing machine to draw the creases with.
    num_lines: The number of lines to draw.
    start_y: The starting y coordinate.
    distance: The distance between lines.
  """
  y = start_y
  fold_kind = crease.Fold.MOUNTAIN
  angle = 180

  for _ in range(num_lines):
    machine.draw_line((0, y), (1, y), fold_kind, angle)
    y += distance
    fold_kind = fold_kind.flip()


def pleat_to_hide(
    machine: drawing_machine.CreasePatternDrawingMachine,
    length: float,
    distance: float,
) -> None:
  """Draws an even number of evenly spaced horizontal pleat lines strictly inside the length.

  Args:
    machine: The drawing machine to draw the creases with.
    length: The total length of the paper segment.
    distance: The distance between adjacent crease lines.
  """
  epsilon = 1e-9
  num_lines = math.ceil(length / distance - epsilon)
  if num_lines % 2 != 0:
    num_lines -= 1
  if num_lines <= 0:
    return
  span = (num_lines - 1) * distance
  slack = length - span
  if distance < slack - epsilon:
    start_y = distance
  else:
    start_y = slack / 2.0
  _pleat(machine, num_lines, start_y=start_y, distance=distance)
