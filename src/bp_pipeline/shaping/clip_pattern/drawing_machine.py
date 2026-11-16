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

"""Drawing machine for crease pattern."""

import dataclasses
import types

from matplotlib import lines as mpl_lines
import numpy as np

from ...crease_pattern import crease
from . import drawing_machine_utils
from ...crease_pattern import point as cp_point
from ...crease_pattern.parser import svg
from . import world_space


@dataclasses.dataclass(frozen=True, kw_only=True)
class Line:
  """Represents a line segment with color and alpha.

  Attributes:
    p1: The start point of the line segment, as a numpy array of any shape (2+,)
    p2: The end point of the line segment, as a numpy array of any shape(2+,)
    color: The color of the line segment, as a string.
    alpha: The opacity of the line segment from 0 to 1.
  """

  p1: np.ndarray
  p2: np.ndarray
  color: str
  alpha: float

  def to_line2d(self) -> mpl_lines.Line2D:
    """Converts Line to matplotlib Line2D, uses only x and y coordinates."""
    return mpl_lines.Line2D(
        [self.p1[0], self.p2[0]],
        [self.p1[1], self.p2[1]],
        color=self.color,
        alpha=self.alpha,
    )

  def to_crease(self) -> crease.Crease:
    """Converts Line to crease.Crease."""
    return crease.Crease(
        cp_point.Point(x=self.p1[0], y=self.p1[1], require_on_paper=False),
        cp_point.Point(x=self.p2[0], y=self.p2[1], require_on_paper=False),
        fold=svg.get_fold(self.color),
        fold_percent=self.alpha,
    )

  def __eq__(self, other: 'Line') -> bool | types.NotImplementedType:
    """Custom Line equality check to handle numpy arrays and p1/p2 swaps."""
    if not isinstance(other, Line):
      return NotImplemented
    points_eq = np.allclose(self.p1, other.p1) and np.allclose(
        self.p2, other.p2
    )
    swapped_points_eq = np.allclose(self.p1, other.p2) and np.allclose(
        self.p2, other.p1
    )
    return (
        (points_eq or swapped_points_eq)
        and self.color == other.color
        and self.alpha == other.alpha
    )


class LineDrawer:
  """Stores lines and allows to add new lines imperatively.

  LineDrawer is a basic object that stores lines (as Line objects that
  supports color and opacity). New line is added by calling start_line(point1)
  and end_line(point2). Meant to be used through CanvasDrawingMachine.

    Attributes:
      lines: A list of Line objects representing the lines drawn.
      color: The color to use for new lines.
      alpha: The opacity to use for new lines.
  """

  def __init__(self):
    self.lines: list[Line] = []
    self.color = 'black'
    self.alpha = 1.0
    self._line_start: np.ndarray | None = None

  def start_line(self, point: np.ndarray) -> None:
    """Stores the start point of a new line."""
    if self._line_start is not None:
      raise ValueError('To start a line, machine must not have an open line.')
    self._line_start = np.array(point)

  def end_line(self, point: np.ndarray) -> None:
    """Completes the line and adds it to the list of lines."""
    if self._line_start is None:
      raise ValueError('To end a line, machine must have an open line.')
    end_point = np.array(point)
    self.lines.append(
        Line(
            p1=self._line_start,
            p2=end_point,
            color=self.color,
            alpha=self.alpha,
        )
    )
    self._line_start = None


class CanvasDrawingMachine:
  """Maintains 3D coordinate system when uses LineDrawer.

  CanvasDrawingMachine is composed of LineDrawer andworld_space.World. It
  uses world_space.World to maintain position and orientation of the drawing
  machine as a 3D coordinate system in global space.
  It has start_line() and end_line() with no arguments. Current position of the
  drawing machine is taken from world_space.World and passed as a start or end
  point of the line.

    Attributes:
      line_machine: A LineDrawer object that stores the lines drawn.
      canvas: A world_space.World object that maintains the position and
        orientation of the drawing machine. Position is 4-dimensional numpy
        array (x, y, z, 1).
  """

  def __init__(self):
    self.line_machine = LineDrawer()
    self.canvas = world_space.World()

  def start_line(self):
    """Starts a new line at the current position of the drawing machine."""
    self.line_machine.start_line(self.canvas.peek_point_at_origin())

  def end_line(self):
    """Ends the current line at the current position of the drawing machine."""
    self.line_machine.end_line(self.canvas.peek_point_at_origin())

  @property
  def lines(self) -> list[Line]:
    """Returns list of Line objects."""
    return self.line_machine.lines

  def get_clipped_lines(self, polygon_points: np.ndarray) -> list[Line]:
    """Clips lines to the polygon. Only x and y coordinates are used."""
    if polygon_points.size == 0:
      return self.lines
    if not self.lines:
      return []
    lines_2d = np.array([[line.p1[:2], line.p2[:2]] for line in self.lines])
    clip_results = drawing_machine_utils.clip_lines_to_polygon(
        lines_2d, polygon_points, fast=True
    )
    return [
        Line(
            p1=segment[0],
            p2=segment[1],
            color=self.lines[line_idx].color,
            alpha=self.lines[line_idx].alpha,
        )
        for line_idx, segment in clip_results
    ]

  def get_lines2d(self) -> list[mpl_lines.Line2D]:
    """Returns list of matplotlib Line2D objects."""
    return [line.to_line2d() for line in self.lines]


class CreasePatternDrawingMachine:
  """Similar to CanvasDrawingMachine but specilized in crease pattern drawing.

  Crease pattern in origami is a set of lines which represent mountain and
  valley creases, which acts as a blueprint for paper folding. Mountains are
  red, valleys are blue, borders or unassigned creases are black. Opacity of
  the line encodes the target fold angle (0 to 180 degrees).

  CreasePatternDrawingMachine is composed of CanvasDrawingMachine and
  maintains mountain and valley colors and fold angle for the next line. It has
  set_mountain() and set_valley(), set_fold_angle_deg(theta). It also has a
  feature that crease type is automatically switched to the opposite based on
  the z-axis of the drawing machine every time it ends a line. This allows to
  re-draw the crease pattern with inverted colors just by changing sign of
  z-axis at the beginning. Like with self.canvas.transform(scale=(1, 1, -1)) or
  a rotation.

  Attributes:
    canvas: A world_space.World object that maintains the position and
      orientation of the drawing machine. Position is 4-dimensional numpy array
      (x, y, z, 1).
    lines: A list of Line objects representing the lines already drawn.
    machine_side: An integer indicating which side (0 or 1) of the machine is on
      the paper right now. if 1, M/V will be inverted from what user sets.
    fold_angle_deg: A fold angle associated with the next crease, translated to
      opacity of the line.
  """

  def __init__(self):
    self._machine = CanvasDrawingMachine()
    self._color_pair = [
        crease.Fold.MOUNTAIN.color(),
        crease.Fold.VALLEY.color(),
    ]
    self._fold_angle_deg = 180

  @property
  def canvas(self) -> world_space.World:
    return self._machine.canvas

  @property
  def lines(self) -> list[Line]:
    return self._machine.lines

  @property
  def machine_side(self) -> int:
    """Returns which side (0 or 1) of the machine is on the paper."""
    matrix = self._machine.canvas.peek_transform()
    if np.isclose(matrix[2, 2], 1):
      return 0
    if np.isclose(matrix[2, 2], -1):
      return 1
    raise ValueError(
        f'Unexpected z-basis coordinate in the matrix {matrix}. Must be -1 or 1'
    )

  def _set_line_features(self):
    if self._color_pair[self.machine_side] == crease.Fold.BORDER.color():
      self._machine.line_machine.alpha = 1.0
    else:
      self._machine.line_machine.alpha = self.fold_angle_deg / 180
    self._machine.line_machine.color = self._color_pair[self.machine_side]

  def _validate_fold_angle(self, theta: float):
    if not 0 <= theta <= 180:
      raise ValueError(
          f'Fold angle must be in the range [0, 180], got {theta}.'
      )

  @property
  def fold_angle_deg(self) -> float:
    return self._fold_angle_deg

  def set_fold_angle_deg(self, theta: float):
    self._validate_fold_angle(theta)
    self._fold_angle_deg = theta

  def set_mountain(self):
    """Sets machine to draw mountain creases."""
    self._color_pair = [
        crease.Fold.MOUNTAIN.color(),
        crease.Fold.VALLEY.color(),
    ]

  def set_valley(self):
    """Sets machine to draw valley creases."""
    self._color_pair = [
        crease.Fold.VALLEY.color(),
        crease.Fold.MOUNTAIN.color(),
    ]

  def set_unassigned(self):
    """Sets machine to draw unassigned creases."""
    self._color_pair = [crease.Fold.BORDER.color(), crease.Fold.BORDER.color()]

  def start_line(self):
    self._machine.start_line()

  def end_line(self):
    self._set_line_features()
    self._machine.end_line()

  def draw_line(
      self,
      start_point: tuple[float, float],
      end_point: tuple[float, float],
      fold_kind: crease.Fold,
      fold_angle_deg: float,
  ):
    """Draws a line relative to the current canvas."""
    match fold_kind:
      case crease.Fold.MOUNTAIN:
        self.set_mountain()
      case crease.Fold.VALLEY:
        self.set_valley()
      case crease.Fold.BORDER:
        self.set_unassigned()
      case _:
        raise ValueError(f'Not supported fold kind: {fold_kind}')
    self.set_fold_angle_deg(fold_angle_deg)

    self.canvas.transform(translate=(start_point[0], start_point[1], 0))
    self.start_line()
    self.canvas.pop_transformation()
    self.canvas.transform(translate=(end_point[0], end_point[1], 0))
    self.end_line()
    self.canvas.pop_transformation()

  def get_clipped_lines(self, polygon_points: np.ndarray) -> list[Line]:
    """Crops lines to the polygon."""
    return self._machine.get_clipped_lines(polygon_points)
