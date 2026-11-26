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

"""Stick figure class."""

import dataclasses
import enum
import math
import typing
from typing import Any, Literal, TypeAlias

from absl import logging
import numpy as np

from . import angle_utils

GridMode: TypeAlias = Literal['v0', 'v1', 'v0_include_width', 'v2', 'v4']
# pytype doesn't support float literals, but pydantic json schema does.
StickWidth: TypeAlias = Literal[0.25, 0.5, 1, 2, 3, 4]  # pyrefly: ignore[invalid-literal]
StickWidthValues = typing.get_args(StickWidth)
DEFAULT_TOLERANCE = 1e-3


class StickValueError(ValueError):
  """Raised when there is an error with a Stick's attributes."""

  def __init__(self, message: str, stick: 'Stick'):
    message = f'{message}; {stick=}'
    super().__init__(message)


class FlapTipShape(enum.Enum):
  UNSET = 'unset'
  NOT_APPLICABLE = 'not_applicable'
  TRIANGULAR = 'triangular'
  RECTANGULAR = 'rectangular'


@dataclasses.dataclass(frozen=False, kw_only=True)
class Stick:
  """A single stick in a stick figure."""

  name: str
  """Semantic name of the stick."""

  start: str
  """Semantic name of the start joint."""

  end: str
  """Semantic name of the end joint."""

  azimuth_angle: float
  """Polar angle in counter-clockwise degrees of the stick from North at the start joint."""

  elevation_angle: float = 0.0
  """Elevation angle in degrees of the stick from the XY plane."""

  length: int
  """Integer length of the stick."""

  width: StickWidth = 1.0
  """Integer width of the stick."""

  shaping_flap: bool = False
  """Whether the stick is a shaping flap (used for 3D effect)."""

  def __post_init__(self):
    # Whether the stick is a flap (only has a joint on one end).
    self.is_flap: bool = False
    # Whether the stick is a river (each joint has len >= 3).
    self.is_river: bool = False
    # Whether the stick is on the symmetry axis.
    self.is_on_symmetry_axis: bool = False
    # The name of the symmetric stick, if one exists.
    self.symmetric_stick_name: str | None = None
    # Whether the stick is pointing up at the start joint. All flaps are down.
    # For rivers, we walk from one side of the tree. If the start joint is
    # reached first, the river is pointing up, othersise its pointing down.
    self.up_at_start_joint: bool = False
    # The shape of the flap tip.
    self.flap_tip_shape: FlapTipShape = FlapTipShape.UNSET

    if (
        not -angle_utils.AZIMUTH_PERIOD
        <= self.azimuth_angle
        <= angle_utils.AZIMUTH_PERIOD
    ):
      raise StickValueError(
          f'Angle {self.azimuth_angle} is not in [-360, 360].', self
      )

    if not (
        -angle_utils.ELEVATION_PERIOD
        <= self.elevation_angle
        <= angle_utils.ELEVATION_PERIOD
    ):
      raise StickValueError(
          f'Elevation angle {self.elevation_angle} is not in [-90, 90].', self
      )

    is_int_length = isinstance(self.length, int) or (
        isinstance(self.length, float) and self.length.is_integer()
    )
    if not is_int_length:
      raise StickValueError(f'Length {self.length} is not an integer.', self)

    if self.width <= 0:
      logging.warning('Stick width is not positive. Setting to 1: %s', self)
      self.width = 1

    is_int_width = isinstance(self.width, int) or (
        isinstance(self.width, float) and self.width.is_integer()
    )
    if not is_int_width and self.width > 1:
      if math.isfinite(self.width):
        logging.warning(
            'Stick widths greater than 1 should be integers: %s', self
        )
        self.width = round(self.width)
      else:
        raise StickValueError(f'Width is not finite: {self.width}', self)

  def to_dict(self, skip_defaults: bool = False) -> dict[str, Any]:
    """Returns a dictionary representation of the Stick object."""
    data = {}
    for field in dataclasses.fields(self):
      if not field.init:
        continue

      value = getattr(self, field.name)
      if skip_defaults and is_default_value(field, value):
        continue

      data[field.name] = value
    return data

  @classmethod
  def from_dict(cls, data: dict[str, Any]) -> 'Stick':
    """Creates a Stick object from a dictionary."""
    init_field_names = {f.name for f in dataclasses.fields(cls) if f.init}
    data = {k: v for k, v in data.items() if k in init_field_names}
    return cls(**data)

  def azimuth_angle_at_joint(self, joint_name: str) -> float:
    """Returns the angle of the stick pointing away from the given joint (-180 to 180 degrees)."""
    match joint_name:
      case self.start:
        return angle_utils.wrap_angle_centered(
            self.azimuth_angle, angle_utils.AZIMUTH_PERIOD
        )
      case self.end:
        return angle_utils.wrap_angle_centered(
            self.azimuth_angle + angle_utils.AZIMUTH_PERIOD / 2,
            angle_utils.AZIMUTH_PERIOD,
        )
      case _:
        raise ValueError(
            f'Stick {self.name} is not connected to joint {joint_name}.'
        )

  def elevation_angle_at_joint(
      self,
      joint_name: str,
  ) -> float:
    """Returns the elevation angle of the stick pointing away from the given joint ([-90, 90] degrees)."""
    match joint_name:
      case self.start:
        return self.elevation_angle
      case self.end:
        return -self.elevation_angle
      case _:
        raise ValueError(
            f'Stick {self.name} is not connected to joint {joint_name}.'
        )

  def get_coords(self, joint_name: str | None) -> tuple[float, float, float]:
    """Get dx, dy, dz for a stick starting from a joint.

    Args:
      joint_name: The name of the joint from which to calculate the coordinates,
        or None to calculate from the origin.

    Returns:
      A tuple containing the change in x, y, and z coordinates (dx, dy, dz).
    """
    if joint_name is None:
      azimuth_deg = self.azimuth_angle
      elevation_deg = self.elevation_angle
    else:
      azimuth_deg = self.azimuth_angle_at_joint(joint_name)
      elevation_deg = self.elevation_angle_at_joint(joint_name)

    azimuth_rad = math.radians(azimuth_deg)
    elevation_rad = math.radians(elevation_deg)

    xy_projection = self.length * math.cos(elevation_rad)
    dx = xy_projection * -math.sin(azimuth_rad)
    dy = xy_projection * math.cos(azimuth_rad)
    dz = self.length * math.sin(elevation_rad)
    return dx, dy, dz

  def other_joint_name(self, joint_name: str) -> str:
    """Returns the other joint of a stick given one joint."""
    if joint_name not in (self.start, self.end):
      raise ValueError(f'Joint {joint_name} is not part of stick {self.name}.')
    return self.end if self.start == joint_name else self.start

  def rotate(self, rotation_matrix: np.ndarray) -> 'Stick':
    """Applies a rotation matrix to the stick and returns a new Stick instance.

    Args:
      rotation_matrix: A 3x3 NumPy array representing the rotation.

    Returns:
      A new Stick object with updated azimuth and elevation angles.
    """
    if rotation_matrix.shape != (3, 3):  # Check if it's a 3x3 matrix
      raise ValueError(
          f'Rotation matrix must be 3x3, got {rotation_matrix.shape}.'
      )

    # 1. Convert current stick orientation to a Cartesian vector
    p_initial = np.array(self.get_coords(None))

    # 2. Apply the rotation
    p_final = rotation_matrix @ p_initial

    # 3. Convert the new vector back to spherical coordinates
    new_azimuth, new_elevation = angle_utils.cartesian_to_spherical(p_final)

    # 4. Return a new Stick instance with the updated angles
    return dataclasses.replace(
        self, azimuth_angle=new_azimuth, elevation_angle=new_elevation
    )

  def shared_joint_name(self, other: 'Stick') -> list[str]:
    """Returns the name of the joint shared by two sticks if it exists."""
    shared_joints = list(
        set([self.start, self.end]).intersection([other.start, other.end])
    )
    assert len(shared_joints) < 2
    return shared_joints

  def none_shared_joint_name(self, other: 'Stick') -> str:
    """Returns a name of a joint not shared by two sticks."""
    shared_joints = list(
        set([self.start, self.end]).intersection([other.start, other.end])
    )
    assert len(shared_joints) < 2
    return self.start if self.start not in shared_joints else self.end

  def share_joint(self, other: 'Stick') -> bool:
    """Returns whether the stick shares a joint with another stick."""
    return bool(self.shared_joint_name(other))

  @property
  def requires_level_shifter(self) -> bool:
    """Whether the stick is packed to be level shifted."""
    if self.width <= 1:
      return False
    # Flaps must have length > width, while rivers are not restricted.
    return not self.is_flap or self.length > self.width


def get_flaps(sticks: list[Stick]) -> list[Stick]:
  """Returns all the flaps in a list of sticks."""
  return [s for s in sticks if s.is_flap]


def get_flaps_sorted_by_length(sticks: list[Stick]) -> list[Stick]:
  """Returns flaps sorted by length in descending order."""
  return sorted(get_flaps(sticks), key=lambda s: s.length, reverse=True)


def get_rivers(sticks: list[Stick]) -> list[Stick]:
  """Returns all the rivers in a list of sticks."""
  return [s for s in sticks if s.is_river]


def is_default_value(field: dataclasses.Field[Any], value: Any) -> bool:
  """Checks if a value is the default for a dataclass field."""
  if field.default is not dataclasses.MISSING:
    return value == field.default
  if field.default_factory is not dataclasses.MISSING:
    return value == field.default_factory()
  return False
