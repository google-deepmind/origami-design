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

"""A simple class for managing a 3D scene with a stack of coordinate systems."""

import dataclasses
import numpy as np
from scipy.spatial import transform as spt


@dataclasses.dataclass(frozen=True, kw_only=True)
class Point3D:
  """Represents a point in 3D space."""

  x: float
  y: float
  z: float

  def to_homogeneous(self) -> np.ndarray:
    """Converts the point to a 4D homogeneous vector."""
    return np.array([self.x, self.y, self.z, 1.0])

  @classmethod
  def from_homogeneous(cls, vec: np.ndarray) -> "Point3D":
    """Creates a Point3D from a 4D homogeneous vector."""
    if vec[3] == 0:  # Avoid division by zero for vectors/points at infinity
      return cls(x=np.inf, y=np.inf, z=np.inf)
    return cls(x=vec[0] / vec[3], y=vec[1] / vec[3], z=vec[2] / vec[3])

  def __repr__(self) -> str:
    return f"Point3D(x={self.x:.2f}, y={self.y:.2f}, z={self.z:.2f})"


class World:
  """Manages a 3D scene with a stack of coordinate systems.

  Transformations are applied using transform() and reverted using
  pop_transformation(). This allows for creating nested coordinate
  systems. For example:

  world = World()
  world.transform(translate=(10, 0, 0))  # Push a translation
  world.add_point(1, 1, 0)               # This point is at (11, 1, 0) globally

  try:
      # Use a try...finally block to ensure transformations are popped
      world.transform(rotate_deg=90)     # Push a rotation
      world.add_point(0, 1, 0)           # This point is at (10, 1, 0) globally
  finally:
      world.pop_transformation()         # Pop the rotation

  world.pop_transformation()             # Pop the translation
  """

  def __init__(self):
    # The transformation stack starts with the global coordinate system.
    self._transform_stack: list[np.ndarray] = [np.identity(4)]
    # This list will store tuples of (Point3D, transform_to_global_at_creation).
    self._points_log: list[tuple[Point3D, np.ndarray]] = []

  @property
  def transform_stack_size(self) -> int:
    """Returns the number of transformations currently on the stack."""
    return len(self._transform_stack)

  def transform(
      self,
      *,
      translate: tuple[float, float, float] | np.ndarray = (0, 0, 0),
      rotate_deg: float = 0,
      rotate_axis: tuple[float, float, float] | np.ndarray = (0, 0, 1),
      scale: tuple[float, float, float] | np.ndarray | float = 1.0,
  ) -> None:
    """Calculates a transformation matrix and pushes it to the stack."""
    local_transform = self.transform_matrix(
        translate=translate,
        rotate_deg=rotate_deg,
        rotate_axis=rotate_axis,
        scale=scale,
    )
    self._push_transform(local_transform)

  def transform_matrix(
      self,
      *,
      translate: tuple[float, float, float] | np.ndarray = (0, 0, 0),
      rotate_deg: float = 0,
      rotate_axis: tuple[float, float, float] = (0, 0, 1),
      scale: tuple[float, float, float] | float = 1.0,
  ) -> np.ndarray:
    """Calculates a transformation matrix."""

    if isinstance(scale, (int, float)):
      scale = (scale, scale, scale)

    translation_matrix = np.identity(4)
    translation_matrix[:3, 3] = translate

    rotation_matrix = np.identity(4)
    if rotate_deg != 0:
      rotate_axis = np.array(rotate_axis)
      norm_k = np.linalg.norm(rotate_axis)
      if norm_k > 0:
        rotate_axis = rotate_axis / norm_k
      else:
        raise ValueError("Rotation axis k cannot be a zero vector.")
      rotation = spt.Rotation.from_rotvec(np.deg2rad(rotate_deg) * rotate_axis)
      rotation_matrix[:3, :3] = rotation.as_matrix()

    scale_matrix = np.diag([*scale, 1.0])

    # The final local transformation is Translate -> Rotate -> Scale
    local_transform = translation_matrix @ rotation_matrix @ scale_matrix

    return local_transform

  def peek_transform(self) -> np.ndarray:
    """Returns the current transform matrix."""
    return self._transform_stack[-1]

  def push_local_transform(self, transform: np.ndarray) -> None:
    """Takes a transformation matrix and pushes it to the stack."""
    self._push_transform(transform)

  def pop_transformation(self) -> None:
    """Reverts to the previous coordinate system.

    Pops the current transformation from the stack. Does nothing if only the
    global coordinate system remains.
    """
    if len(self._transform_stack) > 1:
      self._transform_stack.pop()

  def add_point(self, x: float, y: float, z: float):
    """Adds a point defined in the CURRENT coordinate system."""
    point = Point3D(x=x, y=y, z=z)
    # The point is defined locally, so we log it with the current transformation
    # matrix.
    current_transform_to_global = self._transform_stack[-1]
    self._points_log.append((point, current_transform_to_global))

  def add_point_at_origin(self):
    """Adds a point at the origin of the current coordinate system."""
    self.add_point(0, 0, 0)

  def peek_point_at_origin(self) -> np.ndarray:
    """Returns the point at the origin of the current coordinate system."""
    return self.peek_transform() @ np.array([0, 0, 0, 1])

  def get_points_in_current_coords(self) -> list[Point3D]:
    """Computes and returns all points in the current coordinate system."""
    # Get the transformation that maps the current frame to the global frame.
    current_transform_to_global = self._transform_stack[-1]
    # We need its inverse to map points from the global frame back to our
    # current one. This might not work if scale is 0.
    global_to_current_transform = np.linalg.inv(current_transform_to_global)

    points_in_current_cs = []
    # This loop processes each point saved in the log.
    for point, creation_transform_to_global in self._points_log:
      # Step 1: Get the point's coordinates in the global frame.
      p_local_hom = point.to_homogeneous()
      p_global_hom = creation_transform_to_global @ p_local_hom

      # Step 2: Transform the global point into the current coordinate system.
      p_current_hom = global_to_current_transform @ p_global_hom
      points_in_current_cs.append(Point3D.from_homogeneous(p_current_hom))

    return points_in_current_cs

  def get_global_points(self) -> list[Point3D]:
    """Computes and returns all points in the global coordinate system."""
    final_points = []
    for point, transform_matrix in self._points_log:
      p_local_hom = point.to_homogeneous()
      p_global_hom = transform_matrix @ p_local_hom
      final_points.append(Point3D.from_homogeneous(p_global_hom))
    return final_points

  def _push_transform(self, local_transform: np.ndarray) -> None:
    """Calculates new transform and pushes it to the stack."""
    current_global_transform = self._transform_stack[-1]
    # Chain the new local transform with the current one.
    new_global_transform = current_global_transform @ local_transform
    self._transform_stack.append(new_global_transform)
