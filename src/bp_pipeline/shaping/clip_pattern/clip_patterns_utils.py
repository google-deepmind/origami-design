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

"""Utils for clip patterns."""

import enum
import numpy as np


class RotationMode(enum.IntEnum):
  """Rotation crease pattern modes."""

  BETWEEN_0_AND_90 = 0  # 0 < alpha < 90
  BETWEEN_90_AND_180 = 1  # 90 < alpha < 180
  BETWEEN_NEG_90_AND_0 = 2  # -90 < alpha < 0
  BETWEEN_180_AND_NEG_90 = 3  # -180 < alpha < -90

  FIXED_180 = 4  # alpha is exactly 180 or -180
  FIXED_0 = 5  # alpha is exactly 0
  FIXED_90 = 6  # alpha is exactly 90
  FIXED_NEG_90 = 7  # alpha is exactly -90


def normalize_angle(theta_deg: float) -> float:
  """Normalizes the angle to the range (-180, 180]."""
  return (theta_deg + 180) % 360 - 180


def rotation_mode(angle: float) -> tuple[RotationMode, float]:
  """Determines the rotation mode and a normalized angle value.

  Args:
    angle: The rotation angle in degrees.

  Returns:
    A tuple containing:
      - RotationMode: The category of the rotation angle.
      - float: A normalized angle value used for drawing, specific to each mode.
  """
  alpha = normalize_angle(angle)

  if np.isclose(alpha, 180) or np.isclose(alpha, -180):
    return RotationMode.FIXED_180, 180.0
  elif np.isclose(alpha, 0):
    return RotationMode.FIXED_0, 0.0
  elif np.isclose(alpha, 90):
    return RotationMode.FIXED_90, 90.0
  elif np.isclose(alpha, -90):
    return RotationMode.FIXED_NEG_90, -90.0

  if 0 < alpha < 90:
    return RotationMode.BETWEEN_0_AND_90, alpha
  elif 90 < alpha < 180:
    return RotationMode.BETWEEN_90_AND_180, 180.0 - alpha
  elif -90 < alpha < 0:
    return RotationMode.BETWEEN_NEG_90_AND_0, -alpha
  elif -180 < alpha < -90:
    return RotationMode.BETWEEN_180_AND_NEG_90, 180.0 + alpha

  raise ValueError(f"Unexpected angle {angle}")
