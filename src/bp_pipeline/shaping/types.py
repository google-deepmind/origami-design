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

"""Types for box pleating shaping."""

from collections.abc import Sequence
import dataclasses
from typing import Literal

from ..crease_pattern import crease as crease_utils

BankIndex = Literal['FIRST', 'SECOND']


@dataclasses.dataclass(frozen=True, kw_only=True)
class ZigzagParams:
  """Parameters determining the drawing of zigzag creases on a flap.

  Attributes:
    target_flap_angle: Target flap angle in radians. Exactly one of
      `target_flap_angle` and `target_zigzag_depth` must be set.
    fold_order: Sequence of folds detailing the fold order (e.g.
      `[MOUNTAIN, VALLEY]`).
    even_phase: Determines the phase shift of the zigzags.
    depth: How deep into the flap zigzags should be inscribed.
    fold_percent: The percentage to fold the zigzag crease.
    target_zigzag_depth: Target depth of the zigzag tip into the flap.
  """

  target_flap_angle: float | None
  fold_order: Sequence[crease_utils.Fold]
  even_phase: bool
  depth: float = 0.0
  fold_percent: float = 1.0
  target_zigzag_depth: float | None = None

  def __post_init__(self) -> None:
    if not (self.target_flap_angle is None) ^ (
        self.target_zigzag_depth is None
    ):
      raise ValueError(
          'Exactly one of target_flap_angle and target_zigzag_depth must be'
          f' defined, got {self.target_flap_angle} and'
          f' {self.target_zigzag_depth}.'
      )


@dataclasses.dataclass(frozen=True, kw_only=True)
class RiverZigzagParams:
  """Parameters determining the drawing of zigzag creases on a river bank.

  Attributes:
    target_flap_angle: Target flap angle in radians.
    target_zigzag_depth: Target zigzag depth.
    fold_order: Sequence of folds detailing the fold order.
    even_phase: Determines the phase shift of the zigzags.
    bank_index: River bank (`'FIRST'` or `'SECOND'`) to shape.
    fold_percent: The percentage to fold the zigzag crease.
    depth: Depth at which zigzag drawing starts.
  """

  target_flap_angle: float | None
  target_zigzag_depth: float | None = None
  fold_order: Sequence[crease_utils.Fold]
  even_phase: bool
  bank_index: BankIndex
  fold_percent: float = 1.0
  depth: float = 0.0
