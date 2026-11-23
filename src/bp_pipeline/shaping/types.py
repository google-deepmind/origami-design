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
"""Shared parameter types and fold-order constants for origami shaping."""

from collections.abc import Sequence
import dataclasses
from typing import Literal

from ..crease_pattern import crease as crease_utils

type BankIndex = Literal['FIRST', 'SECOND']
type FoldOrder = tuple[crease_utils.Fold, ...]

MOUNTAIN_TO_VALLEY: FoldOrder = (
    crease_utils.Fold.MOUNTAIN,
    crease_utils.Fold.VALLEY,
)
VALLEY_TO_MOUNTAIN: FoldOrder = (
    crease_utils.Fold.VALLEY,
    crease_utils.Fold.MOUNTAIN,
)


def flip_fold_order(fold_order: Sequence[crease_utils.Fold]) -> FoldOrder:
  """Returns a reversed two-element `(MOUNTAIN, VALLEY)` fold order."""
  if len(fold_order) != 2:
    raise ValueError(
        f'Fold order must have length 2 to be flipped, got {len(fold_order)}.'
    )
  return (fold_order[1], fold_order[0])


@dataclasses.dataclass(frozen=True, kw_only=True)
class ZigzagParams:
  """Parameters governing zigzag creases on a flap or river bank.

  Attributes:
    target_flap_angle: Target flap angle in radians. Exactly one of
      `target_flap_angle` and `target_zigzag_depth` must be set.
    target_zigzag_depth: Target zigzag tip depth in grid units. Exactly one of
      `target_flap_angle` and `target_zigzag_depth` must be set.
    fold_order: Repeating mountain/valley fold sequence along the boundary.
    even_phase: Whether the boundary anchor walk uses even phase.
    depth: Inscribe offset in grid units inward from the element boundary.
    fold_percent: Fractional fold angle in `[0, 1]` (`1.0` = 180 degrees).
    bank_index: River bank (`'FIRST'` or `'SECOND'`) when shaping a river, or
      `None` when shaping a flap.
  """

  target_flap_angle: float | None = None
  target_zigzag_depth: float | None = None
  fold_order: Sequence[crease_utils.Fold]
  even_phase: bool
  depth: float = 0.0
  fold_percent: float = 1.0
  bank_index: BankIndex | None = None

  def __post_init__(self) -> None:
    if not (self.target_flap_angle is None) ^ (
        self.target_zigzag_depth is None
    ):
      raise ValueError(
          'Exactly one of target_flap_angle and target_zigzag_depth must be'
          f' defined; got {self.target_flap_angle} and'
          f' {self.target_zigzag_depth}.'
      )
    normalized = tuple(self.fold_order)
    if not normalized:
      raise ValueError('fold_order must be non-empty.')
    object.__setattr__(self, 'fold_order', normalized)


@dataclasses.dataclass(frozen=True, kw_only=True)
class InscribedParams:
  """Parameters governing an inscribed polygon crease on a flap or river bank.

  Attributes:
    depth: Inward offset in grid units from the flap border or river bank.
    fold_order: Repeating mountain/valley fold sequence along the boundary.
    fold_percent: Fractional fold angle in `[0, 1]` (`1.0` = 180 degrees).
    bank_index: River bank (`'FIRST'` or `'SECOND'`) when shaping a river, or
      `None` when shaping a flap.
  """

  depth: float
  fold_order: Sequence[crease_utils.Fold]
  fold_percent: float = 1.0
  bank_index: BankIndex | None = None

  def __post_init__(self) -> None:
    normalized = tuple(self.fold_order)
    if not normalized:
      raise ValueError('fold_order must be non-empty.')
    object.__setattr__(self, 'fold_order', normalized)


@dataclasses.dataclass(frozen=True, kw_only=True)
class ElementShapingParams[P]:
  """Pairs an element name (flap or river) with its shaping parameters."""

  element_name: str
  params: P


type ElementZigzagParams = ElementShapingParams[ZigzagParams]
type ElementInscribedParams = ElementShapingParams[Sequence[InscribedParams]]
