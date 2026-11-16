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

"""Utilities for solving a template for level shifters and then placing them."""

import collections
from collections.abc import Callable, Collection, Iterator, Sequence
import dataclasses
import functools
import itertools
import logging
import math
import traceback
import typing
from typing import NamedTuple, Protocol

from ..crease_pattern import crease as crease_lib
from ..crease_pattern import crease_pattern
from ..crease_pattern import point as point_lib
from ..crease_pattern import visualize
from ..packing import grid
from ..packing import river as river_lib
from ..packing import types as packing_types
from . import shaping_utils
from ..solving import hinges as hinge_lib
from ..solving import utils

TEMP_FOLD = crease_lib.Fold.TRIANGLE
LocalIndex = int
LSPosition = tuple[LocalIndex, LocalIndex]


def get_ls_position_from_percent(
    valid_positions: Sequence[LSPosition], position_percent: float
) -> LSPosition:
  """Returns the level shifter position corresponding to the given percent.

  Args:
    valid_positions: The sorted list of valid level shifter positions.
    position_percent: A float between 0.0 and 1.0 indicating the desired
      position. 0.0 maps to the first valid position, 1.0 maps to the last.

  Returns:
    The level shifter position corresponding to the given percent.
  """
  if not valid_positions:
    raise ValueError("No valid positions provided.")
  if not 0.0 <= position_percent <= 1.0:
    raise ValueError("position_percent must be between 0.0 and 1.0.")

  index = int(round(position_percent * (len(valid_positions) - 1)))
  return valid_positions[index]


def _get_outer_fold(creases: list[crease_lib.Crease]) -> crease_lib.Fold:
  """Returns the fold type of the start and end of a list of creases."""
  if creases[0].intersects_border:
    return creases[-1].fold
  elif creases[-1].intersects_border:
    return creases[0].fold

  assert (
      creases[0].fold == creases[-1].fold
  ), f"First and last creases have different folds: {creases[0]}, {creases[-1]}"
  return creases[0].fold


def _replace_matching_creases(
    creases: list[crease_lib.Crease],
    replacement_creases: list[crease_lib.Crease],
) -> list[crease_lib.Crease]:
  """Removes matching creases from the list of creases."""
  replaced_creases = []
  used_replacement_indices = set()

  for c in creases:
    matched_index = next(
        (i for i, rc in enumerate(replacement_creases) if c.isclose(rc)), -1
    )

    if matched_index != -1:
      used_replacement_indices.add(matched_index)
      if c.fold != crease_lib.Fold.BORDER:
        replaced_creases.append(replacement_creases[matched_index])
        continue

    replaced_creases.append(c)

  for i, rc in enumerate(replacement_creases):
    if i not in used_replacement_indices:
      replaced_creases.append(rc)

  return replaced_creases


def _crease_norm_key(c: crease_lib.Crease) -> tuple[float, float]:
  return (c.p1.norm, c.p1.x)


def _get_central_ridge(
    rectangle: shaping_utils.Rectangle,
) -> crease_lib.Crease | None:
  return next((c for c in rectangle.get_ridges() if not c.is_diagonal), None)


class CrimpRidgePointPair(NamedTuple):
  crimp: point_lib.Point
  ridge: point_lib.Point | None


@dataclasses.dataclass(frozen=True)
class SacrificialFlapTemplate:
  """Handles templating and placement of a level shifter sacrificial flap.

  Attributes:
    side: The side of this flap that is attached to the main flap.
    rectangle: Normalized PackingOutput rectangle corresponding to this flap.
    river_sf_info: Optional metadata about the river side placement.
  """

  side: grid.Direction
  rectangle: shaping_utils.Rectangle
  river_sf_info: packing_types.RiverSfInfo | None = None

  @property
  def size(self) -> int:
    """Size of the sacrificial flap in grid units."""
    return 1

  def __post_init__(self):
    if self.rectangle.radius is None or self.rectangle.radius > 1:
      raise ValueError("Input rectangles should be normalized.")

  @functools.cached_property
  def crimp_hinge(self) -> crease_lib.Crease:
    """The hinge crease to be replaced by a crimp."""
    match self.side:
      case grid.Direction.NORTH:
        return self.rectangle.edges[0]
      case grid.Direction.EAST:
        return self.rectangle.edges[1]
      case grid.Direction.SOUTH:
        return self.rectangle.edges[2]
      case grid.Direction.WEST:
        return self.rectangle.edges[3]
      case _:
        typing.assert_never(self.side)

  @functools.cached_property
  def opposite_hinge(self) -> crease_lib.Crease:
    """The hinge crease opposite to the crimp hinge."""
    match self.side:
      case grid.Direction.NORTH:
        return self.rectangle.edges[2]
      case grid.Direction.EAST:
        return self.rectangle.edges[3]
      case grid.Direction.SOUTH:
        return self.rectangle.edges[0]
      case grid.Direction.WEST:
        return self.rectangle.edges[1]
      case _:
        typing.assert_never(self.side)

  def _get_num_steps_to_crimp(self, grid_size: int) -> int:
    """Returns number of grid steps between crimp and ridge."""
    if self.crimp_hinge.is_horizontal:
      dist = self.rectangle.y_len / 2
    else:
      dist = self.rectangle.x_len / 2
    return int(round(dist * grid_size))

  def _get_crimp_ridge_padding(
      self,
      crimp_creases: Sequence[crease_lib.Crease],
      central_ridge_creases: Sequence[crease_lib.Crease],
  ) -> tuple[int, int]:
    """Returns start and end padding between crimp and ridge creases."""
    crimp_start = crimp_creases[0]
    ridge_start = central_ridge_creases[0]
    crimp_end = crimp_creases[-1]
    ridge_end = central_ridge_creases[-1]
    seg_len = crimp_start.length
    if crimp_start.is_vertical:
      start_diff = abs(ridge_start.p1.y - crimp_start.p1.y)
      end_diff = abs(crimp_end.p2.y - ridge_end.p2.y)
    else:
      start_diff = abs(ridge_start.p1.x - crimp_start.p1.x)
      end_diff = abs(crimp_end.p2.x - ridge_end.p2.x)
    num_pad_start = int(round(start_diff / seg_len))
    num_pad_end = int(round(end_diff / seg_len))
    return num_pad_start, num_pad_end

  def _template_central_ridge_segments(
      self, solved_creases: list[crease_lib.Crease]
  ) -> list[crease_lib.Crease]:
    """Returns segments corresponding to this LS-template's central ridge."""
    central_ridge = _get_central_ridge(self.rectangle)
    assert (
        central_ridge is not None
    ), "Level shifter flaps must have a central ridge."

    solved_ridge_segments = [
        c
        for c in solved_creases
        if c.is_within(central_ridge, match_fold=False)
    ]
    if not solved_ridge_segments:
      raise ValueError(
          "No creases found within the central ridge, did you solve with"
          " correctly preassigned creases?"
      )
    solved_ridge_segments.sort(key=_crease_norm_key)

    fold = _get_outer_fold(solved_ridge_segments)
    # The central ridge creases have the opposite fold than the level shifter
    # template central ridge
    fold = fold.flip()
    return [dataclasses.replace(c, fold=fold) for c in solved_ridge_segments]

  def template_crimp_segments(
      self, solved_creases: Collection[crease_lib.Crease], inside: bool = True
  ) -> list[crease_lib.Crease]:
    """Returns the segments corresponding to the crimped hinge.

    Args:
      solved_creases: The solved creases from which to extract crimp segments.
      inside: Whether to use inside crimping.

    Returns:
      A list of crease segments along the crimp hinge, sorted by norm.

    Raises:
      NotImplementedError: If inside is False.
      ValueError: If no creases are found within the crimp hinge.
    """
    if not inside:
      raise NotImplementedError("Only inside crimping is implemented.")
    solved_creases = [
        c
        for c in solved_creases
        if c.is_within(self.crimp_hinge, match_fold=False)
    ]
    if not solved_creases:
      raise ValueError(
          "No creases found within the edge, did you solve with correctly"
          " preassigned creases?"
      )
    solved_creases.sort(key=_crease_norm_key)

    fold = _get_outer_fold(solved_creases)
    return [dataclasses.replace(c, fold=fold) for c in solved_creases]

  def _align_crimp_and_ridge(
      self,
      crimp_creases: Sequence[crease_lib.Crease],
      central_ridge_creases: Sequence[crease_lib.Crease],
  ) -> list[tuple[crease_lib.Crease, crease_lib.Crease | None]]:
    """Aligns adjacent crimp and ridge creases by padding ridges with Nones."""
    # Calculate padding based on the offset between the start/end of the crimp
    # and ridge. We assume the creases are sorted and parallel.
    num_pad_start, num_pad_end = self._get_crimp_ridge_padding(
        crimp_creases, central_ridge_creases
    )

    padding_start = [None] * num_pad_start
    padding_end = [None] * num_pad_end

    padded_ridges = [*padding_start, *central_ridge_creases, *padding_end]
    if len(padded_ridges) != len(crimp_creases):
      # Fallback or detailed error if calculation failed to match lengths
      raise ValueError(
          "Number of ridges + padding must be equal to crimps:"
          f" {len(padded_ridges)=} != {len(crimp_creases)=}."
          f" pads: {num_pad_start}, {num_pad_end}"
      )

    return list(
        zip(
            crimp_creases,
            padded_ridges,
            strict=True,
        )
    )

  def _aligned_crimp_and_ridge_points(
      self,
      crimp_creases: Sequence[crease_lib.Crease],
      central_ridge_creases: Sequence[crease_lib.Crease],
  ) -> list[CrimpRidgePointPair]:
    """Returns a list of paired crimp and ridge points.

    Pairs each crimp point with its corresponding ridge point, inserting
    None for ridge points where the ridge does not extend to the crimp.

    Args:
      crimp_creases: Sorted crease segments along the crimp hinge.
      central_ridge_creases: Sorted crease segments along the central ridge.

    Returns:
      A list of CrimpRidgePointPair tuples pairing each crimp point with
      its corresponding ridge point (or None if no ridge exists at that
      position).
    """
    paired_points: list[CrimpRidgePointPair] = []
    aligned_creases = self._align_crimp_and_ridge(
        crimp_creases, central_ridge_creases
    )
    crimp_start, ridge_start = aligned_creases[0]
    if ridge_start is None:
      paired_points.append(CrimpRidgePointPair(crimp_start.p1, None))
    else:
      # If an aligned ridge exists, its start must be on the border.
      assert crimp_start.p1.is_border and ridge_start.p1.is_border
      paired_points.append(CrimpRidgePointPair(crimp_start.p1, ridge_start.p1))

    for (crimp_start, ridge_start), (
        crimp_end,
        ridge_end,
    ) in itertools.pairwise(aligned_creases):
      assert crimp_start.p2.isclose(crimp_end.p1)
      crimp_point = crimp_start.p2

      if ridge_start is not None and ridge_end is not None:
        assert ridge_start.p2.isclose(ridge_end.p1)
        ridge_point = ridge_start.p2
      elif ridge_start is not None:
        ridge_point = ridge_start.p2
      elif ridge_end is not None:
        ridge_point = ridge_end.p1
      else:
        ridge_point = None

      paired_points.append(CrimpRidgePointPair(crimp_point, ridge_point))

    crimp_end, ridge_end = aligned_creases[-1]
    if ridge_end is None:
      paired_points.append(CrimpRidgePointPair(crimp_end.p2, None))
    else:
      assert crimp_end.p2.is_border and ridge_end.p2.is_border
      paired_points.append(CrimpRidgePointPair(crimp_end.p2, ridge_end.p2))

    return paired_points

  def _template_inverted_pleats(
      self,
      *,
      solved_creases: Collection[crease_lib.Crease],
      crimp_creases: Sequence[crease_lib.Crease],
      central_ridge_creases: Sequence[crease_lib.Crease],
  ) -> list[crease_lib.Crease]:
    """Returns inverted pleats from between the crimp and ridge."""
    matched_axial_segments = []
    aligned_crimp_and_ridge_points = self._aligned_crimp_and_ridge_points(
        crimp_creases, central_ridge_creases
    )
    for crimp_point, ridge_point in aligned_crimp_and_ridge_points:
      if ridge_point is None:
        continue
      if ridge_point.is_border:
        continue
      axial_segment = crease_lib.Crease(
          crimp_point,
          ridge_point,
          fold=TEMP_FOLD,
      )
      matched_axial_segments.append(
          next(c for c in solved_creases if c.isclose(axial_segment))
      )

    fold = crimp_creases[0].fold
    inverted_axial_segments = []

    _, start_ridge_point = aligned_crimp_and_ridge_points[0]
    if start_ridge_point is not None:  # Ridge touches border
      # Like get_outer_fold, if the flap touches the border we should assign the
      # segments from the (more-constrained) side with the flap vertex
      matched_axial_segments = reversed(matched_axial_segments)
    for c in matched_axial_segments:
      fold = fold.flip()
      inverted_axial_segments.append(dataclasses.replace(c, fold=fold))
    if start_ridge_point is not None:
      # Un-reverse to ensure the pleats are listed in the correct order
      inverted_axial_segments = inverted_axial_segments[::-1]

    return inverted_axial_segments

  def _template_crimp_propagation(
      self,
      *,
      central_ridge_creases: Sequence[crease_lib.Crease],
      crimp_creases: Sequence[crease_lib.Crease],
      grid_size: int,
  ) -> list[crease_lib.Crease]:
    """Propagates crimp creases to fill the gap between hinge and ridge."""
    num_steps_to_crimp = self._get_num_steps_to_crimp(grid_size)
    if num_steps_to_crimp <= 1:
      return []

    ridge_start = central_ridge_creases[0]

    aligned_points = self._aligned_crimp_and_ridge_points(
        crimp_creases, central_ridge_creases
    )
    valid_pair = next(
        (pair for pair in aligned_points if pair.ridge is not None), None
    )
    if not valid_pair:
      return []

    crimp_p, ridge_p = valid_pair
    # Step vector from the ridge towards the crimp.
    step_x = (crimp_p.x - ridge_p.x) / num_steps_to_crimp
    step_y = (crimp_p.y - ridge_p.y) / num_steps_to_crimp

    # Longitudinal vector (along the ridge)
    unit_long_x = ridge_start.p2.x - ridge_start.p1.x
    unit_long_y = ridge_start.p2.y - ridge_start.p1.y

    max_ext_start, max_ext_end = self._get_crimp_ridge_padding(
        crimp_creases, central_ridge_creases
    )

    ridge_start_p = central_ridge_creases[0].p1
    ridge_end_p = central_ridge_creases[-1].p2
    ridge_fold = ridge_start.fold

    propagated_creases = []

    for k in range(1, num_steps_to_crimp):
      # Alternating folds.
      current_fold = ridge_fold.flip() if k % 2 != 0 else ridge_fold
      # Longitudinal extension is limited by the padding.
      ext_start = min(k, max_ext_start)
      ext_end = min(k, max_ext_end)

      # Transverse shift (perpendicular to ridge).
      shift_x = k * step_x
      shift_y = k * step_y

      # Longitudinal shift (parallel to ridge).
      delta_start_x = ext_start * unit_long_x
      delta_start_y = ext_start * unit_long_y
      delta_end_x = ext_end * unit_long_x
      delta_end_y = ext_end * unit_long_y

      p1 = point_lib.Point(
          x=ridge_start_p.x + shift_x - delta_start_x,
          y=ridge_start_p.y + shift_y - delta_start_y,
          require_on_paper=False,
      )
      p2 = point_lib.Point(
          x=ridge_end_p.x + shift_x + delta_end_x,
          y=ridge_end_p.y + shift_y + delta_end_y,
          require_on_paper=False,
      )
      p3 = point_lib.Point(
          x=ridge_end_p.x - shift_x + delta_end_x,
          y=ridge_end_p.y - shift_y + delta_end_y,
          require_on_paper=False,
      )
      p4 = point_lib.Point(
          x=ridge_start_p.x - shift_x - delta_start_x,
          y=ridge_start_p.y - shift_y - delta_start_y,
          require_on_paper=False,
      )

      propagated_creases.extend([
          crease_lib.Crease(p1, p2, fold=current_fold),
          crease_lib.Crease(p2, p3, fold=current_fold),
          crease_lib.Crease(p3, p4, fold=current_fold),
          crease_lib.Crease(p4, p1, fold=current_fold),
      ])

    return [c for c in propagated_creases if not c.is_overlapping_border]

  def ls_template_creases(
      self, solved_creases: list[crease_lib.Crease], grid_size: int
  ) -> list[crease_lib.Crease]:
    """Returns all new creases of the level shifter template.

    Computes crimps, central ridge, inverted pleats, and crimp propagation
    creases that together form the level shifter template.

    Args:
      solved_creases: The solved base creases to template from.
      grid_size: The grid resolution used for crimp propagation.

    Returns:
      A list of all creases comprising the level shifter template.
    """
    crimp_creases = self.template_crimp_segments(solved_creases)
    central_ridge_creases = self._template_central_ridge_segments(
        solved_creases
    )
    inverted_pleats = self._template_inverted_pleats(
        solved_creases=solved_creases,
        crimp_creases=crimp_creases,
        central_ridge_creases=central_ridge_creases,
    )
    propagated_creases = self._template_crimp_propagation(
        central_ridge_creases=central_ridge_creases,
        crimp_creases=crimp_creases,
        grid_size=grid_size,
    )
    all_creases = (
        crimp_creases
        + central_ridge_creases
        + inverted_pleats
        + propagated_creases
    )
    return all_creases

  def convert_to_ls_template(
      self, solved_creases: list[crease_lib.Crease], grid_size: int
  ) -> list[crease_lib.Crease]:
    """Applies this level shifter template to a base CP.

    Replaces matching creases in the solved base with the level shifter
    template creases.

    Args:
      solved_creases: The solved base creases to modify.
      grid_size: The grid resolution.

    Returns:
      A new list of creases with LS template creases applied.
    """
    return _replace_matching_creases(
        solved_creases, self.ls_template_creases(solved_creases, grid_size)
    )

  def _ls_square_creases(
      self,
      front_points: CrimpRidgePointPair,
      mid_points: CrimpRidgePointPair,
      back_points: CrimpRidgePointPair,
      fold: crease_lib.Fold,
  ) -> tuple[list[crease_lib.Crease], crease_lib.Crease]:
    """Returns the creases of the level shifter square."""

    mid_crimp, mid_ridge = mid_points
    assert mid_ridge is not None
    _, front_corner = front_points
    _, back_corner = back_points

    # Find opposite corner with delta
    cr_dx, cr_dy = (
        mid_ridge.x - mid_crimp.x,
        mid_ridge.y - mid_crimp.y,
    )
    opposite_corner = point_lib.Point(
        x=mid_crimp.x + 2 * cr_dx,
        y=mid_crimp.y + 2 * cr_dy,
    )

    creases = []
    for p1, p2 in itertools.pairwise(
        [front_corner, mid_crimp, back_corner, opposite_corner, front_corner]
    ):
      if p1 is not None and p2 is not None:
        creases.append(crease_lib.Crease(p1, p2, fold=fold))

    return creases, crease_lib.Crease(
        mid_crimp, opposite_corner, fold=fold.flip()
    )

  def apply_level_shifter(
      self, template_creases: list[crease_lib.Crease], position: int
  ) -> list[crease_lib.Crease]:
    """Applies a level shifter square at the given position.

    Inserts a level shifter square into the template creases by adding the level
    shifter square, and modifying the appropriate axial crease segments within
    the template.

    Args:
      template_creases: The LS template creases to modify.
      position: The 1-indexed position along the crimp hinge at which to place
        the level shifter.

    Returns:
      A new list of creases with the level shifter applied.

    Raises:
      ValueError: If position is not positive.
    """
    if position <= 0:
      # NOTE: support position 1
      raise ValueError(f"Position must be positive: {position}")

    crimp_creases = self.template_crimp_segments(template_creases)
    ridge_creases = self._template_central_ridge_segments(template_creases)

    ridge_folds = {c.fold for c in ridge_creases}
    if len(ridge_folds) != 1:
      raise ValueError(
          f"All central ridges must have the same fold: {ridge_folds}"
      )
    ridge_fold = ridge_folds.pop()

    aligned_crimp_and_ridge_points = self._aligned_crimp_and_ridge_points(
        crimp_creases,
        ridge_creases,
    )

    aligned_ls_points = aligned_crimp_and_ridge_points[
        position - 1 : position + 2
    ]

    ls_square_creases, ls_axial_crease = self._ls_square_creases(
        *aligned_ls_points, fold=ridge_fold.flip()
    )

    filtered_creases: list[crease_lib.Crease] = []
    for c in template_creases:
      if c.fold != crease_lib.Fold.BORDER and c.is_overlapping(ls_axial_crease):
        filtered_creases.extend(c.split_to_segments(ls_square_creases))
      else:
        filtered_creases.append(c)
    template_creases = filtered_creases

    front_points, mid_points, back_points = aligned_ls_points
    _, mid_ridge = mid_points
    front_crimp, front_ridge = front_points
    back_crimp, back_ridge = back_points

    to_remove = []
    to_remove.append(ls_axial_crease)
    if front_ridge is not None:
      to_remove.append(
          crease_lib.Crease(front_ridge, mid_ridge, fold=TEMP_FOLD)
      )
      to_remove.append(
          crease_lib.Crease(front_crimp, front_ridge, fold=TEMP_FOLD)
      )
    if back_ridge is not None:
      to_remove.append(crease_lib.Crease(mid_ridge, back_ridge, fold=TEMP_FOLD))
      to_remove.append(
          crease_lib.Crease(back_crimp, back_ridge, fold=TEMP_FOLD)
      )

    filtered_creases: list[crease_lib.Crease] = []
    for c in template_creases:
      if c.fold != crease_lib.Fold.BORDER and any(
          c.is_within(c_, match_fold=False) for c_ in to_remove
      ):
        continue
      filtered_creases.append(c)
    filtered_creases.extend(ls_square_creases)
    filtered_creases.append(ls_axial_crease)
    return filtered_creases


@dataclasses.dataclass(frozen=True)
class BorderCrimp:
  """Represents a border crimp that can replace a level shifter."""

  side: grid.Direction
  crimp_hinge: crease_lib.Crease

  def __post_init__(self):
    if self.crimp_hinge.fold != crease_lib.Fold.BORDER:
      raise ValueError(
          f"Crimp hinge must be a border crease: {self.crimp_hinge.fold}."
      )

  @property
  def size(self) -> int:
    """Returns the size of the border crimp."""
    return 1

  def template_crimp_segments(
      self, solved_creases: list[crease_lib.Crease]
  ) -> list[crease_lib.Crease]:
    """Returns the crimp segments of the level shifter template."""
    return self.crimp_hinge.split_to_segments(solved_creases)

  def ls_template_creases(
      self, solved_creases: list[crease_lib.Crease], grid_size: int
  ) -> list[crease_lib.Crease]:
    """No-op that simulates SacrificialFlapTemplate protocol."""
    del solved_creases, grid_size  # Unused.
    return []

  def convert_to_ls_template(
      self, solved_creases: list[crease_lib.Crease], grid_size: int
  ) -> list[crease_lib.Crease]:
    """No-op that simulates SacrificialFlapTemplate protocol."""
    del grid_size  # Unused.
    return solved_creases

  def apply_level_shifter(
      self, template_creases: list[crease_lib.Crease], position: int
  ) -> list[crease_lib.Crease]:
    """No-op that simulates SacrificialFlapTemplate protocol."""
    del position  # Unused.
    return template_creases


@dataclasses.dataclass(frozen=True)
class ElevatedFlap:
  """Represents an elevated main flap with two corresponding sacrificial flaps.

  Attributes:
    main_flap: The PackingOutput rectangle representing the main flap.
    ls_flaps: SacrificialFlapTemplate objects associated with the main flap.
  """

  main_flap: shaping_utils.Rectangle
  ls_flaps: tuple[
      SacrificialFlapTemplate | BorderCrimp,
      SacrificialFlapTemplate | BorderCrimp,
  ]

  @property
  def max_shift(self) -> int:
    """Maximum level shift that can be applied to the main flap."""
    return min(ls_flap.size for ls_flap in self.ls_flaps)

  def axial_paths(
      self, template_creases: Collection[crease_lib.Crease]
  ) -> dict[LSPosition, list[crease_lib.Crease]]:
    """Finds all axial paths within the main flap that connect the LS flaps.

    Args:
      template_creases: Creases from a LS template solved base

    Returns:
      A dictionary mapping level shifter positions to a corresponding axial
      path.
    """
    flap_position_by_point = {}
    crimp_creases = [ls_flap.crimp_hinge for ls_flap in self.ls_flaps]
    for sac_flap_idx, ls_flap in enumerate(self.ls_flaps):
      crimp_segments = ls_flap.template_crimp_segments(template_creases)
      crimp_points = [c.p1 for c in crimp_segments] + [crimp_segments[-1].p2]
      for point_idx, crimp_point in enumerate(crimp_points):
        flap_position_by_point[crimp_point] = (sac_flap_idx, point_idx)

    flap_axials = []
    flap_positions_by_axial_id: dict[int, list[tuple[int, int]]] = {}
    for c in template_creases:
      if not self.main_flap.crease_is_within(c, strict=False):
        continue
      # We rely on elevation to identify axial creases.
      if c.elevation is None and c.fold != crease_lib.Fold.BORDER:
        continue
      if any(c.is_overlapping(crimp_crease) for crimp_crease in crimp_creases):
        continue

      matched_positions = []
      for p, flap_position in flap_position_by_point.items():
        if c.is_tip(p):
          matched_positions.append(flap_position)
      if matched_positions:
        flap_positions_by_axial_id[c.hash_value] = matched_positions
        flap_axials.append(c)

    axial_paths = utils.creases_to_paths(flap_axials)

    axial_paths_by_ls_position = {}
    for axial_path in axial_paths:
      idx_by_flap = {}
      for c in axial_path:
        if c.hash_value not in flap_positions_by_axial_id:
          continue
        for sac_flap_idx, point_idx in flap_positions_by_axial_id[c.hash_value]:
          if sac_flap_idx in idx_by_flap:
            raise ValueError(
                "Axial path has multiple connections to flap"
                f" {sac_flap_idx}: {axial_path}."
            )
          idx_by_flap[sac_flap_idx] = point_idx

      if len(idx_by_flap) != len(self.ls_flaps):
        continue

      ls_position = tuple(idx_by_flap[i] for i in range(len(self.ls_flaps)))
      axial_paths_by_ls_position[ls_position] = axial_path

    return axial_paths_by_ls_position

  def valid_positions(
      self, ls_positions: Collection[LSPosition]
  ) -> Iterator[LSPosition]:
    """Filters provided level shifter positions to only valid positions.

    Args:
      ls_positions: The candidate level shifter positions to filter.

    Yields:
      Valid LSPosition tuples from the input collection.
    """
    n = len(self.ls_flaps)
    # Remove max and min from viable options (no space for level shifter)
    # min is always 0
    idxs_per_flap = list(zip(*ls_positions, strict=True))
    mins = [min(idxs) for idxs in idxs_per_flap]
    maxes = [max(idxs) for idxs in idxs_per_flap]

    for flap_idx in ls_positions:
      # There must also be an adjacent axial (so we leave a gap of 1)
      if not all(mins[i] < flap_idx[i] < maxes[i] for i in range(n)):
        continue
      yield flap_idx

  def _adjust_main_flap_ridges(
      self,
      solved_creases: Sequence[crease_lib.Crease],
      position: LSPosition,
      axial_paths_by_index: dict[LSPosition, Sequence[crease_lib.Crease]],
  ) -> list[crease_lib.Crease]:
    """Adjusts the main flap ridges to accommodate deleted axials."""
    if not self.main_flap.circle_centers:
      # This is a river and thus there are no ridges to adjust.
      return list(solved_creases)
    axial_path = axial_paths_by_index.get(position)
    if axial_path is None:
      raise ValueError(f"No axial path found for position {position}.")
    elif len(axial_path) == 1:
      # The axial path does not intersect a ridge so they are not adjusted.
      return list(solved_creases)
    elif len(axial_path) > 2:
      raise NotImplementedError(
          "Axial path with more than two segments not supported yet:"
          f" {len(axial_path)} {axial_path}"
      )
    ridge_intersection = axial_path[0].vertex_with_crease(axial_path[-1])

    relevant_ridges = [
        c
        for c in self.main_flap.get_ridges()
        if c.point_status(ridge_intersection) != crease_lib.PointStatus.EXTERNAL
    ]
    if not relevant_ridges:
      raise ValueError(f"No relevant ridges found for position {position}.")
    if len(relevant_ridges) > 1:
      raise NotImplementedError(
          f"Multiple relevant ridges found for position {position}:"
          f" {relevant_ridges}"
      )
    relevant_ridge = relevant_ridges[0]

    ridge_segments = []
    non_ridge_creases = []
    for c in solved_creases:
      if (
          c.is_within(relevant_ridge, match_fold=False)
          and c.fold != crease_lib.Fold.BORDER
      ):
        ridge_segments.append(c)
      else:
        non_ridge_creases.append(c)

    deleted_points = []
    filtered_ridges = []
    for c in ridge_segments:
      if c.is_tip(ridge_intersection):
        if (
            any(c.is_tip(p) for p in self.main_flap.circle_centers)
            or c.intersects_border
        ):
          # Just flip the fold of a segment that is at the tip of the flap.
          filtered_ridges.append(dataclasses.replace(c, fold=c.fold.flip()))
        else:
          deleted_points.append(c.p1)
          deleted_points.append(c.p2)
      else:
        filtered_ridges.append(c)

    deleted_points = [
        p for p in deleted_points if not p.isclose(ridge_intersection)
    ]

    ridge_segments = filtered_ridges
    filtered_ridges = []
    for c in ridge_segments:
      tip_point = next((p for p in deleted_points if c.is_tip(p)), None)
      if tip_point is not None:
        other_point = c.p1 if c.p2.isclose(tip_point) else c.p2
        filtered_ridges.append(
            crease_lib.Crease(
                ridge_intersection,
                other_point,
                fold=c.fold,
            )
        )
      else:
        filtered_ridges.append(c)

    return non_ridge_creases + filtered_ridges

  def _adjust_main_flap_axials(
      self,
      solved_creases: Sequence[crease_lib.Crease],
      position: LSPosition,
      axial_paths_by_index: dict[LSPosition, Sequence[crease_lib.Crease]],
  ) -> list[crease_lib.Crease]:
    """Inverts the central axial and removes adjacent axials in the main flap."""
    axial_path = axial_paths_by_index.get(position)
    if axial_path is None:
      raise ValueError(f"No axial path found for position {position}.")
    x, y = position

    adjacent_positions = set()
    for x1, y1 in axial_paths_by_index:
      if x1 == x - 1 or x1 == x + 1 or y1 == y - 1 or y1 == y + 1:
        adjacent_positions.add((x1, y1))

    # At least one of the adjacent paths should exist
    if not adjacent_positions:
      raise ValueError(f"No adjacent positions found for position {position}.")
    elif len(adjacent_positions) > 2:
      raise ValueError(
          f"Multiple adjacent positions found for position {position}:"
          f" {adjacent_positions}."
      )

    to_remove = []
    for adjacent_position in adjacent_positions:
      to_remove.extend(axial_paths_by_index[adjacent_position])

    to_remove.extend(axial_path)
    to_remove = [c for c in to_remove if c.fold != crease_lib.Fold.BORDER]

    filtered_creases = []
    for c in solved_creases:
      if c.fold != crease_lib.Fold.BORDER and any(
          c.isclose(c_) for c_ in to_remove
      ):
        continue
      filtered_creases.append(c)

    for axial_segment in axial_path:
      match axial_segment.elevation:
        case 0:
          new_elevation = 1 + self.max_shift
        case 1:
          new_elevation = 0 - self.max_shift
        case _:
          raise ValueError(
              f"Invalid elevation for axial segment: {axial_segment.elevation}"
          )
      filtered_creases.append(
          dataclasses.replace(
              axial_segment,
              fold=axial_segment.fold.flip(),
              elevation=new_elevation,
          )
      )
    return filtered_creases

  def _adjust_main_flap(
      self, solved_creases: Sequence[crease_lib.Crease], position: LSPosition
  ) -> list[crease_lib.Crease]:
    """Changes creases in the main flap based on the provided position."""
    axial_paths_by_index = self.axial_paths(solved_creases)
    solved_creases = self._adjust_main_flap_ridges(
        solved_creases, position, axial_paths_by_index
    )
    solved_creases = self._adjust_main_flap_axials(
        solved_creases, position, axial_paths_by_index
    )
    return solved_creases

  def apply_level_shifter(
      self,
      template_creases: Sequence[crease_lib.Crease],
      ls_position: LSPosition,
  ) -> list[crease_lib.Crease]:
    """Applies level shifters to the main flap and both sacrificial flaps.

    Args:
      template_creases: The LS template creases to modify.
      ls_position: The level shifter position indices, one per sacrificial flap.

    Returns:
      A new list of creases with level shifters applied to all flaps.
    """
    creases = self._adjust_main_flap(template_creases, ls_position)
    for ls_flap, ls_idx in zip(self.ls_flaps, ls_position, strict=True):
      creases = ls_flap.apply_level_shifter(creases, ls_idx)
    return creases


class LevelShiftable(Protocol):
  """Protocol for objects that can have level shifters applied."""

  @property
  def ls_flaps(self) -> tuple[SacrificialFlapTemplate | BorderCrimp, ...]:
    ...

  def axial_paths(
      self, template_creases: Collection[crease_lib.Crease]
  ) -> dict[LSPosition, list[crease_lib.Crease]]:
    ...

  def valid_positions(
      self, ls_positions: Collection[LSPosition]
  ) -> Iterator[LSPosition]:
    ...

  def apply_level_shifter(
      self,
      template_creases: Sequence[crease_lib.Crease],
      ls_position: LSPosition,
  ) -> list[crease_lib.Crease]:
    ...

  @property
  def main_flap(self) -> shaping_utils.Rectangle:
    ...


@dataclasses.dataclass(frozen=True)
class MirroredElevatedFlapPair:
  """Groups an original and mirrored-copy ElevatedFlap as a single unit.

  Ensures that level shifters are applied at coordinated mirrored positions
  across both halves of a symmetrically-packed element.

  For each SF component, the mirroring reflects the index only when the crimp
  hinge is horizontal (points ordered by x, which reverses under horizontal
  flip). Vertical crimp hinges (points ordered by y) keep the same index.

  Attributes:
    original: The ElevatedFlap for the original (non-copy) element.
    copy: The ElevatedFlap for the mirrored _copy element.
  """

  original: ElevatedFlap
  copy: ElevatedFlap

  def _mirror_position(
      self,
      ls_position: LSPosition,
      original_positions: Collection[LSPosition],
      copy_positions: Collection[LSPosition],
  ) -> LSPosition:
    """Returns the corresponding LSPosition for the mirrored ElevatedFlap."""
    orig_per_flap = list(zip(*original_positions, strict=True))
    copy_per_flap = list(zip(*copy_positions, strict=True))
    mirrored = []
    for i, ls_flap in enumerate(self.original.ls_flaps):
      idx = ls_position[i]
      orig_min = min(orig_per_flap[i])
      copy_min = min(copy_per_flap[i])
      copy_max = max(copy_per_flap[i])
      if ls_flap.crimp_hinge.is_horizontal:
        mirrored.append(copy_max + copy_min - (idx - orig_min + copy_min))
      else:
        mirrored.append(idx - orig_min + copy_min)
    return tuple(mirrored)

  @property
  def ls_flaps(self) -> tuple[SacrificialFlapTemplate | BorderCrimp, ...]:
    """All the sacrificial flap templates for the mirrored pair."""
    return self.original.ls_flaps + self.copy.ls_flaps

  def axial_paths(
      self, template_creases: Collection[crease_lib.Crease]
  ) -> dict[LSPosition, list[crease_lib.Crease]]:
    """Returns the axial paths of the original flap."""
    original_paths = self.original.axial_paths(template_creases)
    copy_paths = self.copy.axial_paths(template_creases)
    if len(original_paths) != len(copy_paths):
      raise ValueError(
          "Original and copy axial_paths have different lengths:"
          f" {len(original_paths)} vs {len(copy_paths)}."
      )
    result = {}
    for pos, original_creases in original_paths.items():
      mirrored_pos = self._mirror_position(pos, original_paths, copy_paths)
      if mirrored_pos in copy_paths:
        result[pos] = original_creases + copy_paths[mirrored_pos]
    return result

  def valid_positions(
      self, ls_positions: Collection[LSPosition]
  ) -> Iterator[LSPosition]:
    yield from self.original.valid_positions(ls_positions)

  def apply_level_shifter(
      self,
      template_creases: Sequence[crease_lib.Crease],
      ls_position: LSPosition,
  ) -> list[crease_lib.Crease]:
    """Applies level shifters to the mirrored pair."""
    original_paths = self.original.axial_paths(template_creases)
    copy_paths = self.copy.axial_paths(template_creases)
    mirrored_pos = self._mirror_position(
        ls_position, original_paths, copy_paths
    )
    creases = self.original.apply_level_shifter(template_creases, ls_position)
    creases = self.copy.apply_level_shifter(creases, mirrored_pos)
    return creases

  @property
  def main_flap(self) -> shaping_utils.Rectangle:
    """The main flap of the 'original' ElevatedFlap."""
    return self.original.main_flap


def _crimp_direction_to_side(
    crimp_direction: packing_types.SidePosition,
) -> grid.Direction:
  match crimp_direction:
    case "bottom":
      return grid.Direction.SOUTH
    case "top":
      return grid.Direction.NORTH
    case "right":
      return grid.Direction.EAST
    case "left":
      return grid.Direction.WEST
    case _:
      typing.assert_never(crimp_direction)


def _sf_clipping_rect(
    s_flaps: Sequence[packing_types.SacrificialFlap],
    grid_size: int,
) -> tuple[float, float, float, float]:
  """Computes unnormalized clipping bounds from sacrificial flap placements.

  Starting from the entire page, each SF constrains its parallel axis:
  horizontal SFs (top/bottom crimp) constrain x-range, vertical SFs
  (left/right crimp) constrain y-range. When exactly two SFs share the same
  orientation, the perpendicular gap between them further constrains the
  other axis.

  Args:
    s_flaps: The sacrificial flaps whose placements define the clipping region.
    grid_size: The grid resolution used to normalize rectangles.

  Returns:
    A clip box tuple (xmin, ymin, xmax, ymax) in unnormalized grid coordinates.
  """
  xmin, ymin = 0, 0
  xmax, ymax = grid_size, grid_size

  horizontal_sfs = []
  vertical_sfs = []

  for sf in s_flaps:
    p = sf.placement
    match sf.crimp_direction:
      case "top" | "bottom":
        horizontal_sfs.append(p)
        xmin = max(xmin, p.offset[0])
        xmax = min(xmax, p.offset[0] + p.size[0])
      case "left" | "right":
        vertical_sfs.append(p)
        ymin = max(ymin, p.offset[1])
        ymax = min(ymax, p.offset[1] + p.size[1])
      case "top_left" | "top_right" | "bottom_left" | "bottom_right":
        raise ValueError(
            f"Corner SFs are not supported in _sf_clipping_rect: {sf}"
        )

  # When exactly two parallel SFs exist, the gap between them constrains
  # the perpendicular axis.
  if len(horizontal_sfs) == 2:
    p1 = horizontal_sfs[0]
    p2 = horizontal_sfs[1]
    y_coords = sorted([
        p1.offset[1],
        p1.offset[1] + p1.size[1],
        p2.offset[1],
        p2.offset[1] + p2.size[1],
    ])
    ymin = max(ymin, y_coords[1])
    ymax = min(ymax, y_coords[2])
  if len(vertical_sfs) == 2:
    p1 = vertical_sfs[0]
    p2 = vertical_sfs[1]
    x_coords = sorted([
        p1.offset[0],
        p1.offset[0] + p1.size[0],
        p2.offset[0],
        p2.offset[0] + p2.size[0],
    ])
    xmin = max(xmin, x_coords[1])
    xmax = min(xmax, x_coords[2])

  return xmin, ymin, xmax, ymax


def _validate_river_sf_group(sfs: Collection[packing_types.SacrificialFlap]):
  """Checks that a collection of river sacrificial flaps forms a valid group.

  Args:
    sfs: A collection of river sacrificial flaps

  Raises:
    ValueError: If there is something wrong with the sf pair.
  """
  if len(orientations := {sf.orientation for sf in sfs}) != 1:
    raise ValueError(
        f"All sfs should have the same orientation: {orientations}"
    )
  if len(element_names := {sf.shifted_element_name for sf in sfs}) != 1:
    raise ValueError(
        f"All sfs should have the same shifted element name: {element_names}"
    )
  for sf1, sf2 in itertools.combinations(sfs, 2):
    if sf1 == sf2:
      raise ValueError("No two sfs should be the same flap.")
    if sf1.shifted_element != sf2.shifted_element:
      raise ValueError("All sfs should have the same shifted element.")
    if not isinstance(sf1.shifted_element, river_lib.PackingRiver):
      raise ValueError("Shifted element should be a river.")


def _get_opposite_border_crimp(
    ls_flap: SacrificialFlapTemplate,
    main_flap_rect: shaping_utils.Rectangle,
) -> BorderCrimp:
  """Adds a BorderCrimp if there's only one LS flap and an opposite border hinge.

  Args:
    ls_flap: The opposite LS flap, expected to contain a single
      SacrificialFlapTemplate.
    main_flap_rect: The main flap rectangle.

  Returns:
    A BorderCrimp on the opposite side.

  Raises:
    ValueError: If no opposite border hinge can be used.
  """

  match ls_flap.side:
    case grid.Direction.NORTH:
      edge = main_flap_rect.edges[0]
    case grid.Direction.EAST:
      edge = main_flap_rect.edges[1]
    case grid.Direction.SOUTH:
      edge = main_flap_rect.edges[2]
    case grid.Direction.WEST:
      edge = main_flap_rect.edges[3]
    case _:
      typing.assert_never(ls_flap.side)
  if edge.is_overlapping_border:
    return BorderCrimp(
        side=ls_flap.side.flip(),
        crimp_hinge=dataclasses.replace(edge, fold=crease_lib.Fold.BORDER),
    )
  raise ValueError(
      "There is only one sacrifical flap and no opposite border hinge: "
      f"{ls_flap}"
  )


def get_elevated_flaps(
    flap_rectangles: list[shaping_utils.Rectangle],
    sacrificial_flaps: list[packing_types.SacrificialFlap],
    grid_size: int,
    group_mirrored: bool = True,
    symmetric_pairs: dict[str, str] | None = None,
) -> list[LevelShiftable]:
  """Constructs ElevatedFlap objects from packing outputs.

  Groups sacrificial flaps by their base flap name and pairs them with
  the corresponding main flap rectangle.

  Args:
    flap_rectangles: The list of all packing output rectangles.
    sacrificial_flaps: The sacrificial flaps to associate with main flaps.
    grid_size: The grid resolution used to normalize rectangles.
    group_mirrored: If True, wraps mirrored original/copy pairs into
      MirroredElevatedFlapPair units.
    symmetric_pairs: Optional mapping from flap name to its symmetric partner
      name, used for pairing non-copy symmetric flaps when group_mirrored is
      True.

  Returns:
    A list of LevelShiftable objects, one per base flap with sacrificial flaps.

  Raises:
    ValueError: If a base flap name is not found in flap_rectangles, or if
      the number of sacrificial flaps per base flap is not exactly 2.
  """
  elevated_flaps = []
  sacrificial_flaps_by_group = collections.defaultdict(list)
  for sf in sacrificial_flaps:
    if sf.is_corner:
      continue
    group_key = sf.shifted_element_name
    if not sf.is_l_shaped and sf.polygon_name.endswith(
        packing_types.COPY_SUFFIX
    ):
      group_key += packing_types.COPY_SUFFIX

    sacrificial_flaps_by_group[group_key].append(sf)

  rectangles_by_name = {r.name: r for r in flap_rectangles}

  for group_key, s_flaps in sacrificial_flaps_by_group.items():
    element_name = s_flaps[0].shifted_element_name
    clip_box = _sf_clipping_rect(s_flaps, grid_size)

    if isinstance(s_flaps[0].shifted_element, river_lib.PackingRiver):
      _validate_river_sf_group(s_flaps)
      river = s_flaps[0].shifted_element
      # Validate that the computed clipping box is within the river rectangles.
      for river_rect in river.rectangles:
        if river_rect.clip(*clip_box) is not None:
          break
      else:
        raise ValueError(
            "Computed main_flap_rect is not within river rectangles:"
            f" {clip_box}"
        )
      main_flap_rect = shaping_utils.Rectangle.from_clip_box(
          name=group_key,
          clip_box=clip_box,
      )
    else:
      if element_name not in rectangles_by_name:
        raise ValueError(
            f"Main flap {element_name} not found in flap_rectangles."
        )
      main_flap_rect = rectangles_by_name[element_name].clip(*clip_box)
      if element_name != group_key:
        object.__setattr__(main_flap_rect, "name", group_key)
    print(main_flap_rect)

    if main_flap_rect is None:
      raise ValueError(
          f"Clipped main flap does not overlap with paper: {group_key}"
      )
    main_flap_rect = main_flap_rect.normalize_rectangle(grid_size)

    ls_flaps_list = []
    for sf in s_flaps:
      side = _crimp_direction_to_side(sf.crimp_direction)
      if sf.polygon_name not in rectangles_by_name:
        raise ValueError(
            f"Sacrificial flap {sf.polygon_name} not found in flap_rectangles."
        )
      ls_flap_rect = rectangles_by_name[sf.polygon_name].normalize_rectangle(
          grid_size
      )
      ls_flaps_list.append(
          SacrificialFlapTemplate(
              side=side,
              rectangle=ls_flap_rect,
              river_sf_info=sf.placement.river_sf_info,
          )
      )

    if len(ls_flaps_list) > 2:
      raise NotImplementedError(
          f"Multiple LS flaps per side are not supported yet: {group_key}"
      )
    elif len(ls_flaps_list) == 1:
      # Check if there is an opposite border hinge that can be used instead
      border_crimp = _get_opposite_border_crimp(
          ls_flaps_list[0], main_flap_rect
      )
      ls_flaps_list.append(border_crimp)

    elevated_flaps.append(
        ElevatedFlap(
            main_flap=main_flap_rect,
            ls_flaps=tuple(ls_flaps_list),
        )
    )
  if group_mirrored:
    return group_mirrored_elevated_flaps(elevated_flaps, symmetric_pairs)
  return elevated_flaps


def group_mirrored_elevated_flaps(
    elevated_flaps: list[ElevatedFlap],
    symmetric_pairs: dict[str, str] | None = None,
) -> list[LevelShiftable]:
  """Groups mirrored elevated flap pairs into a single LevelShiftable unit.

  Pairs are identified by two conventions:
    1. River pairs: ElevatedFlaps sharing the same main_flap.name (river name).
    2. Symmetric sticks: names linked via symmetric_pairs (e.g. left_arm ↔
       right_arm).

  Args:
    elevated_flaps: The elevated flaps to group.
    symmetric_pairs: Optional mapping from flap name to its symmetric partner.

  Returns:
    A list of LevelShiftable objects where mirrored pairs are combined.
  """
  if symmetric_pairs is None:
    symmetric_pairs = {}
  if any(k == v for k, v in symmetric_pairs.items()):
    raise ValueError(
        "Symmetric pairs must be distinct: found "
        f"{[k for k, v in symmetric_pairs.items() if k == v]}."
    )

  groups: dict[str, list[ElevatedFlap]] = collections.defaultdict(list)
  for ef in elevated_flaps:
    name = ef.main_flap.name.removesuffix("_copy")
    partner = symmetric_pairs.get(name)
    if partner and any(e.main_flap.name == partner for e in elevated_flaps):
      group_key = min(name, partner)
    else:
      group_key = name
    groups[group_key].append(ef)

  result: list[LevelShiftable] = []
  for group in groups.values():
    if len(group) == 1:
      result.append(group[0])
    elif len(group) == 2:
      group.sort(key=lambda ef: ef.main_flap.x_start)
      result.append(MirroredElevatedFlapPair(original=group[0], copy=group[1]))
    else:
      raise ValueError(
          f"Expected 1 or 2 elevated flaps per group, got {len(group)}."
      )

  return result


def make_preassignment_cp(
    ls_flaps: Collection[SacrificialFlapTemplate],
    grid_size: int,
) -> crease_pattern.CreasePattern:
  """Makes a preassignment CP that constrains the solver for LS flaps.

  Creates a CreasePattern containing the crimp hinges and opposite hinges
  from each sacrificial flap template, to be used as solver preassignments.

  Args:
    ls_flaps: The sacrificial flap templates to extract hinges from.
    grid_size: The grid resolution for the resulting CreasePattern.

  Returns:
    A CreasePattern containing the preassignment creases.
  """
  crimp_hinges = [ls_flap.crimp_hinge for ls_flap in ls_flaps]
  opposite_hinges = []
  for ls_flap in ls_flaps:
    if not any(c for c in crimp_hinges if c.isclose(ls_flap.opposite_hinge)):
      opposite_hinges.append(
          dataclasses.replace(
              ls_flap.opposite_hinge, fold=crease_lib.Fold.TRIANGLE
          )
      )

  return crease_pattern.CreasePattern(
      creases=shaping_utils.restrict_creases_to_paper(
          opposite_hinges + crimp_hinges
      ),
      grid_size=grid_size,
      light_mode=True,
  )


def get_sorted_crimp_points(
    ls_flap: SacrificialFlapTemplate,
    crimp_creases_segments: list[crease_lib.Crease],
) -> list[point_lib.Point]:
  """Extracts unique points from crimp crease segments and sorts them along the hinge.

  Points are sorted by their norm to ensure a consistent spatial
  order along the axis-aligned crimp hinge (e.g., top-to-bottom or
  left-to-right). The list is reversed if it ends with a flap corner but
  does not start with one, to prefer corner-first ordering.

  Args:
    ls_flap: The sacrificial flap template.
    crimp_creases_segments: A list of crease segments along the crimp hinge.

  Returns:
    A list of unique points from the crimp crease segments, sorted along the
    crimp hinge.
  """
  if not crimp_creases_segments:
    return []
  crimp_points = set(
      itertools.chain.from_iterable(
          (c.p1, c.p2) for c in crimp_creases_segments
      )
  )
  # Sort points by norm.
  sorted_points = sorted(list(crimp_points), key=lambda p: p.norm)

  # Ensure points are ordered along the edge starting from a corner if possible.
  is_first_point_corner = any(
      sorted_points[0].isclose(c) for c in ls_flap.rectangle.corners
  )
  is_last_point_corner = any(
      sorted_points[-1].isclose(c) for c in ls_flap.rectangle.corners
  )
  if not is_first_point_corner and is_last_point_corner:
    sorted_points.reverse()
  return sorted_points


def get_orthogonal_creases(
    ls_flap: SacrificialFlapTemplate,
    point: point_lib.Point,
    split_creases: list[crease_lib.Crease],
) -> list[crease_lib.Crease]:
  """Returns creases orthogonal to crimp hinge passing through a point.

  This function filters `split_creases` to find segments that are:
  1. Perpendicular to the sacrificial flap's crimp hinge.
  2. Within the bounds of the sacrificial flap rectangle.
  3. Lie on the line that passes through `point` and is orthogonal to the
     crimp hinge.

  The returned list of creases is sorted by distance from `point`.

  Args:
    ls_flap: The sacrificial flap template containing the crimp hinge.
    point: The point that defines the orthogonal line.
    split_creases: A list of candidate crease segments to filter.

  Returns:
    A list of orthogonal crease segments passing through `point`, sorted by
    distance from `point`.
  """
  orthogonal_creases = []
  is_vertical_hinge = ls_flap.crimp_hinge.is_vertical

  for c in split_creases:
    # Check if crease `c` lies on the line passing through `point` and
    # orthogonal to `crimp_hinge`.
    if is_vertical_hinge:  # `c` must be horizontal
      if not c.is_horizontal:
        continue
      is_on_line = math.isclose(c.p1.y, point.y)
    else:  # `c` must be vertical
      if not c.is_vertical:
        continue
      is_on_line = math.isclose(c.p1.x, point.x)

    if not is_on_line:
      continue

    if not ls_flap.rectangle.crease_is_within(c, strict=False):
      continue

    orthogonal_creases.append(c)

  # Sort creases by proximity to the point on crimp hinge,
  # so they are ordered from crimp hinge outwards.
  orthogonal_creases.sort(
      key=lambda c: min(
          point_lib.distance(point, c.p1),
          point_lib.distance(point, c.p2),
      )
  )
  return orthogonal_creases


def apply_ls_templates_to_result(
    solve_result: hinge_lib.SolverResult,
    elevated_flaps: Collection[LevelShiftable],
    grid_size: int,
) -> hinge_lib.SolverResult:
  """Applies level shifter templates to a single solver result.

  Converts each sacrificial flap in the solver result into an LS template,
  replacing the original creases.

  Args:
    solve_result: The solver result to transform. Must have base_type NORMAL.
    elevated_flaps: The elevated flaps whose sacrificial flaps will be converted
      to LS templates.
    grid_size: The grid resolution.

  Returns:
    A new SolverResult with LS template creases and base_type LS_TEMPLATE.

  Raises:
    ValueError: If solve_result.base_type is not NORMAL.
  """
  if solve_result.base_type != hinge_lib.BaseType.NORMAL:
    raise ValueError(
        "Cannot apply LS templates to a non-normal base type result:"
        f" {solve_result.base_type}."
    )

  # Apply template conversions to sacrificial flaps
  ls_template_creases = solve_result.crease_pattern.creases
  for elevated_flap in elevated_flaps:
    for ls_flap in elevated_flap.ls_flaps:
      ls_template_creases = ls_flap.convert_to_ls_template(
          ls_template_creases, grid_size
      )

  ls_template_cp = crease_pattern.CreasePattern(
      creases=ls_template_creases,
      grid_size=grid_size,
      light_mode=True,
  )

  return dataclasses.replace(
      solve_result,
      crease_pattern=ls_template_cp,
      base_type=hinge_lib.BaseType.LS_TEMPLATE,
  )


def solve_with_ls_templates(
    solver_config: hinge_lib.SolverConfig,
    packing_cp: crease_pattern.CreasePattern,
    elevated_flaps: Collection[LevelShiftable],
    plot_stages: bool = False,
) -> list[hinge_lib.SolverResult]:
  """Solves a packing CP with constraints and applies level shifter templates.

  Constructs preassignment constraints from the elevated flaps, solves the
  packing crease pattern, and converts each valid result into an LS template.

  Args:
    solver_config: The configuration for the solver. Must not have
      preassignments already set.
    packing_cp: The crease pattern to solve.
    elevated_flaps: The elevated flaps specifying sacrificial flap geometry.
    plot_stages: If True, plot intermediate CPs during solving.

  Returns:
    A list of SolverResult objects with LS templates applied.

  Raises:
    ValueError: If solver_config.preassignments is already set or
      elevated_flaps is empty.
  """

  def _plot_cp(cp):
    if not plot_stages:
      return
    visualize.draw_cp(
        cp, grid_size=solver_config.grid_size, annotate_vertices=False
    )

  if solver_config.preassignments is not None:
    raise ValueError("Preassignments should not already be set.")
  if not elevated_flaps:
    raise ValueError("elevated_flaps must not be empty.")
  if not solver_config.allow_misaligned_symmetric_hinges:
    logging.warning(
        "SolverConfig.allow_misaligned_symmetric_hinges is False, this is not"
        " recommended with level shifter solving"
    )

  grid_size = solver_config.grid_size

  preassignment_cp = make_preassignment_cp(
      [
          sf
          for sf in itertools.chain.from_iterable(
              e.ls_flaps for e in elevated_flaps
          )
          if isinstance(sf, SacrificialFlapTemplate)
      ],
      grid_size,
  )

  _plot_cp(preassignment_cp)

  original_solution_callback = solver_config.solution_callback

  def template_callback(solve_result):
    _plot_cp(solve_result.crease_pattern)

    ls_template_result = apply_ls_templates_to_result(
        solve_result, elevated_flaps, grid_size
    )
    _plot_cp(ls_template_result.crease_pattern)
    if not solver_config.is_valid_crease_pattern(
        ls_template_result.crease_pattern
    ):
      return None
    return original_solution_callback(ls_template_result)

  solver_config = dataclasses.replace(
      solver_config,
      preassignments=preassignment_cp,
      solution_callback=template_callback,
  )

  solver = solver_config.make_from_cp(packing_cp)
  return solver.solve()


def apply_all_level_shifter_combos(
    ls_template_result: hinge_lib.SolverResult,
    elevated_flaps: Collection[LevelShiftable],
    grid_size: int,
    require_all: bool = True,
    is_valid_cp: Callable[
        [crease_pattern.CreasePattern], bool
    ] = lambda cp: True,
) -> Iterator[hinge_lib.SolverResult]:
  """Applies level shifters to an LS template result.

  Tries all valid combinations of level shifter positions across the
  elevated flaps. Positions are sorted by total crease length (shortest
  first), so earlier yielded results have more axials removed.

  Args:
    ls_template_result: The LS template solver result. Must have base_type
      LS_TEMPLATE.
    elevated_flaps: The elevated flaps to apply level shifters to.
    grid_size: The grid resolution.
    require_all: If True, raises ValueError when any elevated flap has no valid
      flat-foldable position.
    is_valid_cp: A callable that validates the resulting CreasePattern.

  Yields:
    SolverResult objects with level shifters applied and base_type
    LEVEL_SHIFTED.

  Raises:
    ValueError: If ls_template_result.base_type is not LS_TEMPLATE, or if
      require_all is True and any flap has no valid position.
  """
  if ls_template_result.base_type != hinge_lib.BaseType.LS_TEMPLATE:
    raise ValueError(
        "Can only apply level shifters to LS template bases: "
        f" {ls_template_result.base_type}."
    )
  valid_positions_per_flap = []
  possible_elevated_flaps = []
  for elevated_flap in elevated_flaps:
    creases = ls_template_result.crease_pattern.creases
    ff_positions = []
    total_lengths = []
    for position in elevated_flap.valid_positions(
        elevated_flap.axial_paths(creases)
    ):
      try:
        ls_creases = elevated_flap.apply_level_shifter(creases, position)
        if crease_pattern.CreasePattern(
            creases=ls_creases, grid_size=grid_size, light_mode=True
        ).local_flat_foldability():
          ff_positions.append(position)
          total_lengths.append(sum(c.length for c in ls_creases))
      except (ValueError, NotImplementedError):
        logging.warning(
            "Failed to apply level shifter to ls template result for position"
            " %s: %s",
            position,
            traceback.format_exc(),
        )
    if ff_positions:
      possible_elevated_flaps.append(elevated_flap)
      # Sort positions by total crease length (prefer shorter total lengths).
      # The shorter the total length, the more axial creases have been removed.
      ff_positions = [
          position for _, position in sorted(zip(total_lengths, ff_positions))
      ]
      valid_positions_per_flap.append(ff_positions)
    elif require_all:
      raise ValueError(
          f"No flat-foldable positions found for flap: {elevated_flap}."
      )

  if not possible_elevated_flaps:
    logging.warning("No flat-foldable positions found for ls template result.")
    return

  for positions in itertools.product(*valid_positions_per_flap):
    try:
      ls_creases = ls_template_result.crease_pattern.creases
      for position, elevated_flap in zip(
          positions, possible_elevated_flaps, strict=True
      ):
        ls_creases = elevated_flap.apply_level_shifter(ls_creases, position)

      ls_cp = crease_pattern.CreasePattern(
          creases=ls_creases,
          grid_size=grid_size,
          light_mode=False,
          merge_trivial=True,
      )
      if not is_valid_cp(ls_cp):
        raise ValueError(
            "Level shifter applied to ls template result is not flat-foldable."
        )
      yield dataclasses.replace(
          ls_template_result,
          crease_pattern=ls_cp,
          base_type=hinge_lib.BaseType.LEVEL_SHIFTED,
      )
    except (ValueError, NotImplementedError):
      logging.warning(
          "Failed to apply level shifter to ls template result: %s",
          traceback.format_exc(),
      )
      continue


def solve_with_max_length_level_shifters(
    solver_config: hinge_lib.SolverConfig,
    packing_cp: crease_pattern.CreasePattern,
    elevated_flaps: Collection[LevelShiftable],
    plot_stages: bool = False,
    require_all: bool = True,
) -> list[hinge_lib.SolverResult]:
  """Solves a packing cp with constraints and applies maximal level shifters.

  This function first solves for level shifter templates, then for each
  flat-foldable template, it applies level shifters, selecting the
  combination of level shifter positions that results in the minimum total
  crease length.

  Args:
    solver_config: The configuration for the solver.
    packing_cp: The crease pattern to solve.
    elevated_flaps: The elevated flaps to apply level shifters to.
    plot_stages: If True, plot intermediate cps during solving.
    require_all: If True, require all elevated flaps to have at least one
      flat-foldable level shifter position for a solution to be valid.

  Returns:
    A list of SolverResult objects with maximal level shifters applied.
  """

  original_solution_callback = solver_config.solution_callback

  def apply_ls_callback(
      ls_template_result: hinge_lib.SolverResult,
  ) -> hinge_lib.SolverResult | None:
    try:
      ls_applied_result = next(
          apply_all_level_shifter_combos(
              ls_template_result,
              elevated_flaps,
              solver_config.grid_size,
              require_all=require_all,
              is_valid_cp=solver_config.is_valid_crease_pattern,
          ),
          None,
      )
    except ValueError:
      ls_applied_result = None
    if ls_applied_result is None:
      return None
    return original_solution_callback(ls_applied_result)

  # Inject the callback to apply level shifters to the results.

  solver_config = dataclasses.replace(
      solver_config,
      solution_callback=apply_ls_callback,
  )

  return solve_with_ls_templates(
      solver_config,
      packing_cp,
      elevated_flaps,
      plot_stages,
  )
