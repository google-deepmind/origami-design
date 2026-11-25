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

"""Utilities for solving level-shifter templates and placing level shifters."""

import collections
from collections.abc import Collection, Iterator, Sequence
import dataclasses
import functools
import itertools
import logging
import typing
from typing import NamedTuple, Protocol

from bp_pipeline import rectangle as shaping_utils
from bp_pipeline.crease_pattern import crease as crease_lib
from bp_pipeline.crease_pattern import crease_pattern
from bp_pipeline.crease_pattern import point as point_lib
from bp_pipeline.crease_pattern import visualize
from bp_pipeline.packing import grid
from bp_pipeline.packing import river as river_lib
from bp_pipeline.packing import types as packing_types
from bp_pipeline.solving import hinges as hinge_lib
from bp_pipeline.solving import utils

TEMP_FOLD = crease_lib.Fold.TRIANGLE
LocalIndex = int
LSPosition = tuple[LocalIndex, LocalIndex]


def get_ls_position_from_percent(
    valid_positions: Sequence[LSPosition], position_percent: float
) -> LSPosition:
  """Returns the level-shifter position corresponding to `position_percent`."""
  if not valid_positions:
    raise ValueError('No valid positions provided.')
  if not 0.0 <= position_percent <= 1.0:
    raise ValueError('position_percent must be between 0.0 and 1.0.')

  index = round(position_percent * (len(valid_positions) - 1))
  return valid_positions[index]


def _get_outer_fold(creases: list[crease_lib.Crease]) -> crease_lib.Fold:
  """Returns the fold type of the non-border end of `creases`."""
  if creases[0].intersects_border:
    return creases[-1].fold
  if creases[-1].intersects_border:
    return creases[0].fold
  if creases[0].fold != creases[-1].fold:
    raise ValueError(
        'First and last creases have different folds:'
        f' {creases[0]}, {creases[-1]}'
    )
  return creases[0].fold


def _replace_matching_creases(
    creases: list[crease_lib.Crease],
    replacement_creases: list[crease_lib.Crease],
) -> list[crease_lib.Crease]:
  """Replaces non-border creases matching `replacement_creases`."""
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
  """Handles templating and placement of a level-shifter sacrificial flap.

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
    """Size of the sacrificial shift step in grid units."""
    return 1

  def __post_init__(self):
    if self.rectangle.radius is None or self.rectangle.radius > 1:
      raise ValueError('Input rectangles should be normalized.')

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
    return round(dist * grid_size)

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
    return round(start_diff / seg_len), round(end_diff / seg_len)

  def _template_central_ridge_segments(
      self, solved_creases: list[crease_lib.Crease]
  ) -> list[crease_lib.Crease]:
    """Returns segments corresponding to this LS-template's central ridge."""
    central_ridge = _get_central_ridge(self.rectangle)
    if central_ridge is None:
      raise ValueError('Level shifter flaps must have a central ridge.')

    solved_ridge_segments = [
        c
        for c in solved_creases
        if c.is_within(central_ridge, match_fold=False)
    ]
    if not solved_ridge_segments:
      raise ValueError(
          'No creases found within the central ridge; did you solve with'
          ' correctly preassigned creases?'
      )
    solved_ridge_segments.sort(key=_crease_norm_key)

    fold = _get_outer_fold(solved_ridge_segments).flip()
    return [
        dataclasses.replace(c, fold=fold, elevation=None)
        for c in solved_ridge_segments
    ]

  def template_crimp_segments(
      self, solved_creases: Collection[crease_lib.Crease], inside: bool = True
  ) -> list[crease_lib.Crease]:
    """Returns the segments corresponding to the crimped hinge."""
    if not inside:
      raise NotImplementedError('Only inside crimping is implemented.')
    matched = [
        c
        for c in solved_creases
        if c.is_within(self.crimp_hinge, match_fold=False)
    ]
    if not matched:
      raise ValueError(
          'No creases found within the crimp edge; did you solve with'
          ' correctly preassigned creases?'
      )
    matched.sort(key=_crease_norm_key)

    fold = _get_outer_fold(matched)
    return [
        dataclasses.replace(c, fold=fold, elevation=None) for c in matched
    ]

  def _align_crimp_and_ridge(
      self,
      crimp_creases: Sequence[crease_lib.Crease],
      central_ridge_creases: Sequence[crease_lib.Crease],
  ) -> list[tuple[crease_lib.Crease, crease_lib.Crease | None]]:
    """Aligns adjacent crimp and ridge creases by padding ridges with Nones."""
    num_pad_start, num_pad_end = self._get_crimp_ridge_padding(
        crimp_creases, central_ridge_creases
    )
    padding_start = [None] * num_pad_start
    padding_end = [None] * num_pad_end
    padded_ridges = [*padding_start, *central_ridge_creases, *padding_end]
    if len(padded_ridges) != len(crimp_creases):
      raise ValueError(
          'Number of ridges + padding must equal crimps:'
          f' {len(padded_ridges)} != {len(crimp_creases)}'
          f' (pads: {num_pad_start}, {num_pad_end}).'
      )
    return list(zip(crimp_creases, padded_ridges, strict=True))

  def _aligned_crimp_and_ridge_points(
      self,
      crimp_creases: Sequence[crease_lib.Crease],
      central_ridge_creases: Sequence[crease_lib.Crease],
  ) -> list[CrimpRidgePointPair]:
    """Returns paired crimp and ridge points along the hinge."""
    paired_points: list[CrimpRidgePointPair] = []
    aligned_creases = self._align_crimp_and_ridge(
        crimp_creases, central_ridge_creases
    )
    crimp_start, ridge_start = aligned_creases[0]
    if ridge_start is None:
      paired_points.append(CrimpRidgePointPair(crimp_start.p1, None))
    else:
      paired_points.append(CrimpRidgePointPair(crimp_start.p1, ridge_start.p1))

    for (c_start, r_start), (c_end, r_end) in itertools.pairwise(
        aligned_creases
    ):
      crimp_point = c_start.p2
      if r_start is not None and r_end is not None:
        ridge_point = r_start.p2
      elif r_start is not None:
        ridge_point = r_start.p2
      elif r_end is not None:
        ridge_point = r_end.p1
      else:
        ridge_point = None
      del c_end
      paired_points.append(CrimpRidgePointPair(crimp_point, ridge_point))

    crimp_end, ridge_end = aligned_creases[-1]
    if ridge_end is None:
      paired_points.append(CrimpRidgePointPair(crimp_end.p2, None))
    else:
      paired_points.append(CrimpRidgePointPair(crimp_end.p2, ridge_end.p2))

    return paired_points

  def _template_inverted_axials(
      self,
      *,
      solved_creases: Collection[crease_lib.Crease],
      crimp_creases: Sequence[crease_lib.Crease],
      central_ridge_creases: Sequence[crease_lib.Crease],
  ) -> list[crease_lib.Crease]:
    """Returns inverted axial creases between the crimp and central ridge."""
    matched_axial_segments = []
    aligned_points = self._aligned_crimp_and_ridge_points(
        crimp_creases, central_ridge_creases
    )
    for crimp_point, ridge_point in aligned_points:
      if ridge_point is None or ridge_point.is_border:
        continue
      axial_segment = crease_lib.Crease(
          crimp_point, ridge_point, fold=TEMP_FOLD
      )
      matched_axial_segments.append(
          next(c for c in solved_creases if c.isclose(axial_segment))
      )

    fold = crimp_creases[0].fold
    inverted_axial_segments = []
    _, start_ridge_point = aligned_points[0]
    iterable = (
        reversed(matched_axial_segments)
        if start_ridge_point is not None
        else matched_axial_segments
    )
    for c in iterable:
      fold = fold.flip()
      elev = c.elevation if c.elevation is not None else 0
      inverted_axial_segments.append(
          dataclasses.replace(c, fold=fold, elevation=elev)
      )
    if start_ridge_point is not None:
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
    if valid_pair is None or valid_pair.ridge is None:
      return []

    crimp_p, ridge_p = valid_pair.crimp, valid_pair.ridge
    step_x = (crimp_p.x - ridge_p.x) / num_steps_to_crimp
    step_y = (crimp_p.y - ridge_p.y) / num_steps_to_crimp
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
      current_fold = ridge_fold.flip() if k % 2 != 0 else ridge_fold
      ext_start = min(k, max_ext_start)
      ext_end = min(k, max_ext_end)
      shift_x = k * step_x
      shift_y = k * step_y
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
    """Returns all new creases of the level-shifter template."""
    crimp_creases = self.template_crimp_segments(solved_creases)
    central_ridge_creases = self._template_central_ridge_segments(
        solved_creases
    )
    inverted_axials = self._template_inverted_axials(
        solved_creases=solved_creases,
        crimp_creases=crimp_creases,
        central_ridge_creases=central_ridge_creases,
    )
    propagated_creases = self._template_crimp_propagation(
        central_ridge_creases=central_ridge_creases,
        crimp_creases=crimp_creases,
        grid_size=grid_size,
    )
    return (
        crimp_creases
        + central_ridge_creases
        + inverted_axials
        + propagated_creases
    )

  def convert_to_ls_template(
      self, solved_creases: list[crease_lib.Crease], grid_size: int
  ) -> list[crease_lib.Crease]:
    """Applies this level-shifter template to `solved_creases`."""
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
    """Returns the diamond creases and central spine of a `2x2` LS square."""
    mid_crimp, mid_ridge = mid_points
    if mid_ridge is None:
      raise ValueError('Level-shifter midpoint must have a central ridge.')
    _, front_corner = front_points
    _, back_corner = back_points

    cr_dx = mid_ridge.x - mid_crimp.x
    cr_dy = mid_ridge.y - mid_crimp.y
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
        mid_crimp, opposite_corner, fold=fold.flip(), elevation=0
    )

  def apply_level_shifter(
      self, template_creases: list[crease_lib.Crease], position: int
  ) -> list[crease_lib.Crease]:
    """Inserts a `2x2` level-shifter diamond at 1-indexed `position`."""
    if position <= 0:
      raise ValueError(f'Position must be positive: {position}')

    crimp_creases = self.template_crimp_segments(template_creases)
    ridge_creases = self._template_central_ridge_segments(template_creases)

    ridge_folds = {c.fold for c in ridge_creases}
    if len(ridge_folds) != 1:
      raise ValueError(
          f'All central ridges must have the same fold: {ridge_folds}'
      )
    ridge_fold = ridge_folds.pop()

    aligned = self._aligned_crimp_and_ridge_points(crimp_creases, ridge_creases)
    aligned_ls_points = aligned[position - 1 : position + 2]
    ls_square_creases, ls_axial_crease = self._ls_square_creases(
        *aligned_ls_points, fold=ridge_fold.flip()
    )

    split_creases: list[crease_lib.Crease] = []
    for c in template_creases:
      if c.fold != crease_lib.Fold.BORDER and c.is_overlapping(ls_axial_crease):
        split_creases.extend(c.split_to_segments(ls_square_creases))
      else:
        split_creases.append(c)

    front_points, mid_points, back_points = aligned_ls_points
    _, mid_ridge = mid_points
    front_crimp, front_ridge = front_points
    back_crimp, back_ridge = back_points

    to_remove = [ls_axial_crease]
    if front_ridge is not None and mid_ridge is not None:
      to_remove.append(
          crease_lib.Crease(front_ridge, mid_ridge, fold=TEMP_FOLD)
      )
      to_remove.append(
          crease_lib.Crease(front_crimp, front_ridge, fold=TEMP_FOLD)
      )
    if back_ridge is not None and mid_ridge is not None:
      to_remove.append(crease_lib.Crease(mid_ridge, back_ridge, fold=TEMP_FOLD))
      to_remove.append(
          crease_lib.Crease(back_crimp, back_ridge, fold=TEMP_FOLD)
      )

    filtered_creases: list[crease_lib.Crease] = []
    for c in split_creases:
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
  """Represents a border crimp that pairs with a single sacrificial flap."""

  side: grid.Direction
  crimp_hinge: crease_lib.Crease

  def __post_init__(self):
    if self.crimp_hinge.fold != crease_lib.Fold.BORDER:
      raise ValueError(
          f'Crimp hinge must be a border crease: {self.crimp_hinge.fold}.'
      )

  @property
  def size(self) -> int:
    """Returns the size of the border crimp."""
    return 1

  def template_crimp_segments(
      self, solved_creases: Collection[crease_lib.Crease]
  ) -> list[crease_lib.Crease]:
    """Returns the crimp segments along the border hinge."""
    return self.crimp_hinge.split_to_segments(list(solved_creases))

  def ls_template_creases(
      self, solved_creases: list[crease_lib.Crease], grid_size: int
  ) -> list[crease_lib.Crease]:
    """No-op satisfying the `SacrificialFlapTemplate` interface."""
    del solved_creases, grid_size
    return []

  def convert_to_ls_template(
      self, solved_creases: list[crease_lib.Crease], grid_size: int
  ) -> list[crease_lib.Crease]:
    """No-op satisfying the `SacrificialFlapTemplate` interface."""
    del grid_size
    return solved_creases

  def apply_level_shifter(
      self, template_creases: list[crease_lib.Crease], position: int
  ) -> list[crease_lib.Crease]:
    """No-op satisfying the `SacrificialFlapTemplate` interface."""
    del position
    return template_creases


@dataclasses.dataclass(frozen=True)
class ElevatedFlap:
  """Represents an elevated main flap with two sacrificial flaps."""

  main_flap: shaping_utils.Rectangle
  ls_flaps: tuple[
      SacrificialFlapTemplate | BorderCrimp,
      SacrificialFlapTemplate | BorderCrimp,
  ]

  @property
  def max_shift(self) -> int:
    """Maximum single-step level shift that can be applied to the main flap."""
    return min(ls_flap.size for ls_flap in self.ls_flaps)

  def axial_paths(
      self, template_creases: Collection[crease_lib.Crease]
  ) -> dict[LSPosition, list[crease_lib.Crease]]:
    """Finds axial paths within `main_flap` connecting both `ls_flaps`."""
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
                'Axial path has multiple connections to flap'
                f' {sac_flap_idx}: {axial_path}.'
            )
          idx_by_flap[sac_flap_idx] = point_idx

      if len(idx_by_flap) != len(self.ls_flaps):
        continue
      ls_position = (idx_by_flap[0], idx_by_flap[1])
      axial_paths_by_ls_position[ls_position] = axial_path

    return axial_paths_by_ls_position

  def valid_positions(
      self, ls_positions: Collection[LSPosition]
  ) -> Iterator[LSPosition]:
    """Filters candidate level-shifter positions to interior positions."""
    if not ls_positions:
      return
    n = len(self.ls_flaps)
    idxs_per_flap = list(zip(*ls_positions, strict=True))
    mins = [min(idxs) for idxs in idxs_per_flap]
    maxes = [max(idxs) for idxs in idxs_per_flap]

    for flap_idx in ls_positions:
      if not all(mins[i] < flap_idx[i] < maxes[i] for i in range(n)):
        continue
      yield flap_idx

  def _adjust_main_flap_ridges(
      self,
      solved_creases: Sequence[crease_lib.Crease],
      position: LSPosition,
      axial_paths_by_index: dict[LSPosition, Sequence[crease_lib.Crease]],
  ) -> list[crease_lib.Crease]:
    """Adjusts main-flap ridges to accommodate deleted flanking axials."""
    if not self.main_flap.circle_centers:
      return list(solved_creases)
    axial_path = axial_paths_by_index.get(position)
    if axial_path is None:
      raise ValueError(f'No axial path found for position {position}.')
    if len(axial_path) == 1:
      return list(solved_creases)
    if len(axial_path) > 2:
      raise NotImplementedError(
          'Axial path with more than two segments not supported yet:'
          f' {len(axial_path)} {axial_path}'
      )
    ridge_intersection = axial_path[0].vertex_with_crease(axial_path[-1])
    if ridge_intersection is None:
      raise ValueError(f'No ridge vertex found for position {position}.')

    relevant_ridges = [
        c
        for c in self.main_flap.get_ridges()
        if c.point_status(ridge_intersection) != crease_lib.PointStatus.EXTERNAL
    ]
    if not relevant_ridges:
      raise ValueError(f'No relevant ridges found for position {position}.')
    if len(relevant_ridges) > 1:
      present_ridges = [
          cand
          for cand in relevant_ridges
          if any(
              sc.fold != crease_lib.Fold.BORDER
              and sc.is_within(cand, match_fold=False)
              for sc in solved_creases
          )
      ]
      if present_ridges:
        relevant_ridges = present_ridges
    if len(relevant_ridges) > 1:
      raise NotImplementedError(
          f'Multiple relevant ridges found for position {position}:'
          f' {relevant_ridges}'
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
            crease_lib.Crease(ridge_intersection, other_point, fold=c.fold)
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
    """Inverts the central axial and removes flanking axials in `main_flap`."""
    axial_path = axial_paths_by_index.get(position)
    if axial_path is None:
      raise ValueError(f'No axial path found for position {position}.')
    x, y = position

    adjacent_positions = {
        (x1, y1)
        for x1, y1 in axial_paths_by_index
        if x1 in (x - 1, x + 1) or y1 in (y - 1, y + 1)
    }
    if not adjacent_positions:
      raise ValueError(f'No adjacent positions found for position {position}.')
    if len(adjacent_positions) > 2:
      raise ValueError(
          f'Multiple adjacent positions found for position {position}:'
          f' {adjacent_positions}.'
      )

    to_remove = []
    for adjacent_position in adjacent_positions:
      to_remove.extend(axial_paths_by_index[adjacent_position])
    to_remove.extend(axial_path)
    to_remove = [c for c in to_remove if c.fold != crease_lib.Fold.BORDER]

    filtered_creases = [
        c
        for c in solved_creases
        if c.fold == crease_lib.Fold.BORDER
        or not any(c.isclose(c_) for c_ in to_remove)
    ]

    for axial_segment in axial_path:
      filtered_creases.append(
          dataclasses.replace(
              axial_segment,
              fold=axial_segment.fold.flip(),
              elevation=0,
          )
      )
    return filtered_creases

  def _adjust_main_flap(
      self, solved_creases: Sequence[crease_lib.Crease], position: LSPosition
  ) -> list[crease_lib.Crease]:
    """Updates ridges and axials in `main_flap` for `position`."""
    axial_paths_by_index = self.axial_paths(solved_creases)
    updated = self._adjust_main_flap_ridges(
        solved_creases, position, axial_paths_by_index
    )
    return self._adjust_main_flap_axials(
        updated, position, axial_paths_by_index
    )

  def apply_level_shifter(
      self,
      template_creases: Sequence[crease_lib.Crease],
      ls_position: LSPosition,
  ) -> list[crease_lib.Crease]:
    """Applies level shifters to `main_flap` and both sacrificial flaps."""
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
  """Groups an original and mirrored-copy `ElevatedFlap` into one unit."""

  original: ElevatedFlap
  copy: ElevatedFlap

  def _mirror_position(
      self,
      ls_position: LSPosition,
      original_positions: Collection[LSPosition],
      copy_positions: Collection[LSPosition],
  ) -> LSPosition:
    """Returns the mirrored `LSPosition` on `copy`."""
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
    return (mirrored[0], mirrored[1])

  @property
  def ls_flaps(self) -> tuple[SacrificialFlapTemplate | BorderCrimp, ...]:
    """All sacrificial flap templates for the mirrored pair."""
    return self.original.ls_flaps + self.copy.ls_flaps

  def axial_paths(
      self, template_creases: Collection[crease_lib.Crease]
  ) -> dict[LSPosition, list[crease_lib.Crease]]:
    """Returns synchronized axial paths across `original` and `copy`."""
    original_paths = self.original.axial_paths(template_creases)
    copy_paths = self.copy.axial_paths(template_creases)
    if len(original_paths) != len(copy_paths):
      raise ValueError(
          'Original and copy axial_paths have different lengths:'
          f' {len(original_paths)} vs {len(copy_paths)}.'
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
    """Applies level shifters to both halves of the mirrored pair."""
    original_paths = self.original.axial_paths(template_creases)
    copy_paths = self.copy.axial_paths(template_creases)
    mirrored_pos = self._mirror_position(
        ls_position, original_paths, copy_paths
    )
    creases = self.original.apply_level_shifter(template_creases, ls_position)
    return self.copy.apply_level_shifter(creases, mirrored_pos)

  @property
  def main_flap(self) -> shaping_utils.Rectangle:
    """The main flap of the original `ElevatedFlap`."""
    return self.original.main_flap


def _crimp_direction_to_side(
    crimp_direction: packing_types.SidePosition,
) -> grid.Direction:
  match crimp_direction:
    case 'bottom':
      return grid.Direction.SOUTH
    case 'top':
      return grid.Direction.NORTH
    case 'right':
      return grid.Direction.EAST
    case 'left':
      return grid.Direction.WEST
    case _:
      typing.assert_never(crimp_direction)


def _sf_clipping_rect(
    s_flaps: Sequence[packing_types.SacrificialFlap],
    grid_size: int,
) -> tuple[float, float, float, float]:
  """Computes unnormalized clipping bounds from sacrificial flap placements."""
  xmin, ymin = 0, 0
  xmax, ymax = grid_size, grid_size
  horizontal_sfs = []
  vertical_sfs = []

  for sf in s_flaps:
    p = sf.placement
    match sf.crimp_direction:
      case 'top' | 'bottom':
        horizontal_sfs.append(p)
        xmin = max(xmin, p.offset[0])
        xmax = min(xmax, p.offset[0] + p.size[0])
      case 'left' | 'right':
        vertical_sfs.append(p)
        ymin = max(ymin, p.offset[1])
        ymax = min(ymax, p.offset[1] + p.size[1])
      case 'top_left' | 'top_right' | 'bottom_left' | 'bottom_right':
        raise ValueError(
            f'Corner SFs are not supported in _sf_clipping_rect: {sf}'
        )

  if len(horizontal_sfs) == 2:
    p1, p2 = horizontal_sfs[0], horizontal_sfs[1]
    y_coords = sorted([
        p1.offset[1],
        p1.offset[1] + p1.size[1],
        p2.offset[1],
        p2.offset[1] + p2.size[1],
    ])
    ymin = max(ymin, y_coords[1])
    ymax = min(ymax, y_coords[2])
  if len(vertical_sfs) == 2:
    p1, p2 = vertical_sfs[0], vertical_sfs[1]
    x_coords = sorted([
        p1.offset[0],
        p1.offset[0] + p1.size[0],
        p2.offset[0],
        p2.offset[0] + p2.size[0],
    ])
    xmin = max(xmin, x_coords[1])
    xmax = min(xmax, x_coords[2])

  return xmin, ymin, xmax, ymax


def _validate_river_sf_group(
    sfs: Collection[packing_types.SacrificialFlap],
) -> None:
  """Validates that river sacrificial flaps form a consistent group."""
  if len(orientations := {sf.orientation for sf in sfs}) != 1:
    raise ValueError(
        f'All sfs should have the same orientation: {orientations}'
    )
  if len(element_names := {sf.shifted_element_name for sf in sfs}) != 1:
    raise ValueError(
        f'All sfs should have the same shifted element name: {element_names}'
    )
  for sf1, sf2 in itertools.combinations(sfs, 2):
    if sf1 == sf2:
      raise ValueError('No two sfs should be the same flap.')
    if sf1.shifted_element != sf2.shifted_element:
      raise ValueError('All sfs should have the same shifted element.')
    if not isinstance(sf1.shifted_element, river_lib.PackingRiver):
      raise ValueError('Shifted element should be a river.')


def _get_opposite_border_crimp(
    ls_flap: SacrificialFlapTemplate,
    main_flap_rect: shaping_utils.Rectangle,
) -> BorderCrimp:
  """Builds a `BorderCrimp` on the opposite border hinge when available."""
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
      'There is only one sacrificial flap and no opposite border hinge:'
      f' {ls_flap}'
  )


def get_elevated_flaps(
    flap_rectangles: Sequence[shaping_utils.Rectangle],
    sacrificial_flaps: Sequence[packing_types.SacrificialFlap],
    grid_size: int,
    group_mirrored: bool = True,
    symmetric_pairs: dict[str, str] | None = None,
) -> list[LevelShiftable]:
  """Constructs `ElevatedFlap` / `MirroredElevatedFlapPair` instances."""
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
      for river_rect in river.rectangles:
        if river_rect.clip(*clip_box) is not None:
          break
      else:
        raise ValueError(
            'Computed main_flap_rect is not within river rectangles:'
            f' {clip_box}'
        )
      main_flap_rect = shaping_utils.Rectangle.from_clip_box(
          name=group_key,
          clip_box=clip_box,
      )
    else:
      if element_name not in rectangles_by_name:
        raise ValueError(
            f'Main flap {element_name} not found in flap_rectangles.'
        )
      main_flap_rect = rectangles_by_name[element_name].clip(*clip_box)
      if main_flap_rect is not None and element_name != group_key:
        object.__setattr__(main_flap_rect, 'name', group_key)

    if main_flap_rect is None:
      raise ValueError(
          f'Clipped main flap does not overlap with paper: {group_key}'
      )
    main_flap_rect = main_flap_rect.normalize_rectangle(grid_size)

    ls_flaps_list: list[SacrificialFlapTemplate | BorderCrimp] = []
    for sf in s_flaps:
      side = _crimp_direction_to_side(
          typing.cast(packing_types.SidePosition, sf.crimp_direction)
      )
      if sf.polygon_name not in rectangles_by_name:
        raise ValueError(
            f'Sacrificial flap {sf.polygon_name} not found in flap_rectangles.'
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
          f'Multiple LS flaps per side are not supported yet: {group_key}'
      )
    if len(ls_flaps_list) == 1:
      first_ls = ls_flaps_list[0]
      if not isinstance(first_ls, SacrificialFlapTemplate):
        raise ValueError(f'Expected SacrificialFlapTemplate, got {first_ls}.')
      ls_flaps_list.append(
          _get_opposite_border_crimp(first_ls, main_flap_rect)
      )

    elevated_flaps.append(
        ElevatedFlap(
            main_flap=main_flap_rect,
            ls_flaps=(ls_flaps_list[0], ls_flaps_list[1]),
        )
    )
  if group_mirrored:
    return group_mirrored_elevated_flaps(elevated_flaps, symmetric_pairs)
  return elevated_flaps


def group_mirrored_elevated_flaps(
    elevated_flaps: list[ElevatedFlap],
    symmetric_pairs: dict[str, str] | None = None,
) -> list[LevelShiftable]:
  """Groups mirrored elevated flap pairs into `MirroredElevatedFlapPair`s."""
  if symmetric_pairs is None:
    symmetric_pairs = {}

  groups: dict[str, list[ElevatedFlap]] = collections.defaultdict(list)
  for ef in elevated_flaps:
    name = ef.main_flap.name.removesuffix('_copy')
    partner = symmetric_pairs.get(name)
    if (
        partner
        and partner != name
        and any(e.main_flap.name == partner for e in elevated_flaps)
    ):
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
          f'Expected 1 or 2 elevated flaps per group, got {len(group)}.'
      )
  return result


def make_preassignment_cp(
    ls_flaps: Collection[SacrificialFlapTemplate],
    grid_size: int,
) -> crease_pattern.CreasePattern:
  """Builds a solver preassignment `CreasePattern` for `ls_flaps`."""
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


def apply_ls_templates_to_result(
    solve_result: hinge_lib.SolverResult,
    elevated_flaps: Collection[LevelShiftable],
    grid_size: int,
) -> hinge_lib.SolverResult:
  """Converts sacrificial flaps in `solve_result` into `LS_TEMPLATE` creases."""
  if solve_result.base_type != hinge_lib.BaseType.NORMAL:
    raise ValueError(
        'Cannot apply LS templates to a non-normal base type result:'
        f' {solve_result.base_type}.'
    )

  ls_template_creases = list(solve_result.crease_pattern.creases)
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
    *,
    record_stages: bool = False,
    initial_stages_out: (
        list[tuple[str, crease_pattern.CreasePattern]] | None
    ) = None,
) -> list[hinge_lib.SolverResult]:
  """Solves `packing_cp` with sacrificial preassignments into `LS_TEMPLATE`s."""

  def _plot_cp(cp: crease_pattern.CreasePattern) -> None:
    if plot_stages:
      visualize.draw_cp(
          cp, grid_size=solver_config.grid_size, annotate_vertices=False
      )

  if solver_config.preassignments is not None:
    raise ValueError('Preassignments should not already be set.')
  if not elevated_flaps:
    raise ValueError('elevated_flaps must not be empty.')
  if not solver_config.allow_misaligned_symmetric_hinges:
    logging.warning(
        'SolverConfig.allow_misaligned_symmetric_hinges is False; this is not'
        ' recommended with level-shifter solving.'
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

  def template_callback(
      solve_result: hinge_lib.SolverResult,
  ) -> hinge_lib.SolverResult | None:
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
  solver = solver_config.make_from_cp(
      packing_cp, record_stages=record_stages or initial_stages_out is not None
  )
  if initial_stages_out is not None and solver.initial_stage_snapshots:
    initial_stages_out.clear()
    initial_stages_out.extend(solver.initial_stage_snapshots)
  return solver.solve()
