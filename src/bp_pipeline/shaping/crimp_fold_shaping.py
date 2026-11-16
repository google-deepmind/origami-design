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

"""Crimp fold shaping methods.

Crimp folds consists of an inscribed rectangle placed inside a flap with zigzags
applied along the inscribed rectangle. The zigzags can alternate between MM and
VV folds.
"""

import dataclasses
from typing import Sequence

from ..crease_pattern import crease as crease_utils
from ..crease_pattern import crease_pattern as cp_utils
from ..crease_pattern import point as point_utils
from . import inscribed_rectangle_shaping
from . import shaping_elements
from . import types as shaping_types
from . import zigzag_shaping
from .. import timeout as timeout_lib


@dataclasses.dataclass(kw_only=True, frozen=True)
class CrimpFoldParams:
  """Params determining placement of a crimp fold.

  Attributes:
    depth: How deep (in grid units) is the crimp fold inside the flap.
    fold_order: The fold order (M/V) of the inscribed rectangle defining the
      crimp fold.
    fold_percent: The fold percent of the crimp fold.
    zigzag_flap_angle: The angle of the flap to be shaped.
    zigzag_fold_order: A sequence of folds detailing the fold order; for example
      a fold order of [MOUNTAIN, VALLEY] will have the zigzag crease starting
      with a mountain fold and then alternating valley and mountain
      orientations, whereas a fold order of [VALLEY] will have zigzag creases be
      all valley folds.
    zigzag_even_phase: Determines the phase shift of the zigzags.
    bank_index: The index of the bank to be shaped (only for river shaping).
  """

  # parameters controlling the placement of inscribed rectangle.
  depth: int
  fold_order: Sequence[crease_utils.Fold]
  fold_percent: float

  # parameters controlling the placement of zigzags.
  zigzag_flap_angle: float
  zigzag_fold_order: Sequence[crease_utils.Fold]
  zigzag_even_phase: bool

  # parameters controlling the placement of rivers.
  bank_index: shaping_types.BankIndex | None = None


@dataclasses.dataclass(kw_only=True, frozen=True)
class CrimpFoldShapingParams:
  """Wrapper around flap name + list of crimp fold params.

  Attributes:
    element_name: The name of the element to be shaped.
    crimp_fold_params: The list of crimp fold params determining the placement
      of the folds.
  """

  element_name: str
  crimp_fold_params: Sequence[CrimpFoldParams]


def _get_zigzag_point_fold_counts(
    zigzag_creases: list[crease_utils.Crease]
) -> dict[point_utils.Point, dict[crease_utils.Fold, int]]:
  """Counts folds of zigzag creases connected to each unique point."""
  point_fold_counts = {}

  for zz_crease in zigzag_creases:
    timeout_lib.maybe_timeout_cooperatively()
    for point in [zz_crease.p1, zz_crease.p2]:
      # Find if an already-keyed point is close to the current point.
      found_key = None
      for key in point_fold_counts:
        if point.isclose(key):
          found_key = key
          break
      if found_key:
        counts = point_fold_counts[found_key]
        counts[zz_crease.fold] = counts.get(zz_crease.fold, 0) + 1
      else:
        point_fold_counts[point] = {zz_crease.fold: 1}
  return point_fold_counts


def fix_flat_foldability(
    processed_creases: list[crease_utils.Crease],
    inscribed_rectangle_creases: list[crease_utils.Crease],
    zigzag_creases: list[crease_utils.Crease],
) -> list[crease_utils.Crease]:
  """Adjusts fold assignments to improve flat-foldability around crimp folds.

  Resolves non-flat-foldable vertices after adding crimp fold creases.
  It identifies original creases connecting zigzags to inscribed rectangles
  and flips their fold order (M/V) if they are part of any non-flat-foldable
  vertex.

  Args:
    processed_creases: Creases from the original CP split by the new creases.
    inscribed_rectangle_creases: Inscribed rectangle creases from crimp fold.
    zigzag_creases: Zigzag creases from crimp fold.

  Returns:
    A list of creases with potentially modified fold orders.
  """
  inscribed_points = set()
  for c in inscribed_rectangle_creases:
    inscribed_points.add(c.p1)
    inscribed_points.add(c.p2)

  zigzag_points = set()
  for c in zigzag_creases:
    zigzag_points.add(c.p1)
    zigzag_points.add(c.p2)

  # Remove common points (intersections of zigzag and inscribed rectangles)
  common_inscribed = set()
  common_zigzag = set()
  for ip in inscribed_points:
    for zp in zigzag_points:
      if ip.isclose(zp):
        common_inscribed.add(ip)
        common_zigzag.add(zp)
        break
  inscribed_points -= common_inscribed
  zigzag_points -= common_zigzag

  # Precompute counts of zigzag creases per point and fold.
  zigzag_point_fold_counts = _get_zigzag_point_fold_counts(zigzag_creases)

  creases_to_modify = list(processed_creases)
  # Iterate through each processed crease. If a crease connects a zigzag point
  # to an inscribed rectangle point, and if those endpoints are shared by
  # multiple (two) zigzag creases with the same fold, then flip the
  # fold order of this crease to fix flat-foldability.
  for i, c in enumerate(creases_to_modify):
    timeout_lib.maybe_timeout_cooperatively()
    if not c.mountain_or_valley():
      continue

    p1_on_zigzag = any(p.isclose(c.p1) for p in zigzag_points)
    p2_on_zigzag = any(p.isclose(c.p2) for p in zigzag_points)
    p1_on_inscribed = any(p.isclose(c.p1) for p in inscribed_points)
    p2_on_inscribed = any(p.isclose(c.p2) for p in inscribed_points)

    should_flip = False
    target_point = None
    if p1_on_zigzag and p2_on_inscribed:
      target_point = c.p1
    elif p2_on_zigzag and p1_on_inscribed:
      target_point = c.p2

    if target_point:
      # Find the fold count for the target_point.
      close_zz_point = None
      for key in zigzag_point_fold_counts:
        if target_point.isclose(key):
          close_zz_point = key
          break
      if close_zz_point:
        count = zigzag_point_fold_counts[close_zz_point].get(c.fold, 0)
        if count > 1:
          should_flip = True

    if should_flip:
      creases_to_modify[i] = dataclasses.replace(c, fold=c.fold.flip())

  return creases_to_modify


def get_crimp_fold_creases(
    cp: cp_utils.CreasePattern,
    grid_size: int,
    flaps: Sequence[shaping_elements.ShapingFlap],
    element_shaping_params: Sequence[CrimpFoldShapingParams],
    *,
    rivers: Sequence[shaping_elements.ShapingRiver] | None = None,
    central_crease_removed: bool = True,
    snap_to_grid: bool = False,
) -> tuple[list[crease_utils.Crease], list[crease_utils.Crease]]:
  """Generates crimp fold creases (inscribed and zigzag) without flat-foldability fix.

  Args:
    cp: The crease pattern (used for fold_step).
    grid_size: The size of the crease pattern.
    flaps: The flaps to shape.
    element_shaping_params: The parameters for shaping the crimp folds.
    rivers: The rivers defining the rivers in the crease pattern.
    central_crease_removed: Whether the central crease is removed.
    snap_to_grid: Whether to snap the zigzag creases to the grid.

  Returns:
    A tuple containing:
      - inscribed_rectangle_creases: List of creases for inscribed rectangles.
      - zigzag_creases: List of creases for zigzags.
  """
  inscribed_rectangle_creases = []
  zigzag_creases = []

  if rivers is None:
    rivers = []

  flaps_by_name = {flap.name: flap for flap in flaps}
  rivers_by_name = {river.name: river for river in rivers}

  for param in element_shaping_params:
    element_name = param.element_name
    flap = flaps_by_name.get(element_name)
    river = rivers_by_name.get(element_name)

    if flap is None and river is None:
      raise ValueError(f'Element to be shaped not found in CP: {element_name}')

    # We apply crimp fold one at a time, with each fold having its own
    # inscribed rectangle and zigzag.
    for crimp_fold_param in param.crimp_fold_params:
      if flap is not None:
        inscribed_rectangle_creases_per_fold, inscribed_element_per_fold = (
            inscribed_rectangle_shaping.get_inscribed_rectangle_shaping_creases_and_rectangles(
                grid_size=grid_size,
                flap_rectangles=[flap],
                flap_shaping_params=[
                    inscribed_rectangle_shaping.InscribedRectangleFlapShapingParams(
                        flap_name=element_name,
                        inscribed_rectangles_params=[
                            inscribed_rectangle_shaping.InscribedRectangleParams(
                                depth=crimp_fold_param.depth,
                                fold_order=crimp_fold_param.fold_order,
                                fold_percent=crimp_fold_param.fold_percent,
                            )
                        ],
                    )
                ],
                central_crease_removed=central_crease_removed,
                fold_step=cp.num_fold_steps,
            )
        )
      else:
        inscribed_rectangle_creases_per_fold = (
            inscribed_rectangle_shaping.get_inscribed_river_shaping_creases(
                rivers=[river],
                river_shaping_params=[
                    inscribed_rectangle_shaping.InscribedRiverShapingParams(
                        element_name=element_name,
                        inscribed_river_params=[
                            inscribed_rectangle_shaping.InscribedRiverParams(
                                depth=crimp_fold_param.depth,
                                fold_order=crimp_fold_param.fold_order,
                                fold_percent=crimp_fold_param.fold_percent,
                                bank_index=crimp_fold_param.bank_index,
                            )
                        ],
                    )
                ],
                central_crease_removed=central_crease_removed,
            )
        )
        assert river is not None
        inscribed_element_per_fold = [
            river.shrink_along_bank(
                crimp_fold_param.bank_index, crimp_fold_param.depth
            )
        ]

      if crimp_fold_param.fold_percent != 0:
        inscribed_rectangle_creases.extend(inscribed_rectangle_creases_per_fold)

      if crimp_fold_param.zigzag_flap_angle != 0:
        if flap is not None:
          zigzag_params = shaping_types.ZigzagParams(
              target_flap_angle=crimp_fold_param.zigzag_flap_angle,
              target_zigzag_depth=None,
              # Hardcode the fold order to be M/V because the zigzag
              # shaping does not always produce MMVV creases.
              # We will fix it after the creases are generated.
              fold_order=[
                  crease_utils.Fold.MOUNTAIN,
                  crease_utils.Fold.VALLEY,
              ],
              even_phase=crimp_fold_param.zigzag_even_phase,
          )
        else:
          zigzag_params = shaping_types.RiverZigzagParams(
              target_flap_angle=crimp_fold_param.zigzag_flap_angle,
              target_zigzag_depth=None,
              bank_index=crimp_fold_param.bank_index,
              # Hardcode the fold order to be M/V because the zigzag
              # shaping does not always produce MMVV creases.
              # We will fix it after the creases are generated.
              fold_order=[
                  crease_utils.Fold.MOUNTAIN,
                  crease_utils.Fold.VALLEY,
              ],
              even_phase=crimp_fold_param.zigzag_even_phase,
          )

        zigzag_crease = zigzag_shaping.get_zigzag_shaping_creases(
            grid_size=grid_size,
            rectangles=inscribed_element_per_fold if flap else [],
            rivers=inscribed_element_per_fold if river else [],
            element_shaping_params=[
                zigzag_shaping.ZigzagShapingParamsForElement(
                    element_name=inscribed_element_per_fold[0].name,
                    zigzag_params=zigzag_params,
                )
            ],
            snap_to_grid=snap_to_grid,
            fold_step=cp.num_fold_steps,
        )

        # Fix the fold order from zigzag shaping.
        # NOTE: We can remove this when zigzag produces MMVV creases.
        fold_order = list(crimp_fold_param.zigzag_fold_order)
        zigzag_crease = [
            dataclasses.replace(crease, fold=fold_order[idx % len(fold_order)])
            for idx, crease in enumerate(zigzag_crease)
        ]

        if not zigzag_crease:
          raise ValueError(
              f'Zigzag creases were not added to element: {element_name},'
              f' param: {crimp_fold_param}'
          )
        zigzag_creases.extend(zigzag_crease)

  return inscribed_rectangle_creases, zigzag_creases


def assign_crimp_fold_orders(
    inscribed_creases: list[crease_utils.Crease],
    zigzag_creases: list[crease_utils.Crease],
    fold_order: Sequence[crease_utils.Fold],
    zigzag_fold_order: Sequence[crease_utils.Fold],
) -> tuple[list[crease_utils.Crease], list[crease_utils.Crease]]:
  """Assigns fold orders to crimp fold creases.

  Args:
    inscribed_creases: The inscribed rectangle creases.
    zigzag_creases: The zigzag creases.
    fold_order: The fold order for inscribed rectangle creases.
    zigzag_fold_order: The fold order for zigzag creases.

  Returns:
    A tuple of (updated_inscribed_creases, updated_zigzag_creases).
  """
  new_inscribed = [
      dataclasses.replace(c, fold=fold_order[i % len(fold_order)])
      for i, c in enumerate(inscribed_creases)
  ]
  new_zigzag = [
      dataclasses.replace(c, fold=zigzag_fold_order[i % len(zigzag_fold_order)])
      for i, c in enumerate(zigzag_creases)
  ]
  return new_inscribed, new_zigzag


def shape_cp_with_crimp_fold(
    cp: cp_utils.CreasePattern,
    grid_size: int,
    flaps: Sequence[shaping_elements.ShapingFlap],
    element_shaping_params: Sequence[CrimpFoldShapingParams],
    *,
    rivers: Sequence[shaping_elements.ShapingRiver] | None = None,
    central_crease_removed: bool = True,
    snap_to_grid: bool = False,
) -> cp_utils.CreasePattern:
  """Shapes a crease pattern with crimp folds.

  Crimp folds are a combination of an inscribed rectangle and zigzags.

  Args:
    cp: The crease pattern to be shaped.
    grid_size: The size of the crease pattern.
    flaps: The flaps to shape.
    element_shaping_params: The parameters for shaping the crimp folds.
    rivers: The rivers defining the rivers in the crease pattern.
    central_crease_removed: Whether the central crease is removed from the
      crease pattern.
    snap_to_grid: Whether to snap the zigzag creases to the grid.

  Returns:
    The crease pattern with the added crimp folds.
  """
  inscribed_rectangle_creases, zigzag_creases = get_crimp_fold_creases(
      cp=cp,
      grid_size=grid_size,
      flaps=flaps,
      element_shaping_params=element_shaping_params,
      rivers=rivers,
      central_crease_removed=central_crease_removed,
      snap_to_grid=snap_to_grid,
  )
  inscribed_and_zigzag_creases = list(inscribed_rectangle_creases) + list(
      zigzag_creases
  )
  tips, _ = cp_utils.get_tips_and_grid_creases(
      cp.creases + inscribed_and_zigzag_creases, grid_size=-1)
  processed_creases = cp_utils.split_creases(
      cp.creases, inscribed_and_zigzag_creases, tips)
  modified_creases = fix_flat_foldability(
      processed_creases, inscribed_rectangle_creases, zigzag_creases)

  return cp_utils.CreasePattern(
      creases=modified_creases + inscribed_and_zigzag_creases,
      light_mode=True,
  )
