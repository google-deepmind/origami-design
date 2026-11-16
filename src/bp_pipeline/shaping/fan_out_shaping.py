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

"""Fan out shaping function and utilities."""

import dataclasses
import enum
from typing import Any, Collection, Sequence

from ..crease_pattern import crease as crease_utils
from ..crease_pattern import crease_pattern as cp_utils
from . import flat_foldable
from ..crease_pattern import point as point_utils
from . import shaping_utils
from .. import timeout as timeout_lib


Point = point_utils.Point
Crease = crease_utils.Crease
CreasePattern = cp_utils.CreasePattern
Fold = crease_utils.Fold


class Orientation(str, enum.Enum):
  """An enum class to represent which crease orientation(s) to apply fan out shaping to."""

  HORIZONTAL = 'horizontal'
  VERTICAL = 'vertical'


def _has_horizontal(valid_orientations: Collection[Any]) -> bool:
  return any(
      getattr(o, 'value', o) in ('horizontal', Orientation.HORIZONTAL)
      for o in valid_orientations
  )


def _has_vertical(valid_orientations: Collection[Any]) -> bool:
  return any(
      getattr(o, 'value', o) in ('vertical', Orientation.VERTICAL)
      for o in valid_orientations
  )


@dataclasses.dataclass(frozen=True, kw_only=True)
class FanOutShapingParams:
  """Parameters for fan out shaping.

  Fields:
    valid_fold_types: valid fold types (M/V) to apply fan out shaping to.
    valid_orientations: valid orientations (horizontal/vertical) to apply fan
      out shaping to.
    rectangles: rectangles to constrain fan out shaping to.
    move_delta: the amount to move the border point by. E.g. if this is -0.5,
      then the border point will be moved -0.5 grid spaces left (if the crease
      is horizontal) or -0.5 grid spaces up (if the crease is vertical).
  """

  valid_fold_types: Collection[Fold]
  valid_orientations: Collection[Orientation]
  rectangles: list[shaping_utils.Rectangle]
  move_delta: float


@dataclasses.dataclass(kw_only=True)
class FannedOutCrease:
  """Stores a fanned-out crease and its corresponding original crease and rectangle."""
  fanned_out_crease: Crease
  original_crease: Crease
  rectangle: shaping_utils.Rectangle


@dataclasses.dataclass(kw_only=True)
class FanOutCandidates:
  """Result of the crease-finding stage for a given fold-type/orientation."""

  modified_creases: list[Crease]
  unmodified_creases: list[Crease]
  modified_rectangles: set[shaping_utils.Rectangle]
  avg_modified_crease_length: float


def _is_valid_horizontal_fan_out_crease(
    crease: Crease, valid_orientations: Collection[Orientation]
) -> bool:
  """Checks if the crease is a valid horizontal fan out crease.

  The crease must:
    - be horizontal
    - be extending from the border but not extending all the way to the opposite
      border
    - has an orientation that is in the valid_orientations

  Args:
    crease: The crease to check.
    valid_orientations: The orientations to check.

  Returns:
    True if the crease is a valid horizontal fan out crease, False otherwise.
  """
  return (
      _has_horizontal(valid_orientations)
      and crease.is_horizontal
      and (
          crease.left_point.close_to_left_border
          ^ crease.right_point.close_to_right_border
      )
  )


def _is_valid_vertical_fan_out_crease(
    crease: Crease, valid_orientations: Collection[Orientation]
) -> bool:
  """Checks if the crease is a valid vertical fan out crease.

  The crease must:
    - be vertical
    - be extending from the border but not extending all the way to the opposite
      border
    - has an orientation that is in the valid_orientations

  Args:
    crease: The crease to check.
    valid_orientations: The orientations to check.

  Returns:
    True if the crease is a valid vertical fan out crease, False otherwise.
  """
  return (
      _has_vertical(valid_orientations)
      and crease.is_vertical
      and (
          crease.top_point.close_to_top_border
          ^ crease.bottom_point.close_to_bottom_border
      )
  )


def get_fanned_out_crease(
    crease: Crease,
    rectangle: shaping_utils.Rectangle,
    valid_fold_types: Collection[Fold],
    valid_orientations: Collection[Orientation],
    move_delta: float,
) -> Crease | None:
  """Apply fan out shaping to a crease if it is valid.

  A valid crease is crease that:
  - doesn't overlap the border
  - is within the rectangle
  - is vertical/horizontal
  - has one tip that touches the border and
  one tip that does not
  - has a fold type that is in the valid_fold_types
  - has an orientation that is in the valid_orientations

  Args:
    crease: The crease to check.
    rectangle: The rectangle to constrain the fan out shaping to.
    valid_fold_types: valid fold types (M/V) to apply fan out shaping to.
    valid_orientations: valid orientations (horizontal/vertical) to apply fan
      out shaping to.
    move_delta: the amount to move the border point by. E.g. if this is -0.5,
      then the border point will be moved -0.5 grid spaces left (if the crease
      is horizontal) or -0.5 grid spaces up (if the crease is vertical).

  Returns:
    The fanned out crease if it is valid, None otherwise.
  """
  if not valid_fold_types:
    raise ValueError('valid_fold_types must not be empty.')
  if not valid_orientations:
    raise ValueError('valid_orientations must not be empty.')

  if (
      (crease.fold not in valid_fold_types)
      or (not rectangle.crease_is_within(crease))
      or crease.is_overlapping_border
  ):
    return None

  if _is_valid_horizontal_fan_out_crease(crease, valid_orientations):
    modified_point, unmodified_point = (
        (crease.left_point, crease.right_point)
        if crease.close_to_left_border
        else (crease.right_point, crease.left_point)
    )
    try:
      modified_point = Point(
          x=modified_point.x, y=modified_point.y + move_delta
      )
    except ValueError:  # Fanned out crease not within the [0, 1] border.
      return None
  elif _is_valid_vertical_fan_out_crease(crease, valid_orientations):
    modified_point, unmodified_point = (
        (crease.top_point, crease.bottom_point)
        if crease.close_to_top_border
        else (crease.bottom_point, crease.top_point)
    )
    try:
      modified_point = Point(
          x=modified_point.x + move_delta, y=modified_point.y
      )
    except ValueError:  # Fanned out crease not within the [0, 1] border.
      return None
  else:  # Crease is not valid for fan out shaping.
    return None

  return Crease(
      modified_point,
      unmodified_point,
      fold=crease.fold,
      fold_percent=crease.fold_percent,
  )


def _hashable_rectangle(
    rect: shaping_utils.Rectangle,
) -> shaping_utils.Rectangle:
  """Ensures `circle_centers` is a tuple so `Rectangle` can be stored in sets."""
  if isinstance(rect.circle_centers, list):
    return dataclasses.replace(rect, circle_centers=tuple(rect.circle_centers))
  return rect


def _point_in_clipped_rect(pt: Point, rect: shaping_utils.Rectangle) -> bool:
  clipped = rect.clip()
  if clipped is None:
    return False
  tol = point_utils.DEFAULT_TOLERANCE
  return (
      clipped.x_start - tol <= pt.x <= clipped.x_end + tol
      and clipped.y_start - tol <= pt.y <= clipped.y_end + tol
  )


def _split_merged_border_creases(
    cp: CreasePattern,
    shaping_params: Sequence[FanOutShapingParams],
) -> list[Crease]:
  """Splits border-touching H/V creases that extend past a vertex in `rectangles`."""
  rects: list[shaping_utils.Rectangle] = []
  for p in shaping_params:
    for r in p.rectangles:
      clipped = r.clip()
      if clipped is not None:
        rects.append(clipped)
  if not rects:
    return list(cp.creases)
  orientations = set()
  for p in shaping_params:
    for o in p.valid_orientations:
      orientations.add(o)
  tol = point_utils.DEFAULT_TOLERANCE
  vertex_pts = [
      v.point
      for v in cp.vertices
      if any(
          r.x_start - tol <= v.point.x <= r.x_end + tol
          and r.y_start - tol <= v.point.y <= r.y_end + tol
          for r in rects
      )
  ]
  if not vertex_pts:
    return list(cp.creases)

  out: list[Crease] = []
  for c in cp.creases:
    if c.fold == Fold.BORDER:
      out.append(c)
      continue
    want_split = False
    if _has_horizontal(orientations) and c.is_horizontal:
      if (
          c.left_point.close_to_left_border
          ^ c.right_point.close_to_right_border
      ) and any(
          r.y_start - tol <= c.p1.y <= r.y_end + tol
          and not r.crease_is_within(c)
          for r in rects
      ):
        want_split = True
    elif _has_vertical(orientations) and c.is_vertical:
      if (
          c.top_point.close_to_top_border
          ^ c.bottom_point.close_to_bottom_border
      ) and any(
          r.x_start - tol <= c.p1.x <= r.x_end + tol
          and not r.crease_is_within(c)
          for r in rects
      ):
        want_split = True
    if not want_split:
      out.append(c)
      continue
    internal_pts = [
        pt
        for pt in vertex_pts
        if c.point_status(pt) == crease_utils.PointStatus.INTERNAL
    ]
    if not internal_pts:
      out.append(c)
      continue
    pts = sorted(internal_pts + [c.p1, c.p2], key=lambda p: (p.x, p.y))
    dedup = [pts[0]]
    for pt in pts[1:]:
      if not pt.isclose(dedup[-1]):
        dedup.append(pt)
    for p_a, p_b in zip(dedup, dedup[1:]):
      out.append(dataclasses.replace(c, p1_init=p_a, p2_init=p_b))
  return out


def _collect_potential_fanned_creases(
    creases: Sequence[Crease],
    shaping_params: Sequence[FanOutShapingParams],
    grid_size: int,
    *,
    allow_border_tip_only: bool,
) -> tuple[list[Crease], list[FannedOutCrease]]:
  """Scans `creases` for candidate fanned-out creases."""
  full_paper = shaping_utils.Rectangle(
      name='paper', x_start=0.0, y_start=0.0, x_len=1.0, y_len=1.0
  )
  unmodified_creases: list[Crease] = []
  potential_fanned_out_creases: list[FannedOutCrease] = []
  for crease in creases:
    timeout_lib.maybe_timeout_cooperatively()
    potential_fanned_out_crease: FannedOutCrease | None = None

    for shaping_param in shaping_params:
      move_delta = shaping_param.move_delta / grid_size

      for rectangle in shaping_param.rectangles:
        clipped = rectangle.clip()
        if clipped is None:
          continue
        if not allow_border_tip_only:
          if not rectangle.crease_is_within(crease):
            continue
          target_rect = clipped
          record_rect = _hashable_rectangle(rectangle)
        else:
          border_pt = (
              crease.p1
              if crease.p1.is_border
              else (crease.p2 if crease.p2.is_border else None)
          )
          if border_pt is None or not _point_in_clipped_rect(
              border_pt, rectangle
          ):
            continue
          target_rect = full_paper
          min_x = min(clipped.x_start, crease.p1.x, crease.p2.x)
          max_x = max(clipped.x_end, crease.p1.x, crease.p2.x)
          min_y = min(clipped.y_start, crease.p1.y, crease.p2.y)
          max_y = max(clipped.y_end, crease.p1.y, crease.p2.y)
          record_rect = shaping_utils.Rectangle(
              name=rectangle.name,
              x_start=min_x,
              y_start=min_y,
              x_len=max(max_x - min_x, 1e-6),
              y_len=max(max_y - min_y, 1e-6),
          )

        fanned_out_crease = get_fanned_out_crease(
            crease,
            target_rect,
            shaping_param.valid_fold_types,
            shaping_param.valid_orientations,
            move_delta,
        )
        if fanned_out_crease:
          potential_fanned_out_crease = FannedOutCrease(
              fanned_out_crease=fanned_out_crease,
              original_crease=crease,
              rectangle=record_rect,
          )
          break

      if potential_fanned_out_crease is not None:
        break

    if potential_fanned_out_crease is not None:
      potential_fanned_out_creases.append(potential_fanned_out_crease)
    else:
      unmodified_creases.append(crease)

  return unmodified_creases, potential_fanned_out_creases


def _filter_non_intersecting_candidates(
    unmodified_creases: list[Crease],
    potential_fanned_out_creases: Sequence[FannedOutCrease],
) -> tuple[list[FannedOutCrease], list[Crease]]:
  """Filters out fanned creases that illegally cross unmodified creases."""
  kept_fanned: list[FannedOutCrease] = []
  final_unmodified = list(unmodified_creases)
  for fanned_out_crease in potential_fanned_out_creases:
    timeout_lib.maybe_timeout_cooperatively()
    modified_crease = fanned_out_crease.fanned_out_crease
    original_crease = fanned_out_crease.original_crease

    valid_intersection = True
    for unmodified_crease in final_unmodified:
      timeout_lib.maybe_timeout_cooperatively()
      if unmodified_crease.fold == Fold.BORDER:
        continue
      intersection_point = modified_crease.vertex_with_crease(
          unmodified_crease
      )
      if (intersection_point is not None) and (
          not modified_crease.is_tip(intersection_point)
      ):
        final_unmodified.append(original_crease)
        valid_intersection = False
        break

    if valid_intersection:
      kept_fanned.append(fanned_out_crease)
  return kept_fanned, final_unmodified


def _candidates_from_fanned(
    kept_fanned: Sequence[FannedOutCrease],
    unmodified_creases: Sequence[Crease],
) -> FanOutCandidates | None:
  if not kept_fanned:
    return None
  modified_creases = [fc.fanned_out_crease for fc in kept_fanned]
  modified_rectangles = {
      _hashable_rectangle(fc.rectangle) for fc in kept_fanned
  }
  avg_length = sum(c.length for c in modified_creases) / len(modified_creases)
  return FanOutCandidates(
      modified_creases=modified_creases,
      unmodified_creases=list(unmodified_creases),
      modified_rectangles=modified_rectangles,
      avg_modified_crease_length=avg_length,
  )


def _single_pleat_subset_candidates(
    kept_fanned: Sequence[FannedOutCrease],
    unmodified_creases: Sequence[Crease],
    is_left: bool = False,
) -> list[FanOutCandidates]:
  """Builds single-pleat fallback `FanOutCandidates` ordered shortest-first."""
  if len(kept_fanned) <= 1:
    return []

  def _sort_key(i: int):
    oc = kept_fanned[i].original_crease
    mid_x = (oc.p1.x + oc.p2.x) / 2.0
    mid_y = (oc.p1.y + oc.p2.y) / 2.0
    canon_x = 1.0 - mid_x if is_left else mid_x
    return (round(oc.length, 6), round(canon_x, 6), round(mid_y, 6))

  ordered_indices = sorted(
      range(len(kept_fanned)),
      key=_sort_key,
  )
  subsets: list[FanOutCandidates] = []
  for idx in ordered_indices:
    chosen = kept_fanned[idx]
    rest_originals = [
        kept_fanned[j].original_crease
        for j in range(len(kept_fanned))
        if j != idx
    ]
    cand = _candidates_from_fanned(
        [chosen], list(unmodified_creases) + rest_originals
    )
    if cand is not None:
      subsets.append(cand)
  return subsets


def _get_fan_out_candidates_with_fanned(
    cp: CreasePattern,
    shaping_params: Sequence[FanOutShapingParams],
    grid_size: int,
) -> tuple[FanOutCandidates | None, list[FannedOutCrease], list[Crease]]:
  """Internal helper returning both `FanOutCandidates` and per-crease items."""
  unmodified_creases, potential_fanned = _collect_potential_fanned_creases(
      cp.creases,
      shaping_params,
      grid_size,
      allow_border_tip_only=False,
  )
  kept_fanned, final_unmodified = _filter_non_intersecting_candidates(
      unmodified_creases, potential_fanned
  )
  candidates = _candidates_from_fanned(kept_fanned, final_unmodified)
  if candidates is not None:
    return candidates, kept_fanned, final_unmodified

  working_creases = _split_merged_border_creases(cp, shaping_params)
  for allow_border_tip_only in (False, True):
    unmodified_creases, potential_fanned = _collect_potential_fanned_creases(
        working_creases,
        shaping_params,
        grid_size,
        allow_border_tip_only=allow_border_tip_only,
    )
    kept_fanned, final_unmodified = _filter_non_intersecting_candidates(
        unmodified_creases, potential_fanned
    )
    candidates = _candidates_from_fanned(kept_fanned, final_unmodified)
    if candidates is not None:
      return candidates, kept_fanned, final_unmodified
  return None, [], list(cp.creases)


def _get_fan_out_candidates(
    cp: CreasePattern,
    shaping_params: Sequence[FanOutShapingParams],
    grid_size: int,
) -> FanOutCandidates | None:
  """Finds all valid fan-out crease candidates for the given shaping params.

  Args:
    cp: The crease pattern to find candidates in.
    shaping_params: A list of shaping parameters specifying fold types,
      orientations, rectangles, and move deltas.
    grid_size: The grid size of the crease pattern.

  Returns:
    A FanOutCandidates with the modified/unmodified creases and average
    modified crease length, or None if no valid fan-out creases are found.
  """
  candidates, _, _ = _get_fan_out_candidates_with_fanned(
      cp, shaping_params, grid_size
  )
  return candidates


def _validate_shaping_params(
    shaping_params: Sequence[FanOutShapingParams],
    grid_size: int,
) -> None:
  if grid_size <= 0:
    raise ValueError('Grid size must be positive.')
  for shaping_param in shaping_params:
    if abs(shaping_param.move_delta) > grid_size:
      raise ValueError(
          'Move delta must be less than or equal to the grid size.'
      )


def _make_flat_foldable(
    cp: CreasePattern,
    candidates: FanOutCandidates,
    *,
    max_candidates: int | None,
    max_crease_modifications: int | None,
) -> CreasePattern | None:
  """Builds a new CP from candidates and attempts to make it flat-foldable."""
  all_creases = candidates.unmodified_creases + candidates.modified_creases
  if cp.light_mode:
    all_creases = flat_foldable.split_border_creases_at_tips(all_creases)
  potential_new_cp = CreasePattern(
      creases=all_creases,
      light_mode=cp.light_mode,
      merge_trivial=True,
  )

  matched_modified_creases = potential_new_cp.get_matching_creases(
      candidates.modified_creases
  )

  # Apply flat foldability fix first within the modified rectangles, then allow
  # wavefront propagation across rectangle boundaries if the base CP had no
  # pre-existing violations outside the rectangle.
  local_solution = flat_foldable.get_flat_foldable(
      cp=potential_new_cp,
      rectangles=tuple(candidates.modified_rectangles),
      unmodifiable_creases=matched_modified_creases,
      max_candidates=max_candidates,
      max_crease_modifications=max_crease_modifications,
  )
  if local_solution is not None:
    return local_solution

  base_score, _, _ = cp.vertex_flat_foldability_percentage()
  if base_score < 1.0:
    return None
  wavefront_max_cands = (
      25 if max_candidates is None else min(max_candidates, 25)
  )
  wavefront_max_mods = (
      12
      if max_crease_modifications is None
      else min(max_crease_modifications, 12)
  )
  return flat_foldable.get_flat_foldable(
      cp=potential_new_cp,
      rectangles=None,
      unmodifiable_creases=matched_modified_creases,
      max_candidates=wavefront_max_cands,
      max_crease_modifications=wavefront_max_mods,
  )


# NOTE: find better name for this type of shaping
def shape_cp_with_fan_out(
    cp: CreasePattern,
    *,
    shaping_params: Sequence[FanOutShapingParams],
    grid_size: int | None = None,
    max_candidates: int | None = 30,
    max_crease_modifications: int | None = 15,
) -> CreasePattern:
  """Applies fan out shaping to a crease pattern.

  For every horizontal/vertical crease line that extends the border but not all
  the way to the opposite border, move the border point by number of grid points
  specified in `shaping_params.move_delta`.

  This has the effect of "fanning out" some of the flaps that correspond to
  modified crease lines.

  Args:
    cp: The crease pattern to shape.
    shaping_params: A list of shaping parameters, each corresponding to a
      rectangle, fold types, orientations, and move delta. See
      `FanOutShapingParams` for more details
    grid_size: The grid size of the crease pattern. If None, then the grid size
      is inferred from `cp`.
    max_candidates: Maximum number of unique candidate CPs that can be added to
      the priority queue of the flat foldability solver. If None, no limit is
      applied.
    max_crease_modifications: Maximum number of crease modifications (rotations)
      allowed in a single candidate chain. Candidates exceeding this depth are
      not added to the priority queue of the flat foldability solver. If None,
      no limit is applied.

  Returns:
    The crease pattern with fanned out shaping.

  Raises:
    ValueError: If no valid fan out creases are found or the result is not
      flat-foldable.
  """
  if grid_size is None:
    grid_size = cp.grid_size
  _validate_shaping_params(shaping_params, grid_size)

  candidates, kept_fanned, final_unmodified = (
      _get_fan_out_candidates_with_fanned(cp, shaping_params, grid_size)
  )
  if candidates is None:
    raise ValueError('No valid fan out creases found.')

  flat_foldable_cp = _make_flat_foldable(
      cp,
      candidates,
      max_candidates=max_candidates,
      max_crease_modifications=max_crease_modifications,
  )
  if flat_foldable_cp is not None:
    return flat_foldable_cp

  # If fanning all matching pleats at once collides, try single fold types first
  # (when multiple fold types were allowed), then single-pleat subsets.
  if any(len(p.valid_fold_types) > 1 for p in shaping_params):
    for single_fold in (Fold.MOUNTAIN, Fold.VALLEY):
      single_params = [
          dataclasses.replace(
              p,
              valid_fold_types=(
                  [single_fold]
                  if single_fold in p.valid_fold_types
                  else list(p.valid_fold_types)
              ),
          )
          for p in shaping_params
      ]
      sub_cand = _get_fan_out_candidates(cp, single_params, grid_size)
      if sub_cand is not None:
        flat_foldable_cp = _make_flat_foldable(
            cp,
            sub_cand,
            max_candidates=max_candidates,
            max_crease_modifications=max_crease_modifications,
        )
        if flat_foldable_cp is not None:
          return flat_foldable_cp

  for sub_cand in _single_pleat_subset_candidates(
      kept_fanned, final_unmodified
  ):
    timeout_lib.maybe_timeout_cooperatively()
    flat_foldable_cp = _make_flat_foldable(
        cp,
        sub_cand,
        max_candidates=max_candidates,
        max_crease_modifications=max_crease_modifications,
    )
    if flat_foldable_cp is not None:
      return flat_foldable_cp

  raise ValueError('Fanned out crease pattern is not flat-foldable.')


def shape_cp_with_fan_out_heuristic(
    cp: CreasePattern,
    *,
    shaping_params: Sequence[FanOutShapingParams],
    grid_size: int | None = None,
    max_candidates: int | None = 30,
    max_crease_modifications: int | None = 15,
) -> CreasePattern:
  """Applies fan out shaping using a heuristic over all fold/orientation combos.

  Tries all 4 combinations of fold type (MOUNTAIN/VALLEY) and orientation
  (HORIZONTAL/VERTICAL), sorted by descending average length of modified
  creases. Returns the first combination that yields a flat-foldable CP.

  The `valid_fold_types` and `valid_orientations` fields in each shaping param
  must be empty; this function overrides them with each combination.

  Args:
    cp: The crease pattern to shape.
    shaping_params: A list of shaping parameters. `valid_fold_types` and
      `valid_orientations` must be empty collections.
    grid_size: The grid size of the crease pattern. If None, then the grid size
      is inferred from `cp`.
    max_candidates: Maximum number of unique candidate CPs that can be added to
      the priority queue of the flat foldability solver.
    max_crease_modifications: Maximum number of crease modifications allowed in
      a single candidate chain.

  Returns:
    The crease pattern with fanned out shaping.

  Raises:
    ValueError: If valid_fold_types/valid_orientations are non-empty, or if no
      combination yields a flat-foldable result.
  """
  for shaping_param in shaping_params:
    if shaping_param.valid_fold_types:
      raise ValueError(
          'valid_fold_types must be empty for heuristic mode. Use'
          ' shape_cp_with_fan_out for explicit fold type selection.'
      )
    if shaping_param.valid_orientations:
      raise ValueError(
          'valid_orientations must be empty for heuristic mode. Use'
          ' shape_cp_with_fan_out for explicit orientation selection.'
      )

  if grid_size is None:
    grid_size = cp.grid_size
  _validate_shaping_params(shaping_params, grid_size)

  all_candidates: list[FanOutCandidates] = []
  fallback_subsets: list[FanOutCandidates] = []
  for fold_type in (Fold.MOUNTAIN, Fold.VALLEY):
    for orientation in (Orientation.HORIZONTAL, Orientation.VERTICAL):
      combo_params = [
          dataclasses.replace(
              param,
              valid_fold_types=[fold_type],
              valid_orientations=[orientation],
          )
          for param in shaping_params
      ]
      candidates, kept_fanned, final_unmodified = (
          _get_fan_out_candidates_with_fanned(cp, combo_params, grid_size)
      )
      if candidates is not None:
        all_candidates.append(candidates)
        fallback_subsets.extend(
            _single_pleat_subset_candidates(kept_fanned, final_unmodified)
        )

  if not all_candidates:
    raise ValueError('No valid fan out creases found for any combination.')

  all_candidates.sort(key=lambda c: c.avg_modified_crease_length, reverse=True)

  for candidates in all_candidates + fallback_subsets:
    timeout_lib.maybe_timeout_cooperatively()
    flat_foldable_cp = _make_flat_foldable(
        cp,
        candidates,
        max_candidates=max_candidates,
        max_crease_modifications=max_crease_modifications,
    )
    if flat_foldable_cp is not None:
      return flat_foldable_cp

  raise ValueError(
      'No fold type/orientation combination produced a flat-foldable result.'
  )
