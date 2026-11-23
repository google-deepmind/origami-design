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

"""Fan-out shaping by shifting border pleats and repairing vertex angles."""

from collections.abc import Collection, Sequence
import dataclasses
import enum

from .. import timeout as timeout_lib
from ..crease_pattern import crease as crease_utils
from ..crease_pattern import crease_pattern as cp_utils
from ..crease_pattern import flat_foldable
from ..crease_pattern import point as point_utils
from . import shaping_utils

Point = point_utils.Point
Crease = crease_utils.Crease
CreasePattern = cp_utils.CreasePattern
Fold = crease_utils.Fold

_WAVEFRONT_MAX_CANDIDATES = 25
_WAVEFRONT_MAX_MODIFICATIONS = 12


class Orientation(enum.StrEnum):
  """Axis orientation of pleats eligible for fan-out shaping."""

  HORIZONTAL = 'horizontal'
  VERTICAL = 'vertical'


class CandidateMode(enum.StrEnum):
  """Scope mode for gathering candidate border-terminating pleats.

  Attributes:
    AUTO: Tries `STRICT`, then `SPLIT`, then `BORDER_TIP`, returning the first
      non-empty plan.
    STRICT: Unsplit creases wholly within a parameter rectangle.
    SPLIT: Splits border-terminating pleats at rectangle vertices first, then
      selects segments wholly within a parameter rectangle.
    BORDER_TIP: Splits border-terminating pleats at rectangle vertices, then
      selects any pleat whose border endpoint lies inside a parameter rectangle.
  """

  AUTO = 'auto'
  STRICT = 'strict'
  SPLIT = 'split'
  BORDER_TIP = 'border_tip'


@dataclasses.dataclass(frozen=True, kw_only=True)
class FanOutShapingParams:
  """Parameters specifying which border pleats to fan out and by how much.

  Attributes:
    valid_fold_types: Allowed fold types (`Fold.MOUNTAIN`, `Fold.VALLEY`).
    valid_orientations: Allowed pleat orientations (`HORIZONTAL`, `VERTICAL`).
    rectangles: Bounding rectangles restricting which pleats are shifted.
    move_delta: Signed border-tip shift in grid units (along `y` for horizontal
      pleats and along `x` for vertical pleats).
  """

  valid_fold_types: Collection[Fold]
  valid_orientations: Collection[Orientation | str]
  rectangles: Sequence[shaping_utils.Rectangle]
  move_delta: float


@dataclasses.dataclass(frozen=True, kw_only=True)
class FannedOutCrease:
  """A candidate shifted pleat paired with its source crease and repair box."""

  fanned_out_crease: Crease
  original_crease: Crease
  rectangle: shaping_utils.Rectangle


@dataclasses.dataclass(frozen=True, kw_only=True)
class FanOutPlan:
  """Collision-filtered set of fanned-out pleats and untouched base creases."""

  fanned: tuple[FannedOutCrease, ...]
  unmodified_creases: tuple[Crease, ...]

  @property
  def modified_creases(self) -> tuple[Crease, ...]:
    """Shifted replacement creases."""
    return tuple(item.fanned_out_crease for item in self.fanned)

  @property
  def modified_rectangles(self) -> frozenset[shaping_utils.Rectangle]:
    """Rectangles enclosing modified pleats for local flat-foldable repair."""
    return frozenset(item.rectangle for item in self.fanned)

  @property
  def avg_modified_crease_length(self) -> float:
    """Mean length of shifted creases."""
    return sum(item.fanned_out_crease.length for item in self.fanned) / len(
        self.fanned
    )

  def subset(self, indices: Sequence[int]) -> 'FanOutPlan | None':
    """Returns a sub-plan retaining only `indices` as shifted pleats."""
    keep_set = set(indices)
    chosen = tuple(
        item for idx, item in enumerate(self.fanned) if idx in keep_set
    )
    if not chosen:
      return None
    restored = tuple(
        item.original_crease
        for idx, item in enumerate(self.fanned)
        if idx not in keep_set
    )
    return FanOutPlan(
        fanned=chosen,
        unmodified_creases=self.unmodified_creases + restored,
    )

  def single_pleat_subsets(
      self, *, is_left: bool = False
  ) -> list['FanOutPlan']:
    """Returns 1-pleat fallback plans ordered shortest-first."""
    if len(self.fanned) <= 1:
      return []

    def _sort_key(idx: int) -> tuple[float, float, float]:
      orig = self.fanned[idx].original_crease
      mid_x = 0.5 * (orig.p1.x + orig.p2.x)
      mid_y = 0.5 * (orig.p1.y + orig.p2.y)
      canon_x = 1.0 - mid_x if is_left else mid_x
      return (round(orig.length, 6), round(canon_x, 6), round(mid_y, 6))

    plans: list[FanOutPlan] = []
    for idx in sorted(range(len(self.fanned)), key=_sort_key):
      sub = self.subset((idx,))
      if sub is not None:
        plans.append(sub)
    return plans


def _single_border_endpoints(
    crease: Crease, orientation: Orientation
) -> tuple[Point, Point] | None:
  """Returns `(border_tip, interior_tip)` if `crease` hits one border."""
  if orientation == Orientation.HORIZONTAL:
    if not crease.is_horizontal:
      return None
    left_on = crease.left_point.close_to_left_border
    right_on = crease.right_point.close_to_right_border
    if not (left_on ^ right_on):
      return None
    return (
        (crease.left_point, crease.right_point)
        if crease.close_to_left_border
        else (crease.right_point, crease.left_point)
    )
  if not crease.is_vertical:
    return None
  top_on = crease.top_point.close_to_top_border
  bottom_on = crease.bottom_point.close_to_bottom_border
  if not (top_on ^ bottom_on):
    return None
  return (
      (crease.top_point, crease.bottom_point)
      if crease.close_to_top_border
      else (crease.bottom_point, crease.top_point)
  )


def get_fanned_out_crease(
    crease: Crease,
    rectangle: shaping_utils.Rectangle,
    valid_fold_types: Collection[Fold],
    valid_orientations: Collection[Orientation | str],
    move_delta: float,
) -> Crease | None:
  """Shifts the border endpoint of `crease` by `move_delta` if eligible."""
  if not valid_fold_types:
    raise ValueError('valid_fold_types must not be empty.')
  if not valid_orientations:
    raise ValueError('valid_orientations must not be empty.')
  if (
      crease.fold not in valid_fold_types
      or not rectangle.crease_is_within(crease)
      or crease.is_overlapping_border
  ):
    return None

  shifted_tip: Point | None = None
  anchor_tip: Point | None = None
  for orientation, shift_y in (
      (Orientation.HORIZONTAL, True),
      (Orientation.VERTICAL, False),
  ):
    if orientation not in valid_orientations:
      continue
    tips = _single_border_endpoints(crease, orientation)
    if tips is None:
      continue
    border_tip, anchor_tip = tips
    new_x = border_tip.x if shift_y else border_tip.x + move_delta
    new_y = border_tip.y + move_delta if shift_y else border_tip.y
    try:
      shifted_tip = Point(x=new_x, y=new_y)
    except ValueError:
      return None
    break

  if shifted_tip is None or anchor_tip is None:
    return None
  return Crease(
      shifted_tip,
      anchor_tip,
      fold=crease.fold,
      fold_percent=crease.fold_percent,
  )


def _point_in_rect(pt: Point, rect: shaping_utils.Rectangle) -> bool:
  tol = point_utils.DEFAULT_TOLERANCE
  return (
      rect.x_start - tol <= pt.x <= rect.x_end + tol
      and rect.y_start - tol <= pt.y <= rect.y_end + tol
  )


def _split_merged_border_creases(
    cp: CreasePattern,
    shaping_params: Sequence[FanOutShapingParams],
) -> list[Crease]:
  """Splits border-touching H/V pleats at interior vertices in `rectangles`."""
  rects: list[shaping_utils.Rectangle] = []
  orientations: set[Orientation | str] = set()
  for param in shaping_params:
    orientations.update(param.valid_orientations)
    for rect in param.rectangles:
      clipped = rect.clip()
      if clipped is not None:
        rects.append(clipped)
  if not rects:
    return list(cp.creases)

  vertex_pts = [
      v.point
      for v in cp.vertices
      if any(_point_in_rect(v.point, r) for r in rects)
  ]
  if not vertex_pts:
    return list(cp.creases)

  tol = point_utils.DEFAULT_TOLERANCE
  want_h = Orientation.HORIZONTAL in orientations
  want_v = Orientation.VERTICAL in orientations
  out: list[Crease] = []
  for c in cp.creases:
    if c.fold == Fold.BORDER:
      out.append(c)
      continue
    split_h = (
        want_h
        and _single_border_endpoints(c, Orientation.HORIZONTAL) is not None
        and any(
            r.y_start - tol <= c.p1.y <= r.y_end + tol
            and not r.crease_is_within(c)
            for r in rects
        )
    )
    split_v = (
        want_v
        and _single_border_endpoints(c, Orientation.VERTICAL) is not None
        and any(
            r.x_start - tol <= c.p1.x <= r.x_end + tol
            and not r.crease_is_within(c)
            for r in rects
        )
    )
    if split_h or split_v:
      out.extend(crease_utils.split_crease_at_points(c, vertex_pts))
    else:
      out.append(c)
  return out


def _collect_and_filter_plan(
    creases: Sequence[Crease],
    shaping_params: Sequence[FanOutShapingParams],
    grid_size: int,
    *,
    allow_border_tip_only: bool,
) -> FanOutPlan | None:
  """Collects candidate fanned pleats and filters interior crossings."""
  full_paper = shaping_utils.Rectangle(
      name='paper', x_start=0.0, y_start=0.0, x_len=1.0, y_len=1.0
  )
  unmodified: list[Crease] = []
  potential: list[FannedOutCrease] = []

  for crease in creases:
    timeout_lib.maybe_timeout_cooperatively()
    matched: FannedOutCrease | None = None
    for param in shaping_params:
      delta = param.move_delta / grid_size
      for rect in param.rectangles:
        clipped = rect.clip()
        if clipped is None:
          continue
        if not allow_border_tip_only:
          if not rect.crease_is_within(crease):
            continue
          target_rect = clipped
          record_rect = rect
        else:
          border_pt = (
              crease.p1
              if crease.p1.is_border
              else (crease.p2 if crease.p2.is_border else None)
          )
          if border_pt is None or not _point_in_rect(border_pt, clipped):
            continue
          target_rect = full_paper
          min_x = min(clipped.x_start, crease.p1.x, crease.p2.x)
          max_x = max(clipped.x_end, crease.p1.x, crease.p2.x)
          min_y = min(clipped.y_start, crease.p1.y, crease.p2.y)
          max_y = max(clipped.y_end, crease.p1.y, crease.p2.y)
          record_rect = shaping_utils.Rectangle(
              name=rect.name,
              x_start=min_x,
              y_start=min_y,
              x_len=max(max_x - min_x, 1e-6),
              y_len=max(max_y - min_y, 1e-6),
          )
        fanned = get_fanned_out_crease(
            crease,
            target_rect,
            param.valid_fold_types,
            param.valid_orientations,
            delta,
        )
        if fanned is not None:
          matched = FannedOutCrease(
              fanned_out_crease=fanned,
              original_crease=crease,
              rectangle=record_rect,
          )
          break
      if matched is not None:
        break
    if matched is not None:
      potential.append(matched)
    else:
      unmodified.append(crease)

  kept: list[FannedOutCrease] = []
  for item in potential:
    timeout_lib.maybe_timeout_cooperatively()
    shifted = item.fanned_out_crease
    crosses = False
    for other in unmodified:
      timeout_lib.maybe_timeout_cooperatively()
      if other.fold == Fold.BORDER:
        continue
      hit = shifted.vertex_with_crease(other)
      if hit is not None and not shifted.is_tip(hit):
        unmodified.append(item.original_crease)
        crosses = True
        break
    if not crosses:
      kept.append(item)

  if not kept:
    return None
  return FanOutPlan(fanned=tuple(kept), unmodified_creases=tuple(unmodified))


def find_fan_out_plan(
    cp: CreasePattern,
    shaping_params: Sequence[FanOutShapingParams],
    grid_size: int,
    *,
    mode: CandidateMode = CandidateMode.AUTO,
) -> FanOutPlan | None:
  """Finds collision-free candidate fan-out pleats under `mode`."""
  if mode in (CandidateMode.AUTO, CandidateMode.STRICT):
    plan = _collect_and_filter_plan(
        cp.creases, shaping_params, grid_size, allow_border_tip_only=False
    )
    if plan is not None or mode == CandidateMode.STRICT:
      return plan

  split_creases = _split_merged_border_creases(cp, shaping_params)
  if mode in (CandidateMode.AUTO, CandidateMode.SPLIT):
    plan = _collect_and_filter_plan(
        split_creases, shaping_params, grid_size, allow_border_tip_only=False
    )
    if plan is not None or mode == CandidateMode.SPLIT:
      return plan

  return _collect_and_filter_plan(
      split_creases, shaping_params, grid_size, allow_border_tip_only=True
  )


def resolve_fan_out_plan(
    cp: CreasePattern,
    plan: FanOutPlan,
    *,
    max_candidates: int | None = 30,
    max_crease_modifications: int | None = 15,
) -> CreasePattern | None:
  """Materializes `plan` and repairs resulting flat-foldability violations."""
  all_creases: list[Crease] = list(plan.unmodified_creases) + list(
      plan.modified_creases
  )
  if cp.light_mode:
    all_creases = flat_foldable.split_border_creases_at_tips(all_creases)
  candidate_cp = CreasePattern(
      creases=all_creases, light_mode=cp.light_mode, merge_trivial=True
  )
  locked = candidate_cp.get_matching_creases(list(plan.modified_creases))
  local = flat_foldable.get_flat_foldable(
      cp=candidate_cp,
      rectangles=tuple(plan.modified_rectangles),
      unmodifiable_creases=locked,
      max_candidates=max_candidates,
      max_crease_modifications=max_crease_modifications,
  )
  if local is not None:
    return local

  base_score, _, _ = cp.vertex_flat_foldability_percentage()
  if base_score < 1.0:
    return None
  wave_cands = (
      _WAVEFRONT_MAX_CANDIDATES
      if max_candidates is None
      else min(max_candidates, _WAVEFRONT_MAX_CANDIDATES)
  )
  wave_mods = (
      _WAVEFRONT_MAX_MODIFICATIONS
      if max_crease_modifications is None
      else min(max_crease_modifications, _WAVEFRONT_MAX_MODIFICATIONS)
  )
  return flat_foldable.get_flat_foldable(
      cp=candidate_cp,
      rectangles=None,
      unmodifiable_creases=locked,
      max_candidates=wave_cands,
      max_crease_modifications=wave_mods,
  )


def _validate_shaping_params(
    shaping_params: Sequence[FanOutShapingParams], grid_size: int
) -> None:
  if grid_size <= 0:
    raise ValueError('Grid size must be positive.')
  for param in shaping_params:
    if abs(param.move_delta) > grid_size:
      raise ValueError('Move delta must not exceed grid size.')


def shape_cp_with_fan_out(
    cp: CreasePattern,
    *,
    shaping_params: Sequence[FanOutShapingParams],
    grid_size: int | None = None,
    max_candidates: int | None = 30,
    max_crease_modifications: int | None = 15,
) -> CreasePattern:
  """Applies explicit fan-out shaping and restores local flat-foldability."""
  resolved_grid = cp.grid_size if grid_size is None else grid_size
  _validate_shaping_params(shaping_params, resolved_grid)

  plan = find_fan_out_plan(cp, shaping_params, resolved_grid)
  if plan is None:
    raise ValueError('No valid fan out creases found.')

  candidate_plans: list[FanOutPlan] = [plan]
  if any(len(p.valid_fold_types) > 1 for p in shaping_params):
    for single_fold in (Fold.MOUNTAIN, Fold.VALLEY):
      single_params = [
          dataclasses.replace(
              p,
              valid_fold_types=(
                  (single_fold,)
                  if single_fold in p.valid_fold_types
                  else tuple(p.valid_fold_types)
              ),
          )
          for p in shaping_params
      ]
      sub_plan = find_fan_out_plan(cp, single_params, resolved_grid)
      if sub_plan is not None:
        candidate_plans.append(sub_plan)
  candidate_plans.extend(plan.single_pleat_subsets())

  for cand_plan in candidate_plans:
    timeout_lib.maybe_timeout_cooperatively()
    solved = resolve_fan_out_plan(
        cp,
        cand_plan,
        max_candidates=max_candidates,
        max_crease_modifications=max_crease_modifications,
    )
    if solved is not None:
      return solved
  raise ValueError('Fanned out crease pattern is not flat-foldable.')


def shape_cp_with_fan_out_heuristic(
    cp: CreasePattern,
    *,
    shaping_params: Sequence[FanOutShapingParams],
    grid_size: int | None = None,
    max_candidates: int | None = 30,
    max_crease_modifications: int | None = 15,
) -> CreasePattern:
  """Searches all `(MOUNTAIN/VALLEY, HORIZONTAL/VERTICAL)` fan-out plans."""
  for param in shaping_params:
    if param.valid_fold_types:
      raise ValueError(
          'valid_fold_types must be empty for heuristic mode; use '
          'shape_cp_with_fan_out for explicit selection.'
      )
    if param.valid_orientations:
      raise ValueError(
          'valid_orientations must be empty for heuristic mode; use '
          'shape_cp_with_fan_out for explicit selection.'
      )

  resolved_grid = cp.grid_size if grid_size is None else grid_size
  _validate_shaping_params(shaping_params, resolved_grid)

  primary: list[FanOutPlan] = []
  fallbacks: list[FanOutPlan] = []
  for fold_type in (Fold.MOUNTAIN, Fold.VALLEY):
    for orientation in (Orientation.HORIZONTAL, Orientation.VERTICAL):
      combo = [
          dataclasses.replace(
              p,
              valid_fold_types=(fold_type,),
              valid_orientations=(orientation,),
          )
          for p in shaping_params
      ]
      plan = find_fan_out_plan(cp, combo, resolved_grid)
      if plan is not None:
        primary.append(plan)
        fallbacks.extend(plan.single_pleat_subsets())

  if not primary:
    raise ValueError('No valid fan out creases found for any combination.')
  primary.sort(key=lambda p: p.avg_modified_crease_length, reverse=True)

  for plan in primary + fallbacks:
    timeout_lib.maybe_timeout_cooperatively()
    solved = resolve_fan_out_plan(
        cp,
        plan,
        max_candidates=max_candidates,
        max_crease_modifications=max_crease_modifications,
    )
    if solved is not None:
      return solved
  raise ValueError(
      'No fold type/orientation combination produced a flat-foldable result.'
  )
