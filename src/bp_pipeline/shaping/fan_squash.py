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

"""Fan-out and squash-fold actuators for spreading pleats and ridges."""

from collections.abc import Iterator, Sequence
import dataclasses
import datetime
import math
import time

from .. import folder_3d
from .. import timeout as timeout_lib
from ..crease_pattern import crease as crease_utils
from ..crease_pattern import crease_pattern as crease_pattern_lib
from ..crease_pattern import flat_foldable
from . import fan_out_shaping
from . import fidelity
from . import shaping_elements
from . import shaping_utils
from . import squash_sink_shaping
from . import state as state_lib

_FOLD_TYPES = {
    'mountain': crease_utils.Fold.MOUNTAIN,
    'valley': crease_utils.Fold.VALLEY,
}
_ORIENTATIONS = {
    'horizontal': fan_out_shaping.Orientation.HORIZONTAL,
    'vertical': fan_out_shaping.Orientation.VERTICAL,
}

# Solver & search constants for incremental pleat fanning.
_SOLVER_CANDIDATES_PER_PLEAT = 12
_SOLVER_MODS_PER_PLEAT = 5
_MIN_SOLVER_CANDIDATES = 12
_MIN_SOLVER_MODS = 6
_MAX_SOLVER_CANDIDATES = 60
_MAX_SOLVER_MODS = 25
_PLEAT_DELTA_FACTORS = (1.0, 0.5, 0.25)
_MAX_FAN_PLEATS = 6
_MAX_BILATERAL_PLEATS_PER_SIDE = 3
_MAX_MISSED_PLEATS = 2
_MIN_PLEAT_LENGTH_FRACTION = 0.5
_PLEAT_RECT_MARGIN = 1.5
_MIN_FAN_WIDTH_RATIO = 1.05


@dataclasses.dataclass(frozen=True)
class _FanPlan:
  """Ordered pleats and target shifts for one `(orientation, fold_type)`."""

  label: str
  orientation: fan_out_shaping.Orientation
  kept_fanned: tuple[fan_out_shaping.FannedOutCrease, ...]
  unmodified: tuple[crease_utils.Crease, ...]
  order: tuple[int, ...]
  deltas: tuple[float, ...]
  bilateral: bool
  avg_length: float


@dataclasses.dataclass(frozen=True)
class _FanCandidate:
  """Verified fan-out result with width and pleat diagnostics."""

  state: state_lib.ShapingState
  width_ratio: float
  new_creases: int
  pleats: int
  label: str
  weak: bool

  def beats(self, other: '_FanCandidate | None') -> bool:
    """Returns True if `self` outranks `other`."""
    if other is None:
      return True
    if self.weak != other.weak:
      return not self.weak
    if not math.isclose(self.width_ratio, other.width_ratio, abs_tol=1e-3):
      return self.width_ratio > other.width_ratio
    if self.pleats != other.pleats:
      return self.pleats > other.pleats
    return self.new_creases > other.new_creases


def _require_flap(
    state: state_lib.ShapingState, element_name: str
) -> shaping_elements.ShapingFlap:
  """Returns the `ShapingFlap` for `element_name`, rejecting rivers."""
  elem = state.element(element_name)
  if not isinstance(elem, shaping_elements.ShapingFlap):
    raise ValueError(
        f'{element_name!r} is a river; fan_out_fold and squash_fold require a'
        ' flap.'
    )
  return elem


def _flap_rect(flap: shaping_elements.ShapingFlap) -> shaping_utils.Rectangle:
  return shaping_utils.Rectangle(
      name=flap.name,
      x_start=flap.x_start,
      y_start=flap.y_start,
      x_len=flap.x_len,
      y_len=flap.y_len,
  )


def _chain_rect(
    state: state_lib.ShapingState, element_name: str
) -> shaping_utils.Rectangle | None:
  """Returns the merged chain rectangle when `element_name` is in a chain."""
  _, merged_names = state.tree.merge_linear_chains()
  flaps, _ = state.merged_shaping_elements
  for merged_name, sub_names in merged_names.items():
    if element_name in sub_names and len(sub_names) > 1:
      for flap in flaps:
        if flap.name == merged_name:
          return _flap_rect(flap)
  return None


def _base_section_widths(
    state: state_lib.ShapingState, element_name: str
) -> dict[float, float]:
  """Measures baseline 3-D section widths of `element_name` at 0.25/0.5/0.75."""
  try:
    model = folder_3d.fold_3d(state.pattern)
    geom = fidelity.measure(state.tree, state.paper_layout, model)
  except (RuntimeError, ValueError, AssertionError, KeyError):
    return {}
  return {
      frac: float(
          geom['sections'].get(f'{element_name}@{frac}', {}).get('width', 0.0)
      )
      for frac in (0.25, 0.5, 0.75)
  }


def _section_width_ratio(
    state: state_lib.ShapingState,
    element_name: str,
    base_widths: dict[float, float],
    candidate: crease_pattern_lib.CreasePattern,
    *,
    model: folder_3d.FoldedModel | None = None,
) -> float:
  """Returns candidate-to-baseline folded section width ratio."""
  if not base_widths:
    return 1.0
  try:
    folded = model or folder_3d.fold_3d(candidate)
    geom = fidelity.measure(state.tree, state.paper_layout, folded)
  except (RuntimeError, ValueError, AssertionError, KeyError):
    return 1.0
  for frac in (0.5, 0.25, 0.75):
    w0 = base_widths.get(frac, 0.0)
    if w0 > 0.0:
      w1 = float(
          geom['sections'].get(f'{element_name}@{frac}', {}).get('width', 0.0)
      )
      return round(w1 / w0, 3)
  return 1.0


# ------------------------------------------------------------------------------
# Fan-out shaping.
# ------------------------------------------------------------------------------


def _spans_paper(
    rect: shaping_utils.Rectangle, orientation: fan_out_shaping.Orientation
) -> bool:
  clipped = rect.clip()
  if clipped is None:
    return False
  if orientation == fan_out_shaping.Orientation.VERTICAL:
    return clipped.x_start <= 0.05 and clipped.x_end >= 0.95
  return clipped.y_start <= 0.05 and clipped.y_end >= 0.95


def _split_half_rects(
    rect: shaping_utils.Rectangle,
    orientation: fan_out_shaping.Orientation,
    grid_size: int,
) -> tuple[shaping_utils.Rectangle, shaping_utils.Rectangle]:
  """Splits a paper-spanning rectangle around the symmetry midline."""
  clipped = rect.clip()
  if clipped is None:
    return rect, rect
  gap = 0.25 / max(grid_size, 1)
  if orientation == fan_out_shaping.Orientation.VERTICAL:
    return (
        shaping_utils.Rectangle(
            name=f'{rect.name}_a',
            x_start=0.0,
            y_start=clipped.y_start,
            x_len=0.5 - gap,
            y_len=clipped.y_len,
        ),
        shaping_utils.Rectangle(
            name=f'{rect.name}_b',
            x_start=0.5 + gap,
            y_start=clipped.y_start,
            x_len=0.5 - gap,
            y_len=clipped.y_len,
        ),
    )
  return (
      shaping_utils.Rectangle(
          name=f'{rect.name}_a',
          x_start=clipped.x_start,
          y_start=0.0,
          x_len=clipped.x_len,
          y_len=0.5 - gap,
      ),
      shaping_utils.Rectangle(
          name=f'{rect.name}_b',
          x_start=clipped.x_start,
          y_start=0.5 + gap,
          x_len=clipped.x_len,
          y_len=0.5 - gap,
      ),
  )


def _pleat_rect(
    fanned: fan_out_shaping.FannedOutCrease, grid_size: int
) -> shaping_utils.Rectangle:
  """Builds a tight local solver box around one fanned pleat."""
  margin = _PLEAT_RECT_MARGIN / max(grid_size, 1)
  pts = (
      fanned.original_crease.p1,
      fanned.original_crease.p2,
      fanned.fanned_out_crease.p1,
      fanned.fanned_out_crease.p2,
  )
  x0 = max(0.0, min(p.x for p in pts) - margin)
  x1 = min(1.0, max(p.x for p in pts) + margin)
  y0 = max(0.0, min(p.y for p in pts) - margin)
  y1 = min(1.0, max(p.y for p in pts) + margin)
  return shaping_utils.Rectangle(
      name=fanned.rectangle.name,
      x_start=x0,
      y_start=y0,
      x_len=max(x1 - x0, 1e-6),
      y_len=max(y1 - y0, 1e-6),
  )


def _pleat_coord(
    fanned: fan_out_shaping.FannedOutCrease,
    orientation: fan_out_shaping.Orientation,
) -> float:
  oc = fanned.original_crease
  if orientation == fan_out_shaping.Orientation.VERTICAL:
    return 0.5 * (oc.p1.x + oc.p2.x)
  return 0.5 * (oc.p1.y + oc.p2.y)


def _middle_out(indices: Sequence[int]) -> list[int]:
  if not indices:
    return []
  mid = (len(indices) - 1) // 2
  ordered = [indices[mid]]
  for step in range(1, len(indices)):
    for idx in (mid + step, mid - step):
      if 0 <= idx < len(indices):
        ordered.append(indices[idx])
  return ordered


def _mirror_crease(
    crease: crease_utils.Crease, vertical: bool
) -> crease_utils.Crease:
  if vertical:
    p1 = dataclasses.replace(crease.p1_init, x=1.0 - crease.p1_init.x)
    p2 = dataclasses.replace(crease.p2_init, x=1.0 - crease.p2_init.x)
  else:
    p1 = dataclasses.replace(crease.p1_init, y=1.0 - crease.p1_init.y)
    p2 = dataclasses.replace(crease.p2_init, y=1.0 - crease.p2_init.y)
  return crease_utils.Crease(
      p1,
      p2,
      fold=crease.fold,
      fold_percent=crease.fold_percent,
      elevation=crease.elevation,
  )


def _is_mirror_symmetric(
    pattern: crease_pattern_lib.CreasePattern, vertical: bool
) -> bool:
  keys = {(state_lib.crease_key(c), c.fold) for c in pattern.creases}
  mirrored = {
      (state_lib.crease_key(_mirror_crease(c, vertical)), c.fold)
      for c in pattern.creases
  }
  return keys == mirrored


def _mirror_half(
    pattern: crease_pattern_lib.CreasePattern,
    *,
    vertical: bool,
    keep_second: bool,
) -> crease_pattern_lib.CreasePattern:
  """Mirrors one shaped half of `pattern` onto the opposite half."""
  tol = 1e-6
  creases: list[crease_utils.Crease] = []
  for c in pattern.creases:
    a, b = (
        (c.p1_init.x, c.p2_init.x) if vertical else (c.p1_init.y, c.p2_init.y)
    )
    lo, hi = (a, b) if a <= b else (b, a)
    in_first = hi <= 0.5 + tol
    in_second = lo >= 0.5 - tol
    if (in_first and keep_second) or (in_second and not keep_second):
      continue
    creases.append(c)
    if (in_second and keep_second) or (in_first and not keep_second):
      creases.append(_mirror_crease(c, vertical))
  return crease_pattern_lib.CreasePattern(
      creases=creases,
      grid_size=-1,
      light_mode=pattern.light_mode,
      tolerance=pattern.tolerance,
  )


def _bounded_flat_foldable(
    timeout_s: float,
    *,
    cp: crease_pattern_lib.CreasePattern,
    rectangles: tuple[shaping_utils.Rectangle, ...] | None,
    unmodifiable_creases: list[crease_utils.Crease],
    max_candidates: int,
    max_crease_modifications: int,
) -> tuple[crease_pattern_lib.CreasePattern | None, bool]:
  """Calls `flat_foldable.get_flat_foldable` under a cooperative timeout."""
  limit = datetime.timedelta(seconds=max(0.05, timeout_s))
  outer = timeout_lib.get_timeout()
  if outer is not None:
    limit = min(limit, outer.remaining)
    if limit <= datetime.timedelta(0):
      return None, True
  try:
    with timeout_lib.cooperative_timeout(limit):
      solved = flat_foldable.get_flat_foldable(
          cp=cp,
          rectangles=rectangles,
          unmodifiable_creases=unmodifiable_creases,
          max_candidates=max_candidates,
          max_crease_modifications=max_crease_modifications,
      )
      return solved, False
  except timeout_lib.CooperativeTimeoutError:
    return None, True


def _find_fan_out_plan_prefer_more(
    cp: crease_pattern_lib.CreasePattern,
    params: Sequence[fan_out_shaping.FanOutShapingParams],
    grid_size: int,
) -> fan_out_shaping.FanOutPlan | None:
  """Selects AUTO vs BORDER_TIP fan-out plan, preferring more creases."""
  strict_plan = fan_out_shaping.find_fan_out_plan(
      cp, params, grid_size, mode=fan_out_shaping.CandidateMode.AUTO
  )
  loose_plan = fan_out_shaping.find_fan_out_plan(
      cp, params, grid_size, mode=fan_out_shaping.CandidateMode.BORDER_TIP
  )
  if loose_plan is not None and (
      strict_plan is None or len(loose_plan.fanned) > len(strict_plan.fanned)
  ):
    return loose_plan
  return strict_plan


def _solve_fan_candidates(
    state: state_lib.ShapingState,
    candidates: fan_out_shaping.FanOutPlan,
) -> Iterator[crease_pattern_lib.CreasePattern]:
  """Yields local and whole-paper flat-foldable solutions for `candidates`."""
  n_pleats = max(1, len(candidates.modified_creases))
  max_cands = min(
      _MAX_SOLVER_CANDIDATES,
      max(_MIN_SOLVER_CANDIDATES, _SOLVER_CANDIDATES_PER_PLEAT * n_pleats),
  )
  max_mods = min(
      _MAX_SOLVER_MODS, max(_MIN_SOLVER_MODS, _SOLVER_MODS_PER_PLEAT * n_pleats)
  )
  pattern = state.pattern
  all_creases = list(candidates.unmodified_creases) + list(
      candidates.modified_creases
  )
  if pattern.light_mode:
    all_creases = flat_foldable.split_border_creases_at_tips(all_creases)
  potential = crease_pattern_lib.CreasePattern(
      creases=all_creases,
      grid_size=-1,
      light_mode=pattern.light_mode,
      merge_trivial=True,
  )
  matched = potential.get_matching_creases(list(candidates.modified_creases))
  timeout_s = state.config.fan_out_solver_timeout_s
  local, timed_out = _bounded_flat_foldable(
      timeout_s,
      cp=potential,
      rectangles=tuple(candidates.modified_rectangles),
      unmodifiable_creases=matched,
      max_candidates=max_cands,
      max_crease_modifications=max_mods,
  )
  if local is not None:
    yield local
  if timed_out or not pattern.local_flat_foldability():
    return
  whole, _ = _bounded_flat_foldable(
      timeout_s,
      cp=potential,
      rectangles=None,
      unmodifiable_creases=matched,
      max_candidates=min(max_cands, 25),
      max_crease_modifications=min(max_mods, 12),
  )
  if whole is not None:
    yield whole


def _accept_fan_cp(
    state: state_lib.ShapingState,
    solved_cp: crease_pattern_lib.CreasePattern,
    *,
    open_percent: float,
    verify_fold: bool,
) -> state_lib.ShapingState | None:
  """Restores borders, opens new creases when requested, and checks folding."""
  cleaned = state_lib.preserve_base_borders(state.pattern, solved_cp)
  if state_lib.new_crease_count(state.pattern, cleaned) == 0:
    return None
  old_keys = {state_lib.crease_key(c) for c in state.pattern.creases}
  new_keys = {
      state_lib.crease_key(c)
      for c in cleaned.creases
      if c.mountain_or_valley() and state_lib.crease_key(c) not in old_keys
  }
  pcts = [open_percent]
  if open_percent < 1.0:
    pcts.extend(p for p in (0.7, 0.85, 1.0) if p > open_percent)
  fallback: state_lib.ShapingState | None = None
  for pct in pcts:
    target = (
        state_lib.with_fold_percent(cleaned, new_keys, pct)
        if pct < 1.0
        else cleaned
    )
    accepted, _, model = state_lib.accept_pattern(
        state,
        target,
        allow_3d_vertices=pct < 1.0,
        verify_fold=verify_fold and state.config.check_flat_foldability,
    )
    if accepted is None:
      continue
    if (
        pct < 1.0
        and model is not None
        and model.vertex_error
        > fidelity.DEFAULT_CONFIG.clean_gate_max_vertex_error
    ):
      if fallback is None:
        fallback = accepted
      continue
    return accepted
  return fallback


def _build_fan_plan(
    state: state_lib.ShapingState,
    rect: shaping_utils.Rectangle,
    orientation: fan_out_shaping.Orientation,
    fold_types: Sequence[crease_utils.Fold],
    *,
    move_delta: float,
    is_left: bool,
    spread: bool,
) -> _FanPlan | None:
  """Builds an ordered pleat plan for one `(rect, orientation, fold_types)`."""
  gs = state.solving_grid_size
  bilateral = _spans_paper(rect, orientation)
  signed_delta = move_delta
  if (
      is_left
      and orientation == fan_out_shaping.Orientation.VERTICAL
      and not bilateral
  ):
    signed_delta = -move_delta
  if bilateral:
    first, second = _split_half_rects(rect, orientation, gs)
    rect_deltas = [(first, signed_delta), (second, -signed_delta)]
  else:
    rect_deltas = [(rect, signed_delta)]
  params = [
      fan_out_shaping.FanOutShapingParams(
          valid_fold_types=list(fold_types),
          valid_orientations=[orientation],
          rectangles=[r],
          move_delta=d,
      )
      for r, d in rect_deltas
  ]
  cand = _find_fan_out_plan_prefer_more(state.pattern, params, gs)
  if cand is None or not cand.fanned:
    return None
  kept = cand.fanned
  unmod = cand.unmodified_creases
  max_len = max(fc.original_crease.length for fc in kept)
  long_idx = sorted(
      (
          i
          for i, fc in enumerate(kept)
          if fc.original_crease.length >= _MIN_PLEAT_LENGTH_FRACTION * max_len
      ),
      key=lambda i: _pleat_coord(kept[i], orientation),
  )
  if not long_idx:
    return None

  order: list[int] = []
  deltas: list[float] = []
  if bilateral:
    left_idx = [
        i for i in long_idx if _pleat_coord(kept[i], orientation) < 0.5
    ][:_MAX_BILATERAL_PLEATS_PER_SIDE]
    right_idx = [
        i
        for i in reversed(long_idx)
        if _pleat_coord(kept[i], orientation) >= 0.5
    ][:_MAX_BILATERAL_PLEATS_PER_SIDE]
    pairs = list(zip(left_idx, right_idx))
    for k, (li, ri) in enumerate(pairs):
      scale = (
          1.0 - 0.45 * (k / max(1, len(pairs) - 1)) if len(pairs) >= 2 else 1.0
      )
      d = signed_delta * scale
      order.extend((li, ri))
      deltas.extend((d, -d))
  else:
    order = _middle_out(long_idx)[:_MAX_FAN_PLEATS]
    coords = [_pleat_coord(kept[i], orientation) for i in order]
    lo, hi = min(coords), max(coords)
    span = hi - lo
    for coord in coords:
      if span <= 1e-6:
        deltas.append(signed_delta)
      elif spread:
        centered = 2.0 * (coord - lo) / span - 1.0
        if abs(centered) < 1e-4:
          centered = 0.35
        deltas.append(abs(move_delta) * centered)
      else:
        rel = (coord - lo) / span if signed_delta >= 0 else (hi - coord) / span
        deltas.append(signed_delta * (0.55 + 0.45 * rel))

  if not order:
    return None
  label = f'{orientation.value}/' + '+'.join(
      ft.name.lower() for ft in fold_types
  )
  return _FanPlan(
      label=label,
      orientation=orientation,
      kept_fanned=tuple(kept),
      unmodified=tuple(unmod),
      order=tuple(order),
      deltas=tuple(deltas),
      bilateral=bilateral,
      avg_length=cand.avg_modified_crease_length,
  )


def _fan_single_pleat(
    state: state_lib.ShapingState,
    pleat: fan_out_shaping.FannedOutCrease,
    orientation: fan_out_shaping.Orientation,
    *,
    move_delta: float,
    open_percent: float,
    verify_fold: bool,
) -> state_lib.ShapingState | None:
  """Solves a single-pleat fan step on `state.pattern`."""
  if math.isclose(move_delta, 0.0, abs_tol=1e-4):
    return None
  gs = state.solving_grid_size
  if abs(move_delta) > gs:
    move_delta = math.copysign(float(gs), move_delta)
  params = [
      fan_out_shaping.FanOutShapingParams(
          valid_fold_types=[pleat.original_crease.fold],
          valid_orientations=[orientation],
          rectangles=[_pleat_rect(pleat, gs)],
          move_delta=move_delta,
      )
  ]
  plan = _find_fan_out_plan_prefer_more(state.pattern, params, gs)
  if plan is None:
    return None
  kept = plan.fanned
  unmod = plan.unmodified_creases
  target_key = state_lib.crease_key(pleat.original_crease)
  chosen = next(
      (
          i
          for i, fc in enumerate(kept)
          if state_lib.crease_key(fc.original_crease) == target_key
      ),
      None,
  )
  if chosen is None:
    return None
  rest = tuple(kept[j].original_crease for j in range(len(kept)) if j != chosen)
  chosen_item = dataclasses.replace(
      kept[chosen], rectangle=_pleat_rect(kept[chosen], gs)
  )
  bundle = fan_out_shaping.FanOutPlan(
      fanned=(chosen_item,),
      unmodified_creases=unmod + rest,
  )
  for solved_cp in _solve_fan_candidates(state, bundle):
    accepted = _accept_fan_cp(
        state, solved_cp, open_percent=open_percent, verify_fold=verify_fold
    )
    if accepted is not None:
      return accepted
  return None


def _run_incremental_fan(
    state: state_lib.ShapingState,
    plan: _FanPlan,
    *,
    open_percent: float,
    verify_fold: bool,
    deadline: float,
) -> tuple[state_lib.ShapingState, int]:
  """Grows a fan one pleat at a time, adapting delta factors on failure."""
  current = state
  fanned = 0
  first_factor = 0
  misses = 0
  for idx, delta in zip(plan.order, plan.deltas):
    for k in range(first_factor, len(_PLEAT_DELTA_FACTORS)):
      if time.perf_counter() >= deadline:
        return current, fanned
      nxt = _fan_single_pleat(
          current,
          plan.kept_fanned[idx],
          plan.orientation,
          move_delta=delta * _PLEAT_DELTA_FACTORS[k],
          open_percent=open_percent,
          verify_fold=verify_fold,
      )
      if nxt is not None:
        current = nxt
        fanned += 1
        first_factor = k
        break
    else:
      misses += 1
      if misses >= _MAX_MISSED_PLEATS:
        break
  return current, fanned


def _evaluate_plan(
    state: state_lib.ShapingState,
    element_name: str,
    plan: _FanPlan,
    base_widths: dict[float, float],
    *,
    open_percent: float,
    verify_fold: bool,
    deadline: float,
) -> _FanCandidate | None:
  """Executes `plan` (using half-solve + mirror on bilateral symmetric CPs)."""
  vertical = plan.orientation == fan_out_shaping.Orientation.VERTICAL
  if plan.bilateral and _is_mirror_symmetric(state.pattern, vertical):
    best: _FanCandidate | None = None
    for keep_second in (True, False):
      if time.perf_counter() >= deadline:
        break
      pairs = [
          (i, d)
          for i, d in zip(plan.order, plan.deltas)
          if (_pleat_coord(plan.kept_fanned[i], plan.orientation) >= 0.5)
          == keep_second
      ]
      half_plan = dataclasses.replace(
          plan,
          order=tuple(i for i, _ in pairs),
          deltas=tuple(d for _, d in pairs),
      )
      half_state, count = _run_incremental_fan(
          state,
          half_plan,
          open_percent=open_percent,
          verify_fold=verify_fold,
          deadline=deadline,
      )
      if count == 0:
        continue
      mirrored = _mirror_half(
          half_state.pattern, vertical=vertical, keep_second=keep_second
      )
      accepted, _, model = state_lib.accept_pattern(
          state,
          mirrored,
          allow_3d_vertices=open_percent < 1.0,
          verify_fold=verify_fold and state.config.check_flat_foldability,
      )
      if accepted is None:
        continue
      ratio = _section_width_ratio(
          state, element_name, base_widths, accepted.pattern, model=model
      )
      cand = _FanCandidate(
          state=accepted,
          width_ratio=ratio,
          new_creases=state_lib.new_crease_count(
              state.pattern, accepted.pattern
          ),
          pleats=2 * count,
          label=plan.label,
          weak=bool(base_widths and ratio < _MIN_FAN_WIDTH_RATIO),
      )
      if cand.beats(best):
        best = cand
    if best is not None:
      return best

  out_state, count = _run_incremental_fan(
      state,
      plan,
      open_percent=open_percent,
      verify_fold=verify_fold,
      deadline=deadline,
  )
  if count == 0:
    return None
  ratio = _section_width_ratio(
      state, element_name, base_widths, out_state.pattern
  )
  return _FanCandidate(
      state=out_state,
      width_ratio=ratio,
      new_creases=state_lib.new_crease_count(state.pattern, out_state.pattern),
      pleats=count,
      label=plan.label,
      weak=bool(base_widths and ratio < _MIN_FAN_WIDTH_RATIO),
  )


def fan_out_fold(
    state: state_lib.ShapingState,
    *,
    element_name: str,
    move_delta: float = 1.0,
    orientation: str | None = None,
    fold_type: str | None = None,
    spread: bool = False,
    open_percent: float = 1.0,
    verify_fold: bool = True,
) -> state_lib.ToolOutcome:
  """Fans out border accordion pleats of a flap to flare wings, fins, or tails.

  When `orientation` is omitted, both horizontal and vertical pleat families
  across mountain and valley assignments are evaluated and ranked automatically.

  Args:
    state: Current shaping state.
    element_name: Target flap name.
    move_delta: Border shift in grid units (non-zero; sign chooses direction).
    orientation: `'horizontal'`, `'vertical'`, or `None` for automatic search.
    fold_type: `'mountain'`, `'valley'`, or `None` to try individual and
      combined pleat polarities.
    spread: Fan pleats outward from the flap center with opposing signs.
    open_percent: Fold fraction in `(0, 1]` applied to new fan creases.
    verify_fold: Require the deterministic 3-D folder to converge cleanly.

  Returns:
    `ToolOutcome` containing the updated state and fan diagnostics.
  """
  if math.isclose(move_delta, 0.0):
    raise ValueError('move_delta must be non-zero.')
  if not 0.0 < open_percent <= 1.0:
    raise ValueError(f'open_percent must be in (0, 1], got {open_percent}.')

  if orientation is None:
    orientations = (
        fan_out_shaping.Orientation.VERTICAL,
        fan_out_shaping.Orientation.HORIZONTAL,
    )
  else:
    ori = _ORIENTATIONS.get(orientation.lower())
    if ori is None:
      raise ValueError(
          "orientation must be 'horizontal', 'vertical', or None; got"
          f' {orientation!r}.'
      )
    orientations = (ori,)

  if fold_type is None:
    fold_sets: tuple[tuple[crease_utils.Fold, ...], ...] = (
        (crease_utils.Fold.MOUNTAIN,),
        (crease_utils.Fold.VALLEY,),
        (crease_utils.Fold.MOUNTAIN, crease_utils.Fold.VALLEY),
    )
  else:
    ft = _FOLD_TYPES.get(fold_type.lower())
    if ft is None:
      raise ValueError(
          f"fold_type must be 'mountain', 'valley', or None; got {fold_type!r}."
      )
    fold_sets = ((ft,),)

  def _impl(st: state_lib.ShapingState) -> state_lib.ToolOutcome:
    flap = _require_flap(st, element_name)
    rects = [_flap_rect(flap)]
    chain = _chain_rect(st, element_name)
    if chain is not None:
      rects.append(chain)
    is_left = state_lib.is_left_element(st, element_name)
    base_widths = _base_section_widths(st, element_name) if verify_fold else {}
    deadline = time.perf_counter() + max(1.0, st.config.fan_out_time_budget_s)

    deltas = [float(move_delta)]
    if orientation is None or abs(move_delta) >= 1.0:
      opp = -float(move_delta)
      if not spread and opp not in deltas:
        deltas.append(opp)

    best: _FanCandidate | None = None
    saw_pleats = False
    for rect in rects:
      for delta in deltas:
        if time.perf_counter() >= deadline:
          break
        plans: list[_FanPlan] = []
        for ori in orientations:
          for fts in fold_sets:
            plan = _build_fan_plan(
                st,
                rect,
                ori,
                fts,
                move_delta=delta,
                is_left=is_left,
                spread=spread,
            )
            if plan is not None:
              plans.append(plan)
        plans.sort(key=lambda p: p.avg_length, reverse=True)
        for plan in plans:
          saw_pleats = True
          if time.perf_counter() >= deadline:
            break
          cand = _evaluate_plan(
              st,
              element_name,
              plan,
              base_widths,
              open_percent=open_percent,
              verify_fold=verify_fold,
              deadline=deadline,
          )
          if cand is not None and cand.beats(best):
            best = cand
      if best is not None and not best.weak:
        break

    if best is None:
      if not saw_pleats:
        raise ValueError(
            f'fan_out_fold({element_name}): no border pleats match'
            f' orientation={orientation!r}, fold_type={fold_type!r}.'
        )
      raise ValueError(
          f'fan_out_fold({element_name}): candidate pleats could not be made'
          ' flat-foldable and cleanly foldable.'
      )

    relaxed_pct: float | None = None
    if open_percent < 1.0:
      old_keys = {state_lib.crease_key(c) for c in st.pattern.creases}
      new_pcts = sorted({
          round(c.fold_percent, 4)
          for c in best.state.pattern.creases
          if c.mountain_or_valley() and state_lib.crease_key(c) not in old_keys
      })
      if new_pcts and not any(
          math.isclose(p, open_percent, abs_tol=1e-3) for p in new_pcts
      ):
        relaxed_pct = new_pcts[0]

    fb_suffix = (
        f' [fallback open={relaxed_pct:g} from {open_percent:g}]'
        if relaxed_pct is not None
        else ''
    )
    resolved: dict[str, object] | None = None
    if relaxed_pct is not None:
      resolved = {
          'element_name': element_name,
          'move_delta': move_delta,
          'spread': spread,
          'open_percent': relaxed_pct,
          'verify_fold': verify_fold,
      }
      if orientation is not None:
        resolved['orientation'] = orientation
      if fold_type is not None:
        resolved['fold_type'] = fold_type
    note = (
        f'{element_name}: fanned {best.pleats} pleat(s) ({best.label}),'
        f' +{best.new_creases} creases, width {best.width_ratio:.2f}x'
        f'{fb_suffix}.'
    )
    return state_lib.ToolOutcome(
        state=best.state, note=note, resolved_arguments=resolved
    )

  return state_lib.run_with_3d_overlay(state, _impl, verify_fold=verify_fold)


# ------------------------------------------------------------------------------
# Squash-sink shaping.
# ------------------------------------------------------------------------------


def squash_ridges(
    flap: shaping_elements.ShapingFlap,
) -> list[crease_utils.Crease]:
  """Returns center-to-corner ridges of `flap` in mirror-canonical order."""
  centers = flap.circle_centers
  if len(centers) == 1:
    raw = [
        crease_utils.Crease(centers[0], corner, fold=crease_utils.Fold.MOUNTAIN)
        for corner in flap.corners
    ]
  else:
    central = crease_utils.Crease(
        centers[0], centers[1], fold=crease_utils.Fold.MOUNTAIN
    )
    raw = [r for r in flap.get_ridges() if not r.is_collinear(central)]
  directed: list[crease_utils.Crease] = []
  for r in raw:
    p1, p2 = r.p1, r.p2
    if any(p2.isclose(c) for c in centers) and not any(
        p1.isclose(c) for c in centers
    ):
      directed.append(
          crease_utils.Crease(p2, p1, fold=r.fold, elevation=r.elevation)
      )
    else:
      directed.append(r)
  is_left = (flap.x_start + flap.x_len * 0.5) < 0.5

  def _sort_key(r: crease_utils.Crease) -> tuple[float, float, float, float]:
    x1 = 1.0 - r.p1.x if is_left else r.p1.x
    x2 = 1.0 - r.p2.x if is_left else r.p2.x
    return (round(x1, 5), round(r.p1.y, 5), round(x2, 5), round(r.p2.y, 5))

  directed.sort(key=_sort_key)
  return directed


def ridge_segment_counts(
    pattern: crease_pattern_lib.CreasePattern,
    flap: shaping_elements.ShapingFlap,
) -> list[int]:
  """Counts crease segments along each squashable ridge of `flap`."""
  rects = [flap.to_legacy_rectangle()]
  counts: list[int] = []
  for ridge in squash_ridges(flap):
    try:
      start = squash_sink_shaping.get_start_side_of_ridge(ridge, ridge, rects)
      segs = squash_sink_shaping.get_oriented_ordered_crease_segments_on_ridge(
          pattern, ridge, start
      )
      counts.append(len(segs))
    except (ValueError, AssertionError):
      counts.append(0)
  return counts


def _line_signature(
    crease: crease_utils.Crease,
) -> tuple[float, float, float]:
  """Returns canonical normalized line equation `(nx, ny, d)` for `crease`."""
  dx = crease.p2_init.x - crease.p1_init.x
  dy = crease.p2_init.y - crease.p1_init.y
  norm = math.hypot(dx, dy)
  if norm < 1e-9:
    return (0.0, 0.0, 0.0)
  nx, ny = -dy / norm, dx / norm
  if (nx, ny) < (-nx, -ny):
    nx, ny = -nx, -ny
  d = nx * crease.p1_init.x + ny * crease.p1_init.y
  return (round(nx, 4), round(ny, 4), round(d, 4))


def _midpoint_in_rect(
    crease: crease_utils.Crease, rect: shaping_utils.Rectangle
) -> bool:
  x0 = min(rect.x_start, rect.x_start + rect.x_len) - 1e-4
  x1 = max(rect.x_start, rect.x_start + rect.x_len) + 1e-4
  y0 = min(rect.y_start, rect.y_start + rect.y_len) - 1e-4
  y1 = max(rect.y_start, rect.y_start + rect.y_len) + 1e-4
  mx = 0.5 * (crease.p1_init.x + crease.p2_init.x)
  my = 0.5 * (crease.p1_init.y + crease.p2_init.y)
  return x0 <= mx <= x1 and y0 <= my <= y1


def _open_squash_pattern(
    before: crease_pattern_lib.CreasePattern,
    shaped: crease_pattern_lib.CreasePattern,
    ridge: crease_utils.Crease,
    rect: shaping_utils.Rectangle,
    *,
    strategy: str,
    open_percent: float,
) -> crease_pattern_lib.CreasePattern:
  """Partially opens a squash kite and optional continuation pleats in 3-D."""
  old_keys = {state_lib.crease_key(c) for c in before.creases}
  new_creases = [
      c
      for c in shaped.creases
      if c.mountain_or_valley() and state_lib.crease_key(c) not in old_keys
  ]
  new_keys = {state_lib.crease_key(c) for c in new_creases}
  if not new_keys or strategy == 'all_new':
    return state_lib.with_fold_percent(shaped, new_keys, open_percent)

  ridge_line = _line_signature(ridge)
  axial_and_cross = {
      state_lib.crease_key(c)
      for c in new_creases
      if _line_signature(c) == ridge_line
      or (not c.is_horizontal and not c.is_vertical)
  }
  if strategy == 'axial_and_cross':
    return state_lib.with_fold_percent(
        shaped, axial_and_cross or new_keys, open_percent
    )

  kite_vertices: set[tuple[float, float]] = set()
  for c in new_creases:
    kite_vertices.add((round(c.p1_init.x, 5), round(c.p1_init.y, 5)))
    kite_vertices.add((round(c.p2_init.x, 5), round(c.p2_init.y, 5)))
  pleat_lines = {
      _line_signature(c)
      for c in shaped.creases
      if c.mountain_or_valley()
      and (c.is_horizontal or c.is_vertical)
      and _midpoint_in_rect(c, rect)
      and (
          (round(c.p1_init.x, 5), round(c.p1_init.y, 5)) in kite_vertices
          or (round(c.p2_init.x, 5), round(c.p2_init.y, 5)) in kite_vertices
      )
  }
  target = new_keys | {
      state_lib.crease_key(c)
      for c in shaped.creases
      if c.mountain_or_valley()
      and _midpoint_in_rect(c, rect)
      and _line_signature(c) in pleat_lines
  }
  return state_lib.with_fold_percent(shaped, target, open_percent)


def _mirror_ridge_pairs(
    flap: shaping_elements.ShapingFlap,
    ridges: Sequence[crease_utils.Crease],
    counts: Sequence[int],
) -> list[tuple[int, int]]:
  """Finds symmetric `(i, j)` ridge pairs with at least 2 segments each."""
  axes: list[tuple[float | None, float | None]] = []
  if flap.contains_central_crease:
    axes.append((0.5, None))
  x_mid = flap.x_start + flap.x_len * 0.5
  y_mid = flap.y_start + flap.y_len * 0.5
  axes.extend(
      [(x_mid, None), (None, y_mid)]
      if flap.y_len >= flap.x_len
      else [(None, y_mid), (x_mid, None)]
  )
  usable = sorted(
      (i for i, c in enumerate(counts) if c >= 2),
      key=lambda i: counts[i],
      reverse=True,
  )
  pairs: list[tuple[int, int]] = []
  for ax, ay in axes:
    used: set[int] = set()
    for i in usable:
      if i in used:
        continue
      a = ridges[i]
      for j in usable:
        if j == i or j in used:
          continue
        b = ridges[j]

        def _refl(
            p: object, q: object, ax: float | None = ax, ay: float | None = ay
        ) -> bool:
          px, py = getattr(p, 'x'), getattr(p, 'y')
          qx, qy = getattr(q, 'x'), getattr(q, 'y')
          if ax is not None:
            return math.isclose(px, 2.0 * ax - qx, abs_tol=1e-4) and (
                math.isclose(py, qy, abs_tol=1e-4)
            )
          return (
              ay is not None
              and math.isclose(px, qx, abs_tol=1e-4)
              and math.isclose(py, 2.0 * ay - qy, abs_tol=1e-4)
          )

        if (_refl(a.p1, b.p1) and _refl(a.p2, b.p2)) or (
            _refl(a.p1, b.p2) and _refl(a.p2, b.p1)
        ):
          pair = (i, j)
          if pair not in pairs and (j, i) not in pairs:
            pairs.append(pair)
          used.update(pair)
          break
  return pairs


def _squash_one_ridge(
    state: state_lib.ShapingState,
    flap: shaping_elements.ShapingFlap,
    ridges: Sequence[crease_utils.Crease],
    ridge_index: int,
    *,
    start_idx: int | None,
    end_idx: int | None,
    open_percent: float,
    verify_fold: bool,
    base_widths: dict[float, float],
    min_ratio: float = 1.15,
) -> tuple[state_lib.ShapingState, float, float, bool]:
  """Squashes a ridge, returning `(state, width_ratio, used_open, clean)`."""
  rect = flap.to_legacy_rectangle()
  rects = [rect]
  ridge = ridges[ridge_index]
  start_side = squash_sink_shaping.get_start_side_of_ridge(ridge, ridge, rects)
  segs = squash_sink_shaping.get_oriented_ordered_crease_segments_on_ridge(
      state.pattern, ridge, start_side
  )
  can_border_tri = (
      bool(segs)
      and segs[0].p1_init.is_border
      and segs[0].p2_init.is_internal
      and start_idx is None
      and end_idx is None
  )
  if len(segs) < 2 and not can_border_tri:
    raise ValueError(
        f'squash_fold({flap.name}): ridge {ridge_index} has {len(segs)}'
        ' segment(s) (needs >= 2);'
        f' counts={ridge_segment_counts(state.pattern, flap)}.'
    )

  if start_idx is None and end_idx is None:
    spans: list[tuple[int, int]] = []
    if len(segs) >= 4:
      for span in ((1, len(segs) - 1), (1, len(segs)), (0, len(segs))):
        if span[1] - span[0] >= 3 and span not in spans:
          spans.append(span)
    spans.extend((k, k + 2) for k in range(len(segs) - 1))
    if can_border_tri:
      spans.append((0, 1))
  else:
    if end_idx is None:
      s = int(start_idx or 0)
      e = s + 2
    elif start_idx is None:
      e = int(end_idx)
      s = max(0, e - 2)
    else:
      s, e = int(start_idx), int(end_idx)
    s, e = max(0, s), min(len(segs), e)
    if s >= e:
      raise ValueError(
          f'Invalid squash span [{s}, {e}) for ridge with {len(segs)} segments.'
      )
    spans = [(s, e)]

  last_reason = 'no squash candidate changed the crease pattern'
  for s, e in spans:
    sub_ridge = crease_utils.Crease(
        segs[s].p1_init,
        segs[e - 1].p2_init,
        fold=ridge.fold,
        elevation=ridge.elevation,
    )
    try:
      shaped = squash_sink_shaping.shape_cp_with_squash_sink(
          cp=state.pattern, rectangles=rects, ridge=sub_ridge
      )
      shaped = state_lib.preserve_base_borders(state.pattern, shaped)
    except (ValueError, AssertionError) as error:
      last_reason = f'span {s}..{e}: {error}'
      continue
    if state_lib.new_crease_count(state.pattern, shaped) == 0:
      continue
    if not shaped.local_flat_foldability():
      last_reason = f'span {s}..{e}: not flat-foldable'
      continue

    if math.isclose(open_percent, 1.0, abs_tol=1e-6):
      accepted, reason, model = state_lib.accept_pattern(
          state,
          shaped,
          allow_3d_vertices=False,
          verify_fold=verify_fold and state.config.check_flat_foldability,
      )
      if accepted is None:
        last_reason = reason
        continue
      ratio = _section_width_ratio(
          state, flap.name, base_widths, accepted.pattern, model=model
      )
      return accepted, ratio, 1.0, True

    open_trials = [
        (open_percent, 'kite_and_pleats'),
        (open_percent, 'axial_and_cross'),
        (open_percent, 'all_new'),
    ]
    if open_percent < 0.6:
      open_trials.insert(2, (min(0.75, open_percent + 0.2), 'kite_and_pleats'))
    best_pair: (
        tuple[tuple[bool, float], state_lib.ShapingState, float, bool] | None
    ) = None
    for pct, strat in open_trials:
      opened = _open_squash_pattern(
          state.pattern,
          shaped,
          sub_ridge,
          rect,
          strategy=strat,
          open_percent=pct,
      )
      accepted, reason, model = state_lib.accept_pattern(
          state,
          opened,
          allow_3d_vertices=True,
          verify_fold=verify_fold and state.config.check_flat_foldability,
      )
      if accepted is None:
        last_reason = reason
        continue
      clean = (
          model is None
          or model.vertex_error
          <= fidelity.DEFAULT_CONFIG.clean_gate_max_vertex_error
      )
      ratio = _section_width_ratio(
          state, flap.name, base_widths, accepted.pattern, model=model
      )
      rank = (clean, ratio)
      if best_pair is None or rank > best_pair[0]:
        best_pair = (rank, accepted, pct, clean)
      if clean and ratio >= min_ratio:
        return accepted, ratio, pct, clean
    if best_pair is not None:
      return best_pair[1], best_pair[0][1], best_pair[2], best_pair[3]

  raise ValueError(
      f'squash_fold({flap.name}) on ridge {ridge_index} failed: {last_reason}'
  )


def squash_fold(
    state: state_lib.ShapingState,
    *,
    element_name: str,
    ridge_index: int | None = None,
    starting_position_index: int | None = None,
    ending_position_index: int | None = None,
    open_percent: float = 0.45,
    verify_fold: bool = True,
) -> state_lib.ToolOutcome:
  """Squashes a flap along one or a symmetric pair of diagonal ridges.

  Args:
    state: Current shaping state.
    element_name: Target flap name.
    ridge_index: Explicit ridge index in `0..len(ridges)-1`, or `None` to pick
      the best symmetric ridge pair or single ridge automatically.
    starting_position_index: Optional start segment index along the ridge.
    ending_position_index: Optional exclusive end segment index along the ridge.
    open_percent: Fold fraction in `(0, 1]` for squash creases (`1.0` folds
      flat; `0.35`–`0.55` spreads the squashed kite into a broad 3-D plate).
    verify_fold: Require the deterministic 3-D folder to converge cleanly.

  Returns:
    `ToolOutcome` containing the updated state and squash summary.
  """
  if not 0.0 < open_percent <= 1.0:
    raise ValueError(f'open_percent must be in (0, 1], got {open_percent}.')

  def _impl(st: state_lib.ShapingState) -> state_lib.ToolOutcome:
    flap = _require_flap(st, element_name)
    ridges = squash_ridges(flap)
    if not ridges:
      raise ValueError(f'squash_fold({element_name}): flap has no ridges.')
    base_widths = _base_section_widths(st, element_name) if verify_fold else {}

    if ridge_index is not None:
      if not 0 <= ridge_index < len(ridges):
        raise ValueError(
            f'ridge_index={ridge_index} out of range for {element_name}'
            f' (0..{len(ridges) - 1}).'
        )
      out, ratio, used_pct, clean = _squash_one_ridge(
          st,
          flap,
          ridges,
          ridge_index,
          start_idx=starting_position_index,
          end_idx=ending_position_index,
          open_percent=open_percent,
          verify_fold=verify_fold,
          base_widths=base_widths,
      )
      added = state_lib.new_crease_count(st.pattern, out.pattern)
      fb_tags: list[str] = []
      if not math.isclose(used_pct, open_percent, abs_tol=1e-6):
        fb_tags.append(f'fallback open={used_pct:.2f} from {open_percent:.2f}')
      if not clean:
        fb_tags.append('above clean_gate vertex error')
      fb_suffix = f' [{"; ".join(fb_tags)}]' if fb_tags else ''
      resolved: dict[str, object] | None = None
      if not math.isclose(used_pct, open_percent, abs_tol=1e-6):
        resolved = {
            'element_name': element_name,
            'ridge_index': ridge_index,
            'open_percent': used_pct,
            'verify_fold': verify_fold,
        }
        if starting_position_index is not None:
          resolved['starting_position_index'] = starting_position_index
        if ending_position_index is not None:
          resolved['ending_position_index'] = ending_position_index
      return state_lib.ToolOutcome(
          state=out,
          note=(
              f'{element_name}: squashed ridge {ridge_index}'
              f' (open={used_pct:.2f}), +{added} creases, width'
              f' {ratio:.2f}x{fb_suffix}.'
          ),
          resolved_arguments=resolved,
      )

    counts = ridge_segment_counts(st.pattern, flap)
    last_error = (
        f'squash_fold({element_name}): no squashable ridges (counts={counts}).'
    )
    bilateral_failed = False
    if starting_position_index is None and ending_position_index is None:
      flat_only = math.isclose(open_percent, 1.0, abs_tol=1e-6)
      for i, j in _mirror_ridge_pairs(flap, ridges, counts):
        try:
          first, _, pct_i, clean_i = _squash_one_ridge(
              st,
              flap,
              ridges,
              i,
              start_idx=None,
              end_idx=None,
              open_percent=open_percent,
              verify_fold=False if flat_only else verify_fold,
              base_widths={} if flat_only else base_widths,
              min_ratio=1.05,
          )
          second_target = 1.0 if flat_only else pct_i
          both, ratio, pct_j, clean_j = _squash_one_ridge(
              first,
              flap,
              ridges,
              j,
              start_idx=None,
              end_idx=None,
              open_percent=second_target,
              verify_fold=verify_fold,
              base_widths=base_widths,
          )
          added = state_lib.new_crease_count(st.pattern, both.pattern)
          same_pct = math.isclose(pct_i, pct_j, abs_tol=1e-6)
          open_label = (
              f'{pct_j:.2f}' if same_pct else f'{pct_i:.2f}/{pct_j:.2f}'
          )
          fb_tags = []
          if not (
              math.isclose(pct_i, open_percent, abs_tol=1e-6)
              and math.isclose(pct_j, open_percent, abs_tol=1e-6)
          ):
            fb_tags.append(
                f'fallback open={open_label} from {open_percent:.2f}'
            )
          if not (clean_i and clean_j):
            fb_tags.append('above clean_gate vertex error')
          fb_suffix = f' [{"; ".join(fb_tags)}]' if fb_tags else ''
          resolved: dict[str, object] | None = None
          if same_pct and not math.isclose(pct_j, open_percent, abs_tol=1e-6):
            resolved = {
                'element_name': element_name,
                'open_percent': pct_j,
                'verify_fold': verify_fold,
            }
          return state_lib.ToolOutcome(
              state=both,
              note=(
                  f'{element_name}: squashed mirror ridges ({i}, {j})'
                  f' (open={open_label}), +{added} creases, width'
                  f' {ratio:.2f}x{fb_suffix}.'
              ),
              resolved_arguments=resolved,
          )
        except ValueError as error:
          last_error = str(error)
          bilateral_failed = True

    ordered = sorted(
        range(len(ridges)), key=lambda idx: counts[idx], reverse=True
    )
    for idx in ordered:
      if counts[idx] <= 0:
        continue
      try:
        out, ratio, used_pct, clean = _squash_one_ridge(
            st,
            flap,
            ridges,
            idx,
            start_idx=starting_position_index,
            end_idx=ending_position_index,
            open_percent=open_percent,
            verify_fold=verify_fold,
            base_widths=base_widths,
        )
        added = state_lib.new_crease_count(st.pattern, out.pattern)
        fb_tags = []
        if bilateral_failed:
          fb_tags.append('unilateral fallback after bilateral pair failed')
        if not math.isclose(used_pct, open_percent, abs_tol=1e-6):
          fb_tags.append(
              f'fallback open={used_pct:.2f} from {open_percent:.2f}'
          )
        if not clean:
          fb_tags.append('above clean_gate vertex error')
        fb_suffix = f' [{"; ".join(fb_tags)}]' if fb_tags else ''
        resolved: dict[str, object] | None = None
        if not math.isclose(used_pct, open_percent, abs_tol=1e-6):
          resolved = {
              'element_name': element_name,
              'ridge_index': idx,
              'open_percent': used_pct,
              'verify_fold': verify_fold,
          }
          if starting_position_index is not None:
            resolved['starting_position_index'] = starting_position_index
          if ending_position_index is not None:
            resolved['ending_position_index'] = ending_position_index
        return state_lib.ToolOutcome(
            state=out,
            note=(
                f'{element_name}: squashed ridge {idx}'
                f' (open={used_pct:.2f}), +{added} creases, width'
                f' {ratio:.2f}x{fb_suffix}.'
            ),
            resolved_arguments=resolved,
        )
      except ValueError as error:
        last_error = str(error)
    raise ValueError(last_error)

  return state_lib.run_with_3d_overlay(state, _impl, verify_fold=verify_fold)
