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

"""Automatic 3-D pose synthesis recipes built on the shaping actuator toolbox.

Provides four composable posing modes selected via `PoseConfig`:

* `'figure'`: Simultaneous 3-D spherical rotation toward stick-figure angles
  with closed-loop root-frame feedback, bilateral side-crossing repair, and
  optional limb narrowing.
* `'greedy'`: Fast, zero-folder flat-foldable pose combining simultaneous 3-D
  rotation with 2-D fallback bends and optional narrowing.
* `'search'`: Deadline-bounded, symmetry-preserving coordinate ascent over
  anatomical element groups maximizing fidelity while preserving fold gates.
* `'auto'`: Runs `'figure'` followed by `'search'` local refinement within the
  remaining wall-clock budget (falling back to `'greedy'` if unposed).
"""

from collections.abc import Callable, Mapping, Sequence
import dataclasses
import math
import time
from typing import Any, Literal

from .. import bus
from .. import folder_3d
from ..stick_figures import angle_utils
from ..stick_figures import stick_figure as stick_figure_lib
from . import fidelity
from . import heuristics
from . import paper_layout as paper_layout_lib
from . import pose_alignment
from . import shaping_elements
from . import state as state_lib
from . import toolbox

PoseMode = Literal['auto', 'figure', 'search', 'greedy']
_VALID_MODES: tuple[str, ...] = ('auto', 'figure', 'search', 'greedy')

# Element role tags produced by `pose_plan`.
ROLE_ROOT = 'root'
ROLE_RIVER = 'river'
ROLE_THIN_LIMB = 'thin_limb'
ROLE_LIMB = 'limb'
ROLE_SEGMENT = 'segment'
ROLE_WIDE_TIP = 'wide_tip'
ROLE_TIP_DOWN = 'tip_down'
ROLE_SKIP = 'skip'

_FEEDBACK_GAIN = 0.65
_FEEDBACK_MIN_ERROR_DEG = 15.0
_MAX_ELEVATION_DEG = 88.0
_MIN_SEARCH_REMAINDER_S = 1.5


@dataclasses.dataclass(frozen=True, kw_only=True)
class PoseConfig:
  """Configuration for automatic 3-D posing (`bp.pose` and `pose_state`).

  Attributes:
    mode: Recipe strategy (`'auto'`, `'figure'`, `'search'`, or `'greedy'`).
    narrow: Narrow width-1 outer limbs and central tips when clean.
    time_budget_s: Wall-clock budget in seconds for `'search'` refinement
      (`'figure'` runs to completion; `'auto'` allocates remaining budget to
      `'search'` only when `'figure'` has not yet satisfied both `clean_gate`
      and `pose_gate`).
    feedback_rounds: Closed-loop root-frame correction rounds in `'figure'`.
    min_gain: Minimum fidelity gain required to accept a `'search'` step.
  """

  mode: PoseMode = 'auto'
  narrow: bool = False
  time_budget_s: float = 60.0
  feedback_rounds: int = 2
  min_gain: float = 0.05

  def __post_init__(self) -> None:
    if self.mode not in _VALID_MODES:
      raise ValueError(
          f'PoseConfig.mode must be one of {_VALID_MODES}, got {self.mode!r}.'
      )
    if self.time_budget_s <= 0.0:
      raise ValueError(
          f'PoseConfig.time_budget_s must be > 0, got {self.time_budget_s}.'
      )
    if self.feedback_rounds < 0:
      raise ValueError(
          'PoseConfig.feedback_rounds must be >= 0, got'
          f' {self.feedback_rounds}.'
      )
    if self.min_gain < 0.0:
      raise ValueError(
          f'PoseConfig.min_gain must be >= 0, got {self.min_gain}.'
      )


@dataclasses.dataclass(frozen=True, kw_only=True)
class ElementPlan:
  """Anatomical classification and recommended tools for one stick."""

  name: str
  role: str
  partner: str | None
  parent: str | None
  target_az: float
  target_el: float
  width: float
  length: float
  recommended_tools: tuple[str, ...]

  def asdict(self) -> dict[str, Any]:
    """Converts this plan entry to a JSON-friendly dictionary."""
    return dataclasses.asdict(self)


# ------------------------------------------------------------------------------
# Anatomical planning and bilateral geometry helpers.
# ------------------------------------------------------------------------------


def pose_plan(state: state_lib.ShapingState) -> list[ElementPlan]:
  """Classifies every stick of `state.tree` into an anatomical shaping role."""
  tree = state.tree
  topo = paper_layout_lib.topology(tree)
  root_name = heuristics.get_first_stick_for_shaping(
      tree, use_tree_root=state.config.use_tree_root
  ).name
  has_children = {p for p in topo.parent.values() if p is not None}
  _, merged_map = tree.merge_linear_chains()
  chain_tail: set[str] = set()
  for subs in merged_map.values():
    chain_tail.update(subs[1:])
  try:
    flaps, rivers = state.shaping_elements
  except (AssertionError, KeyError, ValueError):
    flaps, rivers = state.merged_shaping_elements
  elem_by_name: dict[str, state_lib.ShapingElement] = {f.name: f for f in flaps}
  elem_by_name.update({r.name: r for r in rivers})

  plans: list[ElementPlan] = []
  for name in topo.order:
    stick = tree[name]
    elem = elem_by_name.get(name)
    partner = state_lib.resolve_symmetric_element(state, name)
    parent = topo.parent.get(name)
    is_leaf = name not in has_children
    spans_axis = elem is not None and state_lib.element_spans_axis(elem)

    if name == root_name:
      role = ROLE_ROOT
      tools: tuple[str, ...] = ('fold_spine',) if spans_axis else ()
    elif stick.length <= 0:
      role = ROLE_SKIP
      tools = ()
    elif stick.shaping_flap:
      # `recommended_tools` is advisory here; `bp.pose` skips `ROLE_SKIP`.
      role = ROLE_SKIP
      tools = ('shape_flap_template',)
    elif stick.is_river:
      role = ROLE_RIVER
      base_tools = ['rotation_3d', 'open_flap']
      if stick.width >= 2.0 or (elem is not None and elem.width >= 2.0):
        base_tools.append('inscribed_flap')
      if partner is None and spans_axis:
        base_tools.append('fold_spine')
      tools = tuple(base_tools)
    elif is_leaf and stick.width >= 2.0:
      role = ROLE_WIDE_TIP
      tools = ('rotation_3d', 'squash_fold', 'fan_out_fold', 'open_flap')
    elif (
        is_leaf
        and stick.elevation_angle <= -25.0
        and parent is not None
        and parent != root_name
    ):
      role = ROLE_TIP_DOWN
      tools = ('rotation_3d', 'crimp_fold', 'reverse_fold', 'bend_to_direction')
    elif name in chain_tail:
      role = ROLE_SEGMENT
      tools = ('rotation_3d', 'crimp_fold', 'reverse_fold', 'curve_flap')
    elif partner is not None and stick.width <= 1.0:
      role = ROLE_THIN_LIMB
      tools = (
          'rotation_3d',
          'thin_limbs_3d',
          'simple_fold',
          'apply_symmetric',
      )
    elif (
        isinstance(elem, shaping_elements.ShapingFlap)
        and spans_axis
        and elem.grid_unit_radius >= 2
    ):
      role = ROLE_LIMB
      tools = (
          'rotation_3d',
          'narrow_central_flap',
          'bend_to_direction',
          'curve_flap',
      )
    else:
      role = ROLE_LIMB
      tools = ('rotation_3d', 'bend_to_direction', 'crimp_fold', 'reverse_fold')

    plans.append(
        ElementPlan(
            name=name,
            role=role,
            partner=partner,
            parent=parent,
            target_az=float(stick.azimuth_angle),
            target_el=float(stick.elevation_angle),
            width=float(stick.width),
            length=float(stick.length),
            recommended_tools=tools,
        )
    )
  return plans


def _nudge_azimuth_toward(
    az: float, target_side_az: float, step: float
) -> float:
  """Moves `az` toward `target_side_az` (`+90` or `-90`) by up to `step` deg."""
  diff = angle_utils.wrap_angle_centered(target_side_az - az, 360.0)
  if abs(diff) <= step:
    return angle_utils.wrap_angle_centered(target_side_az, 360.0)
  return angle_utils.wrap_angle_centered(az + math.copysign(step, diff), 360.0)


def lateral_bias(
    tree: stick_figure_lib.StickFigure,
    *,
    max_bias_deg: float = 15.0,
) -> dict[str, float]:
  """Computes small symmetric outward azimuth nudges for steep paired legs.

  Args:
    tree: Unmerged stick figure.
    max_bias_deg: Maximum outward azimuth shift in degrees.

  Returns:
    Mapping from stick name to signed azimuth delta (degrees), satisfying
    `bias[left] == -bias[right]` across bilaterally paired sticks.
  """
  topo = paper_layout_lib.topology(tree)
  design_pos = paper_layout_lib.stick_figure_joint_positions(tree)
  root = tree[tree.root]
  mid_y = 0.5 * float(design_pos[root.start][1] + design_pos[root.end][1])
  known = {s.name for s in tree.children}
  bias: dict[str, float] = {}
  visited: set[str] = set()

  for name in topo.order:
    if name in visited or name == tree.root:
      continue
    stick = tree[name]
    if stick.is_river or stick.elevation_angle > -35.0:
      continue
    partner = stick.symmetric_stick_name or fidelity.infer_symmetric_stick(
        name, known
    )
    if not partner or partner == name or partner not in tree:
      continue
    dist_y = float(design_pos[topo.distal[name]][1]) - mid_y
    if abs(dist_y) < 1e-3:
      continue
    steepness = min(1.0, max(0.0, (-stick.elevation_angle - 35.0) / 45.0))
    if steepness <= 0.0:
      continue
    side_az = 90.0 if dist_y > 0.0 else -90.0
    shifted = _nudge_azimuth_toward(
        stick.azimuth_angle, side_az, max_bias_deg * steepness
    )
    delta = round(
        angle_utils.wrap_angle_centered(shifted - stick.azimuth_angle, 360.0),
        2,
    )
    if not math.isclose(delta, 0.0, abs_tol=0.5):
      bias[name] = delta
      bias[partner] = -delta
    visited.add(name)
    visited.add(partner)
  return bias


def side_fix_targets(
    tree: stick_figure_lib.StickFigure,
    crossing_names: Sequence[str],
    current_targets: Mapping[str, tuple[float, float]],
    *,
    step_deg: float = 25.0,
) -> dict[str, tuple[float, float]]:
  """Returns updated `(az, el)` targets splaying `crossing_names` outward."""
  topo = paper_layout_lib.topology(tree)
  design_pos = paper_layout_lib.stick_figure_joint_positions(tree)
  root = tree[tree.root]
  mid_y = 0.5 * float(design_pos[root.start][1] + design_pos[root.end][1])
  known = {s.name for s in tree.children}
  updated = {k: (float(v[0]), float(v[1])) for k, v in current_targets.items()}

  expanded: list[str] = []
  for name in crossing_names:
    if name not in tree or name == tree.root:
      continue
    if name not in expanded:
      expanded.append(name)
    partner = tree[name].symmetric_stick_name or fidelity.infer_symmetric_stick(
        name, known
    )
    if partner and partner in tree and partner not in expanded:
      expanded.append(partner)

  for name in expanded:
    stick = tree[name]
    cur_az, cur_el = updated.get(
        name, (float(stick.azimuth_angle), float(stick.elevation_angle))
    )
    offset_y = float(design_pos[topo.distal[name]][1]) - mid_y
    if abs(offset_y) < 1e-4:
      offset_y = math.sin(math.radians(stick.azimuth_angle))
    if abs(offset_y) < 1e-4:
      continue
    side_az = 90.0 if offset_y > 0.0 else -90.0
    new_az = round(_nudge_azimuth_toward(cur_az, side_az, step_deg), 2)
    new_el = cur_el
    if cur_el < -55.0:
      new_el = round(min(-45.0, cur_el + min(15.0, -55.0 - cur_el)), 2)
    updated[name] = (new_az, new_el)
  return updated


def realized_directions(
    state: state_lib.ShapingState,
    *,
    model: folder_3d.FoldedModel | None = None,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
  """Returns `(report, per_stick_directions)` in `rotation_3d`'s root frame."""
  return state_lib.measured_directions(state, model=model)


# ------------------------------------------------------------------------------
# Fidelity evaluation and gate-preserving acceptance.
# ------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class _ScoreSummary:
  """Compact score and gate status for candidate ranking."""

  score: float
  clean_ok: bool
  pose_ok: bool
  direction: float
  num_crossings: int
  vertex_error: float
  report: Mapping[str, Any]


def _score_state(state: state_lib.ShapingState) -> _ScoreSummary:
  """Folds and scores `state.pattern`, returning a fail-safe error summary."""
  ok_fold, reason, model = state_lib.verify_3d_fold(state.pattern)
  if not ok_fold or model is None:
    return _ScoreSummary(
        score=-1.0,
        clean_ok=False,
        pose_ok=False,
        direction=0.0,
        num_crossings=99,
        vertex_error=99.0,
        report={'error': reason},
    )
  try:
    report = fidelity.score_model(
        state.tree,
        state.packing,
        model,
        layout=state.paper_layout,
        pattern=state.pattern,
    ).to_dict()
  except (RuntimeError, ValueError, AssertionError, KeyError) as error:
    return _ScoreSummary(
        score=-1.0,
        clean_ok=False,
        pose_ok=False,
        direction=0.0,
        num_crossings=99,
        vertex_error=float(model.vertex_error),
        report={'error': str(error)},
    )
  clean = bool(report.get('clean_gate', {}).get('passed', False))
  pose = bool(report.get('pose_gate', {}).get('passed', False))
  direction = float(report.get('components', {}).get('direction', 0.0))
  crossings = len(
      pose_alignment.crossing_sticks(report.get('side_crossings') or [])
  )
  return _ScoreSummary(
      score=float(report['fidelity']),
      clean_ok=clean,
      pose_ok=pose,
      direction=direction,
      num_crossings=crossings,
      vertex_error=float(report.get('fold_vertex_error_percent', 0.0)),
      report=report,
  )


def _accept_candidate(
    cand: _ScoreSummary,
    best: _ScoreSummary,
    *,
    min_gain: float = 0.0,
) -> bool:
  """Returns True when `cand` improves on `best` without regressing gates."""
  if cand.score < 0.0:
    return False
  if best.clean_ok and not cand.clean_ok:
    return False
  if cand.clean_ok and not best.clean_ok:
    return True
  if best.pose_ok and not cand.pose_ok:
    return False
  if cand.pose_ok and not best.pose_ok and cand.score >= best.score - 0.15:
    return True
  return cand.score >= best.score + min_gain


def _build_diagnostics(
    *,
    mode: str,
    state: state_lib.ShapingState,
    base_summary: _ScoreSummary | None,
    final_summary: _ScoreSummary | None,
    elapsed: float,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
  """Constructs a flat, JSON-serializable pose diagnostics record."""
  diag: dict[str, Any] = {
      'mode': mode,
      'num_creases': len(state.pattern.creases),
      'base_creases': len(state.base_pattern.creases),
      'is_locally_flat_foldable': state.pattern.local_flat_foldability(),
      'steps_applied': sum(1 for h in state.history if h.get('ok')),
      'narrowed_elements': list(state.narrowed_elements),
      'time_taken': round(elapsed, 3),
  }
  if base_summary is not None and base_summary.score >= 0.0:
    diag['base_fidelity'] = round(base_summary.score, 2)
  if final_summary is not None and final_summary.score >= 0.0:
    diag['fidelity'] = round(final_summary.score, 2)
    diag['clean_gate'] = final_summary.clean_ok
    diag['pose_gate'] = final_summary.pose_ok
    diag['direction'] = round(final_summary.direction, 3)
    diag['vertex_error_percent'] = round(final_summary.vertex_error, 4)
  if extra:
    diag.update(extra)
  return diag


# ------------------------------------------------------------------------------
# 1. Closed-loop stick-figure pose (`figure_pose`).
# ------------------------------------------------------------------------------


def _initial_targets(
    state: state_lib.ShapingState,
    *,
    use_bias: bool = True,
) -> dict[str, tuple[float, float]]:
  """Builds initial `(az, el)` targets for every non-root stick."""
  root_name = heuristics.get_first_stick_for_shaping(
      state.tree, use_tree_root=state.config.use_tree_root
  ).name
  biases = lateral_bias(state.tree) if use_bias else {}
  targets: dict[str, tuple[float, float]] = {}
  for name in paper_layout_lib.topology(state.tree).order:
    if name == root_name:
      continue
    stick = state.tree[name]
    if stick.shaping_flap:
      continue
    az = angle_utils.wrap_angle_centered(
        float(stick.azimuth_angle) + biases.get(name, 0.0), 360.0
    )
    el = max(
        -_MAX_ELEVATION_DEG,
        min(_MAX_ELEVATION_DEG, float(stick.elevation_angle)),
    )
    targets[name] = (round(az, 2), round(el, 2))
  return targets


def _thin_leaf_names(state: state_lib.ShapingState) -> set[str]:
  """Returns unnarrowed width-1 leaf flaps eligible for narrowing."""
  narrowed = set(state.narrowed_elements)
  return {
      p.name
      for p in pose_plan(state)
      if p.role == ROLE_THIN_LIMB and p.width <= 1.0 and p.name not in narrowed
  }


def _run_rotation(
    base_state: state_lib.ShapingState,
    targets: Mapping[str, tuple[float, float]],
    *,
    narrow_names: set[str] | None = None,
    flaps_only: bool = False,
) -> state_lib.ShapingState | None:
  """Applies a single simultaneous `rotation_3d` call to `base_state`."""
  rotations: list[dict[str, Any]] = []
  narrow_set = narrow_names or set()
  for name, (az, el) in targets.items():
    if name not in base_state.tree:
      continue
    if flaps_only and base_state.tree[name].is_river:
      continue
    item: dict[str, Any] = {'element_name': name, 'az': az, 'el': el}
    if name in narrow_set and not base_state.tree[name].is_river:
      item['narrow'] = True
    rotations.append(item)
  if not rotations:
    return None
  next_state, res = toolbox.apply_tool(
      base_state, 'rotation_3d', {'rotations': rotations}
  )
  return next_state if res.ok else None


def _feedback_targets(
    state: state_lib.ShapingState,
    current_targets: Mapping[str, tuple[float, float]],
    directions: Mapping[str, Mapping[str, Any]],
) -> dict[str, tuple[float, float]] | None:
  """Computes symmetrically damped `(az, el)` corrections from realized axes."""
  updated = dict(current_targets)
  changed = False
  visited: set[str] = set()

  for name, (cur_az, cur_el) in current_targets.items():
    if name in visited or name not in state.tree or state.tree[name].is_river:
      continue
    info = directions.get(name)
    if not info:
      continue
    realized = info.get('realized')
    target = info.get('target')
    error = info.get('error')
    if realized is None or target is None or error is None:
      continue
    partner = state_lib.resolve_symmetric_element(state, name)
    has_sym_partner = (
        partner is not None
        and partner != name
        and partner in current_targets
        and partner in directions
    )
    if has_sym_partner:
      p_info = directions[partner]
      p_err = float(p_info.get('error') or 0.0)
      if max(float(error), p_err) < _FEEDBACK_MIN_ERROR_DEG:
        visited.update((name, partner))
        continue
      d_az = angle_utils.wrap_angle_centered(
          float(target[0]) - float(realized[0]), 360.0
      )
      d_el = float(target[1]) - float(realized[1])
      p_real = p_info.get('realized')
      p_tgt = p_info.get('target')
      if p_real is not None and p_tgt is not None:
        pd_az = angle_utils.wrap_angle_centered(
            float(p_tgt[0]) - float(p_real[0]), 360.0
        )
        pd_el = float(p_tgt[1]) - float(p_real[1])
        d_az = 0.5 * (d_az - pd_az)
        d_el = 0.5 * (d_el + pd_el)
      new_az = round(
          angle_utils.wrap_angle_centered(
              cur_az + _FEEDBACK_GAIN * d_az, 360.0
          ),
          2,
      )
      new_el = round(
          max(
              -_MAX_ELEVATION_DEG,
              min(_MAX_ELEVATION_DEG, cur_el + _FEEDBACK_GAIN * d_el),
          ),
          2,
      )
      p_cur_az, p_cur_el = current_targets[partner]
      p_new_az = round(
          angle_utils.wrap_angle_centered(
              p_cur_az - _FEEDBACK_GAIN * d_az, 360.0
          ),
          2,
      )
      p_new_el = round(
          max(
              -_MAX_ELEVATION_DEG,
              min(_MAX_ELEVATION_DEG, p_cur_el + _FEEDBACK_GAIN * d_el),
          ),
          2,
      )
      updated[name] = (new_az, new_el)
      updated[partner] = (p_new_az, p_new_el)
      visited.update((name, partner))
      changed = True
    elif float(error) >= _FEEDBACK_MIN_ERROR_DEG:
      d_az = angle_utils.wrap_angle_centered(
          float(target[0]) - float(realized[0]), 360.0
      )
      d_el = float(target[1]) - float(realized[1])
      new_az = round(
          angle_utils.wrap_angle_centered(
              cur_az + _FEEDBACK_GAIN * d_az, 360.0
          ),
          2,
      )
      new_el = round(
          max(
              -_MAX_ELEVATION_DEG,
              min(_MAX_ELEVATION_DEG, cur_el + _FEEDBACK_GAIN * d_el),
          ),
          2,
      )
      updated[name] = (new_az, new_el)
      visited.add(name)
      changed = True
  return updated if changed else None


def figure_pose(
    state: state_lib.ShapingState,
    *,
    feedback_rounds: int = 2,
    narrow: bool = False,
    from_current: bool = False,
    fix_crossings: bool = True,
    deadline: float | None = None,
) -> tuple[state_lib.ShapingState, dict[str, Any]]:
  """Poses a model to its stick-figure directions with root-frame feedback.

  Args:
    state: Initial shaping state.
    feedback_rounds: Maximum closed-loop `(az, el)` correction rounds.
    narrow: Also narrow thin outer limbs and central flaps when clean.
    from_current: Start from `state` rather than resetting to `base_pattern`.
    fix_crossings: Nudge flaps that cross the sagittal mid-plane outward.
    deadline: Optional `time.monotonic()` deadline.

  Returns:
    `(posed_state, diagnostics)`.
  """
  t0 = time.monotonic()

  def _expired() -> bool:
    return deadline is not None and time.monotonic() >= deadline

  work = state if from_current else state.reset()
  base_eval = _score_state(work)
  best_state = work
  best_eval = base_eval
  best_targets = _initial_targets(work, use_bias=True)
  best_flaps_only = False
  has_rivers = any(s.is_river for s in work.tree.children)

  def _try_round0(
      targets: dict[str, tuple[float, float]],
      cur_state: state_lib.ShapingState,
      cur_eval: _ScoreSummary,
      cur_flaps_only: bool,
  ) -> tuple[state_lib.ShapingState, _ScoreSummary, bool, bool]:
    improved = False
    for flaps_only in (False, True):
      if _expired() and cur_state is not work:
        break
      cand = _run_rotation(work, targets, flaps_only=flaps_only)
      if cand is None:
        continue
      ev = _score_state(cand)
      if _accept_candidate(ev, cur_eval):
        cur_state, cur_eval, cur_flaps_only = cand, ev, flaps_only
        improved = True
      if cur_eval.clean_ok and not flaps_only:
        # Only fall back to flaps_only when river cuts caused strain or lowered
        # fidelity on a model that has rivers.
        if not has_rivers or cur_eval.pose_ok:
          break
    return cur_state, cur_eval, cur_flaps_only, improved

  # Initial global rotation (trying both all-sticks and flaps-only if needed).
  best_state, best_eval, best_flaps_only, _ = _try_round0(
      best_targets, best_state, best_eval, best_flaps_only
  )

  # If outward `lateral_bias` caused fold strain or failed to improve on the
  # base, retry round 0 at exact stick-figure angles and report the fallback.
  unbiased_fallback = False
  if best_state is work or not best_eval.clean_ok:
    unbiased_targets = _initial_targets(work, use_bias=False)
    if unbiased_targets != best_targets and not (
        _expired() and best_state is not work
    ):
      best_state, best_eval, best_flaps_only, improved = _try_round0(
          unbiased_targets, best_state, best_eval, best_flaps_only
      )
      if improved:
        best_targets = unbiased_targets
        unbiased_fallback = True

  if narrow and not _expired():
    thin = _thin_leaf_names(work)
    if thin:
      cand_narrow = _run_rotation(
          work, best_targets, narrow_names=thin, flaps_only=best_flaps_only
      )
      if cand_narrow is not None:
        ev_n = _score_state(cand_narrow)
        if _accept_candidate(ev_n, best_eval):
          best_state, best_eval = cand_narrow, ev_n

  # Closed-loop proportional feedback in `rotation_3d`'s own root frame.
  rounds_used = 0
  active_targets = dict(best_targets)
  for _ in range(max(0, feedback_rounds)):
    if _expired() or best_state is work:
      break
    try:
      _, dirs = state_lib.measured_directions(best_state)
    except (RuntimeError, ValueError, AssertionError, KeyError):
      break
    next_targets = _feedback_targets(work, active_targets, dirs)
    if next_targets is None:
      break
    cand = _run_rotation(work, next_targets, flaps_only=best_flaps_only)
    if cand is None:
      break
    ev = _score_state(cand)
    rounds_used += 1
    active_targets = next_targets
    improves_dir = (
        ev.clean_ok
        and ev.pose_ok >= best_eval.pose_ok
        and ev.direction > best_eval.direction + 0.02
        and ev.score >= best_eval.score - 0.05
    )
    if _accept_candidate(ev, best_eval, min_gain=0.01) or improves_dir:
      best_state, best_eval, best_targets = cand, ev, next_targets

  # Side-crossing repair for paired appendages crossing the body's mid-plane.
  repaired_crossings = False
  if fix_crossings and not _expired() and best_eval.num_crossings > 0:
    crossing = pose_alignment.crossing_sticks(
        best_eval.report.get('side_crossings') or []
    )
    if crossing:
      fixed_targets = side_fix_targets(work.tree, crossing, best_targets)
      cand = _run_rotation(work, fixed_targets, flaps_only=best_flaps_only)
      if cand is not None:
        ev = _score_state(cand)
        fewer_cross = (
            ev.clean_ok == best_eval.clean_ok
            and ev.num_crossings < best_eval.num_crossings
            and ev.score >= best_eval.score - 0.25
        )
        if _accept_candidate(ev, best_eval) or fewer_cross:
          best_state, best_eval, best_targets = cand, ev, fixed_targets
          repaired_crossings = True

  # Optional post-pose clean narrowing on remaining thin tips and central flaps.
  if narrow and not _expired():
    seen_pairs: set[str] = set()
    for plan in pose_plan(best_state):
      if _expired():
        break
      if plan.name in seen_pairs or plan.name in best_state.narrowed_elements:
        continue
      if plan.role == ROLE_THIN_LIMB and plan.partner is not None:
        seen_pairs.update((plan.name, plan.partner))
        cand, res = toolbox.apply_tool(
            best_state,
            'apply_symmetric',
            {
                'tool': 'simple_fold',
                'element_name': plan.name,
                'arguments': {
                    'angle': 45.0,
                    'narrow': True,
                    'verify_fold': True,
                },
            },
        )
        if res.ok:
          ev = _score_state(cand)
          if _accept_candidate(ev, best_eval, min_gain=0.01):
            best_state, best_eval = cand, ev
      elif (
          plan.partner is None
          and plan.width <= 1.0
          and 'narrow_central_flap' in plan.recommended_tools
      ):
        seen_pairs.add(plan.name)
        cand, res = toolbox.apply_tool(
            best_state,
            'narrow_central_flap',
            {'element_name': plan.name, 'tip_width': 0.3, 'verify_fold': True},
        )
        if res.ok:
          ev = _score_state(cand)
          if _accept_candidate(ev, best_eval, min_gain=0.01):
            best_state, best_eval = cand, ev

  diag = _build_diagnostics(
      mode='figure',
      state=best_state,
      base_summary=base_eval,
      final_summary=best_eval,
      elapsed=time.monotonic() - t0,
      extra={
          'feedback_rounds_run': rounds_used,
          'flaps_only': best_flaps_only,
          'repaired_crossings': repaired_crossings,
          'unbiased_fallback': unbiased_fallback,
      },
  )
  return best_state, diag


# ------------------------------------------------------------------------------
# 2. Fast zero-folder flat-foldable pose (`greedy_pose`).
# ------------------------------------------------------------------------------


def greedy_pose(
    state: state_lib.ShapingState,
    *,
    narrow: bool = False,
) -> tuple[state_lib.ShapingState, dict[str, Any]]:
  """Applies a fast flat-foldable pose without invoking the 3-D folder."""
  t0 = time.monotonic()
  current = state.reset()
  targets = _initial_targets(current)

  rotated = None
  if narrow:
    thin = _thin_leaf_names(current)
    if thin:
      rotated = _run_rotation(current, targets, narrow_names=thin)
  if rotated is None:
    rotated = _run_rotation(current, targets)
  if rotated is None:
    unbiased = _initial_targets(current, use_bias=False)
    if unbiased != targets:
      rotated = _run_rotation(current, unbiased)
  if rotated is not None:
    current = rotated

  # If global rotation added nothing (e.g. collinear sticks or blocked cuts),
  # fall back to fast 2-D bilateral/midline folds so every posable model shapes.
  if len(current.pattern.creases) == len(current.base_pattern.creases):
    seen: set[str] = set()
    for plan in pose_plan(current):
      if plan.name in seen or plan.role in (ROLE_ROOT, ROLE_SKIP, ROLE_RIVER):
        continue
      angle = plan.target_el if abs(plan.target_el) >= 10.0 else plan.target_az
      if math.isclose(angle, 0.0, abs_tol=5.0):
        continue
      for tool_name, inner_args in (
          (
              'crimp_fold',
              {'angle': angle, 'mode': '2d', 'verify_fold': False},
          ),
          (
              'reverse_fold',
              {'angle': angle, 'mode': '2d', 'verify_fold': False},
          ),
          ('simple_fold', {'angle': angle, 'verify_fold': False}),
      ):
        if plan.partner is not None and plan.partner != plan.name:
          seen.update((plan.name, plan.partner))
          cand, res = toolbox.apply_tool(
              current,
              'apply_symmetric',
              {
                  'tool': tool_name,
                  'element_name': plan.name,
                  'arguments': inner_args,
              },
          )
        else:
          seen.add(plan.name)
          cand, res = toolbox.apply_tool(
              current, tool_name, {'element_name': plan.name, **inner_args}
          )
        if res.ok and cand.pattern.local_flat_foldability():
          current = cand
          break

  # Optional fast 2-D narrowing and flat wide-tip squashing.
  seen_post: set[str] = set()
  for plan in pose_plan(current):
    if plan.name in seen_post:
      continue
    if (
        narrow
        and plan.role == ROLE_THIN_LIMB
        and plan.name not in current.narrowed_elements
    ):
      if plan.partner is not None:
        seen_post.update((plan.name, plan.partner))
        cand, res = toolbox.apply_tool(
            current,
            'apply_symmetric',
            {
                'tool': 'simple_fold',
                'element_name': plan.name,
                'arguments': {
                    'angle': 45.0,
                    'narrow': True,
                    'verify_fold': False,
                },
            },
        )
      else:
        seen_post.add(plan.name)
        cand, res = toolbox.apply_tool(
            current,
            'simple_fold',
            {
                'element_name': plan.name,
                'angle': 45.0,
                'narrow': True,
                'verify_fold': False,
            },
        )
      if res.ok and cand.pattern.local_flat_foldability():
        current = cand
    elif plan.role == ROLE_WIDE_TIP:
      if plan.partner is not None:
        seen_post.update((plan.name, plan.partner))
        cand, res = toolbox.apply_tool(
            current,
            'apply_symmetric',
            {
                'tool': 'squash_fold',
                'element_name': plan.name,
                'arguments': {'open_percent': 1.0, 'verify_fold': False},
            },
        )
      else:
        seen_post.add(plan.name)
        cand, res = toolbox.apply_tool(
            current,
            'squash_fold',
            {
                'element_name': plan.name,
                'open_percent': 1.0,
                'verify_fold': False,
            },
        )
      if res.ok and cand.pattern.local_flat_foldability():
        current = cand

  diag = _build_diagnostics(
      mode='greedy',
      state=current,
      base_summary=None,
      final_summary=None,
      elapsed=time.monotonic() - t0,
  )
  return current, diag


# ------------------------------------------------------------------------------
# 3. Deadline-bounded anatomical coordinate ascent (`search_pose`).
# ------------------------------------------------------------------------------


def _wrap_call(
    plan: ElementPlan, tool_name: str, arguments: Mapping[str, Any]
) -> tuple[str, dict[str, Any]]:
  """Wraps single-element calls in `apply_symmetric` for paired sticks."""
  if plan.partner is not None and plan.partner != plan.name:
    return (
        'apply_symmetric',
        {
            'tool': tool_name,
            'element_name': plan.name,
            'arguments': dict(arguments),
        },
    )
  return tool_name, {'element_name': plan.name, **dict(arguments)}


def _search_candidates(
    state: state_lib.ShapingState,
    summary: _ScoreSummary,
    *,
    verify_fold: bool,
) -> list[tuple[str, dict[str, Any]]]:
  """Generates ordered, symmetry-preserving candidate calls for `state`."""
  calls: list[tuple[str, dict[str, Any]]] = []
  plans = pose_plan(state)
  plan_by_name = {p.name: p for p in plans}

  # 1. Cold start: try simultaneous 3-D figure rotation first.
  if len(state.pattern.creases) == len(state.base_pattern.creases):
    targets = _initial_targets(state)
    all_rots = [
        {'element_name': n, 'az': az, 'el': el}
        for n, (az, el) in targets.items()
    ]
    flap_rots = [
        r for r in all_rots if not state.tree[r['element_name']].is_river
    ]
    if all_rots:
      calls.append(('rotation_3d', {'rotations': all_rots}))
    if flap_rots and len(flap_rots) < len(all_rots):
      calls.append(('rotation_3d', {'rotations': flap_rots}))

  # 2. Aim badly misaligned flaps (worst error first).
  dir_errors = dict(summary.report.get('direction_error_deg') or {})
  misaligned = sorted(
      (
          (float(err), name)
          for name, err in dir_errors.items()
          if float(err) > 30.0
          and name in plan_by_name
          and plan_by_name[name].role not in (ROLE_ROOT, ROLE_RIVER, ROLE_SKIP)
      ),
      reverse=True,
  )
  seen_aim: set[str] = set()
  for _, name in misaligned[:6]:
    plan = plan_by_name[name]
    if plan.name in seen_aim:
      continue
    seen_aim.add(plan.name)
    if plan.partner:
      seen_aim.add(plan.partner)
    if verify_fold:
      calls.append(_wrap_call(plan, 'bend_to_direction', {'max_candidates': 6}))
    else:
      ang = plan.target_el if abs(plan.target_el) >= 10.0 else 45.0
      calls.append(
          _wrap_call(plan, 'crimp_fold', {'angle': ang, 'verify_fold': False})
      )

  # 3. Flare wide plates, narrow thin limbs/central tips, curve chains, and
  #    add 3-D volume on midline rivers and spine.
  seen_role: set[str] = set()
  for plan in plans:
    if plan.name in seen_role or plan.role in (ROLE_ROOT, ROLE_SKIP):
      continue
    seen_role.add(plan.name)
    if plan.partner:
      seen_role.add(plan.partner)

    if plan.role == ROLE_WIDE_TIP:
      calls.append(
          _wrap_call(
              plan,
              'open_flap',
              {'fold_percent': 0.45, 'verify_fold': verify_fold},
          )
      )
      calls.append(
          _wrap_call(
              plan,
              'squash_fold',
              {'open_percent': 0.45, 'verify_fold': verify_fold},
          )
      )
      calls.append(
          _wrap_call(
              plan,
              'fan_out_fold',
              {
                  'move_delta': 1.0,
                  'open_percent': 0.65,
                  'verify_fold': verify_fold,
              },
          )
      )
    elif (
        plan.role == ROLE_THIN_LIMB and plan.name not in state.narrowed_elements
    ):
      calls.append(
          _wrap_call(
              plan,
              'simple_fold',
              {'angle': 45.0, 'narrow': True, 'verify_fold': verify_fold},
          )
      )
    elif plan.role == ROLE_TIP_DOWN and not dir_errors:
      calls.append(
          _wrap_call(
              plan,
              'crimp_fold',
              {'angle': plan.target_el, 'verify_fold': verify_fold},
          )
      )
    elif (
        plan.partner is None
        and plan.width <= 1.0
        and 'narrow_central_flap' in plan.recommended_tools
        and plan.name not in state.narrowed_elements
    ):
      calls.append((
          'narrow_central_flap',
          {
              'element_name': plan.name,
              'tip_width': 0.3,
              'verify_fold': verify_fold,
          },
      ))
    elif plan.role == ROLE_SEGMENT and plan.partner is None:
      direction = 'down' if plan.target_el < 0.0 else 'up'
      calls.append((
          'curve_flap',
          {
              'element_name': plan.name,
              'n_segments': 2,
              'total_bend_deg': 50.0,
              'direction': direction,
              'verify_fold': verify_fold,
          },
      ))
    elif plan.role == ROLE_RIVER and plan.partner is None:
      calls.append((
          'open_flap',
          {
              'element_name': plan.name,
              'fold_percent': 0.65,
              'verify_fold': verify_fold,
          },
      ))
      if 'inscribed_flap' in plan.recommended_tools:
        calls.append((
            'inscribed_flap',
            {
                'element_name': plan.name,
                'depths': [1],
                'fold_percent': 0.5,
                'verify_fold': verify_fold,
            },
        ))

  calls.append(
      ('fold_spine', {'fold_percent': 0.7, 'verify_fold': verify_fold})
  )
  return calls


def search_pose(
    state: state_lib.ShapingState,
    *,
    time_budget_s: float = 60.0,
    warm_state: state_lib.ShapingState | None = None,
    min_gain: float = 0.05,
    evaluate: Callable[[state_lib.ShapingState], float] | None = None,
) -> tuple[state_lib.ShapingState, dict[str, Any]]:
  """Improves fidelity via deadline-bounded, symmetry-aware coordinate ascent.

  Args:
    state: Unshaped or current shaping state.
    time_budget_s: Maximum wall-clock runtime in seconds.
    warm_state: Optional starting state (e.g. output of `figure_pose`).
    min_gain: Minimum score increase required to accept a candidate edit.
    evaluate: Optional custom scalar evaluator (primarily for testing).

  Returns:
    `(best_state, diagnostics)`.
  """
  t0 = time.monotonic()
  deadline = t0 + max(0.01, float(time_budget_s))

  def _eval(st: state_lib.ShapingState) -> _ScoreSummary:
    if evaluate is None:
      return _score_state(st)
    val = float(evaluate(st))
    return _ScoreSummary(
        score=val,
        clean_ok=True,
        pose_ok=True,
        direction=1.0,
        num_crossings=0,
        vertex_error=0.0,
        report={},
    )

  start_state = warm_state if warm_state is not None else state.reset()
  base_eval = _eval(start_state)
  best_state = start_state
  best_eval = base_eval
  verify = evaluate is None

  candidates_tried = 0
  accepted_steps = 0
  for _ in range(2):
    if time.monotonic() >= deadline:
      break
    improved_in_pass = False
    candidates = _search_candidates(best_state, best_eval, verify_fold=verify)
    for tool_name, args in candidates:
      if time.monotonic() >= deadline:
        break
      candidates_tried += 1
      # Bound per-call `fan_out_fold` budget to the remaining search deadline.
      call_state = best_state
      if tool_name == 'fan_out_fold' or (
          tool_name == 'apply_symmetric' and args.get('tool') == 'fan_out_fold'
      ):
        rem = max(0.25, deadline - time.monotonic())
        capped_cfg = dataclasses.replace(
            best_state.config,
            fan_out_time_budget_s=min(
                best_state.config.fan_out_time_budget_s, rem
            ),
            fan_out_solver_timeout_s=min(
                best_state.config.fan_out_solver_timeout_s, max(0.2, 0.5 * rem)
            ),
        )
        call_state = best_state.replace(config=capped_cfg)
      cand, res = toolbox.apply_tool(call_state, tool_name, args)
      if not res.ok:
        continue
      if call_state.config != best_state.config:
        cand = cand.replace(config=best_state.config)
      ev = _eval(cand)
      if _accept_candidate(ev, best_eval, min_gain=min_gain):
        best_state = cand
        best_eval = ev
        accepted_steps += 1
        improved_in_pass = True
    if not improved_in_pass:
      break

  diag = _build_diagnostics(
      mode='search',
      state=best_state,
      base_summary=base_eval,
      final_summary=best_eval,
      elapsed=time.monotonic() - t0,
      extra={
          'candidates_tried': candidates_tried,
          'accepted_steps': accepted_steps,
      },
  )
  return best_state, diag


# ------------------------------------------------------------------------------
# Unified entry points (`pose_state` and `pose_solved`).
# ------------------------------------------------------------------------------


def pose_state(
    state: state_lib.ShapingState,
    config: PoseConfig | None = None,
) -> tuple[state_lib.ShapingState, dict[str, Any]]:
  """Runs the configured automatic pose recipe on `state`."""
  cfg = config or PoseConfig()
  t0 = time.monotonic()
  deadline = t0 + cfg.time_budget_s

  if cfg.mode == 'greedy':
    return greedy_pose(state, narrow=cfg.narrow)
  if cfg.mode == 'figure':
    return figure_pose(
        state,
        feedback_rounds=cfg.feedback_rounds,
        narrow=cfg.narrow,
    )
  if cfg.mode == 'search':
    return search_pose(
        state,
        time_budget_s=cfg.time_budget_s,
        min_gain=cfg.min_gain,
    )

  # `mode == 'auto'`: figure pose (runs to completion) + budgeted local search
  # when `figure` has not yet satisfied both `clean_gate` and `pose_gate`.
  base_eval = _score_state(state.reset())
  best_state, fig_diag = figure_pose(
      state,
      feedback_rounds=cfg.feedback_rounds,
      narrow=cfg.narrow,
  )
  best_eval = _score_state(best_state)

  search_diag: dict[str, Any] = {}
  remaining = deadline - time.monotonic()
  if (
      not (best_eval.pose_ok and best_eval.clean_ok)
      and remaining >= _MIN_SEARCH_REMAINDER_S
  ):
    refined, search_diag = search_pose(
        state,
        time_budget_s=remaining,
        warm_state=best_state,
        min_gain=cfg.min_gain,
    )
    if refined is not best_state:
      ev_ref = _score_state(refined)
      if _accept_candidate(ev_ref, best_eval):
        best_state, best_eval = refined, ev_ref

  if (
      len(best_state.pattern.creases) == len(best_state.base_pattern.creases)
      and time.monotonic() < deadline
  ):
    greedy_st, _ = greedy_pose(state, narrow=cfg.narrow)
    if len(greedy_st.pattern.creases) > len(greedy_st.base_pattern.creases):
      ev_g = _score_state(greedy_st)
      if _accept_candidate(ev_g, best_eval):
        best_state, best_eval = greedy_st, ev_g

  diag = _build_diagnostics(
      mode='auto',
      state=best_state,
      base_summary=base_eval,
      final_summary=best_eval,
      elapsed=time.monotonic() - t0,
      extra={
          'feedback_rounds_run': fig_diag.get('feedback_rounds_run', 0),
          'search_accepted_steps': search_diag.get('accepted_steps', 0),
          'unbiased_fallback': fig_diag.get('unbiased_fallback', False),
      },
  )
  return best_state, diag


def pose_solved(
    solved: bus.SolvedCreasePattern,
    config: PoseConfig | None = None,
) -> tuple[bus.ShapedCreasePattern, dict[str, Any]]:
  """Runs `pose_state` on `solved`, returning `(ShapedCreasePattern, report)`."""
  cfg = config or PoseConfig()
  t0 = time.monotonic()
  initial = state_lib.ShapingState.from_solved(solved)
  posed, diag = pose_state(initial, cfg)
  elapsed = time.monotonic() - t0
  diag = dict(diag, time_taken=round(elapsed, 3))
  strategy = bus.ShapingStrategy(f'pose_{cfg.mode}')
  shaped = posed.to_shaped(
      solved, strategy=strategy, time_taken=elapsed, diagnostics=diag
  )
  return shaped, diag
