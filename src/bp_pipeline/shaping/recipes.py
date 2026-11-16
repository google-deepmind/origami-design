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

"""Deterministic shaping recipes built on the shaping toolbox.

The `pose` recipe turns the stick figure into a per-element shaping plan (what
each limb should do, with a few candidate tool calls each) and applies it
greedily from the base pattern: for every element the first candidate that
keeps the crease pattern flat-foldable is kept, the others are discarded. The
result is a valid, posed base - a stronger starting point for an agent's
manual shaping than the taskqueue heuristic baseline, whose geometric folds
block later narrowing - together with a report of what worked on which
element, so the agent knows where to intervene.

Nothing here calls a model; the plan is a function of the stick figure only.
"""

from collections.abc import Callable, Iterator, Mapping, Sequence
import contextlib
import dataclasses
import math
import time
from typing import Any

from absl import logging

from . import fan_out_shaping
from .. import bus
from ..stick_figures import traversal
from . import fidelity
from . import toolbox as shaping

# Element roles assigned by `pose_plan`.
ROLE_THIN_LIMB = 'thin_limb'  # Narrow flap (legs, antennae, fingers).
ROLE_LIMB = 'limb'  # Medium flap (arms, necks) bent to its azimuth.
ROLE_SEGMENT = 'segment'  # Member of a merged chain (tail/neck segments).
ROLE_WIDE_TIP = 'wide_tip'  # Wide terminal flap (pincers, heads, fins).
ROLE_TIP_DOWN = 'tip_down'  # Short terminal flap pointing down (stinger).
ROLE_RIVER = 'river'
ROLE_ROOT = 'root'
ROLE_SKIP = 'skip'  # Length-1 elements and other unshapeable ones.

_MIN_ANGLE = 8.0
_MAX_ANGLE = 170.0
_NUDGE = 10.0


@dataclasses.dataclass(frozen=True)
class ElementPlan:
  """Shaping plan for one unmerged element.

  Attributes:
    name: Element (unmerged stick) name.
    role: One of the `ROLE_*` constants.
    parent: Name of the stick towards the root, or None for the root.
    relative_azimuth: Azimuth relative to the parent, in degrees (-180, 180].
    relative_elevation: Elevation relative to the parent, in degrees.
    candidates: Tool calls to try in order (`{"name": ..., "arguments": ...}`).
    reason: Why the element got this role/candidates.
  """

  name: str
  role: str
  parent: str | None
  relative_azimuth: float
  relative_elevation: float
  candidates: tuple[dict[str, Any], ...]
  reason: str

  def asdict(self) -> dict[str, Any]:
    return dataclasses.asdict(self)


def _wrap_degrees(angle: float) -> float:
  """Wraps an angle into (-180, 180]."""
  angle = (angle + 180.0) % 360.0 - 180.0
  return 180.0 if angle == -180.0 else angle


def _turn_angle_deg(first, second) -> float:
  """3-D angle in degrees between two sticks' unit direction vectors."""
  az1 = math.radians(first.azimuth_angle)
  el1 = math.radians(first.elevation_angle)
  az2 = math.radians(second.azimuth_angle)
  el2 = math.radians(second.elevation_angle)
  dot = math.cos(el1) * math.cos(el2) * math.cos(az1 - az2) + math.sin(
      el1
  ) * math.sin(el2)
  return math.degrees(math.acos(min(1.0, max(-1.0, dot))))


def _nudged(angle: float) -> float:
  """Moves an angle away from 0/±90/±180 by `_NUDGE` and clamps it.

  Small angles are visually indistinct and right angles tend to collide with
  the grid; the RL shaping prompts recommend nudging them by ~10 degrees.

  Args:
    angle: Angle in degrees, (-180, 180].

  Returns:
    The nudged angle with magnitude in [_MIN_ANGLE, _MAX_ANGLE].
  """
  sign = 1.0 if angle >= 0 else -1.0
  magnitude = abs(angle)
  for anchor in (0.0, 90.0, 180.0):
    if abs(magnitude - anchor) < _NUDGE:
      magnitude = anchor + _NUDGE if magnitude >= anchor else anchor - _NUDGE
      if anchor == 0.0:
        magnitude = _NUDGE + _MIN_ANGLE
      break
  magnitude = min(max(magnitude, _MIN_ANGLE), _MAX_ANGLE)
  return sign * magnitude


def _parents(tree) -> dict[str, str | None]:
  """Maps every stick to the stick through which the root reaches it."""
  parents: dict[str, str | None] = {}
  root = tree[tree.root]
  parents[root.name] = None
  reached_joints = {root.start: root.name, root.end: root.name}
  for stick in traversal.breadth_first_traversal(tree):
    if stick.name == root.name:
      continue
    if stick.start in reached_joints:
      parents[stick.name] = reached_joints[stick.start]
      far_joint = stick.end
    elif stick.end in reached_joints:
      parents[stick.name] = reached_joints[stick.end]
      far_joint = stick.start
    else:  # Disconnected (should not happen for validated trees).
      parents[stick.name] = None
      far_joint = stick.end
    reached_joints.setdefault(far_joint, stick.name)
  return parents


def _chain_members(tree) -> dict[str, tuple[int, int]]:
  """Maps sticks that belong to a merged linear chain to (position, length)."""
  _, merged_names = tree.merge_linear_chains()
  members = {}
  for names in merged_names.values():
    for position, name in enumerate(names):
      members[name] = (position, len(names))
  return members


def _is_tip(stick) -> bool:
  """True for terminal sticks (one joint is a leaf), i.e. flaps with a tip."""
  return bool(stick.is_flap)


def pose_plan(state: shaping.ShapingState) -> list[ElementPlan]:
  """Derives a per-element shaping plan from the stick figure.

  Roles and candidates (tried in order until one keeps flat-foldability):

  *   thin limbs (width <= 0.5, length >= 2): `simple_fold` with `narrow`
      at the element's azimuth relative to its parent (nudged), then the
      opposite sign, then half the angle.
  *   chain segments (members of a merged linear chain, e.g. tail segments):
      `crimp_fold` with the elevation change along the chain (20-70 deg),
      at depth 0.4 then 0.25, both signs.
  *   wide terminal flaps (width >= 2 with a tip): `fan_out_fold_heuristic`
      with `move_delta` +1 then -1.
  *   short terminal flaps pointing down (elevation <= -30, length >= 2):
      `reverse_fold` 60 deg, then 90 deg.
  *   other flaps (arms, necks): `simple_fold` at the relative azimuth,
      without narrowing.
  *   rivers, the root stick and length-1 elements are left alone.

  Args:
    state: Shaping state (the stick figure and base pattern).

  Returns:
    One plan per unmerged element, in breadth-first order from the root.
  """
  tree = state.tree
  parents = _parents(tree)
  chains = _chain_members(tree)
  plans = []
  for stick in traversal.breadth_first_traversal(tree):
    parent_name = parents.get(stick.name)
    parent = tree[parent_name] if parent_name else None
    rel_az = _wrap_degrees(
        stick.azimuth_angle - (parent.azimuth_angle if parent else 0.0)
    )
    rel_el = stick.elevation_angle - (parent.elevation_angle if parent else 0.0)
    common = dict(
        name=stick.name,
        parent=parent_name,
        relative_azimuth=round(rel_az, 1),
        relative_elevation=round(rel_el, 1),
    )
    if parent is None:
      plans.append(
          ElementPlan(
              role=ROLE_ROOT,
              candidates=(),
              reason='root stick is never shaped',
              **common,
          )
      )
      continue
    if stick.is_river:
      plans.append(
          ElementPlan(
              role=ROLE_RIVER,
              candidates=(),
              reason='rivers are left to manual shaping',
              **common,
          )
      )
      continue

    def call(name, **arguments):
      # pylint: disable-next=cell-var-from-loop
      args_dict = dict(arguments, element_name=stick.name)
      return {'name': name, 'arguments': args_dict}

    if stick.length < 2:
      if stick.width <= 0.5 and _is_tip(stick):
        # Too short for bends, but probes showed length-1 thin tips narrow.
        plans.append(
            ElementPlan(
                role=ROLE_THIN_LIMB,
                candidates=(
                    call(
                        'simple_fold', angle=_MIN_ANGLE, depth=0.0, narrow=True
                    ),
                    call(
                        'simple_fold', angle=-_MIN_ANGLE, depth=0.0, narrow=True
                    ),
                ),
                reason='length-1 thin tip: narrow only',
                **common,
            )
        )
      elif (
          stick.name.endswith('_l')
          or stick.name.endswith('_r')
          or shaping.resolve_symmetric_element(state, stick.name)
      ):
        plans.append(
            ElementPlan(
                role=ROLE_LIMB,
                candidates=(),
                reason='length-1 bilateral element: 3-D rotation',
                **common,
            )
        )
      else:
        plans.append(
            ElementPlan(
                role=ROLE_SKIP,
                candidates=(),
                reason='length-1 elements cannot be shaped',
                **common,
            )
        )
      continue

    if stick.width >= 2 and _is_tip(stick):
      plans.append(
          ElementPlan(
              role=ROLE_WIDE_TIP,
              candidates=(
                  call('open_flap', fold_percent=0.3, ridge_only=True),
                  call('squash_fold', ridge_index=0),
                  call('squash_fold', ridge_index=1),
                  call('fan_out_fold_heuristic', move_delta=1),
                  call('fan_out_fold', orientation='horizontal', move_delta=-1),
              ),
              reason=f'wide terminal flap (width {stick.width:g})',
              **common,
          )
      )
      continue
    if _is_tip(stick) and rel_el <= -30 and stick.length <= 2:
      plans.append(
          ElementPlan(
              role=ROLE_TIP_DOWN,
              candidates=(
                  call(
                      'reverse_fold',
                      angle=60.0,
                      orientation='horizontal',
                      fold_type='valley',
                  ),
                  call(
                      'reverse_fold',
                      angle=90.0,
                      orientation='horizontal',
                      fold_type='valley',
                  ),
                  call('reverse_fold_3d', angle=45.0, depth=0.5),
                  call('reverse_fold_3d', angle=60.0, depth=0.5),
              ),
              reason=f'short terminal flap pointing down ({rel_el:.0f} deg)',
              **common,
          )
      )
      continue
    chain = chains.get(stick.name)
    # Collinear with the parent in plan view (continuing the axis forwards or
    # backwards): the only meaningful operation is a bend by elevation.
    collinear = min(abs(rel_az), 180.0 - abs(rel_az)) < 15.0
    if (chain is not None and chain[0] > 0) or (
        collinear and not _is_tip(stick) and stick.width > 0.5
    ):
      turn_deg = _turn_angle_deg(parent, stick)
      if turn_deg < 8.0:
        plans.append(
            ElementPlan(
                role=ROLE_SKIP,
                candidates=(),
                reason='continues its parent (no azimuth/elevation change)',
                **common,
            )
        )
        continue
      # A segment inside a merged chain: bend it by the 3-D turn angle.
      angle = min(max(turn_deg, 20.0), 70.0)
      position = (
          f'segment {chain[0] + 1}/{chain[1]} of a merged chain'
          if chain
          else 'axis segment'
      )
      plans.append(
          ElementPlan(
              role=ROLE_SEGMENT,
              candidates=(
                  call('crimp_fold', angle=angle, depth=0.4),
                  call('crimp_fold', angle=-angle, depth=0.4),
                  call('crimp_fold', angle=angle, depth=0.25),
                  call('crimp_fold', angle=-angle, depth=0.25),
              ),
              reason=f'{position}; elevation change {rel_el:.0f} deg',
              **common,
          )
      )
      continue
    angle = _nudged(rel_az if abs(rel_az) >= _MIN_ANGLE else 45.0)
    if state.central_crease_removed:
      # With the central crease removed, left/right copies take the same
      # sign; otherwise the mirrored azimuths already give opposite signs.
      angle = abs(angle)
    half = math.copysign(max(abs(angle) / 2.0, _MIN_ANGLE), angle)
    if stick.width <= 0.5:
      plans.append(
          ElementPlan(
              role=ROLE_THIN_LIMB,
              candidates=(
                  call('simple_fold', angle=angle, depth=0.0, narrow=True),
                  call('simple_fold', angle=-angle, depth=0.0, narrow=True),
                  call('simple_fold', angle=half, depth=0.0, narrow=True),
                  call('simple_fold', angle=angle, depth=0.0, narrow=False),
              ),
              reason=(
                  f'thin flap (width {stick.width:g}); relative azimuth'
                  f' {rel_az:.0f} deg'
              ),
              **common,
          )
      )
      continue
    plans.append(
        ElementPlan(
            role=ROLE_LIMB,
            candidates=(
                call('simple_fold', angle=angle, depth=0.0),
                call('simple_fold', angle=-angle, depth=0.0),
                call('simple_fold', angle=half, depth=0.0),
            ),
            reason=(
                f'flap (width {stick.width:g}); relative azimuth'
                f' {rel_az:.0f} deg'
            ),
            **common,
        )
    )
  return plans


# Order in which roles are applied: few, structural operations first; then the
# chain segments (tail, neck) in one 3-D rotation while no neighbouring flap
# has been narrowed yet (narrowing first made the rotation fail with "tips are
# too close" on a 2-river scorpion) - crimps, which shorten segments, only as
# a fallback, alternating so adjacent crimps do not collide; then the many
# narrow folds; tips last.
_ROLE_ORDER = (
    ROLE_LIMB,
    ROLE_THIN_LIMB,
    ROLE_SEGMENT,
    ROLE_TIP_DOWN,
    ROLE_WIDE_TIP,
)


def _rotation_call(tree, names: Sequence[str]) -> dict[str, Any]:
  """A `rotation_3d` call posing `names` at their stick-figure angles."""
  rotations = [
      {
          'element_name': name,
          'el': tree[name].elevation_angle,
          'az': tree[name].azimuth_angle,
          'depth': 0.0,
      }
      for name in names
  ]
  return {'name': 'rotation_3d', 'arguments': {'rotations': rotations}}


def _symmetric_rotation_candidates(
    tree: Any,
    left_name: str,
    right_name: str,
    role: str = '',
) -> list[dict[str, Any]]:
  """Generates candidate 3-D rotations for a bilateral symmetric pair."""
  rot_cands = []
  try:
    rot_cands.append(_rotation_call(tree, [left_name, right_name]))
    az_l = tree[left_name].azimuth_angle
    az_r = tree[right_name].azimuth_angle
    el_l = tree[left_name].elevation_angle
    el_r = tree[right_name].elevation_angle
    # No azimuth-negated candidates: `rotation_3d` takes absolute directions
    # in the figure frame, so negating the azimuth puts the left limb on the
    # right side (crossed legs in the angle investigation).
    if el_l <= -85.0 or el_r <= -85.0:
      parent_l = getattr(tree[left_name], 'parent', None)
      splay_az_l = (
          tree[parent_l].azimuth_angle
          if parent_l and parent_l in tree
          else az_l
      )
      parent_r = getattr(tree[right_name], 'parent', None)
      splay_az_r = (
          tree[parent_r].azimuth_angle
          if parent_r and parent_r in tree
          else az_r
      )
      if abs(splay_az_l - splay_az_r) < 1e-3:
        splay_az_l = 90.0 if az_l >= 0 else -90.0
        splay_az_r = -splay_az_l
      for splay_el in (-75.0, -80.0):
        rot_cands.append({
            'name': 'rotation_3d',
            'arguments': {
                'rotations': [
                    {
                        'element_name': left_name,
                        'el': splay_el,
                        'az': splay_az_l,
                        'depth': 0.0,
                    },
                    {
                        'element_name': right_name,
                        'el': splay_el,
                        'az': splay_az_r,
                        'depth': 0.0,
                    },
                ]
            },
        })
    elif el_l <= -30.0:
      # Keep each limb on its own side: `left_name` is on the side of az_l.
      side = 1.0 if az_l >= 0 else -1.0
      az_pairs = [(az_l, -az_l)]
      for az in (105.0, 120.0, 90.0, 155.0):
        if abs(az - abs(az_l)) > 1e-3:
          az_pairs.append((side * az, -side * az))
      for cand_el in (el_l, -45.0, -60.0, -75.0, -85.0):
        for az_a, az_b in az_pairs:
          rot_cands.append({
              'name': 'rotation_3d',
              'arguments': {
                  'rotations': [
                      {
                          'element_name': left_name,
                          'el': cand_el,
                          'az': az_a,
                          'depth': 0.0,
                      },
                      {
                          'element_name': right_name,
                          'el': cand_el,
                          'az': az_b,
                          'depth': 0.0,
                      },
                  ]
              },
          })
  except Exception:
    pass
  return rot_cands


def _rank_rotation_candidates(
    tree: Any,
    candidates: Sequence[dict[str, Any]],
    left_name: str,
    right_name: str,
) -> list[tuple[float, dict[str, Any]]]:
  """Ranks rotation candidates in microseconds using 2-D alignment + splay."""
  el_l = tree[left_name].elevation_angle if left_name in tree else -90.0
  az_l = tree[left_name].azimuth_angle if left_name in tree else 0.0
  el_r = tree[right_name].elevation_angle if right_name in tree else -90.0
  az_r = tree[right_name].azimuth_angle if right_name in tree else 0.0

  def _spherical_to_cart(
      el_deg: float, az_deg: float
  ) -> tuple[float, float, float]:
    el_rad = math.radians(el_deg)
    az_rad = math.radians(az_deg)
    return (
        math.cos(el_rad) * math.cos(az_rad),
        math.cos(el_rad) * math.sin(az_rad),
        math.sin(el_rad),
    )

  t_l = _spherical_to_cart(el_l, az_l)
  t_r = _spherical_to_cart(el_r, az_r)

  is_vertical_leg = el_l <= -75.0 or el_r <= -75.0

  seen_configs = set()
  scored = []
  for call in candidates:
    rot_list = call.get('arguments', {}).get('rotations', [])
    rot_map = {r['element_name']: r for r in rot_list if isinstance(r, dict)}
    r_l = rot_map.get(left_name)
    r_r = rot_map.get(right_name)
    if not r_l or not r_r:
      continue
    c_el_l = float(r_l.get('el', el_l))
    c_az_l = float(r_l.get('az', az_l))
    c_el_r = float(r_r.get('el', el_r))
    c_az_r = float(r_r.get('az', az_r))

    config_key = (
        round(c_el_l, 1),
        round(c_az_l, 1),
        round(c_el_r, 1),
        round(c_az_r, 1),
    )
    if config_key in seen_configs:
      continue
    seen_configs.add(config_key)

    v_l = _spherical_to_cart(c_el_l, c_az_l)
    v_r = _spherical_to_cart(c_el_r, c_az_r)

    dot_l = v_l[0] * t_l[0] + v_l[1] * t_l[1] + v_l[2] * t_l[2]
    dot_r = v_r[0] * t_r[0] + v_r[1] * t_r[1] + v_r[2] * t_r[2]
    align = 0.5 * (dot_l + dot_r)

    sym_el_diff = abs(c_el_l - c_el_r)
    sym_az_sum = abs(c_az_l + c_az_r)
    sym_az_sum = abs((sym_az_sum + 180.0) % 360.0 - 180.0)
    sym_score = 1.0 - min(1.0, (sym_el_diff + sym_az_sum) / 90.0)

    y_sep = abs(v_l[1] - v_r[1])

    score = 0.70 * align + 0.30 * sym_score
    scored.append((score, call))

  scored.sort(key=lambda x: x[0], reverse=True)
  return scored


def _evaluate_rot_candidates(
    state: shaping.ShapingState,
    candidates: Sequence[dict[str, Any]],
    left_name: str,
    right_name: str,
) -> tuple[shaping.ShapingState | None, dict[str, Any] | None, str, float]:
  """Evaluates ranked rotation candidates with fast early exit and at most 2 folds."""
  ranked = _rank_rotation_candidates(
      state.tree, candidates, left_name, right_name
  )
  if not ranked:
    return None, None, 'no candidates', 0.0

  best_state = None
  best_call = None
  best_score = -1.0
  best_msg = ''

  topo = fidelity.topology(state.tree)
  jl = topo.distal.get(left_name)
  jr = topo.distal.get(right_name)

  folds_attempted = 0
  calls_attempted = 0
  max_folds = 3
  max_calls = 3

  for _, call in ranked:
    calls_attempted += 1
    if calls_attempted > max_calls:
      break
    try:
      cand_state, res = shaping.apply_tool(
          state, call['name'], call['arguments']
      )
      if not res.ok or 'left out' in res.message:
        continue
      if not cand_state.pattern.local_flat_foldability():
        continue

      folds_attempted += 1
      m = fidelity.FoldedModel.from_pattern(cand_state.pattern)
      if m is None or getattr(m, 'vertices_3d', None) is None:
        continue

      score = 0.5
      if jl and jr:
        try:
          g = fidelity.measure(state.tree, state.paper_layout, m)
          fl = g['joints'][jl]['folded']
          fr = g['joints'][jr]['folded']
          tl = g['joints'][jl]['target']
          tr = g['joints'][jr]['target']
          if (
              fl is not None
              and fr is not None
              and tl is not None
              and tr is not None
          ):
            tg = math.dist(tl, tr)
            if tg > 1e-6:
              fg = math.dist(fl, fr)
              gap_ratio = fg / tg
              score = (
                  min(1.0, gap_ratio)
                  if gap_ratio <= 1.5
                  else max(0.1, 1.0 - (gap_ratio - 1.5))
              )
            # A wide gap is not enough: both limbs must point where the
            # stick figure points (a swapped or reversed pair can have the
            # right gap).
            errors = fidelity.components(g)['details']['direction_error_deg']
            pair_errors = [
                errors[n] for n in (left_name, right_name) if n in errors
            ]
            if pair_errors:
              aim = sum(max(0.0, 1.0 - e / 90.0) for e in pair_errors) / len(
                  pair_errors
              )
              score = 0.5 * score + 0.5 * aim
        except Exception:
          pass

      if score > best_score:
        best_score = score
        best_state = cand_state
        best_call = call
        best_msg = res.message
        if score >= 0.85:
          break

      if folds_attempted >= max_folds:
        break
    except Exception:
      pass

  return best_state, best_call, best_msg, best_score


def apply_pose(
    state: shaping.ShapingState,
    plans: Sequence[ElementPlan],
    *,
    roles: Sequence[str] = _ROLE_ORDER,
    spine_fold_percent: float = 0.0,
) -> tuple[shaping.ShapingState, list[dict[str, Any]]]:
  """Applies a pose plan greedily: first flat-foldable candidate per element.

  Symmetric (L, R) pairs are posed together with `apply_symmetric`, preserving
  exact 3-D bilateral symmetry and eliminating sky-legs.
  Chain segments (tails, necks) are posed together per merged chain with one
  `rotation_3d` call at their stick-figure angles, which curls a tail in 3-D
  where crimps only shorten it; if that call fails (or the chain was narrowed),
  segments are rotated one by one and the crimp candidates are the last resort
  (in two passes, every other segment first, so adjacent crimps do not block
  each other).

  Args:
    state: Starting state (normally the base pattern, `state.reset()`).
    plans: Output of `pose_plan`.
    roles: Roles to apply, in order.
    spine_fold_percent: Optional 3-D spine fold percent along the midline.

  Returns:
    The shaped state and one report entry per planned element:
    `{name, role, applied: call | None, attempts: [{call, ok, message}]}`.
  """
  plan_by_name = {p.name: p for p in plans}
  paired_partner: dict[str, str] = {}
  for stick in traversal.breadth_first_traversal(state.tree):
    if stick.is_river or stick.name == state.tree.root:
      continue
    sym = shaping.resolve_symmetric_element(state, stick.name)
    if sym and sym in state.tree:
      paired_partner[stick.name] = sym
      paired_partner[sym] = stick.name

  by_role: dict[str, list[ElementPlan]] = {}
  for plan in plans:
    by_role.setdefault(plan.role, []).append(plan)
  report = []
  done: set[str] = set()

  def attempt(candidate):
    nonlocal state
    new_state, result = shaping.apply_tool(
        state, candidate['name'], candidate['arguments']
    )
    if result.ok:
      state = new_state
    return result

  def run(plan: ElementPlan, extra_candidates=()) -> None:
    if plan.name in paired_partner:
      # Bilateral symmetry guarantee: an element with a resolved symmetric
      # partner must NEVER be folded unilaterally.
      return
    if getattr(state.tree, 'is_symmetric_tree', False):
      # In symmetric trees, unilateral folding of midline elements destroys symmetry.
      return
    attempts = []
    applied = None
    for candidate in tuple(extra_candidates) + plan.candidates:
      result = attempt(candidate)
      attempts.append({
          'call': candidate,
          'ok': result.ok,
          'message': result.message[:200],
      })
      if result.ok:
        applied = candidate
        break
    logging.info(
        'pose %s (%s): %s',
        plan.name,
        plan.role,
        'applied ' + applied['name'] if applied else 'no candidate worked',
    )
    done.add(plan.name)
    report.append({
        'name': plan.name,
        'role': plan.role,
        'applied': applied,
        'attempts': attempts,
    })

  for role in roles:
    plans_for_role = by_role.get(role, [])
    if role == ROLE_SEGMENT and plans_for_role:
      midline_plans = [
          p for p in plans_for_role if p.name not in paired_partner
      ]
      symmetric_plans = [p for p in plans_for_role if p.name in paired_partner]

      if midline_plans:
        if getattr(state.tree, 'is_symmetric_tree', False):
          sagittal_names = [
              stick.name
              for stick in traversal.breadth_first_traversal(state.tree)
              if stick.name != state.tree.root
              and stick.name not in paired_partner
              and not any(
                  body_kw in stick.name.lower()
                  for body_kw in (
                      'thorax',
                      'pelvis',
                      'carapace',
                      'abdomen',
                      'torso',
                      'body',
                  )
              )
              and (
                  abs(stick.azimuth_angle) < 1e-3
                  or abs(abs(stick.azimuth_angle) - 180.0) < 1e-3
              )
          ]
          if sagittal_names:
            joint_call = _rotation_call(state.tree, sagittal_names)
            res = attempt(joint_call)
            logging.info(
                'pose sagittal midline %s: %s',
                sagittal_names,
                res.message[:200],
            )
            if res.ok:
              for name in sagittal_names:
                done.add(name)
                report.append({
                    'name': name,
                    'role': ROLE_SEGMENT,
                    'applied': joint_call,
                    'attempts': [{
                        'call': joint_call,
                        'ok': True,
                        'message': res.message,
                    }],
                })
        else:
          _, merged_names = state.tree.merge_linear_chains()
          chain_key_by_name = {}
          for merged_name, sub_names in merged_names.items():
            for sub_name in sub_names:
              chain_key_by_name[sub_name] = merged_name
          narrowed_chains = {
              chain_key_by_name.get(name, name)
              for name in state.narrowed_elements
          }
          groups: dict[str, list[ElementPlan]] = {}
          for plan in midline_plans:
            key = chain_key_by_name.get(plan.name, plan.name)
            groups.setdefault(key, []).append(plan)

          tips_by_chain: dict[str, list[ElementPlan]] = {}
          for plan in by_role.get(ROLE_TIP_DOWN, []):
            if plan.name not in paired_partner:
              key = chain_key_by_name.get(plan.name)
              if key is not None:
                tips_by_chain.setdefault(key, []).append(plan)

          for chain_key, group_plans in groups.items():
            is_narrowed = chain_key in narrowed_chains
            if not is_narrowed:
              chain_plans = group_plans + tips_by_chain.get(chain_key, [])
              names = [plan.name for plan in chain_plans]
              joint_call = _rotation_call(state.tree, names)
              result = attempt(joint_call)
              logging.info(
                  'pose midline segments %s: %s', names, result.message[:200]
              )
              if result.ok:
                for plan in chain_plans:
                  done.add(plan.name)
                  report.append({
                      'name': plan.name,
                      'role': plan.role,
                      'applied': joint_call,
                      'attempts': [{
                          'call': joint_call,
                          'ok': True,
                          'message': result.message,
                      }],
                  })
                continue
            for parity in (0, 1):
              for index, plan in enumerate(group_plans):
                if index % 2 == parity:
                  extra = (
                      ()
                      if is_narrowed
                      else (_rotation_call(state.tree, [plan.name]),)
                  )
                  run(plan, extra)

      for plan in symmetric_plans:
        if plan.name in done:
          continue
        sym_name = paired_partner[plan.name]
        if sym_name in done:
          done.add(plan.name)
          continue
        partner_plan = plan_by_name.get(sym_name)
        if not partner_plan:
          done.add(plan.name)
          continue

        sym_applied = False
        rot_cands = _symmetric_rotation_candidates(
            state.tree, plan.name, sym_name, ROLE_SEGMENT
        )
        best_rot_state, best_rot_call, best_rot_msg, best_rot_score = (
            _evaluate_rot_candidates(state, rot_cands, plan.name, sym_name)
        )
        if best_rot_state is not None:
          state = best_rot_state
          sym_applied = True
          done.add(plan.name)
          done.add(sym_name)
          report.append({
              'name': plan.name,
              'role': plan.role,
              'applied': best_rot_call,
              'attempts': [{
                  'call': best_rot_call,
                  'ok': True,
                  'message': best_rot_msg[:200],
              }],
          })
          report.append({
              'name': sym_name,
              'role': partner_plan.role,
              'applied': best_rot_call,
              'attempts': [{
                  'call': best_rot_call,
                  'ok': True,
                  'message': best_rot_msg[:200],
              }],
          })
          logging.info(
              'pose symmetric segments %s + %s: applied joint rotation'
              ' (gap_score=%.2f)',
              plan.name,
              sym_name,
              best_rot_score,
          )
          continue

        attempts_l = []
        attempts_r = []
        sym_attempts = 0
        for cand in plan.candidates:
          if sym_attempts >= 2:
            break
          tool_name = cand['name']
          args = dict(cand['arguments'])
          args.pop('element_name', None)
          # Pre-filter with 2-D local flat-foldability before expensive 3-D candidate evaluation
          try:
            test_st, test_res = shaping.apply_tool(
                state, tool_name, {'element_name': plan.name, **args}
            )
            if not test_res.ok or not test_st.pattern.local_flat_foldability():
              continue
          except Exception:
            continue
          sym_args = {
              'tool': tool_name,
              'left_element': plan.name,
              'right_element': sym_name,
              **args,
          }
          sym_attempts += 1
          new_st, res = shaping.apply_tool(state, 'apply_symmetric', sym_args)
          attempts_l.append(
              {'call': cand, 'ok': res.ok, 'message': res.message[:200]}
          )
          attempts_r.append(
              {'call': cand, 'ok': res.ok, 'message': res.message[:200]}
          )
          if res.ok:
            state = new_st
            sym_applied = True
            done.add(plan.name)
            done.add(sym_name)
            report.append({
                'name': plan.name,
                'role': plan.role,
                'applied': cand,
                'attempts': attempts_l,
            })
            report.append({
                'name': sym_name,
                'role': partner_plan.role,
                'applied': cand,
                'attempts': attempts_r,
            })
            logging.info(
                'pose symmetric segments %s + %s: applied %s',
                plan.name,
                sym_name,
                tool_name,
            )
            break

        if not sym_applied:
          done.add(plan.name)
          done.add(sym_name)
          report.append({
              'name': plan.name,
              'role': plan.role,
              'applied': None,
              'attempts': attempts_l,
          })
          report.append({
              'name': sym_name,
              'role': partner_plan.role,
              'applied': None,
              'attempts': attempts_r,
          })
          logging.info(
              'pose symmetric segments %s + %s: rolled back cleanly',
              plan.name,
              sym_name,
          )
      continue

    # Pair symmetric (L, R) elements in plans_for_role
    paired: dict[str, ElementPlan] = {}
    paired_names: set[str] = set()
    for p in plans_for_role:
      if p.name in done or p.name in paired_names:
        continue
      sym_name = shaping.resolve_symmetric_element(state, p.name)
      if sym_name:
        for p_other in plans_for_role:
          if (
              p_other.name == sym_name
              and p_other.name not in done
              and p_other.name not in paired_names
          ):
            paired[p.name] = p_other
            paired[p_other.name] = p
            paired_names.add(p.name)
            paired_names.add(p_other.name)
            break

    for plan in plans_for_role:
      if plan.name in done:
        continue
      if plan.name in paired:
        partner_plan = paired[plan.name]
        if partner_plan.name in done:
          continue

        sym_applied = False
        if role in (ROLE_THIN_LIMB, ROLE_LIMB, ROLE_TIP_DOWN, ROLE_SEGMENT):
          rot_cands = _symmetric_rotation_candidates(
              state.tree, plan.name, partner_plan.name, role
          )
          best_rot_state, best_rot_call, best_rot_msg, best_rot_score = (
              _evaluate_rot_candidates(
                  state, rot_cands, plan.name, partner_plan.name
              )
          )
          if best_rot_state is not None:
            state = best_rot_state
            sym_applied = True
            done.add(plan.name)
            done.add(partner_plan.name)
            report.append({
                'name': plan.name,
                'role': plan.role,
                'applied': best_rot_call,
                'attempts': [{
                    'call': best_rot_call,
                    'ok': True,
                    'message': best_rot_msg[:200],
                }],
            })
            report.append({
                'name': partner_plan.name,
                'role': partner_plan.role,
                'applied': best_rot_call,
                'attempts': [{
                    'call': best_rot_call,
                    'ok': True,
                    'message': best_rot_msg[:200],
                }],
            })
            logging.info(
                'pose symmetric %s + %s: applied rotation (gap_score=%.2f)',
                plan.name,
                partner_plan.name,
                best_rot_score,
            )
            continue

        if role == ROLE_WIDE_TIP and getattr(
            state.tree, 'is_symmetric_tree', False
        ):
          try:
            joint_call = _rotation_call(
                state.tree, [plan.name, partner_plan.name]
            )
            cand_state, res = shaping.apply_tool(
                state, joint_call['name'], joint_call['arguments']
            )
            if res.ok and 'left out' not in res.message:
              state = cand_state
              sym_applied = True
              done.add(plan.name)
              done.add(partner_plan.name)
              report.append({
                  'name': plan.name,
                  'role': plan.role,
                  'applied': joint_call,
                  'attempts': [{
                      'call': joint_call,
                      'ok': True,
                      'message': res.message[:200],
                  }],
              })
              report.append({
                  'name': partner_plan.name,
                  'role': partner_plan.role,
                  'applied': joint_call,
                  'attempts': [{
                      'call': joint_call,
                      'ok': True,
                      'message': res.message[:200],
                  }],
              })
              logging.info(
                  'pose symmetric %s + %s: applied joint rotation',
                  plan.name,
                  partner_plan.name,
              )
              continue
          except Exception:
            pass

        attempts_l = []
        attempts_r = []
        sym_attempts = 0
        for cand in plan.candidates:
          if sym_attempts >= 2:
            break
          tool_name = cand['name']
          args = dict(cand['arguments'])
          args.pop('element_name', None)
          # Pre-filter with 2-D local flat-foldability before expensive 3-D candidate evaluation
          try:
            test_st, test_res = shaping.apply_tool(
                state, tool_name, {'element_name': plan.name, **args}
            )
            if not test_res.ok or not test_st.pattern.local_flat_foldability():
              continue
          except Exception:
            continue
          sym_args = {
              'tool': tool_name,
              'left_element': plan.name,
              'right_element': partner_plan.name,
              **args,
          }
          sym_attempts += 1
          new_st, res = shaping.apply_tool(state, 'apply_symmetric', sym_args)
          attempts_l.append({
              'call': cand,
              'ok': res.ok,
              'message': res.message[:200],
          })
          attempts_r.append({
              'call': cand,
              'ok': res.ok,
              'message': res.message[:200],
          })
          if res.ok:
            state = new_st
            sym_applied = True
            done.add(plan.name)
            done.add(partner_plan.name)
            report.append({
                'name': plan.name,
                'role': plan.role,
                'applied': cand,
                'attempts': attempts_l,
            })
            report.append({
                'name': partner_plan.name,
                'role': partner_plan.role,
                'applied': cand,
                'attempts': attempts_r,
            })
            logging.info(
                'pose symmetric %s + %s: applied %s',
                plan.name,
                partner_plan.name,
                tool_name,
            )
            break
        if not sym_applied:
          try:
            joint_call = _rotation_call(
                state.tree, [plan.name, partner_plan.name]
            )
            cand_state, res = shaping.apply_tool(
                state, joint_call['name'], joint_call['arguments']
            )
            if res.ok and 'left out' not in res.message:
              state = cand_state
              sym_applied = True
              done.add(plan.name)
              done.add(partner_plan.name)
              report.append({
                  'name': plan.name,
                  'role': plan.role,
                  'applied': joint_call,
                  'attempts': (
                      attempts_l + [{
                          'call': joint_call,
                          'ok': True,
                          'message': res.message[:200],
                      }]
                  ),
              })
              report.append({
                  'name': partner_plan.name,
                  'role': partner_plan.role,
                  'applied': joint_call,
                  'attempts': (
                      attempts_r + [{
                          'call': joint_call,
                          'ok': True,
                          'message': res.message[:200],
                      }]
                  ),
              })
              logging.info(
                  'pose symmetric %s + %s: applied joint rotation fallback',
                  plan.name,
                  partner_plan.name,
              )
          except Exception as e:  # pylint: disable=broad-exception-caught
            logging.info(
                'pose symmetric %s + %s: joint rotation fallback failed: %s',
                plan.name,
                partner_plan.name,
                e,
            )
        if not sym_applied:
          logging.info(
              'pose symmetric %s + %s: no symmetric candidate worked, rolling'
              ' back cleanly to preserve bilateral symmetry',
              plan.name,
              partner_plan.name,
          )
          done.add(plan.name)
          done.add(partner_plan.name)
          report.append({
              'name': plan.name,
              'role': plan.role,
              'applied': None,
              'attempts': attempts_l,
          })
          report.append({
              'name': partner_plan.name,
              'role': partner_plan.role,
              'applied': None,
              'attempts': attempts_r,
          })
      else:
        run(plan)

  for plan in plans:
    if plan.name not in done:
      report.append({
          'name': plan.name,
          'role': plan.role,
          'applied': None,
          'attempts': [],
      })

  if spine_fold_percent > 0.0 and 'fold_spine_3d' in shaping.TOOL_NAMES:
    try:
      flaps, rivers = state.shaping_elements
      body_rivers = [
          r
          for r in rivers
          if any(
              kw in r.name.lower()
              for kw in (
                  'thorax',
                  'pelvis',
                  'body',
                  'abdomen',
                  'trunk',
              )
          )
      ]
      if body_rivers:
        all_c = []
        for r in body_rivers:
          all_c.extend(r.bank_1)
          all_c.extend(r.bank_2)
        ys = [c.p1.y for c in all_c] + [c.p2.y for c in all_c]
        if ys:
          spine_args = {
              'fold_percent': spine_fold_percent,
              'y_min': max(0.0, min(ys)),
              'y_max': min(1.0, max(ys)),
              'verify_fold': False,
          }
          new_state, spine_res = shaping.apply_tool(
              state, 'fold_spine_3d', spine_args
          )
          if spine_res.ok:
            state = new_state
            logging.info(
                'apply_pose fold_spine_3d: %s', spine_res.message[:200]
            )
            spine_call = {'name': 'fold_spine_3d', 'arguments': spine_args}
            for entry in report:
              if (
                  any(
                      kw in entry['name'].lower()
                      for kw in ('pelvis', 'thorax', 'body', 'abdomen', 'trunk')
                  )
                  and entry.get('applied') is None
              ):
                entry['applied'] = spine_call
                entry['attempts'].append({
                    'call': spine_call,
                    'ok': True,
                    'message': spine_res.message[:200],
                })
    except Exception as e:  # pylint: disable=broad-exception-caught
      logging.info('apply_pose fold_spine_3d skipped: %s', e)

  return state, report


def pose(
    state: shaping.ShapingState,
    *,
    spine_fold_percent: float = 0.0,
) -> tuple[shaping.ShapingState, list[ElementPlan], list[dict[str, Any]]]:
  """Plans and applies the greedy pose recipe from the base pattern.

  Args:
    state: Shaping state; shaping starts from `state.reset()`.
    spine_fold_percent: Optional 3-D spine fold percent along the midline.

  Returns:
    The posed state, the plan and the application report.
  """
  base_state = state.reset()
  base_rep = fidelity.score_pattern(
      base_state.tree, base_state.packing, base_state.pattern
  )
  base_sym = float(base_rep.get('components', {}).get('symmetry', 0.0))

  plans = pose_plan(base_state)
  posed, report = apply_pose(
      base_state, plans, spine_fold_percent=spine_fold_percent
  )
  posed_rep = fidelity.score_pattern(posed.tree, posed.packing, posed.pattern)
  posed_sym = float(posed_rep.get('components', {}).get('symmetry', 0.0))

  if posed_sym < base_sym:
    logging.warning(
        'pose: symmetry regressed (%.3f -> %.3f)',
        base_sym,
        posed_sym,
    )

  applied = sum(1 for entry in report if entry['applied'])
  logging.info(
      'pose: %d/%d planned elements shaped, %d creases, sym=%.3f',
      applied,
      sum(1 for plan in plans if plan.candidates),
      len(posed.pattern.creases),
      posed_sym,
  )
  return posed, plans, report


# ------------------------------------------------------------------------------
# Figure pose: closed-loop 3-D rotation to the stick figure's own angles.
# ------------------------------------------------------------------------------

# Requested angles are corrected by this fraction of the measured error.
_FIGURE_POSE_GAIN = 0.8
# Sticks closer than this to their target direction are not corrected.
_FIGURE_POSE_TOLERANCE_DEG = 20.0
# Azimuth is ill-defined for near-vertical sticks; only elevation is fed back.
_FIGURE_POSE_VERTICAL_DEG = 75.0
# figure_pose(narrow=True): in-plane turns (degrees) tried for the
# `simple_fold_narrow` applied to each thin limb on the BASE, before the
# rotation. `angle` is the zig-zag's turn (160 reverses a limb, 90 turns it
# ~86 deg), so only small turns survive the rotation; -20 won on every
# moose/scorpion limb, the crab's right legs needed +20 (narrow investigation).
_FIGURE_POSE_NARROW_ANGLES = (-20.0, 20.0, -8.0, 8.0)
# A narrowing is kept only if no stick ends more than this off its target
# (unless it already was, within _FIGURE_POSE_NARROW_SLACK_DEG), the
# direction score drops by at most _FIGURE_POSE_NARROW_DIRECTION_SLACK, the
# folder's vertex error stays <= 1% and the width score rises.
_FIGURE_POSE_STRAY_DEG = 45.0
_FIGURE_POSE_NARROW_SLACK_DEG = 5.0
_FIGURE_POSE_NARROW_DIRECTION_SLACK = 0.02
# Kept narrowings are ranked by fidelity + this weight * width score.
_FIGURE_POSE_NARROW_WIDTH_WEIGHT = 5.0
# Side fix (`fidelity.side_crossings`): flaps folded across (or onto) the
# body's mid-plane are re-targeted with an outward lateral bias (the unit
# target direction plus this much of the lateral axis on the flap's own side,
# renormalized: 0.4 ~ 20 deg, 0.8 ~ 35 deg for a flap pointing down), applied
# cumulatively, together with their lateral ancestors. A biased pose is kept
# when it has fewer crossing sticks and its direction loss rises by at most
# _FIGURE_POSE_SIDE_LOSS_SLACK_DEG.
_FIGURE_POSE_SIDE_BIASES = (0.4, 0.8, 1.2)
_FIGURE_POSE_SIDE_LOSS_SLACK_DEG = 8.0


def realized_directions(
    state: shaping.ShapingState,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
  """Folds `state` and measures every stick's direction in the figure frame.

  See `shaping.measured_directions`: measured in the frame `rotation_3d`
  actuates in (root faces -> unshaped base), not through the whole-figure
  similarity fit of `fidelity.stick_directions`, whose offsets, tilt and
  mirror made the feedback below chase errors that were not there.

  Args:
    state: The shaping state.

  Returns:
    (fidelity report, per-stick `{target: (az, el), realized: (az, el) |
    None, error: degrees | None}`).
  """
  return shaping.measured_directions(state)


def _direction_loss(report: Mapping[str, Any], tree) -> float:
  """Length-weighted mean direction error (degrees) of the located sticks."""
  errors = report.get('direction_error_deg', {})
  total = sum(tree[name].length for name in errors)
  if not total:
    return 180.0
  return sum(tree[name].length * err for name, err in errors.items()) / total


def _num_crossing(report: Mapping[str, Any] | None) -> int:
  """Number of sticks `fidelity.side_crossings` flags in a report."""
  if not report:
    return 0
  return len(fidelity.crossing_sticks(report.get('side_crossings') or []))


def lateral_bias(
    azimuth: float, elevation: float, side: float, amount: float
) -> tuple[float, float]:
  """Turns an (az, el) direction toward the lateral axis on `side`.

  Args:
    azimuth: Azimuth (degrees; figure frame, +90 = left).
    elevation: Elevation (degrees).
    side: +1 to push toward the left (+y), -1 toward the right.
    amount: Lateral component added to the unit direction before renormalizing.

  Returns:
    The biased (azimuth, elevation), rounded to 0.1 degree.
  """
  az, el = math.radians(azimuth), math.radians(elevation)
  vector = [
      math.cos(el) * math.cos(az),
      math.cos(el) * math.sin(az) + math.copysign(amount, side),
      math.sin(el),
  ]
  norm = math.sqrt(sum(v * v for v in vector))
  vector = [v / norm for v in vector]
  return (
      round(math.degrees(math.atan2(vector[1], vector[0])), 1),
      round(math.degrees(math.asin(max(-1.0, min(1.0, vector[2])))), 1),
  )


def side_fix_targets(
    tree,
    crossings: Sequence[Mapping[str, Any]],
    requested: Mapping[str, Any],
) -> dict[str, float]:
  """Sticks to push outward for `fidelity.side_crossings` entries.

  Every crossing stick and its ancestors up to (excluding) the first stick
  without a mirror partner (the body axis) get the side (+1 left, -1 right)
  the stick figure puts the crossing point on: a foot that crosses is fixed
  by splaying its leg too.

  Args:
    tree: The stick figure.
    crossings: `fidelity.side_crossings` entries.
    requested: The posed elements (only these are returned).

  Returns:
    Stick name -> side.
  """
  parents = _parents(tree)
  targets: dict[str, float] = {}
  for crossing in crossings:
    side = 1.0 if crossing['design_side'] == 'left' else -1.0
    name = crossing['stick']
    while name is not None and name != tree.root:
      if name in requested:
        targets.setdefault(name, side)
      parent = parents.get(name)
      if parent is None or _symmetric_partner_name(tree, parent) is None:
        break
      name = parent
  return targets


def _skipped_elements(message: str, reason: str = 'flat-fold') -> list[str]:
  """Names in a rotation_3d `left out: a (why); b (why).` whose why matches."""
  if 'left out:' not in message:
    return []
  tail = message.split('left out:', 1)[1]
  names = []
  depth = 0
  current = why = ''
  for char in tail:
    if char == '(':
      depth += 1
    elif char == ')':
      depth -= 1
    elif depth == 0 and char in ';.':
      if current.strip() and reason in why:
        names.append(current.strip())
      current = why = ''
      if char == '.':
        break
      continue
    if depth == 0 and char not in '()':
      current += char
    else:
      why += char
  return names


def _skip_nudges(azimuth: float, elevation: float) -> list[tuple[float, float]]:
  """Nudged (az, el) retries for an element rotation_3d left out.

  Azimuth moves up to 22 degrees toward (or 13 away from) the lateral axis
  (+-90; legs splay sideways), elevation is kept, lowered or set to a few flat
  values (crab walk_L3_merus: (112, 34) is left out; (90, 20) and (125, -10)
  rotate it, direction 0.92 -> 1.00). The azimuth-only nudges come first: on
  a planar figure (dragonfly hind wings, el 0) only the azimuth unblocks the
  fold, and every retry is a full rotation_3d.

  Args:
    azimuth: Requested azimuth (degrees).
    elevation: Requested elevation (degrees).

  Returns:
    Candidate (azimuth, elevation) pairs, most similar first; the request
    itself is not included.
  """
  lateral = math.copysign(90.0, azimuth) if azimuth else 0.0
  step = max(-22.0, min(22.0, lateral - azimuth))
  azimuths = [azimuth, azimuth + step, azimuth - math.copysign(13.0, step or 1)]
  elevations = [elevation, elevation - 15.0, 20.0, 10.0, 0.0, -10.0]
  candidates = []
  for el in elevations:
    for az in azimuths:
      candidates.append(
          (round(_wrap_degrees(az), 1), round(max(-89.0, min(89.0, el)), 1))
      )
  requested = (
      round(_wrap_degrees(azimuth), 1),
      round(max(-89.0, min(89.0, elevation)), 1),
  )
  return [c for c in dict.fromkeys(candidates) if c != requested]


def _is_mirrored_request(
    first: Sequence[float], second: Sequence[float], tolerance: float = 1.0
) -> bool:
  """True when two (az, el) requests mirror each other across the body axis."""
  return (
      abs(_wrap_degrees(first[0] + second[0])) <= tolerance
      and abs(first[1] - second[1]) <= tolerance
  )


def _rotates_alone(
    base: shaping.ShapingState, angles: Mapping[str, Sequence[float]]
) -> bool:
  """Whether rotation_3d applies every element in `angles` on the bare base.

  A cheap screen for skip nudges: an element whose fold breaks flat-foldability
  alone is left out of the full figure call too (which adds every other
  element's creases), so a nudge that fails here is not worth a full call.

  Args:
    base: The reset shaping state.
    angles: `{element: (az, el)}` for one element or a mirrored pair.

  Returns:
    True when the call succeeds and leaves none of them out.
  """
  rotations = [
      {'element_name': name, 'az': az, 'el': el}
      for name, (az, el) in angles.items()
  ]
  _, result = shaping.apply_tool(base, 'rotation_3d', {'rotations': rotations})
  if not result.ok:
    return False
  left_out = set(_skipped_elements(result.message, reason=''))
  return not left_out & set(angles)


def _narrow_candidates(state: shaping.ShapingState) -> list[str]:
  """Thin limb roots: flap sticks of width < 1 under a river or wider stick."""
  tree = state.tree
  parents = _parents(tree)
  rivers = {river.name for river in state.shaping_elements[1]}
  names = []
  for name in fidelity.topology(tree).order:
    if name == tree.root or name in rivers or tree[name].width >= 1.0:
      continue
    parent = parents.get(name)
    if parent is None or parent in rivers or tree[parent].width >= 1.0:
      names.append(name)
  return names


def _figure_pose_narrow(
    base: shaping.ShapingState,
    posed_state: shaping.ShapingState,
    requested: Mapping[str, Sequence[float]],
    report: Mapping[str, Any],
    *,
    elements: Sequence[str] | None,
    angles: Sequence[float],
    keep_sides: bool = True,
) -> tuple[Any, list[tuple[str, float | str]], list[dict[str, Any]]]:
  """Greedy narrow-then-rotate on top of a figure_pose result.

  For each thin limb root, every angle in `angles` is tried as
  `simple_fold_narrow(limb, angle)` on the base (after the narrowings kept
  so far), followed by the same `rotation_3d` call as the posed state
  (`allow_narrowed=True`); the best acceptable candidate (see the
  `_FIGURE_POSE_NARROW_*` constants) is kept. Limbs left wide are then tried
  once more after the rotation (`simple_fold_narrow` with its clip fallback).

  Args:
    base: The reset state.
    posed_state: figure_pose's posed (not narrowed) state.
    requested: figure_pose's final requested angles.
    report: Fidelity report of the posed (not narrowed) state.
    elements: Limbs to narrow (default `_narrow_candidates`).
    angles: Narrowing turns to try per limb.
    keep_sides: Reject narrowings that add `fidelity.side_crossings` sticks.

  Returns:
    (best (state, report, directions) or None if nothing was kept, the kept
    `(element, angle)` steps (angle 'post' for pass 2), a per-candidate log).
  """
  rotations = [
      {'element_name': name, 'az': az, 'el': el}
      for name, (az, el) in requested.items()
  ]
  base_errors = report.get('direction_error_deg', {})
  min_direction = (
      report['components']['direction'] - _FIGURE_POSE_NARROW_DIRECTION_SLACK
  )

  def score(rep):
    return rep['fidelity'] + _FIGURE_POSE_NARROW_WIDTH_WEIGHT * (
        rep['components']['width']
    )

  def acceptable(rep, current):
    if rep.get('fold_vertex_error_percent', 0.0) > 1.0:
      return False
    if rep['components']['width'] <= current['components']['width'] + 1e-3:
      return False
    if rep['components']['direction'] < min_direction:
      return False
    if keep_sides and _num_crossing(rep) > _num_crossing(current):
      return False
    return all(
        err <= _FIGURE_POSE_STRAY_DEG
        or err <= base_errors.get(name, 0.0) + _FIGURE_POSE_NARROW_SLACK_DEG
        for name, err in rep.get('direction_error_deg', {}).items()
    )

  prefix = base
  steps: list[tuple[str, float | str]] = []
  best = None
  current = report
  log = []
  for name in elements if elements is not None else _narrow_candidates(base):
    choice = None
    for angle in angles:
      narrowed, result = shaping.apply_tool(
          prefix,
          'simple_fold_narrow',
          {'element_name': name, 'angle': angle, 'fallback_clip_narrow': False},
      )
      if not result.ok:
        log.append(
            {'element': name, 'angle': angle, 'error': result.message[:200]}
        )
        continue
      posed, result = shaping.apply_tool(
          narrowed,
          'rotation_3d',
          {'rotations': rotations, 'allow_narrowed': True},
      )
      if not result.ok:
        log.append(
            {'element': name, 'angle': angle, 'error': result.message[:200]}
        )
        continue
      try:
        rep, directions = realized_directions(posed)
      except Exception as error:  # pylint: disable=broad-exception-caught
        log.append({'element': name, 'angle': angle, 'error': str(error)[:200]})
        continue
      ok = acceptable(rep, current)
      log.append({
          'element': name,
          'angle': angle,
          'kept': ok,
          'fidelity': rep['fidelity'],
          'width': rep['components']['width'],
          'direction': rep['components']['direction'],
      })
      if ok and (choice is None or score(rep) > score(choice[2])):
        choice = (angle, narrowed, rep, posed, directions)
    if choice is not None:
      angle, prefix, current, posed, directions = choice
      steps.append((name, angle))
      best = (posed, current, directions)
  # Pass 2: thin flaps the base pass could not narrow (single-segment flaps
  # such as eyestalks fail the zig-zag) are narrowed AFTER the rotation with
  # the clip-fold fallback (crab eyestalks: width 1.26 -> 0.31, direction
  # kept). Proximal limb segments fail flat-foldability here, so this pass
  # only adds what pass 1 left.
  done = {name for name, _ in steps}
  final = best[0] if best is not None else posed_state
  for name in elements if elements is not None else _narrow_candidates(base):
    if name in done:
      continue
    narrowed, result = shaping.apply_tool(
        final, 'simple_fold_narrow', {'element_name': name, 'angle': 45.0}
    )
    if not result.ok:
      log.append({'element': name, 'pass': 2, 'error': result.message[:200]})
      continue
    try:
      rep, directions = realized_directions(narrowed)
    except Exception as error:  # pylint: disable=broad-exception-caught
      log.append({'element': name, 'pass': 2, 'error': str(error)[:200]})
      continue
    ok = acceptable(rep, current)
    log.append({
        'element': name,
        'pass': 2,
        'kept': ok,
        'fidelity': rep['fidelity'],
        'width': rep['components']['width'],
        'direction': rep['components']['direction'],
    })
    if ok and score(rep) > score(current):
      final, current = narrowed, rep
      steps.append((name, 'post'))
      best = (narrowed, rep, directions)
  return best, steps, log


def figure_pose(
    state: shaping.ShapingState,
    *,
    iterations: int = 2,
    elements: Sequence[str] | None = None,
    narrow: bool = False,
    narrow_elements: Sequence[str] | None = None,
    narrow_angles: Sequence[float] = _FIGURE_POSE_NARROW_ANGLES,
    from_current: bool = False,
    bend_left_out: bool = False,
    fix_crossings: bool = True,
) -> tuple[shaping.ShapingState, dict[str, Any]]:
  """Poses every element at its stick-figure angles with measured feedback.

  One `rotation_3d` call from the base pattern with each element's own
  azimuth/elevation (the designer's root is the reference frame, see
  `ToolConfig.use_designer_root_3d`), then up to `iterations` corrections:
  the folded model is measured in the figure frame and every stick more
  than `_FIGURE_POSE_TOLERANCE_DEG` off gets its requested angles moved by
  `_FIGURE_POSE_GAIN` times the error; the whole call is redone from the base
  (rotations compose from the base posture, not across calls) and kept only
  if the length-weighted direction error drops.

  The angles are absolute directions in the stick figure's frame (x forward,
  y left, z up), never relative turns: this is the fix for limbs posed with
  in-plane turn angles whose sign and plane depend on each flap's layer.

  With `narrow`, thin limbs are then narrowed without losing their
  direction: narrowing AFTER the rotation fails flat-foldability on rotated
  proximal segments, so each limb is narrowed on the base with a small turn
  (`simple_fold_narrow`, `narrow_angles`) and the same rotation is redone on
  top (`rotation_3d(allow_narrowed=True)`); see `_figure_pose_narrow`. Costs
  about len(narrow_angles) folds per thin limb.

  Args:
    state: Shaping state; posing starts from `state.reset()` (or from `state`
      itself with `from_current`).
    iterations: Feedback rounds after the first call.
    elements: Elements to rotate (default: every non-root stick).
    narrow: Narrow the thin limbs after posing (narrow-then-rotate).
    narrow_elements: Limbs to narrow (default: thin flap sticks whose parent is
      a river or a wider stick).
    narrow_angles: Narrowing turns tried per limb, best kept.
    from_current: Pose on top of `state` instead of `state.reset()`, keeping
      flat 2-D shaping applied before (e.g. `fan_out_fold(open_percent=1.0)` on
      wings). Without it the feedback loop discards that shaping; posing fanned
      wings with a single uncorrected `rotation_3d` left them 44 deg off and
      stacked on Pegasus (`peg_E_flatfan_first`).
    bend_left_out: After posing, bend every element `rotation_3d` left out (and
      that is still off target) with `bend_to_direction`, keeping a bend only if
      the direction loss drops. Use with `from_current` after flat
      fans: on Pegasus the fanned wings went from 44.8 deg off and stacked to 29
        deg and spread.
    fix_crossings: Keep every flap on its own side of the body's mid-plane
      (`fidelity.side_crossings`, measured in the root frame): flaps that cross
      (or meet at) the mid-plane are re-targeted with an outward lateral bias
      (`_FIGURE_POSE_SIDE_BIASES`, see `side_fix_targets`), and no feedback,
      nudge, bend or narrowing step is kept if it adds crossing sticks. Eagle
      v3: both talons folded onto the mid-plane (tips at y = -0.03 / 0.02 for
        design +-1.55) and crossed under the body.

  Returns:
    The posed state and a report (`loss` trajectory in degrees, the final
    requested angles, per-stick realized directions and fidelity,
    `side_crossings` and the `side_fix` log; with `narrow` also `narrowed`
    steps, `fidelity_before_narrow`, `narrow_log`).
  """
  start = time.monotonic()
  base = state if from_current else state.reset()
  tree = base.tree
  if elements is None:
    elements = [
        name for name in fidelity.topology(tree).order if name != tree.root
    ]
  requested = {
      name: [float(tree[name].azimuth_angle), float(tree[name].elevation_angle)]
      for name in elements
  }

  def attempt(angles):
    rotations = [
        {'element_name': name, 'az': az, 'el': el}
        for name, (az, el) in angles.items()
    ]
    posed, result = shaping.apply_tool(
        base, 'rotation_3d', {'rotations': rotations}
    )
    if not result.ok:
      return None, result.message, None, None
    try:
      report, directions = realized_directions(posed)
    except Exception as error:  # pylint: disable=broad-exception-caught
      return None, f'fold/measure failed: {error}', None, None
    return posed, result.message, report, directions

  best_state, message, best_report, best_directions = attempt(requested)
  if best_state is None:
    raise ValueError(f'figure_pose: rotation_3d failed: {message}')
  best_loss = _direction_loss(best_report, tree)
  best_requested = {k: list(v) for k, v in requested.items()}
  trajectory: list[float | None] = [round(best_loss, 1)]
  messages = [message[:400]]
  # Elements the library leaves out (their fold breaks flat-foldability) stay
  # flat; feeding their error back only drives the request further off (crab
  # walk_L3_merus: (112, 34) -> (-68, 86), still left out). Retry each with
  # nudged angles instead, and keep it out of the feedback below if no nudge
  # is accepted. Every retry is a full rotation_3d from the base (~11 s on a
  # grid-92 dragonfly), so a mirrored pair is nudged together (mirrored
  # azimuth), each nudge is first screened with a rotation_3d of the group
  # alone on the base (~0.5 s; an element left out alone is left out of the
  # full call too), and the first accepted nudge that lowers the loss is
  # kept (dragonfly hw_r_o: 45 retries, 531 s -> 2 full retries).
  skipped = set(_skipped_elements(message)) & set(best_requested)
  retried: set[str] = set()
  for name in sorted(skipped):
    if name in retried:
      continue
    partner = _symmetric_partner_name(tree, name)
    group = [name]
    if (
        partner is not None
        and partner in skipped
        and partner not in retried
        and partner != name
        and _is_mirrored_request(requested[name], requested[partner])
    ):
      group.append(partner)
    retried.update(group)
    for az, el in _skip_nudges(*requested[name]):
      proposal = {k: list(v) for k, v in best_requested.items()}
      proposal[name] = [az, el]
      for other in group[1:]:
        proposal[other] = [round(_wrap_degrees(-az), 1), el]
      if not _rotates_alone(
          base, {member: proposal[member] for member in group}
      ):
        continue
      posed, msg, report, directions = attempt(proposal)
      if posed is None:
        continue
      still_out = set(_skipped_elements(msg))
      if any(member in still_out for member in group):
        continue
      loss = _direction_loss(report, tree)
      if loss < best_loss and not (
          fix_crossings and _num_crossing(report) > _num_crossing(best_report)
      ):
        best_state, best_report, best_directions = posed, report, directions
        best_loss, best_requested = loss, proposal
        skipped.difference_update(group)
        trajectory.append(round(loss, 1))
        messages.append(msg[:400])
        break
  for _ in range(iterations):
    proposal = {k: list(v) for k, v in best_requested.items()}
    changed = False
    for name, info in best_directions.items():
      if name not in proposal or name in skipped or info['realized'] is None:
        continue
      if tree[name].is_river:
        # A river's realized direction is measured between joints located
        # from the flaps around it, and turning a river moves every flap that
        # hangs from it: feeding its error back moved the triceratops neck
        # request to (-143.8, 10), 166 degrees off, where one plain
        # rotation_3d was right.
        continue
      if (info['error'] or 0.0) <= _FIGURE_POSE_TOLERANCE_DEG:
        continue
      (t_az, t_el), (r_az, r_el) = info['target'], info['realized']
      az, el = proposal[name]
      el = max(-89.0, min(89.0, el + _FIGURE_POSE_GAIN * (t_el - r_el)))
      if abs(t_el) < _FIGURE_POSE_VERTICAL_DEG:
        az = _wrap_degrees(az + _FIGURE_POSE_GAIN * _wrap_degrees(t_az - r_az))
      proposal[name] = [round(az, 1), round(el, 1)]
      changed = True
    if not changed:
      break
    posed, message, report, directions = attempt(proposal)
    messages.append(message[:400])
    if posed is None:
      trajectory.append(None)
      break
    loss = _direction_loss(report, tree)
    trajectory.append(round(loss, 1))
    if loss >= best_loss or (
        fix_crossings and _num_crossing(report) > _num_crossing(best_report)
    ):
      break
    best_state, best_report, best_directions = posed, report, directions
    best_loss, best_requested = loss, proposal
  side_fix: list[dict[str, Any]] = []
  if fix_crossings and _num_crossing(best_report):
    # Flaps folded across (or onto) the mid-plane: push them (and their
    # lateral ancestors) outward and redo the call from the base.
    for amount in _FIGURE_POSE_SIDE_BIASES:
      crossings = best_report.get('side_crossings') or []
      targets = side_fix_targets(tree, crossings, best_requested)
      if not targets:
        break
      proposal = {k: list(v) for k, v in best_requested.items()}
      for name, side in targets.items():
        az, el = proposal[name]
        proposal[name] = list(lateral_bias(az, el, side, amount))
      posed, message, report, directions = attempt(proposal)
      entry: dict[str, Any] = {
          'bias': amount,
          'targets': {name: proposal[name] for name in targets},
      }
      if posed is None:
        entry['error'] = message[:200]
        side_fix.append(entry)
        continue
      loss = _direction_loss(report, tree)
      kept = _num_crossing(report) < _num_crossing(best_report) and (
          loss <= best_loss + _FIGURE_POSE_SIDE_LOSS_SLACK_DEG
      )
      entry.update(
          kept=kept,
          loss=round(loss, 1),
          crossing=fidelity.crossing_sticks(report.get('side_crossings') or []),
      )
      side_fix.append(entry)
      messages.append(message[:400])
      if kept:
        best_state, best_report, best_directions = posed, report, directions
        best_loss, best_requested = loss, proposal
        trajectory.append(round(loss, 1))
        if not _num_crossing(best_report):
          break
  bent: list[dict[str, Any]] = []
  if bend_left_out:
    # rotation_3d leaves flaps out silently (its call still succeeds), e.g.
    # wings fanned flat before posing: they stay folded along the body and
    # read as crossing/stacked (Pegasus: realized az 25 vs target 92). Bend
    # each left-out flap toward its target by measurement instead.
    for name in sorted(skipped):
      info = best_directions.get(name) or {}
      if info.get('realized') is None or (
          (info.get('error') or 0.0) <= _FIGURE_POSE_TOLERANCE_DEG
      ):
        continue
      t_az, t_el = info['target']
      candidate, result = shaping.apply_tool(
          best_state,
          'bend_to_direction',
          {'element_name': name, 'az': t_az, 'el': t_el},
      )
      if not result.ok:
        bent.append({'element': name, 'error': result.message[:200]})
        continue
      try:
        report, directions = realized_directions(candidate)
      except Exception as error:  # pylint: disable=broad-exception-caught
        bent.append({'element': name, 'error': str(error)[:200]})
        continue
      loss = _direction_loss(report, tree)
      kept = loss < best_loss and not (
          fix_crossings and _num_crossing(report) > _num_crossing(best_report)
      )
      bent.append({
          'element': name,
          'kept': kept,
          'error_before': info.get('error'),
          'error_after': (directions.get(name) or {}).get('error'),
      })
      if kept:
        best_state, best_report, best_directions = candidate, report, directions
        best_loss = loss
        trajectory.append(round(loss, 1))
  narrow_summary: dict[str, Any] = {}
  if narrow:
    narrowed, steps, log = _figure_pose_narrow(
        base,
        best_state,
        best_requested,
        best_report,
        elements=narrow_elements,
        angles=narrow_angles,
        keep_sides=fix_crossings,
    )
    narrow_summary = {
        'narrowed': steps,
        'fidelity_before_narrow': best_report['fidelity'],
        'narrow_log': log,
    }
    if narrowed is not None:
      best_state, best_report, best_directions = narrowed
  summary = {
      'loss_deg': trajectory,
      'fidelity': best_report['fidelity'],
      'components': best_report['components'],
      'pose_gate': best_report.get('pose_gate', {}),
      'side_crossings': best_report.get('side_crossings', []),
      'side_fix': side_fix,
      'requested': best_requested,
      'directions': best_directions,
      'messages': messages,
      'bent': bent,
      **narrow_summary,
      'seconds': round(time.monotonic() - start, 1),
  }
  return best_state, summary


# ------------------------------------------------------------------------------
# Search-based pose: coordinate search over axis chains, L/R pairs, and flaps.
# ------------------------------------------------------------------------------

_ORIG_PREPARE_GRAPH_DATA = shaping.geometric_simple_fold_lib.prepare_graph_data
_GRAPH_DATA_CACHE: list[tuple[int, Any, Any]] = []
_MAX_GRAPH_DATA_CACHE = 8


def _cached_prepare_graph_data(crease_pattern: Any) -> Any:
  """Caches `prepare_graph_data` by pattern identity across candidates/groups."""
  pid = id(crease_pattern)
  for cached_id, cached_pattern, graph_data in _GRAPH_DATA_CACHE:
    if cached_id == pid and cached_pattern is crease_pattern:
      return graph_data
  graph_data = _ORIG_PREPARE_GRAPH_DATA(crease_pattern)
  _GRAPH_DATA_CACHE.append((pid, crease_pattern, graph_data))
  if len(_GRAPH_DATA_CACHE) > _MAX_GRAPH_DATA_CACHE:
    _GRAPH_DATA_CACHE.pop(0)
  return graph_data


_ORIG_MAKE_FLAT_FOLDABLE = fan_out_shaping._make_flat_foldable  # pylint: disable=protected-access


def _fast_make_flat_foldable(
    cp: Any,
    candidates: Any,
    *,
    max_candidates: int | None = 4,
    max_crease_modifications: int | None = 4,
) -> Any:
  """Caps `fan_out_shaping._make_flat_foldable` breadth for search speed."""
  return _ORIG_MAKE_FLAT_FOLDABLE(
      cp,
      candidates,
      max_candidates=4 if max_candidates is None else min(max_candidates, 4),
      max_crease_modifications=(
          4
          if max_crease_modifications is None
          else min(max_crease_modifications, 4)
      ),
  )


@contextlib.contextmanager
def _fast_search_patches() -> Iterator[None]:
  """Temporarily enables graph caching and bounded fan-out during search."""
  orig_prep = shaping.geometric_simple_fold_lib.prepare_graph_data
  orig_make_ff = fan_out_shaping._make_flat_foldable  # pylint: disable=protected-access
  shaping.geometric_simple_fold_lib.prepare_graph_data = (
      _cached_prepare_graph_data
  )
  fan_out_shaping._make_flat_foldable = _fast_make_flat_foldable  # pylint: disable=protected-access
  try:
    yield
  finally:
    shaping.geometric_simple_fold_lib.prepare_graph_data = orig_prep
    fan_out_shaping._make_flat_foldable = orig_make_ff  # pylint: disable=protected-access

# A step lists alternative tool calls; the first that succeeds is kept.
Step = tuple[dict[str, Any], ...]


@dataclasses.dataclass(frozen=True)
class Candidate:
  """One way to shape a group of elements.

  Attributes:
    label: Short name for reports, e.g. `rot3d` or `narrow+knee`.
    steps: Steps applied in order; each lists alternative calls.
    required: The first `required` steps must succeed (the rest are optional
      refinements); at least one step must succeed in any case.
  """

  label: str
  steps: tuple[Step, ...]
  required: int = 1


@dataclasses.dataclass(frozen=True)
class Group:
  """Elements posed together: an axis chain, a left/right pair, one flap.

  Attributes:
    name: Group name (merged chain name, `L|R` for pairs).
    kind: `chain` (axis chain), `pair` or `single`.
    elements: Unmerged element names in the group.
    candidates: Alternatives, tried in order.
  """

  name: str
  kind: str
  elements: tuple[str, ...]
  candidates: tuple[Candidate, ...]


_SHAPEABLE_ROLES = (
    ROLE_THIN_LIMB,
    ROLE_LIMB,
    ROLE_SEGMENT,
    ROLE_WIDE_TIP,
    ROLE_TIP_DOWN,
)


def _rotation_step(tree, names: Sequence[str]) -> Step:
  return (_rotation_call(tree, names),)


def _plan_step(plan: ElementPlan) -> Step:
  return tuple(plan.candidates)


def _mirrored_fan_steps(
    names_and_signs: Sequence[tuple[str, float]],
) -> tuple[Step, ...]:
  steps = []
  for name, sign in names_and_signs:
    steps.append((
        {
            'name': 'fan_out_fold_heuristic',
            'arguments': {'element_name': name, 'move_delta': sign},
        },
        {
            'name': 'fan_out_fold',
            'arguments': {
                'element_name': name,
                'orientation': 'horizontal',
                'move_delta': -sign,
            },
        },
    ))
  return tuple(steps)


def _mirrored_squash_steps(
    names_and_signs: Sequence[tuple[str, float]],
) -> tuple[Step, ...]:
  steps = []
  for name, _ in names_and_signs:
    steps.append((
        {
            'name': 'squash_fold',
            'arguments': {'element_name': name, 'ridge_index': 0},
        },
    ))
  return tuple(steps)


def _narrow_alternatives(
    plan: ElementPlan, fold_percent: float | None = None
) -> Step:
  calls = []
  for call in plan.candidates:
    if call['name'] == 'simple_fold' and call['arguments'].get('narrow'):
      arguments = dict(call['arguments'])
      if fold_percent is not None:
        arguments['fold_percent'] = fold_percent
      calls.append({'name': 'simple_fold', 'arguments': arguments})
  return tuple(calls)


def _thin_limbs_3d_step(
    tree,
    plans_by_name: Mapping[str, ElementPlan],
    sides: Sequence[Sequence[str]],
) -> Step:
  """Builds fast `thin_limbs_3d` calls with explicit narrow/bend/knee args."""
  first_plans = [plans_by_name[names[0]] for names in sides if names]
  narrow_options = [_narrow_alternatives(p) for p in first_plans]
  if not narrow_options or not all(narrow_options):
    return ()
  limbs = []
  for side_idx, names in enumerate(sides):
    if not names:
      continue
    proximal = names[0]
    distal = names[1] if len(names) > 1 and tree[names[1]].length >= 2 else None
    narrow_angle = float(narrow_options[side_idx][0]['arguments']['angle'])
    fold_order = 'MV' if tree[proximal].elevation_angle >= 0 else 'VM'
    spec: dict[str, Any] = {
        'proximal': proximal,
        'distal': distal,
        'narrow_angle': narrow_angle,
        'bend': (
            {
                'angle': 180.0,
                'depth': 0.5,
                'fold_percent': 0.5,
                'fold_order': fold_order,
            }
            if distal is None
            else None
        ),
        'knee': (
            {
                'angle': 180.0,
                'depth': 0.25,
                'fold_percent': 0.5,
                'fold_order': 'VM',
            }
            if distal is not None
            else None
        ),
    }
    limbs.append(spec)
  return ({'name': 'thin_limbs_3d', 'arguments': {'limbs': limbs}},)


def _chain_candidates(
    tree,
    plans_by_name: Mapping[str, ElementPlan],
    sides: Sequence[Sequence[str]],
    *,
    include_thin_3d: bool = False,
) -> list[Candidate]:
  """Candidates for one chain (`sides` = [names]) or a pair ([L, R])."""
  shapeable = [
      [n for n in names if plans_by_name[n].role in _SHAPEABLE_ROLES]
      for names in sides
  ]
  if not any(shapeable):
    return []
  all_names = [n for names in shapeable for n in names]
  wide_tips = [
      (names[-1], 1.0)
      for names in shapeable
      if names and plans_by_name[names[-1]].role == ROLE_WIDE_TIP
  ]
  candidates = []
  # 3-D rotation of the whole chain(s) at the stick-figure angles, then fan
  # out or squash wide tips (pincers, palms).
  rotated = [n for n in all_names if plans_by_name[n].role != ROLE_WIDE_TIP]
  if not rotated and any(len(s) > 1 for s in sides):
    # Chain or pair whose non-tip segments had length < 2 (ROLE_SKIP).
    all_names = [n for names in sides for n in names]
    rotated = [n for n in all_names if plans_by_name[n].role != ROLE_WIDE_TIP]
  if rotated:
    candidates.append(
        Candidate(
            'rot3d',
            (_rotation_step(tree, rotated),) + _mirrored_fan_steps(wide_tips),
        )
    )
    if wide_tips:
      candidates.append(
          Candidate(
              'rot3d+squash',
              (_rotation_step(tree, rotated),)
              + _mirrored_squash_steps(wide_tips),
              required=1 + len(wide_tips),
          )
      )
  if all_names and rotated != all_names:
    candidates.append(
        Candidate('rot3d_all', (_rotation_step(tree, all_names),))
    )
  if not rotated and wide_tips:
    candidates.append(
        Candidate(
            'fan_out', _mirrored_fan_steps(wide_tips), required=len(wide_tips)
        )
    )
    candidates.append(
        Candidate(
            'squash',
            _mirrored_squash_steps(wide_tips),
            required=len(wide_tips),
        )
    )
  # Thin limbs: narrow the first stick, plus thin_limbs_3d when applicable.
  first_plans = [plans_by_name[names[0]] for names in shapeable if names]
  is_thin_group = bool(
      first_plans and all(p.role == ROLE_THIN_LIMB for p in first_plans)
  )
  if is_thin_group:
    narrow_steps = tuple(_narrow_alternatives(p) for p in first_plans)
    if all(narrow_steps):
      candidates.append(
          Candidate('narrow', narrow_steps, required=len(narrow_steps))
      )
      if include_thin_3d:
        thin_3d = _thin_limbs_3d_step(tree, plans_by_name, sides)
        if thin_3d:
          candidates.append(Candidate('thin_limbs_3d', (thin_3d,), required=1))
  # The legacy recipe's role-based calls, element by element (skipped on pure
  # thin-limb groups where `narrow` already covers the same simple_fold call).
  has_non_thin_role = any(
      plans_by_name[name].role not in (ROLE_THIN_LIMB, ROLE_TIP_DOWN)
      for names in shapeable
      for name in names
  )
  if not is_thin_group or has_non_thin_role:
    legacy = []
    for role in _ROLE_ORDER:
      for names in shapeable:
        for name in names:
          if (
              plans_by_name[name].role == role
              and plans_by_name[name].candidates
          ):
            legacy.append(_plan_step(plans_by_name[name]))
    if legacy:
      candidates.append(Candidate('legacy', tuple(legacy), required=0))
  return candidates


def _symmetric_partner_name(tree, name: str) -> str | None:
  """Returns the mirror stick name from metadata or `_L`/`_R` naming."""
  mirror = tree[name].symmetric_stick_name
  if mirror:
    return mirror
  for left, right in (('_L', '_R'), ('_l', '_r')):
    if left in name:
      candidate = name.replace(left, right, 1)
    elif right in name:
      candidate = name.replace(right, left, 1)
    else:
      continue
    if candidate in tree and (
        tree[candidate].length == tree[name].length
        and tree[candidate].width == tree[name].width
    ):
      return candidate
  return None


def _body_candidates(
    tree,
    rivers: Sequence[str],
) -> list[Candidate]:
  """Candidates for body/river volume (`rotation_3d` and `shape_angle_width`)."""
  non_root = [r for r in rivers if r != tree.root]
  if not non_root:
    return []
  candidates = []
  target = non_root[0]
  base_el = tree[target].elevation_angle
  # Incremental river-subset `rotation_3d` and `shape_angle_width(only_listed=True)`
  # on top of the posed state (fast, preserves narrowed limbs).
  for delta in (25.0, -35.0):
    el_arch = base_el + delta
    rotations = [
        {
            'element_name': r,
            'el': el_arch if r == target else tree[r].elevation_angle,
            'az': tree[r].azimuth_angle,
            'depth': 0.0,
        }
        for r in rivers
    ]
    candidates.append(
        Candidate(
            f'rot3d_body_{el_arch:+.0f}',
            (
                (
                    {
                        'name': 'rotation_3d',
                        'arguments': {'rotations': rotations},
                    },
                ),
            ),
        )
    )
  for delta in (25.0, -25.0, -35.0):
    el_listed = base_el + delta
    updates = [
        {
            'element_name': r,
            'el': el_listed if r == target else tree[r].elevation_angle,
        }
        for r in rivers
    ]
    candidates.append(
        Candidate(
            f'saw_listed_{el_listed:+.0f}',
            (
                (
                    {
                        'name': 'shape_angle_width',
                        'arguments': {'updates': updates, 'only_listed': True},
                    },
                ),
            ),
        )
    )
  return candidates


def pose_groups(
    state: shaping.ShapingState,
    plans: Sequence[ElementPlan] | None = None,
) -> list[Group]:
  """Groups the plan into chains, left/right pairs, single flaps and body.

  Order: axis chains (tails, necks) first, then pairs of wide limbs (arms,
  antlers, claws), then thin pairs (legs), then single flaps, then body rivers.

  Args:
    state: Shaping state.
    plans: Output of `pose_plan` (computed if None).

  Returns:
    Groups with their candidates (groups without any are dropped).
  """
  if plans is None:
    plans = pose_plan(state)
  tree = state.tree
  plans_by_name = {p.name: p for p in plans}
  _, merged_names = tree.merge_linear_chains()
  chain_of = {}
  for key, names in merged_names.items():
    for name in names:
      chain_of[name] = key
  order = [s.name for s in traversal.breadth_first_traversal(tree)]
  rank = {name: index for index, name in enumerate(order)}
  chains: dict[str, list[str]] = {}
  rivers: list[str] = []
  for name in order:
    if plans_by_name[name].role in (ROLE_ROOT, ROLE_RIVER):
      rivers.append(name)
      continue
    chains.setdefault(chain_of.get(name, name), []).append(name)
  raw_groups: list[tuple[str, str, list[list[str]], tuple[str, ...]]] = []
  done = set()
  for key in sorted(chains, key=lambda k: rank[chains[k][0]]):
    if key in done:
      continue
    names = chains[key]
    mirror = _symmetric_partner_name(tree, names[0])
    partner = chain_of.get(mirror, mirror) if mirror else None
    if partner in chains and partner != key and partner not in done:
      done.update({key, partner})
      sides = [names, chains[partner]]
      kind = 'pair'
      group_name = f'{key}|{partner}'
    else:
      done.add(key)
      sides = [names]
      kind = 'chain' if len(names) > 1 else 'single'
      group_name = key
    elements = tuple(n for side in sides for n in side)
    raw_groups.append((group_name, kind, sides, elements))

  def raw_priority(
      item: tuple[str, str, list[list[str]], tuple[str, ...]],
  ) -> tuple[int, int]:
    _, kind, _, elements = item
    widths = [tree[n].width for n in elements]
    if kind == 'chain':
      level = 0
    elif kind == 'pair':
      level = 1 if max(widths) >= 1 else 2
    else:
      level = 3
    return level, min(rank[n] for n in elements)

  raw_groups.sort(key=raw_priority)
  thin_indices = []
  thin_with_distal_idx = -1
  for idx, (_, _, sides, _) in enumerate(raw_groups):
    shapeable = [
        [n for n in names if plans_by_name[n].role in _SHAPEABLE_ROLES]
        for names in sides
    ]
    first_plans = [plans_by_name[names[0]] for names in shapeable if names]
    if first_plans and all(p.role == ROLE_THIN_LIMB for p in first_plans):
      thin_indices.append(idx)
      if thin_with_distal_idx == -1 and any(
          len(names) > 1 and tree[names[1]].length >= 2 for names in sides
      ):
        thin_with_distal_idx = idx
  thin_3d_idx = (
      thin_with_distal_idx
      if thin_with_distal_idx != -1
      else (thin_indices[0] if len(thin_indices) == 1 else -1)
  )

  groups = []
  for idx, (group_name, kind, sides, elements) in enumerate(raw_groups):
    candidates = _chain_candidates(
        tree,
        plans_by_name,
        sides,
        include_thin_3d=(idx == thin_3d_idx),
    )
    if candidates:
      groups.append(Group(group_name, kind, elements, tuple(candidates)))
  body_candidates = _body_candidates(tree, rivers)
  if body_candidates:
    groups.append(Group('body', 'body', tuple(rivers), tuple(body_candidates)))
  return groups


def _call_cache_key(call: Mapping[str, Any]) -> str:
  return repr((call['name'], sorted(call.get('arguments', {}).items())))


def _apply_candidate(
    state: shaping.ShapingState,
    candidate: Candidate,
    call_cache: (
        dict[
            tuple[int, str],
            tuple[Any, shaping.ShapingState, shaping.ToolResult],
        ]
        | None
    ) = None,
) -> tuple[
    shaping.ShapingState, bool, list[dict[str, Any]], list[dict[str, Any]]
]:
  """Applies a candidate's steps; returns (state, ok, attempts, applied)."""
  attempts, applied = [], []
  current = state
  for index, step in enumerate(candidate.steps):
    chosen = None
    for call in step:
      key = (id(current), _call_cache_key(call))
      if call_cache is not None and key in call_cache:
        _, new_state, result = call_cache[key]
      else:
        tool_args = call['arguments']
        if call['name'] == 'thin_limbs_3d' and 'evaluate' not in tool_args:
          tool_args = dict(tool_args, evaluate=lambda _s, _p: {})
        new_state, result = shaping.apply_tool(current, call['name'], tool_args)
        if call_cache is not None:
          call_cache[key] = (current, new_state, result)
      attempts.append({
          'call': call,
          'ok': result.ok,
          'message': result.message[:160],
      })
      if result.ok:
        current, chosen = new_state, call
        break
    if chosen is not None:
      applied.append(chosen)
    elif index < candidate.required:
      return state, False, attempts, []
  return current, bool(applied), attempts, applied


def fidelity_evaluator(
    state: shaping.ShapingState,
) -> Callable[[shaping.ShapingState], tuple[float, dict[str, Any]]]:
  """Returns `evaluate(state) -> (score, report)` using the fidelity score.

  Patterns the deterministic folder cannot fold score -1.

  Args:
    state: Any state of the base (provides the stick figure and packing).
  """
  layout = fidelity.paper_layout(state.tree, state.packing)
  # Keyed by id() but holding the pattern: a freed candidate's id can be
  # reused by the next candidate, which must not get the old score.
  cache: dict[int, tuple[Any, tuple[float, dict[str, Any]]]] = {}

  def evaluate(
      candidate_state: shaping.ShapingState,
  ) -> tuple[float, dict[str, Any]]:
    pattern = candidate_state.pattern
    key = id(pattern)
    if key in cache and cache[key][0] is pattern:
      return cache[key][1]
    try:
      report = fidelity.score_pattern(
          candidate_state.tree,
          candidate_state.packing,
          candidate_state.pattern,
          layout=layout,
      )
    except (ValueError, AssertionError, KeyError, IndexError) as error:
      res = (-1.0, {'error': str(error)[:200]})
      cache[key] = (pattern, res)
      return res
    res = (float(report['fidelity']), report)
    cache[key] = (pattern, res)
    return res

  return evaluate


@dataclasses.dataclass(frozen=True)
class _Entry:
  score: float
  state: shaping.ShapingState
  choices: tuple[tuple[str, str], ...]


def search_pose(
    state: shaping.ShapingState,
    *,
    time_budget_s: float = 120.0,
    beam_width: int = 1,
    evaluate: (
        Callable[[shaping.ShapingState], tuple[float, dict[str, Any]]] | None
    ) = None,
) -> tuple[
    shaping.ShapingState,
    list[ElementPlan],
    list[dict[str, Any]],
    dict[str, Any],
]:
  """Poses the base by searching tool alternatives per group by fidelity."""
  with _fast_search_patches():
    return _search_pose_impl(
        state,
        time_budget_s=time_budget_s,
        beam_width=beam_width,
        evaluate=evaluate,
    )


def _search_pose_impl(
    state: shaping.ShapingState,
    *,
    time_budget_s: float = 120.0,
    beam_width: int = 1,
    evaluate: (
        Callable[[shaping.ShapingState], tuple[float, dict[str, Any]]] | None
    ) = None,
) -> tuple[
    shaping.ShapingState,
    list[ElementPlan],
    list[dict[str, Any]],
    dict[str, Any],
]:
  """Internal implementation of `search_pose`."""
  start = time.monotonic()
  base = state.reset()
  plans = pose_plan(base)
  groups = pose_groups(base, plans)
  if evaluate is None:
    evaluate = fidelity_evaluator(base)
  base_score, _ = evaluate(base)
  beam = [_Entry(base_score, base, ())]
  summaries = []
  evaluations = 1
  planned_names = {plan.name for plan in plans if plan.candidates}
  element_attempts: dict[str, list[dict[str, Any]]] = {}
  element_applied: dict[str, dict[str, Any]] = {}
  budget_exhausted = False
  call_cache: dict[
      tuple[int, str], tuple[Any, shaping.ShapingState, shaping.ToolResult]
  ] = {}
  for group in groups:
    expansions = [
        _Entry(e.score, e.state, e.choices + ((group.name, 'skip'),))
        for e in beam
    ]
    tried = []
    for entry in beam:
      for candidate in group.candidates:
        if time.monotonic() - start > time_budget_s:
          budget_exhausted = True
          break
        if candidate.label.startswith('saw_listed') and any(
            r['candidate'].startswith('rot3d_body')
            and r.get('fidelity', -1.0) >= 0
            for r in tried
        ):
          continue
        new_state, ok, attempts, applied = _apply_candidate(
            entry.state, candidate, call_cache=call_cache
        )
        record: dict[str, Any] = {
            'candidate': candidate.label,
            'ok': ok,
            'calls': len(attempts),
        }
        for attempt in attempts:
          for name in _call_elements(attempt['call']):
            if name in planned_names:
              element_attempts.setdefault(name, []).append(attempt)
        if ok:
          score, report = evaluate(new_state)
          evaluations += 1
          record['fidelity'] = round(score, 3)
          if 'error' in report:
            record['error'] = report['error']
          if score >= 0:
            expansions.append(
                _Entry(
                    score,
                    new_state,
                    entry.choices + ((group.name, candidate.label),),
                )
            )
            record['applied'] = applied
        tried.append(record)
      if budget_exhausted:
        break
    if (
        budget_exhausted
        and group.kind != 'body'
        and not any('fidelity' in r for r in tried)
    ):
      # Out of time: prefer non-rotation_3d candidates first to avoid slow
      # graph_data recomputation on fallback.
      entry = beam[0]
      fallback_candidates = sorted(
          group.candidates, key=lambda c: c.label.startswith('rot3d')
      )
      for candidate in fallback_candidates:
        new_state, ok, attempts, applied = _apply_candidate(
            entry.state, candidate, call_cache=call_cache
        )
        if ok:
          expansions = [
              _Entry(
                  entry.score,
                  new_state,
                  entry.choices
                  + ((group.name, candidate.label + ' (unevaluated)'),),
              )
          ]
          tried.append({
              'candidate': candidate.label,
              'ok': True,
              'applied': applied,
              'unevaluated': True,
          })
          break
    expansions.sort(key=lambda e: -e.score)
    beam = expansions[: max(1, beam_width)]
    summaries.append({
        'group': group.name,
        'kind': group.kind,
        'elements': list(group.elements),
        'tried': tried,
        'chosen': beam[0].choices[-1][1],
        'score': round(beam[0].score, 3),
    })
  best = beam[0]
  for summary in summaries:
    for record in summary['tried']:
      if record.get('candidate') == summary['chosen'].split(' ')[0] and (
          record.get('applied')
      ):
        for call in record['applied']:
          if call['name'] == 'shape_angle_width' and not call.get(
              'arguments', {}
          ).get('only_listed', False):
            element_applied.clear()
            for name in planned_names:
              element_applied[name] = call
          else:
            for name in _call_elements(call):
              if name in planned_names:
                element_applied.setdefault(name, call)
        break
  final_score = best.score
  if budget_exhausted:
    final_score, _ = evaluate(best.state)
    evaluations += 1
  report = []
  for plan in plans:
    report.append({
        'name': plan.name,
        'role': plan.role,
        'applied': element_applied.get(plan.name),
        'attempts': element_attempts.get(plan.name, [])[:12],
    })
  trajectory = [round(base_score, 3)] + [s['score'] for s in summaries]
  if budget_exhausted and summaries:
    trajectory[-1] = round(final_score, 3)
  candidates_tried = sum(len(s['tried']) for s in summaries)
  candidates_kept = sum(1 for _, choice in best.choices if choice != 'skip')
  summary = {
      'base_fidelity': round(base_score, 3),
      'fidelity': round(final_score, 3),
      'fidelity_trajectory': trajectory,
      'candidates_tried': candidates_tried,
      'candidates_kept': candidates_kept,
      'groups': summaries,
      'choices': [list(c) for c in best.choices],
      'evaluations': evaluations,
      'seconds': round(time.monotonic() - start, 1),
      'budget_exhausted': budget_exhausted,
      'beam_width': beam_width,
  }
  return best.state, plans, report, summary


def _call_elements(call: Mapping[str, Any]) -> list[str]:
  arguments = call.get('arguments', {})
  if 'element_name' in arguments:
    return [arguments['element_name']]
  if 'elements' in arguments:
    return [str(e) for e in arguments['elements']]
  if 'updates' in arguments:
    return [u['element_name'] for u in arguments['updates']]
  if 'limbs' in arguments:
    names = []
    for limb in arguments['limbs']:
      if limb.get('proximal'):
        names.append(limb['proximal'])
      if limb.get('distal'):
        names.append(limb['distal'])
    return names
  return [r['element_name'] for r in arguments.get('rotations', [])]


def sculpt_model(
    state: shaping.ShapingState,
    *,
    max_vertex_error: float = fidelity.CLEAN_GATE_MAX_VERTEX_ERROR,
    require_flat_foldable: bool = True,
    time_budget_s: float = 90.0,
) -> tuple[shaping.ShapingState, dict[str, Any]]:
  """Two-phase clean sculptor: broad plate expansion + pose + articulation."""
  with _fast_search_patches():
    return _sculpt_model_impl(
        state,
        max_vertex_error=max_vertex_error,
        require_flat_foldable=require_flat_foldable,
        time_budget_s=time_budget_s,
    )


def _sculpt_model_impl(
    state: shaping.ShapingState,
    *,
    max_vertex_error: float = fidelity.CLEAN_GATE_MAX_VERTEX_ERROR,
    require_flat_foldable: bool = True,
    time_budget_s: float = 90.0,
) -> tuple[shaping.ShapingState, dict[str, Any]]:
  """Internal implementation of `sculpt_model`."""
  start = time.monotonic()
  deadline = start + max(5.0, float(time_budget_s))
  layout = state.paper_layout
  topo = fidelity.topology(state.tree)
  all_names = set(topo.order)

  def _eval_clean(
      cand_state: shaping.ShapingState,
  ) -> tuple[bool, float, dict[str, Any]]:
    pct, _, non_ff = cand_state.pattern.vertex_flat_foldability_percentage()
    if require_flat_foldable and (non_ff or not math.isclose(pct, 1.0)):
      return False, -1.0, {'non_ff_vertices': len(non_ff)}
    try:
      rep = fidelity.score_pattern(
          cand_state.tree,
          cand_state.packing,
          cand_state.pattern,
          layout=layout,
      )
    except (ValueError, KeyError, ZeroDivisionError) as exc:
      return False, -1.0, {'error': str(exc)}
    verr = float(rep.get('fold_vertex_error_percent', 0.0))
    is_ok = verr <= max_vertex_error
    return is_ok, float(rep['fidelity']), rep

  _, base_fid, base_rep = _eval_clean(state)
  best_state = state
  best_fid = max(0.0, base_fid)
  best_rep = base_rep

  # Phase A: Closed-loop figure_pose foundation (fallback to search_pose).
  pose_budget = min(40.0, 0.50 * max(5.0, float(time_budget_s)))
  pose_summary: dict[str, Any] = {}
  posed_state: shaping.ShapingState | None = None
  try:
    posed_state, pose_summary = figure_pose(
        state, iterations=2, narrow=bool(time_budget_s >= 20.0)
    )
  except ValueError:
    posed_state, _, _, pose_summary = search_pose(
        state, time_budget_s=pose_budget
    )
  pose_clean, pose_fid, pose_rep = _eval_clean(posed_state)
  if not pose_clean and require_flat_foldable:
    flat_pat = fidelity.as_flat_backbone(posed_state.pattern)
    try:
      flat_posed = posed_state.replace(pattern=flat_pat, flat_backbone=None)
      pose_clean, pose_fid, pose_rep = _eval_clean(flat_posed)
      if pose_clean:
        posed_state = flat_posed
    except ValueError:
      pass
  if pose_clean and pose_fid >= best_fid:
    best_state = posed_state
    best_fid = pose_fid
    best_rep = pose_rep
  else:
    rot_specs = [
        {
            'element_name': name,
            'az': float(state.tree[name].azimuth_angle),
            'el': float(state.tree[name].elevation_angle),
        }
        for name in topo.order
        if name != state.tree.root
        and not state.tree[name].is_river
        and state.tree[name].length >= 2
    ]
    if rot_specs:
      rot_state, rot_res = shaping.apply_tool(
          state,
          'rotation_3d',
          {'rotations': rot_specs},
      )
      if rot_res.ok:
        r_clean, r_fid, r_rep = _eval_clean(rot_state)
        if r_clean and r_fid > best_fid:
          best_state = rot_state
          best_fid = r_fid
          best_rep = r_rep

  pose_stage_fid = best_fid

  # Identify broad plates (`width >= 2`, non-river, non-root), pairing L/R.
  visited_plates: set[str] = set()
  plate_groups: list[tuple[str, str | None, float]] = []
  for name in sorted(
      topo.order,
      key=lambda n: (-float(state.tree[n].width), -float(state.tree[n].length)),
  ):
    stick = state.tree[name]
    if (
        name in visited_plates
        or name == state.tree.root
        or stick.is_river
        or stick.width < 2.0
        or stick.length < 2
    ):
      continue
    partner = shaping.resolve_symmetric_element(state, name)
    if partner and partner in all_names and partner != name:
      visited_plates.add(name)
      visited_plates.add(partner)
      plate_groups.append((name, partner, float(stick.width)))
    else:
      visited_plates.add(name)
      plate_groups.append((name, None, float(stick.width)))

  plates_shaped: list[dict[str, Any]] = []
  unexpanded_wide: list[tuple[str, str | None, float]] = []

  def _plate_width(rep: Mapping[str, Any], elem: str) -> float:
    return float(rep.get('widths', {}).get(elem, 0.0))

  for left_name, right_name, target_w in plate_groups:
    if time.monotonic() >= deadline:
      break
    cur_w = _plate_width(best_rep, left_name)
    if right_name is not None:
      cur_w = 0.5 * (cur_w + _plate_width(best_rep, right_name))
    if cur_w >= 0.78 * target_w:
      continue

    if right_name is not None:
      calls = [
          (
              'apply_symmetric',
              {
                  'tool': 'fan_out_fold_heuristic',
                  'left_element': left_name,
                  'right_element': right_name,
                  'move_delta': 1.0,
                  'open_percent': 0.65,
              },
          ),
          (
              'apply_symmetric',
              {
                  'tool': 'squash_fold',
                  'left_element': left_name,
                  'right_element': right_name,
                  'open_percent': 0.35,
              },
          ),
          (
              'apply_symmetric',
              {
                  'tool': 'squash_fold',
                  'left_element': left_name,
                  'right_element': right_name,
                  'open_percent': 1.0,
              },
          ),
          (
              'apply_symmetric',
              {
                  'tool': 'fan_out_fold_heuristic',
                  'left_element': left_name,
                  'right_element': right_name,
                  'move_delta': 1.0,
                  'open_percent': 1.0,
              },
          ),
      ]
    else:
      calls = [
          (
              'squash_fold',
              {'element_name': left_name, 'open_percent': 0.30},
          ),
          (
              'fan_out_fold_heuristic',
              {
                  'element_name': left_name,
                  'move_delta': 1.0,
                  'open_percent': 0.65,
              },
          ),
          (
              'squash_fold',
              {'element_name': left_name, 'open_percent': 1.0},
          ),
          (
              'fan_out_fold_heuristic',
              {
                  'element_name': left_name,
                  'move_delta': 1.0,
                  'open_percent': 1.0,
              },
          ),
          (
              'fan_out_fold_heuristic',
              {
                  'element_name': left_name,
                  'move_delta': -1.0,
                  'open_percent': 1.0,
              },
          ),
      ]

    accepted_plate = False
    for tool_name, kwargs in calls:
      if time.monotonic() >= deadline:
        break
      cand_state, res = shaping.apply_tool(best_state, tool_name, kwargs)
      if not res.ok:
        continue
      c_clean, c_fid, c_rep = _eval_clean(cand_state)
      if not c_clean:
        continue
      new_w = _plate_width(c_rep, left_name)
      if right_name is not None:
        new_w = 0.5 * (new_w + _plate_width(c_rep, right_name))
      pose_ok = bool(c_rep.get('pose_gate', {}).get('passed', False)) or not (
          best_rep.get('pose_gate', {}).get('passed', False)
      )
      if pose_ok and (
          c_fid > best_fid + 0.02
          or (new_w >= max(cur_w * 1.20, cur_w + 0.35) and c_fid >= best_fid - 0.08)
      ):
        best_state = cand_state
        best_fid = c_fid
        best_rep = c_rep
        plates_shaped.append({
            'elements': [left_name]
            if right_name is None
            else [left_name, right_name],
            'tool': tool_name,
            'width_before': round(cur_w, 3),
            'width_after': round(new_w, 3),
            'fidelity': round(c_fid, 3),
        })
        accepted_plate = True
        break
    if not accepted_plate and target_w >= 3.0:
      unexpanded_wide.append((left_name, right_name, target_w))

  # Pre-pose fallback for chain-embedded wide plates (e.g. cobra hood) where
  # post-pose squash was blocked by an earlier rotation_3d crease.
  if unexpanded_wide and time.monotonic() < deadline - 4.0:
    pre_state = state
    pre_shaped_names: set[str] = set()
    for left_name, right_name, _ in unexpanded_wide:
      if right_name is not None:
        for open_pct in (0.35, 0.65, 1.0):
          cand_s, res = shaping.apply_tool(
              pre_state,
              'apply_symmetric',
              {
                  'tool': 'squash_fold',
                  'left_element': left_name,
                  'right_element': right_name,
                  'open_percent': open_pct,
              },
          )
          if res.ok:
            pre_state = cand_s
            pre_shaped_names.update((left_name, right_name))
            break
      else:
        for open_pct in (0.30, 0.60, 1.0):
          cand_s, res = shaping.apply_tool(
              pre_state,
              'squash_fold',
              {'element_name': left_name, 'open_percent': open_pct},
          )
          if res.ok:
            pre_state = cand_s
            pre_shaped_names.add(left_name)
            break
    if pre_shaped_names:
      # Include pre-shaped plates in rotation_3d first so root hinges bend them
      # to target (az, el) while interior squash tiers stay open; fall back to
      # rotating only remaining elements if root-hinge rotation conflicts.
      for skip_preshaped in (False, True):
        rot_specs = [
            {
                'element_name': name,
                'az': float(state.tree[name].azimuth_angle),
                'el': float(state.tree[name].elevation_angle),
            }
            for name in topo.order
            if name != state.tree.root
            and (not skip_preshaped or name not in pre_shaped_names)
            and not state.tree[name].is_river
            and state.tree[name].length >= 2
        ]
        if not rot_specs:
          continue
        cand_s, res = shaping.apply_tool(
            pre_state,
            'rotation_3d',
            {'rotations': rot_specs},
        )
        if res.ok:
          c_clean, c_fid, c_rep = _eval_clean(cand_s)
          pose_ok = bool(c_rep.get('pose_gate', {}).get('passed', False)) or not (
              best_rep.get('pose_gate', {}).get('passed', False)
          )
          if c_clean and pose_ok and c_fid > best_fid + 0.05:
            best_state = cand_s
            best_fid = c_fid
            best_rep = c_rep
            plates_shaped.append({
                'elements': sorted(pre_shaped_names),
                'tool': 'pre_pose_squash+rotation_3d',
                'fidelity': round(c_fid, 3),
            })
            break

  # Phase B: Clean symmetric secondary articulation on unbent paired limbs.
  articulations: list[dict[str, Any]] = []
  seen_chains: set[str] = set()
  chains_map = best_rep.get('limb_articulation', {})
  dir_errs = best_rep.get('direction_error_deg', {})
  for _, chain in layout.chains.items():
    if time.monotonic() >= deadline or len(chain) < 2:
      break
    distal_name = chain[-1]
    if distal_name in seen_chains or state.tree[distal_name].length < 2:
      continue
    partner = shaping.resolve_symmetric_element(state, distal_name)
    if not partner or partner not in all_names:
      continue
    seen_chains.update((distal_name, partner))
    art_key = next(
        (k for k, v in layout.chains.items() if v and v[-1] == distal_name),
        '',
    )
    cur_art = float(chains_map.get(art_key, 1.0))
    cur_derr = max(
        float(dir_errs.get(distal_name, 0.0)),
        float(dir_errs.get(partner, 0.0)),
    )
    if cur_art >= 0.75 and cur_derr <= 35.0:
      continue

    for tool_sub, sub_args in (
        ('crimp_fold', {'angle': 45.0, 'depth': 0.35}),
        ('reverse_fold', {'angle': 45.0, 'depth': 0.35}),
        (
            'simple_fold',
            {
                'angle': 45.0,
                'depth': 0.0,
                'narrow': bool(state.tree[distal_name].width <= 1.0),
            },
        ),
    ):
      if time.monotonic() >= deadline:
        break
      cand_s, res = shaping.apply_tool(
          best_state,
          'apply_symmetric',
          {
              'tool': tool_sub,
              'left_element': distal_name,
              'right_element': partner,
              **sub_args,
          },
      )
      if not res.ok:
        continue
      c_clean, c_fid, c_rep = _eval_clean(cand_s)
      if not c_clean:
        continue
      pose_ok = bool(c_rep.get('pose_gate', {}).get('passed', False)) or not (
          best_rep.get('pose_gate', {}).get('passed', False)
      )
      if pose_ok and c_fid > best_fid + 0.03:
        best_state = cand_s
        best_fid = c_fid
        best_rep = c_rep
        articulations.append({
            'elements': [distal_name, partner],
            'tool': tool_sub,
            'fidelity': round(c_fid, 3),
        })
        break

  pct_ff, _, non_ff_final = (
      best_state.pattern.vertex_flat_foldability_percentage()
  )
  verr_final = float(best_rep.get('fold_vertex_error_percent', 0.0))
  sculpt_report = {
      'base_fidelity': round(max(0.0, base_fid), 3),
      'pose_fidelity': round(pose_stage_fid, 3),
      'fidelity': round(best_fid, 3),
      'vertex_flat_foldability_percentage': round(float(pct_ff), 6),
      'non_flat_foldable_vertices': len(non_ff_final),
      'fold_vertex_error_percent': round(verr_final, 4),
      'clean_gate': fidelity.clean_gate(verr_final, len(non_ff_final)),
      'pose_gate': best_rep.get('pose_gate'),
      'plates_shaped': plates_shaped,
      'articulations_applied': articulations,
      'widths': best_rep.get('widths', {}),
      'components': best_rep.get('components', {}),
      'figure_pose_summary': {
          'fidelity': pose_summary.get('fidelity'),
          'seconds': pose_summary.get('seconds'),
      },
      'seconds': round(time.monotonic() - start, 2),
  }
  return best_state, sculpt_report


def pose_solved(
    solved: bus.SolvedCreasePattern,
    *,
    search: bool = True,
    figure: bool = True,
    narrow: bool = False,
    time_budget_s: float = 120.0,
) -> tuple[bus.ShapedCreasePattern, dict[str, Any]]:
  """Poses a `SolvedCreasePattern` using `figure_pose` and/or `search_pose`.

  Args:
    solved: Flat-foldable base crease pattern bus object.
    search: Whether to run group-based coordinate search if `figure_pose` does
      not pass `pose_gate`.
    figure: Whether to run closed-loop `figure_pose` at stick-figure angles.
    narrow: With `figure=True`, whether to narrow thin limbs on the base first.
    time_budget_s: Wall-time budget in seconds when `search=True`.

  Returns:
    `(shaped_bus_pattern, pose_report)`.
  """
  t0 = time.monotonic()
  state = shaping.ShapingState.from_solved(solved)
  fig_res = None
  if figure:
    try:
      fig_res = figure_pose(state, narrow=narrow)
    except ValueError:
      fig_res = None

  if fig_res is not None and fig_res[1].get('pose_gate', {}).get('passed'):
    best_state, fig_summary = fig_res
    report: dict[str, Any] = {
        'mode': 'figure',
        'fidelity': fig_summary['fidelity'],
        'seconds': fig_summary['seconds'],
        'figure_pose': fig_summary,
    }
  elif search:
    best_state, plans, elements, search_summary = search_pose(
        state, time_budget_s=time_budget_s
    )
    report = dict(
        search_summary,
        mode='search',
        planned=sum(1 for p in plans if p.candidates),
        elements=elements,
    )
  else:
    best_state, plans, elements = pose(state)
    fid_rep = fidelity.score_pattern(
        best_state.tree, best_state.packing, best_state.pattern
    )
    report = {
        'mode': 'greedy',
        'fidelity': fid_rep['fidelity'],
        'planned': sum(1 for p in plans if p.candidates),
        'elements': elements,
    }

  if report['mode'] != 'figure' and fig_res is not None:
    fig_state, fig_summary = fig_res
    cur_fid = float(
        report.get('fidelity')
        or fidelity.score_pattern(
            best_state.tree, best_state.packing, best_state.pattern
        )['fidelity']
    )
    report['figure_pose_fidelity'] = fig_summary['fidelity']
    if fig_summary['fidelity'] > cur_fid:
      best_state = fig_state
      report.update(
          mode='figure',
          fidelity=fig_summary['fidelity'],
          figure_pose=fig_summary,
      )

  elapsed = round(time.monotonic() - t0, 3)
  shaped = best_state.to_shaped(
      solved, strategy=f"pose_{report['mode']}", time_taken=elapsed
  )
  return shaped, report
