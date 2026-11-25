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

"""Topology discovery and validation for composable shaping-flap templates."""

from . import stick as stick_lib
from . import stick_figure as stick_figure_lib

MAX_TAIL_TEMPLATES = 2
MAX_KNEE_TEMPLATES = 4
MAX_SYMMETRIC_KNEE_PAIRS = 2


def has_shaping_flaps(tree: stick_figure_lib.StickFigure) -> bool:
  """Returns whether `tree` contains any `shaping_flap=True` sticks."""
  return any(stick.shaping_flap for stick in tree.children)


def _extend_tail_chain_names(
    first_river: stick_lib.Stick,
    start_joint: str,
    tree: stick_figure_lib.StickFigure,
    claimed: set[str],
    flaps: list[stick_lib.Stick],
    rivers: list[stick_lib.Stick],
) -> None:
  """Extends `(flaps, rivers)` outward along degree-3 `SF + river` joints."""
  curr_stick = first_river
  prev_joint = start_joint
  while True:
    far_joint_name = curr_stick.other_joint_name(prev_joint)
    far_joint = tree.joints[far_joint_name]
    if far_joint.degree != 3:
      break
    others = [s for s in far_joint.sticks if s.name != curr_stick.name]
    if len(others) != 2:
      break
    s1, s2 = others
    if s1.name in claimed or s2.name in claimed:
      break
    s1_is_leaf_sf = s1.shaping_flap and (
        tree.joints[s1.start].degree == 1 or tree.joints[s1.end].degree == 1
    )
    s2_is_leaf_sf = s2.shaping_flap and (
        tree.joints[s2.start].degree == 1 or tree.joints[s2.end].degree == 1
    )
    if s1_is_leaf_sf and not s2.shaping_flap and not s2.is_flap:
      sf_next, rv_next = s1, s2
    elif s2_is_leaf_sf and not s1.shaping_flap and not s1.is_flap:
      sf_next, rv_next = s2, s1
    else:
      break
    flaps.append(sf_next)
    rivers.append(rv_next)
    curr_stick = rv_next
    prev_joint = far_joint_name


def discover_shaping_template_tuples(
    tree: stick_figure_lib.StickFigure,
    *,
    merge_chains: bool = True,
) -> tuple[
    list[
        tuple[
            tuple[stick_lib.Stick, ...],
            tuple[stick_lib.Stick, ...],
            stick_lib.Stick | None,
        ]
    ],
    list[tuple[stick_lib.Stick, stick_lib.Stick, stick_lib.Stick, str]],
    set[str],
]:
  """Discovers raw `(tail_tuples, knee_tuples, tail_claimed)` in `tree`."""
  merged_tree = tree.merge_linear_chains()[0] if merge_chains else tree
  tails: list[
      tuple[
          tuple[stick_lib.Stick, ...],
          tuple[stick_lib.Stick, ...],
          stick_lib.Stick | None,
      ]
  ] = []
  claimed: set[str] = set()

  for stick in merged_tree.children:
    if not stick.shaping_flap or stick.name in claimed:
      continue
    for j_name in (stick.start, stick.end):
      if merged_tree.joints[j_name].degree != 1:
        continue
      next_joint_name = stick.other_joint_name(j_name)
      next_joint = merged_tree.joints[next_joint_name]
      matched: (
          tuple[
              tuple[stick_lib.Stick, ...],
              tuple[stick_lib.Stick, ...],
              stick_lib.Stick | None,
          ]
          | None
      ) = None
      if next_joint.degree == 2:
        first_river = next(
            s for s in next_joint.sticks if s.name != stick.name
        )
        if (
            not first_river.shaping_flap
            and not first_river.is_flap
            and first_river.name not in claimed
        ):
          flaps = [stick]
          rivers = [first_river]
          _extend_tail_chain_names(
              first_river, next_joint_name, merged_tree, claimed, flaps, rivers
          )
          if len(flaps) >= 2:
            matched = (tuple(flaps), tuple(rivers), None)
      elif next_joint.degree == 3 and stick.is_on_symmetry_axis:
        others = [s for s in next_joint.sticks if s.name != stick.name]
        normal_tips = [
            s
            for s in others
            if not s.shaping_flap
            and s.is_flap
            and s.is_on_symmetry_axis
            and s.name not in claimed
            and (
                merged_tree.joints[s.start].degree == 1
                or merged_tree.joints[s.end].degree == 1
            )
        ]
        axis_rivers = [
            s
            for s in others
            if not s.shaping_flap
            and not s.is_flap
            and s.is_on_symmetry_axis
            and s.name not in claimed
        ]
        if len(normal_tips) == 1 and len(axis_rivers) == 1:
          tip_flap = normal_tips[0]
          first_river = axis_rivers[0]
          flaps = [stick]
          rivers = [first_river]
          _extend_tail_chain_names(
              first_river, next_joint_name, merged_tree, claimed, flaps, rivers
          )
          if len(flaps) >= 2 or merged_tree.is_symmetric_tree:
            matched = (tuple(flaps), tuple(rivers), tip_flap)
      if matched is not None:
        tails.append(matched)
        claimed.update(s.name for s in matched[0])
        claimed.update(s.name for s in matched[1])
        if matched[2] is not None:
          claimed.add(matched[2].name)
        break

  knees: list[tuple[stick_lib.Stick, stick_lib.Stick, stick_lib.Stick, str]] = (
      []
  )
  used_joints: set[str] = set()
  for joint_name, joint in merged_tree.joints.items():
    if joint_name in used_joints or joint.degree != 3:
      continue
    sticks = joint.sticks
    if any(s.name in claimed for s in sticks):
      continue
    sf_sticks = [
        s
        for s in sticks
        if s.shaping_flap
        and (
            merged_tree.joints[s.start].degree == 1
            or merged_tree.joints[s.end].degree == 1
        )
    ]
    normal_flaps = [
        s for s in sticks if not s.shaping_flap and not s.is_river and s.is_flap
    ]
    rivers = [s for s in sticks if not s.shaping_flap and s.is_river]
    if len(sf_sticks) == 1 and len(normal_flaps) == 1 and len(rivers) == 1:
      knees.append((sf_sticks[0], normal_flaps[0], rivers[0], joint_name))
      for s in sticks:
        used_joints.add(s.start)
        used_joints.add(s.end)

  return tails, knees, claimed


def validate_shaping_flap_templates(
    tree: stick_figure_lib.StickFigure,
) -> list[str]:
  """Checks template membership and builtin packer count limits on `tree`."""
  if not has_shaping_flaps(tree):
    return []
  merged_tree, _ = tree.merge_linear_chains()
  tails, knees, tail_claimed = discover_shaping_template_tuples(
      merged_tree, merge_chains=False
  )
  covered = set(tail_claimed)
  for sf_stick, _, _, _ in knees:
    covered.add(sf_stick.name)

  errors: list[str] = []
  for stick in merged_tree.children:
    if stick.shaping_flap and stick.name not in covered:
      errors.append(
          f"Shaping flap '{stick.name}' does not match a valid TailTemplate"
          ' (alternating SF-River-SF-... chain, or central tip_flap + SF +'
          ' River) or KneeTemplate (degree-3 joint with 1 shaping_flap, 1'
          ' normal leaf flap, and 1 river).'
      )

  if len(tails) > MAX_TAIL_TEMPLATES:
    errors.append(
        f'At most {MAX_TAIL_TEMPLATES} TailTemplates are supported, got'
        f' {len(tails)}.'
    )
  if merged_tree.is_symmetric_tree:
    seen_pairs: set[str] = set()
    pair_count = 0
    for sf_stick, _, _, _ in knees:
      if sf_stick.name in seen_pairs:
        continue
      seen_pairs.add(sf_stick.name)
      if sf_stick.symmetric_stick_name:
        seen_pairs.add(sf_stick.symmetric_stick_name)
      pair_count += 1
    if pair_count > MAX_SYMMETRIC_KNEE_PAIRS:
      errors.append(
          f'At most {MAX_SYMMETRIC_KNEE_PAIRS} mirrored KneeTemplate pairs'
          f' ({MAX_KNEE_TEMPLATES} total corner knees) are supported on'
          f' symmetric figures, got {pair_count} pairs ({len(knees)} knees).'
      )
  elif len(knees) > MAX_KNEE_TEMPLATES:
    errors.append(
        f'At most {MAX_KNEE_TEMPLATES} corner KneeTemplates are supported, got'
        f' {len(knees)}.'
    )

  return errors
