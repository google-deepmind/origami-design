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

"""Heuristics for determining the shape angles for shaping.

For now we support get_flap_info_dict, which determines the
shape angle for each flap used in the zigzag shaping from the stick figure.
"""

import collections
from collections.abc import Collection, Mapping, Sequence
import copy
import dataclasses
import enum
import itertools
import math
from typing import Any

from absl import logging
import numpy as np

from ..packing import packing_output
from ..packing import types
from ..solving import hinges as hinge_utils
from ..stick_figures import angle_utils
from ..stick_figures import stick as stick_lib
from ..stick_figures import stick_figure as sf
from ..stick_figures import traversal as tr
from . import shaping_elements


class AbsFlapDirection(enum.Enum):
  """The absolute direction of a flap.

  Note that the direction is relative to the root stick, e.g., for a stick with
  two joints A and B, we can assume A->B or B->A as the up direction. Thus the
  absolute direction depende on how we set the root stick direction. That is why
  we need to specify `root_direction_follow_stick_figure` in the following
  functions. If True, we use the stick.start->stick.end as the up direction.
  Otherwise, we use stick.end->stick.start as the up direction.

  Attrs:
    UP: The flap is pointing up.
    DOWN: The flap is pointing down.
    NOT_DETERMINED: The flap direction is not determined.
    REFERENCE: The flap is the reference node (root node).
  """

  UP = 1
  DOWN = 2
  NOT_DETERMINED = 3
  REFERENCE = 4

  def flip(self) -> 'AbsFlapDirection':
    """Flips the direction."""
    if self == AbsFlapDirection.NOT_DETERMINED:
      raise ValueError('Cannot flip a not determined direction.')
    if self == AbsFlapDirection.REFERENCE:
      raise ValueError('Cannot flip a reference direction.')
    if self is AbsFlapDirection.UP:
      return AbsFlapDirection.DOWN
    else:
      return AbsFlapDirection.UP


class FoldState(enum.Enum):
  """The fold state of a node.

  Attrs:
    FOLD_IN: The fold is fold-in (point to the parent).
    FOLD_OUT: The fold is fold-out (point away from the parent).
    NOT_DETERMINED: The fold state is not determined.
    REFERENCE: The node is the reference node (root node).`
  """

  FOLD_IN = 'fold_in'
  FOLD_OUT = 'fold_out'
  NOT_DETERMINED = 'not_determined'
  REFERENCE = 'reference'


def get_first_stick_for_shaping(
    tree: sf.StickFigure,
    *,
    use_tree_root: bool = False,
) -> stick_lib.Stick:
  """Chooses a parent stick for shaping.

  The parent stick is chosen based on the following ordered heuristics unless
  `use_tree_root` is True:
  1. The stick on the symmetry axis.
  2. The stick with joints that require the most packing space.
  3. The stick with the largest degree.
  4. The largest stick.

  Args:
    tree: The stick figure to choose a parent stick for.
    use_tree_root: When True, return `tree[tree.root]` directly so angles are
      measured in the stick figure's declared root frame.

  Returns:
    The chosen parent stick.
  """
  if use_tree_root and tree.root and tree.root in tree:
    return tree[tree.root]
  unmerged_tree = tree
  tree, merged_stick_name_dict = tree.merge_linear_chains()

  candidates = list(tree.children)

  # Heuristic 0 (for testing purposes):
  for stick in candidates:
    if 'IS_ROOT' == stick.name:
      return stick

  # Heuristic 1: On symmetry axis
  on_axis = [s for s in candidates if s.is_on_symmetry_axis]
  if on_axis:
    candidates = on_axis

  # Heuristic 2: Stick with joints requiring most packing space.
  if len(candidates) > 1:
    scores = {}
    for s in candidates:
      score = 0
      if s.start in tree.joints:
        score += tree.joints[s.start].get_required_grid_size(tree)
      if s.end in tree.joints:
        score += tree.joints[s.end].get_required_grid_size(tree)
      scores[s.name] = score

    if scores:
      max_score = max(scores.values())
      candidates = [s for s in candidates if scores[s.name] == max_score]

  # Heuristic 3: Largest degree
  if len(candidates) > 1:

    def get_stick_degree(stick: stick_lib.Stick) -> int:
      return tree.joints[stick.start].degree + tree.joints[stick.end].degree

    max_degree = max(get_stick_degree(s) for s in candidates)
    candidates = [s for s in candidates if get_stick_degree(s) == max_degree]

  # Heuristic 4: Largest length
  if len(candidates) > 1:
    max_length = max(s.length for s in candidates)
    candidates = [s for s in candidates if s.length == max_length]

  # Final tie-breaker: Sort by name
  if len(candidates) > 1:
    candidates.sort(key=lambda s: s.name)

  merged_root_stick = candidates[0]

  # Return stick if it is not merged.
  if merged_root_stick.name not in merged_stick_name_dict:
    return merged_root_stick

  # Map root of merged tree back to unmerged tree
  # by picking sub-stick of merged tree root
  # closest to the merged tree root's start joint.
  root_sub_stick = None
  for sub_stick_name in merged_stick_name_dict[merged_root_stick.name]:
    sub_stick = unmerged_tree[sub_stick_name]
    if merged_root_stick.start in (sub_stick.start, sub_stick.end):
      root_sub_stick = sub_stick

  assert (
      root_sub_stick is not None
  ), f'No sub-stick found for merged root stick {merged_root_stick.name}.'
  return root_sub_stick


def shaping_angles(
    tree: sf.StickFigure,
    *,
    use_tree_root: bool = False,
) -> tuple[stick_lib.Stick, dict[str, tuple[str | None, float, float]]]:
  """Chooses a parent stick and calculates all angles relative to it.

  Args:
    tree: The stick figure to calculate angles for.
    use_tree_root: Whether to use `tree[tree.root]` as the parent stick.

  Returns:
    A tuple of the parent stick and a dictionary mapping stick names to a
    tuple of (parent stick name, relative azimuth, relative elevation).
  """
  parent_stick = get_first_stick_for_shaping(tree, use_tree_root=use_tree_root)
  relative_angles = relative_angles_from_stick(tree, parent_stick)
  return parent_stick, relative_angles


def relative_angles_from_stick(
    tree: sf.StickFigure, first_stick: stick_lib.Stick
) -> dict[str, tuple[str | None, float, float]]:
  """Given the first stick as reference, calculates all angles relative to its parent stick.

  The relative angle points from the parent stick to the child stick. Positive
  angles indicate a counter-clockwise rotation. Negative angles indicate a
  clockwise rotation.

  Args:
    tree: The stick figure to calculate angles for.
    first_stick: The stick to use as the reference for angles.

  Returns:
    A dictionary mapping stick names to a tuple of (parent stick name,
    relative azimuth, relative elevation). The first stick will have `None` as
    its parent and (0, 0) as relative angles.
  """
  relative_angles: dict[str, tuple[str | None, float, float]] = {
      first_stick.name: (None, 0.0, 0.0)
  }
  for stick in tr.breadth_first_traversal(
      tree, start_stick_name=first_stick.name
  ):

    # Explore neighbors at both joints of the current stick.
    for joint_name in (stick.start, stick.end):
      joint = tree.joints.get(joint_name)
      if not joint:
        continue

      # Angle of the current stick pointing away from this joint.
      azimuth_at_joint = stick.azimuth_angle_at_joint(joint_name)
      elevation_at_joint = stick.elevation_angle_at_joint(joint_name)

      for neighbor_stick in joint.sticks:
        # If the neighbor hasn't been processed, `stick` is its parent in the
        # BFS tree.
        if neighbor_stick.name not in relative_angles:
          # Angle of the neighbor stick pointing away from the same joint.
          neighbor_azimuth_at_joint = neighbor_stick.azimuth_angle_at_joint(
              joint_name
          )
          neighbor_elevation_at_joint = neighbor_stick.elevation_angle_at_joint(
              joint_name
          )

          # The angle of the neighbor relative to the current stick.
          relative_azimuth = neighbor_azimuth_at_joint - azimuth_at_joint
          relative_azimuth = angle_utils.wrap_angle_centered(
              relative_azimuth, 360.0
          )

          # The angle of the neighbor relative to the current stick.
          relative_elevation = neighbor_elevation_at_joint - elevation_at_joint

          relative_azimuth = math.radians(relative_azimuth)
          relative_elevation = math.radians(relative_elevation)

          relative_angles[neighbor_stick.name] = (
              stick.name,
              relative_azimuth,
              relative_elevation,
          )

  return relative_angles


def copy_hinge_info_list(
    hinges: Sequence[types.HingeInfo] | None,
) -> list[types.HingeInfo]:
  """Copies a sequence of HingeInfo objects without deepcopying Creases."""
  return [
      dataclasses.replace(h, flap_names=list(h.flap_names))
      for h in hinges or ()
  ]


def update_hinge_info_from_solver_result(
    hinge_info_list: list[types.HingeInfo],
    solver_result: hinge_utils.SolverResult,
) -> list[types.HingeInfo]:
  """Updates the hinge info list with the solver results.

  Args:
    hinge_info_list: A list of `HingeInfo` objects containing information about
      hinges between flaps.
    solver_result: A `SolverResult` object containing the solver results.

  Returns:
    The updated hinge info list (update the `assigned` field)
  """
  for hinge in hinge_info_list:
    if hinge.coordinate is not None:
      crease_ = hinge.coordinate
      if crease_ is not None and any(
          crease_.is_overlapping(h) for h in solver_result.final_state.hinges
      ):
        hinge.assigned = True
  return hinge_info_list


def _find_hinge_info_from_flap_names(
    hinge_info_list: list[types.HingeInfo],
    flap_names_1: str,
    flap_names_2: str,
) -> types.HingeInfo | None:
  """Finds the hinge connecting two flaps from their names.

  Args:
    hinge_info_list: A list of `HingeInfo` objects containing information about
      hinges between flaps.
    flap_names_1: The name of the first flap.
    flap_names_2: The name of the second flap.

  Returns:
    The hinge info if found, otherwise None.
  """
  target_names = sorted((flap_names_1, flap_names_2))
  for hinge in hinge_info_list:
    if len(hinge.flap_names) == 2 and sorted(hinge.flap_names) == target_names:
      return hinge
  return None


@dataclasses.dataclass(kw_only=True)
class Node:
  """Represents a single stick in the stick figure, or a flap in the box-pleating grid."""

  name: str
  # Length of the stick.
  length: int
  is_root: bool
  # Children are now stored by name for easy serialization and lookup.
  children: list[str] = dataclasses.field(default_factory=list)
  # The relative angle element makes with its parent in [-pi, pi]
  relative_azimuth_angle: float
  relative_elevation_angle: float
  hinges: list[types.HingeInfo]
  # Whether the root stick points in the negative-y direction.
  root_direction_follow_stick_figure_root: bool
  parent: str | None = None
  # The fold state of the node. Whether the fold is fold-in (point to the
  # parent), fold-out (point away from the parent), or not determined.
  fold_state: FoldState = FoldState.NOT_DETERMINED
  absolute_direction: AbsFlapDirection = AbsFlapDirection.NOT_DETERMINED
  first_side_is_up: bool = True
  width: float = 1.0

  # Whether the node is a sacrificial flap or has sacrificial flaps.
  is_sacrificial: bool = False
  sacrificial_flaps: Sequence[str] = dataclasses.field(default_factory=list)

  def __post_init__(self):
    if self.is_sacrificial and self.sacrificial_flaps:
      raise ValueError(
          'Sacrificial flaps should not have their own sacrificial flaps.'
      )

  def __deepcopy__(self, memo: Any) -> 'Node':
    del memo
    return dataclasses.replace(
        self,
        children=list(self.children),
        hinges=copy_hinge_info_list(self.hinges),
        sacrificial_flaps=list(self.sacrificial_flaps),
    )

  def to_serializable_dict(self) -> dict[str, Any]:
    """Returns the node as a dictionary."""
    d = dataclasses.asdict(self)
    d['hinges'] = [hinge.flap_names for hinge in self.hinges]
    d['absolute_direction'] = str(self.absolute_direction)
    d['fold_state'] = str(self.fold_state)
    return d

  @property
  def has_assigned_hinge(self) -> bool:
    """Returns whether the node has an assigned hinge."""
    return any(hinge.assigned for hinge in self.hinges)

  @property
  def shape_angle(self) -> float | None:
    """Returns the shape angle of the node in radians, calculated from relative angle and fold state."""
    if self.fold_state == FoldState.REFERENCE:  # Root node has no shape angle.
      assert self.is_root
      return None
    if self.fold_state == FoldState.NOT_DETERMINED:
      raise ValueError(
          'Shape angle is not determined because fold_state is'
          f' {self.fold_state} and relative angle is'
          f' {self.relative_azimuth_angle}'
      )
    if self.fold_state == FoldState.FOLD_IN:
      shape_angle = self.relative_azimuth_angle
    else:
      shape_angle = self.relative_azimuth_angle - np.pi
      if shape_angle < -np.pi:
        shape_angle += 2 * np.pi
    return shape_angle

  @property
  def shape_angle_elevation(self) -> float | None:
    """Returns the shape angle of the node in radians, calculated from relative angle and fold state."""
    if self.fold_state == FoldState.REFERENCE:  # Root node has no shape angle.
      assert self.is_root
      return None
    if self.fold_state == FoldState.NOT_DETERMINED:
      raise ValueError(
          'Shape angle elevation is not determined because fold_state is'
          f' {self.fold_state} and relative angle is'
          f' {self.relative_elevation_angle}'
      )
    shape_angle = self.relative_elevation_angle

    # Wrap the angle to [-pi/2, pi/2]
    shape_angle = angle_utils.wrap_angle_centered(shape_angle, np.pi)

    return shape_angle

  def eq_without_children(self, other: 'Node') -> bool:
    """Returns whether the nodes are equal, ignoring children."""
    return (
        self.name == other.name
        and self.length == other.length
        and self.is_root == other.is_root
        and math.isclose(
            self.relative_azimuth_angle, other.relative_azimuth_angle
        )
        and math.isclose(
            self.relative_elevation_angle, other.relative_elevation_angle
        )
        and self.hinges == other.hinges
        and self.fold_state == other.fold_state
        and self.absolute_direction == other.absolute_direction
    )


def _get_hinge_infos_by_name(name: str, hinge_info_list: list[types.HingeInfo]):
  return [
      hinge_info
      for hinge_info in hinge_info_list
      if name in hinge_info.flap_names
  ]


def convert_stick_figure_to_node_dict(
    stick_figure: sf.StickFigure,
    hinge_info_list: list[types.HingeInfo],
    root_direction_follow_stick_figure_root: bool,
    sacrificial_flaps_by_base_flap: Mapping[str, Collection[str]] | None = None,
    *,
    use_tree_root: bool = False,
) -> tuple[Node, dict[str, Node]]:
  """Converts a StickFigure graph into a tree of Nodes and a lookup dictionary.

  Args:
    stick_figure: The StickFigure instance to convert.
    hinge_info_list: A list of `HingeInfo` objects containing information about
      hinges between flaps.
    root_direction_follow_stick_figure_root: Whether the root stick points in
      the negative-y direction of the stick figure.
    sacrificial_flaps_by_base_flap: A dictionary mapping base flaps to their
      sacrificial flaps.
    use_tree_root: Whether to anchor the node tree at `stick_figure.root`.

  Returns:
      A tuple containing:
      - The root Node of the tree.
      - A dictionary mapping all node names to their Node objects.
  """
  root_stick, relative_angles = shaping_angles(
      stick_figure, use_tree_root=use_tree_root
  )
  root_stick_name = root_stick.name

  # Reconstruct the tree structure from the relative angles.
  children_map: dict[str, list[str]] = collections.defaultdict(list)
  for child_name, (parent_name, _, _) in relative_angles.items():
    if parent_name is not None:
      children_map[parent_name].append(child_name)

  sacrificial_flaps_by_base_flap = sacrificial_flaps_by_base_flap or {}
  sacrificial_flaps = set(
      itertools.chain.from_iterable(sacrificial_flaps_by_base_flap.values())
  )

  # Build the Node objects.
  node_dict: dict[str, Node] = {}
  for stick_name, (parent_name, azimuth, elevation) in relative_angles.items():
    if parent_name is not None:
      # normalize relative angles to lie in range [-pi, pi]
      relative_azimuth_angle = angle_utils.wrap_angle_centered(
          azimuth, 2 * math.pi
      )
      relative_elevation_angle = angle_utils.wrap_angle_centered(
          elevation, 2 * math.pi
      )
      is_root = False
      assert stick_name != root_stick_name
    else:
      relative_azimuth_angle = 0.0
      relative_elevation_angle = 0.0
      is_root = True
      assert stick_name == root_stick_name

    # get all hinge infos incident to the node
    incident_hinge_infos = _get_hinge_infos_by_name(stick_name, hinge_info_list)
    stick_sfs = sacrificial_flaps_by_base_flap.get(stick_name, [])

    node = Node(
        name=stick_name,
        length=stick_figure[stick_name].length,
        is_root=is_root,
        children=children_map.get(stick_name, []),
        parent=parent_name,
        relative_azimuth_angle=relative_azimuth_angle,
        relative_elevation_angle=relative_elevation_angle,
        hinges=incident_hinge_infos,
        root_direction_follow_stick_figure_root=root_direction_follow_stick_figure_root,
        is_sacrificial=stick_name in sacrificial_flaps,
        sacrificial_flaps=stick_sfs,  # pyrefly: ignore[bad-argument-type]
        width=stick_figure[stick_name].width if stick_sfs else 1.0,
    )
    node_dict[stick_name] = node

  return node_dict[root_stick_name], node_dict


def _get_hinge_info_dict_for_connected_nodes(
    node_dict: dict[str, Node],
    root_node_name: str,
    hinge_info_list: list[types.HingeInfo],
    stick_figure: sf.StickFigure,
):
  """Builds an adjacency dictionary with hinge information for connected nodes.

  This function iterates over the node tree structure and for each pair of
  adjacent nodes (sharing a parent or being siblings), it attempts to determine
  if there is a hinge between them.

  Args:
    node_dict: A dictionary mapping node names to Node objects.
    root_node_name: The name of the root node.
    hinge_info_list: A list of `HingeInfo` objects containing information about
      hinges between flaps.
    stick_figure: The StickFigure instance, needed for joint information.

  Returns:
    A dictionary where keys are node names and values are dictionaries mapping
    adjacent node names to their HingeInfo.
  """
  # Pre-calculate which joint each child of the root connects to.
  child_to_joint_map = {}
  if root_node_name in node_dict:
    root_stick = stick_figure[root_node_name]
    root_children_names = node_dict[root_node_name].children
    for joint_name in (root_stick.start, root_stick.end):
      if joint_name in stick_figure.joints:
        for connected_stick in stick_figure.joints[joint_name].sticks:
          if connected_stick.name in root_children_names:
            child_to_joint_map[connected_stick.name] = joint_name

  adj_dict = collections.defaultdict(dict)
  for parent_name, parent_node in node_dict.items():
    children_names = parent_node.children
    for child_name in children_names:

      # Check adjacency with siblings and the parent.
      adjacents = children_names + [parent_name]
      for adj_name in adjacents:
        if child_name == adj_name:  # Skip self-loops
          continue

        is_sibling_of_root_child = (
            parent_name == root_node_name and adj_name != parent_name
        )
        if is_sibling_of_root_child:
          # If connected to different joints on the root stick, they are not
          # adjacent for shaping.
          if child_to_joint_map.get(child_name) != child_to_joint_map.get(
              adj_name
          ):
            logging.info(
                'Skipping %s and %s because they are not connected to the same'
                ' joint on the root stick.',
                child_name,
                adj_name,
            )
            continue

        hinge_info = _find_hinge_info_from_flap_names(
            hinge_info_list,
            child_name,
            adj_name,
        )
        if hinge_info:
          adj_dict[child_name][adj_name] = hinge_info

  return dict(adj_dict)


def _get_absolute_direction_from_non_root_parent_direction(
    parent_flap_direction: AbsFlapDirection | None,
    fold_state: FoldState,
) -> AbsFlapDirection:
  """Determines absolute direction based on parent's direction and fold state."""
  match (parent_flap_direction, fold_state):
    case (AbsFlapDirection.DOWN, FoldState.FOLD_IN):
      return AbsFlapDirection.UP
    case (AbsFlapDirection.DOWN, FoldState.FOLD_OUT):
      return AbsFlapDirection.DOWN
    case (AbsFlapDirection.UP, FoldState.FOLD_IN):
      return AbsFlapDirection.DOWN
    case (AbsFlapDirection.UP, FoldState.FOLD_OUT):
      return AbsFlapDirection.UP
    case _:
      raise ValueError(
          f'Invalid parent flap direction {parent_flap_direction} and fold'
          f' state {fold_state}!'
      )


def _get_first_side_is_up_from_non_root_parent_direction(
    parent_first_side_is_up: bool,
    fold_state: FoldState,
) -> bool:
  """Determines the first_side_is_up based on parent's first_side_is_up and fold state."""
  match (parent_first_side_is_up, fold_state):
    case (True, FoldState.FOLD_IN):
      return False
    case (True, FoldState.FOLD_OUT):
      return True
    case (False, FoldState.FOLD_IN):
      return True
    case (False, FoldState.FOLD_OUT):
      return False
    case _:
      raise ValueError(
          f'Invalid parent first side is out {parent_first_side_is_up} and fold'
          f' state {fold_state}!'
      )


def _get_first_side_is_up_from_root_direction(
    root_stick_name: str,
    child_stick_name: str,
    stick_figure: sf.StickFigure,
    root_direction_follow_stick_figure: bool,
    fold_state: FoldState,
) -> bool:
  """Determines the first_side_is_up of a flap connected to the root stick."""
  root_stick = stick_figure[root_stick_name]
  child_stick = stick_figure[child_stick_name]
  connected_joint_name = (
      {root_stick.start, root_stick.end} & {child_stick.start, child_stick.end}
  ).pop()

  # Determine which joint is the "top" (UP) and "bottom" (DOWN).
  if root_direction_follow_stick_figure:
    top_joint_name = root_stick.end
    bot_joint_name = root_stick.start
  else:
    top_joint_name = root_stick.start
    bot_joint_name = root_stick.end

  # Determine direction based on connection point and fold state.
  if connected_joint_name == top_joint_name:
    if fold_state == FoldState.FOLD_IN:
      return False
    elif fold_state == FoldState.FOLD_OUT:
      return True
    else:
      raise ValueError(
          f'Invalid fold state {fold_state} for the top joint'
          f' {connected_joint_name}!'
      )
  elif connected_joint_name == bot_joint_name:
    if fold_state == FoldState.FOLD_IN:
      return False
    elif fold_state == FoldState.FOLD_OUT:
      return True
    else:
      raise ValueError(
          f'Invalid fold state {fold_state} for the bottom joint'
          f' {connected_joint_name}!'
      )
  else:
    raise ValueError(f'Invalid connected joint {connected_joint_name}!')


def _get_absolute_direction_from_root_direction(
    root_stick_name: str,
    child_stick_name: str,
    stick_figure: sf.StickFigure,
    root_direction_follow_stick_figure: bool,
    fold_state: FoldState,
) -> AbsFlapDirection:
  """Determines the absolute direction of a flap connected to the root stick."""
  root_stick = stick_figure[root_stick_name]
  child_stick = stick_figure[child_stick_name]
  connected_joint_name = (
      {root_stick.start, root_stick.end} & {child_stick.start, child_stick.end}
  ).pop()

  # Determine which joint is the "top" (UP) and "bottom" (DOWN).
  if root_direction_follow_stick_figure:
    top_joint_name = root_stick.end
    bot_joint_name = root_stick.start
  else:
    top_joint_name = root_stick.start
    bot_joint_name = root_stick.end

  # Determine direction based on connection point and fold state.
  if connected_joint_name == top_joint_name:
    if fold_state == FoldState.FOLD_IN:
      return AbsFlapDirection.DOWN
    elif fold_state == FoldState.FOLD_OUT:
      return AbsFlapDirection.UP
    else:
      raise ValueError(
          f'Invalid fold state {fold_state} for the top joint'
          f' {connected_joint_name}!'
      )
  elif connected_joint_name == bot_joint_name:
    if fold_state == FoldState.FOLD_IN:
      return AbsFlapDirection.UP
    elif fold_state == FoldState.FOLD_OUT:
      return AbsFlapDirection.DOWN
    else:
      raise ValueError(
          f'Invalid fold state {fold_state} for the bottom joint'
          f' {connected_joint_name}!'
      )
  else:
    raise ValueError(f'Invalid connected joint {connected_joint_name}!')


def build_sacrificial_flaps_by_base_flap(
    sacrificial_flaps: Collection[types.SacrificialFlap] | None,
) -> dict[str, list[str]]:
  mapping: dict[str, list[str]] = collections.defaultdict(list)
  if sacrificial_flaps:
    for sf_flap in sacrificial_flaps:
      mapping[sf_flap.shifted_element_name].append(sf_flap.polygon_name)
  return dict(mapping)


def get_flap_info_dict_with_hinge_info(
    stick_figure: sf.StickFigure,
    hinge_info_list: list[types.HingeInfo],
    root_direction_follow_stick_figure_root: bool,
    sacrificial_flaps_by_base_flap: Mapping[str, Collection[str]] | None = None,
    *,
    use_tree_root: bool = False,
) -> dict[str, Node]:
  """Gets the flap info in the stick figure with hinge info list input.

  Full logic restructured for clarity:
    1. Convert the stick figure to a tree of nodes using
      stick_figure.shaping_angles().
    2. Determine adjacencies and hinge info between nodes.
    3. Determine fold states (IN/OUT) iteratively until convergence, handling
    sibling interdependencies.
    4. Calculate absolute directions (UP/DOWN).

  Args:
    stick_figure: The StickFigure instance to process.
    hinge_info_list: A list of `HingeInfo` objects containing information about
      hinges between flaps.
    root_direction_follow_stick_figure_root: Whether the root stick's direction
      follows the stick figure's root direction.
    sacrificial_flaps_by_base_flap: A dictionary mapping base flap names to a
      list of sacrificial flap names.
    use_tree_root: Whether to anchor the node tree at `stick_figure.root`.

  Returns:
    A dictionary mapping stick names to their Node info.
  """
  root_node, node_dict = convert_stick_figure_to_node_dict(
      stick_figure,
      hinge_info_list,
      root_direction_follow_stick_figure_root,
      sacrificial_flaps_by_base_flap,
      use_tree_root=use_tree_root,
  )
  root_stick_name = root_node.name

  # Initialize root node state.
  node_dict[root_stick_name].fold_state = FoldState.REFERENCE
  node_dict[root_stick_name].absolute_direction = AbsFlapDirection.REFERENCE

  adj_dict: dict[str, dict[str, types.HingeInfo]] = (
      _get_hinge_info_dict_for_connected_nodes(
          node_dict, root_stick_name, hinge_info_list, stick_figure
      )
  )

  # Build parent map for easy lookup during iteration.
  parent_map = {root_stick_name: None}
  for parent_name, parent_node in node_dict.items():
    for child_name in parent_node.children:
      parent_map[child_name] = parent_name  # pyrefly: ignore[unsupported-operation]

  # Determine Fold States Iteratively.
  # Since fold states can be interdependent among siblings, we iterate until
  # convergence, bounded by the total number of nodes to guarantee termination.
  max_iterations = len(node_dict)
  for _ in range(max_iterations + 1):
    newly_determined = False
    # Iterate over all nodes. Order doesn't matter for convergence.
    for child_name, node in node_dict.items():
      if node.fold_state != FoldState.NOT_DETERMINED:
        continue

      parent_name = parent_map.get(child_name)
      if parent_name is None:
        continue  # Skip root node

      fold_state = FoldState.NOT_DETERMINED
      child_adjacencies = adj_dict.get(child_name, {})

      # Case 1: Check adjacency with the parent.
      if parent_name in child_adjacencies:
        has_hinge = child_adjacencies[parent_name].assigned
        # If hinge is assigned (used), fold IN. If not assigned (gap), fold OUT.
        fold_state = FoldState.FOLD_IN if has_hinge else FoldState.FOLD_OUT

      # Case 2: Check adjacency with siblings (or others adjacent in packing).
      else:
        for adj_flap_name, hinge_info in child_adjacencies.items():
          adj_flap = node_dict[adj_flap_name]
          if adj_flap.fold_state == FoldState.NOT_DETERMINED:
            continue

          # Apply the zigzag rule.
          has_hinge = hinge_info.assigned
          # siblings share same fold state if they share a hinge
          if has_hinge:
            fold_state = adj_flap.fold_state
          # Once determined, we stop checking other neighbors.
          # NOTE: maybe don't break here, but continue to check the
          # other adjacent flaps for consistency.

      # If the fold state is determined, update the node.
      if fold_state != FoldState.NOT_DETERMINED:
        node_dict[child_name].fold_state = fold_state
        newly_determined = True
      # Case 3: Handle flaps with relative angle 0 (e.g., shaping flaps).
      elif np.isclose(node.relative_azimuth_angle, 0.0):
        # Fallback based on parent hinge if available.
        hinge_info = child_adjacencies.get(parent_name)
        has_hinge = hinge_info is not None and hinge_info.assigned
        fold_state = FoldState.FOLD_IN if has_hinge else FoldState.FOLD_OUT
        node_dict[child_name].fold_state = fold_state
        newly_determined = True

    # If a full pass occurs without determining any new states, we have
    # converged (or are stuck).
    if not newly_determined:
      break

  # Assign remaining undetermined flaps as fold out
  # as they do not have a hinge path to their parent
  for node in node_dict.values():
    if node.fold_state == FoldState.NOT_DETERMINED:
      node.fold_state = FoldState.FOLD_OUT

  # Calculate Absolute Directions in BFS order.
  reliable_parents = {root_stick_name}
  for child_name, node in node_dict.items():
    if (
        node.fold_state == FoldState.REFERENCE
        or node.fold_state == FoldState.NOT_DETERMINED
    ):
      continue

    parent_name = parent_map.get(child_name)
    if parent_name not in reliable_parents:
      continue

    # Shaping flaps should not be parents.
    if stick_figure[child_name].shaping_flap:
      continue

    # Calculate absolute direction.
    fold_state = node.fold_state
    if parent_name == root_stick_name:
      absolute_direction = _get_absolute_direction_from_root_direction(
          root_stick_name,
          child_name,
          stick_figure,
          root_direction_follow_stick_figure_root,
          fold_state,
      )
    else:
      absolute_direction = _get_absolute_direction_from_non_root_parent_direction(
          node_dict[parent_name].absolute_direction,  # pyrefly: ignore[bad-index]
          fold_state,
      )

    node.absolute_direction = absolute_direction

    # Calculate first_side_is_up.
    if parent_name == root_stick_name:
      first_side_is_up = _get_first_side_is_up_from_root_direction(
          root_stick_name,
          child_name,
          stick_figure,
          root_direction_follow_stick_figure_root,
          fold_state,
      )
    else:
      first_side_is_up = _get_first_side_is_up_from_non_root_parent_direction(
          node_dict[parent_name].first_side_is_up,  # pyrefly: ignore[bad-index]
          fold_state,
      )
    node.first_side_is_up = first_side_is_up

    # Determine if this node is reliable for its children.
    if not np.isclose(node.relative_azimuth_angle, 0.0):
      reliable_parents.add(child_name)
  return node_dict


def get_hinge_info_list_from_packing_and_solving_output(
    packing: packing_output.PackingOutput,
    solver_result: hinge_utils.SolverResult,
) -> list[types.HingeInfo]:
  """Gets the hinge info list from packing and solving output."""
  hinge_info_list = copy_hinge_info_list(packing.hinges)
  hinge_info_list = update_hinge_info_from_solver_result(
      hinge_info_list, solver_result  # pyrefly: ignore[bad-argument-type]
  )
  return hinge_info_list


def get_flap_info_dict(
    stick_figure: sf.StickFigure,
    packing: packing_output.PackingOutput,
    solver_result: hinge_utils.SolverResult,
    root_direction_follow_stick_figure_root: bool,
) -> dict[str, Node]:
  """Gets the flap info in the stick figure."""
  hinge_info_list = copy_hinge_info_list(packing.hinges)
  hinge_info_list = update_hinge_info_from_solver_result(
      hinge_info_list, solver_result  # pyrefly: ignore[bad-argument-type]
  )
  sacrificial_flaps_by_base_flap = build_sacrificial_flaps_by_base_flap(
      packing.sacrificial_flaps
  )
  return get_flap_info_dict_with_hinge_info(
      stick_figure,
      hinge_info_list,
      root_direction_follow_stick_figure_root,
      sacrificial_flaps_by_base_flap,
  )


def check_validity(
    node_dict: dict[str, Node],
    unmerged_node_dict: dict[str, Node],
    augmented_merged_stick_names: dict[str, list[str]] | None = None,
):
  """Checks the validity of the node dictionary.

  This function checks whether the new unmerged node has been updated correctly
  from the merged node, and the whether the old unmerged node stays the same.

  Args:
    node_dict: The node dictionary from the merged stick figure.
    unmerged_node_dict: The node dictionary from the unmerged stick figure.
    augmented_merged_stick_names: Optional mapping from merged stick names to
      their unmerged split stick names.

  Raises:
    ValueError: If an unmerged non-root node has fold_state NOT_DETERMINED, or
      has absolute_direction NOT_DETERMINED when its merged stick is determined.
  """
  merged_by_unmerged: dict[str, str] = {}
  if augmented_merged_stick_names is not None:
    for merged_name, split_names in augmented_merged_stick_names.items():
      for split_name in split_names:
        merged_by_unmerged[split_name] = merged_name

  for key, node in unmerged_node_dict.items():
    if key not in node_dict:
      if not node.is_root and node.fold_state == FoldState.NOT_DETERMINED:
        raise ValueError(
            f'Unmerged non-root node {key!r} has fold_state NOT_DETERMINED.'
        )
      merged_name = merged_by_unmerged.get(key)
      merged_is_undetermined = (
          merged_name in node_dict
          and node_dict[merged_name].absolute_direction
          == AbsFlapDirection.NOT_DETERMINED
      )
      if (
          not node.is_root
          and not merged_is_undetermined
          and node.absolute_direction == AbsFlapDirection.NOT_DETERMINED
      ):
        raise ValueError(
            f'Unmerged non-root node {key!r} has absolute_direction'
            f' NOT_DETERMINED while its merged flap {merged_name!r} is'
            ' determined.'
        )


def _translate_sacrificial_flaps_to_unmerged(
    sacrificial_flaps_by_base_flap: Mapping[str, Sequence[str]],
    merged_stick_name_dict: dict[str, list[str]],
) -> dict[str, set[str]]:
  unmerged_mapping = collections.defaultdict(set)
  for base_flap_name, sac_flap_names in sacrificial_flaps_by_base_flap.items():
    for unmerged_name in merged_stick_name_dict.get(
        base_flap_name, [base_flap_name]
    ):
      unmerged_mapping[unmerged_name].update(sac_flap_names)
  return dict(unmerged_mapping)


def _augment_merged_stick_name_dict(
    merged_stick_name_dict: dict[str, list[str]],
    unmerged_stick_figure: sf.StickFigure,
) -> dict[str, list[str]]:
  """Augments the merged stick name dict with unsplit sticks."""
  augmented_merged_stick_name_dict = copy.copy(merged_stick_name_dict)
  unmerged_stick_names = set(
      [stick.name for stick in unmerged_stick_figure.children]
  )
  merged_stick_names = set([
      stick.name
      for stick in unmerged_stick_figure.merge_linear_chains()[0].children
  ])
  unsplit_stick_names = unmerged_stick_names.intersection(merged_stick_names)
  for stick_name in unsplit_stick_names:
    augmented_merged_stick_name_dict[stick_name] = [stick_name]
  return augmented_merged_stick_name_dict


def get_flap_info_dict_for_unmerged_stick_figure(
    unmerged_stick_figure: sf.StickFigure,
    merged_stick_name_dict: dict[str, list[str]],
    hinge_info_list: list[types.HingeInfo],
    node_dict: dict[str, Node],
    root_direction_follow_stick_figure_root: bool,
    sacrificial_flaps_by_base_flap: Mapping[str, Sequence[str]] | None = None,
    *,
    use_tree_root: bool = False,
) -> dict[str, Node]:
  """Gets the flap info in the unmerged stick figure.

  This idea is to use the fold states and absolute directions of the merged
  stick figure to determine the fold states and absolute directions of the
  unmerged stick figure.
  - unsplit sticks (one stick in both merged and unmerged tree) should have the
    same fold state and absolute direction.
  - split sticks (one stick in merged tree splits into multiple sticks in
    unmerged tree) will be updated so the first split stick has the same fold
    state and absolute direction as the original merged stick, and the remaining
    split sticks will be fold out and have the same absolute direction as the
    first split stick.

  Args:
    unmerged_stick_figure: The unmerged stick figure.
    merged_stick_name_dict: A dictionary mapping merged stick names to a list of
      split stick names.
    hinge_info_list: A list of `HingeInfo` objects containing information about
      hinges between flaps.
    node_dict: The node dictionary from the merged stick figure.
    root_direction_follow_stick_figure_root: Whether the root stick's direction
      follows the stick figure's root direction.
    sacrificial_flaps_by_base_flap: A mapping from base (merged) flap name to
      sacrificial flap names. Keys are automatically translated to unmerged
      names.
    use_tree_root: Whether to anchor the node tree at
      `unmerged_stick_figure.root`.

  Returns:
    The node dictionary for the unmerged stick figure.
  """
  if sacrificial_flaps_by_base_flap is None:
    unmerged_sac_flaps = None
  else:
    unmerged_sac_flaps = _translate_sacrificial_flaps_to_unmerged(
        sacrificial_flaps_by_base_flap, merged_stick_name_dict
    )
  unmerged_node_dict = get_flap_info_dict_with_hinge_info(
      unmerged_stick_figure,
      hinge_info_list,
      root_direction_follow_stick_figure_root,
      unmerged_sac_flaps,
      use_tree_root=use_tree_root,
  )

  # Add unsplit sticks to merged_stick_name_dict
  augmented_merged_stick_name_dict = _augment_merged_stick_name_dict(
      merged_stick_name_dict, unmerged_stick_figure
  )

  for (
      merged_stick_name,
      split_stick_names,
  ) in augmented_merged_stick_name_dict.items():
    # Set fields for first split stick
    first_split_stick_name = split_stick_names[0]
    unmerged_node_dict[first_split_stick_name].fold_state = node_dict[
        merged_stick_name
    ].fold_state
    unmerged_node_dict[first_split_stick_name].absolute_direction = node_dict[
        merged_stick_name
    ].absolute_direction
    unmerged_node_dict[first_split_stick_name].first_side_is_up = node_dict[
        merged_stick_name
    ].first_side_is_up
    if node_dict[merged_stick_name].is_root:
      unmerged_node_dict[first_split_stick_name].is_root = True

    # NOTE: handle the case where one of split sticks
    # is root better. Will not necessarily be the first split stick.
    # Iterate through remaining split sticks
    for ind, split_stick_name in enumerate(split_stick_names[1:]):
      unmerged_node_dict[split_stick_name].fold_state = FoldState.FOLD_OUT

      # Calculate absolute direction
      # Case 1: the first split stick is root
      if unmerged_node_dict[first_split_stick_name].is_root:
        if ind == 0:
          absolute_direction = _get_absolute_direction_from_root_direction(
              first_split_stick_name,
              split_stick_name,
              unmerged_stick_figure,
              root_direction_follow_stick_figure_root,
              FoldState.FOLD_OUT,
          )
        else:
          absolute_direction = unmerged_node_dict[
              split_stick_names[1:][ind - 1]
          ].absolute_direction
      # Case 2: the first split stick is not root
      else:
        absolute_direction = unmerged_node_dict[
            first_split_stick_name
        ].absolute_direction
      unmerged_node_dict[split_stick_name].absolute_direction = (
          absolute_direction
      )

      # Update first_side_is_up field
      unmerged_node_dict[split_stick_name].first_side_is_up = node_dict[
          merged_stick_name
      ].first_side_is_up

  # Check the validity of the unmerged node dictionary.
  check_validity(
      node_dict, unmerged_node_dict, augmented_merged_stick_name_dict
  )
  return unmerged_node_dict


def get_flap_info_dict_for_unmerged_stick_figure_from_packing_and_solving_output(
    unmerged_stick_figure: sf.StickFigure,
    merged_stick_name_dict: dict[str, list[str]],
    packing: packing_output.PackingOutput,
    solver_result: hinge_utils.SolverResult,
    root_direction_follow_stick_figure_root: bool,
    node_dict: dict[str, Node],
) -> dict[str, Node]:
  """Gets the flap info in the unmerged stick figure from packing and solving output."""
  hinge_info_list = copy_hinge_info_list(packing.hinges)
  hinge_info_list = update_hinge_info_from_solver_result(
      hinge_info_list, solver_result  # pyrefly: ignore[bad-argument-type]
  )
  sacrificial_flaps_by_base_flap = build_sacrificial_flaps_by_base_flap(
      packing.sacrificial_flaps
  )
  return get_flap_info_dict_for_unmerged_stick_figure(
      unmerged_stick_figure,
      merged_stick_name_dict,
      hinge_info_list,
      node_dict,
      root_direction_follow_stick_figure_root,
      sacrificial_flaps_by_base_flap,
  )


def compute_root_stick_direction_follow_stick_figure_root(
    stick_figure: sf.StickFigure,
    flaps: list[shaping_elements.ShapingFlap],
    rivers: list[shaping_elements.ShapingRiver],
    *,
    use_tree_root: bool = False,
) -> bool:
  """Computes if the root stick direction follows the stick figure's root direction.

  This is done by computing how the root's start and end joints are oriented on
  the paper. For river root's, this is equivalent to checking the y-values of
  the river's banks corresponding to root joints. If the on-paper joints are
  aligned with the global coordinate system, return True. Otherwise, return
  False.

  Args:
    stick_figure: The stick figure.
    flaps: The list of shaping flaps.
    rivers: The list of shaping rivers.
    use_tree_root: Whether to look up `stick_figure.root` directly.

  Returns:
    A boolean indicating if the root stick direction follows the stick figure's
    root direction.
  """
  # Step 1: Get the root element.
  root_stick = get_first_stick_for_shaping(
      stick_figure, use_tree_root=use_tree_root
  )
  root_element = shaping_elements.get_element_by_name(
      root_stick.name, flaps, rivers
  )

  # Step 2: determine root_direction_follow_stick_figure_root
  # from root's orientation on paper
  if isinstance(root_element, shaping_elements.ShapingFlap):
    # NOTE: Implement this for flaps.
    # For now, return True as default.
    return True
  assert isinstance(root_element, shaping_elements.ShapingRiver), (
      'Root element is neither a flap nor a river.'
      f' Root element type: {type(root_element)}'
  )
  # Get banks corresponding to each joint
  try:
    start_joint_bank_index = root_element.get_joint_bank_index(
        root_stick.start,
        stick_figure,
        flaps,
        rivers,
    )
  except ValueError as e:
    logging.warning(
        'Could not determine start joint bank index for river %s: %s',
        root_element.name,
        e,
    )
    # Defaulting to first bank.
    start_joint_bank_index = 'FIRST'
  end_joint_bank_index = (
      'SECOND' if start_joint_bank_index == 'FIRST' else 'FIRST'
  )
  start_joint_bank, end_joint_bank = (
      root_element.get_bank_from_index(bank_index)
      for bank_index in (start_joint_bank_index, end_joint_bank_index)
  )
  start_joint_bank_y = shaping_elements.avg_bank_y(start_joint_bank)
  end_joint_bank_y = shaping_elements.avg_bank_y(end_joint_bank)

  # Return True if start joint's y is greater than end joint's y.
  # Otherwise, return False.
  return start_joint_bank_y > end_joint_bank_y


def compute_root_stick_direction_follow_stick_figure_from_packing_output(
    stick_figure: sf.StickFigure,
    packing: packing_output.PackingOutput,
) -> bool:
  """Computes if the root stick direction follows the stick figure's root direction.

  Args:
    stick_figure: The stick figure.
    packing: The packing output.

  Returns:
    A boolean indicating if the root stick direction follows the stick figure's
    root direction.
  """
  assert packing.packing_cp is not None
  shaping_flaps, shaping_rivers = (
      shaping_elements.convert_packing_elements_to_shaping_elements(
          packing.flap_rectangles,  # pyrefly: ignore[bad-argument-type]
          packing.packing_rivers,  # pyrefly: ignore[bad-argument-type]
          packing_grid_size=packing.packing_cp.grid_size,
          solving_grid_size=packing.packing_cp.grid_size,
      )
  )
  return compute_root_stick_direction_follow_stick_figure_root(
      stick_figure, shaping_flaps, shaping_rivers
  )


def get_descendants_names(
    flap_info_dict: dict[str, Node],
    flap_name: str,
) -> list[str]:
  """Gets the descendants names of a given flap."""
  descendants_names = []
  node = flap_info_dict[flap_name]
  children = [flap_info_dict[child_name] for child_name in node.children]
  for child in children:
    descendants_names.append(child.name)
    descendants_names.extend(get_descendants_names(flap_info_dict, child.name))
  return descendants_names
