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

"""Stick figure class."""

import collections
from collections.abc import Callable, Mapping, Sequence
import dataclasses
import functools
import hashlib
import itertools
import json
import logging
import math
from typing import Any, Literal, Self, TypeAlias, TypedDict

import immutabledict
import networkx as nx
import numpy as np

from . import angle_utils
from . import stick as stick_lib

GridMode: TypeAlias = Literal['v1', 'v0_include_width', 'v2']
DEFAULT_TOLERANCE = 1e-3


def get_attached_flap_info(
    river: stick_lib.Stick, joints: Mapping[str, 'Joint']
) -> tuple[int, bool]:
  """Finds information about flaps attached to a river.

  Returns the maximum flap length and an indicator if only flaps are attached.

  Args:
    river: The river stick to check.
    joints: A mapping from joint names to Joint objects.

  Returns:
    A tuple containing:
      - The maximum length of any attached flap.
      - A boolean indicating if all sticks attached to the river's endpoints
        (excluding the river itself) are flaps.
  """
  max_flap_length = 0

  for joint_name in (river.start, river.end):
    joint = joints[joint_name]
    for stick in joint.sticks:
      if stick.name != river.name:
        if stick.is_flap:
          max_flap_length = max(max_flap_length, stick.length)
        else:
          return 0, False

  return max_flap_length, True


def get_longest_path_in_tree(joints_to_sticks: Mapping[str, 'Joint']) -> int:
  """Calculates the length of the longest path in a tree of sticks.

  This is also known as the diameter of the tree, weighted by stick length.

  Args:
    joints_to_sticks: A mapping from joint names to Joint objects.

  Returns:
    The sum of lengths of sticks along the longest path.
  """
  if not joints_to_sticks:
    return 0

  # Built directly here to avoid an import cycle with converters.py.
  graph = nx.Graph()
  stick_by_name = {}
  for joint in joints_to_sticks.values():
    for stick in joint.sticks:
      stick_by_name[stick.name] = stick

  for stick in stick_by_name.values():
    graph.add_edge(stick.start, stick.end, weight=stick.length)

  if not graph.nodes:
    return 0

  diameter = 0
  for component in nx.connected_components(graph):
    subgraph = graph.subgraph(component)
    component_diameter = nx.diameter(subgraph, weight='weight')
    if component_diameter > diameter:
      diameter = component_diameter

  return int(diameter)


def has_overlapping_sticks(
    stick_figure: 'StickFigure', abs_tol: float = DEFAULT_TOLERANCE
) -> bool:
  """Checks if any sticks at a joint are directly overlapping.

  Overlapping sticks are defined as two distinct sticks connected at the same
  joint that point in the same direction when viewed from the joint.

  Args:
    stick_figure: The stick figure to check.
    abs_tol: The absolute tolerance for comparing angles.

  Returns:
    Whether any overlapping sticks are found.
  """
  for joint_name, joint in stick_figure.joints.items():
    if joint.degree < 2:
      continue

    for stick1, stick2 in itertools.combinations(joint.sticks, 2):
      az1 = stick1.azimuth_angle_at_joint(joint_name)
      el1 = stick1.elevation_angle_at_joint(joint_name)
      az2 = stick2.azimuth_angle_at_joint(joint_name)
      el2 = stick2.elevation_angle_at_joint(joint_name)

      # Check for same direction (overlapping)
      if angle_utils.are_azimuth_angles_the_same(
          az1, az2, abs_tol=abs_tol
      ) and math.isclose(el1, el2, abs_tol=abs_tol):
        logging.warning(
            'Sticks %s and %s are overlapping at joint %s.',
            stick1.name,
            stick2.name,
            joint_name,
        )
        return True

  return False


def calculate_grid_size(
    stick_figure: 'StickFigure',
    sticks: list[stick_lib.Stick] | None = None,
    mode: GridMode = 'v2',
    use_longest_path: bool = True,
) -> int:
  """Calculate the required grid size to pack sticks for a list of sticks.

  Args:
    stick_figure: Stick figure to calculate grid size for.
    sticks: If provided, only calculate grid size for this subset of sticks. If
      None, use all sticks in stick_figure.
    mode: The grid size calculation mode. Can be one of: - 'v1': Sqrt(sum of
      lengths^2 / 2). This is also a lower bound, but does not use
      Cauchy-Schwartz. - 'v2': Lower bound based on circle packing theory. The
      grid size is estimated as sqrt(sum of areas), where area for 4 largest
      flaps is l^2, for rivers with flaps max(k*m, k^2) where k is river length
      and m is max flap length, for rivers with rivers k^2, and 2*l^2 for other
      sticks. This is the default mode. - 'v0_include_width': Sum of (lengths +
      width - 1) / 2. This accounts for the width of the stick. Here one unit of
      width, is assumed to be equivalent to a unit of length in the additional
      space required to pack it.
    use_longest_path: If True, grid size is max(mode calculation, longest path
      in tree).

  Returns:
    The calculated grid size as an integer.
  """
  joints = stick_figure.joints
  sticks_to_consider = sticks if sticks is not None else stick_figure.children

  if mode == 'v1':
    value_to_ceil = math.sqrt(
        sum(2.0 * stick.length**2 / 4.0 for stick in sticks_to_consider)
    )
  elif mode == 'v2':
    # This mode computes grid size based on stick types (flaps, rivers).
    # It requires joint information to determine connectivity.
    value_to_ceil = 0.0
    processed_sticks = set()

    # Rule 1: 4 largest flaps (packed as a quarter circle).
    flaps = stick_lib.get_flaps_sorted_by_length(sticks_to_consider)
    max_flaps = min(4, len(flaps))
    for flap in flaps[:max_flaps]:
      value_to_ceil += flap.length**2
      processed_sticks.add(flap.name)

    # Rule 2 & 3: Rivers.
    for river in stick_lib.get_rivers(sticks_to_consider):
      k = river.length
      max_flap_length, has_flaps = get_attached_flap_info(river, joints)
      if has_flaps:
        # River with flaps.
        value_to_ceil += max(k * max_flap_length, k**2)
      else:
        # River attached to rivers
        value_to_ceil += k**2
      processed_sticks.add(river.name)

    # Use the 'v1' formula as a default for any stick not covered by the
    # rules.
    for stick in sticks_to_consider:
      if stick.name not in processed_sticks:
        value_to_ceil += 2 * stick.length**2

    value_to_ceil = math.sqrt(value_to_ceil)
  elif mode == 'v0_include_width':
    value_to_ceil = sum(
        (stick.length + stick.width - 1) / 2.0 for stick in sticks_to_consider
    )
  else:
    raise ValueError(f'Unknown mode: {mode}')

  if use_longest_path:
    longest_path = get_longest_path_in_tree(joints)
    value_to_ceil = max(value_to_ceil, longest_path)
  return int(math.ceil(value_to_ceil))


def calculate_max_grid_size(stick_figure: 'StickFigure') -> int:
  """Calculate the maximum grid size allowed for packing a stick figure.

  Args:
    stick_figure: Stick figure to calculate grid size for.

  Returns:
    The maximum grid size allowed for packing.
  """
  return sum(stick.length for stick in stick_figure.children) * 2


@dataclasses.dataclass(frozen=True, kw_only=True)
class Joint:
  """Specifies an angle at a joint.

  Attributes:
    name: The name of the joint.
    sticks: A list of Stick objects connected at this joint, sorted by azimuth
      angle.
  """

  name: str
  sticks: list[stick_lib.Stick]

  def __post_init__(self):
    if not self.sticks:
      raise ValueError('Joint must have at least one stick.')

    self.sticks.sort(key=lambda stick: stick.azimuth_angle_at_joint(self.name))

  def get_required_grid_size(
      self, stick_figure: 'StickFigure', mode: GridMode = 'v2'
  ) -> int:
    """Get the required grid size to pack sticks in the joint."""
    return calculate_grid_size(
        stick_figure, sticks=self.sticks, mode=mode, use_longest_path=False
    )

  @property
  def degree(self) -> int:
    """Returns the degree of the joint."""
    return len(self.sticks)


def get_stick_connectivity(
    sticks: list[stick_lib.Stick],
) -> dict[str, list[str]]:
  """Returns the connectivity of the sticks."""
  assert all(stick.is_river for stick in sticks), 'All sticks must be rivers.'
  connectivity: dict[str, list[str]] = {stick.name: [] for stick in sticks}
  for stick_a, stick_b in itertools.permutations(sticks, 2):
    if stick_a.share_joint(stick_b):
      connectivity[stick_a.name].append(stick_b.name)
  return connectivity


@dataclasses.dataclass(kw_only=True)
class StickFigure:
  """Represents a stick figure.

  Attributes:
    name: The name of the stick figure (e.g., 'cat').
    category: The category the stick figure belongs to (e.g., 'animal').
    color_top: The rgb value for the color of the top side of the origami paper.
    color_bottom: The rgb value for the color of the bottom side of the origami
      paper.
    complexity: An integer indicating the complexity of the figure.
    root: The name of the root stick.
    children: A list of `Stick` objects that make up the figure.
    shaping_flaps_limit: If > 0, mark sticks on symmetry axis as shaping flaps
      if they are flaps and their length is less than this value.
    symmetry_plane_name: The name of the symmetry plane (e.g., 'azimuth',
      'elevation' or '' if not symmetric).
    symmetry_axis: A list of stick names on the axis of symmetry.
    is_symmetric_tree: Whether the stick figure has a symmetry axis.
    tree_id: A hash of the tree json.
    views: An optional list of 7 camera views (triplets of x, y, z coordinates
      for the camera eye) that best showcase the 3D structure and
      recognizability of the stick figure.
  """

  name: str
  category: str
  color_top: list[int] = dataclasses.field(default_factory=lambda: [0, 0, 0])
  color_bottom: list[int] = dataclasses.field(
      default_factory=lambda: [255, 255, 255]
  )
  complexity: int
  root: str
  children: list[stick_lib.Stick]
  views: Sequence[tuple[float, float, float]] = dataclasses.field(
      default_factory=lambda: ()
  )

  def __post_init__(self):
    # Legacy gemini generation option, not used in the current version.
    self.shaping_flaps_limit: int = 0

    if not self.children:
      raise ValueError('Stick figure must have at least one child.')

    if self.complexity not in (1, 2, 3, 4, 5):
      raise ValueError(
          f'Complexity {self.complexity} is not in [1, 2, 3, 4, 5].'
      )

    if self.root not in self._stick_by_name:
      raise ValueError(f'Root stick {self.root} not found in stick figure.')

    for stick in self.children:
      start_joint = self.joints[stick.start]
      end_joint = self.joints[stick.end]
      is_start_leaf = start_joint.degree == 1
      is_end_leaf = end_joint.degree == 1

      if is_start_leaf or is_end_leaf:
        stick.is_flap = True

      # A river requires at least three sticks at each joint.
      start_degree3 = start_joint and start_joint.degree >= 3
      end_degree3 = end_joint and end_joint.degree >= 3

      if start_degree3 and end_degree3:
        stick.is_river = True

    self._set_symmetry_properties(axis='azimuth')
    self._set_direction_properties(axis='azimuth')
    if not self._is_symmetric_tree and not self.is_planar():
      self._set_symmetry_properties(axis='elevation')
      self._set_direction_properties(axis='elevation')

    # Check that all shaping flaps are also flaps.
    for stick in self.children:
      if stick.shaping_flap and not stick.is_flap:
        raise ValueError(
            f'Stick {stick.name} is a shaping flap but not a flap.'
        )

  @functools.cached_property
  def _stick_by_name(self) -> immutabledict.immutabledict[str, stick_lib.Stick]:
    """A dictionary mapping stick names to Sticks."""
    stick_by_name = {}
    for stick in self.children:
      if stick.name in stick_by_name:
        raise ValueError(f'Duplicate stick name {stick.name}.')
      stick_by_name[stick.name] = stick
    return immutabledict.immutabledict(stick_by_name)

  @functools.cached_property
  def joints(self) -> immutabledict.immutabledict[str, Joint]:
    """A dictionary mapping joint names to Joints."""
    joints: dict[str, list[stick_lib.Stick]] = {}
    for stick in self.children:
      joints.setdefault(stick.start, []).append(stick)
      joints.setdefault(stick.end, []).append(stick)

    return immutabledict.immutabledict(
        (name, Joint(name=name, sticks=sticks))
        for name, sticks in joints.items()
    )

  @property
  def tree_id(self) -> str:
    tree_json = self.to_json()
    tree_hash = hashlib.md5(tree_json.encode()).hexdigest()[:16]
    return f'{self.name}_{tree_hash}'

  def __getitem__(self, key: str) -> stick_lib.Stick:
    """Get a stick child by name."""
    if key not in self._stick_by_name:
      raise KeyError(f'Stick {key} not found in stick figure.')
    return self._stick_by_name[key]

  def __contains__(self, key: str) -> bool:
    """Checks if a stick with the given name exists."""
    return key in self._stick_by_name

  @property
  def symmetry_axis(self) -> list[str]:
    return self._symmetry_axis

  @property
  def symmetry_joints(self) -> set[str]:
    """The names of all joints on the symmetry axis."""
    joint_names = set()
    for stick_name in self.symmetry_axis:
      stick = self[stick_name]
      joint_names.add(stick.start)
      joint_names.add(stick.end)
    return joint_names

  @property
  def symmetry_plane_name(self) -> angle_utils.Angles:
    return self._symmetry_plane_name

  @property
  def is_symmetric_tree(self) -> bool:
    return self._is_symmetric_tree

  def get_packing_gridsize(self) -> int:
    """Get the require grid size of the stick figure for packing."""
    grid_size = calculate_grid_size(self, use_longest_path=True)
    if self.is_symmetric_tree and grid_size % 2 == 1:
      grid_size += 1
    return grid_size

  def get_number_children(self) -> int:
    """Get the number of children in the stick figure."""
    return len(self.children)

  def get_number_joints(self) -> int:
    """Get the number of joints in the stick figure."""
    return len(self.joints)

  def is_planar(self) -> bool:
    """Checks if the stick figure is planar (all sticks on the XY plane)."""
    return all(math.isclose(s.elevation_angle, 0.0) for s in self.children)

  def to_dict(self, skip_defaults: bool = False) -> dict[str, Any]:
    """Returns a dictionary representation of the StickFigure object."""
    data = {}
    for field in dataclasses.fields(self):
      if not field.init:
        continue

      if field.name == 'children':
        value = [
            stick.to_dict(skip_defaults=skip_defaults)
            for stick in self.children
        ]
      else:
        value = getattr(self, field.name)

      if skip_defaults and stick_lib.is_default_value(field, value):
        continue

      data[field.name] = value
    return data

  def to_json(
      self, indent: int | None = None, skip_defaults: bool = False
  ) -> str:
    return json.dumps(self.to_dict(skip_defaults=skip_defaults), indent=indent)

  @classmethod
  def from_dict(cls, data: dict[str, Any]) -> 'StickFigure':
    """Creates a StickFigure object from a dictionary.

    Args:
      data: A dictionary representing the stick figure.

    Returns:
      A StickFigure object.
    """
    init_field_names = {f.name for f in dataclasses.fields(cls) if f.init}
    filtered_data = {k: v for k, v in data.items() if k in init_field_names}
    sticks_data = filtered_data.pop('children', [])
    children = [stick_lib.Stick.from_dict(s) for s in sticks_data]
    views = filtered_data.get('views', None)
    if views is not None:
      filtered_data['views'] = tuple(tuple(v) for v in views)
    return cls(**filtered_data, children=children)

  @classmethod
  def from_json(cls, json_str: str) -> 'StickFigure':
    """Creates a StickFigure object from a JSON string.

    Args:
      json_str: A JSON string representing the stick figure.

    Returns:
      A StickFigure object.
    """
    return cls.from_dict(json.loads(json_str))

  def _extend_chain(
      self,
      stick: stick_lib.Stick,
      degree2_joints: set[str],
      ignore_shaping_flaps: bool = False,
  ) -> collections.deque[stick_lib.Stick]:
    """Extend a linear chain from a stick."""
    if ignore_shaping_flaps and stick.shaping_flap:
      return collections.deque([stick])
    chain = collections.deque([stick])
    visited_in_chain = {stick.name}
    for joint_name, prepend in [(stick.start, True), (stick.end, False)]:
      curr = stick
      while joint_name in degree2_joints:
        s1, s2 = self.joints[joint_name].sticks
        next_stick = s2 if s1.name == curr.name else s1
        if next_stick.name in visited_in_chain:
          break
        if prepend:
          chain.appendleft(next_stick)
        else:
          chain.append(next_stick)
        visited_in_chain.add(next_stick.name)
        joint_name = next_stick.other_joint_name(joint_name)
        curr = next_stick

    return chain

  def merge_linear_chains(
      self, ignore_shaping_flaps: bool = True
  ) -> tuple['StickFigure', dict[str, list[str]]]:
    """Merges all linear chains of sticks into single sticks.

    Args:
      ignore_shaping_flaps: If True, degree-2 joints incident to shaping_flap
        sticks are preserved so shaping-flap template hinges remain unmerged.

    Returns:
      A new StickFigure with linear chains merged and a dictionary mapping
      the names of the merged sticks to the names of the sticks that were
      merged.
    """
    new_children = []
    visited_sticks = set()
    degree2_joints = {
        j_name
        for j_name, j in self.joints.items()
        if j.degree == 2
        and (
            not ignore_shaping_flaps
            or not any(s.shaping_flap for s in j.sticks)
        )
    }
    new_root = self.root

    merged_stick_name_dict = {}
    for stick in self.children:
      if stick.name in visited_sticks:
        continue

      # Build chain by extending from stick in both directions.
      chain = self._extend_chain(stick, degree2_joints, ignore_shaping_flaps)
      visited_sticks.update(s.name for s in chain)

      if len(chain) == 1:
        new_children.append(dataclasses.replace(stick))
        continue

      stick_names = [s.name for s in chain]
      new_stick_name = '_'.join(stick_names)
      merged_stick_name_dict[new_stick_name] = stick_names

      start_joint_name = (
          chain[0].start if chain[0].end in degree2_joints else chain[0].end
      )
      end_joint_name = (
          chain[-1].start if chain[-1].end in degree2_joints else chain[-1].end
      )

      if end_joint_name in degree2_joints:
        raise ValueError(f'Cycle detected involving sticks: {stick_names}')

      total_length = sum(s.length for s in chain)
      max_width = max(s.width for s in chain)

      x, y, z = 0.0, 0.0, 0.0
      current_joint = start_joint_name
      for stick in chain:
        dx, dy, dz = stick.get_coords(current_joint)
        x, y, z = x + dx, y + dy, z + dz
        current_joint = stick.other_joint_name(current_joint)

      new_azimuth_angle = math.degrees(math.atan2(-x, y))
      xy_distance = math.hypot(x, y)
      new_elevation_angle = math.degrees(math.atan2(z, xy_distance))
      new_stick = stick_lib.Stick(
          name=new_stick_name,
          start=start_joint_name,
          end=end_joint_name,
          azimuth_angle=new_azimuth_angle,
          elevation_angle=new_elevation_angle,
          length=total_length,
          width=max_width,
      )
      new_children.append(new_stick)

      if self.root in stick_names:
        new_root = new_stick_name

    return (
        dataclasses.replace(self, root=new_root, children=new_children),
        merged_stick_name_dict,
    )

  def simplify_lengths(self) -> Self:
    """Divides all stick lengths by their greatest common divisor."""
    divisor = math.gcd(*[stick.length for stick in self.children])

    if divisor <= 1:
      return self

    sticks = [
        dataclasses.replace(stick, length=stick.length // divisor)
        for stick in self.children
    ]
    return dataclasses.replace(self, children=sticks)

  def get_largest_joint_name(self) -> str:
    """Finds the name of the joint with the largest required grid size."""
    return max(
        self.joints.keys(),
        key=lambda j_name: self.joints[j_name].get_required_grid_size(self),
    )

  def get_tip(self, connectivity: dict[str, list[str]]) -> str | None:
    """Returns a tip of the connectivity dict."""
    if len(connectivity) == 1:
      return list(connectivity.keys())[0]
    for stick_name, neighbors in connectivity.items():
      shared_joint_names = [
          self[stick_name].shared_joint_name(self[neighbor])
          for neighbor in neighbors
      ]
      if len(shared_joint_names) == 1:
        return stick_name
    return None

  @functools.cached_property
  def rivers(self) -> list[stick_lib.Stick]:
    """Returns all the rivers in the stick figure."""
    return [s for s in self.children if s.is_river]

  def _set_direction_properties(self, axis: angle_utils.Angles):
    """Sets the up_at_start_joint property for all rivers. Flaps are down.

    We first identify a tip river across the main axis. Then we walk the
    connectivity of rivers. Every time a joint is reached for the first time,
    we check whether the river is pointing up or down.

    Args:
      axis: The axis to consider for symmetry ('azimuth' or 'elevation').
    """
    # Identify the main axis.
    if self.is_symmetric_tree:
      main_axis_names = self.symmetry_axis
      main_axis = [self[name] for name in main_axis_names]
    else:
      # If the tree is not symmetric, use the longest axis chain.
      candidate_axes = self._get_candidate_axes(axis)
      main_axis = max(candidate_axes, key=len)
    # Get all rivers on the main axis.
    rivers_on_main_axis = [stick for stick in main_axis if stick.is_river]
    rivers = rivers_on_main_axis if rivers_on_main_axis else self.rivers
    if len(rivers) < 2:
      # If there is less than two rivers, it does not matter.
      return
    # Identify the tip river on the main axis. There are only two and we pick
    # the first one.
    main_axis_river_connectivity = get_stick_connectivity(rivers)
    tip = self.get_tip(main_axis_river_connectivity)
    if tip is None:
      # If there is no tip river, it does not matter.
      return

    other_river_name = main_axis_river_connectivity[tip][0]
    joint_name = self[tip].none_shared_joint_name(self[other_river_name])
    active_sticks_and_joints = [(tip, joint_name)]
    visited_sticks = [tip]
    all_river_connectivity = get_stick_connectivity(self.rivers)
    # Walk the connectivity of rivers.
    while active_sticks_and_joints:
      curr_stick_name, joint_name = active_sticks_and_joints.pop()
      river_starts_at_joint = joint_name == self[curr_stick_name].start
      river_up_at_start = river_starts_at_joint
      self[curr_stick_name].up_at_start_joint = river_up_at_start
      river_up_at_joint = (
          river_up_at_start if river_starts_at_joint else not river_up_at_start
      )
      # After the first river is assigned, we assign the flaps.
      self.assign_sticks(joint_name, river_up_at_joint)
      # Continue with the other rivers.
      for neighbor_name in all_river_connectivity[curr_stick_name]:
        neighbor_shared_joint_name = self[curr_stick_name].shared_joint_name(
            self[neighbor_name]
        )[0]
        if neighbor_name not in visited_sticks:
          active_sticks_and_joints.append(
              (neighbor_name, neighbor_shared_joint_name)
          )
          visited_sticks.append(neighbor_name)

  def _set_symmetry_properties(self, axis: angle_utils.Angles) -> None:
    """Checks if the stick figure has an axis of symmetry.

    If the stick figure is symmetric, it updates the stick figure/sticks
    accordingly. If multiple axes exist, it returns the first axis it finds.

    Args:
      axis: The axis to consider for symmetry ('azimuth' or 'elevation').
    """
    self._symmetry_axis: list[str] = []
    self._symmetry_plane_name: angle_utils.Angles = 'azimuth'
    self._is_symmetric_tree: bool = False
    candidate_axes = self._get_candidate_axes(axis)

    for axis_chain in candidate_axes:
      (
          is_symmetric_result,
          symmetric_stick_mappings,
      ) = self._is_symmetric_around_axis(axis_chain, axis)
      if is_symmetric_result:
        axis_stick_names = {s.name for s in axis_chain}
        for stick in self.children:
          stick.is_on_symmetry_axis = stick.name in axis_stick_names
          if (
              stick.is_on_symmetry_axis
              and stick.is_flap
              and stick.length <= self.shaping_flaps_limit
          ):
            stick.shaping_flap = True
        if symmetric_stick_mappings is not None:
          for stick_name, symmetric_name in symmetric_stick_mappings.items():
            self[stick_name].symmetric_stick_name = symmetric_name
        self._is_symmetric_tree = True
        self._symmetry_axis = [s.name for s in axis_chain]
        self._symmetry_plane_name = axis
        return

  def _get_uniaxial_collinear_sticks(
      self,
      start_stick: stick_lib.Stick,
      visited_sticks: set[str],
      axis: angle_utils.Angles,
  ) -> tuple[list[stick_lib.Stick], set[str]]:
    """Finds all sticks that lie on the same line as start_stick.

    A line is defined by sticks connected at joints that are collinear.
    Collinearity is determined by the angles of the sticks at the joint.
    We check for collinearity on the axis of interest, not both axes.

    Args:
      start_stick: The stick to start finding the line from.
      visited_sticks: A set of stick names that have already been assigned to a
        line.
      axis: The axis to consider for collinearity ('azimuth' or 'elevation').

    Returns:
      A tuple containing:
        - A list of sticks that form the line.
        - A set of names of the sticks in the line.
    """
    if start_stick.name in visited_sticks:
      return [], set()

    line_sticks = []
    q = collections.deque([start_stick])
    visited_on_line = {start_stick.name}

    while q:
      current_stick = q.popleft()
      line_sticks.append(current_stick)

      for joint_name in (current_stick.start, current_stick.end):
        joint = self.joints[joint_name]
        get_angle = self._get_angle_fn(axis)

        current_angle = get_angle(current_stick, joint_name)
        for neighbor_stick in joint.sticks:
          if (
              neighbor_stick.name not in visited_on_line
              and neighbor_stick.name not in visited_sticks
          ):
            neighbor_angle = get_angle(neighbor_stick, joint_name)
            # For collinearity, angles pointing away from joint must be same or
            # opposite. For elevation angles, they must be opposites.
            if axis == 'azimuth':
              collinear = angle_utils.are_azimuth_angles_collinear(
                  current_angle, neighbor_angle
              )
            else:
              collinear = math.isclose(
                  current_angle, -neighbor_angle, abs_tol=1e-3
              )
            if collinear:
              q.append(neighbor_stick)
              visited_on_line.add(neighbor_stick.name)

    return line_sticks, visited_on_line

  def _get_candidate_axes(
      self, axis: angle_utils.Angles
  ) -> list[list[stick_lib.Stick]]:
    """Partitions sticks into straight line chains, potential symmetry axes.

    Args:
      axis: The axis to consider for collinearity ('azimuth' or 'elevation').

    Returns:
      A list of chains of sticks.
    """
    chains = []
    visited = set()
    for stick in self.children:
      if stick.name not in visited:
        chain_deque, current_visited = self._get_uniaxial_collinear_sticks(
            stick, visited, axis
        )
        chains.append(list(chain_deque))
        visited.update(current_visited)
    return chains

  def _find_mirrored_stick(
      self,
      source_stick: stick_lib.Stick,
      source_joint: str,
      candidate_sticks: list[stick_lib.Stick],
      candidate_joint: str,
      axis_angle: float,
      axis: angle_utils.Angles,
  ) -> stick_lib.Stick | None:
    """Finds a stick in a list that mirrors source_stick."""
    get_primary_angle = self._get_angle_fn(axis)
    get_secondary_angle = self._get_angle_fn(
        'elevation' if axis == 'azimuth' else 'azimuth'
    )
    if axis == 'azimuth':
      primary_comparator = angle_utils.are_azimuth_angles_the_same
      secondary_comparator = lambda a, b: math.isclose(a, b, abs_tol=1e-3)
    else:  # elevation
      primary_comparator = lambda a, b: math.isclose(a, b, abs_tol=1e-3)
      secondary_comparator = angle_utils.are_azimuth_angles_the_same

    refl_primary_angle = angle_utils.get_reflection_angle(
        get_primary_angle(source_stick, source_joint),
        axis_angle,
        axis,
    )
    source_secondary_angle = get_secondary_angle(source_stick, source_joint)

    for s in candidate_sticks:
      if (
          s.length == source_stick.length
          and primary_comparator(
              get_primary_angle(s, candidate_joint),
              refl_primary_angle,
          )
          and secondary_comparator(
              get_secondary_angle(s, candidate_joint),
              source_secondary_angle,
          )
      ):
        return s
    return None

  def _are_subtrees_mirrored(
      self,
      s1: stick_lib.Stick,
      s2: stick_lib.Stick,
      on_axis_joint_name: str,
      unaccounted: set[str],
      axis_joints: set[str],
      axis_angle: float,
      axis: angle_utils.Angles,
  ) -> tuple[bool, dict[str, str | None]]:
    """Checks if two subtrees rooted at s1 and s2 are mirrored.

    This method performs a breadth-first search simultaneously on two subtrees
    to check for mirror symmetry. The terms "parent" and "child" are used in the
    context of this traversal. A "parent" stick is the stick that was just
    traversed to reach the current joint. "Child" sticks are all other sticks
    connected to that joint, representing the next sticks to be explored.

    Args:
      s1: The root of the first subtree.
      s2: The root of the second subtree.
      on_axis_joint_name: The joint on the symmetry axis where s1 and s2
        connect.
      unaccounted: A set of unaccounted stick names.
      axis_joints: A set of joint names on the symmetry axis.
      axis_angle: The axis to consider for symmetry.
      axis: The axis to consider for symmetry ('azimuth' or 'elevation').

    Returns:
      A tuple of (is_mirrored, symmetric_mappings).
    """
    if s2.name not in unaccounted:
      return False, {}
    symmetric_mappings = {s1.name: s2.name, s2.name: s1.name}
    unaccounted.remove(s2.name)

    # Breadth-first search to traverse both subtrees simultaneously.
    # q contains tuples of (stick_a, parent_joint_a, stick_b, parent_joint_b)
    # where stick_a and stick_b are corresponding mirrored sticks.
    q = collections.deque([(s1, on_axis_joint_name, s2, on_axis_joint_name)])
    while q:
      stick_a, parent_joint_a, stick_b, parent_joint_b = q.popleft()

      # The joints at the other end of the sticks.
      joint_a_name = stick_a.other_joint_name(parent_joint_a)
      joint_b_name = stick_b.other_joint_name(parent_joint_b)

      if (joint_a_name in axis_joints) != (joint_b_name in axis_joints):
        return False, {}

      joint_a = self.joints[joint_a_name]
      joint_b = self.joints[joint_b_name]

      # Child sticks are all sticks connected to the joint except the parent.
      children_a = [s for s in joint_a.sticks if s.name != stick_a.name]
      children_b = [s for s in joint_b.sticks if s.name != stick_b.name]
      children_a_names = {s.name for s in children_a}
      children_b_names = {s.name for s in children_b}

      if (
          len(children_a) != len(children_b)
          or not children_a_names.issubset(unaccounted)
          or not children_b_names.issubset(unaccounted)
      ):
        return False, {}

      unmatched_children_b = list(children_b)
      for child_a in children_a:
        child_b = self._find_mirrored_stick(
            child_a,
            joint_a_name,
            unmatched_children_b,
            joint_b_name,
            axis_angle,
            axis,
        )
        if not child_b:
          return False, {}
        q.append((child_a, joint_a_name, child_b, joint_b_name))
        symmetric_mappings[child_a.name] = child_b.name
        symmetric_mappings[child_b.name] = child_a.name
        unmatched_children_b.remove(child_b)
      unaccounted -= children_a_names
      unaccounted -= children_b_names
    return True, symmetric_mappings  # pyrefly: ignore[bad-return]

  def _check_off_axis_stick(
      self,
      current_stick: stick_lib.Stick,
      unaccounted: set[str],
      axis_joints: set[str],
      axis_joints_to_sticks: dict[str, list[stick_lib.Stick]],
      axis: angle_utils.Angles,
  ) -> tuple[bool, dict[str, str | None]]:
    """Checks an off-axis stick for symmetry.

    The stick is either a flap on the axis, or the root of a subtree that must
    have a mirrored counterpart.

    Args:
      current_stick: The stick to check, connected to the axis of symmetry.
      unaccounted: A set of unaccounted stick names. This set is modified.
      axis_joints: A set of joint names on the symmetry axis.
      axis_joints_to_sticks: A dict mapping axis joint names to axis sticks.
      axis: The axis to consider for symmetry ('azimuth' or 'elevation').

    Returns:
      A tuple of (is_mirrored, symmetric_mappings).
    """
    candidate_joint = next(
        (
            j
            for j in (current_stick.start, current_stick.end)
            if j in axis_joints
        ),
        None,
    )
    if not candidate_joint:
      return False, {}

    candidate_axis = axis_joints_to_sticks.get(candidate_joint)
    if not candidate_axis:
      return False, {}
    get_angle = self._get_angle_fn(axis)
    axis_angle = get_angle(candidate_axis[0], candidate_joint)

    joint = self.joints[candidate_joint]
    candidate_sticks = [s for s in joint.sticks if s.name in unaccounted]
    mirror_stick = self._find_mirrored_stick(
        current_stick,
        candidate_joint,
        candidate_sticks,
        candidate_joint,
        axis_angle,
        axis,
    )

    if not mirror_stick:
      # If stick has no mirror, check if it is a shaping flap that can be
      # on axis of symmetry.
      # In this case it is adjusted to lie on the axis of symmetry.
      if (
          current_stick.is_flap
          and current_stick.length <= self.shaping_flaps_limit
      ):
        match axis:
          case 'azimuth':
            current_stick.azimuth_angle = axis_angle
          case 'elevation':
            current_stick.elevation_angle = axis_angle
        return True, {current_stick.name: None}
      return False, {}
    return self._are_subtrees_mirrored(
        current_stick,
        mirror_stick,
        candidate_joint,
        unaccounted,
        axis_joints,
        axis_angle,
        axis,
    )

  def _get_max_flap_angle_sort_key(
      self,
      joint: Joint,
      axis_joints: set[str],
      axis_angle: float,
  ) -> float:
    """Returns sort key from max angle of symmetric half flaps.

    This key is used to sort joints in get_symmetric_connecting_joints.
    It finds stick in joint_name which are flaps, not on symmetry axis,
    and in symmetric half (relative angle > 0), and returns max angle
    among them. If no such stick exists, it returns -inf.

    Args:
      joint: The joint to calculate the sort key for.
      axis_joints: A set of joint names on the symmetry axis.
      axis_angle: The primary angle of the symmetry axis.

    Returns:
      The maximum angle of flaps in the symmetric half, or -float('inf').
    """
    get_angle = self._get_angle_fn()

    max_angle = -float('inf')
    for stick in joint.sticks:
      if stick.is_flap and not stick.is_on_symmetry_axis:
        relative_angle = self._get_relative_angle_to_axis(
            stick, axis_joints, axis_angle
        )
        if relative_angle > DEFAULT_TOLERANCE:
          max_angle = max(max_angle, get_angle(stick, joint.name))
    return max_angle

  def get_symmetric_connecting_joints(self) -> list[str]:
    """Returns joints connecting symmetric halves or river chain endpoints.

    If the stick figure is not symmetric or has no symmetry axis,
    an empty list is returned.

    If rivers are present on the symmetry axis, it returns the two joints
    at the ends of the river chain on the symmetry axis.
    It assumes rivers on the symmetry axis form a single chain.

    If no rivers are on the symmetry axis, it returns joints on the
    symmetry axis that are connected to any off-axis sticks.

    If exactly two joints are returned, they are ordered such that
    the joint connected to stick in symmetric half with largest angle comes
    first.
    Otherwise, they are in alphabetical order.

    Returns:
      A list of joint names.
    """
    if not self.is_symmetric_tree or not self.symmetry_axis:
      return []

    axis_rivers = [
        stick
        for stick_name in self.symmetry_axis
        if (stick := self[stick_name]).is_river
    ]

    if axis_rivers:
      joint_river_degree = collections.Counter(
          itertools.chain.from_iterable(
              (river.start, river.end) for river in axis_rivers
          )
      )

      # Joints with river degree 1 are endpoints of river chain on symmetry
      # axis.
      result = [
          joint_name
          for joint_name, degree in joint_river_degree.items()
          if degree == 1
      ]
    else:
      # No rivers on symmetry axis: return joints connecting to off-axis sticks.
      axis_stick_names = set(self.symmetry_axis)
      axis_joints, _ = self._get_axis_joints_and_angle()

      result = []
      for joint_name in sorted(axis_joints):
        joint = self.joints[joint_name]
        if any(stick.name not in axis_stick_names for stick in joint.sticks):
          result.append(joint_name)

    if len(result) == 2:
      axis_joints_set, axis_angle = self._get_axis_joints_and_angle()

      result.sort(
          key=lambda joint_name: self._get_max_flap_angle_sort_key(
              self.joints[joint_name], axis_joints_set, axis_angle
          ),
          reverse=True,
      )
    else:
      result.sort()
    return result

  def _get_angle_fn(
      self, axis: angle_utils.Angles | None = None
  ) -> Callable[[stick_lib.Stick, str], float]:
    """Returns the angle function based on symmetry plane.

    If symmetry is across elevation, it returns elevation_angle_at_joint,
    otherwise it returns azimuth_angle_at_joint.

    Args:
      axis: The axis to return the angle function for. If None, it uses
        self.symmetry_plane_name.

    Returns:
      A callable that takes a stick and a joint name and returns an angle.
    """
    if axis is None:
      axis = self.symmetry_plane_name
    if axis == 'elevation':
      return lambda stick, joint: stick.elevation_angle_at_joint(joint)
    else:
      return lambda stick, joint: stick.azimuth_angle_at_joint(joint)

  def _get_period_for_symmetry(self) -> float:
    """Returns the angle period based on symmetry plane.

    Returns:
      90.0 if symmetry is across elevation, 360.0 otherwise.
    """
    if self.symmetry_plane_name == 'elevation':
      return angle_utils.ELEVATION_PERIOD
    else:
      return angle_utils.AZIMUTH_PERIOD

  def _get_axis_joints_and_angle(self) -> tuple[set[str], float]:
    """Returns joints on the symmetry axis and a representative angle for it.

    This method calculates a single representative angle for all sticks on the
    symmetry axis. The angle is determined by taking the minimum of all axis
    stick angles after wrapping them to [0, period/2].

    Returns:
      A tuple containing:
        - A set of joint names that lie on the symmetry axis.
        - A positive angle that all axis sticks lie on.
    """
    if not self.symmetry_axis:
      raise ValueError(
          f'Stick figure {self.name} is not symmetric, cannot get axis joints.'
      )

    if self.symmetry_plane_name == 'elevation':

      def get_canonical_angle(stick, joint):
        return abs(stick.elevation_angle_at_joint(joint))

    else:

      def get_canonical_angle(stick, joint):
        canonical_angle = stick.azimuth_angle_at_joint(joint) % (
            angle_utils.AZIMUTH_PERIOD // 2
        )
        if math.isclose(
            canonical_angle,
            angle_utils.AZIMUTH_PERIOD // 2,
            abs_tol=DEFAULT_TOLERANCE,
        ):
          canonical_angle = 0.0
        return canonical_angle

    axis_joints_to_sticks = collections.defaultdict(list)
    for stick_name in self.symmetry_axis:
      stick = self[stick_name]
      axis_joints_to_sticks[stick.start].append(stick)
      axis_joints_to_sticks[stick.end].append(stick)

    angles = []
    for joint_name, sticks in axis_joints_to_sticks.items():
      # Wrap to [0, 180] for azimuth and positive for elevation.
      angles.extend(get_canonical_angle(stick, joint_name) for stick in sticks)

    # Check that angles are close enough to be considered the same.
    angles.sort()
    for angle1, angle2 in itertools.pairwise(angles + [angles[0]]):
      assert math.isclose(
          angle1, angle2, abs_tol=DEFAULT_TOLERANCE
      ), f'Not all canonical symmetry axis angles are close: {angles}'

    return set(axis_joints_to_sticks.keys()), min(angles)

  def _get_relative_angle_to_axis(
      self, stick: stick_lib.Stick, axis_joints: set[str], axis_angle: float
  ) -> float:
    """Calculates stick angle relative to symmetry axis.

    The relative angle is the angle between the stick and the axis stick
    at the joint they share, wrapped to [-period/2, period/2).

    Args:
      stick: The stick to calculate the relative angle for.
      axis_joints: A set of joint names on the symmetry axis.
      axis_angle: The positive angle that all axis sticks lie on.

    Returns:
      The relative angle in degrees.
    """
    get_angle = self._get_angle_fn()
    angle_period = self._get_period_for_symmetry()

    joint = stick.start if stick.start in axis_joints else stick.end
    relative_angle = angle_utils.wrap_angle_centered(
        get_angle(stick, joint) - axis_angle, angle_period
    )
    return relative_angle

  def get_symmetric_half(self) -> 'StickFigure':
    """Returns a stick figure representing one symmetric half.

    Returns:
      A stick figure containing only sticks on one symmetric half.
    """
    if not self.is_symmetric_tree:
      raise ValueError(
          f'Stick figure {self.name} is not symmetric, cannot get half.'
      )

    half_stick_names = set(self.symmetry_axis)
    axis_joints, axis_angle = self._get_axis_joints_and_angle()

    # Sticks attached to the axis.
    stick_candidates = [
        stick
        for stick in self.children
        if not stick.is_on_symmetry_axis
        and any(
            joint_name in axis_joints for joint_name in (stick.start, stick.end)
        )
    ]

    # Populate queue with all sticks that are attached to the axis on one side
    # of the tree.
    q = collections.deque()
    for stick in stick_candidates:
      # Skip sticks that are already in the half or have a mirror in the half.
      mirror_stick_in_half = (
          stick.symmetric_stick_name
          and stick.symmetric_stick_name in half_stick_names
      )
      if (stick.name in half_stick_names) or mirror_stick_in_half:
        continue

      relative_angle = self._get_relative_angle_to_axis(
          stick, axis_joints, axis_angle
      )
      if self.symmetry_plane_name == 'elevation':
        is_on_axis = math.isclose(relative_angle, 0.0, abs_tol=1e-3)
      else:
        is_on_axis = angle_utils.are_azimuth_angles_collinear(
            relative_angle, 0.0, abs_tol=1e-3
        )

      # Select sticks with relative_angle > 0 to make sure we select one half.
      if relative_angle > DEFAULT_TOLERANCE or (
          is_on_axis and stick.symmetric_stick_name is None
      ):
        q.append(stick)
        half_stick_names.add(stick.name)

    # Run BFS on these sticks to find all connected sticks in this half.
    while q:
      curr = q.popleft()
      for joint_name in (curr.start, curr.end):
        joint = self.joints[joint_name]
        for neighbor in joint.sticks:
          if neighbor.name not in half_stick_names and (
              neighbor.symmetric_stick_name is None
              or neighbor.symmetric_stick_name not in half_stick_names
          ):
            half_stick_names.add(neighbor.name)
            q.append(neighbor)

    children = [
        dataclasses.replace(self[name]) for name in sorted(half_stick_names)
    ]
    new_root = self.root
    if new_root not in half_stick_names:
      new_root = self.symmetry_axis[0]

    sf = StickFigure(
        name=self.name,
        category=self.category,
        complexity=self.complexity,
        root=new_root,
        children=children,
    )
    for stick in sf.children:
      original_stick = self[stick.name]
      stick.is_on_symmetry_axis = original_stick.is_on_symmetry_axis
      stick.symmetric_stick_name = original_stick.symmetric_stick_name
      stick.is_river = original_stick.is_river
    return sf

  def straighten_symmetry_axis(self) -> 'StickFigure':
    """Returns a new StickFigure with axis sticks oriented consistently.

    When the symmetry axis is a straight line, the axis sticks may have
    inconsistent directions (e.g. one stick at azimuth=0 and the next at
    azimuth=180 with different elevations). This causes the 3D joint positions
    to "curl back" instead of progressing monotonically along the axis.

    This method walks the axis chain from one tip to the other and ensures
    every axis stick's azimuth_angle_at_joint(walking_joint) matches the
    forward direction established by the first stick. Sticks pointing backward
    have their start/end swapped (angles are unchanged since they are defined
    relative to start).

    Returns:
      A new StickFigure with straightened axis sticks.

    Raises:
      ValueError: If the stick figure has no symmetry axis.
      NotImplementedError: If the symmetry plane is elevation.
    """
    if not self.is_symmetric_tree or not self.symmetry_axis:
      raise ValueError(
          f'Stick figure {self.name} has no symmetry axis to straighten.'
      )

    if self.symmetry_plane_name == 'elevation':
      raise NotImplementedError(
          'straighten_symmetry_axis is not implemented for elevation symmetry.'
      )

    axis_sticks = [self[name] for name in self.symmetry_axis]

    joint_to_axis_sticks: dict[str, list[stick_lib.Stick]] = (
        collections.defaultdict(list)
    )
    for stick in axis_sticks:
      joint_to_axis_sticks[stick.start].append(stick)
      joint_to_axis_sticks[stick.end].append(stick)

    tip_joints = [
        j for j, sticks in joint_to_axis_sticks.items() if len(sticks) == 1
    ]
    if not tip_joints:
      return self

    current_joint = tip_joints[0]
    visited: set[str] = set()
    forward_azimuth: float | None = None
    names_to_swap: set[str] = set()

    for _ in range(len(axis_sticks)):
      next_stick = None
      for stick in joint_to_axis_sticks[current_joint]:
        if stick.name not in visited:
          next_stick = stick
          break
      if next_stick is None:
        break

      visited.add(next_stick.name)
      az_from_joint = next_stick.azimuth_angle_at_joint(current_joint)

      if forward_azimuth is None:
        forward_azimuth = az_from_joint
      else:
        is_backward = (
            math.cos(math.radians(az_from_joint - forward_azimuth)) < 0
        )
        if is_backward:
          names_to_swap.add(next_stick.name)

      current_joint = next_stick.other_joint_name(current_joint)

    if not names_to_swap:
      return self

    new_children = []
    for stick in self.children:
      if stick.name in names_to_swap:
        new_children.append(
            dataclasses.replace(stick, start=stick.end, end=stick.start)
        )
      else:
        new_children.append(dataclasses.replace(stick))

    return StickFigure(
        name=self.name,
        category=self.category,
        complexity=self.complexity,
        root=self.root,
        children=new_children,
        views=self.views,
    )

  def _is_symmetric_around_axis(
      self, axis_sticks: list[stick_lib.Stick], axis: angle_utils.Angles
  ) -> tuple[bool, dict[str, str | None] | None]:
    """Checks for symmetry around a given axis of sticks.

    Args:
      axis_sticks: A list of sticks that form the axis of symmetry.
      axis: The axis to consider for symmetry ('azimuth' or 'elevation').

    Returns:
      A tuple containing a boolean indicating whether the figure is symmetric,
      and
      a dictionary of symmetric mappings.
    """
    axis_stick_names = {s.name for s in axis_sticks}
    axis_joints_to_sticks = collections.defaultdict(list)
    for s in axis_sticks:
      axis_joints_to_sticks[s.start].append(s)
      axis_joints_to_sticks[s.end].append(s)
    axis_joints = set(axis_joints_to_sticks.keys())

    unaccounted = {
        s.name for s in self.children if s.name not in axis_stick_names
    }

    symmetric_stick = {s.name: None for s in axis_sticks}

    while unaccounted:
      stick_name = None
      for s_name in sorted(unaccounted):
        if any(
            j in axis_joints for j in (self[s_name].start, self[s_name].end)
        ):
          stick_name = s_name
          break
      # If no unaccounted stick is connected to the axis, the remaining sticks
      # form a disconnected component, so the figure is not symmetric.
      if stick_name is None:
        return False, None

      check_stick = self[stick_name]
      unaccounted.remove(stick_name)
      is_mirrored, new_mappings = self._check_off_axis_stick(
          check_stick,
          unaccounted,
          axis_joints,
          axis_joints_to_sticks,
          axis,
      )
      if not is_mirrored:
        return False, None
      symmetric_stick.update(new_mappings)  # pyrefly: ignore[no-matching-overload]

    return True, symmetric_stick  # pyrefly: ignore[bad-return]

  def align_root_stick_to_target_azimuth_elevation(
      self, target_azimuth: float = 0.0, target_elevation: float = 0.0
  ) -> 'StickFigure':
    """Rotates the figure to align the root stick.

    The entire figure is rotated so that the root is aligned with the given
    target azimuth and elevation angles.

    Args:
      target_azimuth: The desired azimuth angle in degrees for the stick.
      target_elevation: The desired elevation angle in degrees for the stick.

    Returns:
      A new, rotated StickFigure instance.
    """

    # 1. Get the starting vector of the source stick
    source_stick = self[self.root]
    vec_start = np.array(source_stick.get_coords(None))

    # 2. Calculate the target vector from the desired angles.
    # We create a temporary stick to reuse the conversion logic.
    temp_target_stick = dataclasses.replace(
        source_stick,
        azimuth_angle=target_azimuth,
        elevation_angle=target_elevation,
    )
    vec_target = np.array(temp_target_stick.get_coords(None))

    # 3. Get the required rotation matrix to transform the start to the target
    rotation_matrix = angle_utils.get_rotation_matrix_to_align(
        vec_start, vec_target
    )

    # 4. Apply this single rotation to all children in the figure
    new_children = [stick.rotate(rotation_matrix) for stick in self.children]

    # 5. Rotate any views (camera eye coordinates)
    new_views = tuple(tuple(rotation_matrix @ np.array(v)) for v in self.views)

    # Return a new StickFigure instance with the rotated children and views
    return dataclasses.replace(self, children=new_children, views=new_views)

  def subdivide_sticks(
      self, stick_lengths_dict: dict[str, list[int]]
  ) -> tuple['StickFigure', dict[str, list[str]]]:
    """Subdivides multiple sticks into multiple sticks of specified lengths.

    It creates new sticks with names '{stick_name}_0', '{stick_name}_1', ...
    and new joints with names '{stick_name}_j0', '{stick_name}_j1', ...
    The new sticks will inherit angles and width from the original stick.

    Args:
      stick_lengths_dict: A dictionary mapping stick names to a list of positive
        integers specifying the lengths of the new sticks. The sum of these
        lengths must be equal to the length of the stick being subdivided.

    Returns:
      A new StickFigure instance with the sticks subdivided and a dictionary
      mapping the original stick names to their new stick names in order.

    Raises:
      ValueError: If any stick_name is not found, if sum of any substick_lengths
        does not match stick length, or if any substick_lengths is empty or
        contains non-positive values.
    """
    for stick_name in stick_lengths_dict:
      if stick_name not in self._stick_by_name:
        raise ValueError(
            f'Stick {stick_name} not found in stick figure {self.name}.'
        )

    new_children = []
    any_change = False
    new_root = self.root
    new_stick_names_dict = {}

    for stick in self.children:
      stick_name = stick.name
      if stick_name not in stick_lengths_dict:
        new_children.append(stick)
        continue

      substick_lengths = stick_lengths_dict[stick_name]
      if sum(substick_lengths) != stick.length:
        raise ValueError(
            f'Sum of substick lengths {sum(substick_lengths)} does not match'
            f' stick length {stick.length}.'
        )

      if any(length <= 0 for length in substick_lengths):
        raise ValueError('Substick lengths must be positive.')

      if len(substick_lengths) == 1:
        new_children.append(stick)
        new_stick_names_dict[stick_name] = [stick_name]
      else:
        any_change = True
        new_sticks_list = []
        current_joint_name = stick.start
        num_substicks = len(substick_lengths)
        for i, length in enumerate(substick_lengths):
          if i < num_substicks - 1:
            next_joint_name = f'{stick_name}_j{i}'
          else:
            next_joint_name = stick.end

          new_stick = dataclasses.replace(
              stick,
              name=f'{stick_name}_{i}',
              start=current_joint_name,
              end=next_joint_name,
              length=length,
          )
          new_sticks_list.append(new_stick)
          current_joint_name = next_joint_name

        new_children.extend(new_sticks_list)
        new_stick_names_dict[stick_name] = [s.name for s in new_sticks_list]

        if stick_name == self.root:
          new_root = f'{stick_name}_0'

    if any_change:
      return (
          dataclasses.replace(self, children=new_children, root=new_root),
          new_stick_names_dict,
      )
    return self, new_stick_names_dict

  def update_tree(
      self,
      azimuth_updates: dict[str, float],
      elevation_updates: dict[str, float],
      width_updates: dict[str, float],
  ) -> 'StickFigure':
    """Updates the tree with the given updates to angles and widths.

    Args:
      azimuth_updates: A dictionary mapping element names to azimuth updates.
      elevation_updates: A dictionary mapping element names to elevation
        updates.
      width_updates: A dictionary mapping element names to width updates.

    Returns:
      The updated stick figure.
    """
    new_children = []
    any_change = False
    for stick in self.children:
      if stick.name in azimuth_updates:
        azimuth = angle_utils.wrap_angle_centered(
            azimuth_updates.get(stick.name), period=360  # pyrefly: ignore[bad-argument-type]
        )
      else:
        azimuth = stick.azimuth_angle
      if stick.name in elevation_updates:
        elevation = angle_utils.wrap_angle_centered(
            elevation_updates.get(stick.name), period=180  # pyrefly: ignore[bad-argument-type]
        )
      else:
        elevation = stick.elevation_angle
      width = width_updates.get(stick.name, stick.width)

      if (
          azimuth != stick.azimuth_angle
          or elevation != stick.elevation_angle
          or width != stick.width
      ):
        if width != stick.width and not any(
            math.isclose(width, v) for v in stick_lib.StickWidthValues
        ):
          raise ValueError(
              f'Width {width} is not supported for stick {stick.name}.'
          )

        any_change = True
        new_children.append(
            dataclasses.replace(
                stick,
                azimuth_angle=azimuth,
                elevation_angle=elevation,
                width=width,
            )
        )
      else:
        new_children.append(stick)

    if any_change:
      return dataclasses.replace(self, children=new_children)

    return self

  def assign_sticks(self, joint_name: str, river_up_at_joint: bool):
    """Assign `up_at_start_joint` for flap sticks.

    This method sets the `up_at_start_joint` property for flap sticks connected
    to the given `joint_name`, based on whether a single river is present
    at the joint and the orientation of that river.

    Args:
      joint_name: The name of the joint.
      river_up_at_joint: Whether the river connected at this joint is pointing
        "up" when viewed from this joint.
    """
    sticks_in_joint = [
        stick
        for stick in self.children
        if stick.start == joint_name or stick.end == joint_name
    ]
    has_single_river = sum([s.is_river for s in sticks_in_joint]) == 1

    # If there is only one river, the flaps should point in the other direction.
    # Otherwise, they should point down.

    for stick in sticks_in_joint:
      if stick.is_river:
        continue
      if has_single_river:
        up_at_start_joint = not river_up_at_joint
      else:
        up_at_start_joint = False
      # If the stick is not starting at the joint we flip the orientation.
      if stick.end == joint_name:
        up_at_start_joint = not up_at_start_joint
      stick.up_at_start_joint = up_at_start_joint


class TreeUpdate(TypedDict, total=False):
  """A dictionary representing an update to a stick figure tree.

  Attributes:
    element_name: The name of the stick to update.
    depth: The depth at which to split the stick (0 to 1).
    azimuth: The azimuth angle to update to.
    elevation: The elevation angle to update to.
    width: The width to update to.
  """

  element_name: str
  depth: float
  azimuth: float
  elevation: float
  width: float


def compute_merge_converter_for_modified_tree(
    base_tree: StickFigure,
    modified_tree: StickFigure,
) -> dict[str, str]:
  """Computes a merge converter dict.

  Maps names of merged sticks in original (unmerged) tree
  to names of merged sticks in the modified tree (after stick splitting).
  NOTE: assumes merged joints of base_tree and modified_tree are the same.

  Args:
    base_tree: The stick figure before splitting.
    modified_tree: The stick figure after splitting.

  Returns:
    A dictionary mapping names of all merged sticks in the modified tree
     to names of sticks in the original merged tree.
  """
  base_merged_tree, _ = base_tree.merge_linear_chains()
  modified_merged_tree, _ = modified_tree.merge_linear_chains()
  if len(base_merged_tree.children) != len(modified_merged_tree.children):
    raise ValueError(
        'Number of merged sticks in base tree and modified tree are different.'
    )

  # Iterate through all sticks in base tree and find corresponding stick
  # in modified tree with the same joints.
  merge_converter = dict()
  base_merged_sticks = base_merged_tree.children
  modified_merged_sticks = modified_merged_tree.children
  modified_merged_stick_by_joints = {
      tuple(sorted((stick.start, stick.end))): stick
      for stick in modified_merged_sticks
  }
  for base_stick in base_merged_sticks:
    # Find corresponding stick in modified tree with same joints.
    base_stick_joints = tuple(sorted((base_stick.start, base_stick.end)))
    modified_stick = modified_merged_stick_by_joints[base_stick_joints]
    merge_converter[modified_stick.name] = base_stick.name

  return merge_converter


def get_int_depth_from_frac_depth(
    frac_depth: float,
    stick: stick_lib.Stick,
    min_depth: int | None = None,
    max_depth: int | None = None,
) -> int:
  """Returns the integer depth from the fractional depth."""
  if frac_depth < 0 or frac_depth > 1:
    raise ValueError(f'Invalid frac_depth: {frac_depth}')
  if min_depth is None:
    min_depth = 0
  if max_depth is None:
    max_depth = stick.length

  abs_depth = min(max_depth, max(min_depth, round(frac_depth * stick.length)))
  return abs_depth


def split_by_frac_depth_and_update_tree(
    tree: StickFigure,
    updates: Sequence[TreeUpdate],
) -> tuple[StickFigure, list[str], dict[str, str]]:
  """Splits and updates the tree based on the provided updates.

  Args:
    tree: The stick figure to update.
    updates: A sequence of TreeUpdate dictionaries.

  Returns:
    A tuple containing:
      - The updated stick figure.
      - A list of names of the modified sticks (after splitting).
      - A dictionary mapping names of merged sticks in the modified tree to
        names of merged sticks in the original merged tree.
  """
  # First convert the fractional depths to int depths and split the sticks.
  depth_dict = collections.defaultdict(list)
  for update in updates:
    element_name = update['element_name']
    if element_name not in tree:
      logging.warning('Element %s not found in tree.', element_name)
      continue
    if 'depth' not in update:
      continue

    stick = tree[element_name]
    # NOTE: we restrict the depth to be at most stick.length - 1 as otherwise
    # the stick would be too short.
    abs_depth = get_int_depth_from_frac_depth(
        update['depth'], stick, max_depth=stick.length - 1
    )
    if abs_depth > 0:
      depth_dict[element_name].append(abs_depth)

  lengths_dict = {}
  for name, depths in depth_dict.items():
    stick_length = tree[name].length
    split_points = sorted(list(set(depths)))
    substick_lengths = []
    current_pos = 0
    for point in split_points:
      substick_lengths.append(point - current_pos)
      current_pos = point
    substick_lengths.append(stick_length - current_pos)
    lengths_dict[name] = substick_lengths

  for name in depth_dict:
    depth_dict[name] = sorted(list(set(depth_dict[name])))

  modified_tree, split_stick_names_dict = tree.subdivide_sticks(lengths_dict)

  split_stick_dict = {}
  for name, stick_names in split_stick_names_dict.items():
    lengths = lengths_dict[name]
    depth = 0
    for i, length in enumerate(lengths):
      for _ in range(length):
        split_stick_dict[(name, depth)] = stick_names[i]
        depth += 1

  # Now update the split tree with the provided updates.
  azimuth_updates = {}
  elevation_updates = {}
  width_updates = {}
  element_names = []

  for update in updates:
    element_name = update['element_name']
    if element_name not in tree:
      logging.warning('Element %s not found in tree.', element_name)
      continue

    stick = tree[element_name]
    abs_depth = get_int_depth_from_frac_depth(
        update.get('depth', 0.0), stick, max_depth=stick.length - 1
    )
    if element_name in split_stick_names_dict:
      target_name = split_stick_dict[(element_name, abs_depth)]
    else:
      target_name = element_name
    element_names.append(target_name)

    if 'azimuth' in update:
      azimuth_updates[target_name] = update['azimuth']
    if 'elevation' in update:
      elevation_updates[target_name] = update['elevation']
    if 'width' in update and update['width'] > 0:
      width_updates[target_name] = update['width']

  updated_tree = modified_tree.update_tree(
      azimuth_updates,
      elevation_updates,
      width_updates,
  )

  # Finally compute the merge converter for the modified tree.
  merge_converter = compute_merge_converter_for_modified_tree(
      tree, updated_tree
  )

  return updated_tree, element_names, merge_converter
