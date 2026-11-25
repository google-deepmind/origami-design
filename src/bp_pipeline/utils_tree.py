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

"""Utility functions for working with trees, constraints, and shared helpers."""

import dataclasses
import typing
from typing import Any

import networkx as nx

from .crease_pattern import crease
from .packing import config as config_lib
from .packing import types
from .packing import utils_packing_flaps
from .stick_figures import packing_utils
from .stick_figures import stick as stick_lib
from .stick_figures import stick_figure
from .stick_figures import traversal

if typing.TYPE_CHECKING:
  from . import bps_types
  from .packing import utils_tracker


def deduplicate(creases: list[crease.Crease]) -> list[crease.Crease]:
  """Removes duplicate creases using CreaseWithExactMatch equality."""
  return [
      c.crease
      for c in crease.Creases.create_from_creases(creases=creases).creases
  ]


def is_stick_up(
    stick: Any, joint: str, up_down_method: str = 'flaps_down'
) -> bool:
  """Returns the up/down orientation of the stick at the given joint."""
  match up_down_method:
    case 'azimuth':
      azimuth_angle = stick.azimuth_angle_at_joint(joint) % 360
      if stick.is_river and (azimuth_angle == 90 or azimuth_angle == 270):
        return joint == stick.start
      return (0 <= azimuth_angle <= 90) or (270 <= azimuth_angle <= 360)
    case 'flaps_down':
      if joint == stick.start:
        return stick.up_at_start_joint
      elif joint == stick.end:
        return not stick.up_at_start_joint
      else:
        raise ValueError(f'Joint {joint} is not part of stick {stick.name}.')
    case _:
      raise ValueError(f'Unknown up/down method: {up_down_method}.')


@dataclasses.dataclass(frozen=False)
class FoldGraphData:
  """A view of the graph-specific fields from a FOLD dataset."""

  vertices_coords: list[list[float]]
  edges_vertices: list[types.Edge]

  @classmethod
  def from_dict(cls, fold_data: types.FoldGraphDataDict) -> FoldGraphData:
    """Factory method to create an instance from a raw FOLD dictionary."""
    edges_as_tuples = [
        (types.VertexIndex(u), types.VertexIndex(v))
        for u, v in fold_data.get('edges_vertices', [])
    ]
    return cls(
        vertices_coords=fold_data.get('vertices_coords', []),
        edges_vertices=edges_as_tuples,
    )


def _calculate_area_for_river_side(
    sticks: list[stick_lib.Stick],
    packed_polygons: set[str],
    flap_to_pocket: dict[str, packing_utils.FlapPocket],
    river_indices_in_traversal: dict[str, int],
    current_river_name: str,
    use_level_shifter: bool = False,
) -> float:
  """Calculates the area requirement for one side of a river."""
  area = 0.0
  for stick in sticks:
    if stick.is_river:
      continue
    if stick.name in packed_polygons:
      continue
    flap_pocket = flap_to_pocket.get(stick.name)
    is_packed_intermediately = False
    if flap_pocket:
      boundary_rivers = flap_pocket['boundary_rivers']
      if any(
          river_indices_in_traversal.get(br, float('inf'))
          < river_indices_in_traversal[current_river_name]
          for br in boundary_rivers
          if br != current_river_name
      ):
        is_packed_intermediately = True
    if not is_packed_intermediately:
      if use_level_shifter and stick.requires_level_shifter:
        margin = utils_packing_flaps.level_shifter_center_margin(
            round(stick.width)
        )
        eff_len = stick.length + margin
        area += eff_len**2
      else:
        area += stick.length**2
  return area


def compute_constraints_along_traversal(
    tree: stick_figure.StickFigure,
    config: config_lib.PackingConfig | None = None,
    tracker: utils_tracker.CreasePatternTracker | None = None,
    pockets_info: list[packing_utils.FlapPocket] | None = None,
) -> list[types.TraversalStep]:
  """Traverses the tree and computes active constraints at each step."""
  if config is None:
    config = config_lib.PackingConfig()
  traversal_order = list(
      traversal.packing_traversal(
          tree,
          start_stick_name=config.start_stick_name,
          mode=config.traversal_mode,
      )
  )
  if not traversal_order:
    return []

  all_constraints = compute_edge_constraints(tree)
  if not all_constraints:
    raise ValueError('No constraints found.')

  if pockets_info is None:
    pockets_info = packing_utils.precompute_flap_pockets(tree, config=config)
  river_indices_in_traversal = {
      stick.name: i for i, stick in enumerate(traversal_order) if stick.is_river
  }
  flap_to_pocket = {}
  for pocket in pockets_info:
    for flap in pocket['flaps']:
      flap_to_pocket[flap.name] = pocket

  results: list[types.TraversalStep] = []
  traversed_sticks_so_far: list[stick_lib.Stick] = []
  visited_sticks_in_traversal: set[str] = set()
  sticks_traversed_until_previous_river: set[str] = set()

  for current_stick in traversal_order:
    constraints = {}
    if config.enforce_proximity_constraints:
      for other_stick in traversal_order:
        if other_stick.name == current_stick.name:
          continue
        if other_stick.name in all_constraints[current_stick.name]:
          constraint = all_constraints[current_stick.name][other_stick.name]
          constraints[other_stick.name] = constraint
    else:
      for prev_stick in traversed_sticks_so_far:
        constraint = all_constraints.get(current_stick.name, {}).get(
            prev_stick.name
        )
        if constraint and not (
            constraint['type'] == types.ConstraintType.EQUAL
            and constraint['value'] == 0
        ):
          constraints[prev_stick.name] = constraint
    step: types.TraversalStep = {
        'stick': current_stick,
        'constraints': constraints,
    }

    if current_stick.is_river:
      sticks1 = traversal.get_sticks_from_river_side(
          tree,
          current_stick.start,
          current_stick,
          sticks_traversed_until_previous_river,
      )
      sticks2 = traversal.get_sticks_from_river_side(
          tree,
          current_stick.end,
          current_stick,
          sticks_traversed_until_previous_river,
      )
      packed_polygons = set()
      if tracker:
        packed_polygons = set(tracker.packed_polygons.keys())
      sum1 = _calculate_area_for_river_side(
          sticks1,
          packed_polygons,
          flap_to_pocket,
          river_indices_in_traversal,
          current_stick.name,
          use_level_shifter=config.use_level_shifter,
      )
      sum2 = _calculate_area_for_river_side(
          sticks2,
          packed_polygons,
          flap_to_pocket,
          river_indices_in_traversal,
          current_stick.name,
          use_level_shifter=config.use_level_shifter,
      )
      step['area_sides'] = {current_stick.start: sum1, current_stick.end: sum2}

      sticks_traversed_until_previous_river = visited_sticks_in_traversal.copy()
      sticks_traversed_until_previous_river.add(current_stick.name)

    results.append(step)
    traversed_sticks_so_far.append(current_stick)
    visited_sticks_in_traversal.add(current_stick.name)

  return results


def convert_to_fold(
    tree_dict: bps_types.BoxPleatingTreeData,
) -> types.FoldGraphDataDict:
  """Converts a JSON object from the Box Pleating Studio format to a simplified FOLD format."""
  if tree_dict is None:
    raise ValueError('Example has no tree_data to convert.')

  try:
    design = tree_dict['design']
    tree = design['tree']
  except KeyError as e:
    raise ValueError('Failed to get nodes') from e

  nodes = tree.get('nodes', [])
  edges = tree.get('edges', [])

  vertices_coords: list[list[float]] = []
  node_id_to_vertex_index: dict[int, int] = {}

  nodes.sort(key=lambda n: n['id'])

  for i, node in enumerate(nodes):
    vertices_coords.append([node['x'], node['y']])
    node_id_to_vertex_index[node['id']] = i

  edges_vertices: list[list[int]] = []
  edge_lengths: list[float | None] = []

  for edge in edges:
    n1_id = edge['n1']
    n2_id = edge['n2']
    v1_index = node_id_to_vertex_index[n1_id]
    v2_index = node_id_to_vertex_index[n2_id]
    edges_vertices.append([v1_index, v2_index])
    edge_lengths.append(edge.get('length'))

  fold_data: types.FoldGraphDataDict = {
      'file_creator': 'Box Pleating Studio Converter',
      'file_title': design.get('title', 'Untitled'),
      'file_description': design.get('description', ''),
      'vertices_coords': vertices_coords,
      'edges_vertices': edges_vertices,
      'edge_lengths': edge_lengths,
  }

  return fold_data


def _precompute_adjacent_pairs(
    tree: stick_figure.StickFigure,
) -> set[frozenset[str]]:
  """Pre-computes a set of adjacent edge pairs from the vertices_angles data."""
  adjacent_edge_pairs = set()

  for joint in tree.joints.values():
    num_sticks = len(joint.sticks)

    for i in range(num_sticks):
      stick1 = joint.sticks[i]
      stick2 = joint.sticks[(i + 1) % num_sticks]
      adjacent_edge_pairs.add(frozenset((stick1.name, stick2.name)))

  return adjacent_edge_pairs


def _compute_coincident_constraint(
    edge_name1: str, edge_name2: str, adjacent_pairs: set[frozenset[str]]
) -> types.EdgeConstraint:
  """Computes the constraint for edges that share a vertex."""
  if frozenset((edge_name1, edge_name2)) in adjacent_pairs:
    return {'type': types.ConstraintType.EQUAL, 'value': 0.0}
  else:
    return {'type': types.ConstraintType.UNSPECIFIED, 'value': 'NA'}


def _compute_non_adjacent_constraint(
    graph: nx.Graph, edge1: stick_lib.Stick, edge2: stick_lib.Stick
) -> types.EdgeConstraint:
  """Computes the constraint for a pair of non-adjacent edges."""
  u1, v1 = edge1.start, edge1.end
  u2, v2 = edge2.start, edge2.end
  path_lengths = [
      nx.shortest_path_length(graph, u1, u2, weight='length'),
      nx.shortest_path_length(graph, u1, v2, weight='length'),
      nx.shortest_path_length(graph, v1, u2, weight='length'),
      nx.shortest_path_length(graph, v1, v2, weight='length'),
  ]
  min_path_dist = float(min(path_lengths))
  return {'type': types.ConstraintType.GREATER_THAN, 'value': min_path_dist}


def compute_edge_constraints(
    tree: stick_figure.StickFigure,
) -> dict[str, dict[str, types.EdgeConstraint]]:
  """Computes distance constraints between all unique pairs of edges in a tree."""
  edges = tree.children
  num_edges = len(edges)

  if num_edges < 2:
    raise ValueError(
        'The "edges_vertices" and "edge_lengths" keys are required to compute'
        ' edge constraints. Ensure that the number of edges is at least 2.'
    )

  adjacent_edge_pairs = _precompute_adjacent_pairs(tree)
  graph = nx.Graph()
  for stick in tree.children:
    graph.add_edge(stick.start, stick.end, length=stick.length, stick=stick)

  constraints: dict[str, dict[str, types.EdgeConstraint]] = {
      stick.name: {} for stick in edges
  }

  for i in range(num_edges):
    for j in range(i + 1, num_edges):
      edge1 = edges[i]
      edge2 = edges[j]
      name1, name2 = edge1.name, edge2.name

      joints1 = {edge1.start, edge1.end}
      joints2 = {edge2.start, edge2.end}

      if not joints1.isdisjoint(joints2):
        constraint = _compute_coincident_constraint(
            name1, name2, adjacent_edge_pairs
        )
      else:
        constraint = _compute_non_adjacent_constraint(graph, edge1, edge2)

      constraints[name1][name2] = constraint
      constraints[name2][name1] = constraint

  return constraints


def _compare_joints(
    tree1: stick_figure.StickFigure,
    tree2: stick_figure.StickFigure,
) -> list[str]:
  """Checks if two trees have the same number and types of sticks at each joint."""
  errors = []
  joints1 = tree1.joints
  joints2 = tree2.joints
  if len(joints1) != len(joints2):
    errors.append(
        f'Different number of joints: {len(joints1)} != {len(joints2)}'
    )
  for joint_name in joints1:
    if joint_name not in joints2:
      errors.append(f'Joint {joint_name} not found in tree2')
      continue
    if len(joints1[joint_name].sticks) != len(joints2[joint_name].sticks):
      errors.append(
          f'Different number of sticks at joint {joint_name}: '
          f'{len(joints1[joint_name].sticks)} != '
          f'{len(joints2[joint_name].sticks)}'
      )
    for stick1 in joints1[joint_name].sticks:
      joint2_stick_map = {
          stick.name: stick for stick in joints2[joint_name].sticks
      }
      stick2 = joint2_stick_map.get(stick1.name, None)
      if stick2 is None:
        errors.append(
            f'Stick {stick1.name} not found at joint {joint_name} in tree2'
        )
        continue
      if stick1.length != stick2.length:
        errors.append(
            f'Different length for stick {stick1.name} at joint {joint_name}: '
            f'{stick1.length} != {stick2.length}'
        )
  return errors


def _compare_traversal_order(
    tree1: stick_figure.StickFigure,
    tree2: stick_figure.StickFigure,
    config: config_lib.PackingConfig,
) -> list[str]:
  """Checks if two trees have the same packing traversal order."""
  errors = []
  traversal_1 = compute_constraints_along_traversal(tree1, config)
  traversal_2 = compute_constraints_along_traversal(tree2, config)
  if len(traversal_1) != len(traversal_2):
    errors.append(
        f'Different number of steps in traversal: {len(traversal_1)} !='
        f' {len(traversal_2)}'
    )
    return errors
  for step1, step2 in zip(traversal_1, traversal_2):
    if step1['stick'].name != step2['stick'].name:
      errors.append(
          f"Different stick at traversal step: {step1['stick'].name} !="
          f" {step2['stick'].name}"
      )
  return errors


def get_trees_are_packing_equivalent_errors(
    tree1: stick_figure.StickFigure,
    tree2: stick_figure.StickFigure,
    config: config_lib.PackingConfig | None = None,
    check_traversal_order: bool = True,
) -> list[str]:
  """Checks if two trees are equivalent in the packing process."""
  if config is None:
    config = config_lib.PackingConfig()
  errors = _compare_joints(tree1, tree2)
  if check_traversal_order:
    errors.extend(_compare_traversal_order(tree1, tree2, config))
  return errors


def check_trees_are_packing_equivalent(
    tree1: stick_figure.StickFigure,
    tree2: stick_figure.StickFigure,
    config: config_lib.PackingConfig | None = None,
    check_traversal_order: bool = True,
) -> bool:
  """Checks if two trees are equivalent in the packing process."""
  if config is None:
    config = config_lib.PackingConfig()
  errors = get_trees_are_packing_equivalent_errors(
      tree1, tree2, config, check_traversal_order
  )
  return not errors


def get_merge_converter_from_substick_map(
    tree: stick_figure.StickFigure,
    substick_map: dict[str, str],
) -> dict[str, str]:
  """Makes a merge converter from the stick name mapping."""
  merge_converter = dict()
  merged_sf, _ = tree.merge_linear_chains()
  merged_sf_dict = merged_sf.to_dict()
  for stick in merged_sf_dict['children']:
    stick_name = stick['name']
    for sub_stick_name in substick_map:
      if sub_stick_name not in stick_name:
        continue
      if stick_name not in merge_converter:
        merge_converter[stick_name] = substick_map[sub_stick_name]
      assert merge_converter[stick_name] == substick_map[sub_stick_name], (
          f'Inconsistent merge converter for {stick_name}: '
          f'{merge_converter[stick_name]} and {substick_map[sub_stick_name]}'
      )
  return merge_converter


def _preprocess_shaped_tree(
    shaped_tree: stick_figure.StickFigure,
    substick_map: dict[str, str],
    joint_map: dict[str, str],
) -> stick_figure.StickFigure:
  """Preprocesses a shaped tree to make it compatible with the base tree."""
  merged_shaped_tree, _ = shaped_tree.merge_linear_chains()
  merge_converter = get_merge_converter_from_substick_map(
      shaped_tree, substick_map
  )

  merged_shaped_tree_dict = merged_shaped_tree.to_dict()
  for stick in merged_shaped_tree_dict['children']:
    stick_name = stick['name']
    if stick_name in merge_converter:
      stick['name'] = merge_converter[stick_name]
      if merged_shaped_tree_dict['root'] == stick_name:
        merged_shaped_tree_dict['root'] = merge_converter[stick_name]
      if stick['start'] in joint_map:
        stick['start'] = joint_map[stick['start']]
      if stick['end'] in joint_map:
        stick['end'] = joint_map[stick['end']]
  merged_shaped_tree = stick_figure.StickFigure.from_dict(
      merged_shaped_tree_dict
  )
  return merged_shaped_tree


def get_shaped_tree_packing_equivalent_errors(
    base_tree: stick_figure.StickFigure,
    shaped_tree: stick_figure.StickFigure,
    substick_map: dict[str, str],
    joint_map: dict[str, str],
    config: config_lib.PackingConfig | None = None,
    check_traversal_order: bool = True,
) -> list[str]:
  """Checks if a shaped tree is packing equivalent to the base tree."""
  if config is None:
    config = config_lib.PackingConfig()
  merged_shaped_tree = _preprocess_shaped_tree(
      shaped_tree, substick_map, joint_map
  )
  return get_trees_are_packing_equivalent_errors(
      base_tree,
      merged_shaped_tree,
      config,
      check_traversal_order=check_traversal_order,
  )


def check_shaped_tree_is_packing_equivalent(
    base_tree: stick_figure.StickFigure,
    shaped_tree: stick_figure.StickFigure,
    substick_map: dict[str, str],
    joint_map: dict[str, str],
    config: config_lib.PackingConfig | None = None,
    check_traversal_order: bool = True,
) -> bool:
  """Checks if a shaped tree is packing equivalent to the base tree."""
  if config is None:
    config = config_lib.PackingConfig()
  merged_shaped_tree = _preprocess_shaped_tree(
      shaped_tree, substick_map, joint_map
  )
  return check_trees_are_packing_equivalent(
      base_tree,
      merged_shaped_tree,
      config,
      check_traversal_order=check_traversal_order,
  )
