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
"""Heuristic utilities for tree subdivision and 2D simple-fold parameters."""

import collections
from collections.abc import Mapping, Sequence
import copy
import dataclasses
import math
from typing import TypedDict

from absl import logging

from .. import timeout as timeout_lib
from ..crease_pattern import crease as crease_utils
from ..crease_pattern import crease_pattern as cp_utils
from ..crease_pattern import point as point_utils
from ..packing import packing_output as packing_output_utils
from ..packing import types as packing_types
from ..solving import hinges as hinge_utils
from ..stick_figures import stick_figure as sf_utils
from ..stick_figures import traversal as tr
from . import heuristics
from . import shaping_elements
from . import shaping_utils
from . import types as shaping_types
from . import zigzag_shaping

MIN_ANGLE = math.radians(20)
# Small positive offset avoids coinciding consecutive shallow-fold creases.
LARGE_ANGLE = math.pi - MIN_ANGLE / 2 + 0.05
NINETY_DEGREE_THRESHOLD = math.pi / 16
HINGE_FF_SHIFT_DEPTH = 0.3
PI_THRESHOLD = 1e-3


# ---------------------------------------------------------------------------
# Angle and fold-order utilities
# ---------------------------------------------------------------------------


def clip_angle_by_min_angle(
    theta: float, min_angle: float = MIN_ANGLE
) -> float:
  """Preserves the sign of `theta` while clamping `|theta| >= min_angle`."""
  return shaping_utils.sign(theta) * max(min_angle, abs(theta))


def _clip_angle_by_depth(theta: float, max_depth: float) -> float:
  """Clamps `|theta|` upward so its zigzag depth fits within `max_depth`."""
  min_theta = shaping_utils.fold_angle_from_zigzag_depth(max_depth)
  return shaping_utils.sign(theta) * max(min_theta, abs(theta))


def _is_nontrivial_angle(theta: float | None) -> bool:
  """Validates `theta in (-pi, pi)` and returns whether `|theta| > 0`."""
  if theta is None:
    return False
  if not point_utils.in_range(theta, -math.pi, math.pi, strict=True):
    raise ValueError(f'Shape angle {theta} must lie strictly in (-pi, pi).')
  return not math.isclose(theta, 0.0)


def apply_small_angle_flip(target_flap_angle: float) -> list[float]:
  """Splits an acute non-negative flap angle into two shallow obtuse folds."""
  tol = crease_utils.DEFAULT_TOLERANCE
  if target_flap_angle < -tol:
    raise ValueError(
        'Target flap angle must be non-negative to apply small-angle flip,'
        f' got {target_flap_angle}.'
    )
  if target_flap_angle >= (math.pi / 2) - tol or math.isclose(
      target_flap_angle, 0.0, abs_tol=tol
  ):
    return [target_flap_angle]
  small = max(MIN_ANGLE, target_flap_angle)
  return [LARGE_ANGLE, (2.0 * math.pi - small) - LARGE_ANGLE]


def split_acute_zigzag_params(
    element_params: shaping_types.ElementZigzagParams,
    *,
    inscribe_by_epsilon: bool = True,
) -> list[shaping_types.ElementZigzagParams]:
  """Splits an acute zigzag request into two opposing-order shallow requests."""
  zp = element_params.params
  if zp.target_flap_angle is None:
    raise ValueError(
        'ZigzagParams.target_flap_angle must be set for acute-angle splitting.'
    )
  angles = apply_small_angle_flip(zp.target_flap_angle)
  if len(angles) == 1:
    return [element_params]
  depth = zp.depth + (HINGE_FF_SHIFT_DEPTH if inscribe_by_epsilon else 0.0)
  flipped_order = shaping_types.flip_fold_order(zp.fold_order)
  return [
      shaping_types.ElementShapingParams(
          element_name=element_params.element_name,
          params=dataclasses.replace(
              zp, target_flap_angle=angles[0], depth=depth
          ),
      ),
      shaping_types.ElementShapingParams(
          element_name=element_params.element_name,
          params=dataclasses.replace(
              zp,
              target_flap_angle=angles[1],
              depth=depth,
              fold_order=flipped_order,
          ),
      ),
  ]


def shift_zigzag_depth(
    element_params: shaping_types.ElementZigzagParams,
    delta: float = HINGE_FF_SHIFT_DEPTH,
) -> shaping_types.ElementZigzagParams:
  """Inscribes `element_params` deeper by `delta` grid units to avoid hinges."""
  zp = element_params.params
  return dataclasses.replace(
      element_params, params=dataclasses.replace(zp, depth=zp.depth + delta)
  )


# ---------------------------------------------------------------------------
# Stick-figure subdivision and shaping data preparation
# ---------------------------------------------------------------------------


class ShapingData(TypedDict):
  """Prepared flaps, rivers, heuristic tree nodes, and outer river banks."""

  shaping_flaps: list[shaping_elements.ShapingFlap]
  shaping_rivers: list[shaping_elements.ShapingRiver]
  flap_info: dict[str, heuristics.Node]
  outer_bank_indices: dict[str, shaping_types.BankIndex]


def _subdivide_unmerged_elements(
    flaps: Sequence[shaping_elements.ShapingFlap],
    rivers: Sequence[shaping_elements.ShapingRiver],
    outer_bank_indices: Mapping[str, shaping_types.BankIndex],
    unmerged_flap_info: Mapping[str, heuristics.Node],
    merged_stick_names: Mapping[str, Sequence[str]],
) -> tuple[
    list[shaping_elements.ShapingFlap], list[shaping_elements.ShapingRiver]
]:
  """Splits merged flaps and rivers into concentric per-joint sub-elements."""
  out_flaps: list[shaping_elements.ShapingFlap] = []
  for flap in flaps:
    sub_names = merged_stick_names.get(flap.name)
    if sub_names is None:
      out_flaps.append(flap)
      continue
    cur = flap
    last_idx = len(sub_names) - 1
    for idx, sub_name in enumerate(sub_names):
      length = unmerged_flap_info[sub_name].length
      if math.isclose(cur.grid_unit_radius, length):
        if idx != last_idx:
          raise ValueError(
              f'Sub-stick {sub_name!r} exhausted flap {flap.name!r} early.'
          )
        out_flaps.append(cur.rename_and_copy(sub_name))
      else:
        out_flaps.append(
            cur.rename_and_copy(sub_name, max_shaping_depth=length)
        )
        cur = cur.shrink(length)

  out_rivers: list[shaping_elements.ShapingRiver] = []
  for river in rivers:
    sub_names = merged_stick_names.get(river.name)
    if sub_names is None:
      out_rivers.append(river)
      continue
    cur = river
    bank = outer_bank_indices[river.name]
    last_idx = len(sub_names) - 1
    for idx, sub_name in enumerate(sub_names):
      length = unmerged_flap_info[sub_name].length
      if math.isclose(cur.width, length):
        if idx != last_idx:
          raise ValueError(
              f'Sub-stick {sub_name!r} exhausted river {river.name!r} early.'
          )
        out_rivers.append(cur.rename_and_copy(sub_name))
      else:
        out_rivers.append(cur.get_circumbscribed_river(length, sub_name, bank))
        cur = cur.shrink_along_bank(bank, length)

  return out_flaps, out_rivers


def get_outer_bank_indices(
    stick_figure: sf_utils.StickFigure,
    shaping_flaps: list[shaping_elements.ShapingFlap],
    shaping_rivers: list[shaping_elements.ShapingRiver],
    *,
    use_tree_root: bool = False,
) -> dict[str, shaping_types.BankIndex]:
  """Maps each river name to its outer bank index (`'FIRST'` or `'SECOND'`)."""
  outer_banks: dict[str, shaping_types.BankIndex] = {}
  root_name = heuristics.get_first_stick_for_shaping(
      stick_figure, use_tree_root=use_tree_root
  ).name
  for river in shaping_rivers:
    try:
      bank = river.get_outer_bank_index(
          stick_figure, root_name, shaping_flaps, shaping_rivers
      )
    except ValueError as err:
      logging.warning(
          'Could not determine outer bank index for river %s: %s',
          river.name,
          err,
      )
      bank = river.longer_bank_index
    outer_banks[river.name] = bank
  return outer_banks


def _rename_elements_for_merge_converter(
    shaping_flaps: list[shaping_elements.ShapingFlap],
    shaping_rivers: list[shaping_elements.ShapingRiver],
    packing_to_tree: Mapping[str, str],
) -> tuple[
    list[shaping_elements.ShapingFlap], list[shaping_elements.ShapingRiver]
]:
  """Renames packing elements to their merged tree names."""
  flaps: list[shaping_elements.ShapingFlap] = []
  rivers: list[shaping_elements.ShapingRiver] = []
  for packing_name, tree_name in packing_to_tree.items():
    element = shaping_elements.get_element_by_name(
        packing_name, shaping_flaps, shaping_rivers
    )
    if isinstance(element, shaping_elements.ShapingFlap):
      flaps.append(element.rename_and_copy(tree_name))
    elif isinstance(element, shaping_elements.ShapingRiver):
      rivers.append(element.rename_and_copy(tree_name))
    else:
      raise ValueError(
          f'Unsupported shaping element {type(element)} for {packing_name!r}.'
      )
  return flaps, rivers


def _order_chains_from_root(
    unmerged_sf: sf_utils.StickFigure,
    merged_chains: Mapping[str, Sequence[str]],
    *,
    use_tree_root: bool = False,
) -> dict[str, list[str]]:
  """Orders each merged linear chain from proximal (near root) to distal."""
  sub_to_merged = {}
  for merged_name, sub_names in merged_chains.items():
    for sub_name in sub_names:
      sub_to_merged[sub_name] = merged_name

  ordered = {k: list(v) for k, v in merged_chains.items()}
  root = heuristics.get_first_stick_for_shaping(
      unmerged_sf, use_tree_root=use_tree_root
  )
  seen: set[str] = set()
  for stick in tr.breadth_first_traversal(
      unmerged_sf, start_stick_name=root.name
  ):
    merged_name = sub_to_merged.get(stick.name)
    if merged_name is None or merged_name in seen:
      continue
    chain = ordered[merged_name]
    if (
        use_tree_root
        and stick.name == root.name
        and stick.name not in (chain[0], chain[-1])
    ):
      seen.add(merged_name)
      continue
    if stick.name not in (chain[0], chain[-1]):
      raise ValueError(
          f'First visited sub-stick {stick.name!r} is not an endpoint of'
          f' chain {chain}.'
      )
    if chain[-1] == stick.name:
      ordered[merged_name] = chain[::-1]
    seen.add(merged_name)
  return ordered


def get_heuristics_shaping_data(
    stick_figure: sf_utils.StickFigure,
    shaping_flaps: list[shaping_elements.ShapingFlap],
    shaping_rivers: list[shaping_elements.ShapingRiver],
    hinge_info_list: Sequence[packing_types.HingeInfo],
    central_crease_removed: bool,
    *,
    shape_with_unmerged_tree: bool = False,
    merge_converter: Mapping[str, str] | None = None,
    sacrificial_flaps: Sequence[packing_types.SacrificialFlap] = (),
    use_tree_root: bool = False,
) -> ShapingData:
  """Prepares flaps, rivers, `flap_info`, and river outer banks for shaping."""
  unmerged_sf = stick_figure
  merged_sf, raw_chains = stick_figure.merge_linear_chains()
  merged_chains = _order_chains_from_root(
      unmerged_sf, raw_chains, use_tree_root=use_tree_root
  )
  merged_sf = merged_sf.align_root_stick_to_target_azimuth_elevation()

  hinges = list(hinge_info_list)
  if merge_converter:
    if sacrificial_flaps:
      raise NotImplementedError(
          'Sacrificial flaps are unsupported with merge_converter renaming.'
      )
    packing_to_tree = {v: k for k, v in merge_converter.items()}
    shaping_flaps, shaping_rivers = _rename_elements_for_merge_converter(
        shaping_flaps, shaping_rivers, packing_to_tree
    )
    hinges = [
        dataclasses.replace(
            h,
            flap_names=tuple(packing_to_tree.get(n, n) for n in h.flap_names),
        )
        for h in hinges
    ]

  root_dir = heuristics.compute_root_stick_direction_follow_stick_figure_root(
      merged_sf, shaping_flaps, shaping_rivers, use_tree_root=use_tree_root
  )
  sac_map = heuristics.build_sacrificial_flaps_by_base_flap(sacrificial_flaps)
  flap_info = heuristics.get_flap_info_dict_with_hinge_info(
      merged_sf, hinges, root_dir, sac_map, use_tree_root=use_tree_root
  )
  outer_banks = get_outer_bank_indices(
      merged_sf, shaping_flaps, shaping_rivers, use_tree_root=use_tree_root
  )

  if shape_with_unmerged_tree:
    aligned_unmerged = (
        unmerged_sf.align_root_stick_to_target_azimuth_elevation()
    )
    unmerged_info = heuristics.get_flap_info_dict_for_unmerged_stick_figure(
        unmerged_stick_figure=aligned_unmerged,
        merged_stick_name_dict=merged_chains,
        hinge_info_list=hinges,
        root_direction_follow_stick_figure_root=root_dir,
        node_dict=flap_info,
        sacrificial_flaps_by_base_flap=sac_map,
        use_tree_root=use_tree_root,
    )
    shaping_flaps, shaping_rivers = _subdivide_unmerged_elements(
        flaps=shaping_flaps,
        rivers=shaping_rivers,
        outer_bank_indices=outer_banks,
        unmerged_flap_info=unmerged_info,
        merged_stick_names=merged_chains,
    )
    flap_info = unmerged_info
    outer_banks = get_outer_bank_indices(
        aligned_unmerged,
        shaping_flaps,
        shaping_rivers,
        use_tree_root=use_tree_root,
    )

  shaping_flaps, shaping_rivers = (
      shaping_elements.adjust_shaping_elements_for_central_crease_removal(
          shaping_flaps, shaping_rivers, outer_banks, central_crease_removed
      )
  )
  return ShapingData(
      shaping_flaps=shaping_flaps,
      shaping_rivers=shaping_rivers,
      flap_info=flap_info,
      outer_bank_indices=outer_banks,
  )


# ---------------------------------------------------------------------------
# Boundary elevation, phase inference, and hinge-safe fold order
# ---------------------------------------------------------------------------


def _point_elevation(
    point: point_utils.Point, creases: Sequence[crease_utils.Crease]
) -> int | None:
  """Returns the first known elevation among creases containing `point`."""
  for c in creases:
    if (
        c.elevation is not None
        and c.point_status(point) != crease_utils.PointStatus.EXTERNAL
    ):
      return c.elevation
  return None


def partition_boundary_by_elevation(
    boundary: shaping_elements.ShapingBoundary,
    grid_size: int,
    creases: Sequence[crease_utils.Crease],
) -> tuple[list[point_utils.Point], list[point_utils.Point]]:
  """Splits boundary grid points into `(elevation_0, elevation_1)` lists."""
  by_elev: dict[int, list[point_utils.Point]] = collections.defaultdict(list)
  start = boundary.creases[0].p1_init
  cur_p, edge_idx = start, 0
  first = True
  while first or not cur_p.isclose(start):
    first = False
    timeout_lib.maybe_timeout_cooperatively()
    elev = _point_elevation(cur_p, creases)
    if elev is not None:
      by_elev[elev].append(cur_p)
    try:
      cur_p, edge_idx = boundary.walk(cur_p, edge_idx, grid_size)
    except ValueError:
      break

  unusual = by_elev.keys() - {0, 1}
  if unusual:
    logging.warning(
        'Ignoring unusual elevations during zigzag phase inference: %s', unusual
    )
  return by_elev[0], by_elev[1]


def _boundary_step_distance(
    start: point_utils.Point,
    target: point_utils.Point,
    boundary: shaping_elements.ShapingBoundary,
    grid_size: int,
) -> int:
  """Counts unit grid steps along `boundary` from `start` to `target`."""
  edge_idx = boundary.get_edge_index(start)
  if edge_idx is None:
    raise ValueError(f'Start point {start} is not on boundary {boundary}.')
  cur_p = start
  steps = 0
  while not cur_p.isclose(target):
    timeout_lib.maybe_timeout_cooperatively()
    steps += 1
    try:
      cur_p, edge_idx = boundary.walk(cur_p, edge_idx, grid_size)
    except ValueError as err:
      raise RuntimeError(
          f'Target point {target} not reached from {start} on boundary.'
      ) from err
  return steps


def _phase_from_border_anchor(
    valley_anchored: bool,
    start_point: point_utils.Point,
    boundary: shaping_elements.ShapingBoundary,
    grid_size: int,
    creases: Sequence[crease_utils.Crease],
) -> bool:
  """Returns whether even phase lands anchors on the target elevation parity."""
  valley_pts, mountain_pts = partition_boundary_by_elevation(
      boundary, grid_size, creases
  )
  primary = valley_pts if valley_anchored else mountain_pts
  secondary = mountain_pts if valley_anchored else valley_pts
  if primary:
    target = primary[0]
  elif secondary:
    adj = secondary[0]
    adj_edge = boundary.get_edge_index(adj)
    if adj_edge is None:
      raise ValueError(f'Adjacent point {adj} is not on boundary {boundary}.')
    target, _ = boundary.walk(adj, adj_edge, grid_size)
  else:
    raise ValueError(
        'No boundary elevation points found to determine zigzag phase.'
    )
  return (
      _boundary_step_distance(start_point, target, boundary, grid_size) % 2 == 0
  )


def _is_valley_anchored(
    abs_direction: heuristics.AbsFlapDirection, theta: float
) -> bool:
  """Returns whether the zigzag should anchor at elevation-0 border pleats."""
  is_down = abs_direction == heuristics.AbsFlapDirection.DOWN
  return is_down != (theta > 0)


def convert_flap_angle_to_global_phase(
    theta: float,
    absolute_direction: heuristics.AbsFlapDirection,
    flap: shaping_elements.ShapingFlap,
) -> bool:
  """Converts a signed flap angle into its unflipped boundary zigzag phase."""
  start = shaping_utils.compute_anchor_points_start(
      point_utils.midpoint(flap.corners[0], flap.corners[1]),
      flap.grid_size,
      True,
      flap.get_unit_edge_vectors(flap.grid_size)[0],
  )
  boundary = shaping_elements.get_flap_boundary(flap, flap.grid_size)
  return _phase_from_border_anchor(
      _is_valley_anchored(absolute_direction, theta),
      start,
      boundary,
      flap.grid_size,
      flap.creases,
  )


def convert_river_angle_to_global_phase(
    theta: float,
    absolute_direction: heuristics.AbsFlapDirection,
    outer_bank_index: shaping_types.BankIndex,
    river: shaping_elements.ShapingRiver,
) -> bool:
  """Converts a signed river angle into its unflipped outer-bank phase."""
  boundary = river.get_boundary_from_index(outer_bank_index)
  return _phase_from_border_anchor(
      _is_valley_anchored(absolute_direction, theta),
      boundary.creases[0].p1_init,
      boundary,
      river.grid_size,
      river.creases,
  )


def _critical_hinge_checks(
    even_phase: bool,
    grid_size: int,
    rect: shaping_elements.ShapingFlap,
) -> list[tuple[point_utils.Point, point_utils.Point, bool]]:
  """Collects boundary segments whose hinge assignment could violate Maekawa."""
  start = shaping_utils.compute_anchor_points_start(
      point_utils.midpoint(rect.corners[0], rect.corners[1]),
      grid_size,
      even_phase,
      rect.get_unit_edge_vectors(grid_size)[0],
  )
  edge_idx = shaping_utils.get_edge_index(start, rect.corners, rect.edges)
  boundary = shaping_elements.get_flap_boundary(rect, grid_size)
  cur_p, edge_idx = boundary.walk(
      start, edge_idx, grid_size, snap_to_grid=False
  )

  checks: list[tuple[point_utils.Point, point_utils.Point, bool]] = []
  if start.is_internal:
    checks.append((start, cur_p, True))
  same_as_start = False
  record_next = False

  while not cur_p.isclose(start):
    timeout_lib.maybe_timeout_cooperatively()
    prev_p, prev_edge = cur_p, edge_idx
    cur_p, edge_idx = boundary.walk(
        cur_p, edge_idx, grid_size, snap_to_grid=False
    )
    if record_next:
      checks.append((prev_p, cur_p, same_as_start))
      record_next = False
    same_as_start = not same_as_start
    internal_corner = prev_edge != edge_idx and cur_p.is_internal
    entering_paper = prev_p.is_external and cur_p.is_border
    if internal_corner or entering_paper:
      record_next = True
  return checks


def get_fold_order(
    is_even_phase: bool,
    grid_size: int,
    rect: shaping_elements.ShapingFlap,
    hinge_segments: Sequence[crease_utils.Crease],
) -> tuple[shaping_types.FoldOrder, bool]:
  """Picks a conflict-free `(fold_order, is_valid)` against `hinge_segments`."""
  checks = _critical_hinge_checks(is_even_phase, grid_size, rect)
  for order in (
      shaping_types.MOUNTAIN_TO_VALLEY,
      shaping_types.VALLEY_TO_MOUNTAIN,
  ):
    conflict = False
    for p1, p2, same_as_start in checks:
      probe = crease_utils.Crease(
          p1, p2, fold=order[0] if same_as_start else order[1]
      )
      if any(probe.eq_endpoints_and_fold(h) for h in hinge_segments):
        conflict = True
        break
    if not conflict:
      return order, True

  logging.warning(
      'No conflict-free fold order for flap %s (even_phase=%s); defaulting'
      ' to MOUNTAIN_TO_VALLEY.',
      rect.name,
      is_even_phase,
  )
  return shaping_types.MOUNTAIN_TO_VALLEY, False


# ---------------------------------------------------------------------------
# 2D simple-fold heuristic parameter synthesis
# ---------------------------------------------------------------------------


def _apply_context_phase_flips(
    phase: bool,
    element: shaping_elements.ShapingElement,
    flap_info: Mapping[str, heuristics.Node],
    central_crease_removed: bool,
    flip_counts: Mapping[str, int],
) -> bool:
  """Adjusts `phase` for ancestor 180-degree flips and centerline removal."""
  total_flips = 0
  cur: str | None = element.name
  while cur is not None:
    timeout_lib.maybe_timeout_cooperatively()
    total_flips += flip_counts.get(cur, 0)
    node = flap_info.get(cur)
    cur = node.parent if node is not None else None
  if total_flips % 2 == 1:
    phase = not phase
  if central_crease_removed and element.is_right_of_central_crease:
    phase = not phase
  return phase


def _default_element_zigzag_params(
    element: shaping_elements.ShapingElement,
) -> shaping_types.ElementZigzagParams:
  """Returns zero-angle no-op zigzag parameters for `element`."""
  bank: shaping_types.BankIndex | None = (
      'FIRST' if isinstance(element, shaping_elements.ShapingRiver) else None
  )
  return shaping_types.ElementShapingParams(
      element_name=element.name,
      params=shaping_types.ZigzagParams(
          target_flap_angle=0.0,
          fold_order=shaping_types.MOUNTAIN_TO_VALLEY,
          even_phase=True,
          bank_index=bank,
      ),
  )


def get_flap_zigzag_params_from_heuristics(
    grid_size: int,
    rect: shaping_elements.ShapingFlap,
    flap_info: Mapping[str, heuristics.Node],
    hinge_segments: Sequence[crease_utils.Crease],
    central_crease_removed: bool,
    depth: float = 0.0,
    resolve_hinge_ff_with_shifts: bool = False,
    *,
    flip_counts: Mapping[str, int] = (),
) -> shaping_types.ElementZigzagParams:
  """Computes heuristic zigzag parameters for a single flap."""
  node = flap_info[rect.name]
  theta = node.shape_angle
  if (
      rect.grid_unit_radius <= 1.0 + point_utils.DEFAULT_TOLERANCE
      or not _is_nontrivial_angle(theta)
  ):
    return _default_element_zigzag_params(rect)
  if theta is None:
    return _default_element_zigzag_params(rect)

  theta = clip_angle_by_min_angle(theta)
  if abs(abs(theta) - math.pi / 2) < NINETY_DEGREE_THRESHOLD:
    theta = shaping_utils.sign(theta) * (math.pi / 2)
  if rect.max_shaping_depth is not None:
    theta = _clip_angle_by_depth(theta, rect.max_shaping_depth - depth)
  if central_crease_removed and rect.contains_central_crease:
    return _default_element_zigzag_params(rect)

  active_rect = rect.shrink(depth) if depth > 0 else rect
  if active_rect.is_central_flap:
    raw_phase = crease_utils.approx_lte(0.0, theta)
  else:
    raw_phase = convert_flap_angle_to_global_phase(
        theta, node.absolute_direction, active_rect
    )
  flip_map = dict(flip_counts) if flip_counts else {}
  phase = _apply_context_phase_flips(
      raw_phase, active_rect, flap_info, central_crease_removed, flip_map
  )
  fold_order, is_valid = get_fold_order(
      phase, grid_size, active_rect, hinge_segments
  )
  if resolve_hinge_ff_with_shifts and not is_valid:
    depth += HINGE_FF_SHIFT_DEPTH
    if active_rect.max_shaping_depth is not None:
      theta = _clip_angle_by_depth(theta, active_rect.max_shaping_depth - depth)

  return shaping_types.ElementShapingParams(
      element_name=active_rect.name,
      params=shaping_types.ZigzagParams(
          target_flap_angle=abs(theta),
          even_phase=phase,
          fold_order=fold_order,
          depth=depth,
      ),
  )


def _river_zigzag_params_from_heuristics(
    river: shaping_elements.ShapingRiver,
    outer_bank_index: shaping_types.BankIndex,
    flap_info: Mapping[str, heuristics.Node],
    central_crease_removed: bool,
    depth: float,
    flip_counts: Mapping[str, int],
) -> shaping_types.ElementZigzagParams:
  """Computes heuristic zigzag parameters for a single river."""
  node = flap_info[river.name]
  theta = node.shape_angle
  if not _is_nontrivial_angle(theta) or theta is None:
    return _default_element_zigzag_params(river)
  theta = clip_angle_by_min_angle(theta)
  theta = _clip_angle_by_depth(theta, river.max_shaping_depth - depth)
  raw_phase = convert_river_angle_to_global_phase(
      theta, node.absolute_direction, outer_bank_index, river
  )
  phase = _apply_context_phase_flips(
      raw_phase, river, flap_info, central_crease_removed, flip_counts
  )
  return shaping_types.ElementShapingParams(
      element_name=river.name,
      params=shaping_types.ZigzagParams(
          target_flap_angle=abs(theta),
          fold_order=shaping_types.MOUNTAIN_TO_VALLEY,
          even_phase=phase,
          bank_index=outer_bank_index,
          depth=depth,
      ),
  )


def get_inscribed_polygon_fold_order(
    shape_angle_elevation: float,
) -> shaping_types.FoldOrder:
  """Returns MV order for positive elevation angles and VM order otherwise."""
  if shape_angle_elevation > 0:
    return shaping_types.MOUNTAIN_TO_VALLEY
  return shaping_types.VALLEY_TO_MOUNTAIN


def _inscribed_params_from_heuristics(
    element: shaping_elements.ShapingElement,
    outer_bank_indices: Mapping[str, shaping_types.BankIndex],
    flap_info: Mapping[str, heuristics.Node],
    *,
    apply_180_rotation: bool,
    shape_3d: bool,
) -> shaping_types.ElementInscribedParams:
  """Computes heuristic inscribed-polygon parameters for `element`."""
  node = flap_info[element.name]
  if node.is_root:
    return shaping_types.ElementShapingParams(
        element_name=element.name, params=()
    )
  fold_percent = 1.0 if apply_180_rotation else 0.0
  if shape_3d:
    fold_percent = abs(fold_percent - abs(node.shape_angle_elevation) / math.pi)
  fold_order = get_inscribed_polygon_fold_order(node.shape_angle_elevation)

  min_depth = int(node.has_assigned_hinge)
  if element.max_shaping_depth is not None:
    max_depth = element.max_shaping_depth - 1
  elif isinstance(element, shaping_elements.ShapingFlap):
    max_depth = element.grid_unit_radius - 1
  else:
    max_depth = element.width - 1
  depth = min(min_depth, max_depth)

  bank = (
      None
      if isinstance(element, shaping_elements.ShapingFlap)
      else outer_bank_indices[element.name]
  )
  return shaping_types.ElementShapingParams(
      element_name=element.name,
      params=(
          shaping_types.InscribedParams(
              depth=depth,
              fold_order=fold_order,
              fold_percent=fold_percent,
              bank_index=bank,
          ),
      ),
  )


def get_all_simple_fold_params_from_heuristics(
    *,
    flap_info: Mapping[str, heuristics.Node],
    rectangles: Sequence[shaping_elements.ShapingFlap],
    rivers: Sequence[shaping_elements.ShapingRiver],
    outer_bank_indices: Mapping[str, shaping_types.BankIndex],
    grid_size: int,
    hinge_segments: Sequence[crease_utils.Crease],
    apply_flip_to_acute_angles: bool,
    resolve_hinge_ff_with_shifts: bool,
    shape_rivers: bool,
    shape_3d: bool,
    central_crease_removed: bool,
) -> tuple[
    list[shaping_types.ElementZigzagParams],
    list[shaping_types.ElementInscribedParams],
]:
  """Synthesizes BFS-ordered zigzag and inscribed parameters for a model."""
  nodes = copy.deepcopy(dict(flap_info))
  elements: dict[str, shaping_elements.ShapingElement] = {
      **{r.name: r for r in rectangles},
      **{r.name: r for r in rivers},
  }
  roots = [node for node in nodes.values() if node.is_root]
  if len(roots) != 1:
    raise ValueError(
        f'Expected exactly one root node in flap_info, found {len(roots)}.'
    )

  flip_counts: dict[str, int] = collections.defaultdict(int)
  all_zigzag: list[shaping_types.ElementZigzagParams] = []
  all_inscribed: list[shaping_types.ElementInscribedParams] = []
  queue = collections.deque([roots[0].name])

  while queue:
    timeout_lib.maybe_timeout_cooperatively()
    name = queue.popleft()
    element = elements.get(name)
    if element is None:
      continue
    node = nodes[name]
    queue.extend(node.children)
    if element.elevations - {0, 1}:
      logging.info(
          'Skipping element %s with unsupported elevations %s.',
          name,
          element.elevations,
      )
      continue

    angle = node.shape_angle
    rotate_180 = angle is not None and (
        (apply_flip_to_acute_angles and MIN_ANGLE < abs(angle) < math.pi / 2)
        or math.isclose(abs(angle), math.pi, abs_tol=PI_THRESHOLD)
    )
    if rotate_180:
      node.fold_state = heuristics.FoldState.FOLD_IN

    ins_req = _inscribed_params_from_heuristics(
        element,
        outer_bank_indices,
        nodes,
        apply_180_rotation=rotate_180,
        shape_3d=shape_3d,
    )
    for p in ins_req.params:
      if p.fold_percent > 0:
        flip_counts[name] += round(p.fold_percent)

    if angle is None or math.isclose(abs(angle), math.pi):
      zz_req = _default_element_zigzag_params(element)
    else:
      if not ins_req.params or math.isclose(
          ins_req.params[0].fold_percent, 0.0
      ):
        ins_depth = 0.0
      else:
        ins_depth = ins_req.params[0].depth
      if isinstance(element, shaping_elements.ShapingFlap):
        zz_req = get_flap_zigzag_params_from_heuristics(
            grid_size,
            element,
            nodes,
            hinge_segments,
            central_crease_removed,
            ins_depth,
            resolve_hinge_ff_with_shifts,
            flip_counts=flip_counts,
        )
      else:
        zz_req = _river_zigzag_params_from_heuristics(
            element,
            outer_bank_indices[name],
            nodes,
            central_crease_removed,
            ins_depth,
            flip_counts,
        )

    target_angle = zz_req.params.target_flap_angle
    if target_angle is not None and target_angle > 0:
      flip_counts[name] += round(zz_req.params.fold_percent)
      if shape_rivers or not isinstance(element, shaping_elements.ShapingRiver):
        all_zigzag.append(zz_req)

    if any(p.fold_percent > 0 for p in ins_req.params):
      all_inscribed.append(ins_req)

  return all_zigzag, all_inscribed


def get_simple_fold_shaping_params(
    stick_figure: sf_utils.StickFigure,
    grid_size: int,
    shaping_flaps: list[shaping_elements.ShapingFlap],
    shaping_rivers: list[shaping_elements.ShapingRiver],
    hinge_info_list: Sequence[packing_types.HingeInfo],
    central_crease_removed: bool,
    *,
    apply_flip_to_acute_angles: bool = False,
    shape_rivers: bool = False,
    shape_with_unmerged_tree: bool = False,
    resolve_hinge_ff_with_shifts: bool = False,
    merge_converter: Mapping[str, str] | None = None,
    element_name_list: Sequence[str] | None = None,
    sacrificial_flaps: Sequence[packing_types.SacrificialFlap] = (),
) -> tuple[
    list[shaping_types.ElementZigzagParams],
    list[shaping_types.ElementInscribedParams],
    list[shaping_elements.ShapingFlap],
    list[shaping_elements.ShapingRiver],
]:
  """Computes 2D heuristic zigzag/inscribed parameters and adjusted elements."""
  data = get_heuristics_shaping_data(
      stick_figure=stick_figure,
      shaping_flaps=shaping_flaps,
      shaping_rivers=shaping_rivers,
      hinge_info_list=hinge_info_list,
      central_crease_removed=central_crease_removed,
      shape_with_unmerged_tree=shape_with_unmerged_tree,
      merge_converter=merge_converter,
      sacrificial_flaps=sacrificial_flaps,
  )
  hinge_segments = [h.coordinate for h in hinge_info_list]
  zz_params, ins_params = get_all_simple_fold_params_from_heuristics(
      flap_info=data['flap_info'],
      rectangles=data['shaping_flaps'],
      rivers=data['shaping_rivers'],
      outer_bank_indices=data['outer_bank_indices'],
      grid_size=grid_size,
      hinge_segments=hinge_segments,
      apply_flip_to_acute_angles=apply_flip_to_acute_angles,
      resolve_hinge_ff_with_shifts=resolve_hinge_ff_with_shifts,
      shape_rivers=shape_rivers,
      shape_3d=False,
      central_crease_removed=central_crease_removed,
  )
  if element_name_list is not None:
    allowed = set(element_name_list)
    zz_params = [p for p in zz_params if p.element_name in allowed]
    ins_params = [p for p in ins_params if p.element_name in allowed]
  return zz_params, ins_params, data['shaping_flaps'], data['shaping_rivers']


def shape_cp_with_heuristics(
    cp: cp_utils.CreasePattern,
    stick_figure: sf_utils.StickFigure,
    grid_size: int,
    shaping_flaps: list[shaping_elements.ShapingFlap],
    shaping_rivers: list[shaping_elements.ShapingRiver],
    central_crease_removed: bool,
    *,
    hinge_info_list: Sequence[packing_types.HingeInfo] | None = None,
    packing_output: packing_output_utils.PackingOutput | None = None,
    solving_output: hinge_utils.SolverResult | None = None,
    apply_flip_to_acute_angles: bool = False,
    shape_rivers: bool = False,
    shape_with_unmerged_tree: bool = False,
    resolve_hinge_ff_with_shifts: bool = False,
    merge_converter: Mapping[str, str] | None = None,
    narrow: bool = False,
    element_name_list: Sequence[str] | None = None,
    raise_error_on_ff_violation: bool = False,
) -> cp_utils.CreasePattern:
  """Shapes `cp` with 2D stick-figure zigzag and inscribed heuristics."""
  sac_flaps: Sequence[packing_types.SacrificialFlap] = ()
  if hinge_info_list is None:
    if packing_output is None or solving_output is None:
      raise ValueError(
          'Provide either hinge_info_list or both packing_output and'
          ' solving_output.'
      )
    hinges = heuristics.update_hinge_info_from_solver_result(
        copy.deepcopy(packing_output.hinges), solving_output
    )
    sac_flaps = packing_output.sacrificial_flaps or ()
  else:
    hinges = hinge_info_list

  zz_params, ins_params, flaps, rivers = get_simple_fold_shaping_params(
      stick_figure=stick_figure,
      grid_size=grid_size,
      shaping_flaps=shaping_flaps,
      shaping_rivers=shaping_rivers,
      hinge_info_list=hinges,
      central_crease_removed=central_crease_removed,
      apply_flip_to_acute_angles=apply_flip_to_acute_angles,
      shape_rivers=shape_rivers,
      shape_with_unmerged_tree=shape_with_unmerged_tree,
      resolve_hinge_ff_with_shifts=resolve_hinge_ff_with_shifts,
      merge_converter=merge_converter,
      element_name_list=element_name_list,
      sacrificial_flaps=sac_flaps,
  )

  initial_count = len(cp.creases)
  if zz_params:
    cp = zigzag_shaping.shape_cp_with_zigzags(
        cp=cp,
        grid_size=grid_size,
        rectangles=flaps,
        element_shaping_params=zz_params,
        rivers=rivers,
        narrow=narrow,
        raise_error_on_ff_violation=raise_error_on_ff_violation,
    )
  if ins_params:
    cp = zigzag_shaping.shape_cp_with_inscribed_elements(
        cp=cp,
        grid_size=grid_size,
        rectangles=flaps,
        element_shaping_params=ins_params,
        rivers=rivers,
        central_crease_removed=central_crease_removed,
        raise_error_on_ff_violation=raise_error_on_ff_violation,
    )
  if len(cp.creases) == initial_count:
    logging.warning('No shaping creases were added by 2D heuristics.')
  return cp
