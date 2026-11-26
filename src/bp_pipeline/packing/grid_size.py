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

"""Box-pleating packing grid-size Chebyshev (`D_inf`) lower bound."""

import collections
from collections.abc import Callable, Mapping, Sequence
import itertools
import math
from typing import Any, NamedTuple, Protocol


class _SymRectConstraint(NamedTuple):
  """Left-half `[s, N/2 - lx] x [s, N - s]` pigeonhole constraint."""

  count: int
  dist_min: int
  inset_floor: int
  lx_floor: int


class _StickProtocol(Protocol):
  """Structural type for Stick objects."""

  @property
  def name(self) -> str:
    ...

  @property
  def start(self) -> str:
    ...

  @property
  def end(self) -> str:
    ...

  @property
  def length(self) -> int:
    ...

  @property
  def width(self) -> float | int:
    ...

  @property
  def is_flap(self) -> bool:
    ...

  @property
  def is_river(self) -> bool:
    ...

  @property
  def is_on_symmetry_axis(self) -> bool:
    ...

  @property
  def symmetric_stick_name(self) -> str | None:
    ...

  @property
  def requires_level_shifter(self) -> bool:
    ...


class _JointProtocol(Protocol):
  """Structural type for StickFigure joints."""

  @property
  def degree(self) -> int:
    ...

  @property
  def sticks(self) -> Sequence[_StickProtocol]:
    ...


class _StickFigureProtocol(Protocol):
  """Structural type for StickFigure without creating a BUILD cycle."""

  @property
  def children(self) -> Sequence[_StickProtocol]:
    ...

  @property
  def joints(self) -> Mapping[str, _JointProtocol]:
    ...

  @property
  def is_symmetric_tree(self) -> bool:
    ...

  @property
  def symmetry_axis(self) -> Sequence[str]:
    ...

  def __getitem__(self, key: str) -> _StickProtocol:
    ...

  def merge_linear_chains(self) -> tuple['_StickFigureProtocol', Any]:
    ...


def _farthest_in_tree(
    start: str, adj: Mapping[str, Sequence[tuple[str, int]]]
) -> tuple[str, int]:
  """Returns the farthest reachable joint and distance in a tree component."""
  dist: dict[str, int] = {start: 0}
  queue = collections.deque([start])
  farthest_node = start
  farthest_dist = 0
  while queue:
    cur = queue.popleft()
    cur_dist = dist[cur]
    if cur_dist > farthest_dist:
      farthest_dist = cur_dist
      farthest_node = cur
    for nxt, weight in adj[cur]:
      if nxt not in dist:
        dist[nxt] = cur_dist + weight
        queue.append(nxt)
  return farthest_node, farthest_dist


def longest_path_in_joints(joints: Mapping[str, _JointProtocol]) -> int:
  """Calculates the longest weighted path (diameter) across connected joints."""
  if not joints:
    return 0

  stick_by_name: dict[str, _StickProtocol] = {}
  for joint in joints.values():
    for stick in joint.sticks:
      stick_by_name[stick.name] = stick

  if not stick_by_name:
    return 0

  adj: dict[str, list[tuple[str, int]]] = collections.defaultdict(list)
  for stick in stick_by_name.values():
    weight = int(stick.length)
    adj[stick.start].append((stick.end, weight))
    adj[stick.end].append((stick.start, weight))

  visited: set[str] = set()
  diameter = 0
  for root in adj:
    if root in visited:
      continue
    comp: list[str] = []
    queue = collections.deque([root])
    visited.add(root)
    while queue:
      cur = queue.popleft()
      comp.append(cur)
      for nxt, _ in adj[cur]:
        if nxt not in visited:
          visited.add(nxt)
          queue.append(nxt)

    if len(comp) <= 1:
      continue

    edge_count = sum(len(adj[node]) for node in comp) // 2
    if edge_count != len(comp) - 1:
      raise ValueError(
          f'Expected a tree component, got {edge_count} edges for'
          f' {len(comp)} joints.'
      )
    endpoint, _ = _farthest_in_tree(comp[0], adj)
    _, comp_diameter = _farthest_in_tree(endpoint, adj)
    diameter = max(diameter, comp_diameter)

  return int(diameter)


def _flap_margin(
    stick: _StickProtocol,
    use_level_shifter: bool,
    flap_dim_fn: Callable[[Any, bool], int] | None,
) -> int:
  """Returns the per-side margin from `flap_dim_fn` (`0` when unprovided)."""
  if flap_dim_fn is None:
    return 0
  packed_dim = int(flap_dim_fn(stick, use_level_shifter))
  return max(0, (packed_dim - 2 * int(stick.length)) // 2)


def _extract_symmetry_info(
    figure: _StickFigureProtocol,
    use_level_shifter: bool,
    flap_dim_fn: Callable[[Any, bool], int] | None,
) -> tuple[set[str], dict[str, str], set[str]]:
  """Returns symmetry axis, mirror map, and one representative off-axis half."""
  axis_names = set(figure.symmetry_axis)
  axis_joints: set[str] = set()
  for name in axis_names:
    stick = figure[name]
    axis_joints.add(stick.start)
    axis_joints.add(stick.end)

  # Only pair mirror twins when their effective packing footprints match.
  sym_map: dict[str, str] = {}
  for stick in figure.children:
    if stick.is_on_symmetry_axis:
      axis_names.add(stick.name)
      continue
    twin_name = stick.symmetric_stick_name
    if twin_name is None:
      axis_names.add(stick.name)
      continue
    twin = figure[twin_name]
    if (
        stick.length == twin.length
        and stick.width == twin.width
        and stick.requires_level_shifter == twin.requires_level_shifter
        and _flap_margin(stick, use_level_shifter, flap_dim_fn)
        == _flap_margin(twin, use_level_shifter, flap_dim_fn)
    ):
      sym_map[stick.name] = twin_name

  left_half: set[str] = set()
  for stick in figure.children:
    if (
        stick.name in axis_names
        or stick.name not in sym_map
        or stick.name in left_half
        or sym_map[stick.name] in left_half
    ):
      continue
    queue = collections.deque([stick])
    component = []
    seen = {stick.name}
    while queue:
      curr = queue.popleft()
      component.append(curr)
      for joint_name in (curr.start, curr.end):
        if joint_name in axis_joints:
          continue
        for neighbor in figure.joints[joint_name].sticks:
          if neighbor.name not in axis_names and neighbor.name not in seen:
            seen.add(neighbor.name)
            queue.append(neighbor)
    if all(member.name in sym_map for member in component):
      left_half.update(member.name for member in component)

  return axis_names, sym_map, left_half


def _max_separated_leaves(
    root_order: Sequence[str],
    parent_by_joint: Mapping[str, tuple[str, int] | None],
    weighted_adj: Mapping[str, Sequence[tuple[str, int]]],
    eligible_leaves: set[str],
    min_dist: int,
) -> int:
  """Returns max leaf count with pairwise tree distance >= min_dist."""
  best_depth: dict[str, int | None] = {}
  count = 0
  for joint_name in reversed(root_order):
    child_depths = []
    for nxt, weight in weighted_adj[joint_name]:
      parent_entry = parent_by_joint.get(nxt)
      if parent_entry is None or parent_entry[0] != joint_name:
        continue
      child_depth = best_depth[nxt]
      if child_depth is not None:
        child_depths.append(child_depth + weight)
    if not child_depths:
      if joint_name in eligible_leaves:
        best_depth[joint_name] = 0
        count += 1
      else:
        best_depth[joint_name] = None
      continue
    child_depths.sort()
    while (
        len(child_depths) >= 2 and child_depths[0] + child_depths[1] < min_dist
    ):
      child_depths.pop(0)
      count -= 1
    best_depth[joint_name] = child_depths[0]
  return count


def _compute_tree_pigeonhole_floor(
    working_figure: _StickFigureProtocol,
    sticks_to_consider: Sequence[_StickProtocol],
    *,
    is_sym: bool,
    axis_names: set[str],
    sym_map: Mapping[str, str],
    left_half: set[str],
    use_level_shifter: bool,
    flap_dim_fn: Callable[[Any, bool], int] | None,
    is_full_tree: bool,
) -> tuple[int, list[_SymRectConstraint]]:
  """Computes exact D_inf lower bounds via center insets and pigeonholing.

  Assumptions (matching `calculate_v4_grid_size`):
  - Backtracking square-flap placement checks
  (`flap_placement.validate_candidate`
    and any inset margin modeled by `flap_dim_fn`), not `shaping_flap` `2l x
    l'` rectangles or stacking solvers that reserve fewer cells.
  - Mirrored symmetric placement (`[0, N/2]` off-axis confinement) when `is_sym`
    is True, not `allow_paired_axis_placement=True`.

  Each rule derives from Chebyshev (`D_inf`) disk-packing bounds:
  - Single-stick floor: a flap with inset `m` has center in `[m, N - m]^2` and
    at least one corner inside `[0, N]^2`, requiring `N >= l + 2*m` (`2*(l+2*m)`
    when off-axis in a mirrored symmetric packing `[0, N/2]`).
  - Leaf-pair inset distance: leaves `(u, v)` obey `||c_u - c_v||_inf >=
    dist_eff(u, v)` inside `[m_u, N - m_u] x [m_v, N - m_v]`, forcing
    `N >= dist_eff(u, v) + m_u + m_v`.
  - Symmetric mirror-twin x bound: mirror twins `(u, u')` share `y` and satisfy
    `N - 2 * x_u >= dist_eff(u, u')`, forcing `x_u <= N/2 - lx_u` where
    `lx_u = max(l_u + m_u, ceil(dist_eff(u, u') / 2))`.
  - 2D square pigeonhole: `K` leaves separated by `D` inside
    `[s, N - s]^2` cannot share any `D x D` bucket, so
    `ceil((N - 2*s + 1) / D)^2 >= K`, forcing `N >= isqrt(K - 1) * D + 2*s`.

  Args:
    working_figure: Chain-merged stick figure for tree traversal.
    sticks_to_consider: Subset of sticks for single-stick floor constraints.
    is_sym: Whether to enforce mirrored symmetric-placement constraints.
    axis_names: Stick names constrained to lie on the symmetry axis.
    sym_map: Mirror-twin mapping for off-axis sticks with matching footprints.
    left_half: Representative off-axis sticks on one side of the axis.
    use_level_shifter: Whether level-shifter state is passed to `flap_dim_fn`.
    flap_dim_fn: Optional callback returning packed flap side length.
    is_full_tree: Whether all sticks are included (enables global leaf bounds).

  Returns:
    Tuple `(lower_bound, sym_constraints)` with rectangular pigeonhole records.
  """
  min_dim = 1
  for stick in sticks_to_consider:
    if stick.is_river:
      min_dim = max(min_dim, int(stick.length) + 1)
    elif stick.is_flap:
      margin = _flap_margin(stick, use_level_shifter, flap_dim_fn)
      span = int(stick.length) + 2 * margin
      if is_sym and stick.name not in axis_names:
        min_dim = max(min_dim, 2 * span)
      else:
        min_dim = max(min_dim, span)
    else:
      min_dim = max(min_dim, int(stick.length))

  if not is_full_tree:
    return min_dim, []

  joints = working_figure.joints
  weighted_adj: dict[str, list[tuple[str, int]]] = {name: [] for name in joints}
  for stick in working_figure.children:
    if stick.is_flap:
      weight = int(stick.length) + _flap_margin(
          stick, use_level_shifter, flap_dim_fn
      )
    else:
      weight = int(stick.length)
    weighted_adj[stick.start].append((stick.end, weight))
    weighted_adj[stick.end].append((stick.start, weight))

  leaves = [name for name, joint in joints.items() if joint.degree == 1]
  leaf_inset = {
      name: _flap_margin(joints[name].sticks[0], use_level_shifter, flap_dim_fn)
      for name in leaves
  }
  leaf_leff = {
      # NOMUTANTS(UOI): Defensive floor; twin half-dist dominates mirror pairs.
      name: int(joints[name].sticks[0].length) + leaf_inset[name]
      for name in leaves
  }
  leaf_by_stick = {joints[name].sticks[0].name: name for name in leaves}

  leaf_dists: dict[str, dict[str, int]] = {}
  max_path_inset = 0
  all_pair_dists: set[int] = set()
  for leaf in leaves:
    dist = {leaf: 0}
    queue = collections.deque([leaf])
    while queue:
      cur = queue.popleft()
      for nxt, weight in weighted_adj[cur]:
        if nxt not in dist:
          dist[nxt] = dist[cur] + weight
          queue.append(nxt)
    leaf_dists[leaf] = dist
    for other in leaves:
      if (
          leaf != other
          and other in dist
          and joints[leaf].sticks[0].name != joints[other].sticks[0].name
      ):
        max_path_inset = max(
            max_path_inset, dist[other] + leaf_inset[leaf] + leaf_inset[other]
        )
        if dist[other] > 0:
          all_pair_dists.add(dist[other])

  leaf_lx = dict(leaf_leff)
  if is_sym:
    for leaf in leaves:
      stick_name = joints[leaf].sticks[0].name
      twin_name = sym_map.get(stick_name)
      if twin_name and twin_name in leaf_by_stick:
        twin_leaf = leaf_by_stick[twin_name]
        twin_dist = leaf_dists[leaf].get(twin_leaf, 0)
        leaf_lx[leaf] = max(leaf_leff[leaf], (twin_dist + 1) // 2)
      if stick_name in left_half:
        min_dim = max(min_dim, 2 * (leaf_lx[leaf] + leaf_inset[leaf]))

  root = max(joints, key=lambda name: joints[name].degree)
  parent_by_joint: dict[str, tuple[str, int] | None] = {root: None}
  root_order: list[str] = []
  queue = collections.deque([root])
  while queue:
    cur = queue.popleft()
    root_order.append(cur)
    for nxt, weight in weighted_adj[cur]:
      if nxt not in parent_by_joint:
        parent_by_joint[nxt] = (cur, weight)
        queue.append(nxt)

  square_pigeonhole = 0
  if len(leaves) >= 5:
    for inset_floor in sorted(set(leaf_inset.values())):
      eligible = {u for u in leaves if leaf_inset[u] >= inset_floor}
      if len(eligible) < 5:
        continue
      for dist_min in sorted(all_pair_dists):
        indep = _max_separated_leaves(
            root_order, parent_by_joint, weighted_adj, eligible, dist_min
        )
        if indep >= 5:
          side = math.isqrt(indep - 1)
          square_pigeonhole = max(
              square_pigeonhole, side * dist_min + 2 * inset_floor
          )

  sym_constraints: list[_SymRectConstraint] = []
  if left_half:
    left_leaves = [u for u in leaves if joints[u].sticks[0].name in left_half]
    if len(left_leaves) >= 2:
      left_insets = sorted({leaf_inset[u] for u in left_leaves})
      left_lxs = sorted({leaf_lx[u] for u in left_leaves})
      left_dists = sorted({
          dist_val
          for a, b in itertools.combinations(left_leaves, 2)
          if (dist_val := leaf_dists[a].get(b, 0)) > 0
      })
      for s_fl in left_insets:
        for l_fl in left_lxs:
          eligible = {
              u
              for u in left_leaves
              if leaf_inset[u] >= s_fl and leaf_lx[u] >= l_fl
          }
          if len(eligible) < 2:
            continue
          for dist_min in left_dists:
            indep = _max_separated_leaves(
                root_order, parent_by_joint, weighted_adj, eligible, dist_min
            )
            if indep >= 2:
              sym_constraints.append(
                  _SymRectConstraint(
                      count=indep,
                      dist_min=dist_min,
                      inset_floor=s_fl,
                      lx_floor=l_fl,
                  )
              )

  return max(min_dim, max_path_inset, square_pigeonhole), sym_constraints


def _satisfies_sym_rect_pigeonhole(
    grid_size: int, constraints: Sequence[_SymRectConstraint]
) -> bool:
  """Checks left-half `[s, N/2 - lx] x [s, N - s]` pigeonhole bounds.

  Any `K` left-half leaves with pairwise distance `>= D` in
  `[s, N/2 - lx] x [s, N - s]` require
  `ceil((N/2 - lx - s + 1) / D) * ceil((N - 2*s + 1) / D) >= K`.

  Args:
    grid_size: Candidate even grid size `N`.
    constraints: Rectangular pigeonhole constraints.

  Returns:
    True if all symmetric half-grid rectangular pigeonhole bounds hold.
  """
  if grid_size % 2 == 1:
    return False
  half = grid_size // 2
  for constraint in constraints:
    if constraint.dist_min <= 0:
      continue
    box_w = half - constraint.lx_floor - constraint.inset_floor + 1
    box_h = grid_size - 2 * constraint.inset_floor + 1
    if box_w <= 0 or box_h <= 0:
      return False
    buckets_x = (box_w + constraint.dist_min - 1) // constraint.dist_min
    buckets_y = (box_h + constraint.dist_min - 1) // constraint.dist_min
    if buckets_x * buckets_y < constraint.count:
      return False
  return True


def calculate_v4_grid_size(
    stick_figure: _StickFigureProtocol,
    sticks: Sequence[_StickProtocol] | None = None,
    use_longest_path: bool = True,
    use_level_shifter: bool = False,
    *,
    is_symmetric: bool | None = None,
    flap_dim_fn: Callable[[Any, bool], int] | None = None,
) -> int:
  """Computes a Chebyshev (`D_inf`) pigeonhole lower bound on packing grid size.

  Model assumptions:
  - Backtracking square-flap placement checks
  (`flap_placement.validate_candidate`
    for integer grid-aligned centers, border/interior corner reachability, and
    non-overlapping squares), plus any inset margin returned by `flap_dim_fn`.
  - Mirrored symmetric placement (`[0, N/2]` off-axis confinement and even `N`)
    when `is_symmetric` (or `stick_figure.is_symmetric_tree` before chain
    merging when `is_symmetric is None`) is True. Does not apply when
    `allow_paired_axis_placement=True` (for example, `sym_eight_wings` packs at
    grid 8 under paired axis placement).
  - Square flaps (`2l x 2l`), not `shaping_flap` sticks (which pack as
    `2l x l'` rectangles and can pack below `'v2'` and `'v4'`) or solvers
    whose flap dimensions reserve fewer cells.

  Args:
    stick_figure: Stick figure to lower-bound.
    sticks: Optional subset of sticks to bound instead of the full tree.
    use_longest_path: Whether to enforce the tree-diameter lower bound.
    use_level_shifter: Whether to pass level-shifter state to `flap_dim_fn`.
    is_symmetric: Whether mirrored symmetric-packing bounds apply (defaults to
      `stick_figure.is_symmetric_tree` prior to chain merging).
    flap_dim_fn: Optional `(stick, use_level_shifter) -> int` callback (such as
      `utils_packing_flaps.get_flap_dim`) returning the packed flap side. When
      `None`, zero extra flap margin is assumed.

  Returns:
    An integer lower bound on the packing grid side (`N`).
  """
  is_sym = (
      stick_figure.is_symmetric_tree if is_symmetric is None else is_symmetric
  )
  working_figure = stick_figure
  is_full_tree = sticks is None
  if is_full_tree and any(
      joint.degree == 2 for joint in stick_figure.joints.values()
  ):
    working_figure, _ = stick_figure.merge_linear_chains()
  sticks_to_consider = sticks if sticks is not None else working_figure.children

  axis_names, sym_map, left_half = (
      _extract_symmetry_info(working_figure, use_level_shifter, flap_dim_fn)
      if is_sym
      else (set(), {}, set())
  )

  lower_bound, sym_constraints = _compute_tree_pigeonhole_floor(
      working_figure,
      sticks_to_consider,
      is_sym=is_sym,
      axis_names=axis_names,
      sym_map=sym_map,
      left_half=left_half,
      use_level_shifter=use_level_shifter,
      flap_dim_fn=flap_dim_fn,
      is_full_tree=is_full_tree,
  )
  if use_longest_path:
    lower_bound = max(
        lower_bound, longest_path_in_joints(working_figure.joints)
    )
  if is_sym and lower_bound % 2 == 1:
    lower_bound += 1

  while (
      is_sym
      and sym_constraints
      and not _satisfies_sym_rect_pigeonhole(lower_bound, sym_constraints)
  ):
    lower_bound += 2

  return lower_bound
