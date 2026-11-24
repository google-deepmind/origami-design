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

"""Utility functions for box pleating hinges."""

from collections.abc import Callable, Collection, Iterable, Iterator, Mapping, MutableMapping, Sequence
import copy
import dataclasses
import functools
import heapq
import itertools
import json
import math
import time
from typing import Any

from absl import logging
from bp_pipeline import bus
from bp_pipeline import plugins
from bp_pipeline import rectangle as rectangle_lib
from bp_pipeline import timeout as timeout_lib
from bp_pipeline import utils_tree
from bp_pipeline.crease_pattern import crease
from bp_pipeline.crease_pattern import crease_pattern
from bp_pipeline.crease_pattern import global_foldability
from bp_pipeline.crease_pattern import point
from bp_pipeline.crease_pattern import symmetry
from bp_pipeline.crease_pattern import vertex
from bp_pipeline.solving import central_fold_utils
from bp_pipeline.solving import partition as partition_utils
from bp_pipeline.solving import pleats as pleat_utils
from bp_pipeline.solving import ridges as ridge_utils
from bp_pipeline.solving import utils as box_pleating_utils

packing_output = central_fold_utils.packing_output
BaseType = bus.BaseType


def _serialize_solver_state(
    state: State | SolverStateSnapshot,
) -> dict[str, Any]:
  """Serializes the public fields shared by State and SolverStateSnapshot."""
  return {
      'pleats': [c.asdict() for c in state.pleats],
      'unassigned_pleats': [c.asdict() for c in state.unassigned_pleats],
      'hinges': [c.asdict() for c in state.hinges],
      'unsplit_hinges': [c.asdict() for c in state.unsplit_hinges],
      'unvisited_hinges': [c.asdict() for c in state.unvisited_hinges],
      'state_score': state.state_score,
      'ridges': [c.asdict() for c in state.ridges],
      'expanded': state.expanded,
      'solver_config': state.solver_config.asdict(),
      'transform': state.transform.name,
      'partition_progress': state.partition_progress,
  }


def _deserialize_crease_list(
    raw_items: str | bytes | list[Any],
) -> list[crease.Crease]:
  """Deserializes a list of creases from dicts or a JSON-encoded list."""
  items = (
      json.loads(raw_items) if isinstance(raw_items, str | bytes) else raw_items
  )
  return [  # pyrefly: ignore[bad-argument-type]
      crease.Crease.from_dict(item) for item in items
  ]


@dataclasses.dataclass(kw_only=True)
class SolverStateSnapshot:
  """Lightweight final-state snapshot for SolverResult without search-tree internals."""

  pleats: list[crease.Crease]
  unassigned_pleats: list[crease.Crease] = dataclasses.field(
      default_factory=list
  )
  hinges: list[crease.Crease] = dataclasses.field(default_factory=list)
  unsplit_hinges: list[crease.Crease] = dataclasses.field(default_factory=list)
  unvisited_hinges: list[crease.Crease] = dataclasses.field(
      default_factory=list
  )
  ridges: list[crease.Crease] = dataclasses.field(default_factory=list)
  solver_config: SolverConfig
  transform: symmetry.Transform = symmetry.Transform.IDENTITY
  state_score: int = 0
  partition_progress: int = 0
  expanded: bool = True
  _symmetry_score_cache: float | None = dataclasses.field(
      default=None, init=False, repr=False, compare=False
  )

  def symmetry_score(self) -> float:
    return hinge_symmetry_score(self)

  def asdict(self) -> dict[str, Any]:
    """Converts the snapshot to a dictionary matching State.asdict()."""
    return _serialize_solver_state(self)


@dataclasses.dataclass(kw_only=True, frozen=True)
class SolverResult:
  """Holds a successful solution from the solver."""

  crease_pattern: crease_pattern.CreasePattern
  final_state: State | SolverStateSnapshot
  is_globally_flat_foldable: bool = True
  time_taken: float = 0.0  # Time in seconds to find the solution.
  base_type: BaseType = BaseType.NORMAL


is_stick_up = utils_tree.is_stick_up


@dataclasses.dataclass(kw_only=True, frozen=True)
class HingeMetaData:
  """Holds information about a hinge that is used for solving."""

  hinge: crease.Crease
  segments: list[crease.Crease]
  central_pleats: list[crease.Crease]
  solver_config: SolverConfig | None = None
  ban_symmetric: bool = False
  force_non_flat: bool = False
  force_allow_symmetric: bool = False
  _assigned_segments_cache: dict[
      box_pleating_utils.HingeAssignmentType, tuple[crease.Crease, ...]
  ] = dataclasses.field(
      default_factory=dict, init=False, repr=False, compare=False
  )

  def get_assigned_segments(
      self, assignment_type: box_pleating_utils.HingeAssignmentType
  ) -> tuple[crease.Crease, ...]:
    """Returns cached assigned segments for the given assignment type."""
    cached = self._assigned_segments_cache.get(assignment_type)
    if cached is None:
      cached = tuple(assignment_type.assign_segments(self.segments))
      self._assigned_segments_cache[assignment_type] = cached
    return cached

  @functools.cached_property
  def simple(self) -> bool:
    """Returns whether the hinge is simple to solve."""
    return self.hinge.intersects_border and not bool(self.central_pleats)

  @functools.cached_property
  def has_central_flap(self) -> bool:
    """Returns whether the hinge is associated with a central flap.

    A hinge has a central flap if it overlaps with any rectangle in the packing
    information that is classified as an INTERNAL flap type.

    Returns:
      True if the hinge is associated with a central flap, False otherwise.
    """
    if not self.solver_config or not isinstance(
        self.solver_config.packing_information, bus.PackedCreasePattern
    ):
      return False

    packing_info = self.solver_config.packing_information
    if not packing_info.flaps:
      return False
    for flap in packing_info.flaps.values():
      if flap.flap_type == rectangle_lib.FlapType.INTERNAL:
        normalized_rect = flap.rect.normalized(packing_info.grid_size)
        if any(
            self.hinge.is_overlapping(edge) for edge in normalized_rect.edges
        ):
          return True
    return False

  @functools.cached_property
  def allow_symmetric(self) -> bool:
    """Decides if the hinge could be assigned as symmetric.

    A symmetric hinge is trapping a flap in another flap. For that to happen
    it must have an even number of segments, and have a central flap.

    If disallow_trapping_edge_flaps is True, we disallow symmetric
    hinges if there are no central flaps. This will prevent cases where
    a hinge is assigned as symmetric between an edge flap and a river to
    prevent trapping edge flaps.

    Returns:
      True if the hinge could be assigned as symmetric.
    """
    # Hinges on the symmetry axis cannot be symmetric, as they introduce an
    # asymmetric pleat.
    if self.ban_symmetric:
      return False
    if (
        self.solver_config
        and self.solver_config.remove_central_crease
        and self.hinge.midpoint.x == 0.5
    ):
      return self.hinge.is_horizontal
    if self.force_allow_symmetric:
      return len(self.segments) % 2 == 0 and bool(self.central_pleats)
    if self.solver_config and isinstance(
        self.solver_config.packing_information, bus.PackedCreasePattern
    ):
      packing_info = self.solver_config.packing_information
    else:
      packing_info = None
    if self.solver_config and packing_info and packing_info.flaps:
      if (
          self.solver_config.disallow_trapping_edge_flaps
          and not self.has_central_flap
      ):
        # In general, we disallow symmetric hinges between edge flaps and rivers
        # to prevent trapping edge flaps. We allow an exception if the hinge is
        # on the symmetry axis and the solver is configured to remove the
        # central crease since the trapped flap will be freed in the central
        # fold removal.
        if not (
            self.solver_config.remove_central_crease
            and math.isclose(self.hinge.midpoint.x, 0.5)
        ):
          return False
      # If there is only one stick, this is an internal river hinge and cannot
      # be symmetric.
      if len(self._sticks) == 1:
        return False
      elif len(self._sticks) == 2:
        # Never trap a river in another river.
        if self._sticks[0].is_river and self._sticks[1].is_river:
          return False
        # If one of the sticks is a river, and the other is longer, the hinge is
        # flat. In other words, we never trap a river, and a longer flap can't
        # fit in the river.
        elif (
            self._sticks[0].is_river
            and self._sticks[1].length > self._sticks[0].length
        ):
          return False
        elif (
            self._sticks[1].is_river
            and self._sticks[0].length > self._sticks[1].length
        ):
          return False
    return len(self.segments) % 2 == 0 and bool(self.central_pleats)

  @functools.cached_property
  def _sticks(self) -> list[Any]:
    """The river/flap sticks that intersect the hinge."""
    if not self.solver_config or not isinstance(
        self.solver_config.packing_information, bus.PackedCreasePattern
    ):
      return []
    packing_info = self.solver_config.packing_information
    if not packing_info.hinges or packing_info.packed_tree is None:
      return []
    matched = packing_info.hinge_at.find_overlapping(self.hinge)
    if matched is None:
      return []
    sticks = []
    for elem_id in matched.elements:
      if elem_id in packing_info.flaps:
        stick_name = packing_info.flaps[elem_id].stick
      elif elem_id in packing_info.rivers:
        stick_name = packing_info.rivers[elem_id].stick
      else:
        stick_name = elem_id
      sticks.append(packing_info.packed_tree[stick_name])
    if len(sticks) not in (1, 2):
      raise ValueError(
          f'Expected hinge to match 1 or 2 sticks, got {len(sticks)}'
      )
    return sticks

  @functools.cached_property
  def flat(self) -> bool:
    """Returns whether the hinge is flat.

    A hinge is flat, and not assigned during solving, if one of the sticks is
    up and the other is down. For horizontal rivers, the up/down orientation is
    determined by whether the joint is at the start or end of the stick.
    """
    if (
        self.force_non_flat
        or not self.solver_config
        or not self.solver_config.packing_information
        or not self.solver_config.hinges_from_stick_figure
    ):
      return False
    if self._sticks:
      # If one stick is up and the other is down, the hinge is flat.
      if len(self._sticks) == 1:
        # This is an internal river hinge and could be assigned.
        if not self._sticks[0].is_river:
          raise ValueError(
              f'Single-stick hinge expected a river, got {self._sticks[0]}'
          )
        return False
      if len(self._sticks) != 2:
        raise ValueError(
            f'Expected 2 sticks for boundary hinge, got {len(self._sticks)}'
        )
      if any(stick.shaping_flap for stick in self._sticks):
        # If one of the sticks is a shaping flap, the up/down orientation
        # cannot be determined.
        return False
      # Find the shared joint between the two sticks.
      joints_1 = self._sticks[0].start, self._sticks[0].end
      joints_2 = self._sticks[1].start, self._sticks[1].end
      shared_joint = set(joints_1).intersection(joints_2)
      if self.solver_config.allow_constraint_relaxation:
        if len(shared_joint) > 1:
          raise ValueError(
              f'Expected at most 1 shared joint, got {shared_joint}'
          )
        if len(shared_joint) != 1:
          # Non-sibling flaps touching along a 2D packing boundary do not share
          # a stick-figure joint, so their boundary hinge is not forced flat.
          return False
      elif len(shared_joint) != 1:
        raise ValueError(
            f'Expected exactly 1 shared joint between {self._sticks}, got'
            f' {shared_joint}'
        )
      joint = shared_joint.pop()
      # Use the joint to compute the azimuth angle of each stick and the up/down
      # orientation.
      up_down_method = self.solver_config.up_down_method
      up_1 = is_stick_up(self._sticks[0], joint, up_down_method)
      up_2 = is_stick_up(self._sticks[1], joint, up_down_method)
      # A hinge is flat if one stick is up and the other is down.
      return up_1 ^ up_2
    return False


@dataclasses.dataclass(kw_only=True)
class VertexNode:
  """Holds the state of a vertex while we are solving it."""

  vertex_: vertex.Vertex
  score: int
  central_flap: bool
  tip_hashes: set[int] = dataclasses.field(default_factory=set)
  _cloned: bool = True
  _mv_counts: tuple[int, int] | None = dataclasses.field(
      default=None, repr=False, compare=False
  )

  @property
  def creases(self) -> list[crease.Crease]:
    """Returns the creases of the underlying vertex."""
    return self.vertex_.creases

  def ensure_cloned(self) -> vertex.Vertex:
    """Clones vertex_ on first mutation after State.copy_state."""
    if not self._cloned:
      self.vertex_ = self.vertex_.clone()
      self._cloned = True
    self._mv_counts = None
    return self.vertex_

  def add_crease(self, crease_: crease.Crease) -> None:
    """Clones vertex_ on first mutation and adds a crease."""
    self.ensure_cloned().add_crease(crease_)

  def remove_crease(self, crease_: crease.Crease) -> None:
    """Clones vertex_ on first mutation and removes a crease."""
    self.ensure_cloned().remove_crease(crease_)

  def mv_counts(self) -> tuple[int, int]:
    """Returns cached (n_mountains, n_valleys) of vertex_.creases."""
    if self._mv_counts is None:
      n_m = 0
      n_v = 0
      for c in self.vertex_.creases:
        if c.fold == crease.Fold.MOUNTAIN:
          n_m += 1
        elif c.fold == crease.Fold.VALLEY:
          n_v += 1
      self._mv_counts = (n_m, n_v)
    return self._mv_counts

  def copy_shallow(self) -> 'VertexNode':
    """Returns a copy-on-write clone of this VertexNode."""
    return VertexNode(
        vertex_=self.vertex_,
        score=self.score,
        central_flap=self.central_flap,
        tip_hashes=self.tip_hashes,
        _cloned=False,
        _mv_counts=self._mv_counts,
    )

  @classmethod
  def from_hinges_and_pleats(
      cls,
      vertex_: vertex.Vertex,
      hinges: list[crease.Crease],
      unassigned_pleats: list[crease.Crease],
      pleats: list[crease.Crease],
      *,
      check_maekawa: bool = False,
      hinges_tip_index: box_pleating_utils.TipIndex | None = None,
      all_tip_index: box_pleating_utils.TipIndex | None = None,
  ) -> 'VertexNode':
    """Creates a VertexNode, determining if it represents a central flap."""
    if hinges_tip_index is not None:
      central_flap = not hinges_tip_index.has_tip(vertex_.point)
    else:
      central_flap = not any(h.is_tip(vertex_.point) for h in hinges)
    score = vertex_score(
        vertex_, unassigned_pleats, check_maekawa=check_maekawa
    )
    if all_tip_index is not None:
      tip_creases = all_tip_index.creases_at_tip(vertex_.point)
    else:
      tip_creases = [
          c
          for c in pleats + unassigned_pleats + hinges
          if c.is_tip(vertex_.point)
      ]
    tip_hashes = {crease_.hash_value for crease_ in tip_creases}
    return cls(
        vertex_=vertex_,
        score=score,
        central_flap=central_flap,
        tip_hashes=tip_hashes,
    )

  def is_tip(self, crease_: crease.Crease) -> bool:
    return crease_.hash_value in self.tip_hashes


def _four_ridges_vertex(
    vertex_: vertex.Vertex,
    ridges: list[crease.Crease],
    *,
    ridges_tip_index: box_pleating_utils.TipIndex | None = None,
) -> bool:
  """Returns true if the vertex intersects 4 ridges.

  This is often the case for vertices that are the center of a central flap.

  Args:
    vertex_: The vertex to check.
    ridges: The ridges that intersect the vertex.
    ridges_tip_index: Optional spatial tip index over ridges.

  Returns:
    True if the vertex intersects 4 ridges.
  """
  if ridges_tip_index is not None:
    return ridges_tip_index.count_at_tip(vertex_.point) == 4
  return sum(1 for r in ridges if r.is_tip(vertex_.point)) == 4


def _has_misaligned_pleat_intersection(
    unassigned_pleats: Sequence[crease.Crease],
    creases: Sequence[crease.Crease],
    recovery: bool = True,
) -> bool:
  """Checks if any pleat has a non-midpoint internal intersection with creases."""
  for pleat in unassigned_pleats:
    for c in creases:
      v = pleat.vertex_with_crease(c)
      if v:
        is_tip = v.isclose(c.midpoint) or c.is_tip(v)
        if recovery:
          is_tip = is_tip or pleat.is_tip(v)
        if not is_tip:
          return True
  return False


def _filter_encompassing_creases(
    target: crease.Crease, candidates: Sequence[crease.Crease]
) -> list[crease.Crease]:
  """Filters candidate creases whose bounding boxes overlap target."""
  return [c for c in candidates if not c.not_in_encompassing_rectangles(target)]


def _filter_symmetric_partner_creases(
    creases: Sequence[crease.Crease],
) -> list[crease.Crease]:
  """Finds creases with parallel same-fold tip partners."""
  symmetric_creases = []
  for i, c1 in enumerate(creases):
    for j, c2 in enumerate(creases):
      if i == j:
        continue
      if c1.intersect_at_tip(c2) and c1.fold == c2.fold and c1.is_parallel(c2):
        symmetric_creases.append(c1)
        break
  return symmetric_creases


class StateValueError(ValueError):
  """Value error that passes up an associated state."""

  def __init__(self, message: str, state: 'State'):
    super().__init__(message)
    self.state = state


@dataclasses.dataclass(kw_only=True)
class State:
  """Holds the state of the crease pattern while we are solving it.

  Attributes:
    pleats: The pleats that have been added to the crease pattern.
    unassigned_pleats: The pleats that have been added but not been assigned.
    hinges: The hinges that have been added to the crease pattern.
    unsplit_hinges: The hinges that have been added to the crease pattern, but
      not split to segments and not assigned.
    unvisited_hinges: The hinges that have not been added to the crease pattern.
    vertices: The vertices that require solving in the crease pattern.
    state_score: The sum of the vertices score.
    ridges: The ridges that have been added to the crease pattern.
    grid_points: The grid points.
    expanded: Whether the state has been expanded.
    solver_config: The solver config.
    hinge_dict: A dictionary mapping the hinge hash to the hinge metadata.
    transform: The symmetry transform with the highest score based on the hinges
    partition_progress: The number of solved partitions.
    partition_order: The order that the hinge partitions should be solved in.
    relaxed_hinges: Hinge relaxation configurations applied to the state.
  """

  pleats: list[crease.Crease]
  unassigned_pleats: list[crease.Crease]
  hinges: list[crease.Crease]
  unsplit_hinges: list[crease.Crease]
  unvisited_hinges: list[crease.Crease]
  vertices: dict[point.Point, VertexNode]
  state_score: int
  ridges: list[crease.Crease]
  grid_points: tuple[point.Point, ...]
  expanded: bool = False
  solver_config: SolverConfig
  hinge_dict: dict[int, HingeMetaData]
  transform: symmetry.Transform
  partition_progress: int = 0
  partition_order: partition_utils.OrderedPartitions | None
  relaxed_hinges: frozenset[tuple[int, bool, bool]] = frozenset()
  _hash_cache: int | None = dataclasses.field(
      default=None, init=False, repr=False, compare=False
  )
  _symmetry_score_cache: float | None = dataclasses.field(
      default=None, init=False, repr=False, compare=False
  )
  _tip_vertex_points_cache: dict[
      tuple[point.Point, point.Point], tuple[point.Point, ...]
  ] = dataclasses.field(
      default_factory=dict, init=False, repr=False, compare=False
  )
  _pleat_point_status_cache: dict[
      tuple[point.Point, point.Point], tuple[tuple[point.Point, bool], ...]
  ] = dataclasses.field(
      default_factory=dict, init=False, repr=False, compare=False
  )
  _pleat_blocking_hinges_cache: dict[
      crease.CreaseWithExactMatch, tuple[crease.Crease, ...]
  ] = dataclasses.field(
      default_factory=dict, init=False, repr=False, compare=False
  )
  _pleat_candidate_ridges_cache: dict[
      tuple[int, int, crease.CreaseWithExactMatch], tuple[crease.Crease, ...]
  ] = dataclasses.field(
      default_factory=dict, init=False, repr=False, compare=False
  )
  _pleat_misaligned_ridges_cache: dict[
      tuple[int, int, crease.CreaseWithExactMatch], bool
  ] = dataclasses.field(
      default_factory=dict, init=False, repr=False, compare=False
  )
  _partition_vertex_points: tuple[tuple[point.Point, ...], ...] | None = (
      dataclasses.field(default=None, init=False, repr=False, compare=False)
  )

  def invalidate_caches(self) -> None:
    """Invalidates cached hash and symmetry score after state mutation."""
    self._hash_cache = None
    self._symmetry_score_cache = None

  def _get_tip_vertex_points(
      self, crease_: crease.Crease
  ) -> tuple[point.Point, ...]:
    """Returns cached vertex points in self.vertices that are tips of crease_."""
    key = (crease_.p1, crease_.p2)
    pts = self._tip_vertex_points_cache.get(key)
    if pts is None:
      pts = tuple(vp for vp in self.vertices if crease_.is_tip(vp))
      self._tip_vertex_points_cache[key] = pts
    return pts

  def asdict(self) -> dict[str, Any]:
    """Converts the state to a dictionary for serialization."""
    # We skip the serialization of vertices, hinge_dict and partition_order as
    # they are unused in reconstruction.
    return _serialize_solver_state(self)

  @classmethod
  def from_dict(cls, state_dict: dict[str, Any]) -> 'State':
    """Creates a State from a dictionary."""
    # For now, vertices, hinge_dict and partition_order are not deserialized.
    solver_config = SolverConfig(**state_dict['solver_config'])
    return cls(
        pleats=_deserialize_crease_list(state_dict['pleats']),
        unassigned_pleats=_deserialize_crease_list(
            state_dict['unassigned_pleats']
        ),
        hinges=_deserialize_crease_list(state_dict['hinges']),
        unsplit_hinges=_deserialize_crease_list(state_dict['unsplit_hinges']),
        unvisited_hinges=_deserialize_crease_list(
            state_dict['unvisited_hinges']
        ),
        state_score=state_dict['state_score'],
        ridges=_deserialize_crease_list(state_dict['ridges']),
        grid_points=box_pleating_utils.create_grid_points(
            solver_config.grid_size
        ),
        expanded=state_dict['expanded'],
        solver_config=solver_config,
        transform=symmetry.Transform[state_dict['transform']],
        partition_progress=state_dict['partition_progress'],
        hinge_dict={},  # Not deserialized.
        vertices={},  # Not deserialized.
        partition_order=None,  # Not deserialized.
    )

  def copy_state(self) -> 'State':
    """Returns a copy of the state.

    Most of the attributes are copied, ridges are not copied
    because they are never modified. Vertices are deep copied as we may modify
    the creases in the vertex.

    Returns:
      A copy of the state.
    """
    vertices = {
        key: vertex_node.copy_shallow()
        for key, vertex_node in self.vertices.items()
    }
    state = State(
        pleats=copy.copy(self.pleats),
        unassigned_pleats=copy.copy(self.unassigned_pleats),
        hinges=copy.copy(self.hinges),
        unsplit_hinges=copy.copy(self.unsplit_hinges),
        unvisited_hinges=copy.copy(self.unvisited_hinges),
        vertices=vertices,
        state_score=self.state_score,
        ridges=self.ridges,
        grid_points=self.grid_points,
        expanded=False,
        solver_config=self.solver_config,
        hinge_dict=self.hinge_dict,
        transform=self.transform,
        partition_progress=self.partition_progress,
        partition_order=self.partition_order,
        relaxed_hinges=self.relaxed_hinges,
    )
    # pylint: disable=protected-access
    state._tip_vertex_points_cache = self._tip_vertex_points_cache
    state._pleat_point_status_cache = self._pleat_point_status_cache
    state._pleat_blocking_hinges_cache = self._pleat_blocking_hinges_cache
    state._pleat_candidate_ridges_cache = self._pleat_candidate_ridges_cache
    state._pleat_misaligned_ridges_cache = self._pleat_misaligned_ridges_cache
    state._partition_vertex_points = self._partition_vertex_points
    # pylint: enable=protected-access
    return state

  @classmethod
  def init_state(
      cls,
      pleats: list[crease.Crease],
      unassigned_pleats: list[crease.Crease],
      hinges: list[crease.Crease],
      ridges: list[crease.Crease],
      borders: list[crease.Crease],
      solver_config: SolverConfig,
      partition_order: partition_utils.OrderedPartitions | None,
  ) -> 'State':
    """Returns a new state with no hinges or pleats."""
    grid_size = solver_config.grid_size
    if grid_size is None:
      border_coords = [
          int(max(c.p1.x, c.p2.x, c.p1.y, c.p2.y))
          for c in borders or pleats or ridges
      ]
      grid_size = max(border_coords) if border_coords else 0
    init_cp = crease_pattern.CreasePattern(
        borders + pleats + ridges, grid_size=grid_size
    )
    grid_points = box_pleating_utils.create_grid_points(grid_size)
    ridges_tip_index = box_pleating_utils.TipIndex(ridges, grid_size=grid_size)
    hinges_tip_index = box_pleating_utils.TipIndex(hinges, grid_size=grid_size)
    all_tip_index = box_pleating_utils.TipIndex(
        pleats + unassigned_pleats + hinges, grid_size=grid_size
    )
    vertices_ = [
        v
        for v in init_cp.vertices
        if not v.flat_foldability(check_maekawa=solver_config.recovery)
        or _four_ridges_vertex(v, ridges, ridges_tip_index=ridges_tip_index)
    ]
    vertices_ = _vertices_from_crease_intersections(init_cp, hinges, vertices_)
    vertices = {
        v.point: VertexNode.from_hinges_and_pleats(
            v,
            hinges,
            unassigned_pleats,
            pleats,
            check_maekawa=solver_config.recovery,
            hinges_tip_index=hinges_tip_index,
            all_tip_index=all_tip_index,
        )
        for v in vertices_
    }
    if hinges:
      hinge_dict = create_hinge_dict(
          unvisited_hinges=hinges,
          unassigned_pleats=unassigned_pleats,
          pleats=pleats,
          grid_points=grid_points,
          vertices=vertices,
          solver_config=solver_config,
      )
      transform = symmetry.symmetry_axis(
          crease_pattern.CreasePattern(hinges, light_mode=True)
      )
    else:
      hinge_dict = {}
      transform = symmetry.Transform.IDENTITY

    state = cls(
        pleats=pleats[:],
        unassigned_pleats=unassigned_pleats[:],
        unvisited_hinges=hinges[:],
        hinges=[],
        unsplit_hinges=[],
        vertices=vertices,
        state_score=sum(v.score for v in vertices.values()),
        ridges=ridges[:],
        grid_points=grid_points,
        expanded=False,
        solver_config=solver_config,
        hinge_dict=hinge_dict,
        transform=transform,
        partition_order=partition_order,
    )
    if solver_config.recovery:
      state.update_partition_progress()
    return state

  def __hash__(self):
    if self._hash_cache is None:
      base_hash = hash(
          crease.Creases.create_from_creases(creases=self.hinges + self.pleats)
      )
      self._hash_cache = (
          hash((base_hash, self.relaxed_hinges))
          if self.relaxed_hinges
          else base_hash
      )
    return self._hash_cache

  def __eq__(self, other):
    if self is other:
      return True
    return (
        self.relaxed_hinges == other.relaxed_hinges
        and crease.Creases.create_from_creases(
            creases=self.hinges + self.pleats
        )
        == crease.Creases.create_from_creases(
            creases=other.hinges + other.pleats
        )
    )

  def remove_central_crease(self, transform: symmetry.Transform) -> bool:
    """Removes central creases from the state.

    This technique is often used to make the body of the model wider (2 grids
    instead of 1), releasing central flaps and making the problem easier.

    See https://abrashiorigami.com/why-is-central-fold-opening-so-popular/

    The technique depends on what type of creases (axial, hinge, ridge) occupy
    the symmetry axis, each of this cases requires a different implementation.

    Args:
      transform: The symmetry transform to remove the central creases.

    Returns:
      True if the central crease was removed.
    """
    if transform not in [
        symmetry.Transform.HORIZONTAL_FLIP,
        symmetry.Transform.VERTICAL_FLIP,
    ]:
      raise ValueError(
          f'Transform {transform} is not supported for central crease removal.'
      )

    self.invalidate_caches()

    # Remove pleats and hinges on the symmetry axis.
    self.pleats, removed_pleats = (
        central_fold_utils.remove_creases_overlapping_central_crease(
            self.pleats, transform
        )
    )
    self.ridges, removed_ridges = (
        central_fold_utils.remove_creases_overlapping_central_crease(
            self.ridges, transform
        )
    )
    self.hinges, removed_hinges = (
        central_fold_utils.remove_creases_overlapping_central_crease(
            self.hinges, transform
        )
    )

    _, removed_unsplit_hinges = (
        central_fold_utils.remove_creases_overlapping_central_crease(
            self.unsplit_hinges, transform
        )
    )

    central_crease = central_fold_utils.central_crease_from_transform(transform)
    if not central_fold_utils.is_valid_central_crease_removal(
        removed_pleats + removed_ridges + removed_hinges,
        central_crease,
    ):
      self.expanded = True
      return False
    self.pleats = central_fold_utils.reassign_pleats(
        self.pleats,
        removed_unsplit_hinges,
        self.solver_config.grid_size,
        self.ridges,
    )
    self.ridges.extend(
        central_fold_utils.add_diagonals_to_removed_ridges(
            removed_ridges, self.solver_config.grid_size
        )
    )

    self.ridges = (
        central_fold_utils.add_diagonals_to_removed_hinges_and_shorten_flaps(
            self.ridges, removed_unsplit_hinges, self.solver_config.grid_size
        )
    )

    return True

  def in_current_partition(self, point_: point.Point) -> bool:
    """Checks if the vertex point is in the current partition."""
    if self.partition_order is not None:
      return (
          self.partition_order.get_partition_id(point_)
          == self.partition_progress
      )
    return True

  def allow_simple_hinges(self) -> bool:
    """Whether to allow adding hinges that are simple to solve.

    A simple hinge intersect border and does not have a central pleat.
    Such hinges are easiest to solve, as they only influence a single vertex.
    We delay adding them until the end to speed up the search.

    Returns:
      True if hinges that intersect the border can be added.
    """
    unsolved_in_part = [
        v
        for vertex_point, v in self.vertices.items()
        if not v.score and self.in_current_partition(vertex_point)
    ]
    if not unsolved_in_part:
      return True
    if not self.solver_config.recovery:
      simple_tip_hashes = {
          h.hash_value
          for h in self.unvisited_hinges
          if self.hinge_dict[h.hash_value].simple
      }
      return all(
          not v.tip_hashes.isdisjoint(simple_tip_hashes)
          for v in unsolved_in_part
      )
    simple_tip_hashes: set[int] = set()
    non_simple_touch_hashes: set[int] = set()
    hinge_dict = self.hinge_dict
    for h in self.unvisited_hinges:
      h_hash = h.hash_value
      h_meta = hinge_dict[h_hash]
      if h_meta.flat:
        continue
      if h_meta.simple:
        simple_tip_hashes.add(h_hash)
      elif self.in_current_partition(h.p1):
        non_simple_touch_hashes.add(h_hash)
        if h_meta.allow_symmetric:
          for cp_ in h_meta.central_pleats:
            non_simple_touch_hashes.add(cp_.hash_value)
    if not non_simple_touch_hashes:
      return True
    for v in unsolved_in_part:
      tip_hashes = v.tip_hashes
      if not tip_hashes.isdisjoint(simple_tip_hashes):
        continue
      if not tip_hashes.isdisjoint(non_simple_touch_hashes):
        return False
    return True

  def get_valid_hinges(self) -> list[crease.Crease]:
    """Returns the next hinges of the current state.

    Instead of expanding all the remaining hinges, we delay expanding some
    hinges. When partitions are used, this method only returns hinges that
    belong to the current partition being solved.

    Returns:
      The next hinges of the current state.
    """
    next_hinges = [
        h
        for h in self.unvisited_hinges
        if self.in_current_partition(h.p1)
        and not self.hinge_dict[h.hash_value].flat
    ]
    if (
        self.solver_config.recovery
        and not next_hinges
        and self.partition_order is not None
        and self.partition_progress < len(self.partition_order.partitions)
    ):
      unsolved_in_part = [
          vn
          for vp, vn in self.vertices.items()
          if not vn.score and self.in_current_partition(vp)
      ]
      if unsolved_in_part:
        next_hinges = [
            h
            for h in self.unvisited_hinges
            if not self.hinge_dict[h.hash_value].flat
            and any(
                vn.is_tip(h)
                or (
                    self.hinge_dict[h.hash_value].allow_symmetric
                    and any(
                        vn.is_tip(cp_)
                        for cp_ in self.hinge_dict[h.hash_value].central_pleats
                    )
                )
                for vn in unsolved_in_part
            )
        ]
    if (
        self.solver_config.recovery
        and not next_hinges
        and self.partition_order is not None
        and self.partition_progress >= len(self.partition_order.partitions)
    ):
      next_hinges = [
          h
          for h in self.unvisited_hinges
          if not self.hinge_dict[h.hash_value].flat
      ]

    if self.solver_config.remove_central_crease:
      central_crease = central_fold_utils.central_crease_from_transform(
          self.transform
      )
      overlapping_hinges = [
          h for h in next_hinges if h.is_overlapping(central_crease)
      ]
      if overlapping_hinges:
        return overlapping_hinges

    if not self.allow_simple_hinges():
      next_hinges = [
          h for h in next_hinges if not self.hinge_dict[h.hash_value].simple
      ]

    return next_hinges

  def get_next_states(
      self,
      active_states: MutableMapping[int, 'State'],
  ) -> MutableMapping[int, 'State']:
    """Returns the next states of the current state.

    Each state is created by adding one of the unvisited hinges to the
    current state. Each hinge can have multiple options for interleaving and
    folding. We try all the options for each hinge.

    Args:
      active_states: The mapping of active states.

    Returns:
      active_states updated with the next states.
    """
    next_hinges = self.get_valid_hinges()
    next_states_list: list[State] = []
    for hinge in next_hinges:
      interleave_options, fold_options = get_hinge_options(
          hinge, self.hinge_dict, self.unassigned_pleats, self.vertices
      )
      for interleave, fold in itertools.product(
          interleave_options, fold_options
      ):
        state = state_from_hinge(
            self,
            hinge,
            self.hinge_dict,
            assignment_type=box_pleating_utils.HingeAssignmentType(
                fold=fold,
                symmetric=not interleave,
            ),
        )
        if state.state_score < self.state_score:
          continue
        if state.is_invalid(self.state_score):
          if not self.solver_config.allow_constraint_relaxation:
            continue
          relaxed_state = state.with_relaxed_constraints()
          if relaxed_state is state or relaxed_state.is_invalid(
              self.state_score
          ):
            continue
          state = relaxed_state
        next_states_list.append(state)

    if self.solver_config.max_branching_factor is not None:

      def key_fn(s: 'State'):
        return (
            s.partition_progress,
            s.state_score,
            (
                _hinge_symmetry_score(s)
                if self.solver_config.tie_break_by_hinge_symmetry
                else 0
            ),
            -len(s.unsplit_hinges),
            -len(s.hinges),
        )

      next_states_list.sort(key=key_fn, reverse=True)
      next_states_list = next_states_list[
          : self.solver_config.max_branching_factor
      ]

    for state in next_states_list:
      state_hash = hash(state)
      if state_hash in active_states:
        continue
      active_states[state_hash] = state
    self.expanded = True
    return active_states

  def with_relaxed_constraints(self) -> 'State':
    """Returns a new State with heuristic flat/symmetric constraints relaxed for unsolveable vertices."""
    if (
        not self.solver_config.allow_constraint_relaxation
        or not self.unvisited_hinges
        or self.state_score == len(self.vertices)
    ):
      return self
    new_hinge_dict = _relax_hinge_constraints_for_unsolveable_vertices(
        unvisited_hinges=self.unvisited_hinges,
        unassigned_pleats=self.unassigned_pleats,
        vertices=self.vertices,
        hinge_dict=self.hinge_dict,
        unflatten_flat_hinges=self.solver_config.hinges_from_stick_figure,
        copy_on_write=True,
    )
    if new_hinge_dict is self.hinge_dict:
      return self
    relaxed_hinges = frozenset(
        (
            h.hash_value,
            new_hinge_dict[h.hash_value].force_non_flat,
            new_hinge_dict[h.hash_value].force_allow_symmetric,
        )
        for h in self.unvisited_hinges
        if (
            new_hinge_dict[h.hash_value].force_non_flat
            or new_hinge_dict[h.hash_value].force_allow_symmetric
        )
    )
    relaxed_state = self.copy_state()
    relaxed_state.hinge_dict = new_hinge_dict
    relaxed_state.relaxed_hinges = relaxed_hinges
    relaxed_state._pleat_blocking_hinges_cache = {}  # pylint: disable=protected-access
    relaxed_state.invalidate_caches()
    return relaxed_state

  def _has_unsolveable_vertex(self) -> bool:
    """Checks if the state has an unsolveable vertex."""
    if self.state_score == len(self.vertices):
      return False
    odd_changer_hashes: set[int] = set()
    assigned_cp_hashes: set[int] = set()
    unassigned_pleat_hashes: set[int] = set()
    for p in self.unassigned_pleats:
      p_hash = p.hash_value
      odd_changer_hashes.add(p_hash)
      unassigned_pleat_hashes.add(p_hash)
    hinge_dict = self.hinge_dict
    for hinge in self.unvisited_hinges:
      h_hash = hinge.hash_value
      h_meta = hinge_dict[h_hash]
      if self.solver_config.recovery and h_meta.flat:
        continue
      odd_changer_hashes.add(h_hash)
      if not self.solver_config.recovery or h_meta.allow_symmetric:
        for cp in h_meta.central_pleats:
          cp_hash = cp.hash_value
          if cp_hash not in unassigned_pleat_hashes:
            assigned_cp_hashes.add(cp_hash)

    for vertex_point, v in self.vertices.items():
      if v.score:
        continue
      tip_hashes = v.tip_hashes
      odd_changers = 0
      for h in tip_hashes:
        if h in odd_changer_hashes:
          odd_changers += 1
      if not assigned_cp_hashes:
        intersecting = odd_changers
      else:
        assigned_cps = 0
        for h in tip_hashes:
          if h in assigned_cp_hashes:
            assigned_cps += 1
        intersecting = odd_changers + 2 * assigned_cps
      if intersecting == 0:
        if self.solver_config.log_progress:
          n_mountains, n_valleys = v.mv_counts()
          maekawa_diff = abs(abs(n_mountains - n_valleys) - 2)
          logging.debug(
              'vertex %s is unsolveable. maekawa_diff: %d n_creases: %d'
              ' central_flap: %s',
              vertex_point,
              maekawa_diff,
              len(v.creases),
              v.central_flap,
          )
        return True
      n_mountains, n_valleys = v.mv_counts()
      maekawa_diff = abs(abs(n_mountains - n_valleys) - 2)
      if (
          maekawa_diff > intersecting  # Vertex condition 2
          or (
              self.solver_config.recovery
              and (n_mountains + n_valleys) % 2 == 1
              and odd_changers == 0
          )
          or (
              v.central_flap and maekawa_diff > 1 and n_mountains != n_valleys
          )  # Vertex condition 3
      ):
        if self.solver_config.log_progress:
          logging.debug(
              'vertex %s is unsolveable. maekawa_diff: %d n_creases: %d'
              ' central_flap: %s',
              vertex_point,
              maekawa_diff,
              len(v.creases),
              v.central_flap,
          )
        return True
    return False

  def is_invalid(self, parent_score: int) -> bool:
    """Checks if the state is invalid.

    A state is invalid in one of three cases:
      1. The parent state is better than the current state.
      2. There is an unassigned pleat that intersects an unassigned hinge in an
      intermediate point that is not the midpoint.
      3. It has an unsolveable vertex.

    A vertex that is currently not flat foldable is unsolveable in three cases:
      1. There are no more unassigned creases that intersect it.
      2. The number of unassigned creases can't satisfy the Maekawa diff.
      3. There is a central flap that has been assigned with too many pleats and
      is irreversible. Central flaps start with 4 M and 4 V and can only be
      modified by reassigning one of the Valleys to Mountain. Reassigning more
      than one pleat makes them unsolveable.

    Args:
      parent_score: The score of the parent state.

    Returns:
      True if the state is invalid.
    """
    # Condition 1
    if parent_score > self.state_score:
      return True
    if not self.solver_config.allow_misaligned_symmetric_hinges:
      if _has_misaligned_pleat_intersection(
          self.unassigned_pleats,
          self.unsplit_hinges,
          recovery=self.solver_config.recovery,
      ):
        return True
      for unassigned_pleat in self.unassigned_pleats:
        ridges_key = (
            id(self.ridges),
            len(self.ridges),
            unassigned_pleat.exact_match,
        )
        misaligned_ridge = self._pleat_misaligned_ridges_cache.get(ridges_key)
        if misaligned_ridge is None:
          misaligned_ridge = _has_misaligned_pleat_intersection(
              (unassigned_pleat,),
              self.ridges,
              recovery=self.solver_config.recovery,
          )
          self._pleat_misaligned_ridges_cache[ridges_key] = misaligned_ridge
        if misaligned_ridge:
          return True

    return self._has_unsolveable_vertex()

  def assign_pleats(self, force_assign: bool = False) -> list[crease.Crease]:
    """Assigns pleats from the unassigned pleats.

    Pleats are usually not changed while solving the crease pattern. However,
    when a symmetric hinge is added, we reassign the pleat that intersects the
    hinge in the midpoint. This gives some flexibility in solving the pattern.

    The _is_assignable_pleat method decides whether a pleat is assignable now or
    we need to wait with assigning it (because other creases may intersect it).

    If a pleat cannot be assigned now, we compute a temporary assignment for it
    that allows us to compute the score of the vertex that will be updated with
    the pleat when it is finally assigned.

    Args:
      force_assign: If True, all unassigned pleats will be assigned.

    Returns:
      unassigned_pleats that will be assigned in the future and assigned_pleats
      that can be added to the crease pattern.
    """
    unassigned_pleats, assigned_pleats = [], []
    # Unassigned pleats that intersect the border are the easiest to solve,
    # so we assign them last.
    self.unassigned_pleats.sort(key=lambda p: p.intersects_border)
    assigned_pleats_to_remove = []
    for pleat in self.unassigned_pleats:
      # If a pleat intersects hinges that have not been assigned yet, we cannot
      # assign the pleat.
      can_be_assigned = _is_assignable_pleat(pleat, self) or force_assign
      if can_be_assigned:
        assigned_pleats_to_remove.append(pleat)
      ridges_key = (id(self.ridges), len(self.ridges), pleat.exact_match)
      candidate_ridges = self._pleat_candidate_ridges_cache.get(ridges_key)
      if candidate_ridges is None:
        candidate_ridges = tuple(
            _filter_encompassing_creases(pleat, self.ridges)
        )
        self._pleat_candidate_ridges_cache[ridges_key] = candidate_ridges
      candidate_creases = _filter_encompassing_creases(pleat, self.hinges)
      if candidate_ridges:
        candidate_creases.extend(candidate_ridges)
      segments = split_pleat_to_segments(
          pleat.exact_match,
          crease.Creases.create_from_creases(creases=candidate_creases),
          self.grid_points,
      )[:]
      fold_, segments = _fold_to_pleat(pleat, segments, self)

      # Assign the folds to the segments.
      for h in segments:
        new_segment = _make_pleat_segment(
            h.p1,
            h.p2,
            fold_,
            pleat.fold_percent,
            pleat.elevation,
        )
        if can_be_assigned:
          assigned_pleats.append(new_segment)
        else:
          unassigned_pleats.append(new_segment)
        fold_ = new_segment.mountain_to_valley()

    if assigned_pleats_to_remove or assigned_pleats:
      self.invalidate_caches()
    # Remove the assigned pleats from the unassigned pleats.
    for pleat in assigned_pleats_to_remove:
      self.unassigned_pleats.remove(pleat)
    # Add the assigned pleats to the pleats.
    self.pleats.extend(assigned_pleats)
    # Update the vertices with the assigned pleats.
    for pleat in assigned_pleats:
      for vertex_point in self._get_tip_vertex_points(pleat):
        self.vertices[vertex_point].add_crease(pleat)
    return unassigned_pleats

  def update_partition_progress(self):
    """Updates the partition progress of the state.

    partition_progress represents the fact that partitions[:partition_progress]
    have all been solved for flat-foldability.
    partition_progress has two main effects on the search:
      1. Only hinges in the current partition are allowed
        - See `State.get_valid_hinges`
      2. Active states with higher partition_progress are preferred.
        - See `get_state_to_expand`

    In this initial version, we calculate this from scratch each time a hinge
    is added by iterating through each of the allegedly solved partitions.
    If we find a partition that is not solved, we set partition_progress to the
    index of that partition. Otherwise if we check all partitions and they are
    valid, we increment partition_progress to allow the next partition to be
    searched.
    """
    if self.partition_order is None:
      return

    if self._partition_vertex_points is None:
      self._partition_vertex_points = tuple(
          tuple(p for p in partition.vertices if p in self.vertices)
          for partition in self.partition_order.partitions
      )

    if not self.solver_config.recovery:
      for partition_idx, partition_points in enumerate(
          self._partition_vertex_points[: self.partition_progress + 1]
      ):
        if not all(self.vertices[p].score for p in partition_points):
          self.partition_progress = partition_idx
          return
      self.partition_progress += 1
      return

    for partition_idx, partition_points in enumerate(
        self._partition_vertex_points
    ):
      if not all(self.vertices[p].score for p in partition_points):
        # Reset partition progress to the first unsolved partition
        self.partition_progress = partition_idx
        return

    # We have checked all partitions and they are all valid.
    self.partition_progress = len(self.partition_order.partitions)

  def assign_as_symmetric(self, hinges: Collection[crease.Crease]) -> 'State':
    """Assigns provided hinges as symmetric hinges.

    The fold of the symmetric hinges is determined by fold_from_vertex. If the
    hinge cannot be assigned as symmetric, it is assigned as interleaved.

    Args:
      hinges: A collection of hinges to assign as symmetric.

    Returns:
      A new updated state.

    Raises:
      StateValueError: If a hinge cannot be matched to a vertex in the state or
      if the state becomes unsolveable.
    """
    if not hinges:
      logging.warning('No creases in assignment CP.')
      return self

    state = self
    for hinge in hinges:
      # Find matching hinge in state.unvisited_hinges
      matched_hinge = next(
          (h for h in state.unvisited_hinges if h.isclose(hinge)), None
      )
      if matched_hinge is None:
        raise StateValueError(f'No matching hinge found for {hinge}', state)
      if hinge.fold != crease.Fold.HINGE:
        raise StateValueError(f'Unsupported fold type: {hinge.fold}', state)

      interleave_options, fold_options = get_hinge_options(
          hinge, state.hinge_dict, state.unassigned_pleats, state.vertices
      )

      symmetric = (
          False in interleave_options and not matched_hinge.intersects_border
      )
      if not symmetric:
        logging.info(
            'Hinge cannot be symmetric, assigning as interleaved: %s',
            hinge,
        )

      # Determine fold to be the opposite of the ridge at the vertex
      try:
        v = _find_vertex_that_is_tip_of_crease(
            matched_hinge, (vn.vertex_ for vn in state.vertices.values())
        )
        fold = next(c.fold for c in v.creases if c.is_diagonal).flip()
      except (ValueError, StopIteration):
        fold = fold_options[0]
      if fold not in fold_options:
        logging.warning(
            'Hinge fold from ridges (%s) not in recommended fold options: %s',
            fold,
            fold_options,
        )

      assignment_type = box_pleating_utils.HingeAssignmentType(
          fold=fold, symmetric=symmetric
      )

      state = state_from_hinge(
          state, matched_hinge, state.hinge_dict, assignment_type
      )
      if state.is_invalid(parent_score=-1):
        raise StateValueError(
            f'Assigning hinge creates an unsolveable vertex: {hinge}', state
        )
    return state

  def update_vertex_scores(
      self,
      unassigned_pleats: list[crease.Crease],
      *,
      check_maekawa: bool | None = None,
  ):
    """Updates vertex scores.

    Only vertices that have been updated or vertices that have tips with
    unassigned pleats are updated.

    Args:
      unassigned_pleats: The unassigned pleats.
      check_maekawa: Whether to enforce Maekawa's theorem when scoring vertices.
        Defaults to self.solver_config.recovery.
    """
    if check_maekawa is None:
      check_maekawa = self.solver_config.recovery
    if not unassigned_pleats:
      for v in self.vertices.values():
        v.score = v.vertex_.flat_foldability(check_maekawa=check_maekawa)
      return

    cloned_vertices: dict[point.Point, Any] = {}
    for pleat in unassigned_pleats:
      key = (pleat.p1, pleat.p2)
      statuses = self._pleat_point_status_cache.get(key)
      if statuses is None:
        statuses_list: list[tuple[point.Point, bool]] = []
        for vertex_point in self.vertices:
          status = pleat.point_status(vertex_point)
          if status == crease.PointStatus.TIP:
            statuses_list.append((vertex_point, True))
          elif status == crease.PointStatus.INTERNAL:
            statuses_list.append((vertex_point, False))
        statuses = tuple(statuses_list)
        self._pleat_point_status_cache[key] = statuses
      for vertex_point, is_tip in statuses:
        vertex_ = cloned_vertices.get(vertex_point)
        if vertex_ is None:
          vertex_ = self.vertices[vertex_point].vertex_.clone()
          cloned_vertices[vertex_point] = vertex_
        if is_tip:
          vertex_.add_crease(pleat)
        else:
          vertex_.add_crease(
              _make_pleat_segment(vertex_point, pleat.p2, pleat.fold, 1.0, 0)
          )
          vertex_.add_crease(
              _make_pleat_segment(pleat.p1, vertex_point, pleat.fold, 1.0, 0)
          )
    for vertex_point, v in self.vertices.items():
      vertex_ = cloned_vertices.get(vertex_point, v.vertex_)
      v.score = vertex_.flat_foldability(check_maekawa=check_maekawa)

  def symmetry_score(self) -> float:
    return hinge_symmetry_score(self)


class NoSolutionError(RuntimeError):
  """Raised when search is completely exhausted and no solution is found."""


def _vertices_from_crease_intersections(
    cp: crease_pattern.CreasePattern,
    creases: list[crease.Crease],
    vertices: list[vertex.Vertex],
) -> list[vertex.Vertex]:
  """Adds internal vertices of cp that are tip intersections of creases.

  Args:
    cp: The crease pattern.
    creases: The creases to add vertices from.
    vertices: The list of vertices to add to.

  Returns:
    The updated list of vertices.
  """
  new_vertices = []
  for c in creases:
    for v in cp.vertices:
      if v.internal and c.is_tip(v.point):
        if not any(v_.point.isclose(v.point) for v_ in vertices + new_vertices):
          new_vertices.append(v)
  vertices.extend(new_vertices)
  return vertices


def _find_vertex_that_is_tip_of_crease(
    c: crease.Crease, vertices: Iterable[vertex.Vertex]
) -> vertex.Vertex:
  """Returns the vertex that is a tip of a crease."""
  for v in vertices:
    if c.is_tip(v.point):
      return v
  raise ValueError(f'No vertex found for {c}')


def fold_from_vertex(
    c: crease.Crease, vertices: list[vertex.Vertex]
) -> crease.Fold:
  """Returns the fold of a crease based on the vertex it intersects with."""
  v = _find_vertex_that_is_tip_of_crease(c, vertices)
  mountains = sum(c.fold == crease.Fold.MOUNTAIN for c in v.creases)
  valleys = sum(c.fold == crease.Fold.VALLEY for c in v.creases)
  if mountains - valleys > 2:  # 4 1
    return crease.Fold.VALLEY
  elif mountains - valleys == 2:
    return crease.Fold.MOUNTAIN
  elif mountains - valleys == 1:  # 4 3
    return crease.Fold.MOUNTAIN
  elif mountains - valleys == 0:  # 3 2
    return crease.Fold.VALLEY
  elif mountains - valleys == -1:  # 1 4
    return crease.Fold.VALLEY
  elif mountains - valleys == -2:
    return crease.Fold.VALLEY
  else:
    return crease.Fold.MOUNTAIN


def add_pleats_to_unassigned(
    pleats: list[crease.Crease],
    unassigned_pleats: list[crease.Crease],
    ridges: list[crease.Crease],
    hinges: list[crease.Crease],
    grid_size: int | None = None,
) -> tuple[list[crease.Crease], list[crease.Crease]]:
  """Moves pleats with a specific intersection pattern to unassigned pleats.

  A pleat is moved from pleats to unassigned_pleats if it has exactly one point
  that intersects a border, another point that intersects exactly 4 ridges and
  0 hinges, it does not intersect any other pleat at an intermediate point
  (intersections at tips are allowed), and no existing unassigned pleat already
  intersects the non-border point (at most one unassigned pleat per point).

  Args:
    pleats: The currently assigned pleats.
    unassigned_pleats: The currently unassigned pleats.
    ridges: The ridges of the crease pattern.
    hinges: The hinges of the crease pattern.
    grid_size: Optional grid size for integer lattice bucketing.

  Returns:
    A tuple of (updated pleats, updated unassigned_pleats).
  """
  new_pleats = []
  new_unassigned_pleats = list(unassigned_pleats)
  ridges_tip_index = box_pleating_utils.TipIndex(ridges, grid_size=grid_size)
  hinges_tip_index = box_pleating_utils.TipIndex(hinges, grid_size=grid_size)
  unassigned_tip_index = box_pleating_utils.TipIndex(
      new_unassigned_pleats, grid_size=grid_size
  )
  for pleat in pleats:
    p1_is_border = pleat.p1.is_border
    p2_is_border = pleat.p2.is_border
    # Check that exactly one point intersects the border.
    if p1_is_border == p2_is_border:
      new_pleats.append(pleat)
      continue
    # The non-border point is the one that must intersect 4 ridges and 0 hinges.
    non_border_point = pleat.p2 if p1_is_border else pleat.p1
    if ridges_tip_index.count_at_tip(non_border_point) != 4:
      new_pleats.append(pleat)
      continue
    if hinges_tip_index.has_tip(non_border_point):
      new_pleats.append(pleat)
      continue
    # Check that no existing unassigned pleat already covers non_border_point.
    if unassigned_tip_index.has_tip(non_border_point):
      new_pleats.append(pleat)
      continue
    # Check that no other pleat intersects this pleat at an intermediate point.
    other_pleats = [p for p in pleats + unassigned_pleats if p is not pleat]
    has_intermediate_pleat_intersection = any(
        pleat.point_status(intersection) == crease.PointStatus.INTERNAL
        for other in other_pleats + hinges + ridges
        if (intersection := pleat.vertex_with_crease(other)) is not None
    )
    if not has_intermediate_pleat_intersection:
      new_unassigned_pleats.append(pleat)
      unassigned_tip_index.add(pleat)
    else:
      new_pleats.append(pleat)
  return new_pleats, new_unassigned_pleats


def assign_central_ridges(
    ridges: list[crease.Crease],
    pleats: list[crease.Crease],
    vertices: list[vertex.Vertex],
    grid_size: int,
) -> tuple[list[crease.Crease], list[crease.Crease], list[crease.Crease]]:
  """Assigns folds to central ridges.

  Args:
    ridges: The ridges to assign folds to.
    pleats: The pleats to assign folds to.
    vertices: The vertices of the crease pattern.
    grid_size: The size of the grid.

  Returns:
    Assigned ridges, pleats, and a list of unassigned pleats.
  """
  assigned_ridges = []
  unassigned_pleats = []
  grid_points = box_pleating_utils.create_grid_points(grid_size)
  for r in ridges:
    intersects_two_borders = r.p1.is_border and r.p2.is_border
    intersects_exactly_one_border = (
        r.intersects_border and not intersects_two_borders
    )
    if intersects_two_borders:
      # Sometimes there is a weird case where all the vertices of a wide flap
      # are on the border (i.e when all the usual diagonal ridges of the flap
      # are outside the paper). This leaves a central ridge from border to
      # border, with no non-border vertices.
      # See `cp_test_fixtures.CENTRALRIDGENOVERTICES`.
      # In this case, we arbitrarily assign it to MOUNTAIN.
      fold = crease.Fold.MOUNTAIN
    else:
      fold = fold_from_vertex(r, vertices)
    segments = r.split_to_segments(pleats + unassigned_pleats, grid_points)
    interleave = (
        intersects_exactly_one_border
        or len(segments) % 2 == 1  # any odd ridge must be interleaved
        or (intersects_two_borders and r.is_vertical)
    )
    if not interleave:
      central_pleats = _find_central_pleats(segments, pleats)
      for pleat in central_pleats:
        pleats.remove(pleat)
        unassigned_pleats.append(pleat)
    assigned_ridges.extend(
        box_pleating_utils.HingeAssignmentType(
            fold=fold,
            symmetric=not interleave,
        ).assign_segments(segments)
    )
  return assigned_ridges, pleats, unassigned_pleats


def state_from_hinge(
    curr_state: State,
    hinge: crease.Crease,
    hinge_dict: dict[int, HingeMetaData],
    assignment_type: box_pleating_utils.HingeAssignmentType,
) -> State:
  """Creates a new state by adding a hinge to the current state.

  Args:
    curr_state: The current state.
    hinge: The hinge to add to the state.
    hinge_dict: A dictionary mapping the hinge hash to the hinge metadata.
    assignment_type: The chosen assignment type of the hinge.

  Returns:
    The next state or None if the state is not valid.
  """
  next_state = curr_state.copy_state()
  next_state.invalidate_caches()
  hinge_meta = hinge_dict[hinge.hash_value]
  # Update the hinges
  assigned_hinge_segments = hinge_meta.get_assigned_segments(assignment_type)
  for h in assigned_hinge_segments:
    for vertex_point in next_state._get_tip_vertex_points(h):  # pylint: disable=protected-access
      next_state.vertices[vertex_point].add_crease(h)
  next_state.unvisited_hinges.remove(hinge)
  next_state.unsplit_hinges.append(hinge)
  next_state.hinges.extend(assigned_hinge_segments)

  # Update the pleats
  if assignment_type.symmetric:
    # only reassign pleats if the hinge is symmetric
    for pleat in hinge_meta.central_pleats:
      if pleat in next_state.unassigned_pleats:
        continue
      matching_pleats = [
          c_
          for c_ in next_state.pleats
          if c_ == pleat or c_.is_within(pleat, match_fold=False)
      ]
      if not matching_pleats:
        raise ValueError(f'Central pleat {pleat} not found in assigned pleats.')
      for c_ in matching_pleats:
        next_state.pleats.remove(c_)
        for vertex_point in next_state._get_tip_vertex_points(c_):  # pylint: disable=protected-access
          v = next_state.vertices[vertex_point]
          if c_ in v.creases:
            v.remove_crease(c_)
          else:
            for vc in list(v.creases):
              if (
                  vc.fold != crease.Fold.HINGE
                  and vc not in next_state.hinges
                  and vc.is_within(c_, match_fold=False)
              ):
                v.remove_crease(vc)
      next_state.unassigned_pleats.append(pleat)
  unassigned_pleats = next_state.assign_pleats()

  next_state.update_vertex_scores(
      unassigned_pleats, check_maekawa=next_state.solver_config.recovery
  )
  next_state.state_score = sum(v.score for v in next_state.vertices.values())

  # Check if partition is complete
  next_state.update_partition_progress()

  return next_state


def get_hinge_options(
    hinge: crease.Crease,
    hinge_dict: dict[int, HingeMetaData],
    unassigned_pleats: list[crease.Crease],
    vertices: dict[point.Point, VertexNode],
) -> tuple[list[bool], list[crease.Fold]]:
  """A hinge in an origami crease pattern has four options.

  Hinges are split to segments. Each two adjacent segments have different fold
  orientations. The only exception to this rule is a symmetrical hinge, where
  the two segments in the middle have the same fold orientation.

  In addition to these two options, we also have the option to decide if the
  hinge should start with a Mountain or a Valley.

  In some cases, not all four options are available:
  1. If the hinge intersects the border, it has to be interleaved, and its fold
  orientation can be computed from other creases.
  2. If the number of segments is odd, the hinge cannot be symmetrical.
  3. If a central pleat (that intersects the hinge) is already used, the hinge
  has to be symmetrical.

  Args:
    hinge: The hinge to get the options for.
    hinge_dict: A dictionary mapping the hinge hash to the hinge metadata.
    unassigned_pleats: The unassigned pleats.
    vertices: The vertices of the crease pattern.

  Returns:
    A tuple of two lists:
      - A list of booleans indicating whether to interleave the folds.
      - A list of folds to assign to the hinge.
  """
  hinge_metadata = hinge_dict[hinge.hash_value]
  if hinge_metadata.simple:
    interleave_options = [True]
    intersects_unassigned_pleat = any(
        pleat.intersect_at_tip(hinge) for pleat in unassigned_pleats
    )
    border_to_border = hinge.p1.is_border and hinge.p2.is_border
    if intersects_unassigned_pleat or border_to_border:
      # If the hinge is border-to-border there is no vertex to get the fold
      # from.
      fold_options = [crease.Fold.MOUNTAIN, crease.Fold.VALLEY]
    else:
      fold_options = [
          fold_from_vertex(hinge, [v.vertex_ for v in vertices.values()])
      ]

  else:
    # If the hinge does not have an even number of segments or if there is no
    # central pleat, the hinge can only be interleaved.
    interleave_options = (
        [True, False] if hinge_metadata.allow_symmetric else [True]
    )
    for central_pleat in hinge_metadata.central_pleats:
      # Check if a central pleat (that intersects the hinge) is already used.
      # If so, the hinge has to be symmetrical.
      if central_pleat in unassigned_pleats:
        interleave_options.remove(True)
        break
    fold_options = [crease.Fold.MOUNTAIN, crease.Fold.VALLEY]
  return interleave_options, fold_options


def vertex_score(
    v: vertex.Vertex,
    reassigned_pleats: list[crease.Crease],
    *,
    check_maekawa: bool = False,
) -> bool:
  """Predicts the flat_foldability of the vertex."""
  v_copy = v.clone()
  for pleat in reassigned_pleats:
    if pleat.is_tip(v_copy.point):
      v_copy.add_crease(pleat)
  return v_copy.flat_foldability(check_maekawa=check_maekawa)


@functools.lru_cache(maxsize=10000)
def split_pleat_to_segments(
    pleat: crease.CreaseWithExactMatch,
    creases: crease.Creases,
    grid_points: tuple[point.Point, ...],
) -> list[crease.Crease]:
  """Splits a pleat to segments."""
  creases_ = [c.crease for c in creases.creases]
  symmetric_creases = _filter_symmetric_partner_creases(creases_)
  return pleat.crease.split_to_segments(symmetric_creases, grid_points)


@functools.lru_cache(maxsize=65536)
def _make_pleat_segment(
    p1: point.Point,
    p2: point.Point,
    fold: crease.Fold,
    fold_percent: float,
    elevation: int,
) -> crease.Crease:
  """Returns a cached Crease segment for pleat assignment."""
  return crease.Crease(
      p1,
      p2,
      fold=fold,
      fold_percent=fold_percent,
      elevation=elevation,
  )


def _hinge_blocks_pleat(hinge: crease.Crease, pleat: crease.Crease) -> bool:
  """Returns whether hinge intersects pleat at an intermediate point of hinge."""
  v = hinge.vertex_with_crease(pleat)
  return bool(v and pleat_utils.grid_point_intermediate_pleat(v, hinge))


def _is_assignable_pleat(
    pleat: crease.Crease,
    state: State,
) -> bool:
  """Returns whether a pleat can be assigned.

  A pleat can be assigned if it does not intersect an unassigned hinge in an
  intermediate point. Otherwise, the pleat may change orientation if the
  unassigned hinge is added.

  Args:
    pleat: The pleat to check.
    state: The current state of the crease pattern.

  Returns:
    True if the pleat can be assigned.
  """
  if not state.unvisited_hinges:
    return True
  cache = state._pleat_blocking_hinges_cache  # pylint: disable=protected-access
  blocking_hinges = cache.get(pleat.exact_match)
  if blocking_hinges is None:
    blocking_hinges = tuple(
        h_meta.hinge
        for h_meta in state.hinge_dict.values()
        if (not state.solver_config.recovery or not h_meta.flat)
        and _hinge_blocks_pleat(h_meta.hinge, pleat)
    )
    cache[pleat.exact_match] = blocking_hinges
  if not blocking_hinges:
    return True
  unvisited = state.unvisited_hinges
  return not any(h in unvisited for h in blocking_hinges)


def _fold_to_pleat(
    pleat: crease.Crease, segments: list[crease.Crease], state: State
) -> tuple[crease.Fold, list[crease.Crease]]:
  """Assigns the fold to the pleat.

  Following the call to this function, the segments of the pleat will be
  assigned mountain/valley, interleaving at each segment, and starting from
  the returned fold. The segments may be reversed to reflect the right fold.

  A pleat must intersect at least one vertex.

  The fold is assigned as follows:
  1. If the pleat intersects the border, the fold is assigned from the vertex.
  2. If a tip of the pleat does not intersect a vertex, the fold at this tip
  must stay the same.
  3. If the pleat intersects two vertices, the fold is assigned from one of the
  vertices.

  Args:
    pleat: The pleat to assign the fold to.
    segments: The segments of the pleat.
    state: The current state of the crease pattern.

  Returns:
    A tuple of the fold and the segments.
  """
  # Condition 1
  if pleat.intersects_border:
    if segments[0].intersects_border:
      if segments[-1].intersects_border:
        # if both segments intersect the border, any fold can be assigned
        return crease.Fold.MOUNTAIN, segments
      segments.reverse()
    tip_pts = state._get_tip_vertex_points(segments[0])  # pylint: disable=protected-access
    if tip_pts:
      fold_ = fold_from_vertex(
          segments[0], [state.vertices[tip_pts[0]].vertex_]
      )
    else:
      # If no vertex is found, keep the original fold.
      fold_ = segments[0].fold

  # Condition 2
  elif not state._get_tip_vertex_points(segments[0]):  # pylint: disable=protected-access
    fold_ = segments[0].fold
  elif not state._get_tip_vertex_points(segments[-1]):  # pylint: disable=protected-access
    fold_ = segments[-1].fold
    segments.reverse()
  # Condition 3
  else:
    tip_pts = state._get_tip_vertex_points(segments[0])  # pylint: disable=protected-access
    fold_ = fold_from_vertex(segments[0], [state.vertices[tip_pts[0]].vertex_])
  return fold_, segments


def _find_central_pleats(
    segments: list[crease.Crease],
    pleats: list[crease.Crease],
) -> list[crease.Crease]:
  """Find pleats that intersect crease_ at the midpoint.

  If the crease intersects the border, or has an odd number of segments,
  there are no central pleats.

  Args:
    segments: The segments of the crease.
    pleats: The pleats to search for central pleats.

  Returns:
    A list of pleats that intersect crease_ at the midpoint.
  """
  if len(segments) % 2 != 0:
    return []
  central_segments = [
      segments[len(segments) // 2 - 1],
      segments[len(segments) // 2],
  ]
  central_pleats = []
  for pleat in pleats:
    v1 = pleat.vertex_with_crease(central_segments[0])
    v2 = pleat.vertex_with_crease(central_segments[1])
    if v1 is not None and v2 is not None:
      central_pleats.append(pleat)
  return central_pleats


@functools.lru_cache(maxsize=65536)
def _transform_crease(
    c_exact: crease.CreaseWithExactMatch, transform: symmetry.Transform
) -> crease.Crease:
  """Returns cached transformed crease for symmetry scoring."""
  return transform(c_exact.crease)


def hinge_symmetry_score(state: State | SolverStateSnapshot) -> float:
  """Returns the symmetry score of the (unsplit) hinges that were added."""
  if not state.hinges:
    return 0
  if state._symmetry_score_cache is None:  # pylint: disable=protected-access
    state._symmetry_score_cache = symmetry.creases_transform_symmetry(  # pylint: disable=protected-access
        state.hinges, state.transform, dedup_input=False
    )
  return state._symmetry_score_cache  # pylint: disable=protected-access


_hinge_symmetry_score = hinge_symmetry_score


class _StateBucket:
  """Per-(partition_progress, state_score) priority bucket of unexpanded states."""

  __slots__ = ('tie_break_by_hinge_symmetry', 'unheaped', 'heap')

  def __init__(self, tie_break_by_hinge_symmetry: bool):
    self.tie_break_by_hinge_symmetry = tie_break_by_hinge_symmetry
    self.unheaped: list[tuple[int, State]] = []
    self.heap: list[tuple[Any, ...]] = []

  def _make_entry(self, seq: int, state: State) -> tuple[Any, ...]:
    if self.tie_break_by_hinge_symmetry:
      return (
          -_hinge_symmetry_score(state),
          len(state.unsplit_hinges),
          len(state.hinges),
          seq,
          state,
      )
    return (
        len(state.unsplit_hinges),
        len(state.hinges),
        seq,
        state,
    )

  def add(self, seq: int, state: State) -> None:
    """Adds an unexpanded state with insertion sequence seq to the bucket."""
    self.unheaped.append((seq, state))

  def peek(self) -> State | None:
    """Returns the highest-priority unexpanded state in the bucket, or None."""
    while self.unheaped and self.unheaped[-1][1].expanded:
      self.unheaped.pop()
    while self.heap and self.heap[0][-1].expanded:
      heapq.heappop(self.heap)
    if not self.heap:
      if not self.unheaped:
        return None
      if len(self.unheaped) > 1:
        self.unheaped = [
            (u_seq, u_state)
            for u_seq, u_state in self.unheaped
            if not u_state.expanded
        ]
      if not self.unheaped:
        return None
      if len(self.unheaped) == 1:
        return self.unheaped[0][1]
      self.heap = [
          self._make_entry(u_seq, u_state) for u_seq, u_state in self.unheaped
      ]
      heapq.heapify(self.heap)
      self.unheaped.clear()
    elif self.unheaped:
      for u_seq, u_state in self.unheaped:
        if not u_state.expanded:
          heapq.heappush(self.heap, self._make_entry(u_seq, u_state))
      self.unheaped.clear()
    entry = self.heap[0]
    candidate = entry[-1]
    if entry[:-2] != self._make_entry(entry[-2], candidate)[:-2]:
      raise ValueError('State priority mutated after indexing in _StateBucket')
    return candidate


class ActiveStates(MutableMapping[int, State]):
  """Mapping of active states with bucketed O(1) priority lookup."""

  def __init__(self, tie_break_by_hinge_symmetry: bool = True):
    self._states: dict[int, State] = {}
    self._tie_break = tie_break_by_hinge_symmetry
    self._buckets: dict[tuple[int, int], _StateBucket] = {}
    self._bucket_heap: list[tuple[int, int]] = []
    self._seq: int = 0

  def _index_state(self, state: State) -> None:
    self._seq += 1
    if state.expanded:
      return
    key = (state.partition_progress, state.state_score)
    bucket = self._buckets.get(key)
    if bucket is None:
      bucket = _StateBucket(self._tie_break)
      self._buckets[key] = bucket
      heapq.heappush(self._bucket_heap, (-key[0], -key[1]))
    bucket.add(self._seq, state)

  def __getitem__(self, key: int) -> State:
    return self._states[key]

  def __setitem__(self, key: int, state: State) -> None:
    old = self._states.get(key)
    self._states[key] = state
    if old is not None and old is not state and not old.expanded:
      self.rebuild_buckets(self._tie_break)
      return
    self._index_state(state)

  def __delitem__(self, key: int) -> None:
    old = self._states.pop(key)
    if not old.expanded:
      self.rebuild_buckets(self._tie_break)

  def __iter__(self) -> Iterator[int]:
    return iter(self._states)

  def __len__(self) -> int:
    return len(self._states)

  def __contains__(self, key: object) -> bool:
    return key in self._states

  def clear(self) -> None:
    self._states.clear()
    self._buckets.clear()
    self._bucket_heap.clear()
    self._seq = 0

  def rebuild_buckets(self, tie_break_by_hinge_symmetry: bool) -> None:
    """Rebuilds priority buckets if tie_break_by_hinge_symmetry changed."""
    self._tie_break = tie_break_by_hinge_symmetry
    self._buckets.clear()
    self._bucket_heap.clear()
    self._seq = 0
    for state in self._states.values():
      self._index_state(state)

  def best_state(self, tie_break_by_hinge_symmetry: bool) -> State:
    """Returns the highest-priority state to expand."""
    if self._tie_break != tie_break_by_hinge_symmetry:
      self.rebuild_buckets(tie_break_by_hinge_symmetry)
    while self._bucket_heap:
      neg_part, neg_score = self._bucket_heap[0]
      key = (-neg_part, -neg_score)
      bucket = self._buckets.get(key)
      if bucket is not None:
        candidate = bucket.peek()
        if candidate is not None:
          return candidate
        del self._buckets[key]
      heapq.heappop(self._bucket_heap)
    return _linear_get_state_to_expand(
        self._states, tie_break_by_hinge_symmetry=tie_break_by_hinge_symmetry
    )


def _linear_get_state_to_expand(
    active_states: Mapping[int, State],
    tie_break_by_hinge_symmetry: bool = False,
) -> State:
  """Reference linear scan to find the state to expand."""
  unexpanded_states = [v for v in active_states.values() if not v.expanded]
  if not unexpanded_states:
    # Use all states if there are no unexpanded states.
    unexpanded_states = list(active_states.values())

  max_partition_progress = max(s.partition_progress for s in unexpanded_states)
  unexpanded_states = [
      v
      for v in unexpanded_states
      if v.partition_progress == max_partition_progress
  ]

  max_score = max(unexpanded_states, key=lambda s: s.state_score).state_score
  max_states = [v for v in unexpanded_states if v.state_score == max_score]
  if tie_break_by_hinge_symmetry and len(max_states) > 1:
    max_symmetry_score = max(_hinge_symmetry_score(v) for v in max_states)
    max_states = [
        v for v in max_states if _hinge_symmetry_score(v) == max_symmetry_score
    ]
  min_hinges_state = min(
      max_states, key=lambda s: (len(s.unsplit_hinges), len(s.hinges))
  )
  return min_hinges_state


def get_state_to_expand(
    active_states: Mapping[int, State],
    tie_break_by_hinge_symmetry: bool = False,
) -> State:
  """Greedily find the state to expand.

  States are sorted by the following priority:
    1. Unexpanded before expanded
    2. Highest partition progress
    3. Highest state score (vertex flat-foldability)
    4. Fewer unsplit hinges and then fewer split hinges
    5. Hinge symmetry score (if tie_break_by_hinge_symmetry is True)

  Symmetry score is calculated lazily because it's expensive.

  Args:
    active_states: The active states.
    tie_break_by_hinge_symmetry: Whether to tie break by hinge symmetry score.

  Returns:
    The state to expand in the next step.
  """
  if isinstance(active_states, ActiveStates):
    return active_states.best_state(tie_break_by_hinge_symmetry)
  return _linear_get_state_to_expand(
      active_states, tie_break_by_hinge_symmetry=tie_break_by_hinge_symmetry
  )


def _valid_pleat(
    pleat: crease.Crease,
    vertices: dict[point.Point, VertexNode],
) -> bool:
  """Checks if the pleat is valid.

  A pleat is valid if it intersects a vertex, or if it intersects a border.

  Args:
    pleat: The pleat to check.
    vertices: A dictionary mapping point hashes to Vertex objects.

  Returns:
    True if the pleat is valid, False otherwise.
  """
  return (
      any(pleat.is_tip(vertex_point) for vertex_point in vertices.keys())
      or pleat.intersects_border
  )


def create_hinge_dict(
    unvisited_hinges: list[crease.Crease],
    pleats: list[crease.Crease],
    unassigned_pleats: list[crease.Crease],
    vertices: dict[point.Point, VertexNode],
    grid_points: tuple[point.Point, ...],
    solver_config: SolverConfig,
) -> dict[int, HingeMetaData]:
  """Creates a mapping from hinge hash.

  The mapping is to split hinge segments and pleats that intersect the hinge at
  the middle. This is done once to avoid re-computing it for each state. We also
  check if the central pleats intersect a vertex, and remove the ones that do
  not.

  Args:
    unvisited_hinges: A list of hinges that have not yet been added to the
      crease pattern.
    pleats: A list of pleats currently in the crease pattern.
    unassigned_pleats: A list of pleats that are unassigned.
    vertices: A dictionary mapping point hashes to Vertex objects.
    grid_points: A tuple of grid points.
    solver_config: The solver configuration.

  Returns:
    A dictionary mapping hinge hash to HingeMetaData.
  """
  hinge_dict = {}
  banned_symmetric_hinges = []
  if solver_config.preassignments is not None:
    # Add banned symmetric hinges from preassignments
    banned_symmetric_hinges = [
        c
        for c in solver_config.preassignments.creases
        if c.fold is crease.Fold.TRIANGLE
    ]
  for h in unvisited_hinges:
    segments = h.split_to_segments(pleats + unassigned_pleats, grid_points)
    central_pleats = _find_central_pleats(segments, pleats + unassigned_pleats)
    valid_central_pleats = []
    for pleat in central_pleats:
      if _valid_pleat(pleat, vertices):
        valid_central_pleats.append(pleat)

    hinge_dict[h.hash_value] = HingeMetaData(
        hinge=h,
        segments=segments,
        central_pleats=valid_central_pleats,
        solver_config=solver_config,
        ban_symmetric=any(c.isclose(h) for c in banned_symmetric_hinges),
    )

  if not solver_config.allow_constraint_relaxation:
    return hinge_dict
  return _relax_hinge_constraints_for_unsolveable_vertices(
      unvisited_hinges=unvisited_hinges,
      unassigned_pleats=unassigned_pleats,
      vertices=vertices,
      hinge_dict=hinge_dict,
      unflatten_flat_hinges=solver_config.hinges_from_stick_figure,
      copy_on_write=False,
  )


def _non_flat_hinges_and_central_pleats(
    unvisited_hinges: Iterable[crease.Crease],
    hinge_dict: Mapping[int, HingeMetaData],
    unassigned_pleats: list[crease.Crease],
) -> tuple[list[crease.Crease], list[crease.Crease]]:
  """Returns active non-flat hinges and deduplicated candidate central pleats."""
  non_flat_hinges = [
      h for h in unvisited_hinges if not hinge_dict[h.hash_value].flat
  ]
  all_central_pleats = []
  for h in non_flat_hinges:
    h_meta = hinge_dict[h.hash_value]
    if h_meta.allow_symmetric:
      all_central_pleats.extend(h_meta.central_pleats)
  deduped_central_pleats = box_pleating_utils.deduplicate(
      all_central_pleats + unassigned_pleats
  )
  return non_flat_hinges, deduped_central_pleats


def _is_vertex_unsolveable(
    v: VertexNode,
    non_flat_hinges: Iterable[crease.Crease],
    deduped_central_pleats: Iterable[crease.Crease],
    unassigned_pleats: Collection[crease.Crease],
    *,
    check_odd_degree: bool = True,
) -> bool:
  """Returns whether an unsolved vertex cannot be made flat-foldable."""
  intersecting = 0
  odd_changers = 0
  for h in non_flat_hinges:
    if v.is_tip(h):
      intersecting += 1
      odd_changers += 1
  for pleat in deduped_central_pleats:
    if v.is_tip(pleat):
      if pleat in unassigned_pleats:
        intersecting += 1
        odd_changers += 1
      else:
        # If a pleat will be reassigned, it changes the maekawa diff by 2.
        intersecting += 2
  n_mountains, n_valleys = v.mv_counts()
  maekawa_diff = abs(abs(n_mountains - n_valleys) - 2)
  return (
      intersecting == 0  # Vertex condition 1
      or maekawa_diff > intersecting  # Vertex condition 2
      or (
          check_odd_degree
          and (n_mountains + n_valleys) % 2 == 1
          and odd_changers == 0
      )
      or (
          v.central_flap and maekawa_diff > 1 and n_mountains != n_valleys
      )  # Vertex condition 3
  )


def _relax_hinge_constraints_for_unsolveable_vertices(
    unvisited_hinges: list[crease.Crease],
    unassigned_pleats: list[crease.Crease],
    vertices: Mapping[point.Point, VertexNode],
    hinge_dict: dict[int, HingeMetaData],
    unflatten_flat_hinges: bool,
    copy_on_write: bool = False,
) -> dict[int, HingeMetaData]:
  """Relaxes heuristic flat/symmetric restrictions if needed to keep vertices solveable."""
  copied = not copy_on_write
  if unflatten_flat_hinges and any(
      hinge_dict[h.hash_value].flat for h in unvisited_hinges
  ):
    non_flat_hinges, deduped_central_pleats = (
        _non_flat_hinges_and_central_pleats(
            unvisited_hinges, hinge_dict, unassigned_pleats
        )
    )
    to_unflatten: set[int] = set()
    for v in vertices.values():
      if v.score:
        continue
      if _is_vertex_unsolveable(
          v, non_flat_hinges, deduped_central_pleats, unassigned_pleats
      ):
        for h in unvisited_hinges:
          h_meta = hinge_dict[h.hash_value]
          if not h_meta.flat:
            continue
          if v.is_tip(h) or any(
              v.is_tip(pleat) for pleat in h_meta.central_pleats
          ):
            to_unflatten.add(h.hash_value)
    if to_unflatten:
      if not copied:
        hinge_dict = dict(hinge_dict)
        copied = True
      for h_hash in to_unflatten:
        hinge_dict[h_hash] = dataclasses.replace(
            hinge_dict[h_hash], force_non_flat=True
        )

  non_flat_hinges, deduped_central_pleats = _non_flat_hinges_and_central_pleats(
      unvisited_hinges, hinge_dict, unassigned_pleats
  )
  to_allow_symmetric: set[int] = set()
  for v in vertices.values():
    if v.score:
      continue
    if _is_vertex_unsolveable(
        v, non_flat_hinges, deduped_central_pleats, unassigned_pleats
    ):
      for h in non_flat_hinges:
        h_meta = hinge_dict[h.hash_value]
        if (
            h_meta.allow_symmetric
            or h_meta.ban_symmetric
            or len(h_meta.segments) % 2 != 0
            or not h_meta.central_pleats
        ):
          continue
        if any(v.is_tip(pleat) for pleat in h_meta.central_pleats):
          to_allow_symmetric.add(h.hash_value)
  if to_allow_symmetric:
    if not copied:
      hinge_dict = dict(hinge_dict)
    for h_hash in to_allow_symmetric:
      hinge_dict[h_hash] = dataclasses.replace(
          hinge_dict[h_hash], force_allow_symmetric=True
      )
  return hinge_dict


def _infer_grid_from_creases(creases: Sequence[Any]) -> int:
  """Infers integer grid size from crease coordinates."""
  if not creases:
    return 0
  try:
    max_coord = max(max(c.p1.x, c.p1.y, c.p2.x, c.p2.y) for c in creases)
    if max_coord > 1.5:
      return int(round(max_coord))
    vals = set()
    for c in creases:
      for v in (c.p1.x, c.p1.y, c.p2.x, c.p2.y):
        if v > 1e-6:
          vals.add(v)
    if not vals:
      return 0
    for g in range(1, 129):
      if all(abs(v * g - round(v * g)) < 1e-4 for v in vals):
        return g
  except (TypeError, ValueError, AttributeError):
    pass
  return 0


def _safe_infer_cp_grid_size(cp: Any) -> int | None:
  """Safely returns cp.grid_size or infers integer grid size from creases."""
  if cp is None:
    return None
  if hasattr(cp, 'details') and cp.details is not None:
    d = cp.details
    g = getattr(d, 'grid_size', None)
    if g is None and isinstance(d, dict):
      g = d.get('grid_size')
    if g is not None and g > 0:
      return int(g)
  if hasattr(cp, 'packing_cp'):
    cp = cp.packing_cp
    if cp is None:
      return None
  g = getattr(cp, 'grid_size', None)
  if g is not None and g > 0:
    return int(g)
  if hasattr(cp, 'creases') and cp.creases:
    inferred = _infer_grid_from_creases(cp.creases)
    if inferred > 0:
      return inferred
  return None


def valid_crease_pattern(
    cp: crease_pattern.CreasePattern,
    grid_size: int | None,
    local_flat_foldability: bool,
    global_flat_foldability: bool,
    use_python_flat_folder: bool = True,
) -> bool:
  """Validates that the crease pattern is flat foldable."""
  del use_python_flat_folder
  if local_flat_foldability and not cp.local_flat_foldability():
    return False
  if global_flat_foldability:
    eff_grid = (
        grid_size
        if (grid_size is not None and grid_size > 0)
        else (_safe_infer_cp_grid_size(cp) or -1)
    )
    if not global_foldability.is_globally_flat_foldable(
        cp,
        scale=1,
        ndigits=3,
        grid=eff_grid,
    ):
      return False
  return True


def add_solution_if_valid_state(
    borders: list[crease.Crease],
    state: State,
    solver_config: SolverConfig,
    time_taken: float,
) -> SolverResult | None:
  """Adds the crease pattern to the state if it is valid.

  If the remove_central_crease flag is set, the central crease is removed
  before adding the solution.

  Args:
    borders: The border creases of the pattern.
    state: The current state of the crease pattern.
    solver_config: The solver configuration.
    time_taken: The time taken to find the solution.

  Returns:
    A CreasePattern if the state is valid, otherwise None.
  """
  state.assign_pleats(force_assign=True)
  if solver_config.remove_central_crease:
    valid_state = state.remove_central_crease(state.transform)
  else:
    valid_state = True
  if not valid_state:
    return None
  cp = crease_pattern.CreasePattern(
      borders + state.ridges + state.pleats + state.hinges,
      grid_size=solver_config.grid_size,
  )
  if solver_config.is_valid_crease_pattern(cp):
    return SolverResult(
        crease_pattern=cp, final_state=state, time_taken=time_taken
    )
  return None


@dataclasses.dataclass(kw_only=True, frozen=True)
class SolverMetrics:
  """Metrics collected during the hinge crease solving process.

  Attributes:
    step: The current iteration number in the solver's search process.
    unsolved_vertices: Number of vertices that are not yet locally
      flat-foldable.
    states: Number of unique states that exist in the active state queue.
    hinges: Number of hinges that have been added and assigned a fold
    pleats: Number of assigned pleat (axial crease) segments
    partitions: Number of solved partitions. Will always be 0 if partitioning is
      disabled.
    solutions: Number of valid, complete crease patterns found so far.
  """

  # Warning this must be in the same order as the attributes.
  LOG_FORMAT = (
      'step:%d Unsolved vertices:%d #States:%d #Hinges:%d #Pleats:%d'
      ' #Partitions:%d #Solutions:%d'
  )

  step: int
  unsolved_vertices: int
  states: int
  hinges: int
  pleats: int
  partitions: int
  solutions: int

  def log(self):
    """Logs the metrics in a human-readable format as an INFO message."""
    logging.info(self.LOG_FORMAT, *dataclasses.astuple(self))

  def print(self):
    """Prints the metrics in a human-readable format.

    This is useful for debugging the solver in Colab as absl logging is not
    captured in Colab by default. Enable with:
    `SolverConfig(custom_metric_callback=SolverMetrics.print)`.
    """
    print(self.LOG_FORMAT % dataclasses.astuple(self))


class _ClassOrInstanceMethod:
  """Descriptor allowing a method to be called on either a class or an instance."""

  def __init__(self, func: Any):
    self.func = func

  def __get__(self, obj: Any, cls: Any = None) -> Any:
    if obj is not None:
      return lambda *args, **kwargs: self.func(obj, *args, **kwargs)
    return lambda *args, **kwargs: self.func(cls, *args, **kwargs)


@dataclasses.dataclass(kw_only=True, frozen=True)
class SolverConfig:
  """Configuration for the hinge crease Solver.

  Attributes:
    grid_size: The size of the grid to search.
    check_local_flat_foldability: Whether to check for local flat foldability.
    check_global_flat_foldability: Whether to check for global flat foldability.
    use_python_flat_folder: Whether to use python flat folder.
    max_steps: The maximum number of steps to take.
    max_time_limit: The maximum time limit in seconds to solve the problem.
    max_number_of_cps: The maximum number of crease patterns to return.
    log_progress: Whether to log the progress of the search.
    tie_break_by_hinge_symmetry: Whether to tie break by hinge symmetry score.
    remove_central_crease: Whether to remove the central crease.
    use_partition_order: Whether to use partition order to guide search.
    custom_metric_callback: A callback function for solver metrics.
    solution_callback: A callback function that can modify found solutions.
      Returning None from this callback will mark the solution as invalid.
    disallow_trapping_edge_flaps: Whether to disallow trapping edge flaps.
    hinges_from_stick_figure: If True, the solver will only add hinges that are
      consistent with the stick figure.
    packing_information: Bound by make_from_cp; do not pass when constructing
      hyperparameter configs.
    preassignments: Bound by make_from_cp; do not pass when constructing
      hyperparameter configs.
    sort_solutions: Whether to sort the solutions by symmetry and number of
      creases.
    allow_misaligned_symmetric_hinges: Whether to allow misaligned symmetric
      hinges.
    max_attempts_per_cp: The maximum number of attempts to allow per desired
      number of solutions. (attempts are rejected if they are not valid). Total
      attempts = max_attempts_per_cp * max_number_of_cps.
    up_down_method: the method to use for up/down assignments.
    allow_constraint_relaxation: Whether to relax heuristic flat/symmetric
      restrictions when vertices would otherwise be unsolveable.
    recovery: Whether to enable cross-partition hinge fallbacks, odd-degree
      vertex pruning, and multi-partition progress advancement.
    max_branching_factor: The maximum number of next states to generate per
      state.
    solver: The hinge solver backend to use ('default', 'builtin', or any
      external solver registered via plugins.register_solver).
    num_workers: Number of parallel search workers for registered solver plugins
      (defaults to 1 for fork-safety).
  """

  grid_size: int | None = None
  check_local_flat_foldability: bool = True
  check_global_flat_foldability: bool = True
  use_python_flat_folder: bool = True
  max_steps: int | None = 1000
  max_time_limit: float = 300.0
  max_number_of_cps: int = 1
  log_progress: bool = False
  tie_break_by_hinge_symmetry: bool = True
  remove_central_crease: bool = False
  use_partition_order: bool = True
  custom_metric_callback: Callable[[SolverMetrics], None] = dataclasses.field(
      default=lambda _: None, compare=False
  )
  solution_callback: Callable[[SolverResult], SolverResult | None] = (
      dataclasses.field(default=lambda x: x, compare=False)
  )
  disallow_trapping_edge_flaps: bool = False
  packing_information: Any = dataclasses.field(default=None, compare=False)
  hinges_from_stick_figure: bool = False
  preassignments: crease_pattern.CreasePattern | None = dataclasses.field(
      default=None, compare=False
  )
  sort_solutions: bool = True
  allow_misaligned_symmetric_hinges: bool = False
  max_attempts_per_cp: int = 100
  up_down_method: str = 'flaps_down'
  allow_constraint_relaxation: bool = False
  recovery: bool = False
  max_branching_factor: int | None = None
  solver: str = 'default'
  num_workers: int = 1

  @_ClassOrInstanceMethod
  def with_recovery(  # pylint: disable=no-self-argument
      self_or_cls: Any, **overrides: Any
  ) -> 'SolverConfig':
    """Returns a SolverConfig with extended recovery and relaxation enabled."""
    if isinstance(self_or_cls, SolverConfig):
      opts: dict[str, Any] = {
          'allow_constraint_relaxation': True,
          'recovery': True,
      }
      opts.update(overrides)
      return dataclasses.replace(self_or_cls, **opts)
    defaults = {
        'allow_constraint_relaxation': True,
        'recovery': True,
    }
    defaults.update(overrides)
    return self_or_cls(**defaults)

  def asdict(self) -> dict[str, Any]:
    """Returns a dictionary of the serializable solver hyperparameters."""
    config_dict = {}
    for f in dataclasses.fields(self):
      if f.name in ('packing_information', 'preassignments'):
        continue
      field_value = getattr(self, f.name)
      if callable(field_value) or isinstance(
          field_value, crease_pattern.CreasePattern
      ):
        continue
      config_dict[f.name] = field_value
    return config_dict

  @classmethod
  def from_dict(cls, data: Mapping[str, Any]) -> SolverConfig:
    """Constructs a SolverConfig from a dictionary."""
    valid_fields = {
        f.name
        for f in dataclasses.fields(cls)
        if f.init
        and f.name not in (
            'custom_metric_callback',
            'solution_callback',
            'packing_information',
            'preassignments',
        )
    }
    return cls(**{k: v for k, v in data.items() if k in valid_fields})

  def handle_metrics(self, metrics: SolverMetrics):
    """Logs and/or calls a custom callback with the provided metrics.

    Called by the solver at every step.

    Args:
      metrics: The metrics from the current solving step.
    """
    if self.log_progress:
      metrics.log()
    self.custom_metric_callback(metrics)

  def solution_score(
      self, result: SolverResult | bus.SolvedCreasePattern
  ) -> tuple[float, int]:
    """Scores solutions by symmetry and number of hinges."""
    if isinstance(result, bus.SolvedCreasePattern):
      symmetry_score = (
          result.stats.symmetry_score
          if self.tie_break_by_hinge_symmetry
          else 0.0
      )
      return symmetry_score, len(result.creases.creases)
    symmetry_score = (
        hinge_symmetry_score(result.final_state)
        if self.tie_break_by_hinge_symmetry
        else 0
    )
    return symmetry_score, len(result.crease_pattern.creases)

  _solution_score = solution_score

  def is_valid_crease_pattern(self, cp: crease_pattern.CreasePattern) -> bool:
    """Checks if a crease pattern is valid according to this config."""
    resolved_grid_size = (
        self.grid_size
        if self.grid_size is not None
        else _safe_infer_cp_grid_size(cp)
    )
    return valid_crease_pattern(
        cp,
        resolved_grid_size,
        local_flat_foldability=self.check_local_flat_foldability,
        global_flat_foldability=self.check_global_flat_foldability,
        use_python_flat_folder=self.use_python_flat_folder,
    )

  def process_solutions(
      self, solutions: list[SolverResult]
  ) -> list[SolverResult]:
    """Sorts the solutions to maximize symmetry score and number of hinges."""
    if not self.sort_solutions:
      return solutions
    return sorted(solutions, key=self.solution_score, reverse=True)

  def make_from_state(
      self,
      state: State,
      borders: list[crease.Crease],
      *,
      initial_stage_snapshots: tuple[
          tuple[str, crease_pattern.CreasePattern], ...
      ] = (),
      source_cp: crease_pattern.CreasePattern | None = None,
  ) -> Solver:
    return Solver(
        config=self,
        init_state=state,
        borders=borders,
        initial_stage_snapshots=initial_stage_snapshots,
        source_cp=source_cp,
    )

  def bind(
      self,
      *,
      grid_size: int,
      packing_information: Any = None,
      preassignments: crease_pattern.CreasePattern | None = None,
  ) -> SolverConfig:
    """Returns a copy bound to problem-instance state."""
    replacements: dict[str, Any] = {'grid_size': grid_size}
    effective_packing_info = (
        packing_information
        if packing_information is not None
        else self.packing_information
    )
    if isinstance(effective_packing_info, packing_output.PackingOutput):
      effective_packing_info = bus.PackedCreasePattern.from_packing_output(
          effective_packing_info, design=None, grid_size=grid_size
      )
    if effective_packing_info is not None:
      replacements['packing_information'] = effective_packing_info
    if preassignments is not None:
      replacements['preassignments'] = preassignments
    return dataclasses.replace(self, **replacements)

  def make_from_cp(
      self,
      cp: crease_pattern.CreasePattern,
      *,
      grid_size: int | None = None,
      packing_information: Any = None,
      preassignments: crease_pattern.CreasePattern | None = None,
      record_stages: bool = False,
  ) -> Solver:
    """Creates a Solver instance from a crease pattern.

    This method initializes the solver's state by separating the input crease
    pattern into borders, hinges, and ridges. It also performs some initial
    assignments of pleats and ridges.

    Args:
      cp: The input crease pattern.
      grid_size: Optional grid size override for the problem.
      packing_information: Optional packing metadata for stick-figure-aware
        hinge filtering.
      preassignments: Optional crease pattern with preassigned hinges.
      record_stages: Whether to capture intermediate stage snapshots for
        visualization.

    Returns:
      A Solver instance configured with the initial state derived from the CP.
    """
    resolved_grid_size = grid_size
    if resolved_grid_size is None:
      resolved_grid_size = self.grid_size
    if resolved_grid_size is None:
      resolved_grid_size = _safe_infer_cp_grid_size(cp)
    if resolved_grid_size is None or resolved_grid_size <= 0:
      raise ValueError(
          'grid_size must be positive; pass grid_size to'
          ' SolverConfig.make_from_cp or set cp.grid_size.'
      )
    bound_config = self.bind(
        grid_size=resolved_grid_size,
        packing_information=packing_information,
        preassignments=preassignments,
    )

    has_extended_river = False
    if packing_information is not None:
      for attr_name in ('rivers', 'packing_rivers'):
        river_map = getattr(packing_information, attr_name, None)
        if isinstance(river_map, Mapping) and any(
            getattr(r, 'requires_recovery', False) for r in river_map.values()
        ):
          has_extended_river = True
          break
    if (
        plugins.solver_requires_recovery(self.solver) or has_extended_river
    ) and not bound_config.recovery:
      bound_config = bound_config.with_recovery()

    borders = [c for c in cp.creases if c.fold == crease.Fold.BORDER]
    hinges = [c for c in cp.creases if c.fold == crease.Fold.HINGE]
    ridges = [c for c in cp.creases if c.fold == crease.Fold.MOUNTAIN]

    unassigned_pleats = pleat_utils.create_pleats(
        cp, grid_size=resolved_grid_size, recovery=bound_config.recovery
    )
    initial_unassigned_pleats = list(unassigned_pleats) if record_stages else []
    assigned_pleats = pleat_utils.assign_pleats(
        unassigned_pleats,
        grid_size=resolved_grid_size,
        recovery=bound_config.recovery,
    )
    initial_assigned_pleats = list(assigned_pleats) if record_stages else []
    unassigned_ridges, unassigned_central_ridges = (
        box_pleating_utils.split_unassigned_and_central_ridges(ridges)
    )

    assigned_ridges = ridge_utils.assign_ridges(
        unassigned_ridges,
        assigned_pleats,
        grid_size=resolved_grid_size,
        recovery=bound_config.recovery,
    )
    new_cp = crease_pattern.CreasePattern(
        borders + assigned_pleats + assigned_ridges,
        grid_size=resolved_grid_size,
        light_mode=True,
    )
    vertices = [
        v
        for v in new_cp.vertices
        if not v.flat_foldability(check_maekawa=bound_config.recovery)
    ]
    assigned_central_ridges, assigned_pleats, unassigned_pleats = (
        assign_central_ridges(
            unassigned_central_ridges,
            assigned_pleats,
            vertices,
            resolved_grid_size,
        )
    )
    central_assigned_pleats = list(assigned_pleats) if record_stages else []
    central_unassigned_pleats = list(unassigned_pleats) if record_stages else []
    assigned_pleats, unassigned_pleats = add_pleats_to_unassigned(
        assigned_pleats,
        unassigned_pleats,
        ridges,
        hinges,
        grid_size=resolved_grid_size,
    )

    initial_stage_snapshots: tuple[
        tuple[str, crease_pattern.CreasePattern], ...
    ] = ()
    if record_stages:
      initial_stage_snapshots = (
          ('(1) Packing CP', cp),
          (
              '(2) Unassigned Pleats',
              crease_pattern.CreasePattern(
                  borders + ridges + initial_unassigned_pleats,
                  grid_size=resolved_grid_size,
              ),
          ),
          (
              '(3) Assigned Pleats',
              crease_pattern.CreasePattern(
                  borders + ridges + initial_assigned_pleats,
                  grid_size=resolved_grid_size,
              ),
          ),
          # Not new_cp: that light_mode CP omits the still-unassigned central
          # ridges and leaves borders unsplit.
          (
              '(4) Assigned Ridges',
              crease_pattern.CreasePattern(
                  borders
                  + initial_assigned_pleats
                  + assigned_ridges
                  + unassigned_central_ridges,
                  grid_size=resolved_grid_size,
              ),
          ),
          (
              '(5) Central Ridges',
              crease_pattern.CreasePattern(
                  borders
                  + central_assigned_pleats
                  + assigned_ridges
                  + assigned_central_ridges,
                  grid_size=resolved_grid_size,
              ),
          ),
          (
              '(6) + Unassigned Pleats',
              crease_pattern.CreasePattern(
                  borders
                  + central_assigned_pleats
                  + assigned_ridges
                  + assigned_central_ridges
                  + central_unassigned_pleats,
                  grid_size=resolved_grid_size,
              ),
          ),
      )

    partition_order = None
    if bound_config.use_partition_order:
      cp_grid = _safe_infer_cp_grid_size(cp)
      if bound_config.recovery or (
          cp_grid is not None and resolved_grid_size > cp_grid
      ):
        partition_order = partition_utils.OrderedPartitions.from_cp(cp)
      else:
        partition_order = partition_utils.OrderedPartitions(
            tuple(partition_utils.partition_cp(cp))
        )
    init_state = State.init_state(
        pleats=assigned_pleats,
        unassigned_pleats=unassigned_pleats,
        hinges=hinges,
        ridges=assigned_ridges + assigned_central_ridges,
        borders=borders,
        solver_config=bound_config,
        partition_order=partition_order,
    )
    if bound_config.preassignments is not None:
      init_state = init_state.assign_as_symmetric([
          c
          for c in bound_config.preassignments.creases
          if c.fold == crease.Fold.HINGE
      ])

    return bound_config.make_from_state(
        init_state,
        borders,
        initial_stage_snapshots=initial_stage_snapshots,
        source_cp=cp,
    )


@dataclasses.dataclass(kw_only=True)
class Solver:
  """Solves a box-pleating crease pattern by adding hinges.

  Attributes:
    config: The solver configuration.
    init_state: The initial state of the crease pattern.
    borders: The border creases of the pattern.
    initial_stage_snapshots: Optional intermediate stage snapshots captured
      before hinge search.
    source_cp: Optional input packing crease pattern.
    active_states: A dictionary of active states, mapping state hash to the
      State object.
    solutions: A list of found flat-foldable crease patterns and their final
      states.
    step_count: Number of steps executed by the solver.
    start_time: The time when the solver started.
  """

  config: SolverConfig
  init_state: State
  borders: list[crease.Crease]
  initial_stage_snapshots: tuple[
      tuple[str, crease_pattern.CreasePattern], ...
  ] = ()
  source_cp: crease_pattern.CreasePattern | None = None

  active_states: MutableMapping[int, State] = dataclasses.field(
      init=False, default_factory=ActiveStates
  )

  solutions: list[SolverResult] = dataclasses.field(
      init=False, default_factory=list
  )
  _n_solution_attempts: int = dataclasses.field(default=0, init=False)
  step_count: int = dataclasses.field(default=0, init=False)

  # Initialized in solve()
  start_time: float | None = None

  @property
  def is_exhausted(self) -> bool:
    """Returns True if all active states have been completely explored."""
    return not any(not s.expanded for s in self.active_states.values())

  def __post_init__(self):
    self.active_states = ActiveStates(self.config.tie_break_by_hinge_symmetry)
    self.active_states[hash(self.init_state)] = self.init_state

  def state_to_cp(
      self, state: State, remove_central_crease: bool | None = None
  ) -> crease_pattern.CreasePattern:
    """Converts a state to a crease pattern for visualization and validity."""
    temp_state = state.copy_state()
    temp_state.pleats.extend(temp_state.assign_pleats())

    if remove_central_crease is None:
      remove_central_crease = self.config.remove_central_crease
    if remove_central_crease:
      temp_state.remove_central_crease(temp_state.transform)

    return crease_pattern.CreasePattern(
        self.borders
        + temp_state.ridges
        + temp_state.pleats
        + temp_state.hinges,
        grid_size=self.config.grid_size,
    )

  def _calc_metrics(self, step: int, state: State) -> SolverMetrics:
    """Calculates metrics for the current state."""
    return SolverMetrics(
        step=step,
        unsolved_vertices=len(self.init_state.vertices) - state.state_score,
        states=len(self.active_states),
        hinges=(
            len(self.init_state.unvisited_hinges) - len(state.unvisited_hinges)
        ),
        pleats=len(state.pleats),
        partitions=state.partition_progress,
        solutions=len(self.solutions),
    )

  def step(self, i: int = -1) -> tuple[State, bool]:
    """Performs a single step of the search.

    Args:
      i: The current step number, used for logging.

    Returns:
      A tuple containing:
        - The state that was expanded in this step.
        - A boolean indicating if the solver should stop (e.g., because enough
          solutions have been found).
    """
    if self.start_time is None:
      self.start_time = time.perf_counter()
    state = get_state_to_expand(
        self.active_states,
        tie_break_by_hinge_symmetry=self.config.tie_break_by_hinge_symmetry,
    )
    if time.perf_counter() - self.start_time > self.config.max_time_limit:  # pyrefly: ignore[unsupported-operation]
      logging.info('Time limit reached. Stopping solver.')
      return state, True
    if (
        self._n_solution_attempts
        >= self.config.max_number_of_cps * self.config.max_attempts_per_cp
    ):
      logging.warning(
          'Maximum number of validity attempts reached. Stopping solver.'
      )
      state.expanded = True
      return state, True

    # All states are expanded, no more progress can be made.
    if state.expanded:
      logging.info('Could not find any more states to expand.')
      state.expanded = True
      return state, True
    metrics = self._calc_metrics(i, state)
    self.config.handle_metrics(metrics)

    if state.state_score == len(self.init_state.vertices):  # Stopping criteria
      valid_solution = add_solution_if_valid_state(
          self.borders,
          state.copy_state(),
          self.config,
          time.perf_counter() - self.start_time,  # pyrefly: ignore[unsupported-operation]
      )
      if valid_solution:
        valid_solution = self.config.solution_callback(valid_solution)
        if valid_solution:
          self.solutions.append(valid_solution)
          if len(self.solutions) >= self.config.max_number_of_cps:
            return state, True
      self._n_solution_attempts += 1
      if (
          not self.config.recovery
          or valid_solution
          or not state.unvisited_hinges
      ):
        state.expanded = True
        return state, False
    elif not state.unvisited_hinges:
      state.expanded = True
      return state, False

    # get_next_states updates self.active_states in-place.
    state.get_next_states(self.active_states)
    return state, False

  def iter_solutions(self) -> Iterator[SolverResult]:
    """Yields solved crease patterns as the search discovers them."""
    self.start_time = time.perf_counter()

    if self.config.solver not in ('default', 'builtin'):
      solver_fn = plugins.get_solver(self.config.solver)
      if solver_fn is None:
        raise ValueError(
            f'Unknown or unregistered solver {self.config.solver!r}. '
            f'Available: {plugins.list_solvers()}'
        )
      try:
        sols, stats = solver_fn(
            self.init_state,
            self.borders,
            self.config,
            source_cp=self.source_cp,
        )
        self.step_count = (
            getattr(stats, 'solve_calls', stats)
            if hasattr(stats, 'solve_calls')
            else (stats if isinstance(stats, int) else 1)
        )
      finally:
        for s in self.active_states.values():
          s.expanded = True
      for sol in sols:
        self.solutions.append(sol)
        yield sol
      return

    if (
        self.config.remove_central_crease
        and not central_fold_utils.should_try_to_remove_central_crease(
            self.init_state.unvisited_hinges
            + self.init_state.ridges
            + self.init_state.pleats
            + self.init_state.unassigned_pleats,
            self.init_state.unvisited_hinges,
        )
    ):
      raise NoSolutionError('Cannot remove central crease.')
    if self.is_exhausted:
      logging.info('Search frontier is already exhausted.')
      return
    start_step = self.step_count
    max_steps = (
        self.config.max_steps
        if self.config.max_steps is not None
        else 1_000_000
    )
    for i in range(start_step, start_step + max_steps):
      self.step_count = i + 1
      timeout_lib.maybe_timeout_cooperatively()
      prev_len = len(self.solutions)
      _, is_done = self.step(i)
      while prev_len < len(self.solutions):
        yield self.solutions[prev_len]
        prev_len += 1
      if is_done:
        logging.info(
            'Solver found %d solutions after %d steps.',
            len(self.solutions),
            i,
        )
        break

  def solve(self) -> list[SolverResult]:
    """Run the solver and return the solved crease patterns.

    Returns:
      A list of solutions found as SolverResults.

    Raises:
      NoSolutionError: If no solution is found after exhausting the search
        or if the central crease cannot be removed as configured.
    """
    solutions = list(self.iter_solutions())
    if not solutions:
      raise NoSolutionError('No solutions found.')

    if len(solutions) < self.config.max_number_of_cps:
      logging.info(
          'Found %d solutions out of %d.',
          len(solutions),
          self.config.max_number_of_cps,
      )

    return self.config.process_solutions(solutions)


def solve_crease_pattern(
    hinges: list[crease.Crease],
    borders: list[crease.Crease],
    pleats: list[crease.Crease],
    unassigned_pleats: list[crease.Crease],
    ridges: list[crease.Crease],
    grid_size: int,
    check_local_flat_foldability: bool,
    check_global_flat_foldability: bool,
    max_steps: int = 1000,
    max_time_limit: float = 300.0,
    max_number_of_cps: int = 1,
    log_progress: bool = False,
    tie_break_by_hinge_symmetry: bool = True,
    remove_central_crease: bool = False,
    use_python_flat_folder: bool = True,
) -> list[SolverResult]:
  """Finds flat foldable crease patterns by adding hinges and reassigns pleats."""
  solver_config = SolverConfig(
      grid_size=grid_size,
      check_local_flat_foldability=check_local_flat_foldability,
      check_global_flat_foldability=check_global_flat_foldability,
      use_python_flat_folder=use_python_flat_folder,
      max_steps=max_steps,
      max_time_limit=max_time_limit,
      max_number_of_cps=max_number_of_cps,
      log_progress=log_progress,
      tie_break_by_hinge_symmetry=tie_break_by_hinge_symmetry,
      remove_central_crease=remove_central_crease,
  )
  init_state = State.init_state(
      pleats=pleats,
      unassigned_pleats=unassigned_pleats,
      hinges=hinges,
      ridges=ridges,
      borders=borders,
      solver_config=solver_config,
      partition_order=None,
  )
  solver = solver_config.make_from_state(init_state, borders)

  return solver.solve()


def cp_from_packing(
    cp: crease_pattern.CreasePattern,
    grid_size: int = 32,
    sort_solutions_by_num_creases: bool = True,
    print_progress: bool = False,
    **solver_config_kwargs,
) -> list[SolverResult]:
  """Creates a crease pattern from packing crease pattern and grid size."""
  if print_progress:
    if 'custom_metric_callback' in solver_config_kwargs:
      raise ValueError(
          'Only one of print_progress and custom_metric_callback can be set.'
      )
    solver_config_kwargs['custom_metric_callback'] = (
        SolverMetrics.print if print_progress else lambda _: None
    )
  solver_config_kwargs['sort_solutions'] = sort_solutions_by_num_creases
  solver_config = SolverConfig(
      grid_size=grid_size,
      **solver_config_kwargs,
  )
  solver = solver_config.make_from_cp(cp)
  return solver.solve()


def clear_caches() -> None:
  """Clears all LRU caches to free memory.

  This should be called after each task to prevent unbounded memory growth
  when processing many tasks sequentially.
  """
  split_pleat_to_segments.cache_clear()
  _make_pleat_segment.cache_clear()
  _transform_crease.cache_clear()
