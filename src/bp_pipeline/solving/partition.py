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

"""Partitions a box-pleating crease pattern into distinct groups of hinges."""

from __future__ import annotations

from collections.abc import Iterable
import dataclasses
import functools
import itertools
import types
import typing
from typing import Any, Self

from absl import logging
from bp_pipeline.crease_pattern import crease as crease_lib
from bp_pipeline.crease_pattern import crease_pattern
from bp_pipeline.crease_pattern import point as point_lib
from bp_pipeline.crease_pattern import vertex as vertex_lib
from bp_pipeline.solving import utils as box_pleating_utils
import immutabledict

if typing.TYPE_CHECKING:
  from matplotlib import axes
  from matplotlib import figure


@dataclasses.dataclass(kw_only=True, frozen=True)
class HingePartition:
  """A partition of a box-pleating crease pattern.

  This class represents a connected component of hinges and their associated
  ridges and vertices within a larger box-pleating crease pattern.

  Attributes:
    hinges: The hinge creases in this partition.
    ridges: The ridge creases in this partition.
    vertices: An immutabledict of points to Vertex objects for all vertices
      involved in this partition's hinges and ridges.
  """

  hinges: crease_lib.Creases
  ridges: crease_lib.Creases
  vertices: immutabledict.immutabledict[point_lib.Point, vertex_lib.Vertex]

  @functools.cached_property
  def non_border_vertices(
      self,
  ) -> immutabledict.immutabledict[point_lib.Point, vertex_lib.Vertex]:
    """immutabledict of points to vertices that are not on the border of the CP."""
    return immutabledict.immutabledict(
        pv for pv in self.vertices.items() if not pv[0].is_border
    )

  @functools.cached_property
  def _all_creases(self) -> tuple[crease_lib.Crease, ...]:
    return tuple(c.crease for c in self.hinges.creases | self.ridges.creases)

  def union(self, other: Self) -> Self:
    return HingePartition(  # pyrefly: ignore[bad-return]
        hinges=crease_lib.Creases(self.hinges.creases | other.hinges.creases),
        ridges=crease_lib.Creases(self.ridges.creases | other.ridges.creases),
        vertices=self.vertices | other.vertices,
    )

  def __or__(self, other: object) -> Self | types.NotImplementedType:
    if not isinstance(other, HingePartition):
      return NotImplemented
    return self.union(other)  # pyrefly: ignore[bad-argument-type]

  def draw_as_cp(
      self,
      borders: Iterable[crease_lib.Crease] = (),
      **draw_cp_kwargs: Any,
  ) -> tuple[figure.Figure, axes.Axes]:
    """Draws the partition as a CreasePattern.

    Args:
      borders: Additional border creases to include in the drawing.
      **draw_cp_kwargs: Additional keyword arguments passed to
        `visualize.draw_cp`.

    Returns:
      A tuple containing the Matplotlib Figure and Axes objects.
    """
    from bp_pipeline.crease_pattern import visualize  # pylint: disable=g-import-not-at-top

    cp = crease_pattern.CreasePattern(
        [
            *(c.crease for c in self.hinges.creases | self.ridges.creases),
            *borders,
        ],
        light_mode=True,
    )
    # Overwrite the vertices with the ones in this partition
    cp.vertices = list(self.vertices.values())
    cp.sort_vertices()

    return visualize.draw_cp(cp, **draw_cp_kwargs)


def _has_hinges(vertex: vertex_lib.Vertex) -> bool:
  hinge_fold = crease_lib.Fold.HINGE
  return any(c.fold == hinge_fold for c in vertex.creases)


def partition_hinges_and_ridges(
    hinges: Iterable[crease_lib.Crease],
    ridges: Iterable[crease_lib.Crease],
    borders: Iterable[crease_lib.Crease],
    vertices: Iterable[vertex_lib.Vertex],
) -> list[HingePartition]:
  """Partitions a box-pleating crease pattern into distinct groups of hinges.

  This function finds connected components formed by hinges and "internal"
  ridges and borders. Internal ridges/borders are those that do not connect
  two vertices that are both part of a hinge. Each connected component
  is returned as a `HingePartition`. All passed iterators are fully consumed.

  Args:
    hinges: All hinge creases in the crease pattern.
    ridges: All ridge creases in the crease pattern.
    borders: All border creases in the crease pattern.
    vertices: All vertices in the crease pattern.

  Returns:
    A list of `HingePartition` instances, each representing a connected
    component of hinges and their associated internal ridges/borders and
    vertices.
  """
  vertex_by_point = {v.point: v for v in vertices}

  # Some ridges are internal to a flap and others bridge connected hinge paths.
  # We can use the same connected-component traversal to find all the internal
  # ridges as long as we exclude the ones that would bridge hinge paths.
  # i.e. They feature a hinge on both endpoints.
  # This also applies to border creases, which are necessary for usual cases.
  internal_creases = []
  for crease in itertools.chain(ridges, borders):
    p1_has_hinges = _has_hinges(vertex_by_point[crease.p1])
    p2_has_hinges = _has_hinges(vertex_by_point[crease.p2])
    if not (p1_has_hinges and p2_has_hinges):
      # This ridge doesn't connect two hinges, so we should search it
      internal_creases.append(crease)

  # Connected hinge paths are considered a single partition.
  paths = box_pleating_utils.creases_to_paths([*hinges, *internal_creases])

  partitions = []
  for path in paths:
    path_points = {*(c.p1 for c in path), *(c.p2 for c in path)}

    partitions.append(
        HingePartition(
            hinges=crease_lib.Creases.create_from_creases(
                c for c in path if c.fold == crease_lib.Fold.HINGE
            ),
            ridges=crease_lib.Creases.create_from_creases(
                c for c in path if c.fold == crease_lib.Fold.MOUNTAIN
            ),
            vertices=immutabledict.immutabledict(
                {p: vertex_by_point[p] for p in path_points}
            ),
        )
    )

  return partitions


def partition_cp(
    packing_cp: crease_pattern.CreasePattern,
) -> list[HingePartition]:
  """Partitions a box-pleating crease pattern into disjoint groups of hinges."""
  hinges = (c for c in packing_cp.creases if c.fold == crease_lib.Fold.HINGE)
  ridges = (c for c in packing_cp.creases if c.fold == crease_lib.Fold.MOUNTAIN)
  borders = (c for c in packing_cp.creases if c.fold == crease_lib.Fold.BORDER)

  return partition_hinges_and_ridges(
      hinges, ridges, borders, packing_cp.vertices
  )


@dataclasses.dataclass(frozen=True)
class OrderedPartitions:
  """Holds partitions of a box-pleating crease pattern."""

  partitions: tuple[HingePartition, ...]
  bridging_creases: tuple[crease_lib.Crease, ...] = ()

  def __post_init__(self):
    if not self.partitions:
      raise ValueError('OrderedPartitions must have at least one partition.')

    partitions = tuple(
        sorted(self.partitions, key=lambda p: len(p.vertices), reverse=True)
    )
    object.__setattr__(self, 'partitions', partitions)

  @classmethod
  def from_cp(cls, packing_cp: crease_pattern.CreasePattern) -> Self:
    """Creates OrderedPartitions from a packing crease pattern."""
    partitions = tuple(partition_cp(packing_cp))
    in_partition_creases = set()
    for p in partitions:
      in_partition_creases.update(p.hinges.creases)
      in_partition_creases.update(p.ridges.creases)
    bridging_creases = tuple(
        c
        for c in packing_cp.creases
        if c.exact_match not in in_partition_creases
    )
    return cls(partitions=partitions, bridging_creases=bridging_creases)

  def __len__(self) -> int:
    return len(self.partitions)

  @functools.cached_property
  def _point_to_partition_id(self) -> dict[point_lib.Point, int]:
    point_to_partition = {}
    for i, partition in enumerate(self.partitions):
      for v_point in partition.vertices:
        if v_point in point_to_partition:
          raise ValueError(f'Vertex {v_point} belongs to multiple partitions')
        point_to_partition[v_point] = i
    return point_to_partition

  def get_partition_id(self, point: point_lib.Point) -> int:
    """Returns the partition containing the point.

    Args:
      point: The point to find the partition for.

    Raises:
      ValueError: If the point is not contained in any of the partitions.
    """
    partition_id = self._point_to_partition_id.get(point)
    if partition_id is None:
      # Fallback to searching for a crease that the point touches.
      partition_id = self.search_partition_creases(point)
      if partition_id is None:
        raise ValueError(
            f'Point {point} is not in any partition:'
            f' {self._point_to_partition_id.keys()}'
        )
      self._point_to_partition_id[point] = partition_id
    return partition_id

  def search_partition_creases(self, point: point_lib.Point) -> int | None:
    """Returns the partition containing the point by searching all creases."""
    for i, partition in enumerate(self.partitions):
      if any(
          c.point_status(point) != crease_lib.PointStatus.EXTERNAL
          for c in partition._all_creases  # pylint: disable=protected-access
      ):
        logging.info('Found partition %d for point %s', i, str(point))
        return i
    for bridging_crease in self.bridging_creases:
      if bridging_crease.point_status(point) != crease_lib.PointStatus.EXTERNAL:
        p1_id = self._point_to_partition_id.get(bridging_crease.p1)
        p2_id = self._point_to_partition_id.get(bridging_crease.p2)
        if p1_id is not None and p2_id is not None:
          partition_id = min(p1_id, p2_id)
          logging.info(
              'Found bridging partition %d for point %s',
              partition_id,
              str(point),
          )
          return partition_id
    return None
