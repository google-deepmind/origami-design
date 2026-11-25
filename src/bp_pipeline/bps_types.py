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

"""TypedDict schema definitions for Box Pleating Studio (`.bps.json`) files."""

from typing import Any, NotRequired, TypedDict


class SheetDict(TypedDict):
  """Defines the properties of a sheet object."""

  type: str
  height: int
  width: int


class FlapDict(TypedDict):
  """Defines the properties of a flap object in the layout."""

  id: int
  x: int
  y: int
  width: int
  height: int


class ReservationDict(TypedDict):
  """Defines a reserved grid rectangle in the layout."""

  id: str
  x: int
  y: int
  width: int
  height: int


class LayoutDict(TypedDict):
  """Defines the layout containing the sheet and its flaps."""

  sheet: SheetDict
  flaps: list[FlapDict]
  stretches: list[Any]
  reservations: NotRequired[list[ReservationDict]]


class NodeDict(TypedDict):
  """Defines a node in the tree structure."""

  id: int
  name: str
  x: float
  y: float


class EdgeDict(TypedDict):
  """Defines an edge connecting two nodes in the tree."""

  n1: int
  n2: int
  length: float | None


class TreeObjectDict(TypedDict):
  """Defines the core tree structure, including nodes and edges."""

  sheet: SheetDict
  nodes: list[NodeDict]
  edges: list[EdgeDict]


class DesignDict(TypedDict):
  """Contains metadata and the main tree object for the design."""

  title: str
  mode: str
  description: str
  layout: LayoutDict
  tree: TreeObjectDict


class BoxPleatingTreeData(TypedDict):
  """The top-level structure for a Box Pleating Studio (`.bps.json`) file."""

  version: str
  design: DesignDict
