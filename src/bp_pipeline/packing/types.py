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

"""Defines shared data structures and types for the packing solver."""

from collections.abc import Mapping
import dataclasses
import enum
from typing import Any, Iterator, Literal, NewType, NotRequired, TypeAlias, TypeVar, TypedDict, Union

from ..crease_pattern import crease as crease_lib
from ..crease_pattern import point
from ..stick_figures import stick as stick_lib
from . import river as river_lib

VertexIndex = NewType('VertexIndex', int)
EdgeIndex = NewType('EdgeIndex', int)

# An edge is represented by the indices of the two vertices it connects.
Edge: TypeAlias = tuple[VertexIndex, VertexIndex]

# An unordered, hashable representation of an edge, useful for dictionary keys.
EdgeAsVertexSet: TypeAlias = frozenset[VertexIndex]

COPY_SUFFIX = '_copy'
RIVER_SF1_SUFFIX = '__ls_river_sf1'
RIVER_SF2_SUFFIX = '__ls_river_sf2'
AXIS_FILLER_PREFIX = 'axis_filler'


class NoPackingFoundError(Exception):
  """Raised when no packing solution can be found."""


@dataclasses.dataclass(frozen=True)
class RiverSfInfo:
  """Metadata for a river sacrificial flap, attached to its PolygonPlacement.

  Attributes:
    river_name: The name of the river stick this SF belongs to.
    river_side: The river side name (e.g. 'top', 'left', 'inner', 'hugging').
    parent_joint: The stick figure joint this SF connects to.
  """

  river_name: str
  river_side: str
  parent_joint: str


class TraversalMode(enum.StrEnum):
  """The traversal mode to use for packing_traversal."""

  PROXIMITY = 'proximity'
  RIVER_PRIORITY = 'river_priority'
  SIZE = 'size'


class HingeInfoDict(TypedDict):
  """A dictionary representation of HingeInfo."""

  coordinate: dict[str, Any] | None
  assigned: bool
  flap_names: tuple[str, str]


@dataclasses.dataclass(kw_only=True)
class HingeInfo:
  """Information about a hinge in a packing.

  Attributes:
    coordinate: The crease representing the hinge segment.
    assigned: Whether this hinge on the packing gets assigned in the final CP.
    flap_names: Tuple with names of flaps or rivers sharing hinge (1 or 2
      elements).
  """

  coordinate: crease_lib.Crease | None
  assigned: bool
  flap_names: tuple[str, str]

  @classmethod
  def from_dict(
      cls,
      h_dict: dict[
          str,
          Union[
              dict[str, Any] | None,
              bool,
              tuple[str, ...],
          ],
      ],
  ) -> 'HingeInfo':
    """Converts a dictionary to a HingeInfo object."""
    return cls(
        coordinate=crease_lib.Crease.from_dict(h_dict['coordinate']),  # pyrefly: ignore[bad-argument-type]
        assigned=h_dict['assigned'],  # pyrefly: ignore[bad-argument-type]
        flap_names=tuple(h_dict['flap_names']),  # pyrefly: ignore[bad-argument-type]
    )


@dataclasses.dataclass(frozen=True)
class PolygonPlacement:
  """Describes the placement of a single polygon on the grid.

  Attributes:
    offset: The (x, y) coordinate of the top-left corner of polygon's bounding
      box.
    size: The (width, height) of polygon's bounding box.
    center: The normalized center of the polygon's bounding box.
    coords: If not None, list of normalized coords defining a polygon shape.
    river_sf_info: If not None, this contains information about river that this
      is a sacrificial flap for.
  """

  offset: tuple[int, int]
  size: tuple[int, int]
  center: point.Point
  coords: list[tuple[float, float]] | None = None
  river_sf_info: RiverSfInfo | None = None

  @classmethod
  def from_grid_coords(
      cls,
      *,
      offset: tuple[int, int],
      size: tuple[int, int],
      grid_size: int,
      require_on_paper: bool = True,
      river_sf_info: RiverSfInfo | None = None,
  ) -> 'PolygonPlacement':
    """Factory to create a placement with a normalized center.

    Args:
        offset: The (x, y) coordinate of the top-left corner of the polygon.
        size: The (width, height) of the polygon.
        grid_size: The size of the grid.
        require_on_paper: Whether to require the center to be on the paper.
        river_sf_info: If not None, this contains information about river that
          this is a sacrificial flap for.

    Raises:
        ValueError: If grid_size is not positive, if the polygon dimensions are
          not even (center must be on integer grid), or if require_on_paper is
          True and the calculated center falls outside the grid boundaries.
    """
    if grid_size <= 0:
      raise ValueError('Grid size must be positive.')

    # We enforce even dimensions so that the center lies on the integer grid.
    if (size[0] | size[1]) & 1:
      raise ValueError('Center coordinates must be on the integer grid.')

    # The Point constructor will raise ValueError if require_on_paper is True
    # and the normalized center is outside the [0, 1] range, effectively
    # checking grid boundaries.
    inv_grid_size = 1.0 / grid_size
    center_x = (offset[0] + size[0] * 0.5) * inv_grid_size
    center_y = (offset[1] + size[1] * 0.5) * inv_grid_size
    normalized_center = point.Point(
        x=center_x, y=center_y, require_on_paper=require_on_paper
    )
    return cls(
        offset=offset,
        size=size,
        center=normalized_center,
        coords=None,
        river_sf_info=river_sf_info,
    )

  @classmethod
  def from_dict(cls, data: dict[str, Any]) -> 'PolygonPlacement':
    """Creates a PolygonPlacement object from a dictionary."""
    river_sf_data = data.get('river_sf_info')
    river_sf_info = RiverSfInfo(**river_sf_data) if river_sf_data else None
    return cls(
        offset=tuple(data['offset']),
        size=tuple(data['size']),
        center=point.Point.from_dict(data['center']),
        coords=data['coords'],
        river_sf_info=river_sf_info,
    )


class ConstraintType(enum.StrEnum):
  """Defines the types of constraints between edges.

  Attributes:
    EQUAL: Edges must have distance equal to a specific value.
    GREATER_THAN: Edges must have distance greater than a specific value.
    UNSPECIFIED: Edge constraint is not specified.
  """

  EQUAL = 'equal'
  GREATER_THAN = 'greater_than'
  UNSPECIFIED = 'unspecified'


class Constraint(TypedDict):
  """Describes a geometric constraint between two packing objects such as flaps or rivers.

  Attributes:
    type: The category of the constraint (e.g., equal, greater_than).
    value: The numeric value associated with the constraint (e.g., distance).
  """

  type: ConstraintType
  value: Union[int, float, str]


class TraversalStep(TypedDict):
  """Defines a single step in the packing solver's traversal plan.

  Attributes:
    stick: The stick to be placed.
    constraints: A dictionary of all active constraints that this object must
      satisfy against previously placed objects.
    area_sides: If the stick is a river, this contains the area on each side of
      the river, keyed by the joint name at that side. For example, `{'joint_A':
      25.0, 'joint_B': 36.0}`.
  """

  stick: stick_lib.Stick
  constraints: dict[str, Constraint]
  area_sides: NotRequired[dict[str, float]]


RiverPlacementAlias = (
    river_lib.River | river_lib.LShapedRiver | river_lib.HuggingRiver
)


class RiverCandidate(TypedDict):
  """Represents a potential valid placement for a river packing object.

  Attributes:
    stick_name: The stick name this placement corresponds to.
    river: The River object defining the placement.
    packed_flaps: A dictionary of packed flaps for this river placement.
    joint_to_side: A dictionary mapping joint name to river side name (e.g.
      'top', 'bottom', 'inner', 'outer').
    level_shifter_placements: A tuple of two placements for sacrificial flaps.
  """

  stick_name: str
  river: RiverPlacementAlias
  packed_flaps: dict[str, PolygonPlacement]
  joint_to_side: NotRequired[dict[str, str]]
  level_shifter_placements: NotRequired[
      tuple[PolygonPlacement, PolygonPlacement]
  ]


class Candidate(TypedDict):
  """Represents a potential valid placement for a packing object.

  This is the data structure that the `find_possible_*_placements` functions
  are expected to return.

  Attributes:
    type: A string indicating the type of object (e.g., 'FLAP').
    stick_name: The stick name this placement corresponds to.
    placement: The PolygonPlacement object defining the position and size of the
      object on the grid.
    partner_placement: The placement of the symmetric partner, if type is
      'PAIRED_ON_AXIS'.
  """

  type: str
  stick_name: str
  placement: PolygonPlacement
  partner_placement: NotRequired[PolygonPlacement]


CandidateT = TypeVar('CandidateT', Candidate, RiverCandidate)


@dataclasses.dataclass
class StackFrame:
  """Holds the state of a single decision level in the backtracking stack.

  Attributes:
    step_index: The index in the traversal plan this frame corresponds to.
    iterator: An iterator holding the remaining, untried candidate placements
      for this step.
    applied_candidate: The candidate that was chosen from the iterator and
      applied to the grid.
    object_ids: The unique string IDs assigned to the placed objects, used for
      later removal during backtracking.
  """

  step_index: int
  iterator: Iterator[Candidate]
  applied_candidate: Candidate
  object_ids: list[str]


@dataclasses.dataclass
class RiverStackFrame:
  """Holds the state of a single decision level in river backtracking stack.

  Attributes:
    step_index: The index in the traversal plan this frame corresponds to.
    iterator: An iterator holding the remaining, untried candidate placements
      for this step.
    applied_candidate: The candidate that was chosen from the iterator and
      applied to the grid.
    object_ids: The unique string IDs assigned to the placed objects, used for
      later removal during backtracking.
  """

  step_index: int
  iterator: Iterator[RiverCandidate]
  applied_candidate: RiverCandidate
  object_ids: list[str]


class SortedFlap(TypedDict):
  """Represents a flap with its stick name and side length."""

  stick_name: str
  side: int | float


class EdgeConstraint(TypedDict):
  """Describes a distance constraint between two edges.

  Attributes:
    type: The type of constraint.
    value: The value of the constraint.
  """

  type: ConstraintType
  value: float | str


Orientation: TypeAlias = Literal['vertical', 'horizontal', 'corner']
SidePosition: TypeAlias = Literal['top', 'bottom', 'left', 'right']
CornerPosition: TypeAlias = Literal[
    'top_left', 'top_right', 'bottom_left', 'bottom_right'
]
MainFlapPosition: TypeAlias = SidePosition | CornerPosition


def _compute_crimp_direction(
    sf_placement: PolygonPlacement,
    main_placement: PolygonPlacement,
    is_corner: bool,
) -> MainFlapPosition:
  """Computes level shifter crimp direction from placement positions."""
  sacrificial_x, sacrificial_y = sf_placement.offset
  sacrificial_width, sacrificial_height = sf_placement.size
  main_x, main_y = main_placement.offset
  main_width, main_height = main_placement.size

  if is_corner:
    sac_right = sacrificial_x + sacrificial_width
    sac_bottom = sacrificial_y + sacrificial_height
    main_right = main_x + main_width
    main_bottom = main_y + main_height
    if sac_right == main_x and sac_bottom == main_y:
      return 'bottom_right'
    elif sacrificial_x == main_right and sac_bottom == main_y:
      return 'bottom_left'
    elif sac_right == main_x and sacrificial_y == main_bottom:
      return 'top_right'
    elif sacrificial_x == main_right and sacrificial_y == main_bottom:
      return 'top_left'
    raise ValueError(
        f'Cannot determine corner direction for sf at {sf_placement.offset} '
        f'and main at {main_placement.offset}'
    )

  if sacrificial_x + sacrificial_width == main_x:
    return 'right'
  elif main_x + main_width == sacrificial_x:
    return 'left'
  elif sacrificial_y + sacrificial_height == main_y:
    return 'bottom'
  elif main_y + main_height == sacrificial_y:
    return 'top'
  raise ValueError(
      f'Cannot determine crimp direction for sf at {sf_placement.offset} '
      f'and main at {main_placement.offset}'
  )


@dataclasses.dataclass(frozen=True)
class SacrificialFlap:
  """Information about a sacrificial flap.

  A sacrificial flap is a strip of paper generated during the packing process to
  facilitate specific geometric arrangements, such as level shifters, without
  being part of the primary stick figure structure.

  Attributes:
    polygon_name: The name of the polygon representing the flap.
    placement: PolygonPlacement for this sacrificial flap.
    shifted_element_name: The name of the base flap or river associated with the
      flap.
    orientation: The orientation of the flap.
    shifted_element: The placement of the main flap (PolygonPlacement) or the
      river (PackingRiver) that this SF is associated with.
    crimp_direction: The direction of the crimp hinge relative to the SF. For
      side SFs this is a SidePosition, for corner SFs a CornerPosition.
  """

  polygon_name: str
  placement: PolygonPlacement
  shifted_element_name: str
  orientation: Orientation
  shifted_element: PolygonPlacement | river_lib.PackingRiver
  crimp_direction: MainFlapPosition

  @property
  def is_corner(self) -> bool:
    """Whether this sacrificial flap is a corner flap."""
    return self.orientation == 'corner'

  @property
  def is_l_shaped(self) -> bool:
    """Whether this sacrificial flap is one of an L-shaped pair."""
    return not self.polygon_name.removesuffix(COPY_SUFFIX).endswith(
        (RIVER_SF1_SUFFIX, RIVER_SF2_SUFFIX)
    )

  def __post_init__(self):
    if self.is_corner:
      if self.placement.size[0] != self.placement.size[1]:
        raise ValueError(
            f'Corner flap must be square, got size: {self.placement.size}'
        )
      if self.crimp_direction not in (
          'top_left',
          'top_right',
          'bottom_left',
          'bottom_right',
      ):
        raise ValueError(
            f'Inconsistent orientation: {self.orientation} for '
            f'crimp_direction: {self.crimp_direction}'
        )
      return
    if self.crimp_direction in ('left', 'right'):
      if self.orientation != 'vertical':
        raise ValueError(
            f'Inconsistent orientation: {self.orientation} for '
            f'crimp_direction: {self.crimp_direction}'
        )
    if self.crimp_direction in ('top', 'bottom'):
      if self.orientation != 'horizontal':
        raise ValueError(
            f'Inconsistent orientation: {self.orientation} for '
            f'crimp_direction: {self.crimp_direction}'
        )

  @classmethod
  def from_flap_placement(
      cls,
      *,
      polygon_name: str,
      placement: PolygonPlacement,
      shifted_element_name: str,
      orientation: Orientation,
      main_placement: PolygonPlacement,
  ) -> 'SacrificialFlap':
    """Creates a SacrificialFlap, computing crimp_direction from geometry."""
    crimp_direction = _compute_crimp_direction(
        placement, main_placement, is_corner=(orientation == 'corner')
    )
    return cls(
        polygon_name=polygon_name,
        placement=placement,
        shifted_element_name=shifted_element_name,
        orientation=orientation,
        shifted_element=main_placement,
        crimp_direction=crimp_direction,
    )

  @classmethod
  def from_dict(cls, data: dict[str, Any]) -> 'SacrificialFlap':
    """Creates a SacrificialFlap object from a dictionary."""
    shifted_raw = data.get('shifted_element', data.get('main_placement'))
    if shifted_raw is None:
      raise ValueError('Missing shifted_element or main_placement in dict.')

    shifted_element = None
    try:
      shifted_element = PolygonPlacement.from_dict(shifted_raw)
    except (ValueError, KeyError):
      pass
    if shifted_element is None:
      factory = river_lib.get_reconstructed_river_factory()
      if factory is not None and hasattr(factory, 'from_dict'):
        try:
          shifted_element = factory.from_dict(shifted_raw)
        except (ValueError, KeyError):
          pass
    if shifted_element is None:
      raise ValueError(f"Couldn't parse shifted_element: {shifted_raw}")

    crimp_raw = data.get('crimp_direction')
    if crimp_raw is not None:
      crimp_direction = crimp_raw
    else:
      crimp_direction = _compute_crimp_direction(
          PolygonPlacement.from_dict(data['placement']),
          shifted_element,  # pyrefly: ignore[bad-argument-type]
          is_corner=(data['orientation'] == 'corner'),
      )
    return cls(
        polygon_name=data['polygon_name'],
        placement=PolygonPlacement.from_dict(data['placement']),
        shifted_element_name=data.get(  # pyrefly: ignore[bad-argument-type]
            'shifted_element_name', data.get('base_flap_name')
        ),
        orientation=data['orientation'],
        shifted_element=shifted_element,
        crimp_direction=crimp_direction,
    )


class AngleInfo(TypedDict):
  """Describes the angle between two neighboring vertices."""

  neighbor_vertices: list[int]
  angle_degrees: float


class FoldGraphDataDict(TypedDict):
  """Represents the graph-specific structure of a FOLD dictionary."""

  vertices_coords: list[list[float]]
  edges_vertices: list[list[int]]
  file_creator: NotRequired[str]
  file_title: NotRequired[str]
  file_description: NotRequired[str]
  edge_lengths: NotRequired[list[float | None]]
  vertices_angles: NotRequired[dict[str, list[AngleInfo]]]


@dataclasses.dataclass(frozen=True)
class TemplateRiverOptions:
  """Explicit river-solver options for pre-placed shaping-flap templates.

  Attributes:
    preplaced_joint_to_side: Joint-to-side assignments for rivers already placed
      on `initial_tracker` before iterative river solving begins.
    full_tree: Full symmetric stick figure used to mirror enclosed pocket
      placements across the symmetry axis when `mirror_pocket_flaps` is True.
    mirror_pocket_flaps: Whether `pack_intermediate_pocket` mirrors newly packed
      asymmetric pocket flaps across the symmetry axis.
    infer_zero_area_hugging_sides: Whether `_assign_river_sides` infers hugging
      river side assignments from adjacent pre-placed rivers when all remaining
      branch area requirements are zero.
    retry_opposite_hug_side: Whether `generate_hugging_river_candidates` retries
      with the opposite hug side when the primary hug side yields no candidates.
    retry_without_pocket_start_cell: Whether `find_river_candidates` retries
      without `pocket_start_cell` when constrained hugging yields no candidates.
    reorder_pocket_flaps: Whether `pack_intermediate_pocket` reverses a pocket's
      flap traversal order when its last flap has an `EQUAL` constraint to a
      pre-placed boundary river and its first flap does not.
  """

  preplaced_joint_to_side: Mapping[str, Mapping[str, str]] = dataclasses.field(
      default_factory=dict
  )
  full_tree: Any = None
  mirror_pocket_flaps: bool = False
  infer_zero_area_hugging_sides: bool = False
  retry_opposite_hug_side: bool = False
  retry_without_pocket_start_cell: bool = False
  reorder_pocket_flaps: bool = False
