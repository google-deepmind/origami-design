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

"""CreasePatternTracker for 2D grid state during packing."""

import collections
import logging
import re
import typing

import numpy as np
import pydantic

from .. import rectangle as shaping_utils
from ..crease_pattern import crease as crease_lib
from ..stick_figures import stick_figure as stick_figure_lib
from . import grid
from . import river as river_lib
from . import types
from . import utils_packing_flaps


class TrackerState(pydantic.BaseModel):
  """Serializable state of a CreasePatternTracker."""

  model_config = pydantic.ConfigDict(frozen=True, arbitrary_types_allowed=True)

  grid_size: int
  packed_polygons: dict[str, types.PolygonPlacement]
  packed_rivers: dict[str, river_lib.PackingRiver]
  river_banks: dict[str, list[tuple[int, int]]]
  occupancy_mask: bytes
  river_bank_ids: dict[str, list[tuple[int, int]]] = pydantic.Field(
      default_factory=dict
  )

  def to_json(self) -> str:
    return self.model_dump_json()

  @classmethod
  def from_json(cls, json_str: str) -> 'TrackerState':
    return cls.model_validate_json(json_str)


class CreasePatternTracker:
  """Manages the state of a 2D grid for crease pattern packing.

  This class encapsulates the grid data and all operations that can be
  performed on it, such as adding or removing polygons. It prevents
  polygons from overlapping and allows them to be managed by a unique ID.
  """

  def __init__(self, grid_size: int, occupancy_mask: np.ndarray | None = None):
    """Initializes a new, empty tracker grid of a given size.

    Args:
        grid_size: The width and height of the square grid.
        occupancy_mask: If provided, initializes grid with blocked cells.
    """
    if grid_size <= 0:
      raise ValueError('Grid size must be positive.')
    self.grid_size = grid_size
    self.grid: np.ndarray = np.full(
        (grid_size, grid_size),
        fill_value=utils_packing_flaps.EmptyGridCellState,
        dtype=object,
    )
    if occupancy_mask is not None:
      if occupancy_mask.shape != (grid_size, grid_size):
        raise ValueError(
            f'Occupancy mask shape {occupancy_mask.shape} does not match '
            f'grid size ({grid_size}, {grid_size}).'
        )
      self._occupancy_mask = occupancy_mask.copy()
    else:
      self._occupancy_mask: np.ndarray = np.full(
          (grid_size, grid_size), fill_value=False, dtype=bool
      )

    blocked_state = utils_packing_flaps.BlockedGridCellState
    self.grid[self._occupancy_mask] = blocked_state
    self.occupancy_grid = self._occupancy_mask.copy()

    self.packed_polygons: dict[str, types.PolygonPlacement] = {}
    self.packed_rivers: dict[str, river_lib.PackingRiver] = {}
    self.river_banks: dict[str, list[tuple[int, int]]] = {}

  def get_element(
      self, element_id: str
  ) -> types.PolygonPlacement | river_lib.PackingRiver | None:
    """Returns the element with the given ID."""
    return self.packed_polygons.get(
        element_id, self.packed_rivers.get(element_id)
    )

  @property
  def id_grid(self) -> np.ndarray:
    """Returns an array of object IDs."""
    return np.array([s.source_id for s in self.grid.ravel()]).reshape(
        self.grid.shape
    )

  def expand_polygon(
      self, leaf_id: str, dir_or_axis: grid.Direction | grid.Axis, amount: int
  ):
    """Expands an existing polygon in the given direction by the given amount.

    If dir_or_axis is a grid.Direction, the polygon is expanded in that
    direction by the given amount. If it is a grid.Axis, the polygon is expanded
    in both direction of that axis by the given amount.

    Args:
        leaf_id: The ID of the polygon to expand.
        dir_or_axis: The direction (`grid.Direction`) or axis (`grid.Axis`)
          along which to expand.
        amount: The amount by which to expand the polygon. Must be positive.

    Raises:
        ValueError: If the amount is not positive, if the polygon is not
          found in the tracker, or if an axis expansion is attempted on a
          non-border, non-symmetric flap.
    """
    if amount <= 0:
      raise ValueError(f'Amount must be positive: {amount=}')
    if leaf_id not in self.packed_polygons:
      raise ValueError(f'Polygon not found in tracker: {leaf_id=}')

    og_placement = self.packed_polygons[leaf_id]
    offset_x, offset_y = og_placement.offset
    width, height = og_placement.size

    if amount % 2 == 1:
      if not isinstance(dir_or_axis, grid.Axis):
        raise ValueError(
            'Odd deltas are only allowed for axis expansion:'
            f' {amount=} {og_placement}'
        )

    # These match statements must be nested because apparently
    # grid.Axis.HORIZONTAL matches grid.Direction.EAST, etc.
    match dir_or_axis:
      case grid.Direction():
        match dir_or_axis:
          case grid.Direction.NORTH:
            offset_y -= amount
            height += amount
          case grid.Direction.SOUTH:
            height += amount
          case grid.Direction.WEST:
            offset_x -= amount
            width += amount
          case grid.Direction.EAST:
            width += amount
          case _:
            typing.assert_never(dir_or_axis)
      case grid.Axis():
        match dir_or_axis:
          case grid.Axis.VERTICAL:
            offset_y -= amount
            height += 2 * amount
          case grid.Axis.HORIZONTAL:
            offset_x -= amount
            width += 2 * amount
          case _:
            typing.assert_never(dir_or_axis)
      case _:
        typing.assert_never(dir_or_axis)

    new_placement = types.PolygonPlacement.from_grid_coords(
        offset=(offset_x, offset_y),
        size=(width, height),
        grid_size=self.grid_size,
        river_sf_info=og_placement.river_sf_info,
    )
    logging.debug(
        'Removing polygon %s and adding new polygon %s',
        og_placement,
        new_placement,
    )
    self.remove_by_id(leaf_id)
    self.add_polygon(leaf_id, new_placement)

  def add_polygon(self, leaf_id: str, placement: types.PolygonPlacement):
    """Marks cells occupied by a packed polygon, raising an error on overlap.

    Args:
        leaf_id: A unique identifier for the polygon.
        placement: The geometric description of the polygon to add.

    Raises:
        ValueError: If the `leaf_id` already exists or if the new polygon
          would overlap with any existing polygons on the grid.
    """
    if leaf_id in self.packed_polygons or leaf_id in self.packed_rivers:
      raise ValueError(f"Object with ID '{leaf_id}' already exists.")
    state_to_add = utils_packing_flaps.GridCellState(
        type=utils_packing_flaps.GridCellType.POLYGON,
        source_id=leaf_id,
    )

    offset_x, offset_y = placement.offset
    width, height = placement.size

    # 1. Check for overlaps using the utility function on the boolean grid.
    rect = shaping_utils.Rectangle(
        name=leaf_id,
        x_start=offset_x,
        y_start=offset_y,
        x_len=width,
        y_len=height,
    )
    if not utils_packing_flaps.is_unoccupied(
        grid=self.occupancy_grid,
        rect=rect,
    ):
      raise ValueError(
          f"Cannot add polygon '{leaf_id}': Overlap detected "
          f'at offset ({offset_x}, {offset_y}).'
      )

    # 2. If checks pass, update both grids.
    utils_packing_flaps.set_grid_occupation(
        grid=self.occupancy_grid,
        rect=rect,
        occupied=True,
        metadata_grid=self.grid,
        state=state_to_add,
    )

    # 3. Track the successfully added polygon.
    self.packed_polygons[leaf_id] = placement

  def add_river(
      self,
      river_id: str,
      river: river_lib.PackingRiver,
  ):
    """Marks cells occupied by a river, raising an error on overlap.

    Args:
        river_id: A unique identifier for the river.
        river: The river to add.

    Raises:
        ValueError: If the `river_id` already exists or if the new river
          would overlap with any existing polygons or rivers on the grid.
    """
    if river.grid_size != self.grid_size:
      raise ValueError(
          f'River grid size {river.grid_size} does not match tracker grid size'
          f' {self.grid_size}.'
      )
    if river_id in self.packed_rivers:
      raise ValueError(f"Object with ID '{river_id}' already exists.")
    state_to_add = utils_packing_flaps.GridCellState(
        type=utils_packing_flaps.GridCellType.RIVER,
        source_id=river_id,
    )

    if isinstance(river, river_lib.River):
      rects_to_check = [river.rectangle]
    elif hasattr(river, 'rectangles'):
      rects_to_check = river.rectangles
    else:
      raise TypeError(f'Unknown river type: {type(river)}')

    for rect in rects_to_check:
      if not utils_packing_flaps.is_unoccupied(
          grid=self.occupancy_grid,
          rect=rect,
      ):
        raise ValueError(f"Cannot add river '{river_id}': Overlap detected.")

    for rect in rects_to_check:
      utils_packing_flaps.set_grid_occupation(
          grid=self.occupancy_grid,
          rect=rect,
          occupied=True,
          metadata_grid=self.grid,
          state=state_to_add,
      )
    self.packed_rivers[river_id] = river

  def _remove_polygon(self, placement: types.PolygonPlacement):
    """Resets polygon cells on the grid to None. For internal use."""
    offset_x, offset_y = placement.offset
    width, height = placement.size
    rect = shaping_utils.Rectangle(
        name='',
        x_start=offset_x,
        y_start=offset_y,
        x_len=width,
        y_len=height,
    )

    # Update the boolean occupancy grid.
    utils_packing_flaps.set_grid_occupation(
        grid=self.occupancy_grid,
        rect=rect,
        occupied=False,
        metadata_grid=self.grid,
        state=utils_packing_flaps.EmptyGridCellState,
    )

  def _remove_river(
      self,
      river: river_lib.PackingRiver,
  ):
    """Resets river cells on the grid to None."""
    if isinstance(river, river_lib.River):
      rects_to_remove = [river.rectangle]
    elif hasattr(river, 'rectangles'):
      rects_to_remove = river.rectangles
    else:
      raise TypeError(f'Unknown river type: {type(river)}')

    for rect in rects_to_remove:
      utils_packing_flaps.set_grid_occupation(
          grid=self.occupancy_grid,
          rect=rect,
          occupied=False,
          metadata_grid=self.grid,
          state=utils_packing_flaps.EmptyGridCellState,
      )

  def remove_by_id(self, object_id: str):
    """Finds and removes a polygon or river by its unique ID.

    Args:
        object_id: The unique identifier of the object to remove.

    Raises:
        KeyError: If the provided `object_id` is not found in the tracker.
    """
    if object_id in self.packed_polygons:
      placement = self.packed_polygons[object_id]
      self._remove_polygon(placement)
      del self.packed_polygons[object_id]
    elif object_id in self.packed_rivers:
      bank_coords = self.river_banks.pop(object_id, [])
      for y, x in bank_coords:
        if (
            self.grid[y, x]
            and self.grid[y, x].type
            == utils_packing_flaps.GridCellType.RIVER_ADJACENT
        ):
          self.grid[y, x] = utils_packing_flaps.EmptyGridCellState
      river = self.packed_rivers[object_id]
      self._remove_river(river)
      del self.packed_rivers[object_id]
    else:
      raise KeyError(f"Cannot remove object: ID '{object_id}' not found.")

  def get_state(self) -> TrackerState:
    """Serializes the tracker state into a `TrackerState` object.

    This method captures the current state of the tracker, including grid size,
    packed polygons and rivers, river banks, and occupancy mask, into a
    `TrackerState` object for serialization or checkpointing.

    Returns:
      A `TrackerState` object representing the current state of the tracker.
    """
    river_bank_ids = collections.defaultdict(list)
    for y in range(self.grid_size):
      for x in range(self.grid_size):
        cell = self.grid[y, x]
        if (
            isinstance(cell, utils_packing_flaps.GridCellState)
            and cell.type == utils_packing_flaps.GridCellType.RIVER_ADJACENT
        ):
          river_bank_ids[cell.source_id].append((y, x))
    return TrackerState(
        grid_size=self.grid_size,
        packed_polygons=self.packed_polygons,
        packed_rivers=self.packed_rivers,
        river_banks=self.river_banks,
        river_bank_ids=river_bank_ids,
        occupancy_mask=np.packbits(self._occupancy_mask.flatten()).tobytes(),
    )

  @classmethod
  def from_state(cls, state: TrackerState) -> 'CreasePatternTracker':
    """Deserializes a `TrackerState` object into a `CreasePatternTracker`.

    This method reconstructs a `CreasePatternTracker` instance from a
    `TrackerState` object, restoring all polygons, rivers, and grid states.

    Args:
      state: The `TrackerState` object to deserialize.

    Returns:
      A `CreasePatternTracker` instance representing the deserialized state.
    """
    grid_size = state.grid_size
    barr = state.occupancy_mask
    arr_uint8 = np.frombuffer(barr, dtype=np.uint8)
    occupancy_mask = (
        np.unpackbits(arr_uint8)[: grid_size * grid_size]
        .reshape((grid_size, grid_size))
        .astype(bool)
    )
    tracker = cls(grid_size, occupancy_mask=occupancy_mask)
    for river_id, river in state.packed_rivers.items():
      tracker.add_river(river_id, river)  # pyrefly: ignore[bad-argument-type]
    for poly_id, placement in state.packed_polygons.items():
      tracker.add_polygon(poly_id, placement)
    tracker.river_banks = state.river_banks
    for source_id, cells in state.river_bank_ids.items():
      for y, x in cells:
        tracker.grid[y, x] = utils_packing_flaps.GridCellState(
            type=utils_packing_flaps.GridCellType.RIVER_ADJACENT,
            source_id=source_id,
        )
    return tracker


def center_from_placement(
    placement: types.PolygonPlacement,
) -> tuple[float, float]:
  """Returns center of placement in grid coordinates.

  Args:
      placement: The PolygonPlacement object.

  Returns:
      A tuple of (x, y) coordinates of the center of the placement in grid
      coordinates.
  """
  return (
      placement.offset[0] + placement.size[0] / 2.0,
      placement.offset[1] + placement.size[1] / 2.0,
  )


def plot_tracker(
    tracker: CreasePatternTracker | None,
    title: str = 'Solver State',
    ridges: (
        dict[str, list[tuple[tuple[float, float], tuple[float, float]]]] | None
    ) = None,
    creases: list[crease_lib.Crease] | None = None,
    highlight_candidate: types.Candidate | types.RiverCandidate | None = None,
    is_symmetric_packing: bool = False,
    stick_figure: stick_figure_lib.StickFigure | None = None,
):
  """Draws the state of a CreasePatternTracker, optionally with ridges.

  This function visualizes the packed polygons from a tracker object onto a
  matplotlib grid. Each rectangle is labeled with its placement
  order. If a dictionary of ridge creases is provided, they will be
  overlaid on their corresponding polygons.

  Args:
      tracker: The CreasePatternTracker object containing the solved placement.
        Can be None, in which case the function prints a message and exits.
      title: The title to display on the plot.
      ridges: An optional dictionary mapping polygon IDs to a list of line
        segments representing their ridge creases.
      creases: An optional list of creases to draw on the plot.
      highlight_candidate: An optional candidate placement to highlight.
      is_symmetric_packing: If True, draw symmetry axis.
      stick_figure: If provided, used to label flap-adjacent cells with joint
        names.
  """
  if not tracker:
    print(f'Tracker is None for plot with title "{title}".')
    return

  from matplotlib import patches  # pylint: disable=g-import-not-at-top
  import matplotlib.pyplot as plt  # pylint: disable=g-import-not-at-top

  river_start_marker_kwargs = dict(
      marker='*',
      color='limegreen',
      markersize=10,
      zorder=1.8,
  )
  grid_size = tracker.grid_size
  _, ax = plt.subplots(figsize=(8, 8))
  ax.set_title(title, fontsize=16)
  ax.set_xticks(range(grid_size + 1))
  ax.set_yticks(range(grid_size + 1))
  ax.grid(True, which='both', linestyle='--', linewidth=0.5)
  ax.set_xlim(0, grid_size)
  ax.set_ylim(0, grid_size)
  ax.set_aspect('equal', adjustable='box')
  ax.invert_yaxis()

  if is_symmetric_packing:
    ax.axvline(grid_size / 2.0, color='r', linestyle='--', linewidth=1.0)

  flap_adjacent_cells = {}
  for leaf_id, placement in tracker.packed_polygons.items():
    offset_x, offset_y = placement.offset
    width, height = placement.size
    for row in range(offset_y, offset_y + height):
      for col in range(offset_x, offset_x + width):
        for delta_row, delta_col in river_lib.DIRS:
          neighbor_row, neighbor_col = row + delta_row, col + delta_col
          if (
              0 <= neighbor_row < grid_size
              and 0 <= neighbor_col < grid_size
              and not tracker.occupancy_grid[neighbor_row, neighbor_col]
          ):
            coord = (neighbor_row, neighbor_col)
            if coord not in flap_adjacent_cells:
              flap_adjacent_cells[coord] = {
                  'type': 'FLAP_ADJACENT',
                  'source_ids': set(),
              }
            flap_adjacent_cells[coord]['source_ids'].add(leaf_id)

  flap_to_joint = {}
  if stick_figure:
    for stick in stick_figure.children:
      if not stick.is_river:
        j1, j2 = stick.start, stick.end
        d1 = len(stick_figure.joints[j1].sticks)
        d2 = len(stick_figure.joints[j2].sticks)
        if d1 > d2:
          flap_to_joint[stick.name] = j1
        elif d2 > d1:
          flap_to_joint[stick.name] = j2
        else:
          flap_to_joint[stick.name] = j1

  # Draw blocked cells
  for row in range(grid_size):
    for col in range(grid_size):
      match tracker.grid[row, col]:
        case utils_packing_flaps.BlockedGridCellState:
          rect = patches.Rectangle(
              xy=(col, row),
              width=1,
              height=1,
              facecolor='gray',
              edgecolor='gray',
              linewidth=0.5,
              alpha=0.5,
              zorder=0,
          )
          ax.add_patch(rect)
        case utils_packing_flaps.GridCellState(
            type=utils_packing_flaps.GridCellType.RIVER_ADJACENT,
            source_id=source_id,
        ):
          rect = patches.Rectangle(
              xy=(col, row),
              width=1,
              height=1,
              facecolor='orange',
              edgecolor='orange',
              linewidth=0.5,
              alpha=0.5,
              zorder=0,
          )
          ax.add_patch(rect)
          ax.text(
              col + 0.5,
              row + 0.5,
              source_id,
              color='black',
              ha='center',
              va='center',
              weight='bold',
              fontsize=6,
              zorder=1.5,
          )
        case _ if (row, col) in flap_adjacent_cells:
          rect = patches.Rectangle(
              xy=(col, row),
              width=1,
              height=1,
              facecolor='yellow',
              edgecolor='yellow',
              linewidth=0.5,
              alpha=0.5,
              zorder=0,
          )
          ax.add_patch(rect)
          flap_id = next(iter(flap_adjacent_cells[(row, col)]['source_ids']))
          base_poly_id = re.sub(r'_copy$', '', flap_id)
          joint = flap_to_joint.get(base_poly_id, '?')
          ax.text(
              col + 0.5,
              row + 0.5,
              joint,
              color='black',
              ha='center',
              va='center',
              weight='bold',
              fontsize=6,
              zorder=1.5,
          )

  # Draw rivers
  for river_id, river in tracker.packed_rivers.items():
    if isinstance(river, river_lib.River):
      rects_to_draw = [river.rectangle]
      if river.orientation == river_lib.RiverOrientation.HORIZONTAL:
        for i in range(river.width):
          y, x = river.offset + i, 0
          ax.plot(
              x + 0.5,
              y + 0.5,
              **river_start_marker_kwargs,
          )
      else:  # VERTICAL
        for i in range(river.width):
          y, x = tracker.grid_size - 1, river.offset + i
          ax.plot(
              x + 0.5,
              y + 0.5,
              **river_start_marker_kwargs,
          )
    elif hasattr(river, 'rectangles'):
      rects_to_draw = river.rectangles
      if isinstance(river, river_lib.LShapedRiver):
        if river.l_shape_type in [
            river_lib.LShapeType.BOTTOM_LEFT,
            river_lib.LShapeType.TOP_LEFT,
        ]:
          for i in range(river.width):
            ax.plot(
                0.5,
                river.turn_y + i + 0.5,
                **river_start_marker_kwargs,
            )
        elif river.l_shape_type == river_lib.LShapeType.BOTTOM_RIGHT:
          for i in range(river.width):
            ax.plot(
                grid_size - 0.5,
                river.turn_y + i + 0.5,
                **river_start_marker_kwargs,
            )
        elif river.l_shape_type == river_lib.LShapeType.TOP_RIGHT:
          for i in range(river.width):
            ax.plot(
                river.turn_x + i + 0.5,
                grid_size - 0.5,
                **river_start_marker_kwargs,
            )
    else:
      raise TypeError(f'Unknown river type: {type(river)}')

    for i, rect_obj in enumerate(rects_to_draw):
      offset_x, offset_y, width, height = (
          rect_obj.x_start,
          rect_obj.y_start,
          rect_obj.x_len,
          rect_obj.y_len,
      )
      rect = patches.Rectangle(
          xy=(offset_x, offset_y),
          width=width,
          height=height,
          facecolor='cyan',
          edgecolor='blue',
          linewidth=1.5,
          alpha=0.5,
          zorder=1,
      )
      ax.add_patch(rect)
      if i == 0:
        label = f'River {river_id}'
        # For L-shaped rivers, label will be on first rectangle.
        center_x, center_y = offset_x + width / 2, offset_y + height / 2
        ax.text(
            center_x,
            center_y,
            label,
            color='blue',
            ha='center',
            va='center',
            weight='bold',
            fontsize=8,
            zorder=1.5,
        )

  # Draw creases
  if creases:
    for crease in creases:
      ax.plot(
          [crease.p1.x * grid_size, crease.p2.x * grid_size],
          [
              crease.p1.y * grid_size,
              crease.p2.y * grid_size,
          ],
          color=crease.fold.color(),
          linewidth=1.0,
          zorder=3,
      )

  # Draw the packed polygons
  for poly_id, placement in tracker.packed_polygons.items():
    rect = patches.Rectangle(
        xy=(
            placement.offset[0],
            placement.offset[1],
        ),
        width=placement.size[0],
        height=placement.size[1],
        facecolor='royalblue',
        edgecolor='black',
        linewidth=1.5,
        alpha=0.7,
        zorder=2,
    )
    ax.add_patch(rect)
    size_w, size_h = placement.size
    label = f'{poly_id}\n{size_w}x{size_h}'
    if stick_figure:
      stick_name = re.sub(r'_copy$', '', poly_id)
      if stick_name in stick_figure:
        stick = stick_figure[stick_name]
        label = (
            f'{poly_id}\n{size_w}x{size_h}\nstick:{stick.length}x{stick.width}'
        )
    center_x, center_y = center_from_placement(placement)
    ha = 'center'
    va = 'center'
    if center_x < 1:
      ha = 'left'
    elif center_x > grid_size - 1:
      ha = 'right'
    if center_y < 1:
      va = 'top'
    elif center_y > grid_size - 1:
      va = 'bottom'
    ax.text(
        center_x,
        center_y,
        label,
        color='white',
        ha=ha,
        va=va,
        weight='bold',
        fontsize=8,
        zorder=4,
    )

  # Draw the ridge creases if they were provided
  if ridges:
    for poly_id in ridges:
      if poly_id not in tracker.packed_polygons:
        continue
      for p1, p2 in ridges[poly_id]:
        ax.plot(
            [p1[0], p2[0]],
            [p1[1], p2[1]],
            color='red',
            linewidth=1.0,
            zorder=3,
        )

  if highlight_candidate:
    if 'river' in highlight_candidate:
      river: types.RiverPlacementAlias = highlight_candidate['river']  # pyrefly: ignore[bad-assignment]
      rects_to_highlight = river.rectangles
      for rect_obj in rects_to_highlight:
        x, y, w, h = (
            rect_obj.x_start,
            rect_obj.y_start,
            rect_obj.x_len,
            rect_obj.y_len,
        )
        rect = patches.Rectangle(
            xy=(x, y),
            width=w,
            height=h,
            facecolor='none',
            edgecolor='yellow',
            linewidth=2.0,
            linestyle='--',
            zorder=5,
        )
        ax.add_patch(rect)
    elif 'placement' in highlight_candidate:
      candidate: types.Candidate = highlight_candidate
      placements_to_highlight = [candidate['placement']]
      if candidate.get('type') == 'PAIRED_ON_AXIS':
        placements_to_highlight.append(candidate['partner_placement'])
      for placement in placements_to_highlight:
        rect = patches.Rectangle(
            xy=(
                placement.offset[0],
                placement.offset[1],
            ),
            width=placement.size[0],
            height=placement.size[1],
            facecolor='none',
            edgecolor='yellow',
            linewidth=2.0,
            linestyle='--',
            zorder=5,
        )
        ax.add_patch(rect)

  plt.show()
