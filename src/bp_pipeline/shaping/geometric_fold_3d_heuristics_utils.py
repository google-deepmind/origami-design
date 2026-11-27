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

"""Utility functions for geometric simple fold 3D heuristics."""

from __future__ import annotations

import itertools
import logging
import math
import typing

import numpy as np

from ..crease_pattern import crease as crease_utils
from ..crease_pattern import crease_pattern as cp_utils
from ..crease_pattern import point as point_utils
from ..third_party.flat_folder import folded_state
from . import geometric_simple_fold
from . import geometric_utils
from . import shaping_elements
from . import shaping_utils

if typing.TYPE_CHECKING:
  import matplotlib.pyplot as plt  # pylint: disable=g-import-not-at-top,g-bad-import-order

INSCRIBE_FRACTION = 0.1


def point_to_tuple(point: point_utils.Point) -> tuple[float, float]:
  """Converts a Point object to a tuple."""
  return (point.x, point.y)


def tuple_dist(p1: tuple[float, float], p2: tuple[float, float]) -> float:
  """Returns the distance between two points."""
  return point_utils.distance(
      point_utils.Point(x=p1[0], y=p1[1], require_on_paper=False),
      point_utils.Point(x=p2[0], y=p2[1], require_on_paper=False),
  )


def _get_creases_on_point(
    point: point_utils.Point,
    creases: list[crease_utils.Crease],
) -> list[crease_utils.Crease]:
  """Returns the creases containing a given point."""
  return [
      crease
      for crease in creases
      if crease.point_status(point) != crease_utils.PointStatus.EXTERNAL
  ]


def check_point_elevation(
    point: point_utils.Point,
    creases: list[crease_utils.Crease],
) -> int | None:
  """Returns the first valid integer elevation of an incident crease, if any."""
  incident_creases = _get_creases_on_point(point, creases)
  return next(
      (c.elevation for c in incident_creases if c.elevation is not None), None
  )


def get_next_point_edge_index_and_elevation(
    point: point_utils.Point,
    edge_index: int,
    creases: list[crease_utils.Crease],
    boundary: shaping_elements.ShapingBoundary,
    grid_size: int,
) -> tuple[point_utils.Point, int, int | None]:
  """Returns the next point, edge index, and elevation (if any) on `boundary`."""
  next_point, next_edge_index = boundary.walk(point, edge_index, grid_size)
  next_elevation = check_point_elevation(next_point, creases)
  return next_point, next_edge_index, next_elevation


def inscribe_cut_line(
    unfolded_cut_line_start: point_utils.Point,
    element: shaping_elements.ShapingElement,
    inscribe_depth: float = INSCRIBE_FRACTION,
) -> tuple[point_utils.Point, shaping_elements.ShapingElement, bool]:
  """Inscribes the cut line epsilon deep into the given shaping element."""
  outer_bank_index = (
      element.get_outer_bank_index()
      if isinstance(element, shaping_elements.ShapingRiver)
      else None
  )
  shaping_boundary = shaping_elements.get_shaping_boundary(
      element, outer_bank_index
  )
  edge_index = shaping_boundary.get_edge_index(unfolded_cut_line_start)
  if edge_index is None:
    raise ValueError(
        f'Cut line start {unfolded_cut_line_start} does not lie on the shaping'
        f' boundary of element {element.name!r}.'
    )

  match element:
    case shaping_elements.ShapingFlap():
      grid_unit_depth = element.grid_size * element.radius / math.sqrt(2)
      if crease_utils.approx_gte(inscribe_depth, grid_unit_depth):
        logging.warning(
            'Inscribe depth (%s) is greater than the width of the flap (%s). '
            'Avoiding inscribing.',
            inscribe_depth,
            grid_unit_depth,
        )
        return (unfolded_cut_line_start, element, False)
      shrunk_element = element.shrink(inscribe_depth)
    case shaping_elements.ShapingRiver():
      if outer_bank_index is None:
        raise ValueError(
            f'Outer bank index must be set for river {element.name!r}.'
        )
      if crease_utils.approx_gte(inscribe_depth, element.width):
        logging.warning(
            'Inscribe depth (%s) is greater than the width of the river (%s). '
            'Avoiding inscribing.',
            inscribe_depth,
            element.width,
        )
        return unfolded_cut_line_start, element, False
      shrunk_element = element.shrink_along_bank(
          outer_bank_index, inscribe_depth
      )
    case _ as unreachable:
      typing.assert_never(unreachable)

  depth_vector = (
      inscribe_depth * shaping_boundary.depth_unit_vectors[edge_index]
  )
  if shaping_boundary.is_corner_point(unfolded_cut_line_start):
    edge = shaping_boundary.creases[edge_index]
    edge_vector = shaping_boundary.edge_unit_vectors[edge_index]
    is_open_terminal_tip = not shaping_boundary.closed and (
        unfolded_cut_line_start.isclose(edge.p2_init)
    )
    if is_open_terminal_tip:
      # On an open river bank's terminal endpoint, `edge_unit_vectors[-1]`
      # points outward off the segment; reverse to step inward toward `p1_init`.
      edge_vector = -1 * edge_vector

    default_candidate = (depth_vector + 2.0 * edge_vector).point_add(
        unfolded_cut_line_start.relax_paper_requirement()
    )
    inward_steps = 2.0
    if is_open_terminal_tip or (
        unfolded_cut_line_start.on_paper and not default_candidate.on_paper
    ):
      far_end = None
      available_steps = edge.length / edge_vector.norm
      if unfolded_cut_line_start.on_paper:
        clipped_edges = shaping_utils.restrict_creases_to_paper([edge])
        if clipped_edges:
          clipped_edge = clipped_edges[0]
          far_end = (
              clipped_edge.p2_init
              if unfolded_cut_line_start.isclose(clipped_edge.p1_init)
              else clipped_edge.p1_init
          )
          available_steps = (
              point_utils.distance(unfolded_cut_line_start, far_end)
              / edge_vector.norm
          )

      if not crease_utils.approx_gte(available_steps, 2.0):
        # When fewer than 2 on-paper grid steps remain before the paper border
        # or next corner, center the anchor inside the shrunk on-paper span.
        lower = min(inscribe_depth, available_steps)
        upper = (
            available_steps
            if far_end is not None and far_end.is_border
            else max(lower, available_steps - inscribe_depth)
        )
        inward_steps = 0.5 * (lower + upper)
    depth_vector = depth_vector + inward_steps * edge_vector

  new_unfolded_cut_line_start = depth_vector.point_add(unfolded_cut_line_start)
  return new_unfolded_cut_line_start, shrunk_element, True


def _point_2d_to_3d(pt: tuple[float, float]) -> list[float]:
  """Converts a 2D point to a 3D point (z=0)."""
  return [pt[0], pt[1], 0.0]


def visualize_cut_line_on_folded_cp(
    crease_pattern: cp_utils.CreasePattern,
    cut_line_start: tuple[float, float],
    cut_line_end: tuple[float, float],
    highlight_flap: shaping_elements.ShapingElement | None = None,
) -> plt.Figure:
  """Visualizes the cut line super-imposed on the folded crease pattern in 3D."""
  import matplotlib.pyplot as plt  # pylint: disable=g-import-not-at-top
  from mpl_toolkits.mplot3d import art3d  # pylint: disable=g-import-not-at-top

  graph_data = geometric_simple_fold.prepare_graph_data(crease_pattern)
  folded_vertices, _ = folded_state.compute_folded_state(
      graph_data['vertices_coords'],
      graph_data['face_vertices'],
      graph_data['edge_vertices'],
      graph_data['edge_assignments'],
  )

  fig = plt.figure(figsize=(10, 8))
  ax = fig.add_subplot(111, projection='3d')

  for v1_idx, v2_idx in graph_data['edge_vertices']:
    if v1_idx >= len(folded_vertices) or v2_idx >= len(folded_vertices):
      continue

    p1 = folded_vertices[v1_idx]
    p2 = folded_vertices[v2_idx]

    if p1 is None or p2 is None:
      continue

    z1 = p1[2] if len(p1) > 2 else 0
    z2 = p2[2] if len(p2) > 2 else 0

    ax.plot(
        [p1[0], p2[0]],
        [p1[1], p2[1]],
        [z1, z2],
        color='black',
        alpha=0.5,
        linewidth=0.5,
    )

  if highlight_flap:
    if isinstance(highlight_flap, shaping_elements.ShapingFlap):
      tips = [*highlight_flap.corners, *highlight_flap.circle_centers]
    elif isinstance(highlight_flap, shaping_elements.ShapingRiver):
      tips = []
      for c in itertools.chain(highlight_flap.bank_1, highlight_flap.bank_2):
        tips.append(c.p1_init)
        tips.append(c.p2_init)
    else:
      raise ValueError(f'Unsupported element type: {type(highlight_flap)}')

    folded_tips = []
    for tip in tips:
      try:
        folded_tip = geometric_utils.get_folded_pos(
            point=tip,
            unfolded_coords=graph_data['vertices_coords'],
            folded_coords=folded_vertices,  # pyrefly: ignore[bad-argument-type]
            edge_vertices=graph_data['edge_vertices'],
        )
        if folded_tip is not None:
          folded_tips.append(folded_tip)
      except ValueError:
        continue

    ax.plot(
        [p[0] for p in folded_tips],
        [p[1] for p in folded_tips],
        [0 for _ in folded_tips],
        color='blue',
        marker='o',
        linestyle='None',
        markersize=5,
        zorder=20,
    )

    faces_to_highlight = []
    unfolded_coords = graph_data['vertices_coords']
    for v_indices in graph_data['face_vertices']:
      face_unfolded_pts = [unfolded_coords[i] for i in v_indices]

      is_in_flap = True
      for p in face_unfolded_pts:
        pt_obj = point_utils.Point(x=p[0], y=p[1])
        if not highlight_flap.is_internal_point(pt_obj):
          is_in_flap = False
          break

      if is_in_flap:
        face_folded_pts = []
        for i in v_indices:
          p = folded_vertices[i]
          if p is not None:
            z = p[2] if len(p) > 2 else 0
            face_folded_pts.append([p[0], p[1], z])

        if face_folded_pts:
          faces_to_highlight.append(face_folded_pts)

    if faces_to_highlight:
      poly3d = art3d.Poly3DCollection(
          faces_to_highlight, alpha=0.5, facecolors='green', edgecolors='none'
      )
      ax.add_collection3d(poly3d)

  p_start = _point_2d_to_3d(cut_line_start)
  p_end = _point_2d_to_3d(cut_line_end)

  ax.plot(
      [p_start[0], p_end[0]],
      [p_start[1], p_end[1]],
      [p_start[2], p_end[2]],
      color='red',
      linewidth=3,
      label='Cut Line',
      zorder=10,
  )

  ax.scatter(*p_start, color='red', marker='o', s=50)
  ax.scatter(*p_end, color='red', marker='x', s=50)

  ax.set_title('Folded CP with Cut Line')
  ax.legend()
  ax.set_axis_off()
  ax.view_init(elev=90, azim=-90)

  all_points = [p for p in folded_vertices if p is not None]
  if all_points:
    all_points_np = np.array(all_points)
    min_coords = np.min(all_points_np, axis=0)
    max_coords = np.max(all_points_np, axis=0)

    padding = 0.05
    x_range = max(max_coords[0] - min_coords[0], 1e-5)
    y_range = max(max_coords[1] - min_coords[1], 1e-5)

    ax.set_xlim(
        min_coords[0] - x_range * padding, max_coords[0] + x_range * padding
    )
    ax.set_ylim(
        min_coords[1] - y_range * padding, max_coords[1] + y_range * padding
    )
    ax.set_box_aspect((x_range, y_range, 1))

  ax.invert_yaxis()

  return fig
