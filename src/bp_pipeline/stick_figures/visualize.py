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

"""Visualize stick figures."""

from __future__ import annotations

import math
import typing
from typing import Any, Literal, TYPE_CHECKING, TypeAlias

import numpy as np

from . import angle_utils
from . import stick as stick_lib
from . import stick_figure
from . import traversal as tr

if TYPE_CHECKING:
  import matplotlib.pyplot as plt  # pylint: disable=g-import-not-at-top,g-bad-import-order

ViewPoint: TypeAlias = Literal['top', 'side', 'front', 'isometric']
AllViewPoints = typing.get_args(ViewPoint)


def _get_viewpoint_coordinates(
    viewpoint: ViewPoint, round_digits: int | None = None
) -> tuple[float, float, float]:
  """Returns the coordinates of the viewpoint."""
  match viewpoint:
    case 'top':
      view = (0, 0, 1)
    case 'side':
      view = (1, 0, 0)
    case 'front':
      view = (0, 1, 0)
    case 'isometric':
      view = (1, -1, 1)
    case _:
      raise ValueError(f'Unsupported view point: {viewpoint}')
  if round_digits is not None:
    view = tuple(round(v, round_digits) for v in view)
  return view  # pyrefly: ignore[bad-return]


def _projection_at_joint(
    stick: stick_lib.Stick, joint_name: str, viewpoint: ViewPoint
) -> tuple[float, float]:
  """Get the angle and projected length to use for the view point."""
  dx, dy, dz = stick.get_coords(joint_name)

  match viewpoint:
    case 'top':
      # Project onto XY plane
      proj_x, proj_y = dx, dy
    case 'side':
      # Project onto YZ plane
      proj_x, proj_y = dy, dz
    case 'front':
      # Project onto XZ plane
      proj_x, proj_y = dx, dz
    case 'isometric':
      # Project onto isometric plane
      proj_x = (dx - dy) * math.sqrt(3) / 2
      proj_y = dz - (dx + dy) / 2
    case _:
      raise ValueError(f'Unsupported view point: {viewpoint}')

  angle = math.degrees(math.atan2(proj_y, proj_x))
  length = math.hypot(proj_x, proj_y)
  return angle, length


def _get_offset(distance: float, angle_deg: float) -> tuple[float, float]:
  """Calculates x, y offsets from a distance and angle."""
  angle_rad = math.radians(angle_deg)
  offset_x = distance * math.cos(angle_rad)
  offset_y = distance * math.sin(angle_rad)
  return offset_x, offset_y


def _get_angle_offsets(
    tree: stick_figure.StickFigure,
    angle_offset_deg: float,
    viewpoint: ViewPoint,
    angle_tol: float = 0.1,
):
  """Calculate angle offsets for any sticks that overlap at a joint."""
  visual_angle_offsets = {}
  for joint in tree.joints.values():
    offset_count = 0
    projected_sticks = [
        (_projection_at_joint(stick, joint.name, viewpoint)[0], stick)
        for stick in joint.sticks
    ]
    projected_sticks.sort(key=lambda x: x[0])
    for i, (current_angle, stick) in enumerate(projected_sticks):
      prev_angle = projected_sticks[i - 1][0]
      same_angle = angle_utils.are_azimuth_angles_the_same(
          current_angle, prev_angle, angle_tol
      )
      # Use the first stick as the reference, so it isn't double counted.
      if i > 0 and same_angle:
        offset_count += 1
      else:
        offset_count = 0

      visual_angle_offsets[stick.name, joint.name] = (
          offset_count * angle_offset_deg
      )

  return visual_angle_offsets


def visualize_stick_from_all_views(
    tree: stick_figure.StickFigure,
    figsize: int = 5,
    **kwargs,
) -> plt.Figure:
  """Visualize a stick figure from all canonical view points."""
  import matplotlib.pyplot as plt  # pylint: disable=g-import-not-at-top
  viewpoints = typing.get_args(ViewPoint)
  figsize = (figsize * len(viewpoints), figsize)  # pyrefly: ignore[bad-assignment]
  fig, axes = plt.subplots(1, len(viewpoints), figsize=figsize)
  for ax, view in zip(axes, viewpoints):
    visualize_stick_figure(tree, ax=ax, viewpoint=view, **kwargs)
  return fig


def visualize_stick_figure(
    tree: stick_figure.StickFigure,
    *,
    show_stick_label: bool = True,
    show_joint_label: bool | None = None,
    label_offset: float = 0.2,
    figsize: tuple[float, float] = (10, 10),
    angle_offset_deg: float = 8.0,
    min_length: float = 0.5,
    padding: float = 0.5,
    ax: plt.Axes | None = None,
    viewpoint: ViewPoint = 'side',
    use_rectangles: bool = False,
) -> tuple[plt.Figure, plt.Axes]:
  """Visualize a stick figure."""
  import matplotlib.patches as mpatches  # pylint: disable=g-import-not-at-top
  import matplotlib.pyplot as plt  # pylint: disable=g-import-not-at-top
  if ax is None:
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor('white')
  else:
    fig = ax.figure

  num_sticks = len(tree.children)
  if num_sticks <= 16:
    stick_fontsize = 11.0
    joint_fontsize = 9.0
  elif num_sticks <= 24:
    stick_fontsize = 8.0
    joint_fontsize = 7.5
  else:
    stick_fontsize = 6.5
    joint_fontsize = 6.5

  if show_joint_label is None:
    labeled_joints = (
        {j.name for j in tree.joints.values() if len(j.sticks) >= 3}
        if num_sticks > 16
        else set(tree.joints)
    )
  elif show_joint_label:
    labeled_joints = set(tree.joints)
  else:
    labeled_joints = set()

  root_stick = tree[tree.root]
  joint_locations = {root_stick.start: (0, 0)}

  visual_angle_offsets = _get_angle_offsets(tree, angle_offset_deg, viewpoint)
  for stick in tr.breadth_first_traversal(tree):
    next_start, next_end = (
        (stick.start, stick.end)
        if stick.start in joint_locations
        else (stick.end, stick.start)
    )
    x0, y0 = joint_locations[next_start]
    angle_deg, length = _projection_at_joint(stick, next_start, viewpoint)
    offset_deg = visual_angle_offsets[stick.name, next_start]
    angle_deg += offset_deg

    stick_length = max(min_length, length)
    if next_end not in joint_locations:
      # If stick is projected into/out of the page, draw a short stub.
      x_offset, y_offset = _get_offset(stick_length, angle_deg)
      joint_locations[next_end] = (x0 + x_offset, y0 + y_offset)  # pyrefly: ignore[unsupported-operation]

    x1, y1 = joint_locations[next_end]

    # Plot the stick line
    color = 'mediumpurple' if stick.shaping_flap else 'k'
    if use_rectangles:
      rect = mpatches.Rectangle(
          (x0, y0 - stick.width / 2.0),
          stick_length,
          stick.width,
          angle=angle_deg,
          rotation_point=(x0, y0),
          facecolor=color,
          edgecolor=color,
          zorder=1,
          alpha=0.7,
      )
      ax.add_patch(rect)
    else:
      ax.plot(
          (x0, x1),
          (y0, y1),
          color=color,
          linestyle='-',
          linewidth=stick.width,
          zorder=1,
      )

    if show_stick_label:
      # Calculate angle for rotating the label text
      label_offset_x, label_offset_y = _get_offset(label_offset, angle_deg + 45)
      label_x = (x0 + x1) / 2 + label_offset_x
      label_y = (y0 + y1) / 2 + label_offset_y

      # Plot the stick label
      label = f'{stick.name}: {stick.length}'
      if offset_deg > 0:
        label += f' (+{round(offset_deg)}°)'

      ax.text(
          label_x,
          label_y,
          label,
          color='green',
          fontsize=stick_fontsize,
          ha='center',
          va='center',
          rotation=angle_utils.wrap_angle_centered(angle_deg, 180),
          bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=0.1),
          zorder=3,
      )

  joint_marker_size = (
      12 if use_rectangles else (3.5 if num_sticks > 24 else 5)
  )
  for joint_name, (x, y) in joint_locations.items():
    ax.plot(x, y, 'bo', markersize=joint_marker_size, zorder=2, alpha=0.7)
    if joint_name in labeled_joints:
      ax.text(
          x + 0.1,
          y + 0.1,
          joint_name,
          color='red',
          fontsize=joint_fontsize,
          zorder=3,
          rotation=10,
      )

  # Add border padding
  if joint_locations:
    all_x, all_y = zip(*joint_locations.values())
    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    ax.set_xlim(min_x - padding, max_x + padding)
    ax.set_ylim(min_y - padding, max_y + padding)

  ax.set_aspect('equal', adjustable='box')
  viewpoint_coordinates = _get_viewpoint_coordinates(viewpoint, round_digits=3)
  ax.set_title(
      f'{tree.name} ({viewpoint} view \n({viewpoint_coordinates}) |'
      f' complexity={tree.complexity})',
      zorder=4,
      bbox=dict(facecolor='white'),
  )
  ax.axis('off')
  return fig, ax


def visualize_stick_figure_3d(
    tree: stick_figure.StickFigure,
    *,
    figsize: tuple[float, float] = (800, 800),
    show_labels: bool = True,
) -> Any:
  """Visualize a stick figure in 3D using Plotly."""
  try:
    import plotly.graph_objects as go  # pylint: disable=g-import-not-at-top
  except ImportError as exc:
    raise ImportError(
        'plotly is required for 3D stick figure visualization.'
    ) from exc
  root_stick = tree[tree.root]
  joint_locations = {root_stick.start: (0, 0, 0)}
  for stick in tr.breadth_first_traversal(tree):
    start_joint, end_joint = (
        (stick.start, stick.end)
        if stick.start in joint_locations
        else (stick.end, stick.start)
    )

    if end_joint not in joint_locations:
      x0, y0, z0 = joint_locations[start_joint]
      dx, dy, dz = stick.get_coords(start_joint)
      joint_locations[end_joint] = (x0 + dx, y0 + dy, z0 + dz)  # pyrefly: ignore[unsupported-operation]

  traces = []
  labels = dict(x=[], y=[], z=[], text=[])
  stick_x, stick_y, stick_z = [], [], []
  shaping_x, shaping_y, shaping_z = [], [], []

  for stick in tree.children:
    x0, y0, z0 = joint_locations[stick.start]
    x1, y1, z1 = joint_locations[stick.end]
    if stick.shaping_flap:
      shaping_x.extend([x0, x1, None])
      shaping_y.extend([y0, y1, None])
      shaping_z.extend([z0, z1, None])
    else:
      stick_x.extend([x0, x1, None])
      stick_y.extend([y0, y1, None])
      stick_z.extend([z0, z1, None])

    mx, my, mz = ((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2)
    labels['x'].append(mx)
    labels['y'].append(my)
    labels['z'].append(mz)
    labels['text'].append(f'{stick.name}: {stick.length}')

  traces.append(
      go.Scatter3d(
          x=stick_x,
          y=stick_y,
          z=stick_z,
          mode='lines',
          line=dict(color='black', width=6),
          name='Sticks',
          hoverinfo='none',
      )
  )
  traces.append(
      go.Scatter3d(
          x=shaping_x,
          y=shaping_y,
          z=shaping_z,
          mode='lines',
          line=dict(color='mediumpurple', width=6),
          name='Shaping Sticks',
          hoverinfo='none',
      )
  )

  # Plot stick labels
  if show_labels:
    traces.append(
        go.Scatter3d(
            labels,
            mode='text',
            textfont=dict(color='green', size=10),
            name='Labels',
            hoverinfo='none',
        )
    )

  # Plot joint markers
  joint_names, joint_coords = list(zip(*joint_locations.items()))
  joint_coords = np.array(joint_coords).T
  joint_x, joint_y, joint_z = joint_coords

  traces.append(
      go.Scatter3d(
          x=joint_x,
          y=joint_y,
          z=joint_z,
          mode='markers',
          marker=dict(color='blue', size=5, symbol='circle'),
          text=[f'Joint: {name}' for name in joint_names],
          hoverinfo='text',
          name='Joints',
      )
  )

  fig = go.Figure(data=traces)

  scene_dict = dict(
      xaxis=dict(title='X'),
      yaxis=dict(title='Y'),
      zaxis=dict(title='Z'),
  )

  # Set axes limits
  max_coords = joint_coords.max(axis=1)
  min_coords = joint_coords.min(axis=1)
  max_range = np.max(max_coords - min_coords)
  max_range = max(max_range, 1.0)
  mid_x, mid_y, mid_z = (max_coords + min_coords) * 0.5
  scene_dict['xaxis']['range'] = [mid_x - max_range / 2, mid_x + max_range / 2]  # pyrefly: ignore[bad-assignment]
  scene_dict['yaxis']['range'] = [mid_y - max_range / 2, mid_y + max_range / 2]  # pyrefly: ignore[bad-assignment]
  scene_dict['zaxis']['range'] = [mid_z - max_range / 2, mid_z + max_range / 2]  # pyrefly: ignore[bad-assignment]

  fig.update_layout(
      title_text=f'{tree.name} | complexity={tree.complexity}',
      width=figsize[0],
      height=figsize[1],
      scene=scene_dict,
      scene_aspectmode='cube',
      legend=dict(x=0.01, y=0.99),
      template='plotly_white',
  )
  return fig
