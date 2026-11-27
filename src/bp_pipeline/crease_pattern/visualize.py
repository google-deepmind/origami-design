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

"""Visualize a crease pattern as an image."""

from __future__ import annotations

from collections.abc import Sequence
import io
import math
from typing import Any, TYPE_CHECKING

from bp_pipeline.crease_pattern import crease
from bp_pipeline.crease_pattern import crease_pattern
from bp_pipeline.crease_pattern import vertex

if TYPE_CHECKING:
  from matplotlib import axes  # pylint: disable=g-import-not-at-top,g-bad-import-order


def draw_crease(
    ax: axes.Axes,
    creaseline: crease.Crease,
    show_invalid: bool = False,
    lw: float = 0.8,
) -> None:
  """Draws a line representing the crease on a matplotlib axis."""
  from matplotlib import lines  # pylint: disable=g-import-not-at-top
  line = lines.Line2D(
      [creaseline.p1.x, creaseline.p2.x],
      [creaseline.p1.y, creaseline.p2.y],
      color=creaseline.fold.color(),
      alpha=creaseline.fold_percent,
      lw=lw,
  )
  ax.add_line(line)

  if show_invalid and not creaseline.is_valid:
    highlight = lines.Line2D(
        [creaseline.p1.x, creaseline.p2.x],
        [creaseline.p1.y, creaseline.p2.y],
        color='red',
        linewidth=max(4.0, lw * 4.0),
        alpha=0.4,
        zorder=-1,
    )
    ax.add_line(highlight)


def draw_vertex(
    ax: axes.Axes,
    vert: vertex.Vertex,
    only_internal: bool = False,
    show_invalid: bool = False,
    only_flat_foldable: bool = True,
    vertex_size: float = 6.0,
) -> None:
  """Draws a vertex on a matplotlib axis."""
  if only_internal and not vert.internal:
    return
  # If show_not_ff_vertices is True, but show_vertices is False, then we
  # should not show vertices that are flat foldable.
  if only_flat_foldable and vert.flat_foldability():
    return

  point = (vert.point.x, vert.point.y)
  vertex_color = 'g' if vert.flat_foldability() else 'r'
  ax.plot(*point, marker='o', markersize=vertex_size, color=vertex_color)

  if show_invalid and not vert.is_valid:
    from matplotlib import pyplot as plt  # pylint: disable=g-import-not-at-top
    highlight = plt.Circle(point, 0.02, color='red', fill=False, lw=1.5)
    ax.add_patch(highlight)


def draw_cp(
    pattern: crease_pattern.CreasePattern,
    *,
    show_only_not_flat_foldable_vertices: bool = True,
    show_vertices: bool = False,
    annotate_vertices: bool = True,
    annotate_flat_foldability: bool = True,
    show_invalid: bool = False,
    grid_size: int = -1,
    only_internal: bool = False,
    figsize: tuple[float, float] = (10, 10),
    fold_step: int | None = None,
    print_sector_angles_on_non_flat_foldable_vertices: bool = False,
    ax: axes.Axes | None = None,
    flaps: Sequence[Any] | None = None,
    rivers: Sequence[Any] | None = None,
    lw: float = 0.75,
    vertex_size: float = 6.0,
) -> tuple[Any, axes.Axes]:
  """Draws the crease pattern.

  Args:
    pattern: The crease pattern to draw.
    show_only_not_flat_foldable_vertices: Whether to show (only) vertices that
      are not flat foldable.
    show_vertices: Whether to show the vertices.
    annotate_vertices: Whether to annotate the vertices.
    annotate_flat_foldability: Whether to show flat foldability count.
    show_invalid: Highlight invalid vertices.
    grid_size: The number of grid lines to show. If -1, no grid is shown.
    only_internal: Only show internal vertices.
    figsize: The matplotlib figure size.
    fold_step: The fold step to draw creases at. If None, all creases are drawn.
    print_sector_angles_on_non_flat_foldable_vertices: Whether to print the
      sector angles on non-flat-foldable vertices.
    ax: The matplotlib axis to draw on. If None, a new figure and axis will be
      created.
    flaps: Optional flaps to label.
    rivers: Optional rivers to label.
    lw: Linewidth for creases (default 0.75 for crisp, thin rendering).
    vertex_size: Marker size for vertices (default 6.0).

  Returns:
    The matplotlib figure and axis.
  """
  from matplotlib import pyplot as plt  # pylint: disable=g-import-not-at-top
  from matplotlib import ticker  # pylint: disable=g-import-not-at-top

  if ax is None:
    fig, ax = plt.subplots(figsize=figsize)
  else:
    fig = ax.figure
  ax.set_xlim(-0.05, 1.05)
  ax.set_ylim(-0.05, 1.05)
  ax.set_aspect('equal', adjustable='box')

  for c in pattern.creases:
    if fold_step is None or c.fold_step <= fold_step:
      draw_crease(ax, c, show_invalid=show_invalid, lw=lw)

  if show_vertices or show_only_not_flat_foldable_vertices:
    for i, vert in enumerate(pattern.vertices):
      draw_vertex(
          ax,
          vert,
          only_internal=only_internal,
          show_invalid=show_invalid,
          only_flat_foldable=show_only_not_flat_foldable_vertices,
          vertex_size=vertex_size,
      )
      if print_sector_angles_on_non_flat_foldable_vertices:
        if not vert.flat_foldability():
          print(f'Vertex: {i} is not flat foldable with sector angles:')
          for angle in vert.sector_angles():
            print(angle)
      if annotate_vertices:
        ax.annotate(str(i), xy=(vert.point.x, vert.point.y), xycoords='data')

  if annotate_flat_foldability:
    flat_foldable_count = sum(v.flat_foldability() for v in pattern.vertices)
    count_text = f'Flat-foldable: {flat_foldable_count}/{len(pattern.vertices)}'
    # Position text in the top right corner of the axes
    ax.text(
        0.99,
        0.99,
        count_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment='top',
        horizontalalignment='right',
    )

  if grid_size > 0:
    ax.grid(True, ls='--', color='gray', alpha=0.7, zorder=-1)
    tick_interval = 1 / grid_size
    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_interval))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(tick_interval))
  else:
    ax.grid(False)
    ax.axis('off')

  # Label flaps if given (supports both bus.Flap and legacy Rectangle).
  scale = float(grid_size) if grid_size > 0 else 1.0
  if flaps:
    for flap in flaps:
      if hasattr(flap, 'rect'):
        flap_x = (flap.rect.x + flap.rect.w / 2.0) / scale
        flap_y = (flap.rect.y + flap.rect.h / 2.0) / scale
        flap_label = getattr(flap, 'id', getattr(flap, 'name', ''))
      else:
        raw_x = flap.x_start + flap.x_len / 2.0
        raw_y = flap.y_start + flap.y_len / 2.0
        if (raw_x > 1.0 or raw_y > 1.0) and scale > 1.0:
          flap_x, flap_y = raw_x / scale, raw_y / scale
        else:
          flap_x, flap_y = raw_x, raw_y
        flap_label = getattr(flap, 'name', getattr(flap, 'id', ''))
      ax.text(
          flap_x,
          flap_y,
          flap_label,
          ha='center',
          va='center',
          fontsize=12,
          color='black',
          fontweight='bold',
      )

  # Label rivers if given (supports both bus.River and legacy PackingRiver).
  if rivers:
    for river in rivers:
      if hasattr(river, 'rects'):
        rects = river.rects
        if not rects:
          continue
        mid_rect = rects[len(rects) // 2]
        river_x = (mid_rect.x + mid_rect.w / 2.0) / scale
        river_y = (mid_rect.y + mid_rect.h / 2.0) / scale
        river_label = getattr(river, 'id', getattr(river, 'name', ''))
      else:
        rects = river.covering_rectangles
        if not rects:
          continue
        mid_rect = rects[len(rects) // 2]
        raw_x = mid_rect.x_start + mid_rect.x_len / 2.0
        raw_y = mid_rect.y_start + mid_rect.y_len / 2.0
        if (raw_x > 1.0 or raw_y > 1.0) and scale > 1.0:
          river_x, river_y = raw_x / scale, raw_y / scale
        else:
          river_x, river_y = raw_x, raw_y
        river_label = getattr(river, 'name', getattr(river, 'id', ''))
      ax.text(
          river_x,
          river_y,
          river_label,
          ha='center',
          va='center',
          fontsize=12,
          color='black',
          fontweight='bold',
      )

  ax.invert_yaxis()
  return fig, ax


def figure_to_image(
    figure: Any, file_format: str = 'jpg', dpi: float | str = 'figure'
) -> bytes:
  """Converts a matplotlib figure to a bytes image."""
  image_bytes = io.BytesIO()
  figure.savefig(image_bytes, format=file_format, bbox_inches='tight', dpi=dpi)
  image_bytes.seek(0)
  return image_bytes.read()


def folding_grid(
    pattern: crease_pattern.CreasePattern, cols: int = 4, **kwargs
) -> tuple[Any, Any]:
  """Displays a grid of crease pattern folding steps.

  Args:
    pattern: The crease pattern to draw.
    cols: The maximum number of columns in the grid.
    **kwargs: Keyword arguments to pass to draw_cp.

  Returns:
    The matplotlib figure and axes.
  """
  from matplotlib import pyplot as plt  # pylint: disable=g-import-not-at-top

  fold_steps = max(c.fold_step for c in pattern.creases) + 1
  rows, cols = math.ceil(fold_steps / cols), min(fold_steps, cols)
  fig, axs = plt.subplots(
      rows, cols, figsize=(cols * 4, rows * 4), squeeze=False
  )
  for i, ax in enumerate(axs.flatten()):
    ax.axis('off')
    if i < fold_steps:
      draw_cp(pattern, ax=ax, fold_step=i, show_vertices=False, **kwargs)

  plt.tight_layout()
  return fig, axs


def render_cps_to_png_bytes(
    cps: Sequence[crease_pattern.CreasePattern],
) -> bytes:
  """Renders a list of CreasePatterns into a single PNG image as bytes."""
  if not cps:
    return b''
  from matplotlib import pyplot as plt  # pylint: disable=g-import-not-at-top

  num_cps = len(cps)
  grid_size = math.ceil(math.sqrt(num_cps))
  figure, axs = plt.subplots(
      grid_size,
      grid_size,
      figsize=(grid_size * 6, grid_size * 6),
      squeeze=False,
  )

  for i, ax in enumerate(axs.flatten()):
    if i >= num_cps:
      figure.delaxes(ax)
      continue

    cp = cps[i]
    figure, ax = draw_cp(cp, show_invalid=True, grid_size=cp.grid_size, ax=ax)
    ax.set_title(f'Crease-pattern {i+1}')
    ax.axis('off')

  figure.tight_layout()
  # Save the figure to a BytesIO object
  png_bytes = figure_to_image(figure, file_format='png')
  plt.close(figure)
  return png_bytes


def render_highlighted_cps_to_png_bytes(
    cps: Sequence[crease_pattern.CreasePattern],
    packing_outputs: Sequence[Any],
) -> bytes:
  """Renders CPs into a PNG with sacrificial and width-based overlays."""
  if not cps:
    return b''
  from matplotlib import patches  # pylint: disable=g-import-not-at-top
  from matplotlib import pyplot as plt  # pylint: disable=g-import-not-at-top

  sacrificial_suffixes = (
      '_sf1',
      '_sf2',
      '__sacrificial_vertical',
      '__sacrificial_horizontal',
      '__corner',
  )

  num_cps = len(cps)
  grid_size = math.ceil(math.sqrt(num_cps))
  figure, axs = plt.subplots(
      grid_size,
      grid_size,
      figsize=(grid_size * 6, grid_size * 6),
      squeeze=False,
  )

  for i, ax in enumerate(axs.flatten()):
    if i >= num_cps:
      figure.delaxes(ax)
      continue

    cp = cps[i]
    figure, ax = draw_cp(cp, show_invalid=True, grid_size=cp.grid_size, ax=ax)
    ax.set_title(f'Crease-pattern {i+1}')
    ax.axis('off')

    output = packing_outputs[i] if i < len(packing_outputs) else None
    if output and output.tracker:
      gs = output.tracker.grid_size
      stick_fig = output.stick_figure

      if stick_fig:
        for name, placement in output.tracker.packed_polygons.items():
          if any(s in name for s in sacrificial_suffixes):
            continue
          if name not in stick_fig:
            continue
          s = stick_fig[name]
          if not s.is_flap or s.width == 1:
            continue
          ox, oy = placement.offset
          w, h = placement.size
          if s.width > 1:
            alpha = min(s.width / 4.0, 1.0) * 0.5
            color = 'blue'
          else:
            alpha = min((1.0 / s.width) / 4.0, 1.0) * 0.5
            color = 'green'
          ax.add_patch(
              patches.Rectangle(
                  (ox / gs, oy / gs),
                  w / gs,
                  h / gs,
                  facecolor=color,
                  alpha=alpha,
                  zorder=10,
              )
          )

      for name, placement in output.tracker.packed_polygons.items():
        if any(s in name for s in sacrificial_suffixes):
          ox, oy = placement.offset
          w, h = placement.size
          ax.add_patch(
              patches.Rectangle(
                  (ox / gs, oy / gs),
                  w / gs,
                  h / gs,
                  facecolor='purple',
                  alpha=0.3,
                  zorder=10,
              )
          )

  figure.tight_layout()
  png_bytes = figure_to_image(figure, file_format='png')
  plt.close(figure)
  return png_bytes
