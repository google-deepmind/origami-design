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

"""Utilities for drawing folded paper grids."""

import dataclasses
from typing import TypeAlias

from matplotlib import patches
import matplotlib.pyplot as plt

from . import triangle_geometry

CellTriangle: TypeAlias = triangle_geometry.CellTriangle
TriangleCellPosition: TypeAlias = triangle_geometry.TriangleCellPosition

MAGENTA = "#ec008b"
PALE_PINK = "#ecaa8b"

DEFAULT_TRIANGLE_COLOR_MAP = {
    CellTriangle.TOP: MAGENTA,
    CellTriangle.RIGHT: MAGENTA,
    CellTriangle.BOTTOM: MAGENTA,
    CellTriangle.LEFT: MAGENTA,
}

DEFAULT_TRIANGLE_BACK_COLOR_MAP = {
    CellTriangle.TOP: PALE_PINK,
    CellTriangle.RIGHT: PALE_PINK,
    CellTriangle.BOTTOM: PALE_PINK,
    CellTriangle.LEFT: PALE_PINK,
}


@dataclasses.dataclass(kw_only=True, frozen=True)
class TriangleRenderingOptions:
  """Options for rendering triangles."""

  triangle_colors: dict[CellTriangle, str]
  triangle_back_colors: dict[CellTriangle, str]
  opacity: float  # Opacity is in the range [0.0, 1.0].

  def __post_init__(self):
    if not 0.0 <= self.opacity <= 1.0:
      raise ValueError("Opacity must be in the range [0.0, 1.0].")

  @classmethod
  def from_default(
      cls, enable_back_colors: bool = False, opacity: float = 0.2
  ) -> "TriangleRenderingOptions":
    return TriangleRenderingOptions(
        triangle_colors=DEFAULT_TRIANGLE_COLOR_MAP,
        triangle_back_colors=(
            DEFAULT_TRIANGLE_COLOR_MAP
            if not enable_back_colors
            else DEFAULT_TRIANGLE_BACK_COLOR_MAP
        ),
        opacity=opacity,
    )


def plot_triangles_default(triangles: list[TriangleCellPosition]):
  _, ax = plt.subplots(figsize=(16, 16))
  draw_triangles_on_plot(ax, TriangleRenderingOptions.from_default(), triangles)
  plt.show()


def draw_triangles_on_plot(
    ax: plt.Axes,
    options: TriangleRenderingOptions,
    triangles: list[TriangleCellPosition],
):
  """Draws a list of triangles on a matplotlib plot.

  Args:
    ax: The matplotlib axes to draw on.
    options: The rendering options to use.
    triangles: A list of TriangleCellPosition objects to draw.
  """

  if not triangles:
    ax.text(
        0.5,
        0.5,
        "Warning: No triangles provided to draw.",
        horizontalalignment="center",
        verticalalignment="center",
        transform=ax.transAxes,
        color="red",
        fontsize=12,
    )
    ax.axis("off")
    return

  all_cols = [p.col for p in triangles]
  all_rows = [p.row for p in triangles]
  side_color_map = {
      -1: options.triangle_back_colors,
      1: options.triangle_colors,
  }

  for pos in triangles:
    polygon = patches.Polygon(
        pos.vertices_2d(),
        edgecolor="none",
        facecolor=side_color_map[pos.z_sign][pos.tr],
        alpha=options.opacity,
    )
    ax.add_patch(polygon)

  min_col, max_col = min(all_cols), max(all_cols)
  min_row, max_row = min(all_rows), max(all_rows)

  ax.set_xlim(min_col - 1, max_col + 1)
  ax.set_ylim(max_row + 1, min_row - 1)
  ax.set_aspect("equal", adjustable="box")
  ax.axis("off")
