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

"""Function to generate SVG for packing based on shaping flaps and rivers.

Used by clip_pattern_shaping_adapter.
Code was written with Gemini 3.0 Pro.
TODO: Unify util functions with already existing.
"""

from collections.abc import Sequence
import itertools
import math
from typing import Callable

from . import shaping_utils

_LINE_WIDTH = 0.005
_LINE_COLOR = "red"
_BORDER_COLOR = "black"


def _is_close(
    a: float, b: float, rel_tol: float = 1e-9, abs_tol: float = 1e-9
) -> bool:
  """Checks if two floats are effectively equal."""
  return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)


def _get_overlap(
    min1: float, max1: float, min2: float, max2: float
) -> tuple[float, float] | None:
  """Finds the intersection of two ranges."""
  overlap_start = max(min1, min2)
  overlap_end = min(max1, max2)

  if overlap_end > overlap_start and not _is_close(overlap_end, overlap_start):
    return overlap_start, overlap_end
  return None


def _clip_segment_to_unit_square(
    x1: float, y1: float, x2: float, y2: float
) -> tuple[tuple[float, float], tuple[float, float]] | None:
  """Clips a line segment to the [0, 1] unit square."""
  if (
      (x1 < 0 and x2 < 0)
      or (x1 > 1 and x2 > 1)
      or (y1 < 0 and y2 < 0)
      or (y1 > 1 and y2 > 1)
  ):
    return None
  cx1 = max(0, min(1, x1))
  cy1 = max(0, min(1, y1))
  cx2 = max(0, min(1, x2))
  cy2 = max(0, min(1, y2))

  if _is_close(cx1, cx2) and _is_close(cy1, cy2):
    return None

  return (cx1, cy1), (cx2, cy2)


def _find_seams_between_rivers(
    river_a: Sequence[shaping_utils.Rectangle],
    river_b: Sequence[shaping_utils.Rectangle],
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
  """Finds seams between two rivers."""
  seams = []
  for r1, r2 in itertools.product(river_a, river_b):
    r1_x_end = r1.x_start + r1.x_len
    r1_y_end = r1.y_start + r1.y_len
    r2_x_end = r2.x_start + r2.x_len
    r2_y_end = r2.y_start + r2.y_len

    # Check Horizontal Seams
    shared_y = None
    if _is_close(r1.y_start, r2_y_end):
      shared_y = r1.y_start
    elif _is_close(r1_y_end, r2.y_start):
      shared_y = r1_y_end

    if shared_y is not None:
      overlap = _get_overlap(r1.x_start, r1_x_end, r2.x_start, r2_x_end)
      if overlap:
        seams.append(((overlap[0], shared_y), (overlap[1], shared_y)))

    # Check Vertical Seams
    shared_x = None
    if _is_close(r1.x_start, r2_x_end):
      shared_x = r1.x_start
    elif _is_close(r1_x_end, r2.x_start):
      shared_x = r1_x_end

    if shared_x is not None:
      overlap = _get_overlap(r1.y_start, r1_y_end, r2.y_start, r2_y_end)
      if overlap:
        seams.append(((shared_x, overlap[0]), (shared_x, overlap[1])))
  return seams


def _find_river_seams(
    rivers: list[Sequence[shaping_utils.Rectangle]],
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
  """Identifies shared boundaries between different rivers."""
  seams = []
  for river_a, river_b in itertools.combinations(rivers, 2):
    seams.extend(_find_seams_between_rivers(river_a, river_b))
  return seams


def _get_flap_segments(
    rect_list: list[shaping_utils.Rectangle],
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
  """Generates line segments for rectangles clipped to the unit square."""
  segments = []
  for r in rect_list:
    x_end = r.x_start + r.x_len
    y_end = r.y_start + r.y_len

    # Raw sides: Bottom, Right, Top, Left
    raw_lines = [
        (r.x_start, r.y_start, x_end, r.y_start),
        (x_end, r.y_start, x_end, y_end),
        (x_end, y_end, r.x_start, y_end),
        (r.x_start, y_end, r.x_start, r.y_start),
    ]

    for x1, y1, x2, y2 in raw_lines:
      clipped = _clip_segment_to_unit_square(x1, y1, x2, y2)
      if clipped:
        segments.append(clipped)
  return segments


def _round_scale_value(
    value: float, scale_factor: float, ndigits: int
) -> float:
  """Rounds and scales a value."""
  return round(value * scale_factor, ndigits)


def _round_scale_point(
    p: tuple[float, float], scale_factor: float, ndigits: int
) -> tuple[float, float]:
  """Rounds and scales a point."""
  return (
      _round_scale_value(p[0], scale_factor, ndigits),
      _round_scale_value(p[1], scale_factor, ndigits),
  )


def _create_svg_line(
    p1: tuple[float, float],
    p2: tuple[float, float],
    color: str,
    width: float,
) -> str:
  """Helper to format an SVG line string."""
  return (
      f'  <line x1="{p1[0]}" y1="{p1[1]}" '
      f'x2="{p2[0]}" y2="{p2[1]}" '
      f'stroke="{color}" stroke-width="{width}" />'
  )


def _get_round_scale_fn(
    scale_factor: float, ndigits: int
) -> Callable[[tuple[float, float]], tuple[float, float]]:
  """Returns a callback for _round_scale with fixed scale_factor and ndigits."""

  def round_scale_cb(p: tuple[float, float]) -> tuple[float, float]:
    return _round_scale_point(p, scale_factor, ndigits)

  return round_scale_cb


def generate_svg_string(
    flap_rectangles: list[shaping_utils.Rectangle],
    rivers_rectangles: list[Sequence[shaping_utils.Rectangle]],
    scale_factor: float = 1.0,
    ndigits: int = 8,
) -> str:
  """Orchestrates the data gathering and SVG formatting."""
  svg_lines = []
  round_scale = _get_round_scale_fn(scale_factor, ndigits)
  line_width = _round_scale_value(_LINE_WIDTH, scale_factor, ndigits)
  unit_square_edges = [
      ((0, 0), (1, 0)),
      ((1, 0), (1, 1)),
      ((1, 1), (0, 1)),
      ((0, 1), (0, 0)),
  ]
  for start, end in unit_square_edges:
    start = round_scale(start)
    end = round_scale(end)
    svg_lines.append(_create_svg_line(start, end, _BORDER_COLOR, line_width))

  rect_segments = _get_flap_segments(flap_rectangles)
  for start, end in rect_segments:
    start = round_scale(start)
    end = round_scale(end)
    svg_lines.append(_create_svg_line(start, end, _LINE_COLOR, line_width))

  seam_segments = _find_river_seams(rivers_rectangles)
  for start, end in seam_segments:
    start = round_scale(start)
    end = round_scale(end)
    svg_lines.append(_create_svg_line(start, end, _LINE_COLOR, line_width))

  vb_x, vb_y = round_scale((-0.1, -0.1))
  vb_width, vb_height = round_scale((1.2, 1.2))
  header = (
      f'<svg viewBox="{vb_x} {vb_y} {vb_width} {vb_height}"'
      ' xmlns="http://www.w3.org/2000/svg">'
  )
  footer = "</svg>"

  return header + "\n" + "\n".join(svg_lines) + "\n" + footer
