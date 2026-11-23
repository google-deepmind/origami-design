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

"""Maps stick-figure joints and cross-sections onto unit-paper coordinates."""

from collections.abc import Mapping, Sequence
import dataclasses
import math

import numpy as np

from .. import bus
from ..packing import packing_output as packing_output_lib
from ..stick_figures import stick as stick_lib
from ..stick_figures import stick_figure as stick_figure_lib
from ..stick_figures import traversal
from . import bus_adapter

# Fractions from proximal to distal along a stick where sections are sampled.
SECTION_FRACTIONS: tuple[float, ...] = (0.25, 0.5, 0.75)

_SAMPLE_STEP = 0.5
_MAX_SAMPLES_PER_CONTOUR = 96
_PAPER_TOLERANCE = 1e-6
_BASE_RING_MARGIN = 0.05
_MIN_BASE_RING_POINTS = 4


@dataclasses.dataclass(frozen=True, slots=True)
class Topology:
  """Root-oriented tree topology of a `StickFigure`.

  Attributes:
    order: Stick names in breadth-first order starting at the root.
    proximal: Map from stick name to its root-side joint name.
    distal: Map from stick name to its tip-side joint name.
    parent: Map from stick name to its parent stick name (`None` for the root).
  """

  order: tuple[str, ...]
  proximal: Mapping[str, str]
  distal: Mapping[str, str]
  parent: Mapping[str, str | None]


@dataclasses.dataclass(frozen=True, slots=True)
class PaperLayout:
  """Sampled unit-paper contours for stick-figure joints and cross-sections.

  Attributes:
    joints: Map from joint name to `(N, 2)` unit-paper sample points.
    sections: Map from `'<stick>@<fraction>'` to `(N, 2)` sample points.
    chains: Map from packed flap name to its stick chain (proximal first).
    grid_size: Integer packing grid size.
    grid_units_per_tree_unit: Median grid-to-tree length scale factor.
    flap_rectangles: Flap bounding boxes `(x0, x1, y0, y1)` on the unit paper.
  """

  joints: Mapping[str, np.ndarray]
  sections: Mapping[str, np.ndarray]
  chains: Mapping[str, tuple[str, ...]]
  grid_size: int
  grid_units_per_tree_unit: float
  flap_rectangles: tuple[tuple[float, float, float, float], ...]

  @property
  def tree_units_per_paper_unit(self) -> float:
    """Conversion factor from unit-paper lengths to stick-figure tree units."""
    return self.grid_size / self.grid_units_per_tree_unit


@dataclasses.dataclass(frozen=True, slots=True)
class _GridBox:
  """Axis-aligned flap rectangle in `[0, grid_size]` grid coordinates."""

  name: str
  x0: float
  x1: float
  y0: float
  y1: float

  @property
  def width(self) -> float:
    return self.x1 - self.x0

  @property
  def height(self) -> float:
    return self.y1 - self.y0


def stick_vector(stick: stick_lib.Stick) -> np.ndarray:
  """Returns the 3D vector from `stick.start` to `stick.end` in figure frame.

  The figure frame uses `+x` forward (`azimuth=0`), `+y` left (`azimuth=90`),
  and `+z` up (`elevation=90`).

  Args:
    stick: Stick definition in spherical tree coordinates.

  Returns:
    `(3,)` displacement vector in tree units.
  """
  az = math.radians(stick.azimuth_angle)
  el = math.radians(stick.elevation_angle)
  cos_el = math.cos(el)
  return float(stick.length) * np.array(
      [cos_el * math.cos(az), cos_el * math.sin(az), math.sin(el)],
      dtype=float,
  )


def topology(tree: stick_figure_lib.StickFigure) -> Topology:
  """Computes root-oriented `Topology` via breadth-first traversal."""
  root = tree[tree.root]
  proximal: dict[str, str] = {root.name: root.start}
  distal: dict[str, str] = {root.name: root.end}
  parent: dict[str, str | None] = {root.name: None}
  arriving: dict[str, str] = {root.end: root.name, root.start: root.name}
  order: list[str] = []
  for stick in traversal.breadth_first_traversal(tree):
    order.append(stick.name)
    if stick.name == root.name:
      continue
    if stick.start in arriving:
      near, far = stick.start, stick.end
    else:
      near, far = stick.end, stick.start
    proximal[stick.name] = near
    distal[stick.name] = far
    parent[stick.name] = arriving.get(near)
    arriving.setdefault(far, stick.name)
  return Topology(tuple(order), proximal, distal, parent)


def stick_figure_joint_positions(
    tree: stick_figure_lib.StickFigure,
) -> dict[str, np.ndarray]:
  """Computes 3D figure-frame joint positions with root start at the origin."""
  topo = topology(tree)
  positions: dict[str, np.ndarray] = {
      tree[tree.root].start: np.zeros(3, dtype=float)
  }
  for name in topo.order:
    stick = tree[name]
    vec = stick_vector(stick)
    if topo.proximal[name] != stick.start:
      vec = -vec
    positions[topo.distal[name]] = positions[topo.proximal[name]] + vec
  return positions


def _rectangle_contour(
    x0: float, x1: float, y0: float, y1: float, step: float
) -> np.ndarray:
  """Samples evenly spaced points around rectangle `[x0, x1] x [y0, y1]`."""
  w, h = max(x1 - x0, 0.0), max(y1 - y0, 0.0)
  if w == 0.0 and h == 0.0:
    return np.array([[x0, y0]], dtype=float)
  if w == 0.0 or h == 0.0:
    count = max(2, math.ceil(max(w, h) / step) + 1)
    t = np.linspace(0.0, 1.0, count)
    return np.stack([x0 + t * w, y0 + t * h], axis=1)
  perimeter = 2.0 * (w + h)
  count = min(_MAX_SAMPLES_PER_CONTOUR, max(8, math.ceil(perimeter / step)))
  s = np.linspace(0.0, perimeter, count, endpoint=False)
  x = np.where(
      s < w,
      x0 + s,
      np.where(
          s < w + h,
          x1,
          np.where(s < 2.0 * w + h, x1 - (s - w - h), x0),
      ),
  )
  y = np.where(
      s < w,
      y0,
      np.where(
          s < w + h,
          y0 + s - w,
          np.where(s < 2.0 * w + h, y1, y1 - (s - 2.0 * w - h)),
      ),
  )
  return np.stack([x, y], axis=1)


def _on_paper_mask(points: np.ndarray, grid_size: int) -> np.ndarray:
  return np.all(
      (points >= -_PAPER_TOLERANCE) & (points <= grid_size + _PAPER_TOLERANCE),
      axis=1,
  )


def _outside_box_mask(
    points: np.ndarray,
    x0: float,
    x1: float,
    y0: float,
    y1: float,
    inset: float = 0.0,
) -> np.ndarray:
  return ~(
      (points[:, 0] > x0 + inset)
      & (points[:, 0] < x1 - inset)
      & (points[:, 1] > y0 + inset)
      & (points[:, 1] < y1 - inset)
  )


def _flap_contour(box: _GridBox, fraction: float, grid_size: int) -> np.ndarray:
  """Samples unit-paper points at axial `fraction` (`0` = tip, `1` = base)."""
  radius = 0.5 * min(box.width, box.height)
  inset = radius * (1.0 - min(max(fraction, 0.0), 1.0))
  pts = _rectangle_contour(
      box.x0 + inset,
      box.x1 - inset,
      box.y0 + inset,
      box.y1 - inset,
      _SAMPLE_STEP,
  )
  keep = _on_paper_mask(pts, grid_size)
  return np.clip(pts[keep], 0.0, grid_size) / grid_size


def _flap_base_contour(
    box: _GridBox, grid_size: int, others: Sequence[_GridBox]
) -> np.ndarray:
  """Samples points just outside `box` that stay anchored to its base joint."""
  m = _BASE_RING_MARGIN
  pts = _rectangle_contour(
      box.x0 - m, box.x1 + m, box.y0 - m, box.y1 + m, _SAMPLE_STEP
  )
  keep = _on_paper_mask(pts, grid_size)
  for other in others:
    keep &= _outside_box_mask(pts, other.x0, other.x1, other.y0, other.y1)
  if int(keep.sum()) < _MIN_BASE_RING_POINTS:
    return _flap_contour(box, 1.0, grid_size)
  return np.clip(pts[keep], 0.0, grid_size) / grid_size


def _extract_grid_boxes(
    packing: packing_output_lib.PackingOutput | bus.PackedCreasePattern,
) -> tuple[int, list[_GridBox]]:
  """Extracts `(grid_size, flap_boxes)` from a packing record."""
  if isinstance(packing, bus.PackedCreasePattern):
    if packing.grid_size <= 0 or not packing.flaps:
      raise ValueError('PackedCreasePattern has no grid or flap rectangles.')
    boxes: list[_GridBox] = []
    for flap in packing.flaps.values():
      rect = bus_adapter.unclipped_flap_rectangle(
          flap, grid_size=packing.grid_size
      )
      boxes.append(
          _GridBox(
              name=rect.name,
              x0=float(rect.x_start),
              x1=float(rect.x_start + rect.x_len),
              y0=float(rect.y_start),
              y1=float(rect.y_start + rect.y_len),
          )
      )
    return int(packing.grid_size), boxes

  if packing.packing_cp is None or not packing.flap_rectangles:
    raise ValueError('PackingOutput has no crease pattern or flap rectangles.')
  boxes = [
      _GridBox(
          name=rect.name,
          x0=float(rect.x_start),
          x1=float(rect.x_start + rect.x_len),
          y0=float(rect.y_start),
          y1=float(rect.y_start + rect.y_len),
      )
      for rect in packing.flap_rectangles
  ]
  return int(packing.packing_cp.grid_size), boxes


def _sample_river_sections(
    tree: stick_figure_lib.StickFigure,
    topo: Topology,
    all_boxes: Sequence[_GridBox],
    box_by_chain_root: Mapping[str, _GridBox],
    grid_size: int,
    scale_grid: float,
) -> dict[str, np.ndarray]:
  """Samples cross-section contours along non-root river sticks."""
  sections: dict[str, np.ndarray] = {}
  for river_name in topo.order:
    if river_name == tree.root or not tree[river_name].is_river:
      continue
    subtree: list[tuple[_GridBox, float]] = []
    for chain_root, box in box_by_chain_root.items():
      curr: str | None = chain_root
      inter_len = 0.0
      while curr is not None and curr != river_name:
        par = topo.parent.get(curr)
        if par is not None and par != river_name and tree[par].is_river:
          inter_len += float(tree[par].length)
        curr = par
      if curr == river_name:
        subtree.append((box, inter_len))
    if not subtree:
      continue

    subtree_ids = {id(b) for b, _ in subtree}
    outside = [b for b in all_boxes if id(b) not in subtree_ids]
    river_len = float(tree[river_name].length)
    for fraction in (0.0,) + SECTION_FRACTIONS:
      from_distal = (1.0 - fraction) * river_len
      expanded = [
          (box, (from_distal + inter_len) * scale_grid + _BASE_RING_MARGIN)
          for box, inter_len in subtree
      ]
      parts: list[np.ndarray] = []
      for box, exp in expanded:
        pts = _rectangle_contour(
            box.x0 - exp, box.x1 + exp, box.y0 - exp, box.y1 + exp, _SAMPLE_STEP
        )
        keep = _on_paper_mask(pts, grid_size)
        for other_box, other_exp in expanded:
          if other_box is box:
            continue
          keep &= _outside_box_mask(
              pts,
              other_box.x0 - other_exp,
              other_box.x1 + other_exp,
              other_box.y0 - other_exp,
              other_box.y1 + other_exp,
              inset=_PAPER_TOLERANCE,
          )
        for out_box in outside:
          keep &= _outside_box_mask(
              pts, out_box.x0, out_box.x1, out_box.y0, out_box.y1
          )
        if np.any(keep):
          parts.append(np.clip(pts[keep], 0.0, grid_size) / grid_size)
      if parts:
        combined = np.concatenate(parts, axis=0)
        if len(combined) >= _MIN_BASE_RING_POINTS:
          sections[f'{river_name}@{fraction:g}'] = combined
  return sections


def build_paper_layout(
    tree: stick_figure_lib.StickFigure,
    packing: packing_output_lib.PackingOutput | bus.PackedCreasePattern,
) -> PaperLayout:
  """Locates stick-figure joints and cross-sections on the unit paper.

  Joints incident exclusively to river sticks are omitted from `joints`.

  Args:
    tree: Unmerged source `StickFigure`.
    packing: Packed layout (`PackedCreasePattern` or legacy `PackingOutput`).

  Returns:
    Immutable `PaperLayout` of unit-paper sample contours.
  """
  grid_size, boxes = _extract_grid_boxes(packing)
  _, merged_names = tree.merge_linear_chains()
  topo = topology(tree)
  rank = {name: idx for idx, name in enumerate(topo.order)}

  joint_parts: dict[str, list[np.ndarray]] = {}
  sections: dict[str, np.ndarray] = {}
  chains: dict[str, tuple[str, ...]] = {}
  scales: list[float] = []
  unit_rects: list[tuple[float, float, float, float]] = []
  box_by_chain_root: dict[str, _GridBox] = {}

  for box in boxes:
    chain = list(merged_names.get(box.name, [box.name]))
    if not chain or any(name not in tree for name in chain):
      continue
    total_len = float(sum(tree[name].length for name in chain))
    if total_len <= 0.0:
      continue
    unit_rects.append((
        box.x0 / grid_size,
        box.x1 / grid_size,
        box.y0 / grid_size,
        box.y1 / grid_size,
    ))
    scales.append(0.5 * min(box.width, box.height) / total_len)
    ordered_chain = tuple(sorted(chain, key=lambda n: rank[n]))
    chains[box.name] = ordered_chain
    box_by_chain_root[ordered_chain[0]] = box

    distal_first = list(reversed(ordered_chain))
    tip = topo.distal[distal_first[0]]
    joint_parts.setdefault(tip, []).append(_flap_contour(box, 0.0, grid_size))

    others = [other for other in boxes if other is not box]
    dist_from_tip = 0.0
    for idx, stick_name in enumerate(distal_first):
      stick_len = float(tree[stick_name].length)
      for frac in SECTION_FRACTIONS:
        from_tip = dist_from_tip + (1.0 - frac) * stick_len
        sections[f'{stick_name}@{frac:g}'] = _flap_contour(
            box, from_tip / total_len, grid_size
        )
      dist_from_tip += stick_len
      is_chain_base = idx == len(distal_first) - 1
      contour = (
          _flap_base_contour(box, grid_size, others)
          if is_chain_base
          else _flap_contour(box, dist_from_tip / total_len, grid_size)
      )
      joint_parts.setdefault(topo.proximal[stick_name], []).append(contour)

  scale_grid = float(np.median(scales)) if scales else 1.0
  sections.update(
      _sample_river_sections(
          tree, topo, boxes, box_by_chain_root, grid_size, scale_grid
      )
  )
  joints = {
      name: np.concatenate(parts, axis=0)
      for name, parts in joint_parts.items()
      if any(len(p) > 0 for p in parts)
  }
  return PaperLayout(
      joints=joints,
      sections={k: v for k, v in sections.items() if len(v) > 0},
      chains=chains,
      grid_size=grid_size,
      grid_units_per_tree_unit=scale_grid,
      flap_rectangles=tuple(unit_rects),
  )


__all__ = [
    'PaperLayout',
    'SECTION_FRACTIONS',
    'Topology',
    'build_paper_layout',
    'stick_figure_joint_positions',
    'stick_vector',
    'topology',
]
