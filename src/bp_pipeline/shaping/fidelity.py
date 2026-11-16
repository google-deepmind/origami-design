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

"""Objective fidelity score: how well a folded model matches its stick figure.

The deterministic folder turns a (shaped) crease pattern into 3-D geometry:
every point of the paper has a folded 3-D position. The packing tells where
each joint of the stick figure lives on the paper. A flap's tip is its circle
center (or the ridge between its two centers) and the paper at axial distance
`d` from the tip lies on the box-pleating contour at L-infinity distance `d`
from the tip, i.e. on the boundary of the flap rectangle inset by
`radius - d` (the full rectangle boundary maps to the flap's base joint). The
mean folded position of a contour's sample points is the joint's 3-D position
in the model and the contour's folded spread is the flap's width there. On an
unshaped base every stick folds to exactly its stick-figure length (tested).

`measure` returns that geometry (JSON-friendly); `components` turns it into
scores in [0, 1], all invariant to the model's arbitrary frame:

*   `pose`: share of joints within 0.2 / 0.35 figure sizes of their
    stick-figure positions after a robust similarity alignment (rotation,
    uniform scale, translation; mirror images allowed: they look equally
    right).
*   `direction`: length-weighted share of sticks within 45 degrees of their
    stick-figure direction in the aligned frame.
*   `articulation`: per limb (paired appendage), how much of its stick-figure
    bend (knee, elbow, hock) the folded limb shows, and whether it bends at
    all when the stick figure gives it a joint.
*   `separation`: tips that are not clumped (nearest other tip at least 0.35
    of the appendage length away) and left/right pairs that are as far apart
    as in the stick figure (a 2-D fold leaves mirrored flaps stacked).
*   `width`: folded width of every flap stick vs. its stick-figure width
    (unnarrowed legs are planks, unfanned claws are sticks).
*   `depth`: 3-D extent (third vs. first principal extent): a model folded
    flat into one plane or a single bar reads as a stack.
*   `symmetry`: left/right mirror consistency of separated pairs.
*   `volume`: thickness of the body (paper outside all flap rectangles).

`score_model`/`score_pattern` add a headline `fidelity` on a 0-10 scale
(`combine`) and the alignment; `render_rotation` turns the alignment into a
rotation that shows the model upright in the stick figure's frame.
"""

from collections.abc import Mapping, Sequence
import copy
import dataclasses
import math
import os
import re
import time
from typing import Any

from matplotlib import path as mpath
import numpy as np

from .. import bus
from ..crease_pattern import crease as crease_utils
from ..crease_pattern import crease_pattern as crease_pattern_lib
from ..crease_pattern.parser import svg as svg_parser
from ..packing import packing_output as packing_output_lib
from ..stick_figures import stick_figure as stick_figure_lib
from ..stick_figures import traversal
from . import bus_adapter
from . import folder_3d

FIDELITY_REPORT_JSON = 'fidelity_report.json'
# A stick pointing further off than this is a stray flap.
STRAY_ANGLE_DEG = 60.0
# Critic gate (origami-critique): minimum direction component and maximum
# number of stray flaps for a model to pass "naturally splayed, no strays".
POSE_GATE_DIRECTION = 0.7
POSE_GATE_MAX_STRAY = 2
# Side check (`side_crossings`): a flap point the stick figure puts at least
# `SIDE_MIN_DESIGN_OFFSET` tree units to one side of the body's mid-plane must
# stay at least `SIDE_CROSSING_MARGIN` tree units on that side when folded.
# Points beyond the mid-plane have 'crossed' (a left wing folded over to the
# right); points within the margin of it are on the 'midline' (mirrored feet
# meeting and crossing under the body: eagle v3 talon tips at y = 0.01 and
# 0.06 for design offsets of +-1.55). Either fails `pose_gate`.
SIDE_MIN_DESIGN_OFFSET = 0.5
SIDE_CROSSING_MARGIN = 0.25
# Fraction along a flap stick of the mid-flap point that is checked too.
SIDE_SECTION_FRACTION = 0.5
# Maximum deterministic-folder vertex strain (%) for passing `clean_gate`.
CLEAN_GATE_MAX_VERTEX_ERROR = 0.85
_SILHOUETTE_GRID_SIZE = 56
# Fractions along each flap stick where cross-sections are sampled.
SECTION_FRACTIONS = (0.25, 0.5, 0.75)
# Sample spacing along a contour, in packing grid units.
_SAMPLE_STEP = 0.5
_MAX_SAMPLES_PER_CONTOUR = 96
# Folded sticks shorter than this fraction of the figure size have no
# reliable direction (the flap collapsed onto itself).
_MIN_FOLDED_FRACTION = 0.02
_PAPER_TOLERANCE = 1e-6
# A flap's base joint is located on a ring this far (packing grid units)
# outside its rectangle (see `_flap_base_contour`), when at least
# `_MIN_BASE_RING_POINTS` ring points lie on paper outside other flaps.
_BASE_RING_MARGIN = 0.05
_MIN_BASE_RING_POINTS = 4
# Cauchy scale of the robust alignment, in figure sizes.
_ROBUST_SCALE = 0.25
_ROBUST_ITERATIONS = 8
# Width of a uniform segment = sqrt(12) * its RMS spread.
_WIDTH_PER_SPREAD = math.sqrt(12.0)
# A limb "bends" when consecutive folded segments turn by at least this.
_MIN_VISIBLE_BEND_DEG = 20.0
# A stick figure whose joints' third principal extent is below this fraction
# of the first is planar (all elevations 0): `depth` is not required.
PLANAR_FIGURE_FLATNESS = 0.01


# ------------------------------------------------------------------------------
# Stick figure geometry.
# ------------------------------------------------------------------------------


def _stick_vector(stick) -> np.ndarray:
  """Vector from a stick's start joint to its end joint (tree units)."""
  azimuth = math.radians(stick.azimuth_angle)
  elevation = math.radians(stick.elevation_angle)
  return stick.length * np.array([
      math.cos(elevation) * math.cos(azimuth),
      math.cos(elevation) * math.sin(azimuth),
      math.sin(elevation),
  ])


@dataclasses.dataclass(frozen=True)
class Topology:
  """Root-relative structure of an unmerged stick figure.

  Attributes:
    order: Stick names in breadth-first order from the root stick.
    proximal: Stick name -> joint closer to the root stick.
    distal: Stick name -> joint further from the root stick.
    parent: Stick name -> the stick it hangs from (None for the root).
  """

  order: tuple[str, ...]
  proximal: Mapping[str, str]
  distal: Mapping[str, str]
  parent: Mapping[str, str | None]


def topology(tree: stick_figure_lib.StickFigure) -> Topology:
  """Computes `Topology` (the root stick points from its start joint)."""
  root = tree[tree.root]
  proximal = {root.name: root.start}
  distal = {root.name: root.end}
  parent: dict[str, str | None] = {root.name: None}
  arriving = {root.end: root.name, root.start: root.name}
  order = []
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
  """3-D joint positions of the stick figure (root start joint at the origin).

  Args:
    tree: The (unmerged) stick figure.

  Returns:
    Joint name -> position (x forward at azimuth 0, y left, z up), placing
    each stick from its root-side joint, whatever its start/end orientation.
  """
  topo = topology(tree)
  positions = {tree[tree.root].start: np.zeros(3)}
  for name in topo.order:
    stick = tree[name]
    vector = _stick_vector(stick)
    if topo.proximal[name] != stick.start:
      vector = -vector
    positions[topo.distal[name]] = positions[topo.proximal[name]] + vector
  return positions


# ------------------------------------------------------------------------------
# Paper layout: where joints and stick cross-sections live on the paper.
# ------------------------------------------------------------------------------


def _rectangle_contour(
    x0: float, x1: float, y0: float, y1: float, step: float
) -> np.ndarray:
  """Points along the boundary of [x0, x1] x [y0, y1] (degenerate allowed)."""
  width, height = max(x1 - x0, 0.0), max(y1 - y0, 0.0)
  if width == 0.0 and height == 0.0:
    return np.array([[x0, y0]])
  if width == 0.0 or height == 0.0:
    count = max(2, int(math.ceil(max(width, height) / step)) + 1)
    t = np.linspace(0.0, 1.0, count)
    return np.stack([x0 + t * width, y0 + t * height], axis=1)
  perimeter = 2 * (width + height)
  count = min(
      _MAX_SAMPLES_PER_CONTOUR, max(8, int(math.ceil(perimeter / step)))
  )
  s = np.linspace(0.0, perimeter, count, endpoint=False)
  x = np.where(
      s < width,
      x0 + s,
      np.where(
          s < width + height,
          x1,
          np.where(s < 2 * width + height, x1 - (s - width - height), x0),
      ),
  )
  y = np.where(
      s < width,
      y0,
      np.where(
          s < width + height,
          y0 + s - width,
          np.where(s < 2 * width + height, y1, y1 - (s - 2 * width - height)),
      ),
  )
  return np.stack([x, y], axis=1)


def _flap_contour(rectangle, fraction: float, grid_size: int) -> np.ndarray:
  """On-paper points at axial `fraction` of a flap (0 = tip, 1 = base).

  Args:
    rectangle: Packing flap rectangle (packing grid units).
    fraction: Distance from the tip as a fraction of the flap length.
    grid_size: Packing grid size (the paper is [0, grid_size]^2).

  Returns:
    (N, 2) points on the unit paper (possibly empty).
  """
  radius = min(rectangle.x_len, rectangle.y_len) / 2.0
  inset = radius * (1.0 - min(max(fraction, 0.0), 1.0))
  points = _rectangle_contour(
      rectangle.x_start + inset,
      rectangle.x_start + rectangle.x_len - inset,
      rectangle.y_start + inset,
      rectangle.y_start + rectangle.y_len - inset,
      _SAMPLE_STEP,
  )
  on_paper = np.all(
      (points >= -_PAPER_TOLERANCE) & (points <= grid_size + _PAPER_TOLERANCE),
      axis=1,
  )
  return np.clip(points[on_paper], 0.0, grid_size) / grid_size


def _flap_base_contour(
    rectangle, grid_size: int, others: Sequence[Any]
) -> np.ndarray:
  """On-paper points that fold onto a flap's base joint and stay with it.

  The full rectangle boundary (`_flap_contour(rectangle, 1.0, ...)`) folds
  onto the base joint on the base, but a shaping fold through the flap's base
  (`rotation_3d` anchors its cut line on that boundary) carries part of the
  boundary along with the flap: its mean moved towards the rotated flap and
  biased every measured direction from that joint (5-17 degrees on a probe
  whose true error was below 1.5 degrees). The paper just outside the
  rectangle (rivers, hinge corridors) folds to within `_BASE_RING_MARGIN` of
  the joint and is not creased by tools that act on the flap. Points inside
  other flap rectangles are skipped (they move with those flaps).

  Args:
    rectangle: Packing flap rectangle (packing grid units).
    grid_size: Packing grid size.
    others: The other flap rectangles.

  Returns:
    (N, 2) points on the unit paper; the full boundary when fewer than
    `_MIN_BASE_RING_POINTS` points qualify.
  """
  margin = _BASE_RING_MARGIN
  points = _rectangle_contour(
      rectangle.x_start - margin,
      rectangle.x_start + rectangle.x_len + margin,
      rectangle.y_start - margin,
      rectangle.y_start + rectangle.y_len + margin,
      _SAMPLE_STEP,
  )
  keep = np.all(
      (points >= -_PAPER_TOLERANCE) & (points <= grid_size + _PAPER_TOLERANCE),
      axis=1,
  )
  for other in others:
    keep &= ~(
        (points[:, 0] > other.x_start)
        & (points[:, 0] < other.x_start + other.x_len)
        & (points[:, 1] > other.y_start)
        & (points[:, 1] < other.y_start + other.y_len)
    )
  if int(keep.sum()) < _MIN_BASE_RING_POINTS:
    return _flap_contour(rectangle, 1.0, grid_size)
  return np.clip(points[keep], 0.0, grid_size) / grid_size


@dataclasses.dataclass(frozen=True)
class Layout:
  """Paper sample points for joints and stick cross-sections.

  Attributes:
    joints: Joint -> (N, 2) unit-paper points whose folded mean is the joint.
    sections: `<stick>@<fraction>` -> (N, 2) points across the flap stick at
      that fraction from its root-side joint (`SECTION_FRACTIONS`).
    chains: Merged flap name -> its sticks, root side first.
    grid_size: Packing grid size.
    grid_units_per_tree_unit: Packing scale (flap rectangle radius / length).
    flap_rectangles: Flap rectangles on the unit paper (x0, x1, y0, y1).
  """

  joints: Mapping[str, np.ndarray]
  sections: Mapping[str, np.ndarray]
  chains: Mapping[str, tuple[str, ...]]
  grid_size: int
  grid_units_per_tree_unit: float
  flap_rectangles: tuple[tuple[float, float, float, float], ...]


_LAYOUT_CACHE: dict[tuple[int, int], tuple[Any, Any, Layout]] = {}
_MAX_LAYOUT_CACHE = 16


def paper_layout(
    tree: stick_figure_lib.StickFigure,
    packing: packing_output_lib.PackingOutput,
) -> Layout:
  """Locates joints and stick cross-sections on the paper (see `Layout`).

  Joints touched only by rivers are not located.

  Args:
    tree: The unmerged stick figure.
    packing: The packing output of the merged tree.

  Returns:
    The layout.
  """
  cache_key = (id(tree), id(packing))
  cached = _LAYOUT_CACHE.get(cache_key)
  # Identity-checked: a long-lived workbench session can see a new tree or
  # packing reuse a freed object's id.
  if cached is not None and cached[0] is tree and cached[1] is packing:
    return cached[2]
  if packing.packing_cp is None or not packing.flap_rectangles:
    raise ValueError('Packing output has no crease pattern or flap rectangles.')
  grid_size = packing.packing_cp.grid_size
  _, merged_names = tree.merge_linear_chains()
  topo = topology(tree)
  rank = {name: index for index, name in enumerate(topo.order)}
  joints: dict[str, list[np.ndarray]] = {}
  sections: dict[str, np.ndarray] = {}
  chains: dict[str, tuple[str, ...]] = {}
  scales = []
  rectangles = []
  for rectangle in packing.flap_rectangles:
    chain = list(merged_names.get(rectangle.name, [rectangle.name]))
    if not chain or any(name not in tree for name in chain):
      continue
    total = float(sum(tree[name].length for name in chain))
    if total <= 0:
      continue
    rectangles.append((
        rectangle.x_start / grid_size,
        (rectangle.x_start + rectangle.x_len) / grid_size,
        rectangle.y_start / grid_size,
        (rectangle.y_start + rectangle.y_len) / grid_size,
    ))
    scales.append(min(rectangle.x_len, rectangle.y_len) / 2.0 / total)
    chains[rectangle.name] = tuple(sorted(chain, key=lambda n: rank[n]))
    # Walk from the tip towards the root: distance from the tip grows.
    distal_first = sorted(chain, key=lambda name: -rank[name])
    tip = topo.distal[distal_first[0]]
    joints.setdefault(tip, []).append(_flap_contour(rectangle, 0.0, grid_size))
    distance = 0.0
    others = [r for r in packing.flap_rectangles if r is not rectangle]
    for index, name in enumerate(distal_first):
      length = tree[name].length
      for fraction in SECTION_FRACTIONS:
        # `fraction` runs from the root-side joint towards the tip.
        from_tip = distance + (1.0 - fraction) * length
        sections[f'{name}@{fraction:g}'] = _flap_contour(
            rectangle, from_tip / total, grid_size
        )
      distance += length
      if index == len(distal_first) - 1:
        contour = _flap_base_contour(rectangle, grid_size, others)
      else:
        contour = _flap_contour(rectangle, distance / total, grid_size)
      joints.setdefault(topo.proximal[name], []).append(contour)
  scale_grid = float(np.median(scales)) if scales else 1.0
  # Sample cross-sections along non-root river sticks using the outer L-infinity
  # contour around their distal subtree of flap rectangles.
  rect_by_chain_root = {
      chains[r.name][0]: r for r in packing.flap_rectangles if r.name in chains
  }
  for river_name in topo.order:
    if river_name == tree.root or not tree[river_name].is_river:
      continue
    sub_rects: list[tuple[Any, float]] = []
    for chain_root, rect in rect_by_chain_root.items():
      cur = chain_root
      inter_river_len = 0.0
      while cur is not None and cur != river_name:
        parent = topo.parent.get(cur)
        if (
            parent is not None
            and parent != river_name
            and tree[parent].is_river
        ):
          inter_river_len += float(tree[parent].length)
        cur = parent
      if cur == river_name:
        sub_rects.append((rect, inter_river_len))
    if not sub_rects:
      continue
    sub_rect_set = {id(r) for r, _ in sub_rects}
    outside_rects = [
        r for r in packing.flap_rectangles if id(r) not in sub_rect_set
    ]
    river_len = float(tree[river_name].length)
    for fraction in (0.0,) + SECTION_FRACTIONS:
      from_distal = (1.0 - fraction) * river_len
      expanded_boxes = [
          (
              rect,
              (from_distal + inter_len) * scale_grid + _BASE_RING_MARGIN,
          )
          for rect, inter_len in sub_rects
      ]
      parts: list[np.ndarray] = []
      for rect, expand in expanded_boxes:
        pts = _rectangle_contour(
            rect.x_start - expand,
            rect.x_start + rect.x_len + expand,
            rect.y_start - expand,
            rect.y_start + rect.y_len + expand,
            _SAMPLE_STEP,
        )
        keep = np.all(
            (pts >= -_PAPER_TOLERANCE) & (pts <= grid_size + _PAPER_TOLERANCE),
            axis=1,
        )
        for other_rect, other_exp in expanded_boxes:
          if other_rect is rect:
            continue
          keep &= ~(
              (pts[:, 0] > other_rect.x_start - other_exp + _PAPER_TOLERANCE)
              & (
                  pts[:, 0]
                  < other_rect.x_start
                  + other_rect.x_len
                  + other_exp
                  - _PAPER_TOLERANCE
              )
              & (pts[:, 1] > other_rect.y_start - other_exp + _PAPER_TOLERANCE)
              & (
                  pts[:, 1]
                  < other_rect.y_start
                  + other_rect.y_len
                  + other_exp
                  - _PAPER_TOLERANCE
              )
          )
        for outside in outside_rects:
          keep &= ~(
              (pts[:, 0] > outside.x_start)
              & (pts[:, 0] < outside.x_start + outside.x_len)
              & (pts[:, 1] > outside.y_start)
              & (pts[:, 1] < outside.y_start + outside.y_len)
          )
        if np.any(keep):
          parts.append(np.clip(pts[keep], 0.0, grid_size) / grid_size)
      if parts:
        combined_pts = np.concatenate(parts)
        if len(combined_pts) >= _MIN_BASE_RING_POINTS:
          sections[f'{river_name}@{fraction:g}'] = combined_pts
  result = Layout(
      joints={
          joint: np.concatenate(parts)
          for joint, parts in joints.items()
          if sum(len(part) for part in parts)
      },
      sections={k: v for k, v in sections.items() if len(v)},
      chains=chains,
      grid_size=grid_size,
      grid_units_per_tree_unit=scale_grid,
      flap_rectangles=tuple(rectangles),
  )
  if len(_LAYOUT_CACHE) >= _MAX_LAYOUT_CACHE:
    _LAYOUT_CACHE.pop(next(iter(_LAYOUT_CACHE)))
  _LAYOUT_CACHE[cache_key] = (tree, packing, result)
  return result


# ------------------------------------------------------------------------------
# Folded model.
# ------------------------------------------------------------------------------


_FOLD3D_CACHE: dict[str, tuple[Any, float]] = {}
_MAX_FOLD3D_CACHE = 64


_MODEL_PATTERN_CACHE: dict[
    tuple[int, int], tuple[crease_pattern_lib.CreasePattern, 'FoldedModel']
] = {}
_MAX_MODEL_PATTERN_CACHE = 64


@dataclasses.dataclass(frozen=True)
class FoldedModel:
  """3-D geometry of a folded crease pattern, on the unit paper.

  Attributes:
    vertices_2d: (V, 2) crease pattern vertices on the unit paper.
    vertices_3d: (V, 3) folded vertex positions (paper side = 1).
    faces: Vertex indices per face.
    matrices: (F, 4, 4) face transforms acting on unit-paper coordinates.
    vertex_error: The folder's mean vertex (edge strain) error, percent.
    seconds: Wall time of the fold.
  """

  vertices_2d: np.ndarray
  vertices_3d: np.ndarray
  faces: tuple[np.ndarray, ...]
  matrices: np.ndarray
  vertex_error: float
  seconds: float
  non_ff_vertices: int = 0

  @classmethod
  def from_fold3d(cls, fold3d: Any, seconds: float = 0.0) -> 'FoldedModel':
    """Wraps the folder's FOLD3D result (`folder_3d.get_fold`)."""
    vertices_2d = np.asarray(fold3d['vertices_coords_2d'], dtype=float)
    scale = float(vertices_2d.max()) or 1.0
    raw_matrices = fold3d['simulator_auxs'].get('matrices')
    if raw_matrices is None:
      raise ValueError('The folder returned no face transforms.')
    # Conjugate the transforms with the scaling so they act on unit paper.
    to_raw = np.diag([scale, scale, 1.0, 1.0])
    from_raw = np.diag([1.0 / scale, 1.0 / scale, 1.0 / scale, 1.0])
    matrices = np.stack([
        from_raw @ (np.eye(4) if m is None else np.asarray(m)) @ to_raw
        for m in raw_matrices
    ])
    return cls(
        vertices_2d=vertices_2d / scale,
        vertices_3d=np.asarray(fold3d['vertices_coords'], dtype=float) / scale,
        faces=tuple(
            np.asarray(face, dtype=int) for face in fold3d['faces_vertices']
        ),
        matrices=matrices,
        vertex_error=float(fold3d['simulator_auxs'].get('vertex_errors', 0.0)),
        seconds=seconds,
    )

  @classmethod
  def from_svg(cls, svg_string: str) -> 'FoldedModel':
    """Folds an SVG crease pattern with the deterministic folder."""
    cached = _FOLD3D_CACHE.get(svg_string)
    if cached is not None:
      fold3d, seconds = cached
      return cls.from_fold3d(fold3d, seconds)
    start = time.perf_counter()
    fold3d = folder_3d.get_fold(svg_string, simulation_tolerance=10.0)
    seconds = time.perf_counter() - start
    if len(_FOLD3D_CACHE) >= _MAX_FOLD3D_CACHE:
      _FOLD3D_CACHE.pop(next(iter(_FOLD3D_CACHE)))
    _FOLD3D_CACHE[svg_string] = (fold3d, seconds)
    return cls.from_fold3d(fold3d, seconds)

  @classmethod
  def from_pattern(
      cls, pattern: crease_pattern_lib.CreasePattern
  ) -> 'FoldedModel':
    """Folds a `CreasePattern` deterministically via `from_svg`."""
    key = (id(pattern), len(pattern.creases))
    cached = _MODEL_PATTERN_CACHE.get(key)
    # `id()` is reused once a pattern is freed (candidate loops create and
    # drop many patterns with equal crease counts): keep the pattern alive in
    # the cache and check identity, else a new pattern gets an old model.
    if cached is not None and cached[0] is pattern:
      return cached[1]
    _, _, non_ff = pattern.vertex_flat_foldability_percentage()
    model = cls.from_svg(svg_parser.to_svg(pattern, ndigits=8))
    if len(non_ff) != model.non_ff_vertices:
      model = dataclasses.replace(model, non_ff_vertices=len(non_ff))
    if len(_MODEL_PATTERN_CACHE) >= _MAX_MODEL_PATTERN_CACHE:
      _MODEL_PATTERN_CACHE.pop(next(iter(_MODEL_PATTERN_CACHE)))
    _MODEL_PATTERN_CACHE[key] = (pattern, model)
    return model

  @property
  def _face_boxes(self) -> np.ndarray:
    cached = self.__dict__.get('_boxes')
    if cached is None:
      cached = np.array([
          [
              self.vertices_2d[face, 0].min(),
              self.vertices_2d[face, 0].max(),
              self.vertices_2d[face, 1].min(),
              self.vertices_2d[face, 1].max(),
          ]
          for face in self.faces
      ])
      object.__setattr__(self, '_boxes', cached)
    return cached

  def _face_path(self, index: int) -> mpath.Path:
    paths = self.__dict__.setdefault('_paths', {})
    if index not in paths:
      paths[index] = mpath.Path(self.vertices_2d[self.faces[index]])
    return paths[index]

  def locate_faces(self, points: np.ndarray) -> np.ndarray:
    """Index of the face containing each unit-paper point (-1 if none).

    Args:
      points: (N, 2) points on the unit paper.

    Returns:
      (N,) integer face indices.
    """
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    result = np.full(len(points), -1, dtype=int)
    if not len(points):
      return result
    boxes = self._face_boxes
    tolerance = 1e-9
    inside_box = (
        (points[:, None, 0] >= boxes[None, :, 0] - tolerance)
        & (points[:, None, 0] <= boxes[None, :, 1] + tolerance)
        & (points[:, None, 1] >= boxes[None, :, 2] - tolerance)
        & (points[:, None, 1] <= boxes[None, :, 3] + tolerance)
    )
    for index in range(len(points)):
      candidates = [int(f) for f in np.flatnonzero(inside_box[index])]
      for face_index in candidates:
        path = self._face_path(face_index)
        if path.contains_point(points[index], radius=1e-9):
          result[index] = face_index
          break
      else:
        if len(candidates):
          # On an edge between faces: any adjacent face gives the same point.
          distances = [
              np.min(
                  np.linalg.norm(
                      self.vertices_2d[self.faces[f]] - points[index], axis=1
                  )
              )
              for f in candidates
          ]
          result[index] = candidates[int(np.argmin(distances))]
    return result

  def locate(self, points: np.ndarray) -> np.ndarray:
    """Folded 3-D positions of unit-paper points (NaN where not found).

    Args:
      points: (N, 2) points on the unit paper.

    Returns:
      (N, 3) folded positions.
    """
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    faces = self.locate_faces(points)
    result = np.full((len(points), 3), np.nan)
    found = faces >= 0
    if np.any(found):
      homogeneous = np.concatenate(
          [points, np.zeros((len(points), 1)), np.ones((len(points), 1))],
          axis=1,
      )
      transformed = np.einsum(
          'nij,nj->ni', self.matrices[faces[found]], homogeneous[found]
      )
      result[found] = transformed[:, :3]
    return result

  def locate_point(
      self, point: tuple[float, float] | np.ndarray
  ) -> np.ndarray | None:
    """Folded 3-D position of a single unit-paper point, or None."""
    pts = np.asarray(point, dtype=float).reshape(1, 2)
    loc = self.locate(pts)[0]
    if np.any(np.isnan(loc)):
      return None
    return loc

  def locate_groups(
      self, groups: Mapping[str, np.ndarray]
  ) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Mean folded position and RMS spread of each named point group."""
    names = [name for name in groups if len(groups[name])]
    if not names:
      return {}, {}
    located = self.locate(np.concatenate([groups[name] for name in names]))
    means, spreads = {}, {}
    offset = 0
    for name in names:
      count = len(groups[name])
      points = located[offset : offset + count]
      offset += count
      points = points[~np.isnan(points).any(axis=1)]
      if not len(points):
        continue
      mean = points.mean(axis=0)
      means[name] = mean
      spreads[name] = float(np.sqrt(((points - mean) ** 2).sum(axis=1).mean()))
    return means, spreads


# ------------------------------------------------------------------------------
# Alignment.
# ------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Similarity:
  """`target ~ scale * rotation @ source + translation`."""

  rotation: np.ndarray
  scale: float
  translation: np.ndarray
  reflected: bool

  def apply(self, points: np.ndarray) -> np.ndarray:
    return self.scale * np.asarray(points) @ self.rotation.T + self.translation


def fit_similarity(
    source: np.ndarray,
    target: np.ndarray,
    *,
    weights: np.ndarray | None = None,
    allow_reflection: bool = True,
) -> Similarity:
  """Weighted least-squares similarity transform (Umeyama).

  Args:
    source: (N, 3) points to move.
    target: (N, 3) points to match.
    weights: Optional (N,) non-negative weights.
    allow_reflection: Also consider improper rotations (mirror images) and keep
      the better fit.

  Returns:
    The best similarity.
  """
  weights = np.ones(len(source)) if weights is None else np.asarray(weights)
  weights = weights / weights.sum()
  source_mean = weights @ source
  target_mean = weights @ target
  source_centered = source - source_mean
  target_centered = target - target_mean
  covariance = (target_centered * weights[:, None]).T @ source_centered
  u, singular, vt = np.linalg.svd(covariance)
  source_variance = float(weights @ (source_centered**2).sum(axis=1))
  det_sign = 1.0 if np.linalg.det(u @ vt) >= 0 else -1.0
  options = [(np.diag([1.0, 1.0, det_sign]), False)]
  if allow_reflection:
    options.append((np.diag([1.0, 1.0, -det_sign]), True))
  best, best_residual = None, math.inf
  for sign_matrix, reflected in options:
    rotation = u @ sign_matrix @ vt
    scale = (
        float(np.sum(singular * np.diag(sign_matrix))) / source_variance
        if source_variance > 0
        else 0.0
    )
    translation = target_mean - scale * rotation @ source_mean
    candidate = Similarity(rotation, scale, translation, reflected)
    residual = float(
        weights @ ((candidate.apply(source) - target) ** 2).sum(axis=1)
    )
    if residual < best_residual - 1e-12:
      best, best_residual = candidate, residual
  assert best is not None
  return best


def fit_similarity_robust(
    source: np.ndarray,
    target: np.ndarray,
    size: float,
    *,
    allow_reflection: bool = True,
) -> tuple[Similarity, np.ndarray]:
  """Similarity fit with Cauchy reweighting (badly posed flaps don't tilt it).

  Args:
    source: (N, 3) points to move.
    target: (N, 3) points to match.
    size: Figure size; residuals are measured in multiples of it.
    allow_reflection: As in `fit_similarity`.

  Returns:
    The fit and the final per-point weights.
  """
  weights = np.ones(len(source))
  fit = fit_similarity(source, target, allow_reflection=allow_reflection)
  for _ in range(_ROBUST_ITERATIONS):
    residual = np.linalg.norm(fit.apply(source) - target, axis=1) / size
    weights = 1.0 / (1.0 + (residual / _ROBUST_SCALE) ** 2)
    fit = fit_similarity(
        source, target, weights=weights, allow_reflection=allow_reflection
    )
  return fit, weights


def render_rotation(report: Mapping[str, Any]) -> np.ndarray:
  """Proper 3x3 rotation that turns the folded model into the figure's frame.

  Apply it to folded coordinates (e.g. the folder's face polygons) before
  rendering with the stick figure's camera views: the model then stands as
  designed (z up, azimuth 0 forward) instead of in the folder's frame.

  Args:
    report: A fidelity report.

  Returns:
    The rotation matrix.
  """
  return np.asarray(report['alignment']['render_rotation'], dtype=float)


# ------------------------------------------------------------------------------
# Measurement: the geometry everything else is computed from.
# ------------------------------------------------------------------------------


def _infer_symmetric_stick(name: str, all_names: set[str]) -> str | None:
  """Infers bilateral mirror partner stick name if not explicitly set."""
  if (
      name.startswith(('major_', 'minor_'))
      or 'major_' in name
      or 'minor_' in name
  ):
    return None
  candidates = []
  if re.search(r'([_])L([0-9]*)$', name):
    candidates.append(re.sub(r'([_])L([0-9]*)$', r'\g<1>R\2', name))
  elif re.search(r'([_])R([0-9]*)$', name):
    candidates.append(re.sub(r'([_])R([0-9]*)$', r'\g<1>L\2', name))
  if re.search(r'([_])l([0-9]*)$', name):
    candidates.append(re.sub(r'([_])l([0-9]*)$', r'\g<1>r\2', name))
  elif re.search(r'([_])r([0-9]*)$', name):
    candidates.append(re.sub(r'([_])r([0-9]*)$', r'\g<1>l\2', name))
  if '_L_' in name:
    candidates.append(name.replace('_L_', '_R_', 1))
  elif '_R_' in name:
    candidates.append(name.replace('_R_', '_L_', 1))
  if '_l_' in name:
    candidates.append(name.replace('_l_', '_r_', 1))
  elif '_r_' in name:
    candidates.append(name.replace('_r_', '_l_', 1))
  if re.search(r'_L([0-9]+)_', name):
    candidates.append(re.sub(r'_L([0-9]+)_', r'_R\1_', name))
  elif re.search(r'_R([0-9]+)_', name):
    candidates.append(re.sub(r'_R([0-9]+)_', r'_L\1_', name))
  if re.search(r'_l([0-9]+)_', name):
    candidates.append(re.sub(r'_l([0-9]+)_', r'_r\1_', name))
  elif re.search(r'_r([0-9]+)_', name):
    candidates.append(re.sub(r'_r([0-9]+)_', r'_l\1_', name))
  if 'left' in name:
    candidates.append(name.replace('left', 'right', 1))
  elif 'right' in name:
    candidates.append(name.replace('right', 'left', 1))
  for cand in candidates:
    if (
        cand in all_names
        and cand != name
        and not ('major_' in cand or 'minor_' in cand)
    ):
      return cand
  return None


def measure(
    tree: stick_figure_lib.StickFigure,
    layout: Layout,
    model: FoldedModel,
) -> dict[str, Any]:
  """Folded and target geometry of every located joint and section.

  Args:
    tree: The unmerged stick figure.
    layout: `paper_layout(tree, packing)`.
    model: The folded model.

  Returns:
    JSON-friendly geometry: `sticks` (tree structure), `joints` (folded
    position in the model frame and target position, tree units),
    `sections` (folded position and width in tree units), `chains`, the
    model's principal extents and the body thickness.
  """
  topo = topology(tree)
  target = stick_figure_joint_positions(tree)
  folded, _ = model.locate_groups(layout.joints)
  section_means, section_spreads = model.locate_groups(layout.sections)
  to_tree_units = layout.grid_size / layout.grid_units_per_tree_unit
  body = model.vertices_3d[_body_mask(model, layout)]
  body_extents = _principal_extents(body)
  model_extents = _principal_extents(model.vertices_3d)
  return {
      'root': tree.root,
      'root_normal': _root_normal(tree, layout, model),
      'sticks': {
          name: {
              'length': float(tree[name].length),
              'width': float(tree[name].width),
              'proximal': topo.proximal[name],
              'distal': topo.distal[name],
              'parent': topo.parent[name],
              'is_river': bool(tree[name].is_river),
              'symmetric': (
                  tree[name].symmetric_stick_name
                  or _infer_symmetric_stick(name, set(topo.order))
              ),
          }
          for name in topo.order
      },
      'chains': {k: list(v) for k, v in layout.chains.items()},
      'joints': {
          name: {
              'folded': (
                  _rounded_list(folded[name] * to_tree_units)
                  if name in folded
                  else None
              ),
              'target': _rounded_list(target[name]),
          }
          for name in target
      },
      'sections': {
          name: {
              'folded': _rounded_list(section_means[name] * to_tree_units),
              'width': round(
                  _WIDTH_PER_SPREAD * section_spreads[name] * to_tree_units, 4
              ),
          }
          for name in section_means
      },
      'model_extents': _rounded_list(model_extents * to_tree_units),
      'body_extents': _rounded_list(body_extents * to_tree_units),
  }


def _rounded_list(values: np.ndarray) -> list[float]:
  return [round(float(v), 5) for v in values]


def _root_normal(
    tree: stick_figure_lib.StickFigure, layout: Layout, model: FoldedModel
) -> list[float] | None:
  """Dominant paper normal (model frame, sign arbitrary) at the root joints.

  Folded layers alternate orientation, so the normals are combined as an axis
  (principal direction of their outer products), not averaged.

  Args:
    tree: The stick figure.
    layout: The paper layout.
    model: The folded model.

  Returns:
    A unit vector, or None when the root joints are not on the paper.
  """
  root = tree[tree.root]
  parts = [
      layout.joints[j] for j in (root.start, root.end) if j in layout.joints
  ]
  if not parts:
    return None
  faces = model.locate_faces(np.concatenate(parts))
  faces = faces[faces >= 0]
  if not len(faces):
    return None
  normals = model.matrices[faces][:, :3, 2]
  normals /= np.linalg.norm(normals, axis=1, keepdims=True)
  _, vectors = np.linalg.eigh(normals.T @ normals)
  return _rounded_list(vectors[:, -1])


def _principal_extents(points: np.ndarray) -> np.ndarray:
  """Standard deviations along the principal axes, descending."""
  if len(points) < 3:
    return np.zeros(3)
  centered = points - points.mean(axis=0)
  values = np.linalg.svd(centered, compute_uv=False) / math.sqrt(len(points))
  return np.pad(values, (0, max(0, 3 - len(values))))[:3]


def _body_mask(model: FoldedModel, layout: Layout) -> np.ndarray:
  """Vertices outside every flap rectangle (rivers, root region, gussets)."""
  x, y = model.vertices_2d[:, 0], model.vertices_2d[:, 1]
  inside_any = np.zeros(len(x), dtype=bool)
  eps = 1e-6
  for x0, x1, y0, y1 in layout.flap_rectangles:
    inside_any |= (
        (x > x0 + eps) & (x < x1 - eps) & (y > y0 + eps) & (y < y1 - eps)
    )
  return ~inside_any


# ------------------------------------------------------------------------------
# Components and score.
# ------------------------------------------------------------------------------


def _angle_deg(first: np.ndarray, second: np.ndarray) -> float:
  denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
  if denominator <= 0:
    return 180.0
  cosine = float(np.dot(first, second)) / denominator
  return math.degrees(math.acos(min(1.0, max(-1.0, cosine))))


def _turns(points: list[np.ndarray]) -> list[float]:
  """Turning angles (degrees) along a polyline; collapsed segments skipped."""
  segments = [b - a for a, b in zip(points, points[1:])]
  segments = [s for s in segments if np.linalg.norm(s) > 1e-9]
  return [_angle_deg(a, b) for a, b in zip(segments, segments[1:])]


def _flatness(points: np.ndarray) -> float:
  """Third over first principal extent of a point set (0 = flat or a bar)."""
  extents = _principal_extents(points)
  return float(extents[2] / extents[0]) if extents[0] > 0 else 0.0


def _triangulate_faces(faces: Sequence[np.ndarray]) -> np.ndarray:
  """Fan-triangulates polygon faces into a `(T, 3)` integer index array."""
  tris: list[tuple[int, int, int]] = []
  for face in faces:
    if len(face) < 3:
      continue
    v0 = int(face[0])
    for idx in range(1, len(face) - 1):
      tris.append((v0, int(face[idx]), int(face[idx + 1])))
  if not tris:
    return np.zeros((0, 3), dtype=int)
  return np.asarray(tris, dtype=int)


def _rasterize_triangles_2d(
    pts_2d: np.ndarray,
    tris: np.ndarray,
    grid_size: int,
    u_min: float,
    u_max: float,
    v_min: float,
    v_max: float,
) -> np.ndarray:
  """Rasterizes 2-D triangles onto a `(grid_size, grid_size)` boolean mask."""
  mask = np.zeros((grid_size, grid_size), dtype=bool)
  if not len(tris) or u_max <= u_min + 1e-9 or v_max <= v_min + 1e-9:
    return mask
  scale_u = (grid_size - 1) / (u_max - u_min)
  scale_v = (grid_size - 1) / (v_max - v_min)
  pu = (pts_2d[:, 0] - u_min) * scale_u
  pv = (pts_2d[:, 1] - v_min) * scale_v
  for i0, i1, i2 in tris:
    u0, u1, u2 = float(pu[i0]), float(pu[i1]), float(pu[i2])
    v0, v1, v2 = float(pv[i0]), float(pv[i1]), float(pv[i2])
    c_min = max(0, int(math.floor(min(u0, u1, u2))))
    c_max = min(grid_size - 1, int(math.ceil(max(u0, u1, u2))))
    r_min = max(0, int(math.floor(min(v0, v1, v2))))
    r_max = min(grid_size - 1, int(math.ceil(max(v0, v1, v2))))
    if c_min > c_max or r_min > r_max:
      continue
    area2 = (u1 - u0) * (v2 - v0) - (v1 - v0) * (u2 - u0)
    if abs(area2) < 0.25:
      # Edge-on or sub-pixel sliver: mark pixel polyline along longest edge so
      # thin edge-on sheets still register 1-pixel presence without vanishing.
      steps = max(2, min(grid_size, int(max(c_max - c_min, r_max - r_min)) + 2))
      ts = np.linspace(0.0, 1.0, steps)
      for ua, va, ub, vb in ((u0, v0, u1, v1), (u1, v1, u2, v2)):
        cs = np.clip(np.round(ua + ts * (ub - ua)).astype(int), 0, grid_size - 1)
        rs = np.clip(np.round(va + ts * (vb - va)).astype(int), 0, grid_size - 1)
        mask[rs, cs] = True
      continue
    cols = np.arange(c_min, c_max + 1, dtype=float)[None, :]
    rows = np.arange(r_min, r_max + 1, dtype=float)[:, None]
    e0 = (u1 - u0) * (rows - v0) - (v1 - v0) * (cols - u0)
    e1 = (u2 - u1) * (rows - v1) - (v2 - v1) * (cols - u1)
    e2 = (u0 - u2) * (rows - v2) - (v0 - v2) * (cols - u2)
    tol = 0.35 * abs(area2) / max(abs(u1 - u0) + abs(v1 - v0), 1.0)
    if area2 > 0:
      inside = (e0 >= -tol) & (e1 >= -tol) & (e2 >= -tol)
    else:
      inside = (e0 <= tol) & (e1 <= tol) & (e2 <= tol)
    mask[r_min : r_max + 1, c_min : c_max + 1] |= inside
  return mask


def _capsule_mask_2d(
    uu: np.ndarray,
    vv: np.ndarray,
    p0: np.ndarray,
    p1: np.ndarray,
    radius: float,
) -> np.ndarray:
  """Boolean mask of grid coordinates `(uu, vv)` within `radius` of `[p0, p1]`."""
  du = float(p1[0] - p0[0])
  dv = float(p1[1] - p0[1])
  seg_sq = du * du + dv * dv
  ru = uu - float(p0[0])
  rv = vv - float(p0[1])
  if seg_sq <= 1e-12:
    dist_sq = ru * ru + rv * rv
  else:
    t = np.clip((ru * du + rv * dv) / seg_sq, 0.0, 1.0)
    dist_sq = (ru - t * du) ** 2 + (rv - t * dv) ** 2
  return dist_sq <= radius * radius


def _interior_hole_fraction(mask: np.ndarray) -> float:
  """Fraction of enclosed interior void pixels inside `mask`'s silhouette."""
  filled = int(mask.sum())
  if filled < 9:
    return 0.0
  empty = ~mask
  outside = np.zeros_like(mask, dtype=bool)
  outside[0, :] = empty[0, :]
  outside[-1, :] = empty[-1, :]
  outside[:, 0] = empty[:, 0]
  outside[:, -1] = empty[:, -1]
  while True:
    expanded = outside.copy()
    expanded[1:, :] |= outside[:-1, :]
    expanded[:-1, :] |= outside[1:, :]
    expanded[:, 1:] |= outside[:, :-1]
    expanded[:, :-1] |= outside[:, 1:]
    expanded &= empty
    if np.array_equal(expanded, outside):
      break
    outside = expanded
  holes = int((empty & ~outside).sum())
  return float(holes / max(filled + holes, 1))


def _silhouette_and_body_metrics(
    geometry: Mapping[str, Any],
    aligned: Mapping[str, np.ndarray],
    target: Mapping[str, np.ndarray],
    fit: Similarity,
    model: FoldedModel,
    to_tree_units: float,
) -> dict[str, Any]:
  """Multi-view 2-D mesh silhouette recall/precision, hole ratio, and body breadth."""
  sticks = geometry['sticks']
  mesh_v = fit.apply(model.vertices_3d * to_tree_units)
  tris = _triangulate_faces(model.faces)
  target_pts = np.array(list(target.values()), dtype=float)
  max_w = max((float(s['width']) for s in sticks.values()), default=1.0)
  planes = (('xy', 0, 1), ('xz', 0, 2), ('yz', 1, 2))
  grid_size = _SILHOUETTE_GRID_SIZE

  per_stick_plane_recall: dict[str, list[float]] = {n: [] for n in sticks}
  precision_scores: list[float] = []
  side_hole_fraction = 0.0

  for plane_name, ax_u, ax_v in planes:
    t_uv = target_pts[:, [ax_u, ax_v]]
    span_u = float(t_uv[:, 0].max() - t_uv[:, 0].min())
    span_v = float(t_uv[:, 1].max() - t_uv[:, 1].min())
    span = max(span_u, span_v, 1.0)
    pad = 0.35 * max_w + 0.15 * span
    u_min = float(t_uv[:, 0].min()) - pad
    u_max = float(t_uv[:, 0].max()) + pad
    v_min = float(t_uv[:, 1].min()) - pad
    v_max = float(t_uv[:, 1].max()) + pad
    mesh_mask = _rasterize_triangles_2d(
        mesh_v[:, [ax_u, ax_v]], tris, grid_size, u_min, u_max, v_min, v_max
    )
    if plane_name == 'xz':
      side_hole_fraction = _interior_hole_fraction(mesh_mask)

    us = np.linspace(u_min, u_max, grid_size)
    vs = np.linspace(v_min, v_max, grid_size)
    uu, vv = np.meshgrid(us, vs)
    pixel_size = max(
        (u_max - u_min) / max(grid_size - 1, 1),
        (v_max - v_min) / max(grid_size - 1, 1),
    )

    union_loose = np.zeros((grid_size, grid_size), dtype=bool)
    for name, stick in sticks.items():
      t0_3d = target[stick['proximal']]
      t1_3d = target[stick['distal']]
      t0 = t0_3d[[ax_u, ax_v]]
      t1 = t1_3d[[ax_u, ax_v]]
      loose_radius = max(0.55 * float(stick['width']), 0.35 + 1.5 * pixel_size)
      union_loose |= _capsule_mask_2d(uu, vv, t0, t1, loose_radius)

      near, far = stick['proximal'], stick['distal']
      if (
          near in aligned
          and far in aligned
          and float(np.linalg.norm(aligned[far] - aligned[near]))
          >= 0.25 * float(np.linalg.norm(t1_3d - t0_3d))
      ):
        p0_3d = aligned[near]
        p1_3d = aligned[far]
      else:
        p0_3d = t0_3d
        p1_3d = t1_3d
      p0 = p0_3d[[ax_u, ax_v]]
      p1 = p1_3d[[ax_u, ax_v]]
      len_3d = float(np.linalg.norm(p1_3d - p0_3d))
      len_2d = float(np.linalg.norm(p1 - p0))
      if float(stick['width']) >= 2.0:
        radius = max(0.38 * float(stick['width']), 0.85 * pixel_size)
      else:
        radius = max(0.20 * float(stick['width']), 0.45 * pixel_size)
      if len_3d > 1e-6 and len_2d < 0.35 * len_3d:
        continue
      cap = _capsule_mask_2d(uu, vv, p0, p1, radius)
      cap_sum = int(cap.sum())
      if cap_sum > 0:
        per_stick_plane_recall[name].append(
            float((mesh_mask & cap).sum() / cap_sum)
        )

    mesh_sum = int(mesh_mask.sum())
    if mesh_sum > 0:
      precision_scores.append(float((mesh_mask & union_loose).sum() / mesh_sum))

  # For flat plates (wings, hoods, frills), one plane shows the broad face and
  # the orthogonal plane is edge-on, so `max` across non-foreshortened planes
  # credits genuine broad plates while catching 1-D wires/spikes (which are thin
  # in ALL planes!). For body rivers (`is_river` or `root`), require breadth in
  # every non-foreshortened silhouette (`min(recalls)`).
  stick_coverages: dict[str, float] = {}
  weighted_sum = 0.0
  weight_total = 0.0
  wide_coverages: list[float] = []
  body_breadths: list[float] = []
  for name, stick in sticks.items():
    recalls = per_stick_plane_recall[name]
    if not recalls:
      continue
    is_body = bool(stick['is_river'] or name == geometry['root'])
    if is_body:
      cov = float(min(recalls))
    else:
      cov = float(max(recalls))
    full_thresh = 0.80 if float(stick['width']) >= 2.0 else 0.45
    norm_cov = min(1.0, cov / full_thresh)
    stick_coverages[name] = round(norm_cov, 3)
    stick_w = float(stick['length']) * (2.0 if stick['width'] >= 2.0 else 1.0)
    weighted_sum += stick_w * norm_cov
    weight_total += stick_w
    if stick['width'] >= 2.0:
      wide_coverages.append(norm_cov)
    if is_body:
      # Local cross-sectional perpendicular breadth of body/river mid-segments
      # (`t in [0.25, 0.75]`, avoiding leg attachments at the end hubs).
      near, far = stick['proximal'], stick['distal']
      if near in aligned and far in aligned:
        a = aligned[near]
        b = aligned[far]
        axis = b - a
        seg_len = float(np.linalg.norm(axis))
        if seg_len > 1e-6:
          u_axis = axis / seg_len
          rel = mesh_v - a[None, :]
          t_proj = rel @ u_axis
          perp = np.linalg.norm(rel - t_proj[:, None] * u_axis[None, :], axis=1)
          near_seg = (
              (t_proj >= 0.25 * seg_len)
              & (t_proj <= 0.75 * seg_len)
              & (perp <= max(0.75 * float(stick['width']), 0.6))
          )
          if np.count_nonzero(near_seg) >= 3:
            spread = float(np.percentile(perp[near_seg], 80))
            target_half = max(0.32 * float(stick['width']), 0.30)
            body_breadths.append(
                min(1.0, 0.5 * norm_cov + 0.5 * min(1.0, spread / target_half))
            )
          else:
            body_breadths.append(norm_cov)
        else:
          body_breadths.append(norm_cov)

  recall_score = weighted_sum / weight_total if weight_total > 0 else 1.0
  precision_score = (
      float(np.mean(precision_scores)) if precision_scores else 1.0
  )
  wide_score = float(np.mean(wide_coverages)) if wide_coverages else recall_score
  hole_penalty = max(0.0, min(0.35, 1.2 * max(0.0, side_hole_fraction - 0.08)))
  silhouette = max(
      0.0,
      min(
          1.0,
          (0.45 * recall_score + 0.35 * wide_score + 0.20 * precision_score)
          * (1.0 - hole_penalty),
      ),
  )
  local_volume = float(np.mean(body_breadths)) if body_breadths else 0.0
  return {
      'silhouette': silhouette,
      'silhouette_recall': round(recall_score, 4),
      'silhouette_precision': round(precision_score, 4),
      'wide_element_coverage': round(wide_score, 4),
      'side_hole_fraction': round(side_hole_fraction, 4),
      'stick_silhouette_coverage': stick_coverages,
      'local_body_volume': round(local_volume, 4),
  }


def cleanliness_score(
    vertex_error: float = 0.0, non_ff_vertices: int = 0
) -> float:
  """Fold cleanliness in `[0, 1]` from Kawasaki violations and vertex strain."""
  ff_term = (
      1.0
      if non_ff_vertices <= 0
      else float(0.65 ** min(int(non_ff_vertices), 6))
  )
  verr = max(0.0, float(vertex_error))
  if verr <= 0.15:
    strain_term = 1.0
  elif verr <= 1.0:
    strain_term = max(0.30, 1.0 - 0.70 * ((verr - 0.15) / 0.85) ** 1.25)
  else:
    strain_term = 0.10
  return float(ff_term * strain_term)


def clean_gate(
    vertex_error: float = 0.0, non_ff_vertices: int = 0
) -> dict[str, Any]:
  """Hard fold-cleanliness gate: 100% 2-D Kawasaki FF and low folder strain."""
  return {
      'passed': bool(
          non_ff_vertices == 0 and vertex_error <= CLEAN_GATE_MAX_VERTEX_ERROR
      ),
      'locally_flat_foldable': bool(non_ff_vertices == 0),
      'non_flat_foldable_vertices': int(non_ff_vertices),
      'vertex_error_percent': round(float(vertex_error), 4),
  }


def components(
    geometry: Mapping[str, Any],
    *,
    layout: Layout | None = None,
    model: FoldedModel | None = None,
    vertex_error: float = 0.0,
    non_ff_vertices: int = 0,
) -> dict[str, Any]:
  """Scores in [0, 1] and diagnostics from `measure` output.

  Args:
    geometry: The output of `measure`.
    layout: Optional `paper_layout` used to scale mesh vertices for 2-D
      multi-view silhouette and local body breadth scoring.
    model: Optional `FoldedModel` used for 2-D silhouette and body breadth.
    vertex_error: Deterministic folder mean vertex error (percent).
    non_ff_vertices: Count of internal vertices violating 2-D Kawasaki/Maekawa.

  Returns:
    `{'components': {...}, 'details': {...}, 'alignment': {...}}`.
  """
  sticks = geometry['sticks']
  joints = geometry['joints']
  located = sorted(n for n, j in joints.items() if j['folded'] is not None)
  if len(located) < 3:
    raise ValueError(f'Only {len(located)} joints located on the paper.')
  source = np.array([joints[n]['folded'] for n in located])
  goal = np.array([joints[n]['target'] for n in located])
  size = math.sqrt(float(((goal - goal.mean(axis=0)) ** 2).sum(axis=1).mean()))
  fit, robust_weights = fit_similarity_robust(source, goal, size)
  aligned = dict(zip(located, fit.apply(source)))
  target = {n: np.asarray(j['target']) for n, j in joints.items()}
  errors = {
      n: float(np.linalg.norm(aligned[n] - target[n]) / size) for n in located
  }
  pck = 0.5 * (
      np.mean([e < 0.2 for e in errors.values()])
      + np.mean([e < 0.35 for e in errors.values()])
  )
  render_frame, render_rotation_matrix, frame_reason = _render_frame(
      geometry, located, source, goal, robust_weights
  )

  def folded_vector(name: str) -> np.ndarray | None:
    near, far = sticks[name]['proximal'], sticks[name]['distal']
    if near in aligned and far in aligned:
      return aligned[far] - aligned[near]
    return None

  def target_vector(name: str) -> np.ndarray:
    return target[sticks[name]['distal']] - target[sticks[name]['proximal']]

  # Direction in the aligned frame.
  min_length = _MIN_FOLDED_FRACTION * size
  directions = {}
  within = total = stray = flap_total = 0.0
  for name, stick in sticks.items():
    vector = folded_vector(name)
    if vector is None:
      continue
    collapsed = np.linalg.norm(vector) < min_length
    error = 90.0 if collapsed else _angle_deg(vector, target_vector(name))
    directions[name] = round(error, 1)
    total += stick['length']
    within += stick['length'] * (error <= 45.0)
    if not stick['is_river'] and name != geometry['root']:
      flap_total += stick['length']
      stray += stick['length'] * (error > STRAY_ANGLE_DEG)

  # Articulation of limbs: chains of paired appendages.
  def chain_points(chain: list[str], use_target: bool) -> list[np.ndarray]:
    points = []
    for index, name in enumerate(chain):
      near, far = sticks[name]['proximal'], sticks[name]['distal']
      if use_target:
        if index == 0:
          points.append(target[near])
        points.append(target[far])
        continue
      if index == 0:
        if near not in aligned:
          return []
        points.append(aligned[near])
      for fraction in SECTION_FRACTIONS:
        section = geometry['sections'].get(f'{name}@{fraction:g}')
        if section is not None:
          points.append(fit.apply(np.asarray(section['folded'])[None])[0])
      if far not in aligned:
        return []
      points.append(aligned[far])
    return points

  limb_scores = {}
  for merged_name, chain in geometry['chains'].items():
    if not any(sticks[n]['symmetric'] for n in chain):
      continue
    folded_points = chain_points(chain, use_target=False)
    if len(folded_points) < 3:
      continue
    folded_bend = max(_turns(folded_points), default=0.0)
    target_bend = max(_turns(chain_points(chain, use_target=True)), default=0.0)
    if folded_bend < _MIN_VISIBLE_BEND_DEG:
      # A straight limb reads as a stick (critic gate: knees/elbows/hocks),
      # whatever the stick figure says.
      score = 0.0
    else:
      score = min(1.0, folded_bend / max(target_bend, _MIN_VISIBLE_BEND_DEG))
    limb_scores[merged_name] = round(score, 3)
  articulation = (
      float(np.mean(list(limb_scores.values()))) if limb_scores else 1.0
  )

  # Separation: tips not clumped; mirrored pairs apart.
  tips = {}
  for chain in geometry['chains'].values():
    tip = sticks[chain[-1]]['distal']
    if tip in aligned:
      tips[tip] = sum(sticks[n]['length'] for n in chain)
  clumped = []
  for tip, length in tips.items():
    others = [other for other in tips if other != tip]
    if not others:
      continue
    folded_gap = min(np.linalg.norm(aligned[tip] - aligned[o]) for o in others)
    target_gap = min(np.linalg.norm(target[tip] - target[o]) for o in others)
    # Clumped: close in absolute terms (a tuft) and much closer than the
    # stick figure puts it (legs that stand close by design are fine).
    if folded_gap < 0.35 * length and folded_gap < 0.5 * target_gap:
      clumped.append(tip)
  tip_share = 1.0 - len(clumped) / len(tips) if tips else 1.0
  pair_scores = []
  symmetry_terms = []
  mirror = np.array([1.0, -1.0, 1.0])
  stacked_pairs = []
  asymmetric_width_pairs = []
  for name, stick in sticks.items():
    other = stick['symmetric']
    if not other or other not in sticks or name > other:
      continue
    joint_a, joint_b = stick['distal'], sticks[other]['distal']
    if joint_a not in aligned or joint_b not in aligned:
      continue
    target_gap = float(np.linalg.norm(target[joint_a] - target[joint_b]))
    if target_gap < 1e-6:
      continue
    folded_gap = float(np.linalg.norm(aligned[joint_a] - aligned[joint_b]))
    ratio = min(1.0, folded_gap / target_gap)
    pair_scores.append(ratio)
    if ratio < 0.5:
      stacked_pairs.append(f'{name}/{other}')
    mirrored = aligned[joint_b] * mirror
    pair_sym = ratio * max(
        0.0,
        1.0 - float(np.linalg.norm(aligned[joint_a] - mirrored)) / target_gap,
    )
    sec_a = geometry['sections'].get(f'{name}@0.5')
    sec_b = geometry['sections'].get(f'{other}@0.5')
    if (
        sec_a is not None
        and sec_b is not None
        and not stick['is_river']
        and not sticks[other]['is_river']
    ):
      w_a = max(float(sec_a['width']), 1e-6)
      w_b = max(float(sec_b['width']), 1e-6)
      w_hi = max(w_a, w_b)
      w_lo = min(w_a, w_b)
      if w_hi / w_lo > 1.35:
        asymmetric_width_pairs.append(f'{name}/{other}')
        pair_sym *= 0.25 + 0.75 * (w_lo / w_hi)
    symmetry_terms.append(pair_sym)
  pair_share = float(np.mean(pair_scores)) if pair_scores else 1.0
  separation = tip_share * pair_share
  symmetry = float(np.mean(symmetry_terms)) if symmetry_terms else 1.0

  # Width of flap sticks (middle section) vs. the stick figure.
  widths = {}
  width_sum = width_length = 0.0
  for name, stick in sticks.items():
    section = geometry['sections'].get(f'{name}@0.5')
    if section is None or stick['is_river'] or name == geometry['root']:
      continue
    folded_width = max(section['width'], 1e-6)
    widths[name] = round(folded_width, 3)
    ratio = min(folded_width, stick['width']) / max(
        folded_width, stick['width']
    )
    width_sum += stick['length'] * ratio
    width_length += stick['length']
  width = width_sum / width_length if width_length else 1.0

  # 3-D extent of the folded joints relative to the stick figure's: a model
  # collapsed into a bar or a flat stack has none.
  aligned_points = np.array([aligned[n] for n in located])
  folded_flatness = _flatness(aligned_points)
  target_flatness = _flatness(goal)
  if target_flatness < PLANAR_FIGURE_FLATNESS:
    # A planar stick figure (every elevation 0, e.g. the gallery Dragonfly)
    # asks for a flat model: there is no 3-D extent to reproduce, so a flat
    # fold is not a collapse. Dividing by the 0.05 floor scored such models
    # depth 0 (fidelity 3.9 with a perfect pose).
    depth = 1.0
  else:
    depth = min(1.0, folded_flatness / max(target_flatness, 0.05))
  body = geometry['body_extents']
  body_thickness = body[2] / body[0] if body[0] > 0 else 0.0
  extra_details: dict[str, Any] = {}
  if model is not None and layout is not None:
    to_tree_units = layout.grid_size / layout.grid_units_per_tree_unit
    sil = _silhouette_and_body_metrics(
        geometry, aligned, target, fit, model, to_tree_units
    )
    silhouette = float(sil['silhouette'])
    volume_score = float(sil['local_body_volume'])
    extra_details = {k: v for k, v in sil.items() if k != 'silhouette'}
  else:
    silhouette = float(width)
    volume_score = min(1.0, body_thickness / 0.2)
  cleanliness = cleanliness_score(vertex_error, non_ff_vertices)
  # Left/right side check in the alignment frame. A reflected fit is kept: a
  # model folded as a whole mirror image of a symmetric figure looks right;
  # flaps crossing relative to the rest of the model (legs under the body)
  # still show. `root_frame_directions` checks without any fit.
  mid_key = f'@{SIDE_SECTION_FRACTION:g}'
  aligned_sections = {
      key: fit.apply(np.asarray(section['folded'], dtype=float)[None])[0]
      for key, section in geometry['sections'].items()
      if key.endswith((mid_key, '@0', '@0.25'))
      and section.get('folded') is not None
  }
  crossings = side_crossings(
      sticks,
      geometry['root'],
      target,
      aligned,
      realized_sections=aligned_sections,
  )
  return {
      'components': {
          'pose': float(pck),
          'direction': within / total if total else 0.0,
          'articulation': articulation,
          'separation': separation,
          'width': width,
          'depth': depth,
          'symmetry': symmetry,
          'volume': volume_score,
          'silhouette': silhouette,
          'cleanliness': cleanliness,
      },
      'details': {
          'direction_error_deg': directions,
          'stray_fraction': stray / flap_total if flap_total else 0.0,
          'stray_flaps': sorted(
              (
                  n
                  for n, e in directions.items()
                  if e > STRAY_ANGLE_DEG and not sticks[n]['is_river']
              ),
              key=lambda n: -directions[n],
          ),
          'limb_articulation': limb_scores,
          'clumped_tips': clumped,
          'stacked_pairs': stacked_pairs,
          'side_crossings': crossings,
          'asymmetric_width_pairs': asymmetric_width_pairs,
          'widths': widths,
          'joint_error': {k: round(v, 4) for k, v in errors.items()},
          'depth_ratio': round(folded_flatness, 4),
          'target_depth_ratio': round(target_flatness, 4),
          'body_thickness': round(body_thickness, 4),
          'mirrored': fit.reflected,
          'joints_located': len(located),
          'joints_total': len(joints),
          **extra_details,
      },
      'alignment': {
          'rotation': np.round(fit.rotation, 6).tolist(),
          'scale': fit.scale,
          'translation': np.round(fit.translation, 6).tolist(),
          'render_rotation': np.round(render_rotation_matrix, 6).tolist(),
          'render_frame': render_frame,
          'render_frame_reason': frame_reason,
      },
  }


def _frame_from_axes(
    first: np.ndarray, second: np.ndarray
) -> np.ndarray | None:
  """Right-handed orthonormal frame (columns) from two directions."""
  e1 = first / (np.linalg.norm(first) or 1.0)
  e3 = second - np.dot(second, e1) * e1
  if np.linalg.norm(e1) < 0.5 or np.linalg.norm(e3) < 1e-6:
    return None
  e3 /= np.linalg.norm(e3)
  return np.stack([e1, np.cross(e3, e1), e3], axis=1)


def _render_frame(
    geometry: Mapping[str, Any],
    located: list[str],
    source: np.ndarray,
    goal: np.ndarray,
    robust_weights: np.ndarray,
) -> tuple[str, np.ndarray, str]:
  """Rotation that stands the folded model upright in the figure's frame.

  The robust proper similarity is used when it is well posed. A model folded
  into a bar (collinear joints) or matching too few joints falls back to a
  root frame: the root stick's folded direction is mapped to its
  stick-figure direction and the paper normal at the root joints to "up"
  (sign chosen so the joints' heights agree with the stick figure's).

  Args:
    geometry: `measure` output.
    located: Names of located joints (rows of `source`/`goal`).
    source: Folded joint positions (model frame).
    goal: Stick-figure joint positions.
    robust_weights: Final weights of the robust fit.

  Returns:
    (frame name 'stick_figure' | 'root' | 'native', 3x3 rotation, reason).
  """
  extents = _principal_extents(source)
  reason = ''
  if len(located) < 4:
    reason = f'only {len(located)} joints located'
  elif extents[1] < 0.05 * extents[0]:
    reason = 'folded joints are collinear (the model is a bar)'
  elif robust_weights.sum() < 3.0:
    reason = 'fewer than 3 joints agree with the stick figure'
  if not reason:
    fit = fit_similarity(
        source, goal, weights=robust_weights, allow_reflection=False
    )
    return 'stick_figure', fit.rotation, ''
  sticks, joints = geometry['sticks'], geometry['joints']
  root = sticks[geometry['root']]
  near, far = joints[root['proximal']], joints[root['distal']]
  normal = geometry.get('root_normal')
  if near['folded'] is None or far['folded'] is None or normal is None:
    return 'native', np.eye(3), reason + '; root frame unavailable'
  folded_axis = np.asarray(far['folded']) - np.asarray(near['folded'])
  target_axis = np.asarray(far['target']) - np.asarray(near['target'])
  normal = np.asarray(normal, dtype=float)
  # Up/down: heights along the normal should correlate with the figure's z.
  heights = (source - source.mean(axis=0)) @ normal
  if np.dot(heights, goal[:, 2] - goal[:, 2].mean()) < 0:
    normal = -normal
  model_frame = _frame_from_axes(folded_axis, normal)
  figure_frame = _frame_from_axes(target_axis, np.array([0.0, 0.0, 1.0]))
  if model_frame is None or figure_frame is None:
    return 'native', np.eye(3), reason + '; root frame degenerate'
  return 'root', figure_frame @ model_frame.T, reason


# Weights of the headline score: a weighted geometric mean of the components,
# so that failing one criterion (e.g. all legs stacked) cannot hide behind the
# others, like the critic's gates. Planks (`width`) and collapsed/flat models
# (`depth`) dominate the critic's calibration scores; `volume` is reported but
# not weighted until the body measure is validated.
WEIGHTS = {
    'pose': 1.0,
    'direction': 0.5,
    'articulation': 0.5,
    'separation': 1.5,
    'width': 1.5,
    'depth': 2.0,
    'symmetry': 0.5,
    'volume': 0.0,
    'silhouette': 1.0,
    'cleanliness': 1.0,
}
_COMPONENT_FLOOR = 0.05


def combine(
    scores: Mapping[str, float],
    weights: Mapping[str, float] | None = None,
) -> float:
  """Headline 0-10 score: weighted geometric mean of components in [0, 1]."""
  weights = dict(WEIGHTS if weights is None else weights)
  total = sum(weights.values())
  log_sum = sum(
      w * math.log(max(_COMPONENT_FLOOR, min(1.0, scores[k])))
      for k, w in weights.items()
  )
  return 10.0 * math.exp(log_sum / total)


def side_crossings(
    sticks: Mapping[str, Mapping[str, Any]],
    root: str,
    design_joints: Mapping[str, Any],
    realized_joints: Mapping[str, Any],
    *,
    realized_sections: Mapping[str, Any] | None = None,
    min_offset: float = SIDE_MIN_DESIGN_OFFSET,
    margin: float = SIDE_CROSSING_MARGIN,
) -> list[dict[str, Any]]:
  """Flap points folded onto the wrong side of the body's mid-plane.

  The mid-plane is the vertical plane through the root stick (lateral
  coordinate y in the figure frame: x forward, y left, z up). For every
  non-root stick, the distal joint ('tip') and the point at
  `SIDE_SECTION_FRACTION` along the stick ('mid', when `realized_sections`
  has it) are checked when the stick figure puts them at least `min_offset`
  to one side; such a point fails when its folded lateral offset is not at
  least `margin` on that side: 'crossed' when it lies past the mid-plane by
  more than `margin`, 'midline' otherwise (mirrored flaps that meet at the
  mid-plane cross each other there). This covers mirrored pairs (legs crossing
  under the body, wings folded over each other) and any flap with a lateral
  design direction; points on the axis (heads, tails) are never checked.

  Both inputs must be in the same frame orientation (the figure frame), e.g.
  the fidelity alignment of the folded joints or `root_frame_directions`
  positions; translations do not matter (offsets are taken from the root
  joints of each set).

  Args:
    sticks: Stick name -> `{'proximal': joint, 'distal': joint}` (the `measure`
      geometry's `sticks`).
    root: The root stick.
    design_joints: Joint -> stick-figure position.
    realized_joints: Joint -> folded position in the figure frame (None or
      missing when not located).
    realized_sections: `'<stick>@<fraction>'` -> folded position (same frame).
    min_offset: Minimum design offset from the mid-plane to check a point.
    margin: Clearance a folded point must keep on its own side.

  Returns:
    One entry per failing point: `{stick, point ('tip'|'mid'), kind
    ('crossed'|'midline'), design_side ('left'|'right'), design_offset,
    realized_offset}` (tree units), worst (furthest across) first.
  """
  root_joints = (sticks[root]['proximal'], sticks[root]['distal'])

  def mid_y(points) -> float | None:
    ys = [float(points[j][1]) for j in root_joints if points.get(j) is not None]
    return float(np.mean(ys)) if ys else None

  design_mid = mid_y(design_joints)
  realized_mid = mid_y(realized_joints)
  if design_mid is None or realized_mid is None:
    return []
  crossings = []
  for name, stick in sticks.items():
    if name == root:
      continue
    near, far = stick['proximal'], stick['distal']
    if near not in design_joints or far not in design_joints:
      continue
    design_near = np.asarray(design_joints[near], dtype=float)
    design_far = np.asarray(design_joints[far], dtype=float)
    near_on_axis = abs(float(design_near[1]) - design_mid) < min_offset
    checks = [('tip', design_far, realized_joints.get(far), margin)]
    if realized_sections is not None:
      fraction = SIDE_SECTION_FRACTION
      # A stick rooted on the midline starts at y ~ 0 and splays out to its
      # tip, so its mid-section only fails if it crosses past -margin; a
      # distal stick (both joints off-axis) must keep `margin` clearance at
      # its mid-section too.
      mid_threshold = -margin if near_on_axis else margin
      checks.append((
          'mid',
          design_near + fraction * (design_far - design_near),
          realized_sections.get(f'{name}@{fraction:g}'),
          mid_threshold,
      ))
      prox_realized = realized_sections.get(f'{name}@0')
      if prox_realized is None:
        prox_realized = realized_sections.get(f'{name}@0.25')
      if prox_realized is not None:
        checks.append(('proximal', design_far, prox_realized, -margin))
    for point, design, realized, threshold in checks:
      offset = float(design[1]) - design_mid
      if abs(offset) < min_offset or realized is None:
        continue
      folded_offset = float(realized[1]) - realized_mid
      own_side = folded_offset * math.copysign(1.0, offset)
      if own_side < threshold:
        crossings.append({
            'stick': name,
            'point': point,
            'kind': 'crossed' if own_side < -margin else 'midline',
            'design_side': 'left' if offset > 0 else 'right',
            'design_offset': round(offset, 3),
            'realized_offset': round(folded_offset, 3),
        })
  crossings.sort(
      key=lambda c: c['realized_offset']
      * math.copysign(1.0, c['design_offset'])
  )
  return crossings


def crossing_sticks(crossings: Sequence[Mapping[str, Any]]) -> list[str]:
  """Unique stick names of `side_crossings` entries, worst first."""
  return list(dict.fromkeys(c['stick'] for c in crossings))


def pose_gate(
    components_: Mapping[str, float], details: Mapping[str, Any]
) -> dict[str, Any]:
  """The critic's hard direction gate (origami-critique skill).

  Args:
    components_: Fidelity components.
    details: Fidelity details (`stray_flaps`, `direction_error_deg`,
      `side_crossings`).

  Returns:
    `{passed, direction, num_stray, stray, num_crossing, crossing}`: passes
    when at least `POSE_GATE_DIRECTION` of the figure points within 45 degrees
    of the stick figure, at most `POSE_GATE_MAX_STRAY` flaps are stray and no
    flap crosses the body's mid-plane (`side_crossings`).
  """
  stray = list(details.get('stray_flaps', []))
  crossing = crossing_sticks(details.get('side_crossings') or [])
  direction = float(components_.get('direction', 0.0))
  return {
      'passed': bool(
          direction >= POSE_GATE_DIRECTION
          and len(stray) <= POSE_GATE_MAX_STRAY
          and not crossing
      ),
      'direction': round(direction, 3),
      'num_stray': len(stray),
      'stray': stray,
      'num_crossing': len(crossing),
      'crossing': crossing,
  }


def score_model(
    tree: stick_figure_lib.StickFigure,
    packing: packing_output_lib.PackingOutput,
    model: FoldedModel,
    *,
    layout: Layout | None = None,
    pattern: crease_pattern_lib.CreasePattern | None = None,
    include_geometry: bool = False,
) -> dict[str, Any]:
  """Scores a folded model against the stick figure (see module docstring).

  Args:
    tree: The unmerged stick figure.
    packing: Packing output of the merged tree (flap rectangles).
    model: The folded model.
    layout: Precomputed `paper_layout(tree, packing)` (for repeated calls).
    pattern: Optional `CreasePattern` to count non-Kawasaki internal vertices.
    include_geometry: Add the `measure` output under `geometry`.

  Returns:
    A JSON-friendly report; `fidelity` is the 0-10 headline.
  """
  start = time.perf_counter()
  if layout is None:
    layout = paper_layout(tree, packing)
  if pattern is not None:
    _, _, non_ff = pattern.vertex_flat_foldability_percentage()
    non_ff_count = len(non_ff)
  else:
    non_ff_count = int(model.non_ff_vertices)
  geometry = measure(tree, layout, model)
  result = components(
      geometry,
      layout=layout,
      model=model,
      vertex_error=model.vertex_error,
      non_ff_vertices=non_ff_count,
  )
  report = {
      'fidelity': round(combine(result['components']), 2),
      'components': {k: round(v, 3) for k, v in result['components'].items()},
      **result['details'],
      'pose_gate': pose_gate(result['components'], result['details']),
      'clean_gate': clean_gate(model.vertex_error, non_ff_count),
      'alignment': result['alignment'],
      'fold_vertex_error_percent': model.vertex_error,
      'seconds': {
          'fold': round(model.seconds, 3),
          'score': round(time.perf_counter() - start, 3),
      },
  }
  if include_geometry:
    report['geometry'] = geometry
  return report


def azimuth_elevation(
    vector: Sequence[float] | np.ndarray,
) -> tuple[float, float] | None:
  """(azimuth, elevation) in degrees of a 3-D vector, None if degenerate."""
  vector = np.asarray(vector, dtype=float)
  norm = float(np.linalg.norm(vector))
  if norm < 1e-9:
    return None
  azimuth = math.degrees(math.atan2(vector[1], vector[0]))
  elevation = math.degrees(math.asin(max(-1.0, min(1.0, vector[2] / norm))))
  return round(azimuth, 2), round(elevation, 2)


def _figure_to_stick(vector: np.ndarray) -> np.ndarray:
  """Figure frame (x forward, y left, z up) -> `Stick.get_coords` frame."""
  return np.array([-vector[1], vector[0], vector[2]])


def _stick_to_figure(vector: np.ndarray) -> np.ndarray:
  """`Stick.get_coords` frame (az 0 = +y, az 90 = -x) -> figure frame."""
  return np.array([vector[1], -vector[0], vector[2]])


def root_frame_directions(
    tree: stick_figure_lib.StickFigure,
    layout: Layout,
    model: FoldedModel,
    base_model: FoldedModel,
    *,
    root_name: str,
    root_points: np.ndarray,
    up_reference: np.ndarray | None,
    root_alignment: np.ndarray | None = None,
    section: float = 0.5,
    root_axis_points: tuple[np.ndarray, np.ndarray] | None = None,
) -> dict[str, Any] | None:
  """Every stick's direction in the frame `rotation_3d` actuates in.

  `stick_directions` aligns the folded joints to the stick figure with a
  robust similarity (mirror images allowed) and measures each stick from its
  proximal to its distal joint. On a probe whose flaps the folder put within
  1.5 degrees of their requested directions it reported 3-37 degrees: (1) a
  flap rotated by `rotation_3d` turns about a cut line anchored at a corner
  of its base, so the joint-to-tip vector is offset by the flap's half width
  even when the flap's own axis is exact; (2) every such offset tip (and any
  joint the layout misplaces) tilts the whole-model alignment; (3) a
  reflected alignment swaps left and right. Closed-loop corrections fed with
  those numbers push mirrored flaps apart (Pegasus wings: L/R 20-60 degrees
  asymmetric).

  This measurement does not fit the model to the figure. The root element is
  never creased by rotations, so its faces fix the rigid transform from the
  shaped fold back to the unshaped base fold (`model.matrices` composed with
  the inverse of `base_model.matrices` at the same paper points). In the
  base, the root stick's start->end is "forward" and `up_reference` (the
  flat folded state's +z, see `shaping.measured_directions`) is "up", as in
  `heuristics_3d.get_flap_post_dict`. A flap stick's direction is its own
  axis: from the cross-section at `section` to the distal joint; rivers use
  their joints. `root_alignment` maps the figure onto the root frame (what
  `StickFigure.align_root_stick_to_target_azimuth_elevation` applies, in
  `Stick.get_coords` coordinates); results are rotated back to the figure
  frame, so they compare directly with `rotation_3d` arguments.

  Args:
    tree: The unmerged stick figure.
    layout: `paper_layout(tree, packing)`.
    model: The folded (shaped) model.
    base_model: The folded unshaped base.
    root_name: The root stick the 3-D heuristics measure from.
    root_points: (N, 2) unit-paper points inside the root element.
    up_reference: A vector (base model frame) on the "up" side of the root's
      paper normal; None keeps the normal's arbitrary sign.
    root_alignment: 3x3 rotation, figure -> root frame (stick coordinates).
    section: Cross-section fraction used as a flap stick's start.
    root_axis_points: Unit-paper points on the root's start and end side
      (`shaping_elements.get_unfolded_root_vector`), used for "forward" when the
      root's joints are not located (joints touched only by rivers).

  Returns:
    `{'directions': {stick: {target, realized, error}}, 'frame': {...}}` or
    None when the root frame cannot be built.
  """
  root = tree[root_name]
  root_points = np.asarray(root_points, dtype=float).reshape(-1, 2)
  base_faces = base_model.locate_faces(root_points)
  faces = model.locate_faces(root_points)
  transforms = [
      model.matrices[f] @ np.linalg.inv(base_model.matrices[b])
      for f, b in zip(faces, base_faces)
      if f >= 0 and b >= 0
  ]
  if not transforms:
    return None
  transforms = np.array(transforms)
  reference = np.median(transforms, axis=0)
  agreeing = [
      t
      for t in transforms
      if np.abs(t[:3, :3] - reference[:3, :3]).max() < 0.05
  ]
  transform = np.mean(agreeing, axis=0) if agreeing else reference
  to_base = np.linalg.inv(transform)[:3, :3]
  base_joints, _ = base_model.locate_groups({
      j: layout.joints[j] for j in (root.start, root.end) if j in layout.joints
  })
  if root.start not in base_joints or root.end not in base_joints:
    if root_axis_points is None:
      return None
    located = base_model.locate(np.asarray(root_axis_points, dtype=float))
    if np.isnan(located).any():
      return None
    base_joints = {root.start: located[0], root.end: located[1]}
  normals = base_model.matrices[base_faces[base_faces >= 0]][:, :3, 2]
  normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
  _, vectors = np.linalg.eigh(normals.T @ normals)
  normal = vectors[:, -1]
  if up_reference is not None and float(np.dot(normal, up_reference)) < 0:
    normal = -normal
  forward = base_joints[root.end] - base_joints[root.start]
  forward = forward - np.dot(forward, normal) * normal
  if np.linalg.norm(forward) < 1e-9:
    return None
  e1 = forward / np.linalg.norm(forward)
  e3 = normal
  e2 = np.cross(e3, e1)
  frame = np.stack([e1, e2, e3])  # Rows: base model frame -> root frame.
  alignment = np.eye(3) if root_alignment is None else root_alignment
  folded, _ = model.locate_groups(layout.joints)
  sections, _ = model.locate_groups(layout.sections)
  topo = topology(tree)
  directions = {}
  for name in topo.order:
    stick = tree[name]
    target_vector = _stick_vector(stick)
    if topo.proximal[name] != stick.start:
      target_vector = -target_vector
    distal = folded.get(topo.distal[name])
    start = sections.get(f'{name}@{section:g}')
    if start is None:
      start = folded.get(topo.proximal[name])
    realized = error = None
    if name != root_name and distal is not None and start is not None:
      in_root = frame @ (to_base @ (distal - start))
      in_figure = _stick_to_figure(alignment.T @ _figure_to_stick(in_root))
      realized = azimuth_elevation(in_figure)
      if realized is not None:
        error = round(_angle_deg(in_figure, target_vector), 1)
    directions[name] = {
        'target': azimuth_elevation(target_vector),
        'realized': realized,
        'error': error,
    }
  # Positions in the figure frame (tree units, origin at the root stick's
  # midpoint) for the side check: no fit, so a flap on the wrong side of the
  # mid-plane can't be explained away by a reflected alignment.
  to_base_full = np.linalg.inv(transform)
  origin = 0.5 * (base_joints[root.start] + base_joints[root.end])
  to_tree_units = layout.grid_size / layout.grid_units_per_tree_unit

  def to_figure(point: np.ndarray) -> list[float]:
    in_base = to_base_full[:3, :3] @ np.asarray(point) + to_base_full[:3, 3]
    in_root = frame @ (in_base - origin)
    in_figure = _stick_to_figure(alignment.T @ _figure_to_stick(in_root))
    return _rounded_list(in_figure * to_tree_units)

  positions = {name: to_figure(point) for name, point in folded.items()}
  mid_key = f'@{SIDE_SECTION_FRACTION:g}'
  section_positions = {
      key: to_figure(point)
      for key, point in sections.items()
      if key.endswith((mid_key, '@0', '@0.25'))
  }
  crossings = side_crossings(
      {
          name: {'proximal': topo.proximal[name], 'distal': topo.distal[name]}
          for name in topo.order
      },
      root_name,
      stick_figure_joint_positions(tree),
      positions,
      realized_sections=section_positions,
  )
  return {
      'directions': directions,
      'positions': positions,
      'side_crossings': crossings,
      'frame': {
          'root': root_name,
          'agreeing_root_faces': round(len(agreeing) / len(transforms), 3),
          'up_from_reference': up_reference is not None,
      },
  }


def stick_directions(
    tree: stick_figure_lib.StickFigure,
    packing: packing_output_lib.PackingOutput,
    model: FoldedModel,
    *,
    layout: Layout | None = None,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
  """Every stick's realized direction in the stick figure's frame.

  The folded joints are mapped into the figure frame by the fidelity
  alignment (x forward at azimuth 0, y left, z up), so the result compares
  directly with the stick figure's azimuth/elevation (and `rotation_3d`
  arguments).

  Args:
    tree: The unmerged stick figure.
    packing: Packing output of the merged tree.
    model: The folded model.
    layout: Precomputed `paper_layout(tree, packing)`.

  Returns:
    (fidelity report, per-stick `{target: (az, el), realized: (az, el) |
    None, error: degrees | None}`).
  """
  report = score_model(
      tree, packing, model, layout=layout, include_geometry=True
  )
  geometry = report.pop('geometry')
  alignment = report['alignment']
  rotation = np.asarray(alignment['rotation'], dtype=float)
  scale = float(alignment['scale'])
  translation = np.asarray(alignment['translation'], dtype=float)
  joints = geometry['joints']
  if report.get('mirrored'):
    # The score accepts mirror images, but a direction measured through a
    # reflection has the wrong handedness: a left flap reads as the right
    # one, and closed-loop corrections (`recipes.figure_pose`,
    # `shaping.bend_to_direction`) then push it the wrong way. Measure in
    # the best proper frame instead.
    located = sorted(n for n, j in joints.items() if j['folded'] is not None)
    source = np.array([joints[n]['folded'] for n in located])
    goal = np.array([joints[n]['target'] for n in located])
    size = math.sqrt(
        float(((goal - goal.mean(axis=0)) ** 2).sum(axis=1).mean())
    )
    proper, _ = fit_similarity_robust(
        source, goal, size, allow_reflection=False
    )
    rotation, scale, translation = (
        proper.rotation,
        proper.scale,
        proper.translation,
    )
  directions = {}
  for name, stick in geometry['sticks'].items():
    near, far = joints[stick['proximal']], joints[stick['distal']]
    target = np.asarray(far['target']) - np.asarray(near['target'])
    realized = None
    error = report['direction_error_deg'].get(name)
    if near['folded'] is not None and far['folded'] is not None:
      a = scale * rotation @ np.asarray(near['folded']) + translation
      b = scale * rotation @ np.asarray(far['folded']) + translation
      realized = azimuth_elevation(b - a)
      if report.get('mirrored') and error is not None:
        error = round(_angle_deg(b - a, target), 1)
    directions[name] = {
        'target': azimuth_elevation(target),
        'realized': realized,
        'error': error,
    }
  return report, directions


_PATTERN_SCORE_CACHE: dict[
    tuple[int, int, int, int], tuple[tuple[Any, Any, Any], dict[str, Any]]
] = {}
_MAX_PATTERN_SCORE_CACHE = 64


def score_pattern(
    tree: stick_figure_lib.StickFigure,
    packing: packing_output_lib.PackingOutput,
    pattern: crease_pattern_lib.CreasePattern,
    *,
    layout: Layout | None = None,
) -> dict[str, Any]:
  """Folds `pattern` and scores it against `tree` (see `score_model`)."""
  key = (id(tree), id(packing), id(pattern), len(pattern.creases))
  cached = _PATTERN_SCORE_CACHE.get(key)
  # `id()` values are reused once an object is freed: the cache keeps the
  # objects alive and checks identity, so a new pattern that happens to get a
  # freed pattern's id (and crease count) is never served the old score.
  if cached is not None:
    (c_tree, c_packing, c_pattern), result = cached
    if c_tree is tree and c_packing is packing and c_pattern is pattern:
      return result
  result = score_model(
      tree,
      packing,
      FoldedModel.from_pattern(pattern),
      layout=layout,
      pattern=pattern,
  )
  if len(_PATTERN_SCORE_CACHE) >= _MAX_PATTERN_SCORE_CACHE:
    _PATTERN_SCORE_CACHE.pop(next(iter(_PATTERN_SCORE_CACHE)))
  _PATTERN_SCORE_CACHE[key] = ((tree, packing, pattern), result)
  return result


def default_pattern_path(run_dir: str) -> str:
  """The shaped pattern if present, else the base pattern."""
  shaped = os.path.join(run_dir, 'shaping_cp.svg')
  if os.path.exists(shaped):
    return shaped
  return os.path.join(run_dir, 'solving_cp.svg')


def fold_svg(svg_string: str) -> FoldedModel:
  """Folds an SVG crease pattern with the deterministic folder."""
  return FoldedModel.from_svg(svg_string)


class Evaluation(dict):
  """Score report dictionary with attribute-access fit properties."""

  @property
  def fit(self) -> Any:
    class _Fit:

      def __init__(self, rotation):
        self.rotation = np.asarray(rotation)

    return _Fit(self.get('alignment', {}).get('rotation', np.eye(3)))


def evaluate(
    tree: stick_figure_lib.StickFigure,
    packing: packing_output_lib.PackingOutput,
    model: FoldedModel,
) -> Evaluation:
  """Scores folded model against tree."""
  layout = paper_layout(tree, packing)
  return Evaluation(score_model(tree, packing, model, layout=layout))


def locate_joints(
    tree: stick_figure_lib.StickFigure,
    packing: packing_output_lib.PackingOutput,
    model: FoldedModel,
) -> dict[str, list[float] | None]:
  """Locates folded joints in tree units."""
  layout = paper_layout(tree, packing)
  geom = measure(tree, layout, model)
  return {
      name: val.get('folded') for name, val in geom.get('joints', {}).items()
  }


# ------------------------------------------------------------------------------
# 3-D + 2-D multi-tool shaping composition helpers.
# ------------------------------------------------------------------------------


def has_3d_overlay(pattern: crease_pattern_lib.CreasePattern) -> bool:
  """Whether `pattern` has non-Kawasaki 3-D vertices or partial fold angles."""
  if not pattern.local_flat_foldability():
    return True
  return any(
      c.mountain_or_valley()
      and not math.isclose(c.fold_percent, 1.0, abs_tol=1e-5)
      for c in pattern.creases
  )


def as_flat_backbone(
    pattern: crease_pattern_lib.CreasePattern,
) -> crease_pattern_lib.CreasePattern:
  """Returns `pattern` with M/V `fold_percent` normalized to `1.0`."""
  if all(
      not c.mountain_or_valley()
      or math.isclose(c.fold_percent, 1.0, abs_tol=1e-5)
      for c in pattern.creases
  ):
    return pattern
  creases: list[crease_utils.Crease] = []
  for c in pattern.creases:
    if c.mountain_or_valley() and not math.isclose(
        c.fold_percent, 1.0, abs_tol=1e-5
    ):
      creases.append(dataclasses.replace(c, fold_percent=1.0))
    else:
      creases.append(c)
  return crease_pattern_lib.CreasePattern(
      creases=creases, light_mode=True, tolerance=pattern.tolerance
  )


def _point_near_segment(
    px: float, py: float, c: crease_utils.Crease, tol: float = 3e-4
) -> bool:
  """Whether `(px, py)` lies within `tol` of segment `c`."""
  x1, y1, x2, y2 = c.p1.x, c.p1.y, c.p2.x, c.p2.y
  dx, dy = x2 - x1, y2 - y1
  seg_len_sq = dx * dx + dy * dy
  if seg_len_sq <= 1e-12:
    return math.hypot(px - x1, py - y1) <= tol
  t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / seg_len_sq))
  proj_x = x1 + t * dx
  proj_y = y1 + t * dy
  return math.hypot(px - proj_x, py - proj_y) <= tol


def preserves_foldability(
    prior: crease_pattern_lib.CreasePattern | None,
    candidate: crease_pattern_lib.CreasePattern,
    *,
    allow_new_3d_vertices: bool = False,
) -> tuple[bool, str]:
  """Checks 2-D Kawasaki flat-foldability or 3-D foldability preservation.

  When `prior` is None or lacks any 3-D overlay (and `allow_new_3d_vertices` is
  False), requires `candidate` to be 100% 2-D flat-foldable. When `prior`
  already contains 3-D relief creases or partial fold angles, requires every new
  internal vertex outside `prior`'s partial-crease overlay to satisfy 3-D
  Belcastro-Hull foldability (and 2-D Kawasaki/Maekawa unless
  `allow_new_3d_vertices` is True).

  Args:
    prior: Pattern before the current tool step, if any.
    candidate: Candidate pattern after the tool step.
    allow_new_3d_vertices: Whether `candidate` is allowed to introduce new 3-D
      Belcastro-Hull vertices that are not 2-D Kawasaki flat-foldable.

  Returns:
    `(ok, message)`.
  """
  pct, _, after_non_ff = candidate.vertex_flat_foldability_percentage()
  if math.isclose(pct, 1.0):
    return True, 'ok'
  default_err = (
      'Crease pattern is not flat-foldable after shaping:'
      f' {pct:.1%} of vertices satisfy Kawasaki/Maekawa.'
  )
  if prior is None:
    if allow_new_3d_vertices and candidate.is_locally_foldable():
      return True, 'ok'
    return False, default_err
  prior_pct, _, prior_non_ff = prior.vertex_flat_foldability_percentage()
  if (
      math.isclose(prior_pct, 1.0)
      and not has_3d_overlay(prior)
      and not allow_new_3d_vertices
  ):
    return False, default_err
  partial_creases = [
      c
      for c in prior.creases
      if c.mountain_or_valley()
      and not math.isclose(c.fold_percent, 1.0, abs_tol=1e-5)
  ]
  cand_partial_creases = [
      c
      for c in candidate.creases
      if c.mountain_or_valley()
      and not math.isclose(c.fold_percent, 1.0, abs_tol=1e-5)
  ]
  after_non_ff_pts = {
      (round(float(v.point.x), 4), round(float(v.point.y), 4))
      for v in after_non_ff
  }
  prior_non_3d = [v for v in prior.vertices if v.internal and not v.is_foldable]
  for v in candidate.vertices:
    if not v.internal or v.is_foldable:
      continue
    vx, vy = float(v.point.x), float(v.point.y)
    in_prior_3d = any(
        math.hypot(vx - float(pv.point.x), vy - float(pv.point.y)) <= 3e-4
        for pv in prior_non_3d
    )
    if in_prior_3d:
      continue
    on_partial = any(
        _point_near_segment(vx, vy, pc, tol=3e-4) for pc in partial_creases
    )
    if on_partial:
      continue
    if (round(vx, 4), round(vy, 4)) not in after_non_ff_pts and any(
        _point_near_segment(vx, vy, pc, tol=3e-4) for pc in cand_partial_creases
    ):
      continue
    return (
        False,
        (
            'Crease pattern fails 3-D Belcastro-Hull vertex foldability after'
            f' shaping at ({vx:.4f}, {vy:.4f}).'
        ),
    )
  if math.isclose(pct, 1.0) or allow_new_3d_vertices:
    return True, 'ok'
  for v in after_non_ff:
    vx, vy = float(v.point.x), float(v.point.y)
    in_prior = any(
        math.hypot(vx - float(pv.point.x), vy - float(pv.point.y)) <= 3e-4
        for pv in prior_non_ff
    )
    if in_prior:
      continue
    on_partial = any(
        _point_near_segment(vx, vy, pc, tol=3e-4) for pc in partial_creases
    )
    if not on_partial:
      return (
          False,
          (
              f'New non-flat-foldable vertex at ({vx:.4f}, {vy:.4f})'
              f' ({pct:.1%} Kawasaki).'
          ),
      )
  return True, 'ok'


def _find_containing_crease(
    seg: crease_utils.Crease, pool: Sequence[crease_utils.Crease]
) -> crease_utils.Crease | None:
  """Returns the first crease in `pool` geometrically containing `seg`."""
  for other in pool:
    if not seg.not_in_encompassing_rectangles(other) and seg.is_within(
        other, match_fold=False
    ):
      return other
  return None


def replay_3d_overlay(
    prev_backbone: crease_pattern_lib.CreasePattern,
    prev_pattern: crease_pattern_lib.CreasePattern,
    new_backbone: crease_pattern_lib.CreasePattern,
) -> crease_pattern_lib.CreasePattern:
  """Replays 3-D relief creases and partial fold angles onto `new_backbone`.

  Args:
    prev_backbone: Clean 2-D Kawasaki flat-foldable pattern prior to the step.
    prev_pattern: Current 3-D pattern (`prev_backbone` + 3-D relief overlays).
    new_backbone: Updated 2-D Kawasaki flat-foldable pattern after a 2-D tool.

  Returns:
    Combined crease pattern containing both `new_backbone`'s 2-D folds and
    `prev_pattern`'s 3-D overlays.
  """
  prev_merged = copy.deepcopy(prev_backbone)
  prev_merged.merge_creases(only_trivial=False)
  prev_mv = [c for c in prev_merged.creases if c.mountain_or_valley()]

  new_merged = copy.deepcopy(new_backbone)
  new_merged.merge_creases(only_trivial=False)
  new_mv = [c for c in new_merged.creases if c.mountain_or_valley()]

  added_3d: list[crease_utils.Crease] = []
  # `(overlay_crease, preserve_overlay_fold_type)`
  modified_3d: list[tuple[crease_utils.Crease, bool]] = []

  for c in prev_pattern.creases:
    if not c.mountain_or_valley():
      continue
    host = _find_containing_crease(c, prev_mv)
    if host is not None:
      if not math.isclose(
          c.fold_percent, host.fold_percent, abs_tol=1e-5
      ) or not math.isclose(c.fold_percent, 1.0, abs_tol=1e-5):
        modified_3d.append((c, c.fold != host.fold))
    else:
      if _find_containing_crease(c, new_mv) is not None:
        modified_3d.append((c, True))
      else:
        added_3d.append(c)

  if not added_3d and not modified_3d:
    return new_backbone

  if added_3d:
    combined = new_backbone.update_with_shaping_creases(
        added_3d, split_shaping_by_shaping=True
    )
  else:
    combined = new_backbone
  if not modified_3d:
    return combined

  mod_segments = [mc for mc, _ in modified_3d]
  mod_tips = []
  for mc in mod_segments:
    mod_tips.extend([mc.p1, mc.p2])

  split_list: list[crease_utils.Crease] = []
  for c in combined.creases:
    if not c.mountain_or_valley():
      split_list.append(c)
      continue
    strictly_contains = [
        mc
        for mc in mod_segments
        if not mc.not_in_encompassing_rectangles(c)
        and mc.is_within(c, match_fold=False)
        and not c.is_within(mc, match_fold=False)
    ]
    if strictly_contains:
      try:
        split_list.extend(
            c.split_to_segments(
                strictly_contains, tips=mod_tips, tolerance=combined.tolerance
            )
        )
        continue
      except ValueError:
        pass
    split_list.append(c)

  updated_creases: list[crease_utils.Crease] = []
  for c in split_list:
    if not c.mountain_or_valley():
      updated_creases.append(c)
      continue
    matched_mod: tuple[crease_utils.Crease, bool] | None = None
    for mod_c, use_mod_fold in modified_3d:
      if not c.not_in_encompassing_rectangles(mod_c) and c.is_within(
          mod_c, match_fold=False
      ):
        matched_mod = (mod_c, use_mod_fold)
        break
    if matched_mod is not None:
      mod_c, use_mod_fold = matched_mod
      updated_creases.append(
          dataclasses.replace(
              c,
              fold=mod_c.fold if use_mod_fold else c.fold,
              fold_percent=mod_c.fold_percent,
          )
      )
    else:
      updated_creases.append(c)

  return crease_pattern_lib.CreasePattern(
      creases=updated_creases,
      light_mode=True,
      tolerance=new_backbone.tolerance,
  )


def evaluate_fidelity(
    target: (
        bus.SolvedCreasePattern
        | bus.ShapedCreasePattern
        | bus.PipelineResult
    ),
    *,
    tree: stick_figure_lib.StickFigure | bus.DesignTree | None = None,
    include_geometry: bool = False,
) -> dict[str, Any]:
  """Evaluates 0-10 geometric pose fidelity on a `bp_pipeline` stage record.

  Args:
    target: A `SolvedCreasePattern`, `ShapedCreasePattern`, or `PipelineResult`.
    tree: Optional unmerged `StickFigure` or `DesignTree` override. Defaults to
      the source stick figure attached to `target`.
    include_geometry: Whether to include raw `measure()` joint/section coords.

  Returns:
    JSON-serializable fidelity dictionary (`fidelity`, `components`,
    `pose_gate`, `clean_gate`, `alignment`, and per-stick diagnostics).
  """
  if isinstance(target, bus.PipelineResult):
    solved = target.solved
    pattern = (
        target.shaped.creases if target.shaped is not None else solved.creases
    )
  elif isinstance(target, bus.ShapedCreasePattern):
    solved = target.solved
    pattern = target.creases
  elif isinstance(target, bus.SolvedCreasePattern):
    solved = target
    pattern = target.creases
  else:
    raise TypeError(
        'Expected SolvedCreasePattern, ShapedCreasePattern, or PipelineResult;'
        f' got {type(target)!r}'
    )

  packing_output, _ = bus_adapter.legacy_inputs(solved)
  if isinstance(tree, bus.DesignTree):
    resolved_tree: stick_figure_lib.StickFigure | None = tree.source
  elif isinstance(tree, stick_figure_lib.StickFigure):
    resolved_tree = tree
  else:
    resolved_tree = (
        packing_output.unmerged_stick_figure or packing_output.stick_figure
    )
  if resolved_tree is None:
    raise ValueError(
        'evaluate_fidelity() requires a DesignTree on `target` or an explicit'
        ' `tree=` argument.'
    )

  model = FoldedModel.from_pattern(pattern)
  return score_model(
      resolved_tree,
      packing_output,
      model,
      pattern=pattern,
      include_geometry=include_geometry,
  )
