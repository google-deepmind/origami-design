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

"""Geometric 0-10 pose and silhouette fidelity scorer for folded 3D models.

Compares a folded origami mesh (`FoldedModel`) against its source `StickFigure`
using unit-paper joint and cross-section contours (`PaperLayout`) and robust
similarity alignment (`pose_alignment`):

* `pose`: Percentage of joints within `0.2` and `0.35` figure sizes of target.
* `direction`: Length-weighted fraction of sticks within `45` degrees of target.
* `articulation`: Realized bend on bilaterally paired appendage chains.
* `separation`: Freedom from tip clumping and left/right pair stacking.
* `width`: Mid-section flap width match against target stick widths.
* `depth`: 3D thickness ratio relative to the stick figure (1.0 when planar).
* `symmetry`: Bilateral mirror position and width agreement across pairs.
* `volume`: Local cross-sectional body/river breadth (diagnostic, weight `0.0`).
* `silhouette`: Multi-view (`xy`, `xz`, `yz`) rasterized recall, wide-element
  coverage, precision, and side-profile hole penalty.
* `cleanliness`: Local Kawasaki flat-foldability and deterministic fold strain.
"""

from collections.abc import Mapping, Sequence
import dataclasses
import math
import re
import time
import types
from typing import Any, Literal

import numpy as np

from .. import bus
from .. import folder_3d as fold_3d
from ..crease_pattern import crease_pattern as cp_lib
from ..packing import packing_output as packing_output_lib
from ..stick_figures import stick_figure as stick_figure_lib
from . import bus_adapter
from . import paper_layout
from . import pose_alignment

DEFAULT_WEIGHTS: Mapping[str, float] = types.MappingProxyType({
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
})

# Width of a uniform 1D segment equals sqrt(12) times its RMS spread.
_WIDTH_PER_SPREAD = math.sqrt(12.0)
_COMPONENT_FLOOR = 0.05
_MIN_FOLDED_FRACTION = 0.02

_SYMMETRIC_NAME_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r'_L([0-9]*)$'), r'_R\1'),
    (re.compile(r'_R([0-9]*)$'), r'_L\1'),
    (re.compile(r'_l([0-9]*)$'), r'_r\1'),
    (re.compile(r'_r([0-9]*)$'), r'_l\1'),
    (re.compile(r'_L([0-9]*)_'), r'_R\1_'),
    (re.compile(r'_R([0-9]*)_'), r'_L\1_'),
    (re.compile(r'_l([0-9]*)_'), r'_r\1_'),
    (re.compile(r'_r([0-9]*)_'), r'_l\1_'),
    (re.compile(r'left'), 'right'),
    (re.compile(r'right'), 'left'),
)


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class FidelityConfig:
  """Thresholds and component weights controlling fidelity evaluation.

  Attributes:
    weights: Geometric-mean weights per component name.
    direction_tolerance_deg: Max angular error counted as aligned in
      `direction`.
    stray_angle_deg: Angular error above which a non-root flap is flagged stray.
    pose_gate_direction: Minimum `direction` score required by `PoseGate`.
    pose_gate_max_stray: Maximum stray flap count permitted by `PoseGate`.
    clean_gate_max_vertex_error: Maximum fold vertex strain (%) for `CleanGate`.
    min_visible_bend_deg: Minimum turning angle (degrees) for articulated limbs.
    planar_figure_flatness: Target depth ratio below which a figure is planar.
    asymmetric_width_ratio: Max L/R width ratio before penalizing symmetry.
    silhouette_grid_size: Raster resolution per side for orthographic views.
  """

  weights: Mapping[str, float] = DEFAULT_WEIGHTS
  direction_tolerance_deg: float = 45.0
  stray_angle_deg: float = 60.0
  pose_gate_direction: float = 0.7
  pose_gate_max_stray: int = 2
  clean_gate_max_vertex_error: float = 0.85
  min_visible_bend_deg: float = 20.0
  planar_figure_flatness: float = 0.01
  asymmetric_width_ratio: float = 1.35
  silhouette_grid_size: int = 56


DEFAULT_CONFIG = FidelityConfig()


@dataclasses.dataclass(frozen=True, slots=True)
class PoseGate:
  """Hard pose gate: aligned directions, few strays, and no L/R crossings."""

  passed: bool
  direction: float
  stray: tuple[str, ...]
  crossing: tuple[str, ...]

  def to_dict(self) -> dict[str, Any]:
    return {
        'passed': self.passed,
        'direction': round(self.direction, 3),
        'num_stray': len(self.stray),
        'stray': list(self.stray),
        'num_crossing': len(self.crossing),
        'crossing': list(self.crossing),
    }


@dataclasses.dataclass(frozen=True, slots=True)
class CleanGate:
  """Hard fold-cleanliness gate: local flat-foldability and low mesh strain."""

  passed: bool
  locally_flat_foldable: bool
  non_flat_foldable_vertices: int
  vertex_error_percent: float

  def to_dict(self) -> dict[str, Any]:
    return {
        'passed': self.passed,
        'locally_flat_foldable': self.locally_flat_foldable,
        'non_flat_foldable_vertices': self.non_flat_foldable_vertices,
        'vertex_error_percent': round(self.vertex_error_percent, 4),
    }


@dataclasses.dataclass(frozen=True, slots=True)
class AlignmentInfo:
  """Similarity alignment and upright camera rotation in the figure frame."""

  rotation: np.ndarray
  scale: float
  translation: np.ndarray
  render_rotation: np.ndarray
  render_frame: Literal['stick_figure', 'root', 'native']
  render_frame_reason: str

  def to_dict(self) -> dict[str, Any]:
    return {
        'rotation': np.round(self.rotation, 6).tolist(),
        'scale': float(self.scale),
        'translation': np.round(self.translation, 6).tolist(),
        'render_rotation': np.round(self.render_rotation, 6).tolist(),
        'render_frame': self.render_frame,
        'render_frame_reason': self.render_frame_reason,
    }


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class FidelityReport:
  """Typed 0-10 geometric fidelity evaluation report for a folded model."""

  fidelity: float
  components: Mapping[str, float]
  pose_gate: PoseGate
  clean_gate: CleanGate
  alignment: AlignmentInfo
  details: Mapping[str, Any]
  fold_vertex_error_percent: float
  fold_seconds: float
  score_seconds: float
  geometry: Mapping[str, Any] | None = None

  @property
  def render_rotation(self) -> np.ndarray:
    """Proper 3x3 rotation placing the folded model upright in figure frame."""
    return self.alignment.render_rotation

  @property
  def mirrored(self) -> bool:
    """Whether the best similarity fit reflects the model across the midline."""
    return bool(self.details.get('mirrored', False))

  def to_dict(self) -> dict[str, Any]:
    """Serializes this report to a JSON-compatible dictionary."""
    out: dict[str, Any] = {
        'fidelity': round(self.fidelity, 2),
        'components': {
            k: round(float(v), 3) for k, v in self.components.items()
        },
        **self.details,
        'pose_gate': self.pose_gate.to_dict(),
        'clean_gate': self.clean_gate.to_dict(),
        'alignment': self.alignment.to_dict(),
        'fold_vertex_error_percent': float(self.fold_vertex_error_percent),
        'seconds': {
            'fold': round(self.fold_seconds, 3),
            'score': round(self.score_seconds, 3),
        },
    }
    if self.geometry is not None:
      out['geometry'] = self.geometry
    return out


def _round_vec(values: np.ndarray, digits: int = 5) -> list[float]:
  return [round(float(v), digits) for v in values]


def infer_symmetric_stick(name: str, all_names: set[str]) -> str | None:
  """Infers a bilateral mirror name when `symmetric_stick_name` is unset."""
  if 'major_' in name or 'minor_' in name:
    return None
  for pattern, repl in _SYMMETRIC_NAME_RULES:
    if not pattern.search(name):
      continue
    cand = pattern.sub(repl, name, count=1)
    if (
        cand != name
        and cand in all_names
        and 'major_' not in cand
        and 'minor_' not in cand
    ):
      return cand
  return None


_infer_symmetric_stick = infer_symmetric_stick


def _root_normal(
    tree: stick_figure_lib.StickFigure,
    layout: paper_layout.PaperLayout,
    model: fold_3d.FoldedModel,
) -> list[float] | None:
  """Returns the dominant unoriented face normal at the root joints."""
  root = tree[tree.root]
  parts = [
      layout.joints[j] for j in (root.start, root.end) if j in layout.joints
  ]
  if not parts:
    return None
  faces = model.locate_faces(np.concatenate(parts, axis=0))
  faces = faces[faces >= 0]
  if faces.size == 0:
    return None
  axis = pose_alignment.dominant_axis(model.face_matrices[faces][:, :3, 2])
  return None if axis is None else _round_vec(axis)


def _body_mask(
    model: fold_3d.FoldedModel, layout: paper_layout.PaperLayout
) -> np.ndarray:
  """Selects vertices strictly outside every packed flap rectangle."""
  x, y = model.vertices_2d[:, 0], model.vertices_2d[:, 1]
  inside = np.zeros(len(x), dtype=bool)
  eps = 1e-6
  for x0, x1, y0, y1 in layout.flap_rectangles:
    inside |= (x > x0 + eps) & (x < x1 - eps) & (y > y0 + eps) & (y < y1 - eps)
  return ~inside


def measure(
    tree: stick_figure_lib.StickFigure,
    layout: paper_layout.PaperLayout,
    model: fold_3d.FoldedModel,
) -> dict[str, Any]:
  """Measures folded and target joint, cross-section, and body geometry."""
  topo = paper_layout.topology(tree)
  all_names = set(topo.order)
  target = paper_layout.stick_figure_joint_positions(tree)
  folded, _ = model.locate_groups(layout.joints)
  sec_means, sec_spreads = model.locate_groups(layout.sections)
  to_tree = layout.tree_units_per_paper_unit
  body_extents = pose_alignment.principal_extents(
      model.vertices_3d[_body_mask(model, layout)]
  )
  model_extents = pose_alignment.principal_extents(model.vertices_3d)
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
                  or _infer_symmetric_stick(name, all_names)
              ),
          }
          for name in topo.order
      },
      'chains': {k: list(v) for k, v in layout.chains.items()},
      'joints': {
          name: {
              'folded': (
                  _round_vec(folded[name] * to_tree) if name in folded else None
              ),
              'target': _round_vec(target[name]),
          }
          for name in target
      },
      'sections': {
          name: {
              'folded': _round_vec(sec_means[name] * to_tree),
              'width': round(
                  _WIDTH_PER_SPREAD * sec_spreads[name] * to_tree, 4
              ),
          }
          for name in sec_means
      },
      'model_extents': _round_vec(model_extents * to_tree),
      'body_extents': _round_vec(body_extents * to_tree),
  }


def _flatness(points: np.ndarray) -> float:
  extents = pose_alignment.principal_extents(points)
  return float(extents[2] / extents[0]) if extents[0] > 0.0 else 0.0


def _polyline_turns_deg(points: Sequence[np.ndarray]) -> list[float]:
  segs = [b - a for a, b in zip(points, points[1:])]
  segs = [s for s in segs if float(np.linalg.norm(s)) > 1e-9]
  return [pose_alignment.angle_deg(a, b) for a, b in zip(segs, segs[1:])]


def _fan_triangulate(faces: Sequence[np.ndarray]) -> np.ndarray:
  tris: list[tuple[int, int, int]] = []
  for face in faces:
    if len(face) < 3:
      continue
    v0 = int(face[0])
    for idx in range(1, len(face) - 1):
      tris.append((v0, int(face[idx]), int(face[idx + 1])))
  return np.asarray(tris, dtype=int) if tris else np.zeros((0, 3), dtype=int)


def _rasterize_triangles_2d(
    pts_2d: np.ndarray,
    tris: np.ndarray,
    grid_size: int,
    bounds: tuple[float, float, float, float],
) -> np.ndarray:
  """Rasterizes 2D triangles onto a `(grid_size, grid_size)` boolean mask."""
  u_min, u_max, v_min, v_max = bounds
  mask = np.zeros((grid_size, grid_size), dtype=bool)
  if len(tris) == 0 or u_max <= u_min + 1e-9 or v_max <= v_min + 1e-9:
    return mask
  pu = (pts_2d[:, 0] - u_min) * ((grid_size - 1) / (u_max - u_min))
  pv = (pts_2d[:, 1] - v_min) * ((grid_size - 1) / (v_max - v_min))
  for i0, i1, i2 in tris:
    u0, u1, u2 = float(pu[i0]), float(pu[i1]), float(pu[i2])
    v0, v1, v2 = float(pv[i0]), float(pv[i1]), float(pv[i2])
    c_min = max(0, math.floor(min(u0, u1, u2)))
    c_max = min(grid_size - 1, math.ceil(max(u0, u1, u2)))
    r_min = max(0, math.floor(min(v0, v1, v2)))
    r_max = min(grid_size - 1, math.ceil(max(v0, v1, v2)))
    if c_min > c_max or r_min > r_max:
      continue
    area2 = (u1 - u0) * (v2 - v0) - (v1 - v0) * (u2 - u0)
    if abs(area2) < 0.25:
      steps = max(2, min(grid_size, max(c_max - c_min, r_max - r_min) + 2))
      ts = np.linspace(0.0, 1.0, steps)
      for ua, va, ub, vb in ((u0, v0, u1, v1), (u1, v1, u2, v2)):
        cs = np.clip(
            np.round(ua + ts * (ub - ua)).astype(int), 0, grid_size - 1
        )
        rs = np.clip(
            np.round(va + ts * (vb - va)).astype(int), 0, grid_size - 1
        )
        mask[rs, cs] = True
      continue
    cols = np.arange(c_min, c_max + 1, dtype=float)[None, :]
    rows = np.arange(r_min, r_max + 1, dtype=float)[:, None]
    e0 = (u1 - u0) * (rows - v0) - (v1 - v0) * (cols - u0)
    e1 = (u2 - u1) * (rows - v1) - (v2 - v1) * (cols - u1)
    e2 = (u0 - u2) * (rows - v2) - (v0 - v2) * (cols - u2)
    tol = 0.35 * abs(area2) / max(abs(u1 - u0) + abs(v1 - v0), 1.0)
    inside = (
        (e0 >= -tol) & (e1 >= -tol) & (e2 >= -tol)
        if area2 > 0.0
        else (e0 <= tol) & (e1 <= tol) & (e2 <= tol)
    )
    mask[r_min : r_max + 1, c_min : c_max + 1] |= inside
  return mask


def _capsule_mask_2d(
    uu: np.ndarray,
    vv: np.ndarray,
    p0: np.ndarray,
    p1: np.ndarray,
    radius: float,
) -> np.ndarray:
  """Returns a boolean grid mask of points within `radius` of segment `p0-p1`."""
  du, dv = float(p1[0] - p0[0]), float(p1[1] - p0[1])
  seg_sq = du * du + dv * dv
  ru, rv = uu - float(p0[0]), vv - float(p0[1])
  if seg_sq <= 1e-12:
    return ru * ru + rv * rv <= radius * radius
  t = np.clip((ru * du + rv * dv) / seg_sq, 0.0, 1.0)
  return (ru - t * du) ** 2 + (rv - t * dv) ** 2 <= radius * radius


def _interior_hole_fraction(mask: np.ndarray) -> float:
  """Returns enclosed interior void fraction within a 2D silhouette mask."""
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
    grown = outside.copy()
    grown[1:, :] |= outside[:-1, :]
    grown[:-1, :] |= outside[1:, :]
    grown[:, 1:] |= outside[:, :-1]
    grown[:, :-1] |= outside[:, 1:]
    grown &= empty
    if np.array_equal(grown, outside):
      break
    outside = grown
  holes = int((empty & ~outside).sum())
  return float(holes / max(filled + holes, 1))


def _local_segment_breadth(
    mesh_v: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
    width: float,
    norm_cov: float,
) -> float:
  """Computes mid-segment perpendicular breadth score for a body/river stick."""
  axis = end - start
  seg_len = float(np.linalg.norm(axis))
  if seg_len <= 1e-6:
    return norm_cov
  unit = axis / seg_len
  rel = mesh_v - start[None, :]
  proj = rel @ unit
  perp = np.linalg.norm(rel - proj[:, None] * unit[None, :], axis=1)
  near_mid = (
      (proj >= 0.25 * seg_len)
      & (proj <= 0.75 * seg_len)
      & (perp <= max(0.75 * width, 0.6))
  )
  if int(np.count_nonzero(near_mid)) < 3:
    return norm_cov
  spread = float(np.percentile(perp[near_mid], 80))
  target_half = max(0.32 * width, 0.30)
  return min(1.0, 0.5 * norm_cov + 0.5 * min(1.0, spread / target_half))


def _silhouette_and_body_metrics(
    geometry: Mapping[str, Any],
    aligned: Mapping[str, np.ndarray],
    target: Mapping[str, np.ndarray],
    fit: pose_alignment.Similarity,
    model: fold_3d.FoldedModel,
    to_tree_units: float,
    grid_size: int,
) -> dict[str, Any]:
  """Computes multi-view 2D silhouette recall, precision, holes, and breadth."""
  sticks = geometry['sticks']
  mesh_v = fit.apply(model.vertices_3d * to_tree_units)
  tris = _fan_triangulate(model.faces)
  target_pts = np.asarray(list(target.values()), dtype=float)
  max_w = max((float(s['width']) for s in sticks.values()), default=1.0)

  recalls_by_stick: dict[str, list[float]] = {name: [] for name in sticks}
  precisions: list[float] = []
  side_hole_fraction = 0.0

  for plane_name, ax_u, ax_v in (('xy', 0, 1), ('xz', 0, 2), ('yz', 1, 2)):
    t_uv = target_pts[:, [ax_u, ax_v]]
    span = max(
        float(t_uv[:, 0].max() - t_uv[:, 0].min()),
        float(t_uv[:, 1].max() - t_uv[:, 1].min()),
        1.0,
    )
    pad = 0.35 * max_w + 0.15 * span
    bounds = (
        float(t_uv[:, 0].min()) - pad,
        float(t_uv[:, 0].max()) + pad,
        float(t_uv[:, 1].min()) - pad,
        float(t_uv[:, 1].max()) + pad,
    )
    mesh_mask = _rasterize_triangles_2d(
        mesh_v[:, [ax_u, ax_v]], tris, grid_size, bounds
    )
    if plane_name == 'xz':
      side_hole_fraction = _interior_hole_fraction(mesh_mask)

    uu, vv = np.meshgrid(
        np.linspace(bounds[0], bounds[1], grid_size),
        np.linspace(bounds[2], bounds[3], grid_size),
    )
    pixel = max(
        (bounds[1] - bounds[0]) / max(grid_size - 1, 1),
        (bounds[3] - bounds[2]) / max(grid_size - 1, 1),
    )
    union_loose = np.zeros((grid_size, grid_size), dtype=bool)
    for name, stick in sticks.items():
      near, far = stick['proximal'], stick['distal']
      t0_3d, t1_3d = target[near], target[far]
      width = float(stick['width'])
      loose_r = max(0.55 * width, 0.35 + 1.5 * pixel)
      union_loose |= _capsule_mask_2d(
          uu, vv, t0_3d[[ax_u, ax_v]], t1_3d[[ax_u, ax_v]], loose_r
      )
      use_aligned = (
          near in aligned
          and far in aligned
          and float(np.linalg.norm(aligned[far] - aligned[near]))
          >= 0.25 * float(np.linalg.norm(t1_3d - t0_3d))
      )
      p0_3d, p1_3d = (
          (aligned[near], aligned[far]) if use_aligned else (t0_3d, t1_3d)
      )
      p0, p1 = p0_3d[[ax_u, ax_v]], p1_3d[[ax_u, ax_v]]
      len_3d = float(np.linalg.norm(p1_3d - p0_3d))
      if len_3d > 1e-6 and float(np.linalg.norm(p1 - p0)) < 0.35 * len_3d:
        continue
      radius = (
          max(0.38 * width, 0.85 * pixel)
          if width >= 2.0
          else max(0.20 * width, 0.45 * pixel)
      )
      cap = _capsule_mask_2d(uu, vv, p0, p1, radius)
      cap_sum = int(cap.sum())
      if cap_sum > 0:
        recalls_by_stick[name].append(float((mesh_mask & cap).sum() / cap_sum))

    mesh_sum = int(mesh_mask.sum())
    if mesh_sum > 0:
      precisions.append(float((mesh_mask & union_loose).sum() / mesh_sum))

  coverages: dict[str, float] = {}
  weighted_sum = weight_total = 0.0
  wide_covs: list[float] = []
  body_breadths: list[float] = []
  for name, stick in sticks.items():
    recalls = recalls_by_stick[name]
    if not recalls:
      continue
    width = float(stick['width'])
    is_body = bool(stick['is_river'] or name == geometry['root'])
    raw_cov = float(min(recalls)) if is_body else float(max(recalls))
    norm_cov = min(1.0, raw_cov / (0.80 if width >= 2.0 else 0.45))
    coverages[name] = round(norm_cov, 3)
    weight = float(stick['length']) * (2.0 if width >= 2.0 else 1.0)
    weighted_sum += weight * norm_cov
    weight_total += weight
    if width >= 2.0:
      wide_covs.append(norm_cov)
    near, far = stick['proximal'], stick['distal']
    if is_body and near in aligned and far in aligned:
      body_breadths.append(
          _local_segment_breadth(
              mesh_v, aligned[near], aligned[far], width, norm_cov
          )
      )

  recall = weighted_sum / weight_total if weight_total > 0.0 else 1.0
  precision = float(np.mean(precisions)) if precisions else 1.0
  wide = float(np.mean(wide_covs)) if wide_covs else recall
  hole_penalty = max(0.0, min(0.35, 1.2 * max(0.0, side_hole_fraction - 0.08)))
  silhouette = max(
      0.0,
      min(
          1.0,
          (0.45 * recall + 0.35 * wide + 0.20 * precision)
          * (1.0 - hole_penalty),
      ),
  )
  body_vol = float(np.mean(body_breadths)) if body_breadths else 0.0
  return {
      'silhouette': silhouette,
      'silhouette_recall': round(recall, 4),
      'silhouette_precision': round(precision, 4),
      'wide_element_coverage': round(wide, 4),
      'side_hole_fraction': round(side_hole_fraction, 4),
      'stick_silhouette_coverage': coverages,
      'local_body_volume': round(body_vol, 4),
  }


def _score_direction(
    sticks: Mapping[str, Mapping[str, Any]],
    root: str,
    aligned: Mapping[str, np.ndarray],
    target: Mapping[str, np.ndarray],
    size: float,
    config: FidelityConfig,
) -> tuple[float, dict[str, float], float, list[str]]:
  """Scores length-weighted direction accuracy and non-root stray flaps."""
  min_len = _MIN_FOLDED_FRACTION * size
  errors: dict[str, float] = {}
  within = total = stray_len = flap_total = 0.0
  stray_names: list[str] = []
  for name, stick in sticks.items():
    near, far = stick['proximal'], stick['distal']
    if near not in aligned or far not in aligned:
      continue
    vec = aligned[far] - aligned[near]
    tgt = target[far] - target[near]
    collapsed = float(np.linalg.norm(vec)) < min_len
    err = 90.0 if collapsed else pose_alignment.angle_deg(vec, tgt)
    rounded = round(err, 1)
    errors[name] = rounded
    length = float(stick['length'])
    total += length
    if err <= config.direction_tolerance_deg:
      within += length
    if not stick['is_river'] and name != root:
      flap_total += length
      if err > config.stray_angle_deg:
        stray_len += length
        stray_names.append(name)
  stray_names.sort(key=lambda n: -errors[n])
  score = within / total if total > 0.0 else 0.0
  stray_frac = stray_len / flap_total if flap_total > 0.0 else 0.0
  return score, errors, stray_frac, stray_names


def _score_articulation(
    geometry: Mapping[str, Any],
    aligned: Mapping[str, np.ndarray],
    target: Mapping[str, np.ndarray],
    fit: pose_alignment.Similarity,
    config: FidelityConfig,
) -> tuple[float, dict[str, float]]:
  """Scores bend articulation along bilaterally paired appendage chains."""
  sticks = geometry['sticks']
  sections = geometry['sections']

  def _chain_points(
      chain: Sequence[str], *, use_target: bool
  ) -> list[np.ndarray]:
    pts: list[np.ndarray] = []
    for idx, name in enumerate(chain):
      near, far = sticks[name]['proximal'], sticks[name]['distal']
      if use_target:
        if idx == 0:
          pts.append(target[near])
        pts.append(target[far])
        continue
      if idx == 0:
        if near not in aligned:
          return []
        pts.append(aligned[near])
      for frac in paper_layout.SECTION_FRACTIONS:
        sec = sections.get(f'{name}@{frac:g}')
        if sec is not None and sec.get('folded') is not None:
          pts.append(fit.apply(np.asarray(sec['folded'], dtype=float)[None])[0])
      if far not in aligned:
        return []
      pts.append(aligned[far])
    return pts

  limb_scores: dict[str, float] = {}
  for merged_name, chain in geometry['chains'].items():
    if not any(sticks[n]['symmetric'] for n in chain):
      continue
    folded_pts = _chain_points(chain, use_target=False)
    if len(folded_pts) < 3:
      continue
    folded_bend = max(_polyline_turns_deg(folded_pts), default=0.0)
    target_bend = max(
        _polyline_turns_deg(_chain_points(chain, use_target=True)), default=0.0
    )
    min_bend = config.min_visible_bend_deg
    score = (
        0.0
        if folded_bend < min_bend
        else min(1.0, folded_bend / max(target_bend, min_bend))
    )
    limb_scores[merged_name] = round(score, 3)
  mean_score = (
      float(np.mean(list(limb_scores.values()))) if limb_scores else 1.0
  )
  return mean_score, limb_scores


def _score_separation_and_symmetry(
    geometry: Mapping[str, Any],
    aligned: Mapping[str, np.ndarray],
    target: Mapping[str, np.ndarray],
    config: FidelityConfig,
) -> tuple[float, float, list[str], list[str], list[str]]:
  """Scores tip clumping, mirrored pair separation, and bilateral symmetry."""
  sticks = geometry['sticks']
  sections = geometry['sections']
  tips: dict[str, float] = {}
  for chain in geometry['chains'].values():
    tip = sticks[chain[-1]]['distal']
    if tip in aligned:
      tips[tip] = float(sum(sticks[n]['length'] for n in chain))

  clumped: list[str] = []
  for tip, length in tips.items():
    others = [o for o in tips if o != tip]
    if not others:
      continue
    f_gap = min(
        float(np.linalg.norm(aligned[tip] - aligned[o])) for o in others
    )
    t_gap = min(float(np.linalg.norm(target[tip] - target[o])) for o in others)
    if f_gap < 0.35 * length and f_gap < 0.5 * t_gap:
      clumped.append(tip)
  tip_share = 1.0 - len(clumped) / len(tips) if tips else 1.0

  mirror = np.array([1.0, -1.0, 1.0], dtype=float)
  pair_scores: list[float] = []
  sym_terms: list[float] = []
  stacked: list[str] = []
  asym_widths: list[str] = []
  for name, stick in sticks.items():
    other = stick['symmetric']
    if not other or other not in sticks or name > other:
      continue
    ja, jb = stick['distal'], sticks[other]['distal']
    if ja not in aligned or jb not in aligned:
      continue
    t_gap = float(np.linalg.norm(target[ja] - target[jb]))
    if t_gap < 1e-6:
      continue
    f_gap = float(np.linalg.norm(aligned[ja] - aligned[jb]))
    ratio = min(1.0, f_gap / t_gap)
    pair_scores.append(ratio)
    if ratio < 0.5:
      stacked.append(f'{name}/{other}')
    mirror_err = float(np.linalg.norm(aligned[ja] - aligned[jb] * mirror))
    pair_sym = ratio * max(0.0, 1.0 - mirror_err / t_gap)
    sec_a = sections.get(f'{name}@0.5')
    sec_b = sections.get(f'{other}@0.5')
    if (
        sec_a is not None
        and sec_b is not None
        and not stick['is_river']
        and not sticks[other]['is_river']
    ):
      w_a = max(float(sec_a['width']), 1e-6)
      w_b = max(float(sec_b['width']), 1e-6)
      w_lo, w_hi = min(w_a, w_b), max(w_a, w_b)
      if w_hi / w_lo > config.asymmetric_width_ratio:
        asym_widths.append(f'{name}/{other}')
        pair_sym *= 0.25 + 0.75 * (w_lo / w_hi)
    sym_terms.append(pair_sym)

  pair_share = float(np.mean(pair_scores)) if pair_scores else 1.0
  symmetry = float(np.mean(sym_terms)) if sym_terms else 1.0
  return tip_share * pair_share, symmetry, clumped, stacked, asym_widths


def _score_width(
    geometry: Mapping[str, Any],
) -> tuple[float, dict[str, float]]:
  """Scores flap mid-section widths against stick-figure design widths."""
  widths: dict[str, float] = {}
  weighted = total_len = 0.0
  for name, stick in geometry['sticks'].items():
    sec = geometry['sections'].get(f'{name}@0.5')
    if sec is None or stick['is_river'] or name == geometry['root']:
      continue
    folded_w = max(float(sec['width']), 1e-6)
    target_w = float(stick['width'])
    widths[name] = round(folded_w, 3)
    ratio = min(folded_w, target_w) / max(folded_w, target_w)
    length = float(stick['length'])
    weighted += length * ratio
    total_len += length
  return (weighted / total_len if total_len > 0.0 else 1.0), widths


def cleanliness_score(
    vertex_error: float = 0.0, non_ff_vertices: int = 0
) -> float:
  """Returns fold cleanliness in `[0, 1]` from FF violations and strain."""
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
    vertex_error: float = 0.0,
    non_ff_vertices: int = 0,
    *,
    config: FidelityConfig = DEFAULT_CONFIG,
) -> CleanGate:
  """Evaluates the hard fold-cleanliness gate."""
  count = int(non_ff_vertices)
  verr = float(vertex_error)
  return CleanGate(
      passed=count == 0 and verr <= config.clean_gate_max_vertex_error,
      locally_flat_foldable=count == 0,
      non_flat_foldable_vertices=count,
      vertex_error_percent=verr,
  )


def pose_gate(
    component_scores: Mapping[str, float],
    details: Mapping[str, Any],
    *,
    config: FidelityConfig = DEFAULT_CONFIG,
) -> PoseGate:
  """Evaluates the hard pose gate (direction, strays, and L/R crossings)."""
  stray = tuple(details.get('stray_flaps', ()))
  crossing = tuple(
      pose_alignment.crossing_sticks(details.get('side_crossings') or ())
  )
  direction = float(component_scores.get('direction', 0.0))
  passed = (
      direction >= config.pose_gate_direction
      and len(stray) <= config.pose_gate_max_stray
      and not crossing
  )
  return PoseGate(
      passed=passed,
      direction=direction,
      stray=stray,
      crossing=crossing,
  )


def components(
    geometry: Mapping[str, Any],
    *,
    layout: paper_layout.PaperLayout | None = None,
    model: fold_3d.FoldedModel | None = None,
    vertex_error: float = 0.0,
    non_ff_vertices: int = 0,
    config: FidelityConfig = DEFAULT_CONFIG,
) -> tuple[dict[str, float], dict[str, Any], AlignmentInfo]:
  """Computes `[0, 1]` component scores, diagnostic details, and alignment."""
  sticks = geometry['sticks']
  joints = geometry['joints']
  located = sorted(n for n, j in joints.items() if j['folded'] is not None)
  if len(located) < 3:
    raise ValueError(f'Only {len(located)} joints located on the paper.')

  source = np.asarray([joints[n]['folded'] for n in located], dtype=float)
  goal = np.asarray([joints[n]['target'] for n in located], dtype=float)
  size = max(
      math.sqrt(float(((goal - goal.mean(axis=0)) ** 2).sum(axis=1).mean())),
      1e-9,
  )
  fit, robust_weights = pose_alignment.fit_similarity_robust(source, goal, size)
  aligned = dict(zip(located, fit.apply(source), strict=True))
  target = {n: np.asarray(j['target'], dtype=float) for n, j in joints.items()}
  joint_errors = {
      n: float(np.linalg.norm(aligned[n] - target[n]) / size) for n in located
  }
  pose_score = 0.5 * float(
      np.mean([e < 0.2 for e in joint_errors.values()])
      + np.mean([e < 0.35 for e in joint_errors.values()])
  )
  frame_kind, render_rot, frame_reason = pose_alignment.compute_render_frame(
      geometry, located, source, goal, robust_weights
  )

  direction, dir_errors, stray_frac, stray_flaps = _score_direction(
      sticks, geometry['root'], aligned, target, size, config
  )
  articulation, limb_scores = _score_articulation(
      geometry, aligned, target, fit, config
  )
  separation, symmetry, clumped, stacked, asym_widths = (
      _score_separation_and_symmetry(geometry, aligned, target, config)
  )
  width_score, widths = _score_width(geometry)

  folded_flat = _flatness(
      np.asarray([aligned[n] for n in located], dtype=float)
  )
  target_flat = _flatness(goal)
  depth_score = (
      1.0
      if target_flat < config.planar_figure_flatness
      else min(1.0, folded_flat / max(target_flat, 0.05))
  )

  body = geometry['body_extents']
  body_thickness = float(body[2] / body[0]) if body[0] > 0.0 else 0.0
  extra_details: dict[str, Any] = {}
  if model is not None and layout is not None:
    sil = _silhouette_and_body_metrics(
        geometry,
        aligned,
        target,
        fit,
        model,
        layout.tree_units_per_paper_unit,
        config.silhouette_grid_size,
    )
    silhouette = float(sil['silhouette'])
    volume = float(sil['local_body_volume'])
    extra_details = {k: v for k, v in sil.items() if k != 'silhouette'}
  else:
    silhouette = float(width_score)
    volume = min(1.0, body_thickness / 0.2)

  clean = cleanliness_score(vertex_error, non_ff_vertices)
  mid_suffix = f'@{pose_alignment.SIDE_SECTION_FRACTION:g}'
  aligned_sections = {
      key: fit.apply(np.asarray(sec['folded'], dtype=float)[None])[0]
      for key, sec in geometry['sections'].items()
      if key.endswith((mid_suffix, '@0', '@0.25'))
      and sec.get('folded') is not None
  }
  crossings = pose_alignment.find_side_crossings(
      sticks,
      geometry['root'],
      target,
      aligned,
      realized_sections=aligned_sections,
  )

  scores = {
      'pose': pose_score,
      'direction': direction,
      'articulation': articulation,
      'separation': separation,
      'width': width_score,
      'depth': depth_score,
      'symmetry': symmetry,
      'volume': volume,
      'silhouette': silhouette,
      'cleanliness': clean,
  }
  details: dict[str, Any] = {
      'direction_error_deg': dir_errors,
      'stray_fraction': stray_frac,
      'stray_flaps': stray_flaps,
      'limb_articulation': limb_scores,
      'clumped_tips': clumped,
      'stacked_pairs': stacked,
      'side_crossings': [c.to_dict() for c in crossings],
      'asymmetric_width_pairs': asym_widths,
      'widths': widths,
      'joint_error': {k: round(v, 4) for k, v in joint_errors.items()},
      'depth_ratio': round(folded_flat, 4),
      'target_depth_ratio': round(target_flat, 4),
      'body_thickness': round(body_thickness, 4),
      'mirrored': fit.reflected,
      'joints_located': len(located),
      'joints_total': len(joints),
      **extra_details,
  }
  alignment = AlignmentInfo(
      rotation=fit.rotation,
      scale=fit.scale,
      translation=fit.translation,
      render_rotation=render_rot,
      render_frame=frame_kind,
      render_frame_reason=frame_reason,
  )
  return scores, details, alignment


def combine(
    scores: Mapping[str, float],
    weights: Mapping[str, float] | None = None,
) -> float:
  """Combines `[0, 1]` component scores into a 0-10 weighted geometric mean."""
  active = DEFAULT_WEIGHTS if weights is None else weights
  total = float(sum(active.values()))
  if total <= 0.0:
    return 0.0
  log_sum = sum(
      float(w) * math.log(max(_COMPONENT_FLOOR, min(1.0, float(scores[k]))))
      for k, w in active.items()
  )
  return 10.0 * math.exp(log_sum / total)


def score_model(
    tree: stick_figure_lib.StickFigure,
    packing: packing_output_lib.PackingOutput | bus.PackedCreasePattern,
    model: fold_3d.FoldedModel,
    *,
    layout: paper_layout.PaperLayout | None = None,
    pattern: cp_lib.CreasePattern | None = None,
    config: FidelityConfig = DEFAULT_CONFIG,
    include_geometry: bool = False,
) -> FidelityReport:
  """Scores a pre-folded `FoldedModel` against its source `StickFigure`."""
  start = time.perf_counter()
  resolved_layout = (
      paper_layout.build_paper_layout(tree, packing)
      if layout is None
      else layout
  )
  if pattern is not None:
    _, _, non_ff = pattern.vertex_flat_foldability_percentage()
    non_ff_count = len(non_ff)
  else:
    non_ff_count = int(model.non_ff_vertices)

  geometry = measure(tree, resolved_layout, model)
  scores, details, alignment = components(
      geometry,
      layout=resolved_layout,
      model=model,
      vertex_error=model.vertex_error,
      non_ff_vertices=non_ff_count,
      config=config,
  )
  return FidelityReport(
      fidelity=round(combine(scores, config.weights), 2),
      components=scores,
      pose_gate=pose_gate(scores, details, config=config),
      clean_gate=clean_gate(model.vertex_error, non_ff_count, config=config),
      alignment=alignment,
      details=details,
      fold_vertex_error_percent=model.vertex_error,
      fold_seconds=model.seconds,
      score_seconds=time.perf_counter() - start,
      geometry=geometry if include_geometry else None,
  )


def score_pattern(
    tree: stick_figure_lib.StickFigure,
    packing: packing_output_lib.PackingOutput | bus.PackedCreasePattern,
    pattern: cp_lib.CreasePattern,
    *,
    layout: paper_layout.PaperLayout | None = None,
    config: FidelityConfig = DEFAULT_CONFIG,
    include_geometry: bool = False,
) -> FidelityReport:
  """Folds `pattern` in 3D and scores it against `tree`."""
  model = fold_3d.fold_3d(pattern, check_flat_foldability=True)
  return score_model(
      tree,
      packing,
      model,
      layout=layout,
      config=config,
      include_geometry=include_geometry,
  )


def stick_directions(
    tree: stick_figure_lib.StickFigure,
    packing: packing_output_lib.PackingOutput | bus.PackedCreasePattern,
    model: fold_3d.FoldedModel,
    *,
    layout: paper_layout.PaperLayout | None = None,
    config: FidelityConfig = DEFAULT_CONFIG,
) -> tuple[FidelityReport, dict[str, dict[str, Any]]]:
  """Returns `(report, per_stick_directions)` in a proper right-handed frame."""
  full_report = score_model(
      tree, packing, model, layout=layout, config=config, include_geometry=True
  )
  geometry = full_report.geometry
  if geometry is None:
    raise RuntimeError(
        'score_model(include_geometry=True) returned no geometry.'
    )
  align = full_report.alignment
  rot, scale, trans = align.rotation, align.scale, align.translation
  joints = geometry['joints']

  if full_report.mirrored:
    located = sorted(n for n, j in joints.items() if j['folded'] is not None)
    src = np.asarray([joints[n]['folded'] for n in located], dtype=float)
    goal = np.asarray([joints[n]['target'] for n in located], dtype=float)
    size = max(
        math.sqrt(float(((goal - goal.mean(axis=0)) ** 2).sum(axis=1).mean())),
        1e-9,
    )
    proper, _ = pose_alignment.fit_similarity_robust(
        src, goal, size, allow_reflection=False
    )
    rot, scale, trans = proper.rotation, proper.scale, proper.translation

  dir_errors = full_report.details['direction_error_deg']
  directions: dict[str, dict[str, Any]] = {}
  for name, stick in geometry['sticks'].items():
    near, far = joints[stick['proximal']], joints[stick['distal']]
    tgt_vec = np.asarray(far['target'], dtype=float) - np.asarray(
        near['target'], dtype=float
    )
    realized = None
    err = dir_errors.get(name)
    if near['folded'] is not None and far['folded'] is not None:
      a = scale * (rot @ np.asarray(near['folded'], dtype=float)) + trans
      b = scale * (rot @ np.asarray(far['folded'], dtype=float)) + trans
      realized = pose_alignment.azimuth_elevation(b - a)
      if full_report.mirrored and err is not None:
        err = round(pose_alignment.angle_deg(b - a, tgt_vec), 1)
    directions[name] = {
        'target': pose_alignment.azimuth_elevation(tgt_vec),
        'realized': realized,
        'error': err,
    }
  return dataclasses.replace(full_report, geometry=None), directions


def evaluate_fidelity(
    target: (
        bus.SolvedCreasePattern | bus.ShapedCreasePattern | bus.PipelineResult
    ),
    *,
    tree: stick_figure_lib.StickFigure | bus.DesignTree | None = None,
    config: FidelityConfig = DEFAULT_CONFIG,
    include_geometry: bool = False,
) -> FidelityReport:
  """Evaluates 0-10 geometric pose fidelity on a `bp_pipeline` stage record.

  Args:
    target: A `SolvedCreasePattern`, `ShapedCreasePattern`, or `PipelineResult`.
    tree: Optional unmerged `StickFigure` or `DesignTree` override. Defaults to
      `target.packed.design.source`.
    config: Thresholds and component weights for scoring.
    include_geometry: Whether to populate `FidelityReport.geometry`.

  Returns:
    Typed `FidelityReport` containing headline score, component breakdown,
    pose/clean gates, alignment matrices, and per-stick diagnostics.
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
        'Expected SolvedCreasePattern, ShapedCreasePattern, or PipelineResult; '
        f'got {type(target)!r}.'
    )

  packed = solved.packed
  packing_ref: packing_output_lib.PackingOutput | bus.PackedCreasePattern = (
      bus_adapter.legacy_inputs(solved)[0]
      if solved.central_crease_removed
      else packed
  )
  if isinstance(tree, bus.DesignTree):
    resolved_tree: stick_figure_lib.StickFigure | None = tree.source
  elif isinstance(tree, stick_figure_lib.StickFigure):
    resolved_tree = tree
  elif packed.design is not None:
    resolved_tree = packed.design.source
  else:
    resolved_tree = packed.packed_tree
  if resolved_tree is None:
    raise ValueError(
        'evaluate_fidelity() requires a DesignTree on `target` or an explicit '
        '`tree=` argument.'
    )
  return score_pattern(
      resolved_tree,
      packing_ref,
      pattern,
      config=config,
      include_geometry=include_geometry,
  )


__all__ = [
    'AlignmentInfo',
    'CleanGate',
    'DEFAULT_CONFIG',
    'DEFAULT_WEIGHTS',
    'FidelityConfig',
    'FidelityReport',
    'PoseGate',
    'clean_gate',
    'cleanliness_score',
    'combine',
    'components',
    'evaluate_fidelity',
    'infer_symmetric_stick',
    'measure',
    'pose_gate',
    'score_model',
    'score_pattern',
    'stick_directions',
]
