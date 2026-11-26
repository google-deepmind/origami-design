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

"""3D pose similarity alignment, camera frame, and root-frame diagnostics."""

from collections.abc import Mapping, Sequence
import dataclasses
import math
from typing import Any, Literal

import numpy as np

from .. import folder_3d as fold_3d
from ..stick_figures import stick_figure as stick_figure_lib
from . import paper_layout

SIDE_MIN_DESIGN_OFFSET = 0.5
SIDE_CROSSING_MARGIN = 0.25
SIDE_SECTION_FRACTION = 0.5

_ROBUST_CAUCHY_SCALE = 0.25
_ROBUST_ITERATIONS = 8


@dataclasses.dataclass(frozen=True, slots=True)
class Similarity:
  """3D similarity transform `target ~ scale * rotation @ source + shift`."""

  rotation: np.ndarray
  scale: float
  translation: np.ndarray
  reflected: bool

  def apply(self, points: np.ndarray) -> np.ndarray:
    """Applies this similarity transform to `(..., 3)` points."""
    return self.scale * np.asarray(points) @ self.rotation.T + self.translation


@dataclasses.dataclass(frozen=True, slots=True)
class SideCrossing:
  """A stick point folded onto the wrong side (or midline) of the body plane."""

  stick: str
  point: Literal['tip', 'mid', 'proximal']
  kind: Literal['crossed', 'midline']
  design_side: Literal['left', 'right']
  design_offset: float
  realized_offset: float

  def to_dict(self) -> dict[str, Any]:
    return dataclasses.asdict(self)


def angle_deg(first: np.ndarray, second: np.ndarray) -> float:
  """Returns the unsigned angle in degrees between two vectors."""
  denom = float(np.linalg.norm(first) * np.linalg.norm(second))
  if denom <= 0.0:
    return 180.0
  cosine = max(-1.0, min(1.0, float(np.dot(first, second)) / denom))
  return math.degrees(math.acos(cosine))


def azimuth_elevation(
    vector: Sequence[float] | np.ndarray,
) -> tuple[float, float] | None:
  """Returns `(azimuth_deg, elevation_deg)` in figure frame, or `None`."""
  vec = np.asarray(vector, dtype=float)
  norm = float(np.linalg.norm(vec))
  if norm < 1e-9:
    return None
  az = math.degrees(math.atan2(float(vec[1]), float(vec[0])))
  el = math.degrees(math.asin(max(-1.0, min(1.0, float(vec[2]) / norm))))
  return (round(az, 2), round(el, 2))


def principal_extents(points: np.ndarray) -> np.ndarray:
  """Returns descending principal standard deviations `(s1, s2, s3)`."""
  if len(points) < 3:
    return np.zeros(3, dtype=float)
  centered = points - points.mean(axis=0)
  values = np.linalg.svd(centered, compute_uv=False) / math.sqrt(len(points))
  return np.pad(values, (0, max(0, 3 - len(values))))[:3]


def dominant_axis(normals: np.ndarray) -> np.ndarray | None:
  """Returns the principal unit axis of unoriented 3D normal vectors."""
  if len(normals) == 0:
    return None
  norms = np.linalg.norm(normals, axis=1, keepdims=True)
  valid = norms[:, 0] > 1e-12
  if not np.any(valid):
    return None
  unit = normals[valid] / norms[valid]
  _, eigenvectors = np.linalg.eigh(unit.T @ unit)
  return eigenvectors[:, -1]


def fit_similarity(
    source: np.ndarray,
    target: np.ndarray,
    *,
    weights: np.ndarray | None = None,
    allow_reflection: bool = True,
) -> Similarity:
  """Fits a weighted least-squares Umeyama similarity transform."""
  src = np.asarray(source, dtype=float)
  tgt = np.asarray(target, dtype=float)
  w = np.ones(len(src), dtype=float) if weights is None else np.asarray(weights)
  w_sum = float(w.sum())
  w = w / w_sum if w_sum > 0.0 else np.full(len(src), 1.0 / max(len(src), 1))

  src_mean = w @ src
  tgt_mean = w @ tgt
  src_c = src - src_mean
  tgt_c = tgt - tgt_mean
  cov = (tgt_c * w[:, None]).T @ src_c
  u, singular, vt = np.linalg.svd(cov)
  src_var = float(w @ np.sum(src_c * src_c, axis=1))
  det_sign = 1.0 if float(np.linalg.det(u @ vt)) >= 0.0 else -1.0

  options: list[tuple[np.ndarray, bool]] = [
      (np.diag([1.0, 1.0, det_sign]), False)
  ]
  if allow_reflection:
    options.append((np.diag([1.0, 1.0, -det_sign]), True))

  best: Similarity | None = None
  best_residual = math.inf
  for sign_mat, reflected in options:
    rotation = u @ sign_mat @ vt
    scale = (
        float(np.sum(singular * np.diag(sign_mat))) / src_var
        if src_var > 0.0
        else 0.0
    )
    translation = tgt_mean - scale * (rotation @ src_mean)
    cand = Similarity(rotation, scale, translation, reflected)
    residual = float(w @ np.sum((cand.apply(src) - tgt) ** 2, axis=1))
    if best is None or residual < best_residual - 1e-12:
      best = cand
      best_residual = residual
  if best is None:
    raise RuntimeError('Similarity candidates cannot be empty.')
  return best


def fit_similarity_robust(
    source: np.ndarray,
    target: np.ndarray,
    size: float,
    *,
    allow_reflection: bool = True,
    cauchy_scale: float = _ROBUST_CAUCHY_SCALE,
    iterations: int = _ROBUST_ITERATIONS,
) -> tuple[Similarity, np.ndarray]:
  """Fits an outlier-resistant similarity via Cauchy-reweighted IRLS."""
  src = np.asarray(source, dtype=float)
  tgt = np.asarray(target, dtype=float)
  denom = max(float(size), 1e-9)
  weights = np.ones(len(src), dtype=float)
  fit = fit_similarity(src, tgt, allow_reflection=allow_reflection)
  for _ in range(iterations):
    residual = np.linalg.norm(fit.apply(src) - tgt, axis=1) / denom
    weights = 1.0 / (1.0 + (residual / cauchy_scale) ** 2)
    fit = fit_similarity(
        src, tgt, weights=weights, allow_reflection=allow_reflection
    )
  return fit, weights


def _orthonormal_frame(
    primary: np.ndarray, secondary: np.ndarray
) -> np.ndarray | None:
  """Builds a right-handed 3x3 column basis from two vectors, or `None`."""
  p_norm = float(np.linalg.norm(primary))
  if p_norm < 1e-9:
    return None
  e1 = primary / p_norm
  e3 = secondary - float(np.dot(secondary, e1)) * e1
  e3_norm = float(np.linalg.norm(e3))
  if e3_norm < 1e-6:
    return None
  e3 = e3 / e3_norm
  return np.stack([e1, np.cross(e3, e1), e3], axis=1)


def compute_render_frame(
    geometry: Mapping[str, Any],
    located: Sequence[str],
    source: np.ndarray,
    goal: np.ndarray,
    robust_weights: np.ndarray,
) -> tuple[Literal['stick_figure', 'root', 'native'], np.ndarray, str]:
  """Returns `(frame_kind, proper_3x3_rotation, fallback_reason)`."""
  extents = principal_extents(source)
  reason = ''
  if len(located) < 4:
    reason = f'only {len(located)} joints located'
  elif extents[0] <= 0.0 or extents[1] < 0.05 * extents[0]:
    reason = 'folded joints are collinear (the model is a bar)'
  elif float(robust_weights.sum()) < 3.0:
    reason = 'fewer than 3 joints agree with the stick figure'
  if not reason:
    proper = fit_similarity(
        source, goal, weights=robust_weights, allow_reflection=False
    )
    return ('stick_figure', proper.rotation, '')

  sticks = geometry['sticks']
  joints = geometry['joints']
  root = sticks[geometry['root']]
  near, far = joints[root['proximal']], joints[root['distal']]
  raw_normal = geometry.get('root_normal')
  if near['folded'] is None or far['folded'] is None or raw_normal is None:
    return (
        'native',
        np.eye(3, dtype=float),
        f'{reason}; root frame unavailable',
    )

  folded_axis = np.asarray(far['folded'], dtype=float) - np.asarray(
      near['folded'], dtype=float
  )
  target_axis = np.asarray(far['target'], dtype=float) - np.asarray(
      near['target'], dtype=float
  )
  normal = np.asarray(raw_normal, dtype=float)
  heights = (source - source.mean(axis=0)) @ normal
  if float(np.dot(heights, goal[:, 2] - goal[:, 2].mean())) < 0.0:
    normal = -normal

  model_basis = _orthonormal_frame(folded_axis, normal)
  figure_basis = _orthonormal_frame(
      target_axis, np.array([0.0, 0.0, 1.0], dtype=float)
  )
  if model_basis is None or figure_basis is None:
    return (
        'native',
        np.eye(3, dtype=float),
        f'{reason}; root frame degenerate',
    )
  return ('root', figure_basis @ model_basis.T, reason)


def find_side_crossings(
    sticks: Mapping[str, Mapping[str, Any]],
    root: str,
    design_joints: Mapping[str, Any],
    realized_joints: Mapping[str, Any],
    *,
    realized_sections: Mapping[str, Any] | None = None,
    min_offset: float = SIDE_MIN_DESIGN_OFFSET,
    margin: float = SIDE_CROSSING_MARGIN,
    section_fraction: float = SIDE_SECTION_FRACTION,
) -> list[SideCrossing]:
  """Detects lateral stick points that crossed or collapsed to the midline."""
  root_joints = (sticks[root]['proximal'], sticks[root]['distal'])

  def _midline_y(pts: Mapping[str, Any]) -> float | None:
    ys = [float(pts[j][1]) for j in root_joints if pts.get(j) is not None]
    return float(np.mean(ys)) if ys else None

  design_mid = _midline_y(design_joints)
  realized_mid = _midline_y(realized_joints)
  if design_mid is None or realized_mid is None:
    return []

  crossings: list[SideCrossing] = []
  for name, stick in sticks.items():
    if name == root or stick.get('shaping_flap'):
      continue
    near, far = stick['proximal'], stick['distal']
    if near not in design_joints or far not in design_joints:
      continue
    d_near = np.asarray(design_joints[near], dtype=float)
    d_far = np.asarray(design_joints[far], dtype=float)
    near_on_axis = abs(float(d_near[1]) - design_mid) < min_offset

    checks: list[
        tuple[Literal['tip', 'mid', 'proximal'], np.ndarray, Any, float]
    ] = [('tip', d_far, realized_joints.get(far), margin)]
    if realized_sections is not None:
      mid_thresh = -margin if near_on_axis else margin
      checks.append((
          'mid',
          d_near + section_fraction * (d_far - d_near),
          realized_sections.get(f'{name}@{section_fraction:g}'),
          mid_thresh,
      ))
      prox = realized_sections.get(f'{name}@0')
      if prox is None:
        prox = realized_sections.get(f'{name}@0.25')
      if prox is not None:
        checks.append(('proximal', d_far, prox, -margin))

    for pt_kind, d_pt, r_pt, thresh in checks:
      offset = float(d_pt[1]) - design_mid
      if abs(offset) < min_offset or r_pt is None:
        continue
      folded_offset = float(r_pt[1]) - realized_mid
      own_side = folded_offset * math.copysign(1.0, offset)
      if own_side < thresh:
        crossings.append(
            SideCrossing(
                stick=name,
                point=pt_kind,
                kind='crossed' if own_side < -margin else 'midline',
                design_side='left' if offset > 0.0 else 'right',
                design_offset=round(offset, 3),
                realized_offset=round(folded_offset, 3),
            )
        )

  crossings.sort(
      key=lambda c: c.realized_offset * math.copysign(1.0, c.design_offset)
  )
  return crossings


def crossing_sticks(
    crossings: Sequence[SideCrossing | Mapping[str, Any]],
) -> list[str]:
  """Returns unique stick names from `crossings`, worst first."""
  names = (
      c.stick if isinstance(c, SideCrossing) else str(c['stick'])
      for c in crossings
  )
  return list(dict.fromkeys(names))


def _figure_to_stick_coords(vec: np.ndarray) -> np.ndarray:
  """Converts figure-frame `(x=fwd, y=left, z=up)` to `Stick.get_coords`."""
  return np.array([-vec[1], vec[0], vec[2]], dtype=float)


def _stick_to_figure_coords(vec: np.ndarray) -> np.ndarray:
  """Converts `Stick.get_coords` `(az0=+y, az90=-x)` to figure frame."""
  return np.array([vec[1], -vec[0], vec[2]], dtype=float)


def root_frame_directions(
    tree: stick_figure_lib.StickFigure,
    layout: paper_layout.PaperLayout,
    model: fold_3d.FoldedModel,
    base_model: fold_3d.FoldedModel,
    *,
    root_name: str,
    root_points: np.ndarray,
    up_reference: np.ndarray | None,
    root_alignment: np.ndarray | None = None,
    section: float = 0.5,
    root_axis_points: tuple[np.ndarray, np.ndarray] | None = None,
) -> dict[str, Any] | None:
  """Measures per-stick directions directly in the unshaped root frame.

  Unlike similarity-fitted directions, this tracks rigid root-face transforms
  from `base_model` to `model`, preventing global tilt or mirror flips from
  biasing closed-loop angle corrections.

  Args:
    tree: Unmerged source `StickFigure`.
    layout: Precomputed `PaperLayout` for `tree`.
    model: Folded shaped model.
    base_model: Folded unshaped base model.
    root_name: Root stick anchoring the frame.
    root_points: `(N, 2)` unit-paper sample points inside the root element.
    up_reference: Optional `+z` reference vector in `base_model` coordinates.
    root_alignment: Optional 3x3 rotation mapping figure to root frame in
      `Stick.get_coords` coordinates.
    section: Cross-section fraction used as proximal anchor for flap axes.
    root_axis_points: Optional `(start_pt, end_pt)` fallback on the unit paper
      when root joints are incident only to rivers.

  Returns:
    Dictionary with `'directions'`, `'positions'`, `'side_crossings'`, and
    `'frame'`, or `None` if the root frame is degenerate.
  """
  root = tree[root_name]
  pts = np.asarray(root_points, dtype=float).reshape(-1, 2)
  base_faces = base_model.locate_faces(pts)
  shaped_faces = model.locate_faces(pts)
  transforms = [
      model.face_matrices[f] @ np.linalg.inv(base_model.face_matrices[b])
      for f, b in zip(shaped_faces, base_faces, strict=True)
      if f >= 0 and b >= 0
  ]
  if not transforms:
    return None

  stacked = np.asarray(transforms, dtype=float)
  ref = np.median(stacked, axis=0)
  agreeing = [
      t for t in stacked if float(np.abs(t[:3, :3] - ref[:3, :3]).max()) < 0.05
  ]
  consensus = np.mean(agreeing, axis=0) if agreeing else ref
  to_base_full = np.linalg.inv(consensus)
  to_base_rot = to_base_full[:3, :3]

  root_joint_pts = {
      j: layout.joints[j] for j in (root.start, root.end) if j in layout.joints
  }
  base_joints, _ = base_model.locate_groups(root_joint_pts)
  if root.start not in base_joints or root.end not in base_joints:
    if root_axis_points is None:
      return None
    loc = base_model.locate(np.asarray(root_axis_points, dtype=float))
    if np.isnan(loc).any():
      return None
    base_joints = {root.start: loc[0], root.end: loc[1]}

  normal = dominant_axis(
      base_model.face_matrices[base_faces[base_faces >= 0]][:, :3, 2]
  )
  if normal is None:
    return None
  if up_reference is not None and float(np.dot(normal, up_reference)) < 0.0:
    normal = -normal

  forward = base_joints[root.end] - base_joints[root.start]
  forward = forward - float(np.dot(forward, normal)) * normal
  fwd_norm = float(np.linalg.norm(forward))
  if fwd_norm < 1e-9:
    return None
  e1 = forward / fwd_norm
  frame = np.stack([e1, np.cross(normal, e1), normal], axis=0)
  alignment = (
      np.eye(3, dtype=float)
      if root_alignment is None
      else np.asarray(root_alignment, dtype=float)
  )

  def _rotate_to_figure(vec_in_root: np.ndarray) -> np.ndarray:
    return _stick_to_figure_coords(
        alignment.T @ _figure_to_stick_coords(vec_in_root)
    )

  folded, _ = model.locate_groups(layout.joints)
  sections, _ = model.locate_groups(layout.sections)
  topo = paper_layout.topology(tree)
  directions: dict[str, dict[str, Any]] = {}
  for name in topo.order:
    stick = tree[name]
    tgt_vec = paper_layout.stick_vector(stick)
    if topo.proximal[name] != stick.start:
      tgt_vec = -tgt_vec
    distal_pt = folded.get(topo.distal[name])
    start_pt = sections.get(f'{name}@{section:g}')
    if start_pt is None:
      start_pt = folded.get(topo.proximal[name])
    realized = None
    err = None
    if name != root_name and distal_pt is not None and start_pt is not None:
      in_fig = _rotate_to_figure(frame @ (to_base_rot @ (distal_pt - start_pt)))
      realized = azimuth_elevation(in_fig)
      if realized is not None:
        err = round(angle_deg(in_fig, tgt_vec), 1)
    directions[name] = {
        'target': azimuth_elevation(tgt_vec),
        'realized': realized,
        'error': err,
    }

  origin = 0.5 * (base_joints[root.start] + base_joints[root.end])
  to_tree = layout.tree_units_per_paper_unit

  def _point_to_figure(pt: np.ndarray) -> list[float]:
    in_base = to_base_rot @ np.asarray(pt, dtype=float) + to_base_full[:3, 3]
    scaled = _rotate_to_figure(frame @ (in_base - origin)) * to_tree
    return [round(float(v), 5) for v in scaled]

  positions = {k: _point_to_figure(v) for k, v in folded.items()}
  mid_suffix = f'@{SIDE_SECTION_FRACTION:g}'
  sec_positions = {
      k: _point_to_figure(v)
      for k, v in sections.items()
      if k.endswith((mid_suffix, '@0', '@0.25'))
  }
  stick_endpoints = {
      n: {
          'proximal': topo.proximal[n],
          'distal': topo.distal[n],
          'shaping_flap': bool(tree[n].shaping_flap),
      }
      for n in topo.order
  }
  crossings = find_side_crossings(
      stick_endpoints,
      root_name,
      paper_layout.stick_figure_joint_positions(tree),
      positions,
      realized_sections=sec_positions,
  )
  return {
      'directions': directions,
      'positions': positions,
      'side_crossings': [c.to_dict() for c in crossings],
      'frame': {
          'root': root_name,
          'agreeing_root_faces': round(len(agreeing) / len(stacked), 3),
          'up_from_reference': up_reference is not None,
      },
  }


__all__ = [
    'SIDE_CROSSING_MARGIN',
    'SIDE_MIN_DESIGN_OFFSET',
    'SIDE_SECTION_FRACTION',
    'SideCrossing',
    'Similarity',
    'angle_deg',
    'azimuth_elevation',
    'compute_render_frame',
    'crossing_sticks',
    'dominant_axis',
    'find_side_crossings',
    'fit_similarity',
    'fit_similarity_robust',
    'principal_extents',
    'root_frame_directions',
]
