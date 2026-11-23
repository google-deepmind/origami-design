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

"""Helpers for composing 2D flat-foldable shaping with 3D relief overlays."""

from collections.abc import Sequence
import copy
import dataclasses
import math

from ..crease_pattern import crease as crease_lib
from ..crease_pattern import crease_pattern as cp_lib
from ..crease_pattern import vertex as vertex_lib

_FOLD_PERCENT_TOL = 1e-5
_OVERLAY_PROXIMITY_TOL = 3e-4


def _is_partial_mv(crease: crease_lib.Crease) -> bool:
  return crease.mountain_or_valley() and not math.isclose(
      crease.fold_percent, 1.0, abs_tol=_FOLD_PERCENT_TOL
  )


def has_3d_overlay(pattern: cp_lib.CreasePattern) -> bool:
  """Returns True if `pattern` has non-Kawasaki vertices or partial folds."""
  if not pattern.local_flat_foldability():
    return True
  return any(_is_partial_mv(c) for c in pattern.creases)


def as_flat_backbone(pattern: cp_lib.CreasePattern) -> cp_lib.CreasePattern:
  """Returns `pattern` with all mountain/valley `fold_percent` set to `1.0`."""
  if not any(_is_partial_mv(c) for c in pattern.creases):
    return pattern
  creases = [
      dataclasses.replace(c, fold_percent=1.0) if _is_partial_mv(c) else c
      for c in pattern.creases
  ]
  return cp_lib.CreasePattern(
      creases=creases, light_mode=True, tolerance=pattern.tolerance
  )


def _point_near_segment(
    px: float,
    py: float,
    crease: crease_lib.Crease,
    tol: float = _OVERLAY_PROXIMITY_TOL,
) -> bool:
  """Returns True if `(px, py)` lies within `tol` of segment `crease`."""
  x1, y1, x2, y2 = crease.p1.x, crease.p1.y, crease.p2.x, crease.p2.y
  dx, dy = x2 - x1, y2 - y1
  len_sq = dx * dx + dy * dy
  if len_sq <= 1e-12:
    return math.hypot(px - x1, py - y1) <= tol
  t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / len_sq))
  return math.hypot(px - (x1 + t * dx), py - (y1 + t * dy)) <= tol


def _point_near_any_vertex(
    px: float,
    py: float,
    vertices: Sequence[vertex_lib.Vertex],
    tol: float = _OVERLAY_PROXIMITY_TOL,
) -> bool:
  return any(
      math.hypot(px - float(v.point.x), py - float(v.point.y)) <= tol
      for v in vertices
  )


def preserves_foldability(
    prior: cp_lib.CreasePattern | None,
    candidate: cp_lib.CreasePattern,
    *,
    allow_new_3d_vertices: bool = False,
) -> tuple[bool, str]:
  """Validates that `candidate` preserves 2D Kawasaki or 3D foldability."""
  pct, _, after_non_ff = candidate.vertex_flat_foldability_percentage()
  if math.isclose(pct, 1.0):
    return (True, 'ok')
  default_err = (
      'Crease pattern is not flat-foldable after shaping: '
      f'{pct:.1%} of vertices satisfy Kawasaki/Maekawa.'
  )
  if prior is None:
    if allow_new_3d_vertices and candidate.is_locally_foldable():
      return (True, 'ok')
    return (False, default_err)

  prior_pct, _, prior_non_ff = prior.vertex_flat_foldability_percentage()
  if (
      math.isclose(prior_pct, 1.0)
      and not has_3d_overlay(prior)
      and not allow_new_3d_vertices
  ):
    return (False, default_err)

  prior_partial = [c for c in prior.creases if _is_partial_mv(c)]
  cand_partial = [c for c in candidate.creases if _is_partial_mv(c)]
  prior_non_3d = [v for v in prior.vertices if v.internal and not v.is_foldable]

  for v in candidate.vertices:
    if not v.internal or v.is_foldable:
      continue
    vx, vy = float(v.point.x), float(v.point.y)
    if _point_near_any_vertex(vx, vy, prior_non_3d):
      continue
    if any(_point_near_segment(vx, vy, c) for c in prior_partial):
      continue
    if not _point_near_any_vertex(vx, vy, after_non_ff) and any(
        _point_near_segment(vx, vy, c) for c in cand_partial
    ):
      continue
    return (
        False,
        (
            'Crease pattern fails 3D Belcastro-Hull vertex foldability after '
            f'shaping at ({vx:.4f}, {vy:.4f}).'
        ),
    )

  if allow_new_3d_vertices:
    return (True, 'ok')

  for v in after_non_ff:
    vx, vy = float(v.point.x), float(v.point.y)
    if _point_near_any_vertex(vx, vy, prior_non_ff):
      continue
    if any(_point_near_segment(vx, vy, c) for c in prior_partial):
      continue
    return (
        False,
        (
            f'New non-flat-foldable vertex at ({vx:.4f}, {vy:.4f}) '
            f'({pct:.1%} Kawasaki).'
        ),
    )
  return (True, 'ok')


def _find_containing_crease(
    seg: crease_lib.Crease, pool: Sequence[crease_lib.Crease]
) -> crease_lib.Crease | None:
  """Returns the first crease in `pool` geometrically containing `seg`."""
  for other in pool:
    if not seg.not_in_encompassing_rectangles(other) and seg.is_within(
        other, match_fold=False
    ):
      return other
  return None


def _merged_mv_creases(
    pattern: cp_lib.CreasePattern,
) -> list[crease_lib.Crease]:
  merged = copy.deepcopy(pattern)
  merged.merge_creases(only_trivial=False)
  return [c for c in merged.creases if c.mountain_or_valley()]


def replay_3d_overlay(
    prev_backbone: cp_lib.CreasePattern,
    prev_pattern: cp_lib.CreasePattern,
    new_backbone: cp_lib.CreasePattern,
) -> cp_lib.CreasePattern:
  """Replays 3D relief creases and partial fold angles onto `new_backbone`."""
  prev_mv = _merged_mv_creases(prev_backbone)
  new_mv = _merged_mv_creases(new_backbone)

  added_3d: list[crease_lib.Crease] = []
  modified_3d: list[tuple[crease_lib.Crease, bool]] = []
  for c in prev_pattern.creases:
    if not c.mountain_or_valley():
      continue
    host = _find_containing_crease(c, prev_mv)
    if host is not None:
      pct_differs = not math.isclose(
          c.fold_percent, host.fold_percent, abs_tol=_FOLD_PERCENT_TOL
      )
      if pct_differs or _is_partial_mv(c):
        modified_3d.append((c, c.fold != host.fold))
    elif _find_containing_crease(c, new_mv) is not None:
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
    mod_tips.append(mc.p1)
    mod_tips.append(mc.p2)

  split_creases: list[crease_lib.Crease] = []
  for c in combined.creases:
    if not c.mountain_or_valley():
      split_creases.append(c)
      continue
    strictly_inside = [
        mc
        for mc in mod_segments
        if not mc.not_in_encompassing_rectangles(c)
        and mc.is_within(c, match_fold=False)
        and not c.is_within(mc, match_fold=False)
    ]
    if strictly_inside:
      try:
        split_creases.extend(
            c.split_to_segments(
                strictly_inside, tips=mod_tips, tolerance=combined.tolerance
            )
        )
        continue
      except ValueError:
        pass
    split_creases.append(c)

  updated: list[crease_lib.Crease] = []
  for c in split_creases:
    if not c.mountain_or_valley():
      updated.append(c)
      continue
    match: tuple[crease_lib.Crease, bool] | None = None
    for mod_c, use_mod_fold in modified_3d:
      if not c.not_in_encompassing_rectangles(mod_c) and c.is_within(
          mod_c, match_fold=False
      ):
        match = (mod_c, use_mod_fold)
        break
    if match is None:
      updated.append(c)
    else:
      mod_c, use_mod_fold = match
      updated.append(
          dataclasses.replace(
              c,
              fold=mod_c.fold if use_mod_fold else c.fold,
              fold_percent=mod_c.fold_percent,
          )
      )
  return cp_lib.CreasePattern(
      creases=updated, light_mode=True, tolerance=new_backbone.tolerance
  )


__all__ = [
    'as_flat_backbone',
    'has_3d_overlay',
    'preserves_foldability',
    'replay_3d_overlay',
]
