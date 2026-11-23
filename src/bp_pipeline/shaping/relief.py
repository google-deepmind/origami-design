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

"""Clip-pattern, central taper, inscribed step, and 3-D relief actuators."""

from collections.abc import Callable, Sequence
import dataclasses
import math
from typing import Any

import numpy as np

from .. import folder_3d
from ..crease_pattern import crease as crease_utils
from ..crease_pattern import crease_pattern as crease_pattern_lib
from ..crease_pattern import point as point_utils
from . import clip_pattern
from . import fidelity
from . import geometric_simple_fold
from . import overlay_3d
from . import paper_layout as paper_layout_lib
from . import shaping_elements
from . import state as state_lib
from . import types as shaping_types
from . import zigzag_shaping

_CLIP_MARGIN = 0.1
_CENTRAL_SAMPLE_STEP = 0.25
_CENTRAL_MIN_MIRROR_MATCH = 0.95
_INSCRIBED_ORDERS: dict[str, list[crease_utils.Fold]] = {
    'MV': [crease_utils.Fold.MOUNTAIN, crease_utils.Fold.VALLEY],
    'VM': [crease_utils.Fold.VALLEY, crease_utils.Fold.MOUNTAIN],
    'M': [crease_utils.Fold.MOUNTAIN],
    'V': [crease_utils.Fold.VALLEY],
}
_OPPOSITE_ORDER = {'MV': 'VM', 'VM': 'MV', 'M': 'V', 'V': 'M'}


def _dedup_floats(values: Sequence[float]) -> list[float]:
  out: list[float] = []
  for raw in values:
    val = float(raw)
    if val >= 0.0 and not any(math.isclose(val, x, abs_tol=1e-6) for x in out):
      out.append(val)
  return out


def _line_geom_key(
    crease: crease_utils.Crease,
) -> tuple[float, float, float, float]:
  p1 = (round(float(crease.p1_init.x), 6), round(float(crease.p1_init.y), 6))
  p2 = (round(float(crease.p2_init.x), 6), round(float(crease.p2_init.y), 6))
  return (*p1, *p2) if p1 <= p2 else (*p2, *p1)


def _dedup_lines(
    creases: Sequence[crease_utils.Crease],
) -> list[crease_utils.Crease]:
  seen: set[tuple[float, float, float, float]] = set()
  out: list[crease_utils.Crease] = []
  for crease in creases:
    key = _line_geom_key(crease)
    if key not in seen:
      seen.add(key)
      out.append(crease)
  return out


def _cancel_paired_lines(
    creases: Sequence[crease_utils.Crease],
) -> list[crease_utils.Crease]:
  counts: dict[tuple[float, float, float, float], int] = {}
  for crease in creases:
    key = _line_geom_key(crease)
    counts[key] = counts.get(key, 0) + 1
  return [
      crease for crease in creases if counts[_line_geom_key(crease)] % 2 == 1
  ]


# ------------------------------------------------------------------------------
# 1. Clip-pattern drawing templates (`fold`, `turn_and_narrow`,
#    `bend_when_turn_and_narrow`, `pleat_to_hide`).
# ------------------------------------------------------------------------------


def _clip_pattern_once(
    state: state_lib.ShapingState,
    *,
    element_name: str,
    template: clip_pattern.TemplateName,
    step: float,
    bend_angle: float,
    rotate_angle: float,
    enable_narrow: bool,
    narrow_n_times: int,
    narrow_symmetric: bool | None,
    pleat_distance: float,
    length: float | None,
    global_refine: bool,
    verify_fold: bool,
) -> state_lib.ToolOutcome:
  """Executes one clip-pattern pass with a fixed `global_refine` mode."""
  flaps, rivers = state.shaping_elements
  river = next((r for r in rivers if r.name == element_name), None)
  flap: shaping_elements.ShapingFlap | None = None
  if river is not None:
    rects = river.covering_rectangles
    target_rects = rects
    river_width: float | None = float(river.width)
    step_from_root = 0.0
  else:
    root_info = state.flap_hierarchy.get_root_flap_and_distance(element_name)
    if root_info is None:
      raise ValueError(
          f'Unknown element {element_name!r}; known: {state.element_names()}.'
      )
    root_name, dist = root_info
    root_flap = next((f for f in flaps if f.name == root_name), None)
    if root_flap is None:
      raise ValueError(f'Root flap {root_name!r} of {element_name!r} missing.')
    elem = state.element(element_name)
    if not isinstance(elem, shaping_elements.ShapingFlap):
      raise ValueError(f'{element_name!r} is not a flap.')
    flap = elem
    rects = [root_flap.to_legacy_rectangle()]
    target_rects = rects if global_refine else [flap.to_legacy_rectangle()]
    river_width = None
    step_from_root = float(dist * state.solving_grid_size)

  adapter = state.clip_pattern_adapter
  tpl_enum = clip_pattern.TemplateName

  def _draw_merged(draw_pattern: Any) -> list[crease_utils.Crease]:
    raw_creases = adapter.shape_rectangles(
        rects,
        draw_pattern,
        bounding_rectangles=target_rects,
        global_refine=global_refine,
    )
    return crease_utils.merge_creases(
        _dedup_lines(raw_creases), only_trivial=False
    )

  if template == tpl_enum.FOLD:
    if (
        math.isclose(bend_angle, 0.0)
        and math.isclose(rotate_angle, 0.0)
        and not enable_narrow
    ):
      raise ValueError(
          'clip_pattern_fold(fold) needs non-zero bend_angle, rotate_angle, or'
          ' enable_narrow=True.'
      )
    allow_3d = not math.isclose(bend_angle, 0.0)
    if river_width is not None:
      if enable_narrow:
        raw_steps = (
            [step, 0.0] if step > 0.0 and river_width - step < 2.0 else [step]
        )
      else:
        raw_steps = [step, 0.5] if step < 0.5 else [step]
      steps = _dedup_floats(raw_steps)
      margins = (0.0, _CLIP_MARGIN) if enable_narrow else (_CLIP_MARGIN, 0.0)
    else:
      extra = [step]
      if step < 0.5:
        extra.append(0.5)
      if enable_narrow and narrow_n_times >= 3 and step < 1.0:
        extra.append(1.0)
      steps = _dedup_floats(extra)
      margins = (_CLIP_MARGIN, 0.0) if enable_narrow else (_CLIP_MARGIN,)

    last_reason = 'no shaping creases generated'
    for cand_step in steps:
      if not enable_narrow:
        syms: Sequence[bool] = (True,)
      elif narrow_symmetric is not None:
        syms = (bool(narrow_symmetric),)
      elif river_width is not None and river_width - cand_step < 3.0 - 1e-4:
        syms = (False, True)
      else:
        syms = (True, False)

      if river_width is not None:
        eff_len: float | None = river_width
      elif flap is not None and not global_refine:
        eff_len = min(float(flap.grid_unit_radius), 4.0)
      else:
        eff_len = None
      for sym in syms:
        for margin in margins:
          draw = clip_pattern.build_fold_pattern(
              margin=margin,
              step=step_from_root + cand_step,
              bend_angle=bend_angle,
              rotate_angle=rotate_angle,
              enable_narrow=enable_narrow,
              narrow_symmetric=sym,
              narrow_n_times=narrow_n_times,
              narrow_length=eff_len,
          )
          try:
            raw_creases = adapter.shape_rectangles(
                rects,
                draw,
                bounding_rectangles=target_rects,
                global_refine=global_refine,
            )
          except (ValueError, AssertionError) as error:
            last_reason = str(error)
            continue
          if not raw_creases:
            continue
          can_cancel = (
              river_width is not None
              and enable_narrow
              and not sym
              and narrow_n_times == 2
              and math.isclose(river_width - cand_step, 2.0, abs_tol=1e-4)
          )
          cancel_opts: Sequence[bool] = (
              (True, False) if can_cancel else (False,)
          )
          for cancel in cancel_opts:
            creases = (
                _cancel_paired_lines(raw_creases) if cancel else raw_creases
            )
            if not creases:
              continue
            shaped = state.pattern.update_with_shaping_creases(creases)
            accepted, reason, _ = state_lib.accept_pattern(
                state,
                shaped,
                allow_3d_vertices=allow_3d,
                verify_fold=verify_fold,
            )
            if accepted is not None:
              delta = len(accepted.pattern.creases) - len(state.pattern.creases)
              return state_lib.ToolOutcome(
                  state=accepted,
                  note=(
                      f'{element_name}: clip fold(step={cand_step:g},'
                      f' narrow={enable_narrow}, sym={sym}), +{delta} creases.'
                  ),
              )
            last_reason = reason
    raise ValueError(
        f'clip_pattern_fold(fold) on {element_name} failed: {last_reason}'
    )

  if template == tpl_enum.TURN_AND_NARROW:
    if flap is None:
      raise ValueError(
          f"{element_name!r} is a river; 'turn_and_narrow' requires a flap."
      )
    base_int = round(step_from_root)
    if global_refine:
      turn_len: float | None = None
    elif length is not None:
      turn_len = float(length)
    else:
      turn_len = float(flap.grid_unit_radius)
    last_reason = 'no foldable turn_and_narrow candidate'
    for cand_step in _dedup_floats([round(step), step, 0.0, 1.0]):
      total_step = (
          float(base_int + round(cand_step))
          if math.isclose(cand_step, round(cand_step), abs_tol=1e-6)
          else step_from_root + cand_step
      )
      draw = clip_pattern.build_template_pattern(
          template_name=template,
          margin=0.0,
          step=total_step,
          bend_angle=0.0,
          length=turn_len,
      )
      try:
        creases = _draw_merged(draw)
      except (ValueError, AssertionError) as error:
        last_reason = str(error)
        continue
      if not creases:
        continue
      shaped = state.pattern.update_with_shaping_creases(creases)
      accepted, reason, _ = state_lib.accept_pattern(
          state, shaped, allow_3d_vertices=True, verify_fold=verify_fold
      )
      if accepted is not None:
        delta = len(accepted.pattern.creases) - len(state.pattern.creases)
        return state_lib.ToolOutcome(
            state=accepted,
            note=(
                f'{element_name}: turn_and_narrow(step={cand_step:g}),'
                f' +{delta} creases.'
            ),
        )
      last_reason = reason
    raise ValueError(f'turn_and_narrow on {element_name} failed: {last_reason}')

  if template == tpl_enum.BEND_WHEN_TURN_AND_NARROW:
    if flap is None:
      raise ValueError(
          f"{element_name!r} is a river; 'bend_when_turn_and_narrow' requires a"
          ' flap.'
      )
    if math.isclose(bend_angle, 0.0):
      raise ValueError(
          'bend_angle must be non-zero for bend_when_turn_and_narrow.'
      )

    def _try_bend_on(
        base_st: state_lib.ShapingState, steps_to_try: Sequence[float]
    ) -> tuple[state_lib.ShapingState | None, float, str]:
      err = 'no foldable bend_when_turn_and_narrow candidate'
      for bw_step in steps_to_try:
        draw = clip_pattern.build_template_pattern(
            template_name=template,
            margin=0.0,
            step=step_from_root + bw_step,
            bend_angle=bend_angle,
        )
        try:
          creases = _draw_merged(draw)
        except (ValueError, AssertionError) as error:
          err = str(error)
          continue
        if not creases:
          continue
        shaped = base_st.pattern.update_with_shaping_creases(creases)
        accepted, reason, _ = state_lib.accept_pattern(
            base_st, shaped, allow_3d_vertices=True, verify_fold=verify_fold
        )
        if accepted is not None:
          return accepted, bw_step, 'ok'
        err = reason
      return None, 0.0, err

    direct_steps = _dedup_floats([
        step,
        step + 0.25,
        max(1.25, float(round(step)) + 0.25),
        max(1.5, float(round(step)) + 0.5),
    ])
    accepted, used_step, last_reason = _try_bend_on(state, direct_steps)
    if accepted is not None:
      delta = len(accepted.pattern.creases) - len(state.pattern.creases)
      return state_lib.ToolOutcome(
          state=accepted,
          note=(
              f'{element_name}: bend_when_turn_and_narrow(angle={bend_angle:g},'
              f' step={used_step:g}), +{delta} creases.'
          ),
      )

    for tn_step in _dedup_floats([max(0.0, float(round(step) - 1)), 0.0]):
      try:
        turned = _clip_pattern_once(
            state,
            element_name=element_name,
            template=tpl_enum.TURN_AND_NARROW,
            step=tn_step,
            bend_angle=0.0,
            rotate_angle=0.0,
            enable_narrow=False,
            narrow_n_times=2,
            narrow_symmetric=None,
            pleat_distance=pleat_distance,
            length=length,
            global_refine=global_refine,
            verify_fold=verify_fold,
        )
      except ValueError as error:
        last_reason = str(error)
        continue
      bw_steps = _dedup_floats(
          [max(tn_step + 1.25, step), tn_step + 1.25, tn_step + 1.5]
      )
      chained, used_step, last_reason = _try_bend_on(turned.state, bw_steps)
      if chained is not None:
        delta = len(chained.pattern.creases) - len(state.pattern.creases)
        return state_lib.ToolOutcome(
            state=chained,
            note=(
                f'{element_name}: turn_and_narrow(step={tn_step:g}) +'
                f' bend(angle={bend_angle:g}, step={used_step:g}),'
                f' +{delta} creases.'
            ),
        )
    raise ValueError(
        f'bend_when_turn_and_narrow on {element_name} failed: {last_reason}'
    )

  # Template 4: PLEAT_TO_HIDE.
  if pleat_distance <= 0.0:
    raise ValueError(f'pleat_distance must be > 0, got {pleat_distance}.')
  if length is not None and length <= 0.0:
    raise ValueError(f'length must be > 0, got {length}.')

  last_reason = 'no foldable pleat_to_hide candidate'
  for cand_step in _dedup_floats([step, 0.5] if step < 0.5 else [step]):
    if length is not None:
      span = float(length)
    elif river_width is not None:
      rem = river_width - cand_step
      span = rem if global_refine else min(2.0, max(0.5, rem))
    else:
      rem = float(flap.grid_unit_radius) - cand_step if flap else 1.0
      span = rem if global_refine else min(2.0, max(1.0, rem))
    margin = _CLIP_MARGIN if math.isclose(cand_step, 0.0) else 0.0
    total_step = step_from_root + cand_step
    for cand_dist in _dedup_floats([pleat_distance, 0.5, 0.25]):
      if span <= cand_dist * 1.05:
        continue
      draw = clip_pattern.build_template_pattern(
          template_name=template,
          margin=margin,
          step=total_step,
          bend_angle=0.0,
          length=total_step + margin + span,
          pleat_distance=cand_dist,
      )
      try:
        creases = _draw_merged(draw)
      except (ValueError, AssertionError) as error:
        last_reason = str(error)
        continue
      if not creases:
        continue
      shaped = state.pattern.update_with_shaping_creases(creases)
      accepted, reason, _ = state_lib.accept_pattern(
          state, shaped, allow_3d_vertices=False, verify_fold=verify_fold
      )
      if accepted is not None:
        delta = len(accepted.pattern.creases) - len(state.pattern.creases)
        return state_lib.ToolOutcome(
            state=accepted,
            note=(
                f'{element_name}: pleat_to_hide(step={cand_step:g},'
                f' span={span:.2f}, dist={cand_dist:g}), +{delta} creases.'
            ),
        )
      last_reason = reason
  raise ValueError(f'pleat_to_hide on {element_name} failed: {last_reason}')


def clip_pattern_fold(
    state: state_lib.ShapingState,
    *,
    element_name: str,
    template_name: str = 'fold',
    step: float = 0.0,
    bend_angle: float = 0.0,
    rotate_angle: float = 0.0,
    enable_narrow: bool = False,
    narrow_n_times: int = 2,
    narrow_symmetric: bool | None = None,
    pleat_distance: float = 0.5,
    length: float | None = None,
    global_refine: bool = False,
    verify_fold: bool = True,
) -> state_lib.ToolOutcome:
  """Draws a layer-aware clip-pattern template across a flap or river."""
  if int(narrow_n_times) < 2:
    raise ValueError(f'narrow_n_times must be >= 2, got {narrow_n_times}.')
  if step < 0.0:
    raise ValueError(f'step must be >= 0, got {step}.')
  template = clip_pattern.TemplateName(template_name)

  def _impl(st: state_lib.ShapingState) -> state_lib.ToolOutcome:
    refine_modes = (True,) if global_refine else (False, True)
    last_error: ValueError | None = None
    for ref in refine_modes:
      try:
        out = _clip_pattern_once(
            st,
            element_name=element_name,
            template=template,
            step=float(step),
            bend_angle=float(bend_angle),
            rotate_angle=float(rotate_angle),
            enable_narrow=enable_narrow,
            narrow_n_times=int(narrow_n_times),
            narrow_symmetric=narrow_symmetric,
            pleat_distance=float(pleat_distance),
            length=length,
            global_refine=ref,
            verify_fold=verify_fold,
        )
        if len(out.state.pattern.creases) > len(st.pattern.creases):
          return out
      except ValueError as error:
        last_error = error
    if last_error is not None:
      raise last_error
    raise ValueError(
        f'clip_pattern_fold({element_name}) generated no new creases.'
    )

  return state_lib.run_with_3d_overlay(state, _impl, verify_fold=verify_fold)


# ------------------------------------------------------------------------------
# 2. Symmetric tapering of a central flap (`narrow_central_flap`).
# ------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class CentralFlapFrame:
  """Folded-form local coordinate frame `(origin, u, v)` of a central flap."""

  origin: np.ndarray
  u: np.ndarray
  v: np.ndarray
  s_min: float
  s_max: float
  t_hinge: float
  t_tip: float

  @property
  def width(self) -> float:
    """Folded lateral span of the flap."""
    return self.s_max - self.s_min

  def to_folded(self, s: float, t: float) -> tuple[float, float]:
    """Maps local `(s, t)` coordinates back into 2-D folded-form space."""
    pt = self.origin + s * self.u + t * self.v
    return float(pt[0]), float(pt[1])


def _points_in_polygon(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
  """Even-odd point-in-polygon check including boundary segments."""
  inside = np.zeros(len(points), dtype=bool)
  on_edge = np.zeros(len(points), dtype=bool)
  px, py = points[:, 0], points[:, 1]
  n = len(polygon)
  for i in range(n):
    x1, y1 = polygon[i]
    x2, y2 = polygon[(i + 1) % n]
    cross = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
    within = (
        (np.minimum(x1, x2) - 1e-9 <= px)
        & (px <= np.maximum(x1, x2) + 1e-9)
        & (np.minimum(y1, y2) - 1e-9 <= py)
        & (py <= np.maximum(y1, y2) + 1e-9)
    )
    on_edge |= within & (np.abs(cross) < 1e-9)
    straddles = (y1 > py) != (y2 > py)
    with np.errstate(divide='ignore', invalid='ignore'):
      x_cross = x1 + (py - y1) * (x2 - x1) / (y2 - y1)
    inside ^= straddles & (px < x_cross)
  return inside | on_edge


def _map_points_to_folded(
    graph_data: geometric_simple_fold.GraphData, points: np.ndarray
) -> np.ndarray:
  """Maps unit-paper points to 2-D folded positions via per-face isometries."""
  vertices = np.asarray(graph_data['vertices_coords'], dtype=float)
  folded = graph_data['folded_vertices']
  out = np.full((len(points), 2), np.nan)
  todo = np.ones(len(points), dtype=bool)
  for face in graph_data['face_vertices']:
    if len(face) < 3 or any(folded[i] is None for i in face):
      continue
    src = vertices[face]
    lo, hi = src.min(axis=0) - 1e-9, src.max(axis=0) + 1e-9
    idx = np.flatnonzero(
        todo
        & (points[:, 0] >= lo[0])
        & (points[:, 0] <= hi[0])
        & (points[:, 1] >= lo[1])
        & (points[:, 1] <= hi[1])
    )
    if not idx.size:
      continue
    mask = _points_in_polygon(points[idx], src)
    if not mask.any():
      continue
    dst = np.asarray([folded[i][:2] for i in face], dtype=float)
    affine, *_ = np.linalg.lstsq(
        np.c_[src, np.ones(len(face))], dst, rcond=None
    )
    hit = idx[mask]
    out[hit] = np.c_[points[hit], np.ones(len(hit))] @ affine
    todo[hit] = False
  return out


def central_flap_frame(
    flap: shaping_elements.ShapingFlap,
    graph_data: geometric_simple_fold.GraphData,
) -> CentralFlapFrame:
  """Measures a flap's axial direction and lateral width in the folded form."""
  rect = flap.to_legacy_rectangle()
  gs = flap.grid_size
  x0 = max(0.0, min(rect.x_start, rect.x_start + rect.x_len))
  x1 = min(1.0, max(rect.x_start, rect.x_start + rect.x_len))
  y0 = max(0.0, min(rect.y_start, rect.y_start + rect.y_len))
  y1 = min(1.0, max(rect.y_start, rect.y_start + rect.y_len))
  step = _CENTRAL_SAMPLE_STEP / max(gs, 1)
  gx, gy = np.meshgrid(
      np.arange(x0, x1 + step * 0.5, step),
      np.arange(y0, y1 + step * 0.5, step),
  )
  points = np.stack([gx.ravel(), gy.ravel()], axis=1)
  centers = np.asarray([[c.x, c.y] for c in flap.circle_centers], dtype=float)
  clamped = np.clip(points, centers.min(axis=0), centers.max(axis=0))
  dist = np.abs(points - clamped).max(axis=1) * gs
  radius = float(flap.grid_unit_radius)
  keep = dist <= radius + 1e-6
  points, dist = points[keep], dist[keep]
  folded = _map_points_to_folded(graph_data, points)
  valid = ~np.isnan(folded).any(axis=1)
  folded, dist = folded[valid], dist[valid]
  if len(folded) < 8:
    raise ValueError(f'{flap.name}: could not locate flap in 2-D folded form.')
  band = min(0.5, radius * 0.25) + 1e-6
  tip_mask = dist <= band
  hinge_mask = dist >= radius - band
  if not tip_mask.any() or not hinge_mask.any():
    raise ValueError(f'{flap.name}: could not isolate tip and hinge bands.')
  origin = folded[hinge_mask].mean(axis=0)
  axis = folded[tip_mask].mean(axis=0) - origin
  norm = float(np.linalg.norm(axis))
  if norm < 1e-9:
    raise ValueError(f'{flap.name}: degenerate folded flap axis.')
  v = axis / norm
  u = np.array([-v[1], v[0]], dtype=float)
  rel = folded - origin
  s, t = rel @ u, rel @ v
  return CentralFlapFrame(
      origin=origin,
      u=u,
      v=v,
      s_min=float(s.min()),
      s_max=float(s.max()),
      t_hinge=float(t.min()),
      t_tip=float(t.max()),
  )


def central_flap_taper_lines(
    frame: CentralFlapFrame,
    *,
    tip_width: float,
    start_offset_units: float,
    grid_size: int,
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
  """Returns the two symmetric taper cut lines in folded-form coordinates."""
  half = frame.width * 0.5
  center_s = 0.5 * (frame.s_min + frame.s_max)
  t_start = frame.t_hinge - start_offset_units / max(grid_size, 1)
  return [
      (
          frame.to_folded(center_s + side * half, t_start),
          frame.to_folded(center_s + side * half * tip_width, frame.t_tip),
      )
      for side in (-1.0, 1.0)
  ]


def _signed_mirror_match(
    creases: Sequence[crease_utils.Crease], axis_x: float = 0.5
) -> float:
  """Fraction of `creases` whose mirror segment has matching M/V polarity."""
  if not creases:
    return 0.0

  def _sig(
      x1: float, y1: float, x2: float, y2: float, fold: crease_utils.Fold
  ) -> tuple[tuple[float, float], tuple[float, float], crease_utils.Fold]:
    a = (round(x1, 5), round(y1, 5))
    b = (round(x2, 5), round(y2, 5))
    return (a, b, fold) if a <= b else (b, a, fold)

  sigs = {_sig(c.p1.x, c.p1.y, c.p2.x, c.p2.y, c.fold) for c in creases}
  hits = sum(
      _sig(2.0 * axis_x - c.p1.x, c.p1.y, 2.0 * axis_x - c.p2.x, c.p2.y, c.fold)
      in sigs
      for c in creases
  )
  return hits / len(creases)


def narrow_central_flap(
    state: state_lib.ShapingState,
    *,
    element_name: str,
    tip_width: float = 0.25,
    start_offset: float = 0.0,
    fold_type: str = 'auto',
    require_symmetric: bool = True,
    verify_fold: bool = True,
) -> state_lib.ToolOutcome:
  """Tapers both halves of a central flap symmetrically in the folded form."""
  if not 0.0 <= tip_width < 1.0:
    raise ValueError(f'tip_width must be in [0, 1), got {tip_width}.')
  if start_offset < 0.0:
    raise ValueError(f'start_offset must be >= 0, got {start_offset}.')
  if fold_type not in ('auto', 'valley', 'mountain'):
    raise ValueError(
        f"fold_type must be 'auto', 'valley', or 'mountain'; got {fold_type!r}."
    )

  def _impl(st: state_lib.ShapingState) -> state_lib.ToolOutcome:
    elem = st.element(element_name)
    if not isinstance(elem, shaping_elements.ShapingFlap):
      raise ValueError(
          f'{element_name!r} is a river; narrow_central_flap requires a flap.'
      )
    if elem.grid_unit_radius <= 1.0 + 1e-6:
      raise ValueError(f'{element_name!r} has length 1; nothing to taper.')
    if overlay_3d.has_3d_overlay(st.pattern):
      raise ValueError(
          'narrow_central_flap requires a 2-D flat backbone; apply before 3-D'
          ' relief.'
      )
    graph_data = st.graph_data
    frame = central_flap_frame(elem, graph_data)
    gs = elem.grid_size
    lines = central_flap_taper_lines(
        frame,
        tip_width=tip_width,
        start_offset_units=start_offset,
        grid_size=gs,
    )
    limiting = list(st.element_polygon(element_name))
    if start_offset > 0.0:
      parent = paper_layout_lib.topology(st.tree).parent.get(element_name)
      if parent is None:
        raise ValueError(
            f'{element_name!r} has no parent for start_offset > 0.'
        )
      limiting.extend(st.element_polygon(parent))

    fold_map = {
        'valley': crease_utils.Fold.VALLEY,
        'mountain': crease_utils.Fold.MOUNTAIN,
    }
    order = ('valley', 'mountain') if fold_type == 'auto' else (fold_type,)
    reasons: list[str] = []
    for name in order:
      creases: list[crease_utils.Crease] = []
      for start, end in lines:
        creases.extend(
            geometric_simple_fold.get_geometric_fold_creases(
                crease_pattern=st.pattern,
                cut_line_start=start,
                cut_line_end=end,
                fold_assignment=fold_map[name],
                limiting_polygon=limiting,
                graph_data=graph_data,
            )
        )
      if not creases:
        reasons.append(f'{name}: cut lines missed flap')
        continue
      mirror = (
          _signed_mirror_match(creases) if elem.contains_central_crease else 1.0
      )
      if (
          require_symmetric
          and elem.contains_central_crease
          and mirror < _CENTRAL_MIN_MIRROR_MATCH
      ):
        reasons.append(f'{name}: asymmetric creases (match={mirror:.2f})')
        continue
      try:
        shaped = st.pattern.update_with_shaping_creases(creases)
      except (ValueError, AssertionError, IndexError) as error:
        reasons.append(f'{name}: {error}')
        continue
      accepted, reason, _ = state_lib.accept_pattern(
          st, shaped, allow_3d_vertices=False, verify_fold=verify_fold
      )
      if accepted is None:
        reasons.append(f'{name}: {reason}')
        continue
      delta = len(accepted.pattern.creases) - len(st.pattern.creases)
      return state_lib.ToolOutcome(
          state=accepted,
          note=(
              f'{element_name}: tapered ({name}) width {frame.width * gs:.2f}'
              f'->{frame.width * gs * tip_width:.2f}g, +{delta} creases.'
          ),
      )
    raise ValueError(
        f'narrow_central_flap({element_name}) failed: {"; ".join(reasons)}'
    )

  return state_lib.run_with_3d_overlay(state, _impl, verify_fold=verify_fold)


# ------------------------------------------------------------------------------
# 3. Concentric stepped relief (`inscribed_flap`).
# ------------------------------------------------------------------------------


def inscribed_flap(
    state: state_lib.ShapingState,
    *,
    element_name: str,
    depths: int | Sequence[int] = (1,),
    fold_order: str = 'MV',
    fold_percent: float = 0.5,
    verify_fold: bool = True,
) -> state_lib.ToolOutcome:
  """Adds concentric inscribed pleat steps inside a flap or river."""
  if not 0.0 < fold_percent <= 1.0:
    raise ValueError(f'fold_percent must be in (0, 1], got {fold_percent}.')
  order_key = fold_order.upper()
  if order_key not in _INSCRIBED_ORDERS:
    raise ValueError(
        f'fold_order must be one of {sorted(_INSCRIBED_ORDERS)}, got'
        f' {fold_order!r}.'
    )
  depth_list = (
      [int(depths)]
      if isinstance(depths, int)
      else sorted({int(d) for d in depths})
  )
  if not depth_list:
    raise ValueError('depths must be non-empty.')

  def _impl(st: state_lib.ShapingState) -> state_lib.ToolOutcome:
    elem = st.element(element_name)
    is_flap = isinstance(elem, shaping_elements.ShapingFlap)
    max_depth = (
        int(elem.grid_unit_radius) - 1
        if is_flap
        else int(round(elem.width)) - 1
    )
    if max_depth < 1:
      raise ValueError(
          f'{element_name!r} needs radius/width >= 2 for inscribed_flap.'
      )
    for d in depth_list:
      if not 1 <= d <= max_depth:
        raise ValueError(
            f'depth={d} out of range for {element_name!r} (valid'
            f' 1..{max_depth}).'
        )

    flaps, rivers = st.shaping_elements
    banks: Sequence[shaping_types.BankIndex | None]
    if is_flap:
      banks = (None,)
    else:
      primary = elem.longer_bank_index
      other: shaping_types.BankIndex = (
          'SECOND' if primary == 'FIRST' else 'FIRST'
      )
      banks = (primary, other)

    allow_3d = not math.isclose(fold_percent, 1.0, abs_tol=1e-6)
    last_reason = 'no shaping creases generated'
    for key in (order_key, _OPPOSITE_ORDER[order_key]):
      folds_pair = _INSCRIBED_ORDERS[key]
      for bank in banks:
        if is_flap:
          params: Sequence[shaping_types.InscribedParams] = [
              shaping_types.InscribedParams(
                  depth=d,
                  fold_order=folds_pair,
                  fold_percent=float(fold_percent),
              )
              for d in depth_list
          ]
        else:
          assert bank is not None
          params = [
              shaping_types.InscribedParams(
                  depth=d,
                  fold_order=folds_pair,
                  fold_percent=float(fold_percent),
                  bank_index=bank,
              )
              for d in depth_list
          ]
        bundle = shaping_types.ElementShapingParams(
            element_name=element_name, params=params
        )
        try:
          shaped = zigzag_shaping.shape_cp_with_inscribed_elements(
              cp=st.pattern,
              grid_size=elem.grid_size,
              rectangles=flaps,
              rivers=rivers,
              element_shaping_params=[bundle],
              central_crease_removed=st.central_crease_removed,
              light_mode=not st.config.check_all_crease_intersections,
              raise_error_on_ff_violation=False,
          )
        except (ValueError, AssertionError, IndexError) as error:
          last_reason = str(error)
          continue
        delta = len(shaped.creases) - len(st.pattern.creases)
        if delta <= 0:
          continue
        accepted, reason, _ = state_lib.accept_pattern(
            st, shaped, allow_3d_vertices=allow_3d, verify_fold=verify_fold
        )
        if accepted is not None:
          return state_lib.ToolOutcome(
              state=accepted,
              note=(
                  f'{element_name}: inscribed depths={depth_list} order={key}'
                  f' fold_percent={fold_percent:g}, +{delta} creases.'
              ),
          )
        last_reason = reason
    raise ValueError(f'inscribed_flap({element_name}) failed: {last_reason}')

  return state_lib.run_with_3d_overlay(state, _impl, verify_fold=verify_fold)


# ------------------------------------------------------------------------------
# 4. Partial 3-D opening of flaps and rivers (`open_flap`).
# ------------------------------------------------------------------------------


def _plate_breadth(
    model: folder_3d.FoldedModel,
    flap: shaping_elements.ShapingFlap,
    grid_size: int,
) -> float | None:
  """Measures secondary singular value (grid units) of a flap's 3-D plate."""
  rect = flap.to_legacy_rectangle()
  x0 = max(0.0, min(rect.x_start, rect.x_start + rect.x_len))
  x1 = min(1.0, max(rect.x_start, rect.x_start + rect.x_len))
  y0 = max(0.0, min(rect.y_start, rect.y_start + rect.y_len))
  y1 = min(1.0, max(rect.y_start, rect.y_start + rect.y_len))
  if x1 - x0 <= 2e-4 or y1 - y0 <= 2e-4:
    return None
  gx, gy = np.meshgrid(
      np.linspace(x0 + 1e-4, x1 - 1e-4, 15),
      np.linspace(y0 + 1e-4, y1 - 1e-4, 15),
  )
  pts = np.stack([gx.ravel(), gy.ravel()], axis=1)
  located = model.locate(pts)
  located = located[~np.isnan(located).any(axis=1)]
  if len(located) < 3:
    return None
  sv = np.linalg.svd(
      located - located.mean(axis=0), compute_uv=False
  ) / math.sqrt(len(located))
  return float(sv[1] * grid_size)


def _flap_open_selectors(
    flap: shaping_elements.ShapingFlap, *, ridge_only: bool
) -> list[tuple[str, Callable[[crease_utils.Crease], bool]]]:
  """Returns candidate crease selectors for opening a flap into a 3-D plate."""
  rect = flap.to_legacy_rectangle()
  rx0 = min(rect.x_start, rect.x_start + rect.x_len)
  rx1 = max(rect.x_start, rect.x_start + rect.x_len)
  ry0 = min(rect.y_start, rect.y_start + rect.y_len)
  ry1 = max(rect.y_start, rect.y_start + rect.y_len)
  rbox = (max(0.0, rx0), min(1.0, rx1), max(0.0, ry0), min(1.0, ry1))

  cboxes: list[tuple[float, float, float, float]] = []
  if flap.circle_centers:
    c0 = flap.circle_centers[0]
    rad = min(
        abs(c0.x - rx0), abs(rx1 - c0.x), abs(c0.y - ry0), abs(ry1 - c0.y)
    )
    for cc in flap.circle_centers:
      bx0, bx1 = max(0.0, cc.x - rad), min(1.0, cc.x + rad)
      by0, by1 = max(0.0, cc.y - rad), min(1.0, cc.y + rad)
      if bx1 - bx0 > 1e-4 and by1 - by0 > 1e-4:
        cboxes.append((bx0, bx1, by0, by1))

  ridges = flap.get_ridges()

  def _in_box(pt: Any, box: tuple[float, float, float, float]) -> bool:
    x0, x1, y0, y1 = box
    return x0 - 1e-4 <= pt.x <= x1 + 1e-4 and y0 - 1e-4 <= pt.y <= y1 + 1e-4

  def _strict(pt: Any, box: tuple[float, float, float, float]) -> bool:
    if not _in_box(pt, box):
      return False
    x0, x1, y0, y1 = box
    for xv in (x0, x1):
      if 1e-4 < xv < 1.0 - 1e-4 and abs(pt.x - xv) < 1e-4:
        return False
    for yv in (y0, y1):
      if 1e-4 < yv < 1.0 - 1e-4 and abs(pt.y - yv) < 1e-4:
        return False
    return True

  def _strict_cb(pt: Any) -> bool:
    return any(_strict(pt, b) for b in cboxes)

  def _is_ridge(c: crease_utils.Crease) -> bool:
    return any(c.is_collinear(r) for r in ridges)

  ridge_preds: list[tuple[str, Callable[[crease_utils.Crease], bool]]] = [
      (
          'ridges',
          lambda c: _in_box(c.p1_init, rbox)
          and _in_box(c.p2_init, rbox)
          and _is_ridge(c),
      ),
      (
          'cbox_ridges',
          lambda c: (_strict_cb(c.p1_init) or _strict_cb(c.p2_init))
          and _is_ridge(c),
      ),
  ]
  if ridge_only:
    return ridge_preds
  return [
      (
          'cbox_vert',
          lambda c: (_strict_cb(c.p1_init) or _strict_cb(c.p2_init))
          and c.is_vertical,
      ),
      (
          'cbox_horiz',
          lambda c: (_strict_cb(c.p1_init) or _strict_cb(c.p2_init))
          and c.is_horizontal,
      ),
      (
          'rect_vert',
          lambda c: (_strict(c.p1_init, rbox) or _strict(c.p2_init, rbox))
          and c.is_vertical,
      ),
      (
          'rect_horiz',
          lambda c: (_strict(c.p1_init, rbox) or _strict(c.p2_init, rbox))
          and c.is_horizontal,
      ),
      (
          'rect_axis',
          lambda c: (_strict(c.p1_init, rbox) or _strict(c.p2_init, rbox))
          and (c.is_horizontal or c.is_vertical),
      ),
      (
          'rect_all',
          lambda c: _in_box(c.p1_init, rbox) and _in_box(c.p2_init, rbox),
      ),
      *ridge_preds,
  ]


def _open_river(
    state: state_lib.ShapingState,
    river: shaping_elements.ShapingRiver,
    *,
    fold_percent: float,
    verify_fold: bool,
) -> state_lib.ToolOutcome:
  """Partially opens interior axis-aligned accordion pleats of a river."""
  rects = list(river.covering_legacy_rectangles)
  if not rects:
    raise ValueError(f'open_flap({river.name}): river has no rectangles.')
  margin = 0.25 / max(float(state.solving_grid_size), 1.0)
  boxes = [
      (
          max(0.0, min(r.x_start, r.x_start + r.x_len)),
          max(0.0, min(r.y_start, r.y_start + r.y_len)),
          min(1.0, max(r.x_start, r.x_start + r.x_len)),
          min(1.0, max(r.y_start, r.y_start + r.y_len)),
      )
      for r in rects
  ]
  keys = {
      state_lib.crease_key(c)
      for c in state.pattern.creases
      if c.mountain_or_valley()
      and (c.is_horizontal or c.is_vertical)
      and any(
          x0 + margin <= 0.5 * (c.p1_init.x + c.p2_init.x) <= x1 - margin
          and y0 + margin <= 0.5 * (c.p1_init.y + c.p2_init.y) <= y1 - margin
          for x0, y0, x1, y1 in boxes
      )
  }
  if not keys:
    raise ValueError(
        f'open_flap({river.name}): no interior river accordion creases found.'
    )

  prior_bb = state.flat_backbone or (
      overlay_3d.as_flat_backbone(state.pattern)
      if state.pattern.local_flat_foldability()
      else None
  )
  ladder = [fold_percent] + [
      p for p in (0.65, 0.75, 0.85, 0.92) if p > fold_percent
  ]
  last_reason = 'no clean 3-D river opening'
  for pct in _dedup_floats(ladder):
    cand = state_lib.with_fold_percent(state.pattern, keys, pct)
    accepted, reason, model = state_lib.accept_pattern(
        state,
        cand,
        allow_3d_vertices=True,
        verify_fold=verify_fold,
        max_vertex_error=fidelity.DEFAULT_CONFIG.clean_gate_max_vertex_error,
    )
    if accepted is None:
      last_reason = reason
      continue
    out = accepted.replace(flat_backbone=prior_bb)
    strain = f', strain {model.vertex_error:.3f}%' if model is not None else ''
    relaxed = not math.isclose(pct, fold_percent, abs_tol=1e-6)
    fb_note = (
        f' [fallback from fold_percent={fold_percent:g}: {last_reason}]'
        if relaxed
        else ''
    )
    resolved: dict[str, object] | None = None
    if relaxed:
      resolved = {
          'element_name': river.name,
          'fold_percent': pct,
          'verify_fold': verify_fold,
      }
    return state_lib.ToolOutcome(
        state=out,
        note=(
            f'{river.name}: opened {len(keys)} river creases at'
            f' fold_percent={pct:g}{strain}{fb_note}.'
        ),
        resolved_arguments=resolved,
    )
  raise ValueError(f'open_flap({river.name}) failed: {last_reason}')


def open_flap(
    state: state_lib.ShapingState,
    *,
    element_name: str,
    fold_percent: float = 0.35,
    ridge_only: bool = False,
    verify_fold: bool = True,
) -> state_lib.ToolOutcome:
  """Spreads a folded flap plate or river channel into 3-D via partial folds.

  On flaps, tries localized interior pleat and diagonal ridge families and keeps
  the clean candidate maximizing 3-D plate breadth (shells, hoods, wings, ears).
  On rivers, relaxes interior axis-aligned accordion pleats for 3-D body volume.

  Args:
    state: Current shaping state.
    element_name: Target flap or river name.
    fold_percent: Target fold fraction in `(0, 1)` (smaller opens wider).
    ridge_only: Restrict flap opening to diagonal 45-degree ridges.
    verify_fold: Require clean 3-D folding within `CLEAN_GATE_MAX_VERTEX_ERROR`.

  Returns:
    `ToolOutcome` with the opened pattern and preserved `flat_backbone`.
  """
  if not 0.0 < fold_percent < 1.0:
    raise ValueError(f'fold_percent must be in (0, 1), got {fold_percent}.')
  elem = state.element(element_name)
  if isinstance(elem, shaping_elements.ShapingRiver):
    return _open_river(
        state, elem, fold_percent=float(fold_percent), verify_fold=verify_fold
    )

  prior_bb = state.flat_backbone or (
      overlay_3d.as_flat_backbone(state.pattern)
      if state.pattern.local_flat_foldability()
      else None
  )
  base_breadth: float | None = None
  if verify_fold:
    ok_b, _, m_before = state_lib.verify_3d_fold(state.pattern)
    if ok_b and m_before is not None:
      base_breadth = _plate_breadth(m_before, elem, state.solving_grid_size)

  selectors = _flap_open_selectors(elem, ridge_only=ridge_only)
  pcts = _dedup_floats(
      [fold_percent, 0.5, 0.65, 0.8] if fold_percent < 0.8 else [fold_percent]
  )
  seen_keysets: set[frozenset[tuple[float, float, float, float]]] = set()
  best: tuple[float, state_lib.ShapingState, str, int, float] | None = None
  last_reason = 'no interior flap creases matched'

  for label, pred in selectors:
    keys = frozenset(
        state_lib.crease_key(c)
        for c in state.pattern.creases
        if c.mountain_or_valley() and pred(c)
    )
    if not keys or keys in seen_keysets:
      continue
    seen_keysets.add(keys)
    for pct in pcts:
      cand = state_lib.with_fold_percent(state.pattern, set(keys), pct)
      if state_lib.crease_multiset(cand) == state_lib.crease_multiset(
          state.pattern
      ):
        continue
      accepted, reason, model = state_lib.accept_pattern(
          state,
          cand,
          allow_3d_vertices=True,
          verify_fold=verify_fold,
          max_vertex_error=fidelity.DEFAULT_CONFIG.clean_gate_max_vertex_error,
      )
      if accepted is None:
        last_reason = f'{label}@{pct:g}: {reason}'
        continue
      out = accepted.replace(flat_backbone=prior_bb)
      if not verify_fold or model is None:
        return state_lib.ToolOutcome(
            state=out,
            note=(
                f'{element_name}: opened {len(keys)} creases ({label}) at'
                f' fold_percent={pct:g}.'
            ),
        )
      breadth = _plate_breadth(model, elem, state.solving_grid_size) or 0.0
      gain = (
          breadth / base_breadth
          if base_breadth and base_breadth > 1e-4
          else breadth
      )
      if best is None or gain > best[0]:
        best = (gain, out, label, len(keys), pct)
      if base_breadth and gain >= 1.25:
        break
    if best is not None and base_breadth and best[0] >= 1.35:
      break

  if best is None:
    raise ValueError(f'open_flap({element_name}) failed: {last_reason}')
  gain, out, label, count, pct = best
  relaxed = not math.isclose(pct, fold_percent, abs_tol=1e-6)
  fb_note = f' [fallback from fold_percent={fold_percent:g}]' if relaxed else ''
  resolved: dict[str, object] | None = None
  if relaxed:
    resolved = {
        'element_name': element_name,
        'fold_percent': pct,
        'ridge_only': ridge_only,
        'verify_fold': verify_fold,
    }
  return state_lib.ToolOutcome(
      state=out,
      note=(
          f'{element_name}: opened {count} creases ({label}) at'
          f' fold_percent={pct:g}, breadth {gain:.2f}x{fb_note}.'
      ),
      resolved_arguments=resolved,
  )


# ------------------------------------------------------------------------------
# 5. Symmetry-axis sagittal spine fold (`fold_spine`).
# ------------------------------------------------------------------------------


def _central_spine_bounds(
    state: state_lib.ShapingState, axis_x: float = 0.5, tol: float = 0.025
) -> tuple[float, float, str]:
  """Finds y-span of midline rivers or root flap straddling `x = axis_x`."""
  flaps, rivers = state.shaping_elements
  topo = paper_layout_lib.topology(state.tree)

  def _is_midline(name: str) -> bool:
    partner = state_lib.resolve_symmetric_element(state, name)
    return partner is None or partner == name

  spans: list[tuple[int, float, float, str]] = []
  for river in rivers:
    if not _is_midline(river.name):
      continue
    segs = list(river.bank_1) + list(river.bank_2)
    if not segs:
      continue
    xs = [c.p1.x for c in segs] + [c.p2.x for c in segs]
    ys = [c.p1.y for c in segs] + [c.p2.y for c in segs]
    if min(xs) - tol <= axis_x <= max(xs) + tol:
      depth = 0
      curr: str | None = river.name
      while curr is not None and depth < 16:
        curr = topo.parent.get(curr)
        depth += 1
      spans.append((depth, min(ys), max(ys), river.name))

  if spans:
    min_depth = min(s[0] for s in spans)
    proximal = [s for s in spans if s[0] <= min_depth + 1]
    return (
        min(s[1] for s in proximal),
        max(s[2] for s in proximal),
        '+'.join(s[3] for s in proximal),
    )

  for flap in flaps:
    if flap.name == state.tree.root and flap.contains_central_crease:
      rect = flap.to_legacy_rectangle()
      y0 = max(0.0, min(rect.y_start, rect.y_start + rect.y_len))
      y1 = min(1.0, max(rect.y_start, rect.y_start + rect.y_len))
      return y0, y1, flap.name

  raise ValueError(
      f'fold_spine: no midline element straddles x={axis_x:g}; pass'
      ' element_name explicitly.'
  )


def _next_spine_fold(
    other_diff: int, prev_fold: crease_utils.Fold
) -> crease_utils.Fold:
  """Picks the next spine segment polarity preserving Maekawa `|M - V| = 2`."""
  diff = other_diff + (1 if prev_fold == crease_utils.Fold.MOUNTAIN else -1)
  if abs(diff + 1) == 2 and abs(diff - 1) != 2:
    return crease_utils.Fold.MOUNTAIN
  if abs(diff - 1) == 2 and abs(diff + 1) != 2:
    return crease_utils.Fold.VALLEY
  return prev_fold.flip()


def _build_spine_candidate(
    pattern: crease_pattern_lib.CreasePattern,
    *,
    axis_x: float,
    y0: float,
    y1: float,
    first_fold: crease_utils.Fold,
    pct: float,
    keep_existing: bool,
) -> tuple[crease_pattern_lib.CreasePattern, int] | None:
  """Builds a Maekawa-consistent partial/full spine along `x = axis_x`."""

  def _in_span(c: crease_utils.Crease) -> bool:
    return (
        c.fold != crease_utils.Fold.BORDER
        and abs(c.p1_init.x - axis_x) < 1e-4
        and abs(c.p2_init.x - axis_x) < 1e-4
        and max(c.p1_init.y, c.p2_init.y) > y0 + 1e-4
        and min(c.p1_init.y, c.p2_init.y) < y1 - 1e-4
    )

  covered = sorted(
      (
          max(y0, min(c.p1_init.y, c.p2_init.y)),
          min(y1, max(c.p1_init.y, c.p2_init.y)),
      )
      for c in pattern.creases
      if _in_span(c)
  )
  gaps: list[crease_utils.Crease] = []
  cursor = y0
  for seg_lo, seg_hi in covered:
    if seg_lo > cursor + 1e-4:
      gaps.append(
          crease_utils.Crease(
              point_utils.Point(x=axis_x, y=cursor),
              point_utils.Point(x=axis_x, y=seg_lo),
              fold=first_fold,
          )
      )
    cursor = max(cursor, seg_hi)
  if y1 > cursor + 1e-4:
    gaps.append(
        crease_utils.Crease(
            point_utils.Point(x=axis_x, y=cursor),
            point_utils.Point(x=axis_x, y=y1),
            fold=first_fold,
        )
    )

  base = pattern.update_with_shaping_creases(gaps) if gaps else pattern
  spine_indices = [i for i, c in enumerate(base.creases) if _in_span(c)]
  if not spine_indices:
    return None
  spine_set = set(spine_indices)
  spine_indices.sort(
      key=lambda i: (
          min(base.creases[i].p1_init.y, base.creases[i].p2_init.y),
          max(base.creases[i].p1_init.y, base.creases[i].p2_init.y),
      )
  )

  def _other_mv_diff(y_val: float) -> int:
    diff = 0
    for idx, c in enumerate(base.creases):
      if idx in spine_set or not c.mountain_or_valley():
        continue
      hit_p1 = (
          abs(c.p1_init.x - axis_x) < 2e-4 and abs(c.p1_init.y - y_val) < 2e-4
      )
      hit_p2 = (
          abs(c.p2_init.x - axis_x) < 2e-4 and abs(c.p2_init.y - y_val) < 2e-4
      )
      if hit_p1 or hit_p2:
        diff += 1 if c.fold == crease_utils.Fold.MOUNTAIN else -1
    return diff

  assigned: dict[int, crease_utils.Fold] = {}
  prev_fold: crease_utils.Fold | None = None
  prev_hi = y0
  for idx in spine_indices:
    seg = base.creases[idx]
    lo = min(seg.p1_init.y, seg.p2_init.y)
    hi = max(seg.p1_init.y, seg.p2_init.y)
    if keep_existing and not gaps and seg.mountain_or_valley():
      cur_fold = seg.fold
    elif prev_fold is None or abs(lo - prev_hi) > 5e-4:
      cur_fold = first_fold
    else:
      cur_fold = _next_spine_fold(_other_mv_diff(lo), prev_fold)
    assigned[idx] = cur_fold
    prev_fold = cur_fold
    prev_hi = hi

  updated: list[crease_utils.Crease] = []
  for idx, c in enumerate(base.creases):
    if idx in assigned:
      updated.append(
          crease_utils.Crease(
              c.p1_init,
              c.p2_init,
              fold=assigned[idx],
              fold_percent=pct,
              elevation=c.elevation,
          )
      )
    else:
      updated.append(c)
  cand = crease_pattern_lib.CreasePattern(
      creases=updated,
      grid_size=-1,
      light_mode=True,
      tolerance=pattern.tolerance,
  )
  return cand, len(assigned)


def fold_spine(
    state: state_lib.ShapingState,
    *,
    fold_percent: float = 0.65,
    element_name: str | None = None,
    fold_type: str = 'mountain',
    verify_fold: bool = True,
) -> state_lib.ToolOutcome:
  """Partially folds the `x = 0.5` spine for 3-D arch or body tenting."""
  if not 0.0 < fold_percent <= 1.0:
    raise ValueError(f'fold_percent must be in (0, 1], got {fold_percent}.')
  fold_key = fold_type.lower()
  if fold_key not in ('mountain', 'valley'):
    raise ValueError(
        f"fold_type must be 'mountain' or 'valley', got {fold_type!r}."
    )
  target_fold = (
      crease_utils.Fold.MOUNTAIN
      if fold_key == 'mountain'
      else crease_utils.Fold.VALLEY
  )
  axis_x = 0.5
  tol = 0.025

  if element_name is not None:
    elem = state.element(element_name)
    if isinstance(elem, shaping_elements.ShapingFlap):
      rect = elem.to_legacy_rectangle()
      y_min = max(0.0, min(rect.y_start, rect.y_start + rect.y_len))
      y_max = min(1.0, max(rect.y_start, rect.y_start + rect.y_len))
    else:
      segs = list(elem.bank_1) + list(elem.bank_2)
      if not segs:
        raise ValueError(f'fold_spine({element_name}): river has no banks.')
      ys = [c.p1.y for c in segs] + [c.p2.y for c in segs]
      y_min, y_max = max(0.0, min(ys)), min(1.0, max(ys))
    label = element_name
  else:
    y_min, y_max, label = _central_spine_bounds(state, axis_x=axis_x, tol=tol)

  spans = [(y_min, y_max)]
  if y_min > 1e-4 or y_max < 1.0 - 1e-4:
    spans.append((0.0, 1.0))

  pcts = _dedup_floats(
      [fold_percent, 0.8, 0.88, 0.93, 0.5] if fold_percent < 1.0 else [1.0]
  )
  last_reason = 'no foldable spine configuration'
  strain_reason: str | None = None

  for y0, y1 in spans:
    for pct in pcts:
      for keep_existing, first_fold in (
          (True, target_fold),
          (False, target_fold),
          (False, target_fold.flip()),
      ):
        try:
          built = _build_spine_candidate(
              state.pattern,
              axis_x=axis_x,
              y0=y0,
              y1=y1,
              first_fold=first_fold,
              pct=pct,
              keep_existing=keep_existing,
          )
        except (ValueError, AssertionError) as error:
          last_reason = str(error)
          continue
        if built is None:
          continue
        cand, n_spine = built
        accepted, reason, model = state_lib.accept_pattern(
            state,
            cand,
            allow_3d_vertices=pct < 1.0,
            verify_fold=verify_fold,
        )
        if accepted is None:
          if reason.startswith('vertex error') and strain_reason is None:
            strain_reason = reason
          last_reason = reason
          continue
        relief = ''
        if model is not None and model.vertices_3d.size:
          z_span = float(
              model.vertices_3d[:, 2].max() - model.vertices_3d[:, 2].min()
          )
          relief = f', z_span={z_span:.3f}'
        relaxed = not math.isclose(pct, fold_percent, abs_tol=1e-6)
        fb_note = (
            f' [fallback from fold_percent={fold_percent:g}: {last_reason}]'
            if relaxed
            else ''
        )
        resolved: dict[str, object] | None = None
        if relaxed:
          resolved = {
              'fold_percent': pct,
              'fold_type': fold_type,
              'verify_fold': verify_fold,
          }
          if element_name is not None:
            resolved['element_name'] = element_name
        return state_lib.ToolOutcome(
            state=accepted,
            note=(
                f'{label}: folded {n_spine} spine crease(s) at'
                f' fold_percent={pct:g}{relief}{fb_note}.'
            ),
            resolved_arguments=resolved,
        )

  raise ValueError(
      f'fold_spine({label}) failed: {strain_reason or last_reason}'
  )
