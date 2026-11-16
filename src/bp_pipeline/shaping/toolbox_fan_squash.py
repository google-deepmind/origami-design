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

"""Fan-out and squash shaping tool implementations for the shaping toolbox.

Provides `fan_out_fold_impl`, `fan_out_fold_heuristic_impl`, and
`squash_fold_impl` with `open_percent` support, merged-chain and border-pleat
discovery, on-paper ridge selection, border-clipped triangular squashes, and
deterministic-folder verification. This module does not import `shaping`.
"""

from collections.abc import Sequence
import dataclasses
import datetime
import math
import os
import time
from typing import Any

from absl import logging

from ..crease_pattern import crease as crease_utils
from ..crease_pattern import crease_pattern as crease_pattern_lib
from ..crease_pattern.parser import svg as svg_parser
from . import folder_3d as folder
from .. import timeout as timeout_lib
from . import fidelity
from . import fan_out_shaping
from . import flat_foldable
from . import shaping_elements
from . import shaping_utils
from . import squash_sink_shaping

_FOLD_TYPES = {
    'mountain': crease_utils.Fold.MOUNTAIN,
    'valley': crease_utils.Fold.VALLEY,
}


@dataclasses.dataclass
class FanSquashReport:
  """Details of the last `fan_out` or `squash` call for tool notes."""

  tool: str = ''
  element_name: str = ''
  note: str = ''
  num_new_creases: int = 0
  width_ratio: float = 1.0
  locality: float = 1.0
  open_percent: float = 1.0
  elapsed_s: float = 0.0


LAST_REPORT = FanSquashReport()


def last_report_message() -> str:
  """Returns the note from the last call (empty string if none)."""
  if not LAST_REPORT.tool:
    return ''
  metrics = []
  if LAST_REPORT.open_percent < 1.0:
    metrics.append(f'open={LAST_REPORT.open_percent:.2f}')
  if LAST_REPORT.num_new_creases > 0:
    metrics.append(f'+{LAST_REPORT.num_new_creases} creases')
  if LAST_REPORT.width_ratio > 0.0 and not math.isclose(
      LAST_REPORT.width_ratio, 1.0, rel_tol=1e-2
  ):
    metrics.append(f'width={LAST_REPORT.width_ratio:.2f}x')
  if LAST_REPORT.locality > 0.0:
    metrics.append(f'locality={LAST_REPORT.locality:.2f}')
  if LAST_REPORT.elapsed_s > 0.0:
    metrics.append(f'{LAST_REPORT.elapsed_s:.2f}s')
  prefix = (
      f'{LAST_REPORT.tool}({LAST_REPORT.element_name}): {", ".join(metrics)}'
  )
  if LAST_REPORT.note:
    return f'{prefix}. {LAST_REPORT.note}'
  return prefix


# Vertex error (percent) above which the fold stage refuses a pattern
# (`lib.fold_and_render`, `shaping._MAX_RENDERABLE_VERTEX_ERROR`). Candidates
# above it must be rejected here too: on an eagle base (grid 22, central crease
# removed) `fan_out_fold` kept a 14-crease fan whose fold had 2.9-3.9 % vertex
# error, reported success, and then every later tool failed ("the current
# pattern does not fold", `rotation_3d` FF violations) and `--stage=fold`
# raised.
MAX_RENDERABLE_VERTEX_ERROR = 1.0


def folds(pattern: crease_pattern_lib.CreasePattern) -> tuple[bool, str]:
  """Checks whether the deterministic folder folds `pattern` cleanly.

  "Cleanly" means locally flat-foldable (or 3-D Belcastro-Hull foldable when
  3-D relief creases are present), the deterministic folder succeeds and its
  vertex error stays within the fold stage's limit
  (`MAX_RENDERABLE_VERTEX_ERROR`).

  Args:
    pattern: Crease pattern to check.

  Returns:
    `(ok, reason)`.
  """
  ok, reason, _ = _fold_once(pattern)
  return ok, reason


def _fold_once(
    pattern: crease_pattern_lib.CreasePattern,
) -> tuple[bool, str, Any]:
  """Like `folds` but also returns the folded model (None on failure)."""
  if (
      not pattern.local_flat_foldability()
      and not pattern.is_locally_foldable()
      and not fidelity.has_3d_overlay(pattern)
  ):
    return False, 'local foldability failed', None
  start = time.perf_counter()
  try:
    model = fidelity.FoldedModel.from_pattern(pattern)
  except Exception as error:  # pylint: disable=broad-exception-caught
    return False, f'folder: {str(error)[:120]}', None
  vertex_error = getattr(model, 'vertex_error', 0.0)
  if vertex_error > MAX_RENDERABLE_VERTEX_ERROR:
    return (
        False,
        (
            f'folder: vertex error {vertex_error:.2f}% >'
            f' {MAX_RENDERABLE_VERTEX_ERROR}% (the fold stage limit)'
        ),
        None,
    )
  return True, f'folds ({time.perf_counter() - start:.2f}s)', model


def _crease_key(
    crease: crease_utils.Crease, ndigits: int = 6
) -> tuple[float, float, float, float]:
  """Canonical rounded endpoints `(x1, y1, x2, y2)` of a crease segment."""
  p1 = (round(crease.p1_init.x, ndigits), round(crease.p1_init.y, ndigits))
  p2 = (round(crease.p2_init.x, ndigits), round(crease.p2_init.y, ndigits))
  return (*p1, *p2) if p1 <= p2 else (*p2, *p1)


def _new_crease_count(
    before: crease_pattern_lib.CreasePattern,
    after: crease_pattern_lib.CreasePattern,
) -> int:
  """Counts creases in `after` whose geometry is not in `before`."""
  old_keys = {_crease_key(c) for c in before.creases}
  return sum(1 for c in after.creases if _crease_key(c) not in old_keys)


def _compute_flap_locality(
    before: crease_pattern_lib.CreasePattern,
    after: crease_pattern_lib.CreasePattern,
    element: shaping_elements.ShapingFlap,
) -> float:
  """Computes fraction of new/modified creases inside element's padded bbox."""
  old_keys = {_crease_key(c) for c in before.creases}
  new_creases = [c for c in after.creases if _crease_key(c) not in old_keys]
  if not new_creases:
    return 1.0
  x0 = min(element.x_start, element.x_start + element.x_len)
  x1 = max(element.x_start, element.x_start + element.x_len)
  y0 = min(element.y_start, element.y_start + element.y_len)
  y1 = max(element.y_start, element.y_start + element.y_len)
  tol = 0.02
  inside = 0
  for c in new_creases:
    mx = (c.p1_init.x + c.p2_init.x) / 2.0
    my = (c.p1_init.y + c.p2_init.y) / 2.0
    if (x0 - tol <= mx <= x1 + tol) and (y0 - tol <= my <= y1 + tol):
      inside += 1
  return round(inside / len(new_creases), 3)


def _bilateral_counterpart(name: str) -> str | None:
  """Returns symmetric counterpart name for bilateral flap pairs."""
  if name.endswith('_L'):
    return name[:-2] + '_R'
  if name.endswith('_R'):
    return name[:-2] + '_L'
  if name.endswith('_l'):
    return name[:-2] + '_r'
  if name.endswith('_r'):
    return name[:-2] + '_l'
  if name.endswith('_left'):
    return name[:-5] + '_right'
  if name.endswith('_right'):
    return name[:-5] + '_left'
  if name.startswith('left_'):
    return 'right_' + name[5:]
  if name.startswith('right_'):
    return 'left_' + name[6:]
  return None


def _mirror_crease_pattern_x(
    cp: crease_pattern_lib.CreasePattern,
) -> crease_pattern_lib.CreasePattern:
  """Reflects a CreasePattern across x = 0.5."""
  mirrored_creases = []
  for c in cp.creases:
    p1 = dataclasses.replace(c.p1_init, x=1.0 - c.p1_init.x)
    p2 = dataclasses.replace(c.p2_init, x=1.0 - c.p2_init.x)
    mirrored_creases.append(
        crease_utils.Crease(
            p1,
            p2,
            fold=c.fold,
            fold_percent=c.fold_percent,
            elevation=c.elevation,
        )
    )
  mirrored = crease_pattern_lib.CreasePattern(
      creases=mirrored_creases, light_mode=cp.light_mode
  )
  mirrored.tolerance = cp.tolerance
  return mirrored


def _get_base_width_dict(
    state: Any,
    element_name: str,
) -> tuple[Any, dict[float, float]]:
  """Precomputes layout and baseline section widths for element_name."""
  try:
    from . import fidelity  # pylint: disable=g-import-not-at-top

    layout = fidelity.paper_layout(state.tree, state.packing)
    m0 = fidelity.FoldedModel.from_pattern(state.pattern)
    g0 = fidelity.measure(state.tree, layout, m0)
    w0_dict = {
        frac: g0['sections'].get(f'{element_name}@{frac}', {}).get('width', 0.0)
        for frac in (0.25, 0.5, 0.75)
    }
    return layout, w0_dict
  except Exception:  # pylint: disable=broad-exception-caught
    return None, {}


def _measure_candidate_width_ratio(
    tree: Any,
    layout: Any,
    w0_dict: dict[float, float] | None,
    candidate_pattern: crease_pattern_lib.CreasePattern,
    element_name: str,
    model: Any = None,
) -> float:
  """Measures 3-D section width ratio of `element_name` against precomputed baseline.

  Args:
    tree: The stick figure.
    layout: `fidelity.paper_layout` of the base.
    w0_dict: Base section widths (`frac -> width`) of the element.
    candidate_pattern: The candidate pattern.
    element_name: The element.
    model: Its folded model, when the caller already folded it.

  Returns:
    The width ratio (1.0 when it cannot be measured).
  """
  if not w0_dict or layout is None:
    return 1.0
  try:
    from . import fidelity  # pylint: disable=g-import-not-at-top

    m1 = model or fidelity.FoldedModel.from_pattern(candidate_pattern)
    g1 = fidelity.measure(tree, layout, m1)
    w0 = w0_dict.get(0.5, 0.0)
    w1 = g1['sections'].get(f'{element_name}@0.5', {}).get('width', 0.0)
    if w0 <= 0.0:
      for frac in (0.25, 0.75):
        w0 = w0_dict.get(frac, 0.0)
        w1 = g1['sections'].get(f'{element_name}@{frac}', {}).get('width', 0.0)
        if w0 > 0.0:
          break
    ratio = (
        round(w1 / w0, 3) if w0 > 0.0 else (1.0 if w1 == 0.0 else round(w1, 3))
    )
    return ratio
  except Exception:  # pylint: disable=broad-exception-caught
    pass
  return 1.0


def _measure_width_ratio(
    state_before: Any,
    state_after: Any,
    element_name: str,
) -> float:
  """Measures 3-D section width ratio of `element_name` after vs before."""
  layout, w0_dict = _get_base_width_dict(state_before, element_name)
  return _measure_candidate_width_ratio(
      state_before.tree, layout, w0_dict, state_after.pattern, element_name
  )


def _pt_on_segment(
    pt: Any,
    p1: Any,
    p2: Any,
    tol: float = 1e-4,
) -> bool:
  """Returns True if point `pt` lies along the segment between `p1` and `p2`."""
  x_min, x_max = min(p1.x, p2.x), max(p1.x, p2.x)
  y_min, y_max = min(p1.y, p2.y), max(p1.y, p2.y)
  if not (
      x_min - tol <= pt.x <= x_max + tol and y_min - tol <= pt.y <= y_max + tol
  ):
    return False
  return (
      abs((p2.y - p1.y) * (pt.x - p1.x) - (p2.x - p1.x) * (pt.y - p1.y)) < tol
  )


def _preserve_base_borders(
    base_cp: crease_pattern_lib.CreasePattern,
    new_cp: crease_pattern_lib.CreasePattern,
) -> crease_pattern_lib.CreasePattern:
  """Restores base CP border segmentation, splitting only where new creases meet the border."""
  old_keys = {_crease_key(c) for c in base_cp.creases}
  new_internal = [
      c
      for c in new_cp.creases
      if c.fold != crease_utils.Fold.BORDER and _crease_key(c) not in old_keys
  ]
  border_touch_pts = []
  for c in new_internal:
    for pt in (c.p1_init, c.p2_init):
      if pt.is_border:
        border_touch_pts.append(pt)

  unique_border_pts = {}
  for pt in border_touch_pts:
    unique_border_pts[(round(pt.x, 6), round(pt.y, 6))] = pt
  border_touch_pts = list(unique_border_pts.values())

  current_borders = [
      c for c in base_cp.creases if c.fold == crease_utils.Fold.BORDER
  ]
  new_borders = []
  for bc in current_borders:
    pts_on_seg = []
    for pt in border_touch_pts:
      if not (
          pt.isclose(bc.p1_init) or pt.isclose(bc.p2_init)
      ) and _pt_on_segment(pt, bc.p1_init, bc.p2_init):
        pts_on_seg.append(pt)
    if not pts_on_seg:
      new_borders.append(bc)
      continue
    vx = bc.p2_init.x - bc.p1_init.x
    vy = bc.p2_init.y - bc.p1_init.y
    seg_len_sq = vx * vx + vy * vy
    pts_on_seg.sort(
        key=lambda p: (
            ((p.x - bc.p1_init.x) * vx + (p.y - bc.p1_init.y) * vy) / seg_len_sq
        )
    )
    all_pts = [bc.p1_init] + pts_on_seg + [bc.p2_init]
    for p_a, p_b in zip(all_pts, all_pts[1:]):
      new_borders.append(
          crease_utils.Crease(
              p_a, p_b, fold=crease_utils.Fold.BORDER, elevation=bc.elevation
          )
      )

  non_borders = [
      c for c in new_cp.creases if c.fold != crease_utils.Fold.BORDER
  ]
  result_cp = crease_pattern_lib.CreasePattern(
      creases=non_borders + new_borders, light_mode=new_cp.light_mode
  )
  result_cp.tolerance = new_cp.tolerance
  return result_cp


def _open_new_creases(
    before: crease_pattern_lib.CreasePattern,
    after: crease_pattern_lib.CreasePattern,
    fold_percent: float,
) -> crease_pattern_lib.CreasePattern:
  """Returns `after` with creases not in `before` set to `fold_percent`."""

  def key9(crease: crease_utils.Crease) -> tuple[float, ...]:
    ends = sorted([
        (round(crease.p1_init.x, 9), round(crease.p1_init.y, 9)),
        (round(crease.p2_init.x, 9), round(crease.p2_init.y, 9)),
    ])
    return (*ends[0], *ends[1])

  old = {key9(crease) for crease in before.creases}
  creases = []
  for crease in after.creases:
    if key9(crease) in old:
      creases.append(crease)
    else:
      creases.append(
          crease_utils.Crease(
              crease.p1_init,
              crease.p2_init,
              fold=crease.fold,
              fold_percent=fold_percent,
              elevation=crease.elevation,
          )
      )
  opened = crease_pattern_lib.CreasePattern(
      creases=creases, light_mode=after.light_mode
  )
  opened.tolerance = after.tolerance
  return opened


def _direction_key(crease: crease_utils.Crease) -> tuple[float, float]:
  """Canonical unit direction of a crease (sign-normalized, rounded)."""
  dx = crease.p2_init.x - crease.p1_init.x
  dy = crease.p2_init.y - crease.p1_init.y
  norm = math.hypot(dx, dy)
  if norm < 1e-12:
    return (0.0, 0.0)
  dx, dy = dx / norm, dy / norm
  if dx < -1e-9 or (abs(dx) <= 1e-9 and dy < 0):
    dx, dy = -dx, -dy
  return (round(dx, 4) + 0.0, round(dy, 4) + 0.0)


def _line_key(crease: crease_utils.Crease) -> tuple[float, float, float]:
  """Key of the infinite line through a crease: `(direction, offset)`."""
  ux, uy = _direction_key(crease)
  offset = -uy * crease.p1_init.x + ux * crease.p1_init.y
  return (ux, uy, round(offset, 4) + 0.0)


def _midpoint_in_rectangle(
    crease: crease_utils.Crease,
    rectangle: shaping_utils.Rectangle,
    tol: float = 1e-6,
) -> bool:
  """Whether the crease midpoint lies inside `rectangle` clipped to the paper."""
  x0 = max(0.0, min(rectangle.x_start, rectangle.x_start + rectangle.x_len))
  x1 = min(1.0, max(rectangle.x_start, rectangle.x_start + rectangle.x_len))
  y0 = max(0.0, min(rectangle.y_start, rectangle.y_start + rectangle.y_len))
  y1 = min(1.0, max(rectangle.y_start, rectangle.y_start + rectangle.y_len))
  mx = (crease.p1_init.x + crease.p2_init.x) / 2.0
  my = (crease.p1_init.y + crease.p2_init.y) / 2.0
  return x0 - tol <= mx <= x1 + tol and y0 - tol <= my <= y1 + tol


def _with_fold_percent(
    pattern: crease_pattern_lib.CreasePattern,
    target_keys: set[tuple[float, float, float, float]],
    fold_percent: float,
) -> crease_pattern_lib.CreasePattern:
  """Returns `pattern` with the creases in `target_keys` set to `fold_percent`."""
  creases = []
  for c in pattern.creases:
    if c.fold != crease_utils.Fold.BORDER and _crease_key(c) in target_keys:
      creases.append(
          crease_utils.Crease(
              c.p1_init,
              c.p2_init,
              fold=c.fold,
              fold_percent=fold_percent,
              elevation=c.elevation,
          )
      )
    else:
      creases.append(c)
  opened = crease_pattern_lib.CreasePattern(
      creases=creases, light_mode=pattern.light_mode
  )
  opened.tolerance = pattern.tolerance
  return opened


def _open_squash_creases(
    before: crease_pattern_lib.CreasePattern,
    shaped: crease_pattern_lib.CreasePattern,
    computed_ridge: crease_utils.Crease,
    strategy: str,
    fold_percent: float,
    rectangle: shaping_utils.Rectangle | None = None,
) -> crease_pattern_lib.CreasePattern:
  """Opens creases in `shaped` using targeted kite/axial/cross crease masks.

  Strategies: `axial_and_cross`, `axial_only`, `cross_only`, `all_kite`,
  `all_new` (every new crease) and `kite_and_flap_pleats`: the kite's axial
  and cross creases plus the squash's continuation pleats (`new_side_creases`
  of `apply_squash_sink_to_segment`, which replace the long side pleats) and
  every complete pleat line parallel to them that runs through `rectangle`.
  Opening only the 1-unit kite leaves those pleats folded flat (180 degree
  accordion), which pinches the rest of the flap back into a stick right
  above the kite (cobra hood, Phase 10: width 1.05x); opening the whole pleat
  lines lets the flap flare along its length with the kite as the gusset.
  Whole lines (all collinear segments) are opened so every vertex they cross
  stays rigidly consistent.

  Args:
    before: Pattern before the squash.
    shaped: Pattern after the squash.
    computed_ridge: The squashed ridge span.
    strategy: One of the strategies above.
    fold_percent: `fold_percent` for the opened creases.
    rectangle: The flap rectangle (`kite_and_flap_pleats` only).

  Returns:
    The opened pattern.
  """
  old_keys = {_crease_key(c) for c in before.creases}
  new_creases = [c for c in shaped.creases if _crease_key(c) not in old_keys]
  if not new_creases:
    return shaped

  deg: dict[tuple[float, float], int] = {}
  for c in new_creases:
    p1 = (round(c.p1_init.x, 5), round(c.p1_init.y, 5))
    p2 = (round(c.p2_init.x, 5), round(c.p2_init.y, 5))
    deg[p1] = deg.get(p1, 0) + 1
    deg[p2] = deg.get(p2, 0) + 1

  rx = computed_ridge.p2_init.x - computed_ridge.p1_init.x
  ry = computed_ridge.p2_init.y - computed_ridge.p1_init.y

  axial_keys = set()
  cross_keys = set()
  perimeter_keys = set()
  kite_keys = set()
  for c in new_creases:
    p1 = (round(c.p1_init.x, 5), round(c.p1_init.y, 5))
    p2 = (round(c.p2_init.x, 5), round(c.p2_init.y, 5))
    k = _crease_key(c)
    if deg[p1] > 1 and deg[p2] > 1:
      kite_keys.add(k)
      dx = c.p2_init.x - c.p1_init.x
      dy = c.p2_init.y - c.p1_init.y
      dot = rx * dx + ry * dy
      cross = rx * dy - ry * dx
      if math.isclose(cross, 0.0, abs_tol=1e-4) or c.is_collinear(
          computed_ridge
      ):
        axial_keys.add(k)
      elif math.isclose(dot, 0.0, abs_tol=1e-4):
        cross_keys.add(k)
      else:
        perimeter_keys.add(k)

  if strategy == 'axial_and_cross':
    target_keys = axial_keys | cross_keys
  elif strategy == 'axial_only':
    target_keys = axial_keys
  elif strategy == 'cross_only':
    target_keys = cross_keys
  elif strategy == 'all_kite':
    target_keys = kite_keys
  elif strategy == 'kite_and_flap_pleats':
    side_creases = [c for c in new_creases if _crease_key(c) not in kite_keys]
    pleat_dirs = {_direction_key(c) for c in side_creases}
    ridge_line = _line_key(computed_ridge)
    pleat_lines = {
        _line_key(c)
        for c in shaped.creases
        if c.fold != crease_utils.Fold.BORDER
        and _direction_key(c) in pleat_dirs
        and _line_key(c) != ridge_line
        and (rectangle is None or _midpoint_in_rectangle(c, rectangle))
    }
    target_keys = (
        axial_keys
        | cross_keys
        | {_crease_key(c) for c in side_creases}
        | {
            _crease_key(c)
            for c in shaped.creases
            if c.fold != crease_utils.Fold.BORDER
            and _line_key(c) in pleat_lines
        }
    )
  else:  # all_new
    target_keys = {_crease_key(c) for c in new_creases}

  if not target_keys:
    target_keys = {_crease_key(c) for c in new_creases}

  return _with_fold_percent(shaped, target_keys, fold_percent)


def _flap_rectangle(state: Any, element_name: str) -> shaping_utils.Rectangle:
  """Returns the legacy rectangle of a flap, rejecting rivers."""
  element = state.element(element_name)
  if not isinstance(element, shaping_elements.ShapingFlap):
    raise ValueError(
        f'{element_name} is a {type(element).__name__}; this tool only'
        ' applies to flaps.'
    )
  return shaping_utils.Rectangle(
      name=element.name,
      x_start=element.x_start,
      y_start=element.y_start,
      x_len=element.x_len,
      y_len=element.y_len,
  )


def _chain_rectangle(
    state: Any, element_name: str
) -> shaping_utils.Rectangle | None:
  """Returns the legacy rectangle of the merged chain containing `element_name`."""
  _, merged_names = state.tree.merge_linear_chains()
  flaps, _ = state.merged_shaping_elements
  for merged_name, sub_names in merged_names.items():
    if element_name in sub_names and len(sub_names) > 1:
      for flap in flaps:
        if flap.name == merged_name:
          return shaping_utils.Rectangle(
              name=flap.name,
              x_start=flap.x_start,
              y_start=flap.y_start,
              x_len=flap.x_len,
              y_len=flap.y_len,
          )
  return None


# ------------------------------------------------------------------------------
# Fan-out shaping implementations.
# ------------------------------------------------------------------------------


# Wall-clock budget of one fan-out search (both tools). The search tries up to
# 2 orientations x 2 rectangles x 4 deltas x 2 fold types; each attempt runs
# the flat-foldability solver and the folder, so without a budget checked
# between attempts a single call could run for minutes (Matrix 10: 169 s on a
# crab pincer).
FAN_OUT_TIME_BUDGET_S = 20.0

# Wall-clock limit of one flat-foldability solve. The solver only limits
# candidate counts, and every candidate rebuilds the crease pattern (~0.1 s on
# a grid-30 base): a whole-paper solve of a 6-pleat scorpion-claw fan ran for
# minutes. Single-pleat solutions, when they exist, take 0.4-1.6 s.
SOLVER_TIMEOUT_S = 2.0

# Multiplier applied to both wall-clock limits above, read from the
# ORIGAMI_SHAPING_TIME_SCALE environment variable (default 1). Under heavy
# machine load the same fan can hit a limit on one run and not on the next
# (eagle v3, load ~40: wing_R fanned 1.58x on one run and 3.35x on another;
# with the limits raised 4x the 3.35x result reproduced exactly).
_TIME_SCALE_ENV = 'ORIGAMI_SHAPING_TIME_SCALE'


def _time_scale() -> float:
  try:
    scale = float(os.environ.get(_TIME_SCALE_ENV, '1'))
  except ValueError:
    return 1.0
  return scale if scale > 0 else 1.0


# Flat-foldability solver limits scale with the number of pleats fanned at
# once (every fanned pleat needs 1-2 crease rotations around its tip).
SOLVER_CANDIDATES_PER_PLEAT = 12
SOLVER_MODIFICATIONS_PER_PLEAT = 5
MIN_SOLVER_CANDIDATES = 12
MIN_SOLVER_MODIFICATIONS = 6
MAX_SOLVER_CANDIDATES = 60
MAX_SOLVER_MODIFICATIONS = 25

# Fans are grown one pleat at a time on the already solved pattern
# (`_fan_incrementally`): the simultaneous solve of a 4-5 pleat eagle wing or
# a 2-per-side eagle tail times out, the single-pleat steps take ~0.5 s each.
# Every pleat is tried at the requested delta, then at these fractions of it:
# the solver only fixes a pleat tilted by up to ~3 degrees (eagle wing, pleat
# length 13: delta 0.5 folds, 0.75 does not; eagle tail, length 5: the
# outermost pleat takes 1.0, the interior ones 0.25).
PLEAT_DELTA_FACTORS = (1.0, 0.5, 0.25)
# Most pleats fanned on one flap (more unrolls the flap instead of fanning it).
MAX_FAN_PLEATS = 6
# A fan stops after this many pleats that fold at no delta factor.
MAX_MISSED_PLEATS = 2
# Most pleats fanned on each side of a flap that spans the paper.
MAX_BILATERAL_PLEATS_PER_SIDE = 3

# Pleats shorter than this fraction of the longest fanned pleat are hinge stubs
# (1-unit creases of a neighbouring flap inside the flap rectangle), not part
# of the flap's accordion; they are never fanned.
MIN_PLEAT_LENGTH_FRACTION = 0.5

# Margin (grid units) around a fanned pleat for the local flat-foldability
# solve (see `_pleat_rectangle`).
PLEAT_RECTANGLE_MARGIN = 1.5

# A fan that widens the flap less than this is "weak": the tool only moved a
# hinge stub (an eagle tail without pleats: +5 creases, 1.00x). Weak fans are
# kept only when no other fan is found, with a note.
MIN_FAN_WIDTH_RATIO = 1.05


def _over_budget(start_time: float) -> bool:
  return (
      time.perf_counter() - start_time > FAN_OUT_TIME_BUDGET_S * _time_scale()
  )


def _pleat_rectangle(fanned: Any, grid_size: int) -> shaping_utils.Rectangle:
  """Rectangle around a fanned pleat for the local flat-foldability solve.

  Covers the original and the fanned crease plus `PLEAT_RECTANGLE_MARGIN`
  grid units (the pleat's tip vertex and its neighbours on the spine). The
  library records the whole flap rectangle instead, which on a tail stretched
  across the paper lets the solver rotate creases anywhere in a half-paper
  strip and times out.

  Args:
    fanned: The fanned pleat.
    grid_size: Solving grid size.

  Returns:
    The clipped rectangle.
  """
  margin = PLEAT_RECTANGLE_MARGIN / grid_size
  points = [
      fanned.original_crease.p1,
      fanned.original_crease.p2,
      fanned.fanned_out_crease.p1,
      fanned.fanned_out_crease.p2,
  ]
  x0 = max(0.0, min(p.x for p in points) - margin)
  x1 = min(1.0, max(p.x for p in points) + margin)
  y0 = max(0.0, min(p.y for p in points) - margin)
  y1 = min(1.0, max(p.y for p in points) + margin)
  return shaping_utils.Rectangle(
      name=fanned.rectangle.name,
      x_start=x0,
      y_start=y0,
      x_len=x1 - x0,
      y_len=y1 - y0,
  )


def _subset_candidate(
    kept_fanned: Sequence[Any],
    unmodified_creases: Sequence[crease_utils.Crease],
    indices: Sequence[int],
    grid_size: int,
) -> fan_out_shaping.FanOutCandidates | None:
  """`FanOutCandidates` fanning only `indices` of `kept_fanned` (tight rectangles)."""
  chosen = set(indices)
  rest = [
      kept_fanned[j].original_crease
      for j in range(len(kept_fanned))
      if j not in chosen
  ]
  cand = fan_out_shaping._candidates_from_fanned(  # pylint: disable=protected-access
      [kept_fanned[i] for i in indices], list(unmodified_creases) + rest
  )
  if cand is not None:
    cand.modified_rectangles = {
        _pleat_rectangle(kept_fanned[i], grid_size) for i in indices
    }
  return cand


def _fan_candidates_with_fanned(
    pattern: crease_pattern_lib.CreasePattern,
    params: Sequence[fan_out_shaping.FanOutShapingParams],
    grid_size: int,
) -> tuple[Any, list[Any], list[crease_utils.Crease]]:
  """Like `fan_out_shaping._get_fan_out_candidates_with_fanned`, finding more.

  The library returns the first pass that finds anything: creases inside the
  rectangle, else border-touching creases whose border point is inside it.
  On an eagle wing the strict pass finds only a 1-unit hinge stub of the
  neighbouring leg inside the wing rectangle and hides the wing's five long
  pleats, which run past the rectangle to the body's spine. Both passes are
  run and the one with more pleats wins.

  Args:
    pattern: Base pattern.
    params: Fan-out parameters.
    grid_size: Solving grid size.

  Returns:
    `(candidates or None, kept fanned pleats, unmodified creases)`.
  """
  # pylint: disable=protected-access
  cand, kept, unmod = fan_out_shaping._get_fan_out_candidates_with_fanned(
      pattern, params, grid_size
  )
  working = fan_out_shaping._split_merged_border_creases(pattern, params)
  # pylint: enable=protected-access
  loose_unmod, loose_potential = (
      fan_out_shaping._collect_potential_fanned_creases(  # pylint: disable=protected-access
          working, params, grid_size, allow_border_tip_only=True
      )
  )
  loose_kept, loose_unmod = (
      fan_out_shaping._filter_non_intersecting_candidates(  # pylint: disable=protected-access
          loose_unmod, loose_potential
      )
  )
  loose_cand = fan_out_shaping._candidates_from_fanned(  # pylint: disable=protected-access
      loose_kept, loose_unmod
  )
  if loose_cand is not None and (cand is None or len(loose_kept) > len(kept)):
    return loose_cand, loose_kept, loose_unmod
  return cand, kept, unmod


def _spans_paper(
    rectangle: shaping_utils.Rectangle,
    orientation: fan_out_shaping.Orientation,
) -> bool:
  """Whether the flap rectangle reaches both borders across the fan axis."""
  clipped = rectangle.clip()
  if clipped is None:
    return False
  tol = 0.02
  if orientation == fan_out_shaping.Orientation.VERTICAL:
    return clipped.x_start <= tol and clipped.x_end >= 1.0 - tol
  return clipped.y_start <= tol and clipped.y_end >= 1.0 - tol


def _split_rectangle(
    rectangle: shaping_utils.Rectangle,
    orientation: fan_out_shaping.Orientation,
    grid_size: int,
) -> tuple[shaping_utils.Rectangle, shaping_utils.Rectangle]:
  """Halves of `rectangle` on either side of the paper's centre line.

  The centre column (row) is left out so a pleat on the axis is not fanned.

  Args:
    rectangle: The flap rectangle (spanning the paper, see `_spans_paper`).
    orientation: Orientation of the fanned pleats.
    grid_size: Solving grid size.

  Returns:
    `(first, second)`: the left/right (top/bottom) halves.
  """
  clipped = rectangle.clip()
  assert clipped is not None
  gap = 0.5 / grid_size
  if orientation == fan_out_shaping.Orientation.VERTICAL:
    first = shaping_utils.Rectangle(
        name=f'{rectangle.name}_a',
        x_start=0.0,
        y_start=clipped.y_start,
        x_len=0.5 - gap,
        y_len=clipped.y_len,
    )
    second = shaping_utils.Rectangle(
        name=f'{rectangle.name}_b',
        x_start=0.5 + gap,
        y_start=clipped.y_start,
        x_len=0.5 - gap,
        y_len=clipped.y_len,
    )
  else:
    first = shaping_utils.Rectangle(
        name=f'{rectangle.name}_a',
        x_start=clipped.x_start,
        y_start=0.0,
        x_len=clipped.x_len,
        y_len=0.5 - gap,
    )
    second = shaping_utils.Rectangle(
        name=f'{rectangle.name}_b',
        x_start=clipped.x_start,
        y_start=0.5 + gap,
        x_len=clipped.x_len,
        y_len=0.5 - gap,
    )
  return first, second


def _solve_flat_foldable(
    **kwargs,
) -> tuple[crease_pattern_lib.CreasePattern | None, bool]:
  """`flat_foldable.get_flat_foldable` bounded by `SOLVER_TIMEOUT_S`.

  Uses the library's cooperative timeout (checked inside the solver loops),
  never exceeding an enclosing cooperative timeout.

  Args:
    **kwargs: Arguments of `flat_foldable.get_flat_foldable`.

  Returns:
    `(solution or None, timed_out)`.
  """
  limit = datetime.timedelta(seconds=SOLVER_TIMEOUT_S * _time_scale())
  outer = timeout_lib.get_timeout()
  if outer is not None:
    limit = min(limit, outer.remaining)
    if limit <= datetime.timedelta(0):
      return None, True
  try:
    with timeout_lib.cooperative_timeout(limit):
      return flat_foldable.get_flat_foldable(**kwargs), False
  except timeout_lib.CooperativeTimeoutError:
    return None, True


def _flat_foldable_fans(
    pattern: crease_pattern_lib.CreasePattern,
    candidates: fan_out_shaping.FanOutCandidates,
    max_candidates: int,
    max_mods: int,
):
  """Yields flat-foldable patterns for `candidates`.

  Like `fan_out_shaping._make_flat_foldable`: solved inside the fanned
  rectangles first, then (if the base pattern was flat-foldable) across the
  whole paper. Unlike it, both solutions are yielded (the local one sometimes
  solves but does not fold cleanly), every solve is bounded by
  `SOLVER_TIMEOUT_S`, and the whole-paper solve is skipped when the local one
  timed out (it would too).

  Args:
    pattern: The base pattern.
    candidates: Fan-out candidates (modified and unmodified creases).
    max_candidates: Solver candidate limit.
    max_mods: Solver crease-modification limit.

  Yields:
    Flat-foldable crease patterns, local solution first.
  """
  all_creases = candidates.unmodified_creases + candidates.modified_creases
  if pattern.light_mode:
    all_creases = flat_foldable.split_border_creases_at_tips(all_creases)
  potential = crease_pattern_lib.CreasePattern(
      creases=all_creases, light_mode=pattern.light_mode, merge_trivial=True
  )
  matched = potential.get_matching_creases(candidates.modified_creases)
  local, timed_out = _solve_flat_foldable(
      cp=potential,
      rectangles=tuple(candidates.modified_rectangles),
      unmodifiable_creases=matched,
      max_candidates=max_candidates,
      max_crease_modifications=max_mods,
  )
  if local is not None:
    yield local
  if timed_out:
    return
  base_score, _, _ = pattern.vertex_flat_foldability_percentage()
  if base_score < 1.0:
    return
  whole, _ = _solve_flat_foldable(
      cp=potential,
      rectangles=None,
      unmodifiable_creases=matched,
      max_candidates=max_candidates,
      max_crease_modifications=max_mods,
  )
  if whole is not None:
    yield whole


def _verified_fan_state(
    state: Any,
    ff_cp: crease_pattern_lib.CreasePattern,
    open_percent: float,
) -> Any | None:
  """Restores borders, applies `open_percent` and checks that the fan folds."""
  ff_cp = _preserve_base_borders(state.pattern, ff_cp)
  if _new_crease_count(state.pattern, ff_cp) == 0:
    return None

  open_candidates = [open_percent]
  if open_percent < 1.0:
    for fallback_pct in (0.7, 0.85, 1.0):
      if fallback_pct > open_percent:
        open_candidates.append(fallback_pct)

  first_valid: Any = None
  for pct in open_candidates:
    target_cp = (
        _open_new_creases(state.pattern, ff_cp, pct) if pct < 1.0 else ff_cp
    )
    try:
      result_state = state.with_pattern(target_cp)
    except ValueError:
      continue

    if state.config.check_flat_foldability:
      ok, _ = folds(result_state.pattern)
      if not ok:
        continue
      if pct < 1.0:
        verr = fidelity.FoldedModel.from_pattern(
            result_state.pattern
        ).vertex_error
        if verr > fidelity.CLEAN_GATE_MAX_VERTEX_ERROR:
          if first_valid is None:
            first_valid = result_state
          continue
    return result_state
  return first_valid


def _try_fan_out_candidate(
    state: Any,
    candidates: fan_out_shaping.FanOutCandidates,
    open_percent: float,
    *,
    max_candidates: int | None = None,
    max_mods: int | None = None,
) -> Any | None:
  """Makes `candidates` flat-foldable, applies `open_percent`, verifies folding.

  Solver limits default to `SOLVER_*_PER_PLEAT` x pleats (between the `MIN_`
  and `MAX_` constants).

  Args:
    state: Current state.
    candidates: Fan-out candidates.
    open_percent: `fold_percent` of the new creases.
    max_candidates: Solver candidate limit override.
    max_mods: Solver crease-modification limit override.

  Returns:
    The fanned state, or None if no solution folds.
  """
  pleats = len(candidates.modified_creases)
  if max_candidates is None:
    max_candidates = min(
        MAX_SOLVER_CANDIDATES,
        max(MIN_SOLVER_CANDIDATES, SOLVER_CANDIDATES_PER_PLEAT * pleats),
    )
  if max_mods is None:
    max_mods = min(
        MAX_SOLVER_MODIFICATIONS,
        max(MIN_SOLVER_MODIFICATIONS, SOLVER_MODIFICATIONS_PER_PLEAT * pleats),
    )
  for ff_cp in _flat_foldable_fans(
      state.pattern, candidates, max_candidates, max_mods
  ):
    result_state = _verified_fan_state(state, ff_cp, open_percent)
    if result_state is not None:
      return result_state
  return None


@dataclasses.dataclass
class _FanResult:
  """A working fan: its state and the metrics reported in `LAST_REPORT`."""

  state: Any
  ratio: float
  new_creases: int
  locality: float
  pleats: int
  label: str = ''
  # Deltas (grid units) of the fanned pleats.
  deltas: tuple[float, ...] = ()
  # Did not widen the flap measurably (see `MIN_FAN_WIDTH_RATIO`).
  weak: bool = False

  def better_than(self, other: '_FanResult | None') -> bool:
    """A widening fan beats a weak one, then more pleats, then wider."""
    if other is None:
      return True
    if self.weak != other.weak:
      return not self.weak
    if self.pleats != other.pleats:
      return self.pleats > other.pleats
    return self.ratio > other.ratio


@dataclasses.dataclass
class _FanPlan:
  """The pleats of one flap to fan for one fold type, in fanning order."""

  label: str
  kept_fanned: list[Any]
  unmodified: list[crease_utils.Crease]
  order: list[int]  # Indices into `kept_fanned`.
  deltas: list[float]  # Border-point shift (grid units) of each `order` entry.
  bilateral: bool


def _pleat_coordinate(
    fanned: Any, orientation: fan_out_shaping.Orientation
) -> float:
  """Position of a pleat across the fan axis (x of a vertical pleat)."""
  oc = fanned.original_crease
  if orientation == fan_out_shaping.Orientation.VERTICAL:
    return (oc.p1.x + oc.p2.x) / 2.0
  return (oc.p1.y + oc.p2.y) / 2.0


def _middle_out(items: Sequence[int]) -> list[int]:
  """Reorders position-sorted indices from the middle outwards."""
  if not items:
    return []
  mid = (len(items) - 1) // 2
  out = [items[mid]]
  for step in range(1, len(items)):
    for index in (mid + step, mid - step):
      if 0 <= index < len(items) and len(out) < len(items):
        out.append(items[index])
  return out


def _fan_plan(
    state: Any,
    rectangle: shaping_utils.Rectangle,
    orientation: fan_out_shaping.Orientation,
    fold_types: Sequence[crease_utils.Fold],
    move_delta: float,
    is_left: bool,
) -> _FanPlan | None:
  """Collects the pleats of a flap to fan and the order to fan them in.

  Hinge stubs (short pleats) are dropped. A flap that spans the paper across
  the fan axis (an eagle tail stretched along the top border) is fanned on
  both sides with opposite deltas, outermost pleats first and alternating
  sides, up to `MAX_BILATERAL_PLEATS_PER_SIDE` per side; any other flap is
  fanned from the middle pleat outwards (the end pleats, next to a 45-degree
  ridge or a neighbouring flap, are the ones that fail), up to
  `MAX_FAN_PLEATS`.

  Args:
    state: Current state.
    rectangle: Flap (or chain) rectangle.
    orientation: Pleat orientation.
    fold_types: Fold types of the pleats to fan.
    move_delta: Requested border-point shift (grid units).
    is_left: Whether the flap is on the left half of the paper.

  Returns:
    The plan, or None when the flap has no pleat to fan.
  """
  grid_size = state.solving_grid_size
  bilateral = _spans_paper(rectangle, orientation)
  if (
      is_left
      and orientation == fan_out_shaping.Orientation.VERTICAL
      and not bilateral
  ):
    delta = -move_delta
  else:
    delta = move_delta
  if bilateral:
    first, second = _split_rectangle(rectangle, orientation, grid_size)
    rect_deltas = [(first, delta), (second, -delta)]
  else:
    rect_deltas = [(rectangle, delta)]
  params = [
      fan_out_shaping.FanOutShapingParams(
          valid_fold_types=list(fold_types),
          valid_orientations=[orientation],
          rectangles=[rect],
          move_delta=rect_delta,
      )
      for rect, rect_delta in rect_deltas
  ]
  candidates, kept_fanned, unmodified = _fan_candidates_with_fanned(
      state.pattern, params, grid_size
  )
  if candidates is None:
    return None
  label = f'{orientation.value}/{"+".join(f.name.lower() for f in fold_types)}'
  max_len = max(fc.original_crease.length for fc in kept_fanned)
  long_pleats = sorted(
      (
          i
          for i, fc in enumerate(kept_fanned)
          if fc.original_crease.length >= MIN_PLEAT_LENGTH_FRACTION * max_len
      ),
      key=lambda i: _pleat_coordinate(kept_fanned[i], orientation),
  )
  if bilateral:
    first_side = [
        i
        for i in long_pleats
        if _pleat_coordinate(kept_fanned[i], orientation) < 0.5
    ][:MAX_BILATERAL_PLEATS_PER_SIDE]
    second_side = [
        i
        for i in reversed(long_pleats)
        if _pleat_coordinate(kept_fanned[i], orientation) >= 0.5
    ][:MAX_BILATERAL_PLEATS_PER_SIDE]
    order: list[int] = []
    deltas: list[float] = []
    pairs = list(zip(first_side, second_side))
    n_pairs = len(pairs)
    for pair_idx, pair in enumerate(pairs):
      scale = 1.0 - 0.45 * (pair_idx / (n_pairs - 1)) if n_pairs >= 2 else 1.0
      graded = delta * scale
      order.extend(pair)
      deltas.extend([graded, -graded])
  else:
    order = _middle_out(long_pleats)[:MAX_FAN_PLEATS]
    if len(order) >= 2:
      coords = [_pleat_coordinate(kept_fanned[i], orientation) for i in order]
      c_min, c_max = min(coords), max(coords)
      span = c_max - c_min
      if span > 1e-6:
        deltas = []
        for coord in coords:
          rel = (coord - c_min) / span if delta >= 0 else (c_max - coord) / span
          deltas.append(delta * (0.55 + 0.45 * rel))
      else:
        deltas = [delta] * len(order)
    else:
      deltas = [delta] * len(order)
  if not order:
    return None
  return _FanPlan(
      label=label,
      kept_fanned=list(kept_fanned),
      unmodified=list(unmodified),
      order=order,
      deltas=deltas,
      bilateral=bilateral,
  )


def _mirror_crease(
    crease: crease_utils.Crease, vertical: bool
) -> crease_utils.Crease:
  """Mirror image of a crease across x = 0.5 (`vertical`) or y = 0.5."""
  if vertical:
    p1 = dataclasses.replace(crease.p1_init, x=1.0 - crease.p1_init.x)
    p2 = dataclasses.replace(crease.p2_init, x=1.0 - crease.p2_init.x)
  else:
    p1 = dataclasses.replace(crease.p1_init, y=1.0 - crease.p1_init.y)
    p2 = dataclasses.replace(crease.p2_init, y=1.0 - crease.p2_init.y)
  return crease_utils.Crease(
      p1,
      p2,
      fold=crease.fold,
      fold_percent=crease.fold_percent,
      elevation=crease.elevation,
  )


def _is_mirror_symmetric(
    pattern: crease_pattern_lib.CreasePattern, vertical: bool
) -> bool:
  """Whether `pattern` equals its mirror image (geometry and fold types)."""
  keys = {(_crease_key(c), c.fold) for c in pattern.creases}
  mirrored = {
      (_crease_key(_mirror_crease(c, vertical)), c.fold)
      for c in pattern.creases
  }
  return keys == mirrored


def _mirror_half(
    pattern: crease_pattern_lib.CreasePattern,
    vertical: bool,
    keep_second: bool,
) -> crease_pattern_lib.CreasePattern:
  """Replaces one half of `pattern` by the mirror image of the other half.

  Halves are x < 0.5 / x > 0.5 for `vertical` pleats, else y < 0.5 / y > 0.5;
  `keep_second` keeps the second half. Creases crossing the centre line are
  kept once.

  Args:
    pattern: A pattern one half of which was shaped.
    vertical: Orientation of the fanned pleats.
    keep_second: Whether the shaped half is the second one.

  Returns:
    The symmetric pattern.
  """
  tol = 1e-6
  creases = []
  for c in pattern.creases:
    if vertical:
      lo, hi = sorted((c.p1_init.x, c.p2_init.x))
    else:
      lo, hi = sorted((c.p1_init.y, c.p2_init.y))
    in_first = hi <= 0.5 + tol
    in_second = lo >= 0.5 - tol
    if (in_first and keep_second) or (in_second and not keep_second):
      continue
    creases.append(c)
    if (in_second and keep_second) or (in_first and not keep_second):
      creases.append(_mirror_crease(c, vertical))
  mirrored = crease_pattern_lib.CreasePattern(
      creases=creases, light_mode=pattern.light_mode
  )
  mirrored.tolerance = pattern.tolerance
  return mirrored


def _fan_one_pleat(
    state: Any,
    pleat: Any,
    orientation: fan_out_shaping.Orientation,
    move_delta: float,
    open_percent: float,
) -> Any | None:
  """Fans one pleat of `state` (a single-pleat solve); None if it won't fold."""
  grid_size = state.solving_grid_size
  params = [
      fan_out_shaping.FanOutShapingParams(
          valid_fold_types=[pleat.original_crease.fold],
          valid_orientations=[orientation],
          rectangles=[_pleat_rectangle(pleat, grid_size)],
          move_delta=move_delta,
      )
  ]
  _, kept, unmodified = _fan_candidates_with_fanned(
      state.pattern, params, grid_size
  )
  key = _crease_key(pleat.original_crease)
  index = next(
      (
          i
          for i, fc in enumerate(kept)
          if _crease_key(fc.original_crease) == key
      ),
      None,
  )
  if index is None:
    return None
  single = _subset_candidate(kept, unmodified, [index], grid_size)
  if single is None:
    return None
  return _try_fan_out_candidate(state, single, open_percent)


def _fan_incrementally(
    state: Any,
    plan: _FanPlan,
    orientation: fan_out_shaping.Orientation,
    open_percent: float,
    start_time: float,
) -> tuple[Any, list[float]]:
  """Fans the plan's pleats one at a time, skipping those that do not fold.

  Each pleat is tried at its planned delta times `PLEAT_DELTA_FACTORS` in
  turn and keeps the first that folds; the next pleat starts from the factor
  that worked (a failed solve costs a `SOLVER_TIMEOUT_S` timeout, and the
  workable tilt is a property of the flap's accordion). The fan stops at the
  second pleat that folds at no factor: the fold's edge strain grows with
  every fanned pleat (an eagle tail's second pleat per side lifts it above
  `MAX_RENDERABLE_VERTEX_ERROR`), so the remaining pleats would fail too.

  Args:
    state: State before the fan.
    plan: The fan plan.
    orientation: Pleat orientation.
    open_percent: Requested `fold_percent`.
    start_time: Start of the search (time budget).

  Returns:
    `(state, deltas of the fanned pleats)`.
  """
  current = state
  deltas: list[float] = []
  first_factor = 0
  misses = 0
  for index, delta in zip(plan.order, plan.deltas):
    for k in range(first_factor, len(PLEAT_DELTA_FACTORS)):
      if _over_budget(start_time):
        return current, deltas
      factor = PLEAT_DELTA_FACTORS[k]
      next_state = _fan_one_pleat(
          current,
          plan.kept_fanned[index],
          orientation,
          delta * factor,
          open_percent,
      )
      if next_state is not None:
        current = next_state
        deltas.append(delta * factor)
        first_factor = k
        break
    else:
      misses += 1
      if misses >= MAX_MISSED_PLEATS:
        break
  return current, deltas


def _measure_fan(
    state: Any,
    element: shaping_elements.ShapingFlap,
    element_name: str,
    result_state: Any,
    layout: Any,
    w0_dict: dict[float, float] | None,
    label: str,
    deltas: Sequence[float],
) -> _FanResult:
  """Builds a `_FanResult` (weak when the fan did not widen the flap)."""
  ratio = _measure_candidate_width_ratio(
      state.tree, layout, w0_dict, result_state.pattern, element_name
  )
  return _FanResult(
      state=result_state,
      ratio=ratio,
      new_creases=_new_crease_count(state.pattern, result_state.pattern),
      locality=_compute_flap_locality(
          state.pattern, result_state.pattern, element
      ),
      pleats=len(deltas),
      label=label,
      deltas=tuple(deltas),
      weak=bool(layout is not None and w0_dict and ratio < MIN_FAN_WIDTH_RATIO),
  )


def _fan_plan_result(
    state: Any,
    element: shaping_elements.ShapingFlap,
    element_name: str,
    plan: _FanPlan,
    orientation: fan_out_shaping.Orientation,
    open_percent: float,
    layout: Any,
    w0_dict: dict[float, float] | None,
    start_time: float,
) -> _FanResult | None:
  """Fans a plan one pleat at a time and measures the result.

  A bilateral plan on a mirror-symmetric pattern fans one side only (each
  side is tried, the better fan wins) and mirrors the result onto the other
  side: the flat-foldability fixes of one side's pleats block the other
  side's (an eagle tail: x = 23, 21 fan after x = 25, then x = 3, 5 fail),
  and the solver's fixes are not mirror images of each other anyway.
  Otherwise both sides are fanned in turn.

  Args:
    state: Current shaping state.
    element: The flap being fanned.
    element_name: Name of the flap.
    plan: The candidate fan plan.
    orientation: Horizontal or vertical.
    open_percent: Degree of 3-D opening.
    layout: Base width layout.
    w0_dict: Baseline widths along the flap.
    start_time: Search start timestamp.

  Returns:
    The fan result, or None if no pleat could be fanned.
  """
  vertical = orientation == fan_out_shaping.Orientation.VERTICAL
  if plan.bilateral and _is_mirror_symmetric(state.pattern, vertical):
    best = None
    for second_side in (True, False):
      if _over_budget(start_time):
        break
      side = [
          (index, delta)
          for index, delta in zip(plan.order, plan.deltas)
          if (_pleat_coordinate(plan.kept_fanned[index], orientation) >= 0.5)
          == second_side
      ]
      side_plan = dataclasses.replace(
          plan,
          order=[index for index, _ in side],
          deltas=[delta for _, delta in side],
      )
      side_state, deltas = _fan_incrementally(
          state, side_plan, orientation, open_percent, start_time
      )
      if not deltas:
        continue
      mirrored = _mirror_half(
          side_state.pattern, vertical, keep_second=second_side
      )
      try:
        symmetric_state = state.with_pattern(mirrored)
      except ValueError:
        continue
      if (
          state.config.check_flat_foldability
          and not folds(symmetric_state.pattern)[0]
      ):
        continue
      result = _measure_fan(
          state,
          element,
          element_name,
          symmetric_state,
          layout,
          w0_dict,
          plan.label,
          [-d for d in deltas] + deltas,
      )
      if result.better_than(best):
        best = result
    if best is not None:
      return best
  result_state, deltas = _fan_incrementally(
      state, plan, orientation, open_percent, start_time
  )
  if not deltas:
    return None
  return _measure_fan(
      state,
      element,
      element_name,
      result_state,
      layout,
      w0_dict,
      plan.label,
      deltas,
  )


def _best_fan(
    state: Any,
    element: shaping_elements.ShapingFlap,
    element_name: str,
    rectangle: shaping_utils.Rectangle,
    orientations: Sequence[fan_out_shaping.Orientation],
    fold_type_sets: Sequence[Sequence[crease_utils.Fold]],
    move_delta: float,
    is_left: bool,
    open_percent: float,
    layout: Any,
    w0_dict: dict[float, float] | None,
    start_time: float,
) -> tuple[_FanResult | None, bool]:
  """Best fan over orientations x fold types for one rectangle and delta.

  Args:
    state: Current shaping state.
    element: The flap being fanned.
    element_name: Name of the flap.
    rectangle: Bounding rectangle for pleats.
    orientations: Orientations to try.
    fold_type_sets: Mountain/valley combinations to try.
    move_delta: Shift distance in grid units.
    is_left: Whether flap is on the left half of the paper.
    open_percent: Degree of 3-D opening.
    layout: Base width layout.
    w0_dict: Baseline widths along the flap.
    start_time: Search start timestamp.

  Returns:
    `(best result or None, whether any pleat could be fanned)`.
  """
  best = None
  saw_pleats = False
  for orientation in orientations:
    for fold_types in fold_type_sets:
      if _over_budget(start_time):
        return best, saw_pleats
      plan = _fan_plan(
          state, rectangle, orientation, fold_types, move_delta, is_left
      )
      if plan is None:
        continue
      saw_pleats = True
      result = _fan_plan_result(
          state,
          element,
          element_name,
          plan,
          orientation,
          open_percent,
          layout,
          w0_dict,
          start_time,
      )
      if result is not None and result.better_than(best):
        best = result
  return best, saw_pleats


def _record_fan_report(
    best: _FanResult, open_percent: float, start_time: float, note: str
) -> None:
  """Fills `LAST_REPORT` from a search result."""
  LAST_REPORT.num_new_creases = best.new_creases
  LAST_REPORT.locality = best.locality
  LAST_REPORT.width_ratio = best.ratio
  LAST_REPORT.open_percent = open_percent
  LAST_REPORT.elapsed_s = round(time.perf_counter() - start_time, 3)
  if note:
    LAST_REPORT.note = note


def _fan_deltas(state: Any, move_delta: float) -> list[float]:
  """Deltas tried in order: the requested one, then its opposite.

  Smaller magnitudes are tried per pleat (`PLEAT_DELTA_FACTORS`).

  Args:
    state: Current shaping state.
    move_delta: Requested move delta in grid units.

  Returns:
    List of deltas to try.
  """
  if abs(move_delta) > state.solving_grid_size:
    return []
  return [move_delta, -move_delta]


def _fan_notes(best: _FanResult) -> list[str]:
  """Note fragments describing how the fan was found."""
  notes = [f'{best.pleats} pleats' if best.pleats > 1 else '1 pleat']
  if best.deltas:
    notes[-1] += ' (deltas ' + ', '.join(f'{d:g}' for d in best.deltas) + ')'
  if best.weak:
    notes.append(
        f'no measurable widening ({best.ratio:.2f}x): the flap has no long'
        ' border pleats to fan'
    )
  return notes


def fan_out_fold_impl(
    state: Any,
    *,
    element_name: str,
    orientation: str,
    move_delta: float,
    fold_type: str | None = None,
    open_percent: float = 0.5,
) -> Any:
  """Fans out a flap by moving border crease points, with orientation/sign fallbacks.

  Args:
    state: Current `ShapingState`.
    element_name: Flap name.
    orientation: `'horizontal'` or `'vertical'`.
    move_delta: Grid units to shift border points.
    fold_type: Optional `'mountain'` or `'valley'`; both tried (separately, the
      wider fan wins) when None.
    open_percent: `fold_percent` in `(0, 1]` for newly created creases (default
      0.5 for 3-D flare; 1.0 for flat).

  Returns:
    Updated `ShapingState`.
  """
  global LAST_REPORT
  LAST_REPORT = FanSquashReport(tool='fan_out_fold', element_name=element_name)

  if not 0.0 < open_percent <= 1.0:
    raise ValueError(f'open_percent must be in (0, 1], got {open_percent}.')
  if math.isclose(move_delta, 0.0):
    raise ValueError('move_delta must be non-zero.')

  rectangle = _flap_rectangle(state, element_name)
  element = state.element(element_name)
  is_left = (element.x_start + element.x_len / 2.0) < 0.5
  counterpart = _bilateral_counterpart(element_name)
  if is_left and counterpart is not None and counterpart in state.tree:
    try:
      mirrored_state = state.replace(
          pattern=_mirror_crease_pattern_x(state.pattern)
      )
      res_m = fan_out_fold_impl(
          mirrored_state,
          element_name=counterpart,
          orientation=orientation,
          move_delta=move_delta,
          fold_type=fold_type,
          open_percent=open_percent,
      )
      result_pattern = _mirror_crease_pattern_x(res_m.pattern)
      result_state = state.replace(pattern=result_pattern)
      ok = True
      if state.config.check_flat_foldability:
        ok, _ = folds(result_state.pattern)
      if ok:
        LAST_REPORT.element_name = element_name
        LAST_REPORT.note = LAST_REPORT.note.replace(counterpart, element_name)
        return result_state
    except ValueError:
      pass
    LAST_REPORT = FanSquashReport(
        tool='fan_out_fold', element_name=element_name
    )

  layout, w0_dict = _get_base_width_dict(state, element_name)
  req_ori = fan_out_shaping.Orientation(orientation)
  other_ori = (
      fan_out_shaping.Orientation.VERTICAL
      if req_ori == fan_out_shaping.Orientation.HORIZONTAL
      else fan_out_shaping.Orientation.HORIZONTAL
  )

  if fold_type is None:
    # One polarity at a time: moving a flap's mountain pleats (or its valley
    # pleats) fans the accordion, moving both the same way only skews it.
    fold_type_sets = [[crease_utils.Fold.MOUNTAIN], [crease_utils.Fold.VALLEY]]
  else:
    key = fold_type.lower()
    if key not in _FOLD_TYPES:
      raise ValueError(
          f"fold_type must be 'mountain' or 'valley', got {fold_type!r}."
      )
    fold_type_sets = [[_FOLD_TYPES[key]]]

  rects_to_try = [('flap', rectangle)]
  chain_rect = _chain_rectangle(state, element_name)
  if chain_rect is not None:
    rects_to_try.append(('chain', chain_rect))
  deltas_to_try = _fan_deltas(state, move_delta)

  saw_any_candidates = False
  start_time = time.perf_counter()
  weak: tuple[_FanResult, str] | None = None

  for ori_try in (req_ori, other_ori):
    if _over_budget(start_time):
      break
    for scope_name, rect_try in rects_to_try:
      if _over_budget(start_time):
        break
      for delta_try in deltas_to_try:
        if _over_budget(start_time):
          break
        best, saw_pleats = _best_fan(
            state,
            element,
            element_name,
            rect_try,
            [ori_try],
            fold_type_sets,
            delta_try,
            is_left,
            open_percent,
            layout,
            w0_dict,
            start_time,
        )
        saw_any_candidates = saw_any_candidates or saw_pleats
        if best is None:
          continue
        notes = _fan_notes(best)
        if ori_try != req_ori:
          notes.append(f'orientation={ori_try.value}')
        if fold_type is None:
          notes.append(best.label.split('/')[-1])
        if not math.isclose(delta_try, move_delta):
          notes.append(f'move_delta={delta_try:g}')
        if scope_name == 'chain':
          notes.append(f'chain={rect_try.name}')
        note = f'Fanned {element_name}: {", ".join(notes)}.'
        if best.weak:
          if weak is None:
            weak = (best, note)
          continue
        _record_fan_report(best, open_percent, start_time, note)
        return best.state

  if weak is not None:
    best, note = weak
    _record_fan_report(best, open_percent, start_time, note)
    return best.state
  if not saw_any_candidates:
    raise ValueError(f'No valid fan out creases found for {element_name}.')
  raise ValueError(
      f'Fanned out crease pattern for {element_name} is not flat-foldable.'
  )


def fan_out_fold_heuristic_impl(
    state: Any,
    *,
    element_name: str,
    move_delta: float = 1.0,
    open_percent: float = 0.5,
) -> Any:
  """Fans out a flap by automatically picking fold type and orientation.

  Args:
    state: Current `ShapingState`.
    element_name: Flap name.
    move_delta: Grid units to shift border points.
    open_percent: `fold_percent` in `(0, 1]` for newly created creases (default
      0.5 for 3-D flare; 1.0 for flat).

  Returns:
    Updated `ShapingState`.
  """
  global LAST_REPORT
  LAST_REPORT = FanSquashReport(
      tool='fan_out_fold_heuristic', element_name=element_name
  )

  if not 0.0 < open_percent <= 1.0:
    raise ValueError(f'open_percent must be in (0, 1], got {open_percent}.')
  if math.isclose(move_delta, 0.0):
    raise ValueError('move_delta must be non-zero.')

  rectangle = _flap_rectangle(state, element_name)
  element = state.element(element_name)
  is_left = (element.x_start + element.x_len / 2.0) < 0.5
  counterpart = _bilateral_counterpart(element_name)
  if is_left and counterpart is not None and counterpart in state.tree:
    try:
      mirrored_state = state.replace(
          pattern=_mirror_crease_pattern_x(state.pattern)
      )
      res_m = fan_out_fold_heuristic_impl(
          mirrored_state,
          element_name=counterpart,
          move_delta=move_delta,
          open_percent=open_percent,
      )
      result_pattern = _mirror_crease_pattern_x(res_m.pattern)
      result_state = state.replace(pattern=result_pattern)
      ok = True
      if state.config.check_flat_foldability:
        ok, _ = folds(result_state.pattern)
      if ok:
        LAST_REPORT.element_name = element_name
        LAST_REPORT.note = LAST_REPORT.note.replace(counterpart, element_name)
        return result_state
    except ValueError:
      pass
    LAST_REPORT = FanSquashReport(
        tool='fan_out_fold_heuristic', element_name=element_name
    )

  layout, w0_dict = _get_base_width_dict(state, element_name)
  rects_to_try = [('flap', rectangle)]
  chain_rect = _chain_rectangle(state, element_name)
  if chain_rect is not None:
    rects_to_try.append(('chain', chain_rect))
  deltas_to_try = _fan_deltas(state, move_delta)
  fold_type_sets = [[crease_utils.Fold.MOUNTAIN], [crease_utils.Fold.VALLEY]]
  orientations = [
      fan_out_shaping.Orientation.HORIZONTAL,
      fan_out_shaping.Orientation.VERTICAL,
  ]

  saw_any_candidates = False
  start_time = time.perf_counter()
  weak: tuple[_FanResult, str] | None = None

  for scope_name, rect_try in rects_to_try:
    if _over_budget(start_time):
      break
    for delta_try in deltas_to_try:
      if _over_budget(start_time):
        break
      best, saw_pleats = _best_fan(
          state,
          element,
          element_name,
          rect_try,
          orientations,
          fold_type_sets,
          delta_try,
          is_left,
          open_percent,
          layout,
          w0_dict,
          start_time,
      )
      saw_any_candidates = saw_any_candidates or saw_pleats
      if best is None:
        continue
      notes = [best.label] + _fan_notes(best)
      if not math.isclose(delta_try, move_delta):
        notes.append(f'move_delta={delta_try:g}')
      if scope_name == 'chain':
        notes.append(f'chain={rect_try.name}')
      note = f'Selected {", ".join(notes)} on {element_name}.'
      if best.weak:
        if weak is None:
          weak = (best, note)
        continue
      _record_fan_report(best, open_percent, start_time, note)
      return best.state

  if weak is not None:
    best, note = weak
    _record_fan_report(best, open_percent, start_time, note)
    return best.state
  if not saw_any_candidates:
    raise ValueError(
        f'No valid fan out creases found for any combination on {element_name}.'
    )
  raise ValueError(
      'No fold type/orientation combination produced a flat-foldable result'
      f' for {element_name}.'
  )


# ------------------------------------------------------------------------------
# Squash-fold implementation.
# ------------------------------------------------------------------------------

# Wall-clock budget of the flap-flare search of `squash_fold` (each candidate
# runs the folder and the fidelity measurement, ~0.1-0.3 s).
FLARE_TIME_BUDGET_S = 6.0
# Target band of the flap-flare width ratio (a broad plate, not an unrolled
# sheet).
FLARE_MIN_RATIO = 1.5
FLARE_MAX_RATIO = 3.5


def _section_widths(
    tree: Any, layout: Any, pattern: crease_pattern_lib.CreasePattern
) -> dict[str, float] | None:
  """Folded section widths (`name@frac` -> width), or None if it won't fold."""
  try:
    from . import fidelity  # pylint: disable=g-import-not-at-top

    model = fidelity.FoldedModel.from_pattern(pattern)
    if getattr(model, 'vertex_error', 0.0) > MAX_RENDERABLE_VERTEX_ERROR:
      return None
    geometry = fidelity.measure(tree, layout, model)
  except Exception:  # pylint: disable=broad-exception-caught
    return None
  return {
      name: float(section.get('width', 0.0) or 0.0)
      for name, section in geometry['sections'].items()
  }


def _element_ratio(
    widths: dict[str, float], base: dict[str, float], element_name: str
) -> float:
  """Mean width ratio over the element's sections (1.0 if unmeasurable)."""
  ratios = []
  for frac in (0.25, 0.5, 0.75):
    key = f'{element_name}@{frac}'
    if base.get(key, 0.0) > 0.0:
      ratios.append(widths.get(key, 0.0) / base[key])
  return sum(ratios) / len(ratios) if ratios else 1.0


def _flare_line_groups(
    pattern: crease_pattern_lib.CreasePattern,
    layout: Any,
    element_names: Sequence[str],
) -> list[tuple[str, set[tuple[float, float, float]]]]:
  """Pleat-line groups crossing the paper sections of `element_names`.

  A stick's folded width is the spread of its paper cross-section (a contour
  around the flap tip, `fidelity.paper_layout`); the pleat lines that cross
  that contour are the ones whose opening widens the stick. Lines are taken
  whole (every collinear segment) so each crossed vertex opens consistently.
  Groups: per direction, lines crossing only these sticks' sections
  (`exclusive`) first, then all crossing lines.

  Args:
    pattern: The crease pattern.
    layout: `fidelity.paper_layout`.
    element_names: The target sticks.

  Returns:
    `(label, line keys)` groups, most local first.
  """
  import numpy as np  # pylint: disable=g-import-not-at-top

  tol = 0.55 / float(layout.grid_size)
  targets = set(element_names)
  own: list[np.ndarray] = []
  other: list[np.ndarray] = []
  for key, points in layout.sections.items():
    arr = np.asarray(points, dtype=float).reshape(-1, 2)
    if not arr.size:
      continue
    (own if key.split('@')[0] in targets else other).append(arr)
  if not own:
    return []
  own_pts = np.concatenate(own)
  other_pts = np.concatenate(other) if other else np.zeros((0, 2))

  def _hits(c: crease_utils.Crease, pts: np.ndarray) -> bool:
    if not len(pts):
      return False
    a = np.array([c.p1_init.x, c.p1_init.y])
    b = np.array([c.p2_init.x, c.p2_init.y])
    ab = b - a
    denom = float(ab @ ab)
    if denom < 1e-12:
      return False
    t = np.clip(((pts - a) @ ab) / denom, 0.0, 1.0)
    d = np.linalg.norm(pts - (a + t[:, None] * ab), axis=1)
    return bool((d <= tol).any())

  own_lines: dict[tuple[float, float, float], bool] = {}
  for c in pattern.creases:
    if c.fold == crease_utils.Fold.BORDER:
      continue
    line = _line_key(c)
    if _hits(c, own_pts):
      own_lines.setdefault(line, True)
  shared: set[tuple[float, float, float]] = set()
  for c in pattern.creases:
    if c.fold == crease_utils.Fold.BORDER:
      continue
    line = _line_key(c)
    if line in own_lines and line not in shared and _hits(c, other_pts):
      shared.add(line)

  by_dir: dict[tuple[float, float], set[tuple[float, float, float]]] = {}
  for line in own_lines:
    by_dir.setdefault(line[:2], set()).add(line)

  def _is_axial(direction: tuple[float, float]) -> int:
    return 0 if (abs(direction[0]) < 0.1 or abs(direction[1]) < 0.1) else 1

  groups: list[tuple[str, set[tuple[float, float, float]]]] = []
  ordered_dirs = sorted(by_dir, key=lambda d: (_is_axial(d), -len(by_dir[d])))
  for direction in ordered_dirs:
    exclusive = by_dir[direction] - shared
    if exclusive:
      groups.append((f'exclusive{direction}', exclusive))
  # Contiguous windows of parallel lines (axial first, smallest first, both
  # ends): lines shared with other sticks widen those too, so prefer the
  # fewest lines that reach the target width.
  windows: list[
      tuple[tuple[int, int], str, set[tuple[float, float, float]]]
  ] = []
  for direction in ordered_dirs:
    lines = sorted(by_dir[direction], key=lambda line: line[2])
    n = len(lines)
    for frac in (1.0 / 3.0, 0.5, 2.0 / 3.0, 1.0):
      size = max(1, int(round(n * frac)))
      starts = {0, n - size}
      if frac < 0.4:
        starts.add((n - size) // 2)
      for begin in sorted(starts):
        window = set(lines[begin : begin + size])
        windows.append((
            (_is_axial(direction), size),
            f'lines{direction}[{begin}:{begin + size}]/{n}',
            window,
        ))
  windows.sort(key=lambda item: item[0])
  seen = [g[1] for g in groups]
  for _, label, window in windows:
    if window not in seen:
      groups.append((label, window))
      seen.append(window)
  return groups


def _flare_flap_pleats(
    state: Any,
    base_pattern: crease_pattern_lib.CreasePattern,
    element_name: str,
    open_percent: float,
    budget_s: float = 0.0,
    max_folds: int = 8,
) -> tuple[crease_pattern_lib.CreasePattern, float, str] | None:
  """Opens the pleat lines crossing a stick's sections so it flares wide.

  Tries the groups of `_flare_line_groups` at a few `fold_percent` values on
  top of `base_pattern` (typically the squashed pattern, whose kite is the
  gusset), keeps candidates that fold within `MAX_RENDERABLE_VERTEX_ERROR`,
  and picks the widest one for the target, penalizing width spilled onto
  other sticks. The search is bounded by a fixed number of folds (not by
  wall-clock time) so that mirrored left/right calls pick the same lines.

  Args:
    state: The state before the squash (tree, packing, base widths).
    base_pattern: Pattern to open the lines on.
    element_name: Target stick.
    open_percent: Requested `fold_percent`.
    budget_s: Wall-clock safety limit (default `FLARE_TIME_BUDGET_S`).
    max_folds: Maximum number of candidate folds to evaluate.

  Returns:
    `(pattern, width ratio of the target vs. state, note)` or None.
  """
  try:
    from . import fidelity  # pylint: disable=g-import-not-at-top

    layout = fidelity.paper_layout(state.tree, state.packing)
  except Exception:  # pylint: disable=broad-exception-caught
    return None
  base = _section_widths(state.tree, layout, state.pattern)
  if not base:
    return None
  groups = _flare_line_groups(base_pattern, layout, [element_name])
  if not groups:
    return None
  pcts = []
  for pct in (open_percent if open_percent < 1.0 else 0.5, 0.6, 0.75):
    if pct not in pcts:
      pcts.append(pct)
  budget_s = max(budget_s or FLARE_TIME_BUDGET_S, 4.0 * FLARE_TIME_BUDGET_S)
  start = time.perf_counter()
  folds_done = 0
  # (rank, pattern, ratio, note); smaller rank is better. Groups come
  # smallest first, so the first candidate inside the target band
  # [FLARE_MIN_RATIO, FLARE_MAX_RATIO] opens the fewest lines: stop there.
  best: tuple[float, crease_pattern_lib.CreasePattern, float, str] | None = None
  for label, lines in groups:
    if folds_done >= max_folds or time.perf_counter() - start > budget_s:
      break
    keys = {
        _crease_key(c)
        for c in base_pattern.creases
        if c.fold != crease_utils.Fold.BORDER and _line_key(c) in lines
    }
    in_band = False
    for pct in pcts:
      if folds_done >= max_folds or time.perf_counter() - start > budget_s:
        break
      candidate = _with_fold_percent(base_pattern, keys, pct)
      if not candidate.local_flat_foldability():
        continue
      folds_done += 1
      widths = _section_widths(state.tree, layout, candidate)
      if widths is None:
        continue
      ratio = _element_ratio(widths, base, element_name)
      spill = max(
          (
              _element_ratio(widths, base, name)
              for name in {k.split('@')[0] for k in base}
              if name != element_name
          ),
          default=1.0,
      )
      if spill > ratio + 0.05:
        continue  # Widens other sticks more than the target.
      in_band = FLARE_MIN_RATIO <= ratio <= FLARE_MAX_RATIO
      if in_band:
        rank = -100.0 + 0.1 * max(0.0, spill - 1.0)
      elif ratio > FLARE_MAX_RATIO:
        rank = ratio  # Over-opened (unrolls the flap): last resort.
      else:
        rank = -ratio
      if best is None or rank < best[0]:
        best = (
            rank,
            candidate,
            ratio,
            (
                f'flared {len(lines)} pleat lines ({label}, open={pct:g},'
                f' other sticks up to {spill:.2f}x)'
            ),
        )
      if in_band:
        break
    if in_band:
      break
  if best is None:
    return None
  return best[1], best[2], best[3]


def _squash_ridges(
    element: shaping_elements.ShapingFlap,
) -> list[crease_utils.Crease]:
  """Center-to-corner ridges of a flap that a squash can open, sorted in mirror-canonical order."""
  centers = element.circle_centers
  if len(centers) == 1:
    raw_ridges = [
        crease_utils.Crease(centers[0], corner, fold=crease_utils.Fold.MOUNTAIN)
        for corner in element.corners
    ]
  else:
    central = crease_utils.Crease(
        centers[0], centers[1], fold=crease_utils.Fold.MOUNTAIN
    )
    raw_ridges = [
        r for r in element.get_ridges() if not r.is_collinear(central)
    ]

  # Orient each ridge directed from circle center to corner/tip:
  directed_ridges = []
  for r in raw_ridges:
    p1, p2 = r.p1, r.p2
    if any(p2.isclose(c) for c in centers) and not any(
        p1.isclose(c) for c in centers
    ):
      directed_ridges.append(
          crease_utils.Crease(p2, p1, fold=r.fold, elevation=r.elevation)
      )
    else:
      directed_ridges.append(r)

  is_left = (element.x_start + element.x_len / 2.0) < 0.5

  def _canon_ridge_key(
      r: crease_utils.Crease,
  ) -> tuple[float, float, float, float]:
    p1_x = 1.0 - r.p1.x if is_left else r.p1.x
    p2_x = 1.0 - r.p2.x if is_left else r.p2.x
    return (round(p1_x, 5), round(r.p1.y, 5), round(p2_x, 5), round(r.p2.y, 5))

  directed_ridges.sort(key=_canon_ridge_key)
  return directed_ridges


def _ridge_segment_counts(
    pattern: crease_pattern_lib.CreasePattern,
    element: shaping_elements.ShapingFlap,
) -> list[int]:
  """Number of crease segments on each squash ridge of `element`."""
  rectangles = [element.to_legacy_rectangle()]
  counts = []
  for ridge in _squash_ridges(element):
    try:
      start_side = squash_sink_shaping.get_start_side_of_ridge(
          ridge, ridge, rectangles
      )
      counts.append(
          len(
              squash_sink_shaping.get_oriented_ordered_crease_segments_on_ridge(
                  pattern, ridge, start_side
              )
          )
      )
    except (ValueError, AssertionError):
      counts.append(-1)
  return counts


def _mirror_ridge_pairs(
    ridges: Sequence[crease_utils.Crease],
    indices: Sequence[int],
    has_segments: Any,
    axis_x: float | None = 0.5,
    axis_y: float | None = None,
) -> list[tuple[int, int]]:
  """Pairs `(i, j)` of ridges that mirror each other across a symmetry axis.

  Only ridges with >= 2 crease segments (a squashable pleat pair) are paired;
  pairs keep the order of `indices` (most segments first).

  Args:
    ridges: Directed squash ridges (`_squash_ridges`).
    indices: Candidate ridge indices in preference order.
    has_segments: `index -> (squashable, segment count)`.
    axis_x: Mirror across the vertical line `x = axis_x` (the paper axis by
      default); None to skip.
    axis_y: Mirror across the horizontal line `y = axis_y`; None to skip.

  Returns:
    The mirror pairs.
  """

  def _mirrored(a: crease_utils.Crease, b: crease_utils.Crease) -> bool:
    def _close(p: Any, q: Any) -> bool:
      if axis_x is not None:
        return math.isclose(
            p.x, 2.0 * axis_x - q.x, abs_tol=1e-4
        ) and math.isclose(p.y, q.y, abs_tol=1e-4)
      assert axis_y is not None
      return math.isclose(p.x, q.x, abs_tol=1e-4) and math.isclose(
          p.y, 2.0 * axis_y - q.y, abs_tol=1e-4
      )

    return (_close(a.p1, b.p1) and _close(a.p2, b.p2)) or (
        _close(a.p1, b.p2) and _close(a.p2, b.p1)
    )

  if axis_x is None and axis_y is None:
    return []
  usable = [i for i in indices if has_segments(i)[1] >= 2]
  pairs = []
  used: set[int] = set()
  for i in usable:
    if i in used:
      continue
    for j in usable:
      if j != i and j not in used and _mirrored(ridges[i], ridges[j]):
        pairs.append((i, j))
        used.update((i, j))
        break
  return pairs


def _flap_axes(
    element: shaping_elements.ShapingFlap,
) -> list[tuple[float | None, float | None]]:
  """Own symmetry axes of a flap's rectangle, `(axis_x, axis_y)` per axis.

  A box-pleated flap is mirror-symmetric about the line through its circle
  centers (its length axis), so the two ridges that leave a circle center
  toward the two corners at the same end mirror each other across it. For a
  square (single-center) flap both axes qualify; the length axis comes first
  for elongated flaps.

  Args:
    element: The flap.

  Returns:
    Axes to try in order.
  """
  x_mid = element.x_start + element.x_len / 2.0
  y_mid = element.y_start + element.y_len / 2.0
  centers = element.circle_centers
  if len(centers) >= 2:
    if math.isclose(centers[0].x, centers[1].x, abs_tol=1e-6):
      return [(centers[0].x, None), (None, y_mid)]
    if math.isclose(centers[0].y, centers[1].y, abs_tol=1e-6):
      return [(None, centers[0].y), (x_mid, None)]
  if element.y_len >= element.x_len:
    return [(x_mid, None), (None, y_mid)]
  return [(None, y_mid), (x_mid, None)]


def _try_squash_single_ridge(
    state: Any,
    element: shaping_elements.ShapingFlap,
    rectangles: list[shaping_utils.Rectangle],
    ridges: Sequence[crease_utils.Crease],
    index: int,
    starting_position_index: int | None,
    ending_position_index: int | None,
    open_percent: float,
    layout: Any = None,
    w0_dict: dict[float, float] | None = None,
    min_ratio: float = 1.15,
) -> tuple[Any, float]:
  """Attempts to squash `ridges[index]`, raising a descriptive `ValueError` on failure.

  Args:
    state: Current state.
    element: The flap.
    rectangles: The flap's legacy rectangle (for the ridge geometry).
    ridges: Directed squash ridges of the flap.
    index: Ridge to squash.
    starting_position_index: Optional first segment of the span.
    ending_position_index: Optional end segment (exclusive) of the span.
    open_percent: `fold_percent` of the opened creases (1.0 = flat kite).
    layout: `fidelity.paper_layout` of the state (width measurement).
    w0_dict: Base section widths of the element (width measurement).
    min_ratio: Width ratio at which the 3-D opening search stops early (a lower
      value for the first ridge of a bilateral pair, whose gain is only half of
      the final one).

  Returns:
    `(new state, width ratio)`.
  """
  ridge = ridges[index]
  start_side = squash_sink_shaping.get_start_side_of_ridge(
      ridge, ridge, rectangles
  )
  segments = squash_sink_shaping.get_oriented_ordered_crease_segments_on_ridge(
      state.pattern, ridge, start_side
  )

  can_border_triangle = (
      len(segments) >= 1
      and segments[0].p1_init.is_border
      and segments[0].p2_init.is_internal
      and starting_position_index is None
      and ending_position_index is None
  )
  if len(segments) < 2 and not can_border_triangle:
    raise ValueError(
        f'No shaping creases found for squash of {element.name}: ridge {index}'
        f' has {len(segments)} crease segment(s), needs >= 2 (a pleat pair'
        ' crossing the ridge). Segments per ridge:'
        f' {_ridge_segment_counts(state.pattern, element)}. Try another'
        ' ridge_index, or fold the flap first (simple_fold/crimp_fold'
        ' zig-zags cross its ridges).'
    )

  if starting_position_index is None and ending_position_index is None:
    span_candidates: list[tuple[int, int]] = []
    if len(segments) >= 4:
      # Multi-pair spans first: a squash over several pleat pairs of the
      # ridge opens a longer stretch of the flap than one pair (k, k + 2).
      # Try strictly-interior `(1, len(segments) - 1)` first so intermediate
      # chain flaps keep a 1-segment hinge buffer at both ends for rotation_3d.
      for span in (
          (1, len(segments) - 1),
          (1, len(segments)),
          (0, len(segments)),
      ):
        if span[1] - span[0] >= 3 and span not in span_candidates:
          span_candidates.append(span)
    span_candidates.extend((k, k + 2) for k in range(len(segments) - 1))
    if can_border_triangle:
      span_candidates.append((0, 1))
  else:
    if ending_position_index is None:
      start = int(starting_position_index or 0)
      end = start + 2
    elif starting_position_index is None:
      end = int(ending_position_index)
      start = max(0, end - 2)
    else:
      start, end = int(starting_position_index), int(ending_position_index)
    end = min(len(segments), end)
    start = max(0, start)
    if start >= end:
      raise ValueError(
          f'starting_position_index {start} must be below'
          f' ending_position_index {end} (ridge has {len(segments)} segments).'
      )
    span_candidates = [(start, end)]

  last_error: ValueError | None = None
  for start, end in span_candidates:
    computed_ridge = crease_utils.Crease(
        segments[start].p1_init,
        segments[end - 1].p2_init,
        fold=ridge.fold,
        elevation=ridge.elevation,
    )
    try:
      shaped = squash_sink_shaping.shape_cp_with_squash_sink(
          cp=state.pattern, rectangles=rectangles, ridge=computed_ridge
      )
      shaped = _preserve_base_borders(state.pattern, shaped)
    except (ValueError, AssertionError) as error:
      # E.g. a kite corner landing within tolerance of an existing vertex
      # ("tips are too close"): try the next span.
      last_error = ValueError(f'span {start}..{end}: {error}')
      continue
    if _new_crease_count(state.pattern, shaped) == 0:
      continue
    if not shaped.local_flat_foldability():
      continue

    if open_percent < 1.0:
      # Candidates in preference order: the requested opening with the two
      # strategies that widen the flap (the kite plus its continuation
      # pleats, then the kite's axial/cross creases), then the kite +
      # pleats at a more closed and a more open setting, then every new
      # crease. Each candidate costs one deterministic fold (~1-2 s), so the
      # search stops at the first clean candidate that visibly widens the
      # flap.
      candidates = [
          (open_percent, 'kite_and_flap_pleats'),
          (open_percent, 'axial_and_cross'),
      ]
      if open_percent < 0.55:
        candidates.append(
            (min(0.7, open_percent + 0.2), 'kite_and_flap_pleats')
        )
      if open_percent > 0.35:
        candidates.append(
            (max(0.3, open_percent - 0.15), 'kite_and_flap_pleats')
        )
      candidates.append((open_percent, 'all_new'))
    else:
      candidates = []

    best_3d_state: Any = None
    best_3d_ratio: float = 0.0
    best_3d_clean: bool = False
    best_3d_pct: float = open_percent
    best_3d_strat: str = ''

    for pct_try, strat in candidates:
      target_cp = _open_squash_creases(
          state.pattern,
          shaped,
          computed_ridge,
          strat,
          pct_try,
          rectangle=rectangles[0],
      )
      try:
        candidate_state = state.with_pattern(target_cp)
      except ValueError as error:
        last_error = error
        continue

      # One deterministic fold per candidate serves the fold check, the
      # clean gate and the width measurement (it used to fold three times).
      cand_clean = True
      model = None
      if state.config.check_flat_foldability:
        ok, reason, model = _fold_once(candidate_state.pattern)
        if not ok:
          last_error = ValueError(reason)
          continue
        cand_clean = model.vertex_error <= fidelity.CLEAN_GATE_MAX_VERTEX_ERROR

      w_ratio = _measure_candidate_width_ratio(
          state.tree,
          layout,
          w0_dict,
          candidate_state.pattern,
          element.name,
          model=model,
      )
      better = best_3d_state is None or (
          (cand_clean, w_ratio) > (best_3d_clean, best_3d_ratio)
      )
      if better:
        best_3d_ratio = w_ratio
        best_3d_clean = cand_clean
        best_3d_state = candidate_state
        best_3d_pct = pct_try
        best_3d_strat = strat
      # A clean opening that visibly widens the flap is good enough.
      if cand_clean and w_ratio >= min_ratio:
        break

    if best_3d_state is not None:
      if not math.isclose(best_3d_pct, open_percent):
        LAST_REPORT.note = (
            f'open_percent={open_percent:g} adjusted to {best_3d_pct:g} on'
            f' {element.name} ({best_3d_strat}) for 3-D foldability.'
        )
      return best_3d_state, best_3d_ratio

    # Flat fallback when open_percent == 1.0 or all 3-D attempts failed:
    target_cp = shaped
    try:
      candidate_state = state.with_pattern(target_cp)
      flat_model = None
      if state.config.check_flat_foldability:
        ok, reason, flat_model = _fold_once(candidate_state.pattern)
        if not ok:
          last_error = ValueError(reason)
        else:
          if open_percent < 1.0:
            LAST_REPORT.note = (
                f'open_percent={open_percent:g} adjusted to 1.0 (flat) on'
                f' {element.name} for foldability.'
            )
          else:
            LAST_REPORT.note = (
                f'Flat kite squash on {element.name} (segments {start}..{end}'
                f' of ridge {index}): the folded width is unchanged by design;'
                ' use open_percent < 1 for a 3-D flare, or fan_out_fold /'
                ' open_flap to widen.'
            )
          w_ratio = _measure_candidate_width_ratio(
              state.tree,
              layout,
              w0_dict,
              candidate_state.pattern,
              element.name,
              model=flat_model,
          )
          return candidate_state, w_ratio
    except ValueError as error:
      last_error = error

  reasons = set()
  start0, end0 = span_candidates[0]
  for first, second in zip(segments[start0:end0], segments[start0 + 1 : end0]):
    try:
      squash_sink_shaping.apply_squash_sink_to_segment(
          first, second, state.pattern, segments
      )
    except ValueError as error:
      reasons.add(str(error))
  if last_error is not None:
    reasons.add(str(last_error))
  hint = ''
  if any(
      seg.p2_init.isclose(center)
      for seg in segments[start0:end0]
      for center in element.circle_centers
  ):
    hint = (
        ' The last segment ends at the flap tip (circle center), which the'
        ' squash sink cannot use: it needs a pair of plain pleat vertices'
        ' before the tip, i.e. >= 3 segments on the ridge (a width-2 flap'
        ' on a raw base has 2). Try fan_out_fold_heuristic (move_delta=+1'
        ' or -1) to broaden a width-2 flap, or a wider stick.'
    )
  raise ValueError(
      f'No shaping creases found for squash of {element.name} (ridge'
      f' {index}, segments {start0}..{end0} of {len(segments)}):'
      f' {"; ".join(sorted(reasons)) or "no segment pair"}. Segments per'
      f' ridge: {_ridge_segment_counts(state.pattern, element)}.{hint}'
  )


def squash_fold_impl(
    state: Any,
    *,
    element_name: str,
    ridge_index: int | None = None,
    starting_position_index: int | None = None,
    ending_position_index: int | None = None,
    open_percent: float = 0.45,
) -> Any:
  """Squashes a flap along a ridge, with on-paper ridge fallback and `open_percent`.

  Args:
    state: Current `ShapingState`.
    element_name: Flap name.
    ridge_index: Ridge index in `0..len(ridges)-1`, or `None` to try all.
    starting_position_index: Optional start segment index along the ridge.
    ending_position_index: Optional end segment index (exclusive).
    open_percent: `fold_percent` in `(0, 1]` for the new squash creases (default
      0.45 for 3-D flare; 1.0 for flat).

  Returns:
    Updated `ShapingState`.
  """
  global LAST_REPORT
  LAST_REPORT = FanSquashReport(tool='squash_fold', element_name=element_name)

  if not 0.0 < open_percent <= 1.0:
    raise ValueError(f'open_percent must be in (0, 1], got {open_percent}.')
  element = state.element(element_name)
  if not isinstance(element, shaping_elements.ShapingFlap):
    raise ValueError(f'{element_name} is not a flap; squash needs a flap.')

  is_left = (element.x_start + element.x_len / 2.0) < 0.5
  counterpart = _bilateral_counterpart(element_name)
  if is_left and counterpart is not None and counterpart in state.tree:
    try:
      mirrored_state = state.replace(
          pattern=_mirror_crease_pattern_x(state.pattern)
      )
      res_m = squash_fold_impl(
          mirrored_state,
          element_name=counterpart,
          ridge_index=ridge_index,
          starting_position_index=starting_position_index,
          ending_position_index=ending_position_index,
          open_percent=open_percent,
      )
      result_pattern = _mirror_crease_pattern_x(res_m.pattern)
      result_state = state.replace(pattern=result_pattern)
      ok = True
      if state.config.check_flat_foldability:
        ok, _ = folds(result_state.pattern)
      if ok:
        LAST_REPORT.element_name = element_name
        LAST_REPORT.note = LAST_REPORT.note.replace(counterpart, element_name)
        return result_state
    except ValueError:
      pass
    LAST_REPORT = FanSquashReport(tool='squash_fold', element_name=element_name)

  rectangles = [element.to_legacy_rectangle()]
  ridges = _squash_ridges(element)
  if not ridges:
    raise ValueError(f'No ridges found for {element_name}.')

  def _ridge_has_segments(idx: int) -> tuple[bool, int]:
    r = ridges[idx]
    try:
      ss = squash_sink_shaping.get_start_side_of_ridge(r, r, rectangles)
      segs = squash_sink_shaping.get_oriented_ordered_crease_segments_on_ridge(
          state.pattern, r, ss
      )
      can_border_tri = (
          len(segs) >= 1
          and segs[0].p1_init.is_border
          and segs[0].p2_init.is_internal
          and starting_position_index is None
          and ending_position_index is None
      )
      return (len(segs) >= 2 or can_border_tri), len(segs)
    except Exception:  # pylint: disable=broad-exception-caught
      return False, 0

  if ridge_index is None:
    ridge_info = [(_ridge_has_segments(i), i) for i in range(len(ridges))]
    valid_indices = [i for (has_segs, _), i in ridge_info if has_segs]
    if valid_indices:
      valid_indices.sort(key=lambda i: _ridge_has_segments(i)[1], reverse=True)
      indices_to_try = valid_indices
    else:
      indices_to_try = list(range(len(ridges)))
    requested_index = None
  else:
    requested_index = int(ridge_index)
    if not 0 <= requested_index < len(ridges):
      raise ValueError(
          f'ridge_index {requested_index} out of range; {element_name} has'
          f' {len(ridges)} ridges.'
      )
    if starting_position_index is None and ending_position_index is None:
      indices_to_try = [requested_index] + [
          (requested_index + offset) % len(ridges)
          for offset in range(1, len(ridges))
          if _ridge_has_segments((requested_index + offset) % len(ridges))[0]
      ]
    else:
      indices_to_try = [requested_index]

  layout, w0_dict = _get_base_width_dict(state, element_name)
  start_time = time.perf_counter()
  first_error: ValueError | None = None
  best_result_state: Any = None
  best_width_ratio: float = 0.0
  best_try_idx: int = 0
  best_report_note: str = ''

  # Axis-centered flap (head, hood, tail on the symmetry line): squash the
  # mirror-image left/right ridges together so one call gives a symmetric
  # bilateral flare instead of a lopsided one-ridge squash. Off-axis flaps
  # (wings, legs, horns, a snout packed in a corner) get the same treatment
  # about their own length axis: the two ridges leaving a circle center
  # toward the corners at the same end mirror each other across it.
  automatic = (
      ridge_index is None
      and starting_position_index is None
      and ending_position_index is None
  )
  axis_centered = (
      abs(element.x_start + element.x_len / 2.0 - 0.5) < 1e-3 and automatic
  )
  bilateral = False
  if automatic:
    axes: list[tuple[float | None, float | None, str]] = []
    if axis_centered:
      axes.append((0.5, None, 'paper axis'))
    for axis_x, axis_y in _flap_axes(element):
      if axis_centered and axis_x is not None and abs(axis_x - 0.5) < 1e-3:
        continue
      axes.append((axis_x, axis_y, 'flap axis'))
    for axis_x, axis_y, axis_label in axes:
      pairs = _mirror_ridge_pairs(
          ridges, indices_to_try, _ridge_has_segments, axis_x, axis_y
      )
      for i, j in pairs:
        try:
          first_state, _ = _try_squash_single_ridge(
              state,
              element,
              rectangles,
              ridges,
              i,
              None,
              None,
              open_percent,
              layout=layout,
              w0_dict=w0_dict,
              min_ratio=1.05,
          )
          first_note = LAST_REPORT.note
          pair_state, pair_ratio = _try_squash_single_ridge(
              first_state,
              element,
              rectangles,
              ridges,
              j,
              None,
              None,
              open_percent,
              layout=layout,
              w0_dict=w0_dict,
          )
        except ValueError as error:
          if first_error is None:
            first_error = error
          continue
        best_result_state = pair_state
        best_width_ratio = pair_ratio
        best_try_idx = i
        best_report_note = ' '.join(
            n
            for n in (
                (
                    f'Squashed mirror ridges {i}+{j} together (bilateral,'
                    f' {axis_label}).'
                ),
                first_note,
                LAST_REPORT.note if LAST_REPORT.note != first_note else '',
            )
            if n
        )
        bilateral = True
        break
      if bilateral:
        break

  for idx, try_idx in enumerate(indices_to_try):
    if best_result_state is not None and bilateral:
      break
    try:
      candidate_result, w_ratio = _try_squash_single_ridge(
          state,
          element,
          rectangles,
          ridges,
          try_idx,
          starting_position_index,
          ending_position_index,
          open_percent,
          layout=layout,
          w0_dict=w0_dict,
      )
      if w_ratio > best_width_ratio or best_result_state is None:
        best_width_ratio = w_ratio
        best_result_state = candidate_result
        best_try_idx = try_idx
        best_report_note = LAST_REPORT.note
        has_segs, seg_cnt = _ridge_has_segments(try_idx)
        if requested_index is not None or w_ratio >= 1.4 or seg_cnt >= 2:
          break
    except ValueError as error:
      if first_error is None:
        first_error = error

  if best_result_state is not None:
    LAST_REPORT.note = best_report_note
    if requested_index is not None and best_try_idx != requested_index:
      LAST_REPORT.note = (
          f'ridge_index {requested_index} had no valid squash on'
          f' {element_name}; used ridge_index {best_try_idx}.'
      )
      logging.info('squash_fold %s: %s', element_name, LAST_REPORT.note)
    result_state = best_result_state
    if (
        open_percent < 1.0
        and best_width_ratio < 1.5
        and starting_position_index is None
        and ending_position_index is None
    ):
      flared = _flare_flap_pleats(
          state,
          result_state.pattern,
          element_name,
          open_percent,
          max_folds=6,
      )
      flare_note = 'Squash kite + {}.'
      if flared is None or flared[1] < FLARE_MIN_RATIO:
        # The kite and the opened lines can conflict at shared vertices; try
        # the lines alone on the unsquashed pattern.
        alone = _flare_flap_pleats(
            state,
            state.pattern,
            element_name,
            open_percent,
            max_folds=6,
        )
        if alone is not None and (flared is None or alone[1] > flared[1]):
          flared = alone
          flare_note = (
              'Kite + flare did not fold within'
              f' {MAX_RENDERABLE_VERTEX_ERROR}%; kept the flare only: {{}}.'
          )
      if flared is not None and flared[1] > best_width_ratio + 0.15:
        try:
          result_state = state.with_pattern(flared[0])
          best_width_ratio = round(flared[1], 3)
          LAST_REPORT.note = ' '.join(
              n for n in (LAST_REPORT.note, flare_note.format(flared[2])) if n
          )
        except ValueError:
          pass
    new_cnt = _new_crease_count(state.pattern, result_state.pattern)
    loc = _compute_flap_locality(state.pattern, result_state.pattern, element)
    elapsed_s = round(time.perf_counter() - start_time, 3)
    LAST_REPORT.num_new_creases = new_cnt
    LAST_REPORT.locality = loc
    LAST_REPORT.width_ratio = best_width_ratio
    LAST_REPORT.open_percent = open_percent
    LAST_REPORT.elapsed_s = elapsed_s
    return result_state

  # Flap-flare fallback: a ridge has a pleat pair to squash but no squash
  # folds, so open the pleat lines that cross the flap's sections (whole
  # lines, verified by the folder). Flaps with no pleat pair on any ridge
  # keep the "No shaping creases" error.
  if (
      requested_index is None
      and open_percent < 1.0
      and starting_position_index is None
      and ending_position_index is None
      and any(_ridge_has_segments(i)[1] >= 2 for i in range(len(ridges)))
  ):
    flared = _flare_flap_pleats(
        state, state.pattern, element_name, open_percent
    )
    if flared is not None and flared[1] >= 1.2:
      try:
        result_state = state.with_pattern(flared[0])
      except ValueError:
        result_state = None
      if result_state is not None:
        LAST_REPORT.num_new_creases = 0
        LAST_REPORT.locality = 1.0
        LAST_REPORT.width_ratio = round(flared[1], 3)
        LAST_REPORT.open_percent = open_percent
        LAST_REPORT.elapsed_s = round(time.perf_counter() - start_time, 3)
        LAST_REPORT.note = (
            f'No ridge squash folded on {element_name}; {flared[2]}.'
        )
        return result_state

  # Partial-open squash fallback:
  # When no 2-D candidate folds cleanly (common for margin-pinned / river-adjacent flaps),
  # partially open creases in the flap rectangle to achieve 3-D broad plate expansion.
  in_rect = [
      c
      for c in state.pattern.creases
      if rectangles[0].crease_is_within(c)
      and c.fold != crease_utils.Fold.BORDER
  ]
  if requested_index is None and in_rect:
    fallback_pct = open_percent if open_percent < 1.0 else 0.45
    ridge_creases = [
        c for c in in_rect if any(c.is_collinear(r) for r in ridges)
    ]
    non_ridge = [c for c in in_rect if c not in ridge_creases]
    candidate_subsets = []
    if non_ridge:
      candidate_subsets.append(('non_ridge_pleats', non_ridge))
    candidate_subsets.append(('all_in_rect', in_rect))
    if ridge_creases:
      candidate_subsets.append(('ridge_only', ridge_creases))

    best_fallback_state = None
    best_fallback_ratio = 0.0
    best_fallback_subset = ''

    for sub_name, subset in candidate_subsets:
      new_creases = []
      split_done = False
      for c in state.pattern.creases:
        if c in subset:
          if not split_done:
            mid = dataclasses.replace(
                c.p1_init,
                x=(c.p1_init.x + c.p2_init.x) / 2.0,
                y=(c.p1_init.y + c.p2_init.y) / 2.0,
            )
            c1 = crease_utils.Crease(
                c.p1,
                mid,
                fold=c.fold,
                fold_percent=fallback_pct,
                elevation=c.elevation,
            )
            c2 = crease_utils.Crease(
                mid,
                c.p2,
                fold=c.fold,
                fold_percent=fallback_pct,
                elevation=c.elevation,
            )
            new_creases.extend([c1, c2])
            split_done = True
          else:
            new_creases.append(
                crease_utils.Crease(
                    c.p1,
                    c.p2,
                    fold=c.fold,
                    fold_percent=fallback_pct,
                    elevation=c.elevation,
                )
            )
        else:
          new_creases.append(c)

      cand_cp = crease_pattern_lib.CreasePattern(
          creases=new_creases, light_mode=state.pattern.light_mode
      )
      cand_cp.tolerance = state.pattern.tolerance
      ok, _ = folds(cand_cp)
      if ok:
        w_ratio = _measure_candidate_width_ratio(
            state.tree, layout, w0_dict, cand_cp, element.name
        )
        if w_ratio > best_fallback_ratio or best_fallback_state is None:
          best_fallback_ratio = w_ratio
          best_fallback_state = state.replace(pattern=cand_cp)
          best_fallback_subset = sub_name
          if w_ratio >= 2.0:
            break

    # A fallback that does not visibly widen the flap is a no-op dressed as a
    # success (BullMoose head: +2 creases, 1.00x): report it as a failure so
    # the caller can pick another tool.
    if best_fallback_state is not None and best_fallback_ratio >= 1.1:
      elapsed_s = round(time.perf_counter() - start_time, 3)
      LAST_REPORT.num_new_creases = _new_crease_count(
          state.pattern, best_fallback_state.pattern
      )
      LAST_REPORT.locality = 1.0
      LAST_REPORT.width_ratio = best_fallback_ratio
      LAST_REPORT.open_percent = fallback_pct
      LAST_REPORT.elapsed_s = elapsed_s
      LAST_REPORT.note = (
          f'Applied partial-open squash on {element_name}'
          f' ({best_fallback_subset}, width={best_fallback_ratio:.2f}x).'
      )
      return best_fallback_state
    if best_fallback_state is not None:
      no_widen = (
          f'Partially opening the flap creases ({best_fallback_subset})'
          f' folded but did not widen {element_name}'
          f' ({best_fallback_ratio:.2f}x); not applied.'
      )
      first_error = ValueError(
          f'{first_error} {no_widen}' if first_error is not None else no_widen
      )

  if first_error is None:
    first_error = ValueError(
        f'No squashable ridge on {element_name}: segments per ridge'
        f' {_ridge_segment_counts(state.pattern, element)}.'
    )
  raise first_error


TOOLS: dict[str, Any] = {
    'fan_out_fold': fan_out_fold_impl,
    'fan_out_fold_heuristic': fan_out_fold_heuristic_impl,
    'squash_fold': squash_fold_impl,
}


def tool_names() -> Sequence[str]:
  return tuple(TOOLS)
