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

"""Phase 7 shaping tools: inscribed relief, clip templates, narrowing, curves.

Provides `inscribed_flap`, upgraded `clip_pattern_fold` (all 4 templates:
`fold`, `turn_and_narrow`, `bend_when_turn_and_narrow`, `pleat_to_hide`, plus
width-2 river narrowing and `narrow_n_times=3`), `pleat_to_hide`,
`simple_fold_narrow`, `curve_flap`, and `level_shifter`.

Like `shaping_fold3d`, this module takes and returns `shaping.ShapingState`
objects (`Any`) without importing `shaping`, so `shaping.py` can import and
register `TOOLS` without circular dependencies.
"""

from collections.abc import Sequence
import dataclasses
import math
from typing import Any

import numpy as np

from ..crease_pattern import crease as crease_utils
from ..crease_pattern import crease_pattern as crease_pattern_lib
from .. import utils_tree
from ..solving import hinges as hinge_lib
from . import level_shifter_utils as utils_ls
from ..crease_pattern.parser import svg as svg_parser
from ..stick_figures import angle_utils
from ..stick_figures import stick_figure as stick_figure_lib
from . import fidelity
from . import toolbox_fold3d as shaping_fold3d
from .clip_pattern import clip_pattern_shaping_adapter
from . import geometric_simple_fold as geometric_simple_fold_lib
from . import heuristics_utils
from . import inscribed_rectangle_shaping
from . import shaping_elements
from . import simple_fold_shaping
from . import types as shaping_types
from . import zigzag_shaping
from . import zigzag_shaping_utils

_CLIP_PATTERN_MARGIN = 0.1
_MIN_SIMPLE_FOLD_ANGLE_DEG = 8.0


@dataclasses.dataclass
class ExtrasReport:
  """Details of the most recent `shaping_extras` tool call."""

  tool: str = ''
  element_name: str = ''
  summary: str = ''
  num_new_creases: int = 0
  details: dict[str, Any] = dataclasses.field(default_factory=dict)


LAST_REPORT = ExtrasReport()


def last_report() -> ExtrasReport:
  """Returns the most recent ExtrasReport."""
  return LAST_REPORT


def last_report_message() -> str:
  """One-line summary of the last call (for `shaping._note`)."""
  if LAST_REPORT.summary:
    return LAST_REPORT.summary
  return f'{LAST_REPORT.tool}({LAST_REPORT.element_name}): no candidate folded.'


# ------------------------------------------------------------------------------
# Shared geometry / foldability helpers.
# ------------------------------------------------------------------------------


def _all_internal_vertices_3d_foldable(
    pattern: crease_pattern_lib.CreasePattern,
) -> bool:
  """Whether every internal vertex satisfies 3-D Belcastro-Hull foldability."""
  return all(v.is_foldable for v in pattern.vertices if v.internal)


def _accept_pattern(
    state: Any,
    pattern: crease_pattern_lib.CreasePattern,
    *,
    allow_3d_vertices: bool,
    verify_fold: bool,
) -> tuple[Any | None, str]:
  """Validates `pattern` (2-D FF or 3-D vertices, plus folder) and wraps it."""
  prior_backbone = getattr(state, 'flat_backbone', None)
  new_state = None
  try:
    new_state = state.with_pattern(pattern)
  except (ValueError, AssertionError) as error:
    if not allow_3d_vertices:
      return None, f'ff: {error}'
    if not _all_internal_vertices_3d_foldable(pattern):
      return None, f'3d_vertex: {error}'
    if prior_backbone is None and state.pattern.local_flat_foldability():
      prior_backbone = fidelity.as_flat_backbone(state.pattern)
    new_state = state.replace(pattern=pattern, flat_backbone=prior_backbone)

  if verify_fold:
    ok, reason = shaping_fold3d.folds(pattern)
    if not ok:
      return None, reason
    model = fidelity.FoldedModel.from_pattern(pattern)
    if model.vertex_error > 1.0:
      return None, f'vertex error {model.vertex_error:.2f}% > 1.0%'
  if fidelity.has_3d_overlay(pattern):
    if allow_3d_vertices:
      if prior_backbone is not None:
        new_state = new_state.replace(flat_backbone=prior_backbone)
      elif state.pattern.local_flat_foldability():
        new_state = new_state.replace(
            flat_backbone=fidelity.as_flat_backbone(state.pattern)
        )
    elif getattr(new_state, 'flat_backbone', None) is None:
      if prior_backbone is not None:
        new_state = new_state.replace(flat_backbone=prior_backbone)
      elif state.pattern.local_flat_foldability():
        new_state = new_state.replace(
            flat_backbone=fidelity.as_flat_backbone(state.pattern)
        )
  return new_state, 'ok'


def _fold_model(
    pattern: crease_pattern_lib.CreasePattern,
) -> fidelity.FoldedModel:
  """Folds `pattern` with the deterministic folder and returns `FoldedModel`."""
  return fidelity.fold_svg(svg_parser.to_svg(pattern, ndigits=8))


def _element_z_span_grid_units(
    model: fidelity.FoldedModel, element: Any, grid_size: int
) -> float | None:
  """Out-of-plane relief thickness (grid units) across `element`'s paper box."""
  if isinstance(element, shaping_elements.ShapingFlap):
    rects = [element.to_legacy_rectangle()]
  else:
    rects = list(element.covering_legacy_rectangles)
  pts = []
  for rect in rects:
    x0 = max(0.0, min(rect.x_start, rect.x_start + rect.x_len))
    x1 = min(1.0, max(rect.x_start, rect.x_start + rect.x_len))
    y0 = max(0.0, min(rect.y_start, rect.y_start + rect.y_len))
    y1 = min(1.0, max(rect.y_start, rect.y_start + rect.y_len))
    if x1 - x0 <= 2e-4 or y1 - y0 <= 2e-4:
      continue
    xs = np.linspace(x0 + 1e-4, x1 - 1e-4, 9)
    ys = np.linspace(y0 + 1e-4, y1 - 1e-4, 9)
    for x in xs:
      for y in ys:
        folded = model.locate_point((float(x), float(y)))
        if folded is not None:
          pts.append(folded)
  if len(pts) < 4:
    return None
  arr = np.asarray(pts, dtype=float)
  centered = arr - arr.mean(axis=0, keepdims=True)
  # Smallest principal spread is out-of-plane relief thickness.
  _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
  if singular_values.size < 3:
    return 0.0
  normal = vh[2]
  projections = centered @ normal
  return float((projections.max() - projections.min()) * max(grid_size, 1))


# ------------------------------------------------------------------------------
# 1. Inscribed rectangle / river shaping (`inscribed_flap`).
# ------------------------------------------------------------------------------

_INSCRIBED_FOLD_ORDERS: dict[
    str, tuple[crease_utils.Fold, crease_utils.Fold]
] = {
    'MV': (crease_utils.Fold.MOUNTAIN, crease_utils.Fold.VALLEY),
    'VM': (crease_utils.Fold.VALLEY, crease_utils.Fold.MOUNTAIN),
    'M': (crease_utils.Fold.MOUNTAIN, crease_utils.Fold.MOUNTAIN),
    'V': (crease_utils.Fold.VALLEY, crease_utils.Fold.VALLEY),
}


def inscribed_flap(
    state: Any,
    *,
    element_name: str,
    depths: Sequence[int] | int = (1,),
    fold_order: str = 'MV',
    fold_percent: float = 0.5,
    verify_fold: bool = True,
) -> Any:
  """Inscribes concentric stepped pleat rings on a flap or river for 3-D relief.

  Ideal for carapace dorsal plates, tergite ridges, shields, and embossed shells
  (`fold_percent` around 0.4-0.6 raises a stepped 3-D dome/terrace; `1.0` sinks
  flat concentric pleats).

  Args:
    state: Current `ShapingState`.
    element_name: Name of a flap or river with width / radius >= 2.
    depths: One integer depth or sequence of integer grid depths (`>= 1` and
      strictly below `grid_unit_radius` for flaps or `width` for rivers).
    fold_order: `'MV'`, `'VM'`, `'M'`, or `'V'`. Automatically retries the
      opposite order if the requested order fails to fold.
    fold_percent: Crease fold fraction in `(0, 1]`; `0.5` gives 90-degree
      stepped 3-D relief, `1.0` folds flat.
    verify_fold: Keep only candidates that fold in the deterministic folder.

  Returns:
    New `ShapingState`.
  """
  LAST_REPORT.tool = 'inscribed_flap'
  LAST_REPORT.element_name = element_name
  LAST_REPORT.summary = ''
  LAST_REPORT.details = {}

  if not 0.0 < fold_percent <= 1.0:
    raise ValueError(f'fold_percent must be in (0, 1], got {fold_percent}.')
  order_key = fold_order.upper()
  if order_key not in _INSCRIBED_FOLD_ORDERS:
    raise ValueError(
        f'fold_order must be one of {sorted(_INSCRIBED_FOLD_ORDERS)}, got'
        f' {fold_order!r}.'
    )
  depth_list = (
      [int(depths)] if isinstance(depths, int) else [int(d) for d in depths]
  )
  if not depth_list:
    raise ValueError('depths must be non-empty.')
  depth_list = sorted(set(depth_list))

  element = state.element(element_name)
  is_flap = isinstance(element, shaping_elements.ShapingFlap)
  max_depth = (
      int(element.grid_unit_radius) - 1
      if is_flap
      else int(round(element.width)) - 1
  )
  if max_depth < 1:
    kind = 'flap radius' if is_flap else 'river width'
    raise ValueError(
        f'{element_name} has {kind} < 2 grid units and cannot take an interior'
        ' inscribed step (needs width/radius >= 2).'
    )
  for d in depth_list:
    if not 1 <= d <= max_depth:
      raise ValueError(
          f'depth={d} out of range for {element_name}; valid integer depths:'
          f' 1..{max_depth}.'
      )

  order_candidates = [order_key]
  opposite = {'MV': 'VM', 'VM': 'MV', 'M': 'V', 'V': 'M'}[order_key]
  order_candidates.append(opposite)

  flaps, rivers = state.shaping_elements
  bank_candidates: list[shaping_types.BankIndex | None]
  if is_flap:
    bank_candidates = [None]
  else:
    primary_bank: shaping_types.BankIndex = element.longer_bank_index
    alt_bank: shaping_types.BankIndex = (
        'SECOND' if primary_bank == 'FIRST' else 'FIRST'
    )
    bank_candidates = [primary_bank, alt_bank]

  z_before = None
  if verify_fold:
    try:
      z_before = _element_z_span_grid_units(
          _fold_model(state.pattern), element, state.solving_grid_size
      )
    except Exception:  # pylint: disable=broad-exception-caught
      z_before = None

  allow_3d = not math.isclose(fold_percent, 1.0, abs_tol=1e-6)
  last_reason = 'no shaping creases generated'
  for try_order in order_candidates:
    folds_pair = _INSCRIBED_FOLD_ORDERS[try_order]
    for bank_idx in bank_candidates:
      if is_flap:
        inscribed_params: Sequence[Any] = [
            inscribed_rectangle_shaping.InscribedRectangleParams(
                depth=d,
                fold_order=folds_pair,
                fold_percent=float(fold_percent),
            )
            for d in depth_list
        ]
      else:
        assert bank_idx is not None
        inscribed_params = [
            inscribed_rectangle_shaping.InscribedRiverParams(
                depth=d,
                fold_order=folds_pair,
                fold_percent=float(fold_percent),
                bank_index=bank_idx,
            )
            for d in depth_list
        ]
      elem_param = inscribed_rectangle_shaping.InscribedShapingParamsForElement(
          element_name=element_name,
          inscribed_params=inscribed_params,
      )
      try:
        shaped = inscribed_rectangle_shaping.shape_cp_with_inscribed_elements(
            cp=state.pattern,
            grid_size=element.grid_size,
            rectangles=flaps,
            rivers=rivers,
            element_shaping_params=[elem_param],
            central_crease_removed=state.central_crease_removed,
            light_mode=not state.config.check_all_crease_intersections,
            raise_error_on_ff_violation=False,
        )
      except (ValueError, AssertionError, IndexError) as error:
        last_reason = str(error)
        continue
      delta_creases = len(shaped.creases) - len(state.pattern.creases)
      if delta_creases <= 0:
        last_reason = '0 new creases inside paper bounds'
        continue
      new_state, reason = _accept_pattern(
          state,
          shaped,
          allow_3d_vertices=allow_3d,
          verify_fold=verify_fold,
      )
      if new_state is None:
        last_reason = reason
        continue

      z_after = None
      if verify_fold:
        try:
          z_after = _element_z_span_grid_units(
              _fold_model(shaped), element, state.solving_grid_size
          )
        except Exception:  # pylint: disable=broad-exception-caught
          z_after = None

      relief_note = ''
      if z_before is not None and z_after is not None:
        relief_note = f', z-relief {z_before:.2f}->{z_after:.2f}g'
      LAST_REPORT.num_new_creases = delta_creases
      LAST_REPORT.details = {
          'depths': depth_list,
          'fold_order': try_order,
          'fold_percent': fold_percent,
          'z_before_grid': z_before,
          'z_after_grid': z_after,
      }
      LAST_REPORT.summary = (
          f'{element_name}: inscribed depths={depth_list} order={try_order}'
          f' fold_percent={fold_percent}; +{delta_creases}'
          f' creases{relief_note}.'
      )
      return new_state

  raise ValueError(
      f'inscribed_flap on {element_name} (depths={depth_list},'
      f' fold_order={fold_order}, fold_percent={fold_percent}) failed:'
      f' {last_reason}'
  )


# ------------------------------------------------------------------------------
# 2. Upgraded `clip_pattern_fold` (all 4 templates + width-2 river narrowing).
# ------------------------------------------------------------------------------


def _line_geom_key(line: Any) -> tuple[float, float, float, float]:
  """Undirected rounded paper-space segment key for duplicate cancellation."""
  p1 = (round(float(line.p1[0]), 6), round(float(line.p1[1]), 6))
  p2 = (round(float(line.p2[0]), 6), round(float(line.p2[1]), 6))
  return (*p1, *p2) if p1 <= p2 else (*p2, *p1)


def _cancel_duplicate_lines(lines: Sequence[Any]) -> list[Any]:
  """Cancels pairs of identical overlapping 180-deg cap lines (mod 2)."""
  counts: dict[tuple[float, float, float, float], int] = {}
  for line in lines:
    key = _line_geom_key(line)
    counts[key] = counts.get(key, 0) + 1
  return [line for line in lines if counts[_line_geom_key(line)] % 2 == 1]


def _dedup_lines(lines: Sequence[Any]) -> list[Any]:
  """Deduplicates geometric lines preserving order."""
  seen = set()
  out = []
  for line in lines:
    key = _line_geom_key(line)
    if key not in seen:
      seen.add(key)
      out.append(line)
  return out


def _dedup_floats(values: Sequence[float]) -> list[float]:
  out: list[float] = []
  for val in values:
    fval = float(val)
    if fval >= 0.0 and not any(
        math.isclose(fval, x, abs_tol=1e-6) for x in out
    ):
      out.append(fval)
  return out


def clip_pattern_fold(
    state: Any,
    *,
    element_name: str,
    step: float = 0.0,
    template_name: str = 'fold',
    bend_angle: float = 0.0,
    rotate_angle: float = 0.0,
    enable_narrow: bool = False,
    narrow_n_times: int = 2,
    narrow_symmetric: bool | None = None,
    pleat_distance: float = 0.5,
    length: float | None = None,
    global_refine: bool = False,
    verify_fold: bool = True,
) -> Any:
  """Draws a shaping template on every layer of a flap or river."""
  kwargs = dict(
      state=state,
      element_name=element_name,
      step=step,
      template_name=template_name,
      bend_angle=bend_angle,
      rotate_angle=rotate_angle,
      enable_narrow=enable_narrow,
      narrow_n_times=narrow_n_times,
      narrow_symmetric=narrow_symmetric,
      pleat_distance=pleat_distance,
      length=length,
      verify_fold=verify_fold,
  )
  if not global_refine:
    try:
      res = _clip_pattern_fold_impl(**kwargs, global_refine=False)
      if len(res.pattern.creases) > len(state.pattern.creases):
        return res
    except Exception:  # pylint: disable=broad-exception-caught
      pass
    return _clip_pattern_fold_impl(**kwargs, global_refine=True)
  return _clip_pattern_fold_impl(**kwargs, global_refine=True)


def _clip_pattern_fold_impl(
    state: Any,
    *,
    element_name: str,
    step: float = 0.0,
    template_name: str = 'fold',
    bend_angle: float = 0.0,
    rotate_angle: float = 0.0,
    enable_narrow: bool = False,
    narrow_n_times: int = 2,
    narrow_symmetric: bool | None = None,
    pleat_distance: float = 0.5,
    length: float | None = None,
    global_refine: bool = False,
    verify_fold: bool = True,
) -> Any:
  """Draws a shaping template on every layer of a flap or river.

  Supports all four templates in `clip_pattern_shaping_adapter.TemplateName`:
  - `'fold'`: bend, rotate, and/or narrow (`narrow_n_times=2` or `3`) on flaps
    and rivers (including width-2 rivers via shared-cap cancellation).
  - `'turn_and_narrow'`: 90-degree 3-D turn + 1/3 narrowing on flaps.
  - `'bend_when_turn_and_narrow'`: 3-D bend along a `turn_and_narrow` flap
    (works after a prior `turn_and_narrow` or auto-chains `turn_and_narrow` on
    an un-narrowed flap).
  - `'pleat_to_hide'`: element-scoped horizontal sink pleats to shorten/hide a
    segment of `length` grid units at spacing `pleat_distance`.

  Args:
    state: Current `ShapingState`.
    element_name: Flap or river name.
    step: Offset in grid units from the proximal end of `element_name`.
    template_name: `'fold'`, `'turn_and_narrow'`, `'bend_when_turn_and_narrow'`,
      or `'pleat_to_hide'`.
    bend_angle: 3-D bend angle in degrees (`fold`, `bend_when_turn_and_narrow`).
    rotate_angle: 2-D rotation angle in degrees (`fold`).
    enable_narrow: Narrow the element (`fold`).
    narrow_n_times: Width divisor (`>= 2`, e.g. `2` or `3`) when narrowing.
    narrow_symmetric: Optional override for symmetric vs asymmetric narrowing.
    pleat_distance: Spacing in grid units between pleats (`pleat_to_hide`).
    length: Span in grid units to pleat (`pleat_to_hide`); defaults to the
      remaining element length/width.
    global_refine: If True, disables localized flap scoping and performs global
      grid refinement across the full chain. Defaults to False (localized).
    verify_fold: Keep only candidates that fold in the deterministic folder.

  Returns:
    New `ShapingState`.
  """
  LAST_REPORT.tool = 'clip_pattern_fold'
  LAST_REPORT.element_name = element_name
  LAST_REPORT.summary = ''
  LAST_REPORT.details = {}

  if int(narrow_n_times) < 2:
    raise ValueError(f'narrow_n_times must be >= 2, got {narrow_n_times}.')
  if step < 0:
    raise ValueError(f'step must be >= 0, got {step}.')
  template = clip_pattern_shaping_adapter.TemplateName(template_name)

  flaps, rivers = state.shaping_elements
  river = next((r for r in rivers if r.name == element_name), None)
  flap_element = None
  if river is not None:
    rectangles = river.covering_rectangles
    target_rectangles = rectangles
    river_width: float | None = float(river.width)
    step_from_root = 0.0
  else:
    root_and_distance = state.flap_hierarchy.get_root_flap_and_distance(
        element_name
    )
    if root_and_distance is None:
      raise ValueError(
          f'element_name={element_name} not found; known elements:'
          f' {state.element_names()}'
      )
    root_name, distance = root_and_distance
    root_flap = next((f for f in flaps if f.name == root_name), None)
    if root_flap is None:
      raise ValueError(f'Root flap {root_name} of {element_name} not found.')
    flap_element = state.element(element_name)
    rectangles = [root_flap.to_legacy_rectangle()]
    target_rectangles = (
        rectangles
        if global_refine or flap_element is None
        else [flap_element.to_legacy_rectangle()]
    )
    river_width = None
    step_from_root = float(distance * state.solving_grid_size)

  adapter = state.clip_pattern_adapter

  # -- Template 1: FOLD -------------------------------------------------------
  if template == clip_pattern_shaping_adapter.TemplateName.FOLD:
    if (
        math.isclose(bend_angle, 0.0)
        and math.isclose(rotate_angle, 0.0)
        and not enable_narrow
    ):
      raise ValueError(
          'At least one of bend_angle, rotate_angle or enable_narrow is needed.'
      )
    allow_3d = not math.isclose(bend_angle, 0.0)
    last_reason = 'no shaping creases produced'

    if river is not None:
      assert river_width is not None
      if enable_narrow:
        step_cands = [float(step)]
        if step > 0.0 and river_width - step < 2.0 - 1e-6:
          step_cands.append(0.0)
        margins = [0.0, _CLIP_PATTERN_MARGIN]
      else:
        step_cands = _dedup_floats(
            [float(step)] + ([0.5] if step < 0.5 else [])
        )
        margins = [_CLIP_PATTERN_MARGIN, 0.0]
    else:
      extra_steps = [float(step)]
      if step < 0.5:
        extra_steps.append(0.5)
      if enable_narrow and int(narrow_n_times) >= 3 and step < 1.0:
        extra_steps.append(1.0)
      step_cands = _dedup_floats(extra_steps)
      margins = (
          [_CLIP_PATTERN_MARGIN, 0.0]
          if enable_narrow
          else [_CLIP_PATTERN_MARGIN]
      )

    for cand_step in step_cands:
      if river_width is not None and enable_narrow:
        rem = river_width - cand_step
        if narrow_symmetric is not None:
          sym_cands = [bool(narrow_symmetric)]
        elif rem >= 3.0 - 1e-4:
          sym_cands = [True, False]
        else:
          sym_cands = [False, True]
      elif enable_narrow:
        sym_cands = (
            [bool(narrow_symmetric)]
            if narrow_symmetric is not None
            else [True, False]
        )
      else:
        sym_cands = [True]

      for sym in sym_cands:
        for margin in margins:
          eff_narrow_length = (
              river_width
              if river_width is not None
              else (
                  min(float(flap_element.grid_unit_radius), 4.0)
                  if flap_element is not None and not global_refine
                  else None
              )
          )
          draw_pattern = (
              clip_pattern_shaping_adapter.get_clip_pattern_fold_draw_pattern(
                  margin=margin,
                  step=step_from_root + cand_step,
                  bend_angle=bend_angle,
                  rotate_angle=rotate_angle,
                  enable_narrow=enable_narrow,
                  narrow_symmetric=sym,
                  narrow_n_times=int(narrow_n_times),
                  narrow_length=eff_narrow_length,
              )
          )
          try:
            raw_lines = adapter.get_clip_pattern_shaping_lines(
                rectangles,
                draw_pattern,
                bounding_rectangles=target_rectangles,
                global_refine=global_refine,
            )
          except (ValueError, AssertionError) as error:
            last_reason = str(error)
            continue
          if not raw_lines:
            last_reason = f'No shaping creases produced for {element_name}.'
            continue

          cancel_modes = [False]
          if (
              river_width is not None
              and enable_narrow
              and not sym
              and int(narrow_n_times) == 2
              and math.isclose(river_width - cand_step, 2.0, abs_tol=1e-4)
          ):
            cancel_modes = [True, False]

          for cancel_dups in cancel_modes:
            lines = (
                _cancel_duplicate_lines(raw_lines) if cancel_dups else raw_lines
            )
            creases = [line.to_crease() for line in lines]
            if not creases:
              continue
            shaped = state.pattern.update_with_shaping_creases(creases)
            new_state, reason = _accept_pattern(
                state,
                shaped,
                allow_3d_vertices=allow_3d,
                verify_fold=verify_fold,
            )
            if new_state is not None:
              delta_c = len(shaped.creases) - len(state.pattern.creases)
              LAST_REPORT.num_new_creases = delta_c
              LAST_REPORT.details = {
                  'template': 'fold',
                  'step': cand_step,
                  'narrow_symmetric': sym,
                  'cancel_midline_caps': cancel_dups,
              }
              LAST_REPORT.summary = (
                  f'{element_name}: clip_pattern_fold(fold, step={cand_step},'
                  f' narrow={enable_narrow}, n={narrow_n_times}, sym={sym});'
                  f' +{delta_c} creases.'
              )
              return new_state
            last_reason = reason
    raise ValueError(
        f'clip_pattern_fold(fold) on {element_name} failed: {last_reason}'
    )

  # -- Template 2: TURN_AND_NARROW --------------------------------------------
  if template == clip_pattern_shaping_adapter.TemplateName.TURN_AND_NARROW:
    if river is not None:
      raise ValueError(
          f"{element_name} is a river; template_name='turn_and_narrow' only"
          " applies to flaps (use template_name='fold' with enable_narrow=True"
          ' for rivers).'
      )
    base_int = round(step_from_root)
    step_cands = _dedup_floats([float(round(step)), float(step), 0.0, 1.0])
    last_reason = 'no foldable 3-D turn_and_narrow candidate'
    for cand_step in step_cands:
      total_step = (
          float(base_int + round(cand_step))
          if math.isclose(cand_step, round(cand_step), abs_tol=1e-6)
          else step_from_root + cand_step
      )
      eff_radius = (
          float(flap_element.grid_unit_radius)
          if flap_element is not None
          else 4.0
      )
      turn_length = (
          None
          if global_refine
          else (float(length) if length is not None else eff_radius)
      )
      draw_pattern = (
          clip_pattern_shaping_adapter.get_clip_pattern_template_draw_pattern(
              template_name=template,
              margin=0.0,
              step=total_step,
              bend_angle=0.0,
              length=turn_length,
          )
      )
      try:
        lines = adapter.get_clip_pattern_shaping_lines(
            rectangles,
            draw_pattern,
            bounding_rectangles=target_rectangles,
            global_refine=global_refine,
        )
      except (ValueError, AssertionError) as error:
        last_reason = str(error)
        continue
      unique_lines = _dedup_lines(lines)
      raw_creases = [line.to_crease() for line in unique_lines]
      creases = crease_utils.merge_creases(raw_creases, only_trivial=False)
      if not creases:
        last_reason = f'No shaping creases produced for {element_name}.'
        continue
      shaped = state.pattern.update_with_shaping_creases(creases)
      if not _all_internal_vertices_3d_foldable(shaped):
        last_reason = 'internal vertex failed 3-D Belcastro-Hull foldability'
        continue
      if verify_fold:
        ok, reason = shaping_fold3d.folds(shaped)
        if not ok:
          last_reason = reason
          continue
      delta_c = len(shaped.creases) - len(state.pattern.creases)
      LAST_REPORT.num_new_creases = delta_c
      LAST_REPORT.details = {'template': 'turn_and_narrow', 'step': cand_step}
      LAST_REPORT.summary = (
          f'{element_name}: turn_and_narrow at step={cand_step};'
          f' +{delta_c} creases.'
      )
      next_bb = getattr(state, 'flat_backbone', None) or (
          fidelity.as_flat_backbone(state.pattern)
          if state.pattern.local_flat_foldability()
          else None
      )
      return state.replace(pattern=shaped, flat_backbone=next_bb)
    raise ValueError(f'turn_and_narrow on {element_name} failed: {last_reason}')

  # -- Template 3: BEND_WHEN_TURN_AND_NARROW ----------------------------------
  if (
      template
      == clip_pattern_shaping_adapter.TemplateName.BEND_WHEN_TURN_AND_NARROW
  ):
    if river is not None:
      raise ValueError(
          f"{element_name} is a river; 'bend_when_turn_and_narrow' only"
          ' applies to flaps.'
      )
    if math.isclose(bend_angle, 0.0):
      raise ValueError(
          'bend_angle must be non-zero for bend_when_turn_and_narrow.'
      )
    last_reason = 'no foldable 3-D bend_when_turn_and_narrow candidate'
    direct_steps = _dedup_floats([
        float(step),
        float(step) + 0.25,
        max(1.25, float(round(step)) + 0.25),
        max(1.5, float(round(step)) + 0.5),
    ])
    for cand_step in direct_steps:
      draw_pattern = (
          clip_pattern_shaping_adapter.get_clip_pattern_template_draw_pattern(
              template_name=template,
              margin=0.0,
              step=step_from_root + cand_step,
              bend_angle=bend_angle,
          )
      )
      try:
        lines = adapter.get_clip_pattern_shaping_lines(
            rectangles,
            draw_pattern,
            bounding_rectangles=target_rectangles,
            global_refine=global_refine,
        )
      except (ValueError, AssertionError) as error:
        last_reason = str(error)
        continue
      unique_lines = _dedup_lines(lines)
      raw_creases = [line.to_crease() for line in unique_lines]
      creases = crease_utils.merge_creases(raw_creases, only_trivial=False)
      if not creases:
        continue
      shaped = state.pattern.update_with_shaping_creases(creases)
      if not _all_internal_vertices_3d_foldable(shaped):
        last_reason = 'flap does not yet have 1/6 turn_and_narrow pleats'
        continue
      if verify_fold:
        ok, reason = shaping_fold3d.folds(shaped)
        if not ok:
          last_reason = reason
          continue
      delta_c = len(shaped.creases) - len(state.pattern.creases)
      LAST_REPORT.num_new_creases = delta_c
      LAST_REPORT.details = {
          'template': 'bend_when_turn_and_narrow',
          'step': cand_step,
          'auto_chained_turn_and_narrow': False,
      }
      LAST_REPORT.summary = (
          f'{element_name}: bend_when_turn_and_narrow(angle={bend_angle},'
          f' step={cand_step}); +{delta_c} creases.'
      )
      next_bb = getattr(state, 'flat_backbone', None) or (
          fidelity.as_flat_backbone(state.pattern)
          if state.pattern.local_flat_foldability()
          else None
      )
      return state.replace(pattern=shaped, flat_backbone=next_bb)

    # Auto-chain `turn_and_narrow` + `bend_when_turn_and_narrow` when called on
    # a flap that has not yet been narrowed with `turn_and_narrow`.
    tn_steps = _dedup_floats([max(0.0, float(round(step) - 1)), 0.0])
    for tn_step in tn_steps:
      try:
        turned = clip_pattern_fold(
            state,
            element_name=element_name,
            step=tn_step,
            template_name='turn_and_narrow',
            length=length,
            global_refine=global_refine,
            verify_fold=verify_fold,
        )
      except ValueError as error:
        last_reason = str(error)
        continue
      for bw_step in _dedup_floats(
          [max(tn_step + 1.25, float(step)), tn_step + 1.25, tn_step + 1.5]
      ):
        draw_pattern = (
            clip_pattern_shaping_adapter.get_clip_pattern_template_draw_pattern(
                template_name=template,
                margin=0.0,
                step=step_from_root + bw_step,
                bend_angle=bend_angle,
            )
        )
        try:
          lines = adapter.get_clip_pattern_shaping_lines(
              rectangles,
              draw_pattern,
              bounding_rectangles=target_rectangles,
              global_refine=global_refine,
          )
        except (ValueError, AssertionError) as error:
          last_reason = str(error)
          continue
        unique_lines = _dedup_lines(lines)
        raw_creases = [line.to_crease() for line in unique_lines]
        creases = crease_utils.merge_creases(raw_creases, only_trivial=False)
        if not creases:
          continue
        shaped = turned.pattern.update_with_shaping_creases(creases)
        if not _all_internal_vertices_3d_foldable(shaped):
          last_reason = 'chained bend failed 3-D vertex foldability'
          continue
        if verify_fold:
          ok, reason = shaping_fold3d.folds(shaped)
          if not ok:
            last_reason = reason
            continue
        delta_c = len(shaped.creases) - len(state.pattern.creases)
        LAST_REPORT.num_new_creases = delta_c
        LAST_REPORT.details = {
            'template': 'bend_when_turn_and_narrow',
            'turn_step': tn_step,
            'step': bw_step,
            'auto_chained_turn_and_narrow': True,
        }
        LAST_REPORT.summary = (
            f'{element_name}: auto-chained turn_and_narrow(step={tn_step}) +'
            f' bend_when_turn_and_narrow(angle={bend_angle}, step={bw_step});'
            f' +{delta_c} creases.'
        )
        next_bb = getattr(turned, 'flat_backbone', None) or getattr(
            state, 'flat_backbone', None
        )
        return turned.replace(pattern=shaped, flat_backbone=next_bb)
    raise ValueError(
        f'bend_when_turn_and_narrow on {element_name} failed: {last_reason}'
    )

  # -- Template 4: PLEAT_TO_HIDE ----------------------------------------------
  if pleat_distance <= 0.0:
    raise ValueError(f'pleat_distance must be > 0, got {pleat_distance}.')
  if length is not None and length <= 0.0:
    raise ValueError(f'length must be > 0 when provided, got {length}.')

  step_cands = _dedup_floats([float(step)] + ([0.5] if step < 0.5 else []))
  dist_cands = _dedup_floats([float(pleat_distance), 0.5, 0.25])
  last_reason = 'no foldable pleat_to_hide candidate'
  for cand_step in step_cands:
    if length is not None:
      span = float(length)
    elif river_width is not None:
      span = (
          river_width - cand_step
          if global_refine
          else min(2.0, max(0.5, river_width - cand_step))
      )
    else:
      assert flap_element is not None
      span = (
          float(flap_element.grid_unit_radius) - cand_step
          if global_refine
          else min(
              2.0, max(1.0, float(flap_element.grid_unit_radius) - cand_step)
          )
      )
    margin = _CLIP_PATTERN_MARGIN if math.isclose(cand_step, 0.0) else 0.0
    total_step = step_from_root + cand_step
    base_length = total_step + margin + span

    for cand_dist in dist_cands:
      if span <= cand_dist * 1.05:
        continue
      draw_pattern = (
          clip_pattern_shaping_adapter.get_clip_pattern_template_draw_pattern(
              template_name=template,
              margin=margin,
              step=total_step,
              bend_angle=0.0,
              length=base_length,
              pleat_distance=cand_dist,
          )
      )
      try:
        lines = adapter.get_clip_pattern_shaping_lines(
            rectangles,
            draw_pattern,
            bounding_rectangles=target_rectangles,
            global_refine=global_refine,
        )
      except (ValueError, AssertionError) as error:
        last_reason = str(error)
        continue
      unique_lines = _dedup_lines(lines)
      raw_creases = [line.to_crease() for line in unique_lines]
      creases = crease_utils.merge_creases(raw_creases, only_trivial=False)
      if not creases:
        continue
      shaped = state.pattern.update_with_shaping_creases(creases)
      new_state, reason = _accept_pattern(
          state, shaped, allow_3d_vertices=False, verify_fold=verify_fold
      )
      if new_state is not None:
        delta_c = len(shaped.creases) - len(state.pattern.creases)
        LAST_REPORT.num_new_creases = delta_c
        LAST_REPORT.details = {
            'template': 'pleat_to_hide',
            'step': cand_step,
            'span': span,
            'pleat_distance': cand_dist,
        }
        LAST_REPORT.summary = (
            f'{element_name}: pleat_to_hide(step={cand_step}, span={span:.2f},'
            f' dist={cand_dist}); +{delta_c} creases.'
        )
        return new_state
      last_reason = reason
  raise ValueError(f'pleat_to_hide on {element_name} failed: {last_reason}')


def pleat_to_hide(
    state: Any,
    *,
    element_name: str,
    length: float | None = None,
    pleat_distance: float = 0.5,
    step: float = 0.0,
    verify_fold: bool = True,
) -> Any:
  """Shortens/sinks part of a flap or river with horizontal pleats."""
  return clip_pattern_fold(
      state,
      element_name=element_name,
      step=step,
      template_name='pleat_to_hide',
      pleat_distance=pleat_distance,
      length=length,
      verify_fold=verify_fold,
  )


# ------------------------------------------------------------------------------
# 3. Reliable narrowing (`simple_fold_narrow`).
# ------------------------------------------------------------------------------


def _preprocess_simple_angle(angle_deg: float) -> float:
  wrapped = angle_utils.wrap_angle_centered(angle_deg, 360.0)
  if abs(wrapped) < _MIN_SIMPLE_FOLD_ANGLE_DEG:
    wrapped = math.copysign(_MIN_SIMPLE_FOLD_ANGLE_DEG, wrapped or 1.0)
  return math.radians(wrapped)


def _try_single_simple_narrow(
    state: Any,
    *,
    element: shaping_elements.ShapingFlap,
    angle: float,
    depth: float,
    fold_percent: float,
    fold_order: str | None,
    verify_fold: bool,
) -> tuple[Any | None, str]:
  """Runs one `(angle, fold_order)` simple-fold narrowing candidate on a flap."""
  orders = {
      'MV': zigzag_shaping_utils.MOUNTAIN_TO_VALLEY,
      'VM': zigzag_shaping_utils.VALLEY_TO_MOUNTAIN,
  }
  angle_rad = _preprocess_simple_angle(angle)
  order_seq = orders[fold_order] if fold_order is not None else orders['MV']
  grid_depth = min(
      max(0, int(element.grid_unit_radius * depth)),
      max(0, element.grid_unit_radius - 1),
  )
  z_params = shaping_types.ZigzagParams(
      target_flap_angle=abs(angle_rad),
      even_phase=angle_rad >= 0,
      fold_order=order_seq,
      fold_percent=float(fold_percent),
      depth=grid_depth,
  )
  elem_param = zigzag_shaping.ZigzagShapingParamsForElement(
      element_name=element.name, zigzag_params=z_params
  )
  flaps, rivers = state.shaping_elements
  if state.config.apply_simple_fold_heuristics:
    flap_info = state.shaping_heuristic_data['flap_info']
    orientations = {
        name: node.absolute_direction for name, node in flap_info.items()
    }
    signed = dataclasses.replace(
        elem_param,
        zigzag_params=dataclasses.replace(
            z_params,
            target_flap_angle=abs(angle_rad)
            * (1.0 if z_params.even_phase else -1.0),
        ),
    )
    try:
      adjusted = list(
          simple_fold_shaping.apply_heuristics_to_shaping_params(
              grid_size=element.grid_size,
              flaps=flaps,
              rivers=rivers,
              element_orientations=orientations,
              hinge_segments=state.hinge_segments,
              element_shaping_params=[signed],
              use_heuristics_for_fold_order=(
                  state.config.use_heuristics_for_fold_order
                  and fold_order is None
              ),
              use_heuristics_for_depth=state.config.use_heuristics_for_depth,
              use_heuristics_for_phase=state.config.use_heuristics_for_phase,
              convert_small_angles=False,
          )
      )
    except (ValueError, AssertionError) as error:
      return None, str(error)
    if not adjusted:
      return None, 'heuristics returned empty params'
  else:
    adjusted = [elem_param]

  try:
    shaped = simple_fold_shaping.shape_cp_with_simple_folds(
        cp=state.pattern,
        element_shaping_params=adjusted,
        grid_size=element.grid_size,
        flaps=flaps,
        rivers=rivers,
        central_crease_removed=state.central_crease_removed,
        narrow=True,
        light_mode=not state.config.check_all_crease_intersections,
    )
  except (ValueError, AssertionError, KeyError, IndexError) as error:
    return None, str(error)

  if len(shaped.creases) == len(state.pattern.creases):
    return None, 'no shaping creases generated'
  new_state, reason = _accept_pattern(
      state, shaped, allow_3d_vertices=False, verify_fold=verify_fold
  )
  if new_state is None:
    return None, reason
  if element.name not in new_state.narrowed_elements:
    new_state = new_state.replace(
        narrowed_elements=new_state.narrowed_elements + (element.name,)
    )
  return new_state, 'ok'


def simple_fold_narrow(
    state: Any,
    *,
    element_name: str,
    angle: float = 45.0,
    depth: float = 0.0,
    fold_percent: float = 1.0,
    fold_order: str | None = None,
    fallback_clip_narrow: bool = True,
    verify_fold: bool = True,
) -> Any:
  """Halves the width of a flap (or narrows a river) with automatic fallbacks.

  Tries `simple_fold(..., narrow=True)` on flaps (retrying `-angle` and explicit
  `'MV'`/`'VM'` fold orders if the folder rejects the first phase due to layer
  strain), and falls back to `clip_pattern_fold(..., enable_narrow=True)` for
  interior flaps or rivers.

  Args:
    state: Current `ShapingState`.
    element_name: Flap or river name.
    angle: Target turn angle in degrees (non-zero; default `45.0`).
    depth: Fractional start depth along the element in `[0, 1)`.
    fold_percent: Fold fraction in `(0, 1]`.
    fold_order: Optional `'MV'` or `'VM'`.
    fallback_clip_narrow: Fall back to `clip_pattern_fold(enable_narrow=True)`
      if zig-zag narrowing fails (e.g. interior flaps without a paper border).
    verify_fold: Keep only candidates that fold in the deterministic folder.

  Returns:
    New `ShapingState` with `element_name` recorded in `narrowed_elements`.
  """
  LAST_REPORT.tool = 'simple_fold_narrow'
  LAST_REPORT.element_name = element_name
  LAST_REPORT.summary = ''
  LAST_REPORT.details = {}

  if math.isclose(angle, 0.0):
    raise ValueError('angle must be non-zero.')
  if fold_order is not None and fold_order not in ('MV', 'VM'):
    raise ValueError(f'fold_order must be MV or VM, got {fold_order!r}.')

  element = state.element(element_name)
  if isinstance(element, shaping_elements.ShapingRiver):
    step_units = max(0.0, float(int(element.width * depth)))
    narrowed = clip_pattern_fold(
        state,
        element_name=element_name,
        step=step_units,
        template_name='fold',
        enable_narrow=True,
        verify_fold=verify_fold,
    )
    if element_name not in narrowed.narrowed_elements:
      narrowed = narrowed.replace(
          narrowed_elements=narrowed.narrowed_elements + (element_name,)
      )
    delta_c = len(narrowed.pattern.creases) - len(state.pattern.creases)
    LAST_REPORT.tool = 'simple_fold_narrow'
    LAST_REPORT.num_new_creases = delta_c
    LAST_REPORT.details = {'method': 'river_clip_narrow', 'step': step_units}
    LAST_REPORT.summary = (
        f'{element_name}: narrowed river via clip_pattern_fold(step='
        f'{step_units}); +{delta_c} creases.'
    )
    return narrowed

  candidates: list[tuple[float, str | None]] = [
      (angle, fold_order),
      (-angle, fold_order),
  ]
  for sgn_angle in (angle, -angle):
    for explicit_order in ('MV', 'VM'):
      pair = (sgn_angle, explicit_order)
      if pair not in candidates:
        candidates.append(pair)

  last_reason = 'no candidate tried'
  for cand_angle, cand_order in candidates:
    new_state, reason = _try_single_simple_narrow(
        state,
        element=element,
        angle=cand_angle,
        depth=depth,
        fold_percent=fold_percent,
        fold_order=cand_order,
        verify_fold=verify_fold,
    )
    if new_state is not None:
      delta_c = len(new_state.pattern.creases) - len(state.pattern.creases)
      LAST_REPORT.num_new_creases = delta_c
      LAST_REPORT.details = {
          'method': 'simple_fold',
          'angle': cand_angle,
          'fold_order': cand_order,
      }
      LAST_REPORT.summary = (
          f'{element_name}: simple_fold_narrow(angle={cand_angle},'
          f' order={cand_order}); +{delta_c} creases.'
      )
      return new_state
    last_reason = reason

  if fallback_clip_narrow:
    step_units = max(0.0, float(element.grid_unit_radius) * float(depth))
    try:
      narrowed = clip_pattern_fold(
          state,
          element_name=element_name,
          step=step_units,
          template_name='fold',
          rotate_angle=0.0,
          enable_narrow=True,
          verify_fold=verify_fold,
      )
      if element_name not in narrowed.narrowed_elements:
        narrowed = narrowed.replace(
            narrowed_elements=narrowed.narrowed_elements + (element_name,)
        )
      delta_c = len(narrowed.pattern.creases) - len(state.pattern.creases)
      LAST_REPORT.tool = 'simple_fold_narrow'
      LAST_REPORT.num_new_creases = delta_c
      LAST_REPORT.details = {'method': 'clip_narrow_fallback'}
      LAST_REPORT.summary = (
          f'{element_name}: narrowed via clip_pattern_fold fallback;'
          f' +{delta_c} creases (zig-zag failed: {last_reason}).'
      )
      return narrowed
    except ValueError as fallback_error:
      last_reason = f'{last_reason}; clip fallback: {fallback_error}'

  raise ValueError(
      f'simple_fold_narrow on {element_name} failed: {last_reason}'
  )


# ------------------------------------------------------------------------------
# 4. Smooth multi-segment curvature (`curve_flap`) & `level_shifter`.
# ------------------------------------------------------------------------------


def _chain_from_element(state: Any, element_name: str) -> list[str]:
  """Returns the linear flap chain suffix starting at `element_name`."""
  _, merged_names = state.tree.merge_linear_chains()
  for sub_names in merged_names.values():
    names = list(sub_names)
    if element_name in names:
      idx = names.index(element_name)
      return names[idx:]
  return [element_name]


def _tip_displacement_alignment(
    before_state: Any,
    after_state: Any,
    chain: Sequence[str],
    direction: str,
    base_cached: tuple[Any, Any, Any] | None = None,
) -> float:
  """Scores how well `after_state` moves the chain tip along `direction`."""
  dir_vecs = {
      'up': np.array([0.0, 0.0, 1.0]),
      'down': np.array([0.0, 0.0, -1.0]),
      'right': np.array([1.0, 0.0, 0.0]),
      'left': np.array([-1.0, 0.0, 0.0]),
      'forward': np.array([0.0, 1.0, 0.0]),
      'backward': np.array([0.0, -1.0, 0.0]),
  }
  goal_dir = dir_vecs.get(direction)
  if goal_dir is None:
    return 0.0
  topo = fidelity.topology(before_state.tree)
  tip_joint = topo.distal[chain[-1]]
  base_joint = topo.proximal[chain[0]]
  try:
    if base_cached is not None:
      m_before, ev_before, j_before = base_cached
    else:
      m_before = _fold_model(before_state.pattern)
      ev_before = fidelity.evaluate(
          before_state.tree, before_state.packing, m_before
      )
      j_before = fidelity.locate_joints(
          before_state.tree, before_state.packing, m_before
      )
    m_after = _fold_model(after_state.pattern)
    j_after = fidelity.locate_joints(
        after_state.tree, after_state.packing, m_after
    )
    if (
        j_before.get(tip_joint) is None
        or j_after.get(tip_joint) is None
        or j_before.get(base_joint) is None
        or j_after.get(base_joint) is None
    ):
      return 0.0
    rel_before = np.asarray(j_before[tip_joint]) - np.asarray(
        j_before[base_joint]
    )
    rel_after = np.asarray(j_after[tip_joint]) - np.asarray(j_after[base_joint])
    delta_stick_space = ev_before.fit.rotation @ (rel_after - rel_before)
    return float(delta_stick_space @ goal_dir)
  except Exception:  # pylint: disable=broad-exception-caught
    return 0.0


def _curve_flap_tree_edit(
    state: Any,
    *,
    element_name: str,
    n_segments: int,
    total_bend_deg: float,
    direction: str,
    verify_fold: bool,
) -> Any:
  """Subdivides `element_name` into `n_segments` sub-sticks on an unshaped CP."""
  if state.pattern is not state.base_pattern:
    raise ValueError(
        "method='tree_edit' requires an unshaped base pattern; call reset() or"
        " use method='auto' / 'crimp_3d' / 'reverse_3d'."
    )
  tree_dict = state.tree.to_dict()
  sticks_key = 'children' if 'children' in tree_dict else 'sticks'
  target_stick = None
  target_idx = -1
  for idx, stick in enumerate(tree_dict[sticks_key]):
    if stick['name'] == element_name:
      target_stick = dict(stick)
      target_idx = idx
      break
  if target_stick is None or target_stick.get('is_river', False):
    raise ValueError(f'{element_name} is not a flap stick in the stick figure.')
  total_len = int(round(float(target_stick['length'])))
  if total_len < n_segments:
    raise ValueError(
        f'{element_name} has integer length {total_len} < n_segments='
        f'{n_segments} for tree_edit.'
    )

  base_len = total_len // n_segments
  rem = total_len % n_segments
  seg_lengths = [base_len + (1 if i < rem else 0) for i in range(n_segments)]
  step_deg = float(total_bend_deg) / float(n_segments)
  el_sign = -1.0 if direction in ('down', 'negative') else 1.0
  az_sign = (
      1.0 if direction == 'right' else (-1.0 if direction == 'left' else 0.0)
  )

  sub_sticks = []
  sub_names = []
  prev_joint = target_stick['start']
  base_el = float(target_stick['elevation_angle'])
  base_az = float(target_stick['azimuth_angle'])
  for i, seg_len in enumerate(seg_lengths):
    sub_name = f'{element_name}_seg{i}'
    end_joint = (
        target_stick['end']
        if i == n_segments - 1
        else f'{element_name}_curve_j{i + 1}'
    )
    frac = i + 1
    el = max(-89.0, min(89.0, base_el + el_sign * step_deg * frac))
    az = angle_utils.wrap_angle_centered(
        base_az + az_sign * step_deg * frac, 360.0
    )
    new_stick = dict(target_stick)
    new_stick.update({
        'name': sub_name,
        'start': prev_joint,
        'end': end_joint,
        'length': seg_len,
        'elevation_angle': el,
        'azimuth_angle': az,
        'symmetric_stick_name': None,
    })
    if i < n_segments - 1:
      new_stick['flap_tip_shape'] = 'rectangular'
    sub_sticks.append(new_stick)
    sub_names.append(sub_name)
    prev_joint = end_joint

  new_sticks = (
      [dict(st) for st in tree_dict[sticks_key][:target_idx]]
      + sub_sticks
      + [dict(st) for st in tree_dict[sticks_key][target_idx + 1 :]]
  )
  for st in new_sticks:
    if st.get('symmetric_stick_name') == element_name:
      st['symmetric_stick_name'] = None
  edited_dict = dict(tree_dict)
  edited_dict[sticks_key] = new_sticks
  if edited_dict.get('root') == element_name:
    edited_dict['root'] = sub_names[0]

  edited_tree = stick_figure_lib.StickFigure.from_dict(edited_dict)
  stick_map: Any = {st['name']: [st['name']] for st in tree_dict[sticks_key]}
  stick_map[element_name] = sub_names
  if not utils_tree.check_trees_are_packing_equivalent(
      state.tree,
      edited_tree,
      state.packing_config,
      check_traversal_order=False,
  ):
    raise ValueError('Subdivided tree is not packing-equivalent to base tree.')
  merge_converter = utils_tree.get_merge_converter_from_substick_map(
      edited_tree, stick_map
  )
  flaps, rivers = state.shaping_elements
  shaped = heuristics_utils.shape_cp_with_heuristics_pipeline_wrapper(
      cp=state.pattern,
      packing_output=state.packing,
      solving_output=state.solving,
      solving_grid_size=state.solving_grid_size,
      stick_figure=edited_tree,
      merge_converter=merge_converter,
      shaping_flaps=flaps,
      shaping_rivers=rivers,
      central_crease_removed=state.central_crease_removed,
      narrow=False,
  )
  new_state, reason = _accept_pattern(
      state, shaped, allow_3d_vertices=False, verify_fold=verify_fold
  )
  if new_state is None:
    raise ValueError(f'tree_edit curve_flap failed: {reason}')
  delta_c = len(shaped.creases) - len(state.pattern.creases)
  LAST_REPORT.num_new_creases = delta_c
  LAST_REPORT.details = {
      'method': 'tree_edit',
      'segments': sub_names,
      'applied_segments': n_segments,
  }
  LAST_REPORT.summary = (
      f'{element_name}: curve_flap(tree_edit, n={n_segments},'
      f' total={total_bend_deg} deg); +{delta_c} creases.'
  )
  return new_state


def curve_flap(
    state: Any,
    *,
    element_name: str,
    n_segments: int = 3,
    total_bend_deg: float = 90.0,
    direction: str = 'up',
    method: str = 'auto',
    fold_percent: float = 1.0,
    verify_fold: bool = True,
) -> Any:
  """Curves a flap or linear flap chain smoothly in one C-curve call.

  Distributes `n_segments` bends (`total_bend_deg / n_segments` each) along the
  linear flap chain starting at `element_name` (if it has sub-segments such as
  `metasoma_1..telson`) or at evenly spaced depths along `element_name`, keeping
  the world-space curvature consistent (a crimp has two reflections so the next
  segment keeps the same angle sign; a reverse fold has one reflection so the
  next segment flips sign to continue curving in the same C-direction).

  Args:
    state: Current `ShapingState`.
    element_name: Proximal flap (or single flap) to curve.
    n_segments: Number of bend stations (`>= 1`, typically `2..5`).
    total_bend_deg: Total cumulative bend angle in degrees (non-zero).
    direction: `'up'`, `'down'`, `'left'`, `'right'`, `'positive'`, or
      `'negative'`.
    method: `'auto'` (prefers `crimp_fold_3d` with `reverse_fold_3d` fallback
      per station), `'crimp_3d'`, `'reverse_3d'`, or `'tree_edit'`.
    fold_percent: Crease fold fraction in `(0, 1]` for 3-D folds.
    verify_fold: Keep only candidates that fold in the deterministic folder.

  Returns:
    New `ShapingState`.
  """
  LAST_REPORT.tool = 'curve_flap'
  LAST_REPORT.element_name = element_name
  LAST_REPORT.summary = ''
  LAST_REPORT.details = {}

  if int(n_segments) < 1:
    raise ValueError(f'n_segments must be >= 1, got {n_segments}.')
  if math.isclose(total_bend_deg, 0.0):
    raise ValueError('total_bend_deg must be non-zero.')
  dir_norm = direction.lower()
  valid_dirs = (
      'up',
      'down',
      'left',
      'right',
      'forward',
      'backward',
      'positive',
      'negative',
  )
  if dir_norm not in valid_dirs:
    raise ValueError(
        f'direction must be one of {valid_dirs}, got {direction!r}.'
    )
  method_norm = method.lower()
  if method_norm not in ('auto', 'crimp_3d', 'reverse_3d', 'tree_edit'):
    raise ValueError(
        "method must be 'auto', 'crimp_3d', 'reverse_3d', or 'tree_edit',"
        f' got {method!r}.'
    )

  element = state.element(element_name)
  if not isinstance(element, shaping_elements.ShapingFlap):
    raise ValueError(f'{element_name} is not a flap; curve_flap needs a flap.')

  n_seg = int(n_segments)
  if method_norm == 'tree_edit':
    return _curve_flap_tree_edit(
        state,
        element_name=element_name,
        n_segments=n_seg,
        total_bend_deg=total_bend_deg,
        direction=dir_norm,
        verify_fold=verify_fold,
    )

  prior_backbone = getattr(state, 'flat_backbone', None)
  work_state = (
      state.replace(pattern=prior_backbone, flat_backbone=None)
      if prior_backbone is not None
      else state
  )

  chain = _chain_from_element(work_state, element_name)
  # Build `(seg_element_name, depth)` stations in proximal -> distal order.
  stations: list[tuple[str, float]] = []
  if len(chain) >= n_seg:
    if n_seg > 1:
      indices = [
          min(len(chain) - 1, int(round(i * (len(chain) - 1) / (n_seg - 1))))
          for i in range(n_seg)
      ]
    else:
      indices = [0]
    used: list[int] = []
    for idx in indices:
      while idx in used and idx + 1 < len(chain):
        idx += 1
      if idx not in used:
        used.append(idx)
    stations = [(chain[idx], 0.35) for idx in used]
  elif len(chain) == 1:
    max_single = min(n_seg, max(1, int(element.grid_unit_radius) // 2), 3)
    single_depths = {
        1: (0.45,),
        2: (0.30, 0.65),
        3: (0.22, 0.48, 0.72),
    }[max_single]
    stations = [(chain[0], d_val) for d_val in single_depths]
  else:
    stations = [(cname, 0.38) for cname in chain]
    last_el = work_state.element(chain[-1])
    if (
        len(stations) < n_seg
        and isinstance(last_el, shaping_elements.ShapingFlap)
        and int(last_el.grid_unit_radius) >= 3
    ):
      stations[-1] = (chain[-1], 0.28)
      stations.append((chain[-1], 0.62))

  if method_norm == 'reverse_3d':
    ops_order = ('reverse_3d',)
  elif method_norm == 'crimp_3d':
    ops_order = ('crimp_3d',)
  else:
    ops_order = ('crimp_3d', 'reverse_3d')

  def _step_mag_for(num_steps: int) -> float:
    return max(
        10.0,
        min(120.0, abs(float(total_bend_deg)) / float(max(1, num_steps))),
    )

  def _run_batch(
      sub_stations: Sequence[tuple[str, float]],
      start_sign: float,
  ) -> tuple[Any | None, list[tuple[str, float, str, float]], float]:
    b_mag = _step_mag_for(len(sub_stations))
    s_state = work_state
    s_sign = start_sign
    s_steps: list[tuple[str, float, str, float]] = []
    for s_name, d_val in sub_stations:
      nxt = None
      op_chosen = None
      for op_kind in ops_order:
        try:
          if op_kind == 'crimp_3d':
            nxt = shaping_fold3d.crimp_fold_3d(
                s_state,
                element_name=s_name,
                angle=s_sign * b_mag,
                depth=d_val,
                fold_percent=fold_percent,
                verify_fold=False,
            )
          else:
            nxt = shaping_fold3d.reverse_fold_3d(
                s_state,
                element_name=s_name,
                angle=s_sign * b_mag,
                depth=d_val,
                fold_percent=fold_percent,
                verify_fold=False,
            )
          op_chosen = op_kind
          break
        except (ValueError, AssertionError):
          continue
      if nxt is None or op_chosen is None:
        return None, [], b_mag
      s_steps.append((s_name, d_val, op_chosen, round(s_sign * b_mag, 1)))
      s_state = nxt
      if op_chosen == 'reverse_3d':
        s_sign = -s_sign
    pct, _, _ = s_state.pattern.vertex_flat_foldability_percentage()
    if pct < 0.90:
      return None, [], b_mag
    if prior_backbone is not None:
      combined = fidelity.replay_3d_overlay(
          prior_backbone, state.pattern, s_state.pattern
      )
      ok_ff, _ = fidelity.preserves_foldability(state.pattern, combined)
      if not ok_ff:
        return None, [], b_mag
      next_bb = (
          fidelity.as_flat_backbone(s_state.pattern)
          if fidelity.has_3d_overlay(combined)
          else None
      )
      s_state = s_state.replace(pattern=combined, flat_backbone=next_bb)
    elif fidelity.has_3d_overlay(s_state.pattern):
      s_state = s_state.replace(
          flat_backbone=fidelity.as_flat_backbone(s_state.pattern)
      )
    return s_state, s_steps, b_mag

  def _predict_curve_sign(elem: str, target_dir_str: str) -> float:
    try:
      gd = getattr(work_state, 'graph_data', None)
      if gd is None:
        return 1.0
      poly = shaping_fold3d.element_polygon(work_state, elem)
      ref = shaping_fold3d.reference_point(work_state, elem, gd)
      axis = (
          shaping_fold3d.geometric_fold_3d_heuristics.get_element_folded_axis(
              gd, poly, ref
          )
      )
      dir_axis = axis.direction
      left_norm = np.array([-dir_axis[1], dir_axis[0]])
      target_map = {
          'up': np.array([0.0, 1.0]),
          'down': np.array([0.0, -1.0]),
          'left': np.array([-1.0, 0.0]),
          'right': np.array([1.0, 0.0]),
      }
      tv = target_map.get(target_dir_str)
      if tv is not None:
        dot = np.dot(tv, left_norm)
        if abs(dot) > 1e-4:
          return 1.0 if dot > 0 else -1.0
    except Exception:  # pylint: disable=broad-exception-caught
      pass
    if target_dir_str in ('up', 'right', 'forward', 'positive'):
      return 1.0 if total_bend_deg >= 0 else -1.0
    return -1.0 if total_bend_deg >= 0 else 1.0

  dir_vec_3d = {
      'up': np.array([0.0, 0.0, 1.0]),
      'down': np.array([0.0, 0.0, -1.0]),
      'left': np.array([0.0, 1.0, 0.0]),
      'right': np.array([0.0, -1.0, 0.0]),
      'forward': np.array([1.0, 0.0, 0.0]),
      'backward': np.array([-1.0, 0.0, 0.0]),
  }.get(dir_norm)
  if dir_vec_3d is not None and total_bend_deg < 0:
    dir_vec_3d = -dir_vec_3d

  topo = fidelity.topology(state.tree)
  prox_joint = topo.proximal.get(chain[0], '')
  dist_joint = topo.distal.get(chain[-1], '')

  def _score_direction_3d(cand_st: Any) -> float:
    if dir_vec_3d is None:
      return 0.0
    try:
      model = fidelity.FoldedModel.from_pattern(cand_st.pattern)
      if model.vertex_error > 1.0:
        return -math.inf
      rep = fidelity.score_model(
          state.tree,
          state.packing,
          model,
          layout=state.paper_layout,
          include_geometry=True,
      )
      geom = rep.get('geometry', {})
      joints = geom.get('joints', {})
      align = rep.get('alignment', {})
      p_fold = joints.get(prox_joint, {}).get('folded')
      d_fold = joints.get(dist_joint, {}).get('folded')
      if p_fold is None or d_fold is None or 'rotation' not in align:
        return 0.0
      rot = np.asarray(align['rotation'], dtype=float)
      scale = float(align['scale'])
      vec = (
          scale
          * rot
          @ (np.asarray(d_fold, dtype=float) - np.asarray(p_fold, dtype=float))
      )
      return float(np.dot(vec, dir_vec_3d))
    except Exception:  # pylint: disable=broad-exception-caught
      return 0.0

  first_elem = stations[0][0] if stations else element_name
  pred_s = _predict_curve_sign(first_elem, dir_norm)
  sign_cands = [pred_s, -pred_s]

  station_batches: list[list[tuple[str, float]]] = [stations]
  if len(stations) > 2:
    station_batches.append([stations[0], stations[-1]])
  if len(stations) > 1:
    station_batches.append([(element_name, 0.45)])

  best_batch_state = None
  best_batch_steps = None
  chosen_mag = _step_mag_for(len(stations))

  for batch in station_batches:
    valid_for_batch: list[
        tuple[float, int, Any, list[tuple[str, float, str, float]], float]
    ] = []
    for idx_s, cand_s in enumerate(sign_cands):
      cand_st, cand_steps, b_mag = _run_batch(batch, cand_s)
      if cand_st is None:
        continue
      if verify_fold:
        if not shaping_fold3d.folds(cand_st.pattern)[0]:
          continue
        dir_score = _score_direction_3d(cand_st)
        if dir_score == -math.inf:
          continue
      else:
        dir_score = 1.0 if idx_s == 0 else 0.0
      valid_for_batch.append((dir_score, -idx_s, cand_st, cand_steps, b_mag))
      if not verify_fold or dir_vec_3d is None:
        break
    if valid_for_batch:
      _, _, best_batch_state, best_batch_steps, chosen_mag = max(
          valid_for_batch, key=lambda item: (item[0], item[1])
      )
      break

  if best_batch_state is not None and best_batch_steps is not None:
    delta_c = len(best_batch_state.pattern.creases) - len(state.pattern.creases)
    LAST_REPORT.num_new_creases = delta_c
    LAST_REPORT.details = {
        'method': method_norm,
        'requested_segments': n_seg,
        'applied_segments': len(best_batch_steps),
        'steps': best_batch_steps,
    }
    LAST_REPORT.summary = (
        f'{element_name}: curve_flap({len(best_batch_steps)}/{n_seg} segments,'
        f' step={chosen_mag:.1f} deg, dir={dir_norm}); +{delta_c} creases.'
    )
    return best_batch_state

  raise ValueError(
      f'curve_flap failed on {element_name}: all candidate bend batches failed'
      ' fold validation.'
  )


# ------------------------------------------------------------------------------
# 5. Verified `level_shifter` wrapper.
# ------------------------------------------------------------------------------

_NO_LEVEL_SHIFTER_MSG = (
    'This base has no level shifters (packing.sacrificial_flaps is empty). '
    'Level shifters must be planned at packing/solving time: pack with '
    'PackingTaskConfig(use_level_shifter=True) (requires a river and flaps '
    'with width >= 2, length > width) and solve with '
    'SolvingTaskConfig(remove_central_crease=False).'
)


def level_shifter(
    state: Any,
    *,
    element_name: str,
    position_percent: float = 1.0,
    verify_fold: bool = True,
) -> Any:
  """Widens an elevated flap with a level shifter, verifying foldability.

  Args:
    state: Current `ShapingState` (must come from a `use_level_shifter=True`
      packing with non-empty `sacrificial_flaps`).
    element_name: Flap (or sub-flap) name with width >= 2.
    position_percent: Position in `[0, 1]` along valid axial level-shifter
      stations (`1.0` = widen full flap; falls back to nearby valid positions if
      needed).
    verify_fold: Keep only candidates that fold in the deterministic folder.

  Returns:
    New `ShapingState`.
  """
  LAST_REPORT.tool = 'level_shifter'
  LAST_REPORT.element_name = element_name
  LAST_REPORT.summary = ''
  LAST_REPORT.details = {}

  if not 0.0 <= position_percent <= 1.0:
    raise ValueError(
        f'position_percent must be in [0, 1], got {position_percent}.'
    )
  if not state.packing.sacrificial_flaps:
    raise ValueError(_NO_LEVEL_SHIFTER_MSG)

  symmetric_pairs = {
      s.name: s.symmetric_stick_name
      for s in state.tree.children
      if s.symmetric_stick_name is not None
  }
  elevated = utils_ls.get_elevated_flaps(
      state.flap_rectangles,
      state.packing.sacrificial_flaps or [],
      state.packing_grid_size,
      group_mirrored=True,
      symmetric_pairs=symmetric_pairs,
  )
  by_name = {f.main_flap.name: f for f in elevated}
  if not by_name:
    raise ValueError('No level-shiftable flaps found in this packing.')

  _, merged_names = state.tree.merge_linear_chains()
  unmerged_to_merged = {
      unmerged: merged
      for merged, unmerged_list in merged_names.items()
      for unmerged in unmerged_list
  }
  merged_name = unmerged_to_merged.get(element_name, element_name)
  flap = by_name.get(merged_name)
  if flap is None:
    # Check if symmetrical partner holds the grouped LevelShiftable.
    sym_partner = symmetric_pairs.get(merged_name)
    if sym_partner is not None:
      flap = by_name.get(sym_partner)
  if flap is None:
    raise ValueError(
        f'{element_name} cannot take a level shifter; candidates:'
        f' {sorted(by_name)}'
    )

  base_crease_sources: list[tuple[str, list[crease_utils.Crease]]] = [
      ('current', list(state.pattern.creases))
  ]
  try:
    converted_creases = list(state.pattern.creases)
    for ls_flap in flap.ls_flaps:
      if hasattr(ls_flap, 'convert_to_ls_template'):
        converted_creases = ls_flap.convert_to_ls_template(
            converted_creases, state.solving_grid_size
        )
    base_crease_sources.append(('converted_template', converted_creases))
  except (ValueError, AssertionError, KeyError):
    pass

  last_reason = 'no level-shifter position succeeded'
  total_valid_positions = 0
  for source_idx, (source_label, base_creases) in enumerate(
      base_crease_sources
  ):
    try:
      valid = sorted(
          flap.valid_positions(flap.axial_paths(base_creases).keys())
      )
    except (ValueError, AssertionError, KeyError) as error:
      last_reason = str(error)
      valid = []
    if valid:
      total_valid_positions = max(total_valid_positions, len(valid))
      requested_pos = utils_ls.get_ls_position_from_percent(
          valid, float(position_percent)
      )
      ordered_positions = [requested_pos] + [
          p for p in reversed(valid) if p != requested_pos
      ]
      for pos in ordered_positions:
        try:
          creases = flap.apply_level_shifter(base_creases, pos)
          candidate = crease_pattern_lib.CreasePattern(
              creases=creases,
              grid_size=state.solving_grid_size,
              light_mode=not state.config.check_all_crease_intersections,
              merge_trivial=True,
          )
        except (ValueError, AssertionError, KeyError) as error:
          last_reason = str(error)
          continue
        new_state, reason = _accept_pattern(
            state, candidate, allow_3d_vertices=False, verify_fold=verify_fold
        )
        if new_state is not None:
          delta_c = len(candidate.creases) - len(state.pattern.creases)
          LAST_REPORT.num_new_creases = delta_c
          LAST_REPORT.details = {
              'position': str(pos),
              'num_valid_positions': len(valid),
              'base_source': source_label,
          }
          LAST_REPORT.summary = (
              f'{element_name}: level_shifter at {pos} ({len(valid)} valid'
              f' positions, base={source_label}); {delta_c:+d} creases.'
          )
          return new_state
        last_reason = reason

    # Lazily solve with LS templates if neither current nor direct conversion
    # produced a flat-foldable level-shifted CP.
    if (
        source_idx == len(base_crease_sources) - 1
        and source_label != 'solved_ls_template'
        and state.packing.packing_cp is not None
    ):
      try:
        solver_config = hinge_lib.SolverConfig(
            grid_size=state.solving_grid_size,
            packing_information=state.packing,
            allow_misaligned_symmetric_hinges=True,
            max_number_of_cps=1,
            use_python_flat_folder=True,
            check_global_flat_foldability=False,
        )
        ls_results = utils_ls.solve_with_ls_templates(
            solver_config, state.packing.packing_cp, elevated
        )
        if ls_results:
          base_crease_sources.append(
              ('solved_ls_template', list(ls_results[0].crease_pattern.creases))
          )
      except (ValueError, AssertionError, KeyError) as error:
        last_reason = str(error)

  if total_valid_positions == 0:
    raise ValueError(f'No valid level-shifter positions for {element_name}.')
  raise ValueError(
      f'level_shifter on {element_name} failed for all {total_valid_positions}'
      f' valid positions: {last_reason}'
  )


# ------------------------------------------------------------------------------
# 6. Central sagittal spine fold (`fold_spine_3d`).
# ------------------------------------------------------------------------------


def _find_body_spine_bounds(
    state: Any, x_center: float = 0.5, tol: float = 0.05
) -> tuple[float, float, str] | None:
  """Finds y_min, y_max of central body river(s) situated along x_center."""
  flaps, rivers = state.shaping_elements
  body_keywords = (
      'body',
      'mesosoma',
      'prosoma',
      'carapace',
      'pelvis',
      'thorax',
      'torso',
      'trunk',
  )
  exclude_keywords = (
      'tail',
      'metasoma',
      'stinger',
      'telson',
      'leg',
      'arm',
      'antler',
      'pedipalp',
      'head',
      'muzzle',
      'eyestalk',
      'cheliped',
      'claw',
      'major',
      'walk',
      '_l',
      '_r',
      '_left',
      '_right',
  )

  candidate_rivers = []
  for r in rivers:
    r_name_lower = r.name.lower()
    if any(ex in r_name_lower for ex in exclude_keywords):
      continue
    if r.width >= 20:
      continue
    all_c = list(r.bank_1) + list(r.bank_2)
    if not all_c:
      continue
    xs = [c.p1.x for c in all_c] + [c.p2.x for c in all_c]
    ys = [c.p1.y for c in all_c] + [c.p2.y for c in all_c]
    if min(xs) - tol <= x_center <= max(xs) + tol:
      candidate_rivers.append((r, min(ys), max(ys)))

  if not candidate_rivers:
    return None

  body_matches = [
      (r, y0, y1)
      for r, y0, y1 in candidate_rivers
      if any(kw in r.name.lower() for kw in body_keywords)
  ]
  chosen = body_matches if body_matches else candidate_rivers
  combined_y_min = min(y0 for _, y0, _ in chosen)
  combined_y_max = max(y1 for _, _, y1 in chosen)
  names = '+'.join(r.name for r, _, _ in chosen)
  return combined_y_min, combined_y_max, names


def fold_spine_3d(
    state: Any,
    *,
    element_name: str | None = None,
    fold_percent: float = 0.65,
    spine_fold_percent: float | None = None,
    fold_type: str = 'mountain',
    x_center: float = 0.5,
    y_min: float = 0.0,
    y_max: float = 1.0,
    tol: float = 0.025,
    verify_fold: bool = True,
) -> Any:
  """Folds central sagittal spine to create 3-D inverted-V standing posture or dome."""
  if spine_fold_percent is not None:
    fold_percent = spine_fold_percent

  LAST_REPORT.tool = 'fold_spine_3d'
  LAST_REPORT.element_name = (
      element_name if element_name is not None else f'spine@{x_center}'
  )
  LAST_REPORT.summary = ''
  LAST_REPORT.details = {}

  if not 0.0 < fold_percent <= 1.0:
    raise ValueError(f'fold_percent must be in (0, 1], got {fold_percent}.')
  target_fold = (
      crease_utils.Fold.MOUNTAIN
      if fold_type.lower() == 'mountain'
      else crease_utils.Fold.VALLEY
  )
  opposite_fold = (
      crease_utils.Fold.VALLEY
      if target_fold == crease_utils.Fold.MOUNTAIN
      else crease_utils.Fold.MOUNTAIN
  )

  y_ranges = []
  if element_name is not None:
    elem = state.element(element_name)
    if isinstance(elem, shaping_elements.ShapingFlap):
      rect = elem.to_legacy_rectangle()
      y0 = max(y_min, min(rect.y_start, rect.y_start + rect.y_len))
      y1 = min(y_max, max(rect.y_start, rect.y_start + rect.y_len))
      y_ranges.append((y0, y1, element_name))
    elif isinstance(elem, shaping_elements.ShapingRiver):
      all_c = list(elem.bank_1) + list(elem.bank_2)
      ys = [c.p1.y for c in all_c] + [c.p2.y for c in all_c]
      if ys:
        y0 = max(y_min, min(ys))
        y1 = min(y_max, max(ys))
        y_ranges.append((y0, y1, element_name))
    if not y_ranges:
      y_ranges.append((y_min, y_max, element_name))
  else:
    body_bounds = _find_body_spine_bounds(state, x_center, tol)
    if body_bounds is None:
      raise ValueError(
          f'fold_spine_3d: base has no central body river along x={x_center}'
      )
    y0, y1, b_name = body_bounds
    y_ranges.append((y0, y1, b_name))

  pct_cands = _dedup_floats(
      [fold_percent, fold_percent * 0.85, fold_percent * 0.7, 0.5, 0.35, 0.2]
  )
  last_err = 'no candidate folded'

  for cand_ymin, cand_ymax, range_label in y_ranges:

    def _is_spine_crease(c: Any) -> bool:
      if abs(c.p1.x - x_center) < 1e-4 and abs(c.p2.x - x_center) < 1e-4:
        c_min = min(c.p1.y, c.p2.y)
        c_max = max(c.p1.y, c.p2.y)
        return (c_min >= cand_ymin - tol) and (c_max <= cand_ymax + tol)
      return False

    spine_creases = [c for c in state.pattern.creases if _is_spine_crease(c)]

    for pct in pct_cands:
      for preserve_folds in (True, False):
        if spine_creases:
          new_creases = []
          for c in state.pattern.creases:
            if _is_spine_crease(c):
              if preserve_folds and c.fold in (
                  crease_utils.Fold.MOUNTAIN,
                  crease_utils.Fold.VALLEY,
              ):
                fld = c.fold
              else:
                fld = target_fold
              new_creases.append(
                  crease_utils.Crease(c.p1, c.p2, fold=fld, fold_percent=pct)
              )
            else:
              new_creases.append(c)
          num_spine = len(spine_creases)
        else:
          y_pts = sorted({
              float(v.point.y)
              for v in state.pattern.vertices
              if (
                  abs(float(v.point.x) - x_center) < 1e-4
                  and cand_ymin - tol <= float(v.point.y) <= cand_ymax + tol
              )
          })
          if len(y_pts) < 2:
            y_pts = [cand_ymin, cand_ymax]
          curr_fold = target_fold if preserve_folds else opposite_fold
          new_creases = list(state.pattern.creases)
          for i in range(len(y_pts) - 1):
            p1 = crease_utils.point.Point(
                x=x_center, y=y_pts[i], require_on_paper=False
            )
            p2 = crease_utils.point.Point(
                x=x_center, y=y_pts[i + 1], require_on_paper=False
            )
            new_creases.append(
                crease_utils.Crease(p1, p2, fold=curr_fold, fold_percent=pct)
            )
          num_spine = len(y_pts) - 1

        cand_pattern = crease_pattern_lib.CreasePattern(
            creases=new_creases,
            light_mode=True,
            tolerance=state.pattern.tolerance,
        )

        if verify_fold:
          ok, msg = shaping_fold3d.folds(cand_pattern)
          if not ok:
            last_err = f'{range_label}: {msg}'
            continue
          m = _fold_model(cand_pattern)
          z_relief = float(
              m.vertices_3d[:, 2].max() - m.vertices_3d[:, 2].min()
          )
        else:
          z_relief = float(1.0 - pct) * 0.25

        delta_c = len(cand_pattern.creases) - len(state.pattern.creases)
        LAST_REPORT.element_name = range_label
        LAST_REPORT.num_new_creases = delta_c
        LAST_REPORT.details = {
            'fold_percent': pct,
            'fold_type': fold_type,
            'x_center': x_center,
            'y_min': cand_ymin,
            'y_max': cand_ymax,
            'z_relief': z_relief,
            'num_spine_creases': num_spine,
        }
        LAST_REPORT.summary = (
            f'fold_spine_3d({range_label}): activated {num_spine} spine creases'
            f' at x={x_center:.3f}, fold_percent={pct:.2f},'
            f' z_relief={z_relief:.3f}; {delta_c:+d} creases.'
        )
        prior_bb = getattr(state, 'flat_backbone', None)
        if prior_bb is None and fidelity.has_3d_overlay(cand_pattern):
          if state.pattern.local_flat_foldability():
            prior_bb = fidelity.as_flat_backbone(state.pattern)
        return state.replace(pattern=cand_pattern, flat_backbone=prior_bb)

  raise ValueError(
      f'fold_spine_3d failed for elements along x={x_center}: {last_err}'
  )


# ------------------------------------------------------------------------------
# 7. Spread parallel limbs / jaws (`spread_fork`).
# ------------------------------------------------------------------------------


def _compose_2d_patterns(base_pattern: Any, pat_a: Any, pat_b: Any) -> Any:
  """Composes two 2-D crease pattern modifications into one combined pattern."""

  def _ck(c: Any) -> tuple[float, float, float, float]:
    return (
        round(float(c.p1.x), 4),
        round(float(c.p1.y), 4),
        round(float(c.p2.x), 4),
        round(float(c.p2.y), 4),
    )

  dict_base = {_ck(c): c for c in base_pattern.creases}
  dict_a = {_ck(c): c for c in pat_a.creases}
  dict_b = {_ck(c): c for c in pat_b.creases}

  combined_creases = []
  for k, c in dict_a.items():
    if k not in dict_base:
      combined_creases.append(c)
  for k, c in dict_b.items():
    if k not in dict_base:
      combined_creases.append(c)
  for k, c in dict_base.items():
    ca = dict_a.get(k)
    cb = dict_b.get(k)
    if ca is not None and (
        ca.fold != c.fold or not math.isclose(ca.fold_percent, c.fold_percent)
    ):
      combined_creases.append(ca)
    elif cb is not None and (
        cb.fold != c.fold or not math.isclose(cb.fold_percent, c.fold_percent)
    ):
      combined_creases.append(cb)
    else:
      combined_creases.append(c)

  return crease_pattern_lib.CreasePattern(
      creases=combined_creases,
      light_mode=True,
      tolerance=base_pattern.tolerance,
  )


def spread_fork(
    state: Any,
    *,
    element_a: str,
    element_b: str,
    spread_angle: float = 35.0,
    depth: float = 0.2,
    narrow: bool = False,
    fold_percent: float = 1.0,
    fold_percent_b: float | None = None,
    out_of_plane: bool = False,
    verify_fold: bool = True,
) -> Any:
  """Spreads parallel limbs/pincers into a visible V-shaped open gap.

  Supports true 3-D out-of-plane V-jaw opening (e.g. for crab dactyl and pollex)
  via differential `fold_percent` or `out_of_plane=True`.

  Args:
    state: Current ShapingState.
    element_a: First flap/segment name (e.g. 'dactyl').
    element_b: Second flap/segment name (e.g. 'pollex').
    spread_angle: Lateral bend angle in degrees (default 35.0).
    depth: Cut depth fraction along limb center line (default 0.2).
    narrow: Whether to pre-narrow limbs before spreading.
    fold_percent: Crease fold fraction for element_a (1.0 = flat).
    fold_percent_b: Optional crease fold fraction for element_b.
    out_of_plane: Whether to create 3-D vertical daylight between jaws (sets
      fold_percent_a=0.75, fold_percent_b=1.0).
    verify_fold: Keep only candidates that fold in the deterministic folder.

  Returns:
    New ShapingState with widened tip distance and/or vertical Z separation.
  """
  LAST_REPORT.tool = 'spread_fork'
  LAST_REPORT.element_name = f'{element_a}+{element_b}'
  LAST_REPORT.summary = ''
  LAST_REPORT.details = {}

  if element_a not in state.tree or element_b not in state.tree:
    raise ValueError(
        f'Both {element_a} and {element_b} must be present in stick figure.'
    )

  if out_of_plane and math.isclose(fold_percent, 1.0):
    fp_a = 0.75
    fp_b = 1.0 if fold_percent_b is None else fold_percent_b
  else:
    fp_a = fold_percent
    fp_b = fold_percent if fold_percent_b is None else fold_percent_b

  prior_backbone = getattr(state, 'flat_backbone', None)
  base_st = (
      state.replace(pattern=prior_backbone, flat_backbone=None)
      if prior_backbone is not None
      else state
  )
  if narrow:
    for el in (element_a, element_b):
      try:
        base_st = simple_fold_narrow(
            base_st, element_name=el, verify_fold=False
        )
      except Exception:  # pylint: disable=broad-exception-caught
        pass

  topo = fidelity.topology(state.tree)
  chain_a = _chain_from_element(base_st, element_a)
  chain_b = _chain_from_element(base_st, element_b)
  tip_joint_a = topo.distal.get(chain_a[-1], '')
  tip_joint_b = topo.distal.get(chain_b[-1], '')
  layout = state.paper_layout
  to_tree_units = layout.grid_size / max(layout.grid_units_per_tree_unit, 1e-6)

  def _measure_tip_distance(
      pat: crease_pattern_lib.CreasePattern,
  ) -> float | None:
    if tip_joint_a not in layout.joints or tip_joint_b not in layout.joints:
      return None
    try:
      model = fidelity.FoldedModel.from_pattern(pat)
      if model.vertex_error > 1.0:
        return None
      means, _ = model.locate_groups({
          tip_joint_a: layout.joints[tip_joint_a],
          tip_joint_b: layout.joints[tip_joint_b],
      })
      if tip_joint_a not in means or tip_joint_b not in means:
        return None
      return float(
          np.linalg.norm(means[tip_joint_a] - means[tip_joint_b])
          * to_tree_units
      )
    except Exception:  # pylint: disable=broad-exception-caught
      return None

  measured_before = (
      _measure_tip_distance(state.pattern) if verify_fold else None
  )
  dist_before = measured_before if measured_before is not None else 0.05

  def _predict_spread_sign(elem: str, other: str) -> float:
    try:
      gd = getattr(base_st, 'graph_data', None)
      if gd is None:
        return 1.0
      poly_e = shaping_fold3d.element_polygon(base_st, elem)
      ref_e = shaping_fold3d.reference_point(base_st, elem, gd)
      axis_e = (
          shaping_fold3d.geometric_fold_3d_heuristics.get_element_folded_axis(
              gd, poly_e, ref_e
          )
      )
      pts_e = shaping_fold3d.geometric_fold_3d_heuristics.element_folded_points(
          gd, poly_e
      )
      c_e = pts_e.mean(axis=0) if len(pts_e) else axis_e.base

      poly_o = shaping_fold3d.element_polygon(base_st, other)
      pts_o = shaping_fold3d.geometric_fold_3d_heuristics.element_folded_points(
          gd, poly_o
      )
      if len(pts_o):
        c_o = pts_o.mean(axis=0)
      else:
        ref_o = shaping_fold3d.reference_point(base_st, other, gd)
        axis_o = (
            shaping_fold3d.geometric_fold_3d_heuristics.get_element_folded_axis(
                gd, poly_o, ref_o
            )
        )
        c_o = axis_o.base

      dir_e = axis_e.direction
      left_norm_e = np.array([-dir_e[1], dir_e[0]])
      rel = c_o - c_e
      side = np.dot(rel, left_norm_e)
      if abs(side) > 1e-4:
        return -1.0 if side > 0 else 1.0
    except Exception:  # pylint: disable=broad-exception-caught
      pass
    return 1.0

  def _try_single(
      st: Any,
      elem: str,
      ang: float,
      fn: Any,
      d: float,
      fp: float,
  ) -> Any | None:
    if math.isclose(ang, 0.0):
      return st
    try:
      return fn(
          st,
          element_name=elem,
          angle=ang,
          depth=d,
          fold_percent=fp,
          verify_fold=False,
      )
    except Exception:  # pylint: disable=broad-exception-caught
      return None

  def _try_limb_all(
      st: Any,
      elem: str,
      cand_angles: Sequence[float],
      fp: float,
  ) -> list[tuple[Any, float]]:
    found: list[tuple[Any, float]] = []
    d_cands = [depth] if depth >= 0.35 else [depth, 0.4]
    for cand_a in cand_angles:
      got_for_sign = False
      for d_val in d_cands:
        for fn in (
            shaping_fold3d.crimp_fold_3d,
            shaping_fold3d.reverse_fold_3d,
        ):
          res = _try_single(st, elem, cand_a, fn, d_val, fp)
          if res is not None:
            found.append((res, cand_a))
            got_for_sign = True
            break
        if got_for_sign:
          break
      if got_for_sign:
        continue
      for d_val in d_cands:
        try:
          import sys  # pylint: disable=g-import-not-at-top

          shaping_mod = sys.modules.get(
              'bp_pipeline.shaping.toolbox'
          )
          if shaping_mod is not None:
            res_st = shaping_mod.simple_fold(
                st,
                element_name=elem,
                angle=cand_a,
                depth=d_val,
                fold_percent=fp,
            )
            pct, _, _ = res_st.pattern.vertex_flat_foldability_percentage()
            if pct >= 0.90:
              found.append((res_st, cand_a))
              break
        except Exception:  # pylint: disable=broad-exception-caught
          pass
    return found

  sign_a = _predict_spread_sign(element_a, element_b)
  cands_a = [sign_a * spread_angle, -sign_a * spread_angle]

  sign_b = _predict_spread_sign(element_b, element_a)
  cands_b = [sign_b * spread_angle, -sign_b * spread_angle]

  results_a = _try_limb_all(base_st, element_a, cands_a, fp_a)
  results_b = _try_limb_all(base_st, element_b, cands_b, fp_b)

  def _finalize_candidate(
      cand_2d_pattern: crease_pattern_lib.CreasePattern,
      narrowed: tuple[str, ...],
  ) -> tuple[Any | None, float]:
    pct, _, _ = cand_2d_pattern.vertex_flat_foldability_percentage()
    if pct < 0.90:
      return None, -1.0
    if prior_backbone is not None:
      final_pat = fidelity.replay_3d_overlay(
          prior_backbone, state.pattern, cand_2d_pattern
      )
      ok_ff, _ = fidelity.preserves_foldability(state.pattern, final_pat)
      if not ok_ff:
        return None, -1.0
    else:
      final_pat = cand_2d_pattern
    if verify_fold:
      if not shaping_fold3d.folds(final_pat)[0]:
        return None, -1.0
      d_after = _measure_tip_distance(final_pat)
      if d_after is None:
        return None, -1.0
    else:
      d_after = dist_before + 0.10
    next_bb = (
        fidelity.as_flat_backbone(cand_2d_pattern)
        if fidelity.has_3d_overlay(final_pat)
        else None
    )
    out_st = base_st.replace(
        pattern=final_pat,
        narrowed_elements=narrowed,
        flat_backbone=next_bb,
    )
    return out_st, d_after

  best_state = None
  best_dist = -1.0
  best_angles = (0.0, 0.0)
  last_err = 'no candidate succeeded'

  # Prefer 2-limb combinations maximizing 3-D tip separation.
  pair_scored: list[tuple[float, int, Any, tuple[float, float]]] = []
  rank_idx = 0
  for st_a, ang_a_used in results_a:
    # Try opposite sign on B first when A succeeded, then same sign.
    opp_b = [item for item in results_b if item[1] * ang_a_used < 0]
    same_b = [item for item in results_b if item[1] * ang_a_used >= 0]
    ordered_b = opp_b + same_b
    for st_b, ang_b_used in ordered_b:
      rank_idx += 1
      comb_pattern = _compose_2d_patterns(
          base_st.pattern, st_a.pattern, st_b.pattern
      )
      narrowed_union = tuple(
          dict.fromkeys((*st_a.narrowed_elements, *st_b.narrowed_elements))
      )
      fin_st, d_val = _finalize_candidate(comb_pattern, narrowed_union)
      if fin_st is not None:
        pair_scored.append((d_val, -rank_idx, fin_st, (ang_a_used, ang_b_used)))
        if not verify_fold:
          break
    if pair_scored and not verify_fold:
      break

  if pair_scored:
    best_dist, _, best_state, best_angles = max(
        pair_scored, key=lambda item: (round(item[0], 4), item[1])
    )
  else:
    single_scored: list[tuple[float, int, Any, tuple[float, float]]] = []
    for st_a, ang_a_used in results_a:
      rank_idx += 1
      fin_st, d_val = _finalize_candidate(st_a.pattern, st_a.narrowed_elements)
      if fin_st is not None:
        single_scored.append((d_val, -rank_idx, fin_st, (ang_a_used, 0.0)))
    for st_b, ang_b_used in results_b:
      rank_idx += 1
      fin_st, d_val = _finalize_candidate(st_b.pattern, st_b.narrowed_elements)
      if fin_st is not None:
        single_scored.append((d_val, -rank_idx, fin_st, (0.0, ang_b_used)))
    if single_scored:
      best_dist, _, best_state, best_angles = max(
          single_scored, key=lambda item: (round(item[0], 4), item[1])
      )

  if best_state is None:
    raise ValueError(
        f'spread_fork({element_a}, {element_b}) failed: {last_err}'
    )

  delta_c = len(best_state.pattern.creases) - len(state.pattern.creases)
  ratio = best_dist / max(dist_before, 1e-6)
  LAST_REPORT.num_new_creases = delta_c
  LAST_REPORT.details = {
      'element_a': element_a,
      'element_b': element_b,
      'dist_before': dist_before,
      'dist_after': best_dist,
      'ratio': ratio,
      'angles': best_angles,
      'out_of_plane': out_of_plane,
      'fold_percent_a': fp_a,
      'fold_percent_b': fp_b,
  }
  oop_str = (
      f', out_of_plane fp=({fp_a:.2f}, {fp_b:.2f})'
      if (out_of_plane or fp_a < 1.0 or fp_b < 1.0)
      else ''
  )
  LAST_REPORT.summary = (
      f'spread_fork({element_a}, {element_b}): tip distance {dist_before:.3f}'
      f' -> {best_dist:.3f} ({ratio:.2f}x){oop_str}; +{delta_c} creases.'
  )
  return best_state


def _circular_mean_deg(angles_deg: Sequence[float]) -> float:
  """Circular mean of angles in degrees, returned in `(-180, 180]`."""
  if not angles_deg:
    return 0.0
  rads = [math.radians(a) for a in angles_deg]
  sin_sum = sum(math.sin(r) for r in rads)
  cos_sum = sum(math.cos(r) for r in rads)
  if abs(sin_sum) < 1e-9 and abs(cos_sum) < 1e-9:
    return float(angles_deg[0])
  deg = math.degrees(math.atan2(sin_sum, cos_sum))
  return 180.0 if deg == -180.0 else deg


def radial_fan(
    state: Any,
    *,
    elements: Sequence[str],
    arc_deg: float | None = None,
    center_az: float | None = None,
    base_el: float | None = None,
    dome_el: float = 0.0,
    fan_wide: bool = True,
    open_percent: float = 0.65,
    verify_fold: bool = True,
) -> Any:
  """Arranges sibling rays along a smooth circular arc and fans broad rays.

  For feathered wings, scalloped frills, and multi-ray tails, posing multiple
  sibling flaps along a circular `(az, el)` arc via `rotation_3d` (followed by
  optional 100%-flat-foldable `squash_fold`/`fan_out_fold_heuristic` on broad
  rays) achieves genuine 60-140 degree circular spread without breaking
  Kawasaki flat-foldability.

  Args:
    state: Current `ShapingState`.
    elements: Ordered sequence of >= 2 ray element names in `state.tree`.
    arc_deg: Optional total azimuth sweep in degrees across `elements`. When
      None, uses each element's stick-figure azimuth.
    center_az: Optional center azimuth in degrees when `arc_deg` is given;
      defaults to the circular mean of `elements`' stick-figure azimuths.
    base_el: Optional baseline elevation in degrees; defaults to each element's
      stick-figure elevation.
    dome_el: Additional circular dome elevation boost at the arc midpoint.
    fan_wide: Whether to also expand rays of `width >= 2` with `squash_fold` or
      `fan_out_fold_heuristic`.
    open_percent: Crease `fold_percent` for wide-ray expansion (`1.0` keeps
      strict 2-D Kawasaki flat-foldability).
    verify_fold: Whether to verify folder convergence and low vertex strain.

  Returns:
    Updated `ShapingState`.

  Raises:
    ValueError: If fewer than 2 valid elements are supplied or shaping fails.
  """
  import sys  # pylint: disable=g-import-not-at-top

  ray_names = [str(name) for name in elements]
  LAST_REPORT.tool = 'radial_fan'
  LAST_REPORT.element_name = ','.join(ray_names)
  LAST_REPORT.summary = ''
  LAST_REPORT.num_new_creases = 0
  LAST_REPORT.details = {}

  if len(ray_names) < 2:
    raise ValueError('radial_fan requires at least 2 ray elements.')
  for name in ray_names:
    if name not in state.tree:
      raise ValueError(f'radial_fan: unknown element {name!r}.')
    if state.tree[name].is_river:
      raise ValueError(f'radial_fan: {name!r} is a river, expected flap rays.')

  shaping_mod = sys.modules.get(
      'bp_pipeline.shaping.toolbox'
  )
  if shaping_mod is None:
    raise ValueError('radial_fan requires shaping module to be loaded.')

  count = len(ray_names)
  sf_azs = [float(state.tree[n].azimuth_angle) for n in ray_names]
  sf_els = [float(state.tree[n].elevation_angle) for n in ray_names]
  c_az = _circular_mean_deg(sf_azs) if center_az is None else float(center_az)

  rotations: list[dict[str, Any]] = []
  for idx, name in enumerate(ray_names):
    frac = idx / (count - 1) - 0.5
    if arc_deg is None and center_az is None:
      az_val = sf_azs[idx]
    else:
      span = 90.0 if arc_deg is None else float(arc_deg)
      az_val = ((c_az + frac * span + 180.0) % 360.0) - 180.0
    el_base = sf_els[idx] if base_el is None else float(base_el)
    el_val = max(
        -89.0,
        min(89.0, el_base + float(dome_el) * math.cos(math.pi * frac)),
    )
    rotations.append({
        'element_name': name,
        'az': round(az_val, 2),
        'el': round(el_val, 2),
    })

  def _is_acceptable(cand_state: Any) -> bool:
    pct, _, non_ff = cand_state.pattern.vertex_flat_foldability_percentage()
    if non_ff or not math.isclose(pct, 1.0):
      return False
    if verify_fold:
      ok_f, _ = shaping_fold3d.folds(cand_state.pattern)
      if not ok_f:
        return False
      verr = fidelity.FoldedModel.from_pattern(cand_state.pattern).vertex_error
      if verr > fidelity.CLEAN_GATE_MAX_VERTEX_ERROR:
        return False
    return True

  # Strategy 1: pose arc with rotation_3d first, then expand wide rays.
  # Strategy 2: expand wide rays on base first, then pose arc with rotation_3d.
  cur_state = shaping_mod.rotation_3d(state, rotations=rotations)
  if not _is_acceptable(cur_state):
    raise ValueError(
        'radial_fan: rotation_3d arc failed foldability or strain gate.'
    )

  expanded_rays: list[str] = []
  if fan_wide:
    open_pct_cands = [float(open_percent)]
    if not math.isclose(open_percent, 1.0, abs_tol=1e-5):
      open_pct_cands.append(1.0)
    for name in ray_names:
      if float(state.tree[name].width) < 2.0:
        continue
      ray_done = False
      for op_pct in open_pct_cands:
        for tool_fn, kwargs in (
            (
                shaping_mod.fan_out_fold_heuristic,
                {
                    'element_name': name,
                    'move_delta': 1.0,
                    'open_percent': op_pct,
                },
            ),
            (
                shaping_mod.fan_out_fold_heuristic,
                {
                    'element_name': name,
                    'move_delta': -1.0,
                    'open_percent': op_pct,
                },
            ),
            (
                shaping_mod.squash_fold,
                {'element_name': name, 'open_percent': op_pct},
            ),
        ):
          try:
            cand = tool_fn(cur_state, **kwargs)
          except Exception:  # pylint: disable=broad-exception-caught
            continue
          if _is_acceptable(cand):
            cur_state = cand
            expanded_rays.append(name)
            ray_done = True
            break
        if ray_done:
          break

  delta_c = len(cur_state.pattern.creases) - len(state.pattern.creases)
  LAST_REPORT.num_new_creases = delta_c
  LAST_REPORT.details = {
      'rotations': rotations,
      'expanded_rays': expanded_rays,
      'open_percent': open_percent,
  }
  LAST_REPORT.summary = (
      f'radial_fan({len(ray_names)} rays, expanded={expanded_rays}):'
      f' +{delta_c} creases.'
  )
  return cur_state


def open_river_3d(
    state: Any,
    *,
    element_name: str,
    fold_percent: float = 0.65,
    max_vertex_error: float = 0.35,
    verify_fold: bool = True,
) -> Any:
  """Partially opens interior accordion pleats of a wide river for 3-D volume.

  Leaves perimeter hinges and paper borders untouched while opening interior
  parallel river pleats along `element_name` to `fold_percent`, rejecting any
  candidate that violates 3-D Belcastro-Hull foldability or exceeds
  `max_vertex_error`.

  Args:
    state: Current `ShapingState`.
    element_name: River element name in `state.tree`.
    fold_percent: Target `fold_percent` in `(0.15, 0.95]` for interior pleats.
    max_vertex_error: Maximum allowed deterministic folder vertex error (%).
    verify_fold: Whether to verify folder convergence and `max_vertex_error`.

  Returns:
    Updated `ShapingState`.

  Raises:
    ValueError: If `element_name` is not a river or no clean 3-D opening folds.
  """
  LAST_REPORT.tool = 'open_river_3d'
  LAST_REPORT.element_name = element_name
  LAST_REPORT.summary = ''
  LAST_REPORT.num_new_creases = 0
  LAST_REPORT.details = {}

  elem = state.element(element_name)
  if not isinstance(elem, shaping_elements.ShapingRiver):
    raise ValueError(f'open_river_3d: {element_name!r} is not a river.')
  rects = list(elem.covering_legacy_rectangles)
  if not rects:
    raise ValueError(f'open_river_3d: river {element_name!r} has no rectangles.')

  scale = float(state.solving_grid_size)
  boxes = [
      (
          max(0.0, min(r.x_start, r.x_start + r.x_len)),
          max(0.0, min(r.y_start, r.y_start + r.y_len)),
          min(1.0, max(r.x_start, r.x_start + r.x_len)),
          min(1.0, max(r.y_start, r.y_start + r.y_len)),
      )
      for r in rects
  ]
  margin = 0.25 / max(scale, 1.0)

  def _strictly_inside_river(c: crease_utils.Crease) -> bool:
    if not c.mountain_or_valley():
      return False
    # Axis-aligned accordion pleats only (preserve 45-deg miter hinges!).
    dx = abs(float(c.p2.x - c.p1.x))
    dy = abs(float(c.p2.y - c.p1.y))
    if dx > 1e-5 and dy > 1e-5:
      return False
    mx = 0.5 * float(c.p1.x + c.p2.x)
    my = 0.5 * float(c.p1.y + c.p2.y)
    for x0, y0, x1, y1 in boxes:
      if (x0 + margin <= mx <= x1 - margin) and (
          y0 + margin <= my <= y1 - margin
      ):
        return True
    return False

  pct_ladder = [float(fold_percent)]
  for fallback_pct in (0.75, 0.85, 0.92):
    if fallback_pct > fold_percent and fallback_pct not in pct_ladder:
      pct_ladder.append(fallback_pct)

  prior_bb = state.flat_backbone or fidelity.as_flat_backbone(state.pattern)
  last_err = 'no interior river accordion creases found'
  for pct_try in pct_ladder:
    modified = 0
    new_creases: list[crease_utils.Crease] = []
    for c in state.pattern.creases:
      if _strictly_inside_river(c):
        new_creases.append(dataclasses.replace(c, fold_percent=pct_try))
        modified += 1
      else:
        new_creases.append(c)
    if modified == 0:
      break
    cand_pat = crease_pattern_lib.CreasePattern(
        creases=new_creases,
        light_mode=True,
        tolerance=state.pattern.tolerance,
    )
    if not _all_internal_vertices_3d_foldable(cand_pat):
      last_err = f'Belcastro-Hull failed at fold_percent={pct_try:g}'
      continue
    verr = 0.0
    if verify_fold:
      ok_f, reason = shaping_fold3d.folds(cand_pat)
      if not ok_f:
        last_err = reason
        continue
      verr = fidelity.FoldedModel.from_pattern(cand_pat).vertex_error
      if verr > max_vertex_error:
        last_err = (
            f'vertex_error={verr:.3f}% exceeds max_vertex_error='
            f'{max_vertex_error:.3f}%'
        )
        continue
    out_state = state.replace(pattern=cand_pat, flat_backbone=prior_bb)
    LAST_REPORT.details = {
        'modified_creases': modified,
        'fold_percent': pct_try,
        'vertex_error_percent': round(verr, 4),
    }
    LAST_REPORT.summary = (
        f'open_river_3d({element_name}): opened {modified} pleats at'
        f' fold_percent={pct_try:g} (strain={verr:.3f}%).'
    )
    return out_state

  raise ValueError(f'open_river_3d({element_name}) failed: {last_err}.')


# ------------------------------------------------------------------------------
# 8. Symmetric narrowing of a flap spanning the central crease
#    (`narrow_central_flap`).
# ------------------------------------------------------------------------------
#
# After central-crease removal a flap whose tip lies on x = 0.5 (e.g. a cobra
# head on the axis) is no longer a doubled flap: its left paper half folds to
# one side of the flap axis and its right half to the other, so the folded flap
# is twice as wide and the zig-zag narrowing machinery (which narrows one
# paper-side flap towards its central ridge) only ever acts on one half (see
# the TODO in `zigzag_shaping_utils._get_flap_zigzag_params_from_heuristics`).
#
# This tool narrows the whole flap in the 2-D folded form instead: it folds the
# two lateral edges of the flap inward along two lines that are mirror images
# about the flap axis, each running from a hinge corner of the flap (or a point
# `start_offset` grid units further into the parent) to the far end of the flap.
# Each line is a full simple fold through every layer it crosses and runs from
# outline to outline of the folded flap, and the two lines never cross inside
# the paper, so the result is flat-foldable by construction:
#   * where a line meets a pleat fold edge of the stack, the vertex has the
#     straight pleat crease plus two new segments that are mirror images about
#     it, with opposite M/V (the adjacent layers have opposite orientation):
#     Kawasaki (alternating angles theta, 180 - theta, 180 - theta, theta) and
#     Maekawa (|M - V| = 2) hold;
#   * vertices on x = 0.5 receive the left- and right-half images of the two
#     mirrored lines symmetrically, so no degree-2 or odd vertex is left on the
#     axis, and nothing is added at the flap/parent junction except on pleat
#     edges.
# The new creases are mirror-symmetric about x = 0.5 in the crease pattern
# (checked and reported as `mirror_match`).

_CENTRAL_FLAP_SAMPLE_STEP = 0.25  # Grid units between sample points.
_CENTRAL_FLAP_MIN_MIRROR_MATCH = 0.95


def _linf_distance_to_tip(
    points: np.ndarray, flap: shaping_elements.ShapingFlap
) -> np.ndarray:
  """L-inf distance (grid units) from unit-paper points to a flap's tip.

  The tip is the axis-aligned segment between the flap's circle centers (a
  point for single-center flaps), so the nearest tip point is the clamp of the
  query point into the centers' bounding box.

  Args:
    points: (N, 2) unit-paper points.
    flap: The flap.

  Returns:
    (N,) distances in grid units.
  """
  centers = np.asarray([[c.x, c.y] for c in flap.circle_centers], dtype=float)
  clamped = np.clip(points, centers.min(axis=0), centers.max(axis=0))
  return np.abs(points - clamped).max(axis=1) * flap.grid_size


def _points_in_polygon(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
  """Even-odd point-in-polygon test (boundary points count as inside)."""
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


def _folded_flat_positions(graph_data: Any, points: np.ndarray) -> np.ndarray:
  """2-D folded-form positions of unit-paper points (NaN where unmapped).

  Each face of the flat folded state is an isometry of the paper; it is fitted
  by least squares on the face's vertices and applied to the points the face
  contains.

  Args:
    graph_data: `geometric_simple_fold.prepare_graph_data` output.
    points: (N, 2) unit-paper points.

  Returns:
    (N, 2) folded positions.
  """
  vertices = np.asarray(graph_data['vertices_coords'], dtype=float)
  folded = graph_data['folded_vertices']
  result = np.full((len(points), 2), np.nan)
  todo = np.ones(len(points), dtype=bool)
  for face in graph_data['face_vertices']:
    if len(face) < 3 or any(folded[i] is None for i in face):
      continue
    src = vertices[face]
    lo, hi = src.min(axis=0) - 1e-9, src.max(axis=0) + 1e-9
    candidates = np.flatnonzero(
        todo
        & (points[:, 0] >= lo[0])
        & (points[:, 0] <= hi[0])
        & (points[:, 1] >= lo[1])
        & (points[:, 1] <= hi[1])
    )
    if not len(candidates):  # pylint: disable=g-explicit-length-test
      continue
    inside = _points_in_polygon(points[candidates], src)
    if not inside.any():
      continue
    dst = np.asarray([folded[i][:2] for i in face], dtype=float)
    affine, *_ = np.linalg.lstsq(
        np.c_[src, np.ones(len(face))], dst, rcond=None
    )
    hit = candidates[inside]
    result[hit] = np.c_[points[hit], np.ones(len(hit))] @ affine
    todo[hit] = False
  return result


@dataclasses.dataclass(frozen=True)
class CentralFlapFrame:
  """Folded-form frame of a flap: hinge origin, lateral `u`, axial `v` axes.

  Attributes:
    origin: Folded position of the hinge centre.
    u: Lateral unit vector.
    v: Unit vector from the hinge towards the flap tip.
    s_min: Smallest lateral coordinate of the folded flap.
    s_max: Largest lateral coordinate of the folded flap.
    t_hinge: Axial coordinate of the hinge.
    t_tip: Axial coordinate of the far end of the flap.
  """

  origin: np.ndarray
  u: np.ndarray
  v: np.ndarray
  s_min: float
  s_max: float
  t_hinge: float
  t_tip: float

  @property
  def width(self) -> float:
    """Folded width of the central flap across the symmetry axis."""
    return self.s_max - self.s_min

  def to_folded(self, s: float, t: float) -> tuple[float, float]:
    point = self.origin + s * self.u + t * self.v
    return float(point[0]), float(point[1])


def central_flap_frame(
    flap: shaping_elements.ShapingFlap, graph_data: Any
) -> CentralFlapFrame:
  """Measures a flap's axis and lateral extent in the 2-D folded form.

  Args:
    flap: The flap (any flap; meant for flaps spanning x = 0.5).
    graph_data: Folded-form graph of the current flat pattern.

  Returns:
    The frame.

  Raises:
    ValueError: If the flap cannot be located in the folded form.
  """
  rect = flap.to_legacy_rectangle()
  gs = flap.grid_size
  x0 = max(0.0, min(rect.x_start, rect.x_start + rect.x_len))
  x1 = min(1.0, max(rect.x_start, rect.x_start + rect.x_len))
  y0 = max(0.0, min(rect.y_start, rect.y_start + rect.y_len))
  y1 = min(1.0, max(rect.y_start, rect.y_start + rect.y_len))
  step = _CENTRAL_FLAP_SAMPLE_STEP / gs
  xs = np.arange(x0, x1 + step / 2, step)
  ys = np.arange(y0, y1 + step / 2, step)
  grid_x, grid_y = np.meshgrid(xs, ys)
  points = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)
  radius = flap.grid_unit_radius
  dist = _linf_distance_to_tip(points, flap)
  keep = dist <= radius + 1e-6
  points, dist = points[keep], dist[keep]
  folded = _folded_flat_positions(graph_data, points)
  ok = ~np.isnan(folded).any(axis=1)
  folded, dist = folded[ok], dist[ok]
  if len(folded) < 8:
    raise ValueError(f'{flap.name}: flap not found in the folded form.')
  tip_band = dist <= min(0.5, radius / 4) + 1e-6
  hinge_band = dist >= radius - min(0.5, radius / 4) - 1e-6
  if not tip_band.any() or not hinge_band.any():
    raise ValueError(f'{flap.name}: cannot locate the flap tip and hinge.')
  origin = folded[hinge_band].mean(axis=0)
  axis = folded[tip_band].mean(axis=0) - origin
  if np.linalg.norm(axis) < 1e-9:
    raise ValueError(f'{flap.name}: degenerate folded flap axis.')
  v = axis / np.linalg.norm(axis)
  u = np.array([-v[1], v[0]])
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
  """The two mirrored taper fold lines (folded-form coordinates)."""
  half = frame.width / 2
  s_c = (frame.s_min + frame.s_max) / 2
  t_start = frame.t_hinge - start_offset_units / grid_size
  lines = []
  for side in (-1.0, 1.0):
    start = frame.to_folded(s_c + side * half, t_start)
    end = frame.to_folded(s_c + side * half * tip_width, frame.t_tip)
    lines.append((start, end))
  return lines


def _mirror_match(
    creases: Sequence[crease_utils.Crease], axis_x: float = 0.5
) -> float:
  """Fraction of creases whose mirror image about x = axis_x is also present."""
  if not creases:
    return 0.0

  def key(x1, y1, x2, y2):
    a, b = (round(x1, 5), round(y1, 5)), (round(x2, 5), round(y2, 5))
    return (a, b) if a <= b else (b, a)

  keys = {key(c.p1.x, c.p1.y, c.p2.x, c.p2.y) for c in creases}
  hits = sum(
      key(2 * axis_x - c.p1.x, c.p1.y, 2 * axis_x - c.p2.x, c.p2.y) in keys
      for c in creases
  )
  return hits / len(creases)


def narrow_central_flap(
    state: Any,
    *,
    element_name: str,
    tip_width: float = 0.25,
    start_offset: float = 0.0,
    fold_type: str = 'auto',
    verify_fold: bool = True,
    require_symmetric: bool = True,
) -> Any:
  """Narrows both halves of a flap that spans the central crease, symmetrically.

  Folds the two lateral edges of the flap inward along two mirrored lines in
  the folded form (a taper from the flap's hinge corners to its far end); see
  the section comment above for why the result is flat-foldable. Use it before
  3-D tools (`rotation_3d`, `crimp_fold_3d`, ...) on axis flaps such as heads
  after central-crease removal, where `simple_fold(narrow=True)` and
  `crimp_fold_3d` only act on one half of the flap.

  Args:
    state: Current `ShapingState`.
    element_name: The flap (typically spanning x = 0.5).
    tip_width: Fraction of the folded flap width kept at its far end, in [0, 1)
      (0 = pointed spade tip, 0.5 = half width).
    start_offset: Grid units into the parent element where the taper starts (0 =
      at the flap's hinge corners, keeping the parent untouched; > 0 also
      narrows the flap's base at the cost of a small triangle of the parent).
    fold_type: 'valley', 'mountain' (in the folded form; both sides use the same
      type) or 'auto' (valley, then mountain).
    verify_fold: Keep only patterns that fold in the deterministic folder.
    require_symmetric: Reject results whose new creases are not mirror images
      about x = 0.5 (only checked for flaps spanning x = 0.5).

  Returns:
    New `ShapingState` (the flap is not added to `narrowed_elements`, whose
    grid-doubling semantics belong to zig-zag narrowing).

  Raises:
    ValueError: On invalid arguments or when no candidate is flat-foldable.
  """
  LAST_REPORT.tool = 'narrow_central_flap'
  LAST_REPORT.element_name = element_name
  LAST_REPORT.summary = ''
  LAST_REPORT.num_new_creases = 0
  LAST_REPORT.details = {}

  if not 0.0 <= tip_width < 1.0:
    raise ValueError(f'tip_width must be in [0, 1), got {tip_width}.')
  if start_offset < 0.0:
    raise ValueError(f'start_offset must be >= 0, got {start_offset}.')
  if fold_type not in ('auto', 'valley', 'mountain'):
    raise ValueError(
        f'fold_type must be auto/valley/mountain, got {fold_type!r}.'
    )
  element = state.element(element_name)
  if not isinstance(element, shaping_elements.ShapingFlap):
    raise ValueError(
        f'{element_name} is a river; narrow_central_flap needs a flap.'
    )
  if element.grid_unit_radius <= 1.0 + 1e-6:
    raise ValueError(f'{element_name} has length 1; nothing to narrow.')
  if fidelity.has_3d_overlay(state.pattern) and state.flat_backbone is None:
    raise ValueError(
        'narrow_central_flap needs a 2-D flat pattern; apply it before 3-D'
        ' tools (rotation_3d, crimp_fold_3d, ...).'
    )

  pattern = state.active_flat_pattern
  graph_data = state.graph_data
  frame = central_flap_frame(element, graph_data)
  gs = element.grid_size
  lines = central_flap_taper_lines(
      frame, tip_width=tip_width, start_offset_units=start_offset, grid_size=gs
  )
  limiting: list[Any] = [element.to_legacy_rectangle()]
  if start_offset > 0:
    parent = fidelity.topology(state.tree).parent.get(element_name)
    if parent is None:
      raise ValueError(f'{element_name} has no parent for start_offset > 0.')
    parent_element = state.element(parent)
    if isinstance(parent_element, shaping_elements.ShapingFlap):
      limiting.append(parent_element.to_legacy_rectangle())
    else:
      limiting.extend(parent_element.covering_legacy_rectangles)
  spans_axis = bool(element.contains_central_crease)

  fold_types = ('valley', 'mountain') if fold_type == 'auto' else (fold_type,)
  folds = {
      'valley': crease_utils.Fold.VALLEY,
      'mountain': crease_utils.Fold.MOUNTAIN,
  }
  reasons = []
  for name in fold_types:
    new_creases: list[crease_utils.Crease] = []
    for start, end in lines:
      new_creases.extend(
          geometric_simple_fold_lib.get_geometric_fold_creases(
              crease_pattern=pattern,
              cut_line_start=start,
              cut_line_end=end,
              fold_assignment=folds[name],
              limiting_polygon=limiting,
              graph_data=graph_data,
          )
      )
    if not new_creases:
      reasons.append(f'{name}: no creases (lines miss the flap)')
      continue
    mirror = _mirror_match(new_creases) if spans_axis else None
    if (
        require_symmetric
        and mirror is not None
        and mirror < _CENTRAL_FLAP_MIN_MIRROR_MATCH
    ):
      reasons.append(f'{name}: asymmetric creases (mirror match {mirror:.2f})')
      continue
    try:
      shaped = pattern.update_with_shaping_creases(new_creases)
    except (ValueError, AssertionError, IndexError) as error:
      reasons.append(f'{name}: {error}')
      continue
    new_state, reason = _accept_pattern(
        state, shaped, allow_3d_vertices=False, verify_fold=verify_fold
    )
    if new_state is None:
      reasons.append(f'{name}: {reason}')
      continue
    # Not recorded in `narrowed_elements`: that field means "grid size doubled
    # by zig-zag narrowing" and makes rotation_3d refuse the whole chain, but
    # the taper keeps the flap's hinge width (start_offset=0), so the 3-D
    # templates of later tools still apply.
    delta = len(new_state.pattern.creases) - len(state.pattern.creases)
    # Width of the narrowed flap at its hinge: the taper line runs from
    # s = half at t_hinge - offset to s = half * tip_width at t_tip.
    offset_f = start_offset / gs
    taper_len = frame.t_tip - frame.t_hinge + offset_f
    base_fraction = 1.0 - (1.0 - tip_width) * offset_f / max(taper_len, 1e-9)
    LAST_REPORT.num_new_creases = delta
    LAST_REPORT.details = {
        'fold_type': name,
        'lines': [[list(a), list(b)] for a, b in lines],
        'folded_width': round(frame.width * gs, 3),
        'tip_width_units': round(frame.width * gs * tip_width, 3),
        'base_width_units': round(frame.width * gs * base_fraction, 3),
        'mirror_match': None if mirror is None else round(mirror, 3),
        'spans_central_crease': spans_axis,
    }
    LAST_REPORT.summary = (
        f'{element_name}: symmetric taper ({name}) from width'
        f' {frame.width * gs:.2f} to {frame.width * gs * tip_width:.2f} grid'
        f' units at the tip (start_offset={start_offset}); +{delta} creases,'
        f' mirror match {LAST_REPORT.details["mirror_match"]}.'
    )
    return new_state
  raise ValueError(
      f'narrow_central_flap on {element_name} failed: {"; ".join(reasons)}'
  )


TOOLS: dict[str, Any] = {
    'inscribed_flap': inscribed_flap,
    'clip_pattern_fold': clip_pattern_fold,
    'pleat_to_hide': pleat_to_hide,
    'simple_fold_narrow': simple_fold_narrow,
    'curve_flap': curve_flap,
    'level_shifter': level_shifter,
    'fold_spine_3d': fold_spine_3d,
    'spread_fork': spread_fork,
    'radial_fan': radial_fan,
    'open_river_3d': open_river_3d,
    'narrow_central_flap': narrow_central_flap,
}


def tool_names() -> Sequence[str]:
  return tuple(TOOLS)
