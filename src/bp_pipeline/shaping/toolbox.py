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

"""Stateful 3-D origami shaping actuators and declarative tool dispatch."""

from collections.abc import Callable, Mapping, Sequence
import dataclasses
import itertools
import math
import os
import traceback
from typing import Any

import numpy as np

from .. import folder_3d
from .. import utils_tree
from ..crease_pattern import crease as crease_utils
from ..crease_pattern import crease_pattern as crease_pattern_lib
from ..stick_figures import angle_utils
from ..stick_figures import stick_figure as stick_figure_lib
from . import combined_heuristics
from . import fan_squash
from . import fold_shaping
from . import folded_region_fold
from . import geometric_fold_3d_heuristics
from . import geometric_simple_fold as geometric_simple_fold_lib
from . import heuristics
from . import heuristics_3d
from . import heuristics_utils
from . import paper_layout as paper_layout_lib
from . import relief
from . import shaping_elements
from . import state as state_lib
from . import types as shaping_types

# Public state and inspection API re-exported for callers.
MAX_RENDERABLE_VERTEX_ERROR = state_lib.MAX_RENDERABLE_VERTEX_ERROR
ShapingState = state_lib.ShapingState
ToolConfig = state_lib.ToolConfig
ToolResult = state_lib.ToolResult
check_flat_foldability = state_lib.check_flat_foldability
describe = state_lib.describe
deserialize_pattern_svg = state_lib.deserialize_pattern_svg
element_table = state_lib.element_table
flap_orientation_info = state_lib.flap_orientation_info
history_json = state_lib.history_json
measured_directions = state_lib.measured_directions
preserve_3d_overlay = state_lib.preserve_3d_overlay
resolve_symmetric_element = state_lib.resolve_symmetric_element
serialize_pattern_svg = state_lib.serialize_pattern_svg
verify_3d_fold = state_lib.verify_3d_fold

_FOLD_TYPES = {
    'mountain': crease_utils.Fold.MOUNTAIN,
    'valley': crease_utils.Fold.VALLEY,
}
_FOLD_ORDERS = {
    'MV': shaping_types.MOUNTAIN_TO_VALLEY,
    'VM': shaping_types.VALLEY_TO_MOUNTAIN,
}
_DEPTH_OFFSETS = (0.0, 0.15, -0.15, 0.3)
_ROTATION_RETRY_DEPTH = 0.4
_MIN_NARROW_ANGLE_DEG = 8.0
_THIN_BEND_DEPTH = 0.5
_THIN_KNEE_DEPTH = 0.25
_THIN_MIN_GAIN_DEG = 2.0

TOOL_NAMES: tuple[str, ...] = (
    'simple_fold',
    'crimp_fold',
    'reverse_fold',
    'geometric_simple_fold',
    'clip_pattern_fold',
    'narrow_central_flap',
    'fan_out_fold',
    'squash_fold',
    'inscribed_flap',
    'open_flap',
    'fold_spine',
    'rotation_3d',
    'bend_to_direction',
    'thin_limbs_3d',
    'curve_flap',
    'spread_fork',
    'radial_fan',
    'apply_symmetric',
    'tree_edit',
)


def list_shaping_tools() -> tuple[str, ...]:
  """Returns the ordered tuple of canonical shaping tool names."""
  return TOOL_NAMES


def normalize_tool_name(name: str) -> str:
  """Strips an optional namespace prefix (e.g. `'shaping:crimp_fold'`)."""
  return name.split(':', 1)[-1].strip()


# ------------------------------------------------------------------------------
# Shared fold primitives (2-D zig-zag and local 3-D folded-state cuts).
# ------------------------------------------------------------------------------


def _preprocess_angle(angle_deg: float, *, min_deg: float = 0.0) -> float:
  """Wraps `angle_deg` to `(-180, 180]` and converts to clipped radians."""
  wrapped = angle_utils.wrap_angle_centered(float(angle_deg), 360.0)
  if min_deg > 0.0 and abs(wrapped) < min_deg:
    wrapped = math.copysign(min_deg, wrapped or 1.0)
  rad = math.radians(wrapped)
  if math.isclose(rad, 0.0, abs_tol=crease_utils.DEFAULT_TOLERANCE):
    return 0.0
  return heuristics_utils.clip_angle_by_min_angle(rad)


def _zigzag_params(
    state: ShapingState,
    *,
    element_name: str,
    angle: float,
    depth: float,
    fold_percent: float = 1.0,
    fold_order: str | None = None,
    min_deg: float = 0.0,
) -> tuple[state_lib.ShapingElement, shaping_types.ElementZigzagParams]:
  """Constructs an `ElementShapingParams[ZigzagParams]` for a flap or river."""
  if not 0.0 <= depth < 1.0:
    raise ValueError(f'depth must be in [0, 1), got {depth}.')
  if not 0.0 < fold_percent <= 1.0:
    raise ValueError(f'fold_percent must be in (0, 1], got {fold_percent}.')
  order_seq = _FOLD_ORDERS['MV']
  if fold_order is not None:
    key = fold_order.upper()
    if key not in _FOLD_ORDERS:
      raise ValueError(f"fold_order must be 'MV' or 'VM', got {fold_order!r}.")
    order_seq = _FOLD_ORDERS[key]

  elem = state.element(element_name)
  angle_rad = _preprocess_angle(angle, min_deg=min_deg)
  if math.isclose(angle_rad, 0.0):
    raise ValueError('angle must be non-zero.')
  is_flap = isinstance(elem, shaping_elements.ShapingFlap)
  span = int(elem.grid_unit_radius) if is_flap else int(round(elem.width))
  grid_depth = min(max(0, int(span * depth)), max(0, span - 1))
  if is_flap:
    z_params = shaping_types.ZigzagParams(
        target_flap_angle=abs(angle_rad),
        even_phase=angle_rad >= 0.0,
        fold_order=order_seq,
        fold_percent=float(fold_percent),
        depth=grid_depth,
    )
  else:
    banks = state.shaping_heuristic_data['outer_bank_indices']
    bank = banks.get(element_name, elem.longer_bank_index)
    z_params = shaping_types.ZigzagParams(
        target_flap_angle=abs(angle_rad),
        even_phase=angle_rad >= 0.0,
        fold_order=order_seq,
        fold_percent=float(fold_percent),
        depth=grid_depth,
        bank_index=bank,
    )
  return elem, shaping_types.ElementShapingParams(
      element_name=element_name, params=z_params
  )


def _apply_simple_zigzag(
    state: ShapingState,
    elem: state_lib.ShapingElement,
    param: shaping_types.ElementZigzagParams,
    *,
    narrow: bool,
    explicit_order: bool,
    allow_small_angle_split: bool,
) -> crease_pattern_lib.CreasePattern:
  """Runs optional zig-zag heuristics and `shape_cp_with_simple_folds`."""
  flaps, rivers = state.shaping_elements
  params: Sequence[shaping_types.ElementZigzagParams] = [param]
  if state.config.apply_simple_fold_heuristics:
    flap_info = state.shaping_heuristic_data['flap_info']
    orientations = {
        name: node.absolute_direction for name, node in flap_info.items()
    }
    zp = param.params
    signed_angle = abs(zp.target_flap_angle or 0.0) * (
        1.0 if zp.even_phase else -1.0
    )
    signed = dataclasses.replace(
        param,
        params=dataclasses.replace(zp, target_flap_angle=signed_angle),
    )
    adjusted = list(
        fold_shaping.apply_heuristics_to_shaping_params(
            grid_size=elem.grid_size,
            flaps=flaps,
            rivers=rivers,
            element_orientations=orientations,
            hinge_segments=state.hinge_segments,
            element_shaping_params=[signed],
            use_heuristics_for_fold_order=(
                state.config.use_heuristics_for_fold_order
                and not explicit_order
            ),
            use_heuristics_for_depth=state.config.use_heuristics_for_depth,
            use_heuristics_for_phase=state.config.use_heuristics_for_phase,
            convert_small_angles=(
                allow_small_angle_split
                and not narrow
                and state.config.convert_small_angles
            ),
        )
    )
    if adjusted:
      params = adjusted
  return fold_shaping.shape_cp_with_simple_folds(
      cp=state.pattern,
      element_shaping_params=params,
      grid_size=elem.grid_size,
      flaps=flaps,
      rivers=rivers,
      central_crease_removed=state.central_crease_removed,
      narrow=narrow,
      light_mode=not state.config.check_all_crease_intersections,
  )


def _local_3d_cut(
    state: ShapingState,
    *,
    element_name: str,
    depth: float,
    fold_percent: float,
    verify_fold: bool,
    kind: str,
    make_lines: Callable[
        [Any, float, np.ndarray],
        list[tuple[tuple[float, float], tuple[float, float]]],
    ],
) -> state_lib.ToolOutcome:
  """Cuts a flap's layers in the 2-D folded form and inverts M/V polarity."""
  if element_name not in state.tree:
    raise ValueError(
        f'{kind}: unknown element {element_name!r}; known:'
        f' {state.element_names()}.'
    )
  if not 0.0 <= depth < 1.0:
    raise ValueError(f'depth must be in [0, 1), got {depth}.')
  if not 0.0 < fold_percent <= 1.0:
    raise ValueError(f'fold_percent must be in (0, 1], got {fold_percent}.')

  def _impl(st: ShapingState) -> state_lib.ToolOutcome:
    parent = paper_layout_lib.topology(st.tree).parent.get(element_name)
    if parent is None:
      raise ValueError(
          f'{kind}: {element_name!r} is the root stick; choose a non-root flap.'
      )
    graph_data = st.graph_data
    polygon = st.element_polygon(element_name)
    parent_polygon = st.element_polygon(parent)
    axis = folded_region_fold.get_element_folded_axis(
        graph_data, polygon, st.reference_point(element_name, graph_data)
    )
    points = folded_region_fold.element_folded_points(graph_data, polygon)
    scopes: list[tuple[str, list[Any], bool]] = [('element', polygon, True)]
    chain = st.chain_polygon(element_name)
    if chain is not None:
      scopes.append(('chain', chain, True))
      scopes.append(('strip', chain, False))

    offsets = _DEPTH_OFFSETS if verify_fold else (0.0,)
    depths: list[float] = []
    for off in offsets:
      val = round(min(0.95, max(0.05, depth + off)), 3)
      if val not in depths:
        depths.append(val)

    allow_3d = not math.isclose(fold_percent, 1.0, abs_tol=1e-6)
    seen: set[Any] = set()
    last_reason = 'no candidate cut the target flap'
    for (scope_name, limiting, near), cand_depth in itertools.product(
        scopes, depths
    ):
      label = f'{cand_depth:g}/{scope_name}'
      try:
        lines = make_lines(axis, cand_depth, points)
        if near:
          spans = [
              folded_region_fold.line_span(line, points, 0.25 * axis.width)
              for line in lines
          ]
        else:
          spans = [(-math.inf, math.inf) for _ in lines]
        cut_sets = []
        for line, span in zip(lines, spans, strict=True):
          cuts = geometric_simple_fold_lib.get_segment_cut_creases(
              graph_data=graph_data,
              cut_line_start=line[0],
              cut_line_end=line[1],
              span=span,
              limiting_polygon=limiting,
          )
          if not cuts:
            raise ValueError('cut line missed target layers')
          if any(
              math.hypot(c.p1.x - c.p2.x, c.p1.y - c.p2.y)
              < state_lib.MIN_SEGMENT_LENGTH
              for c in cuts
          ):
            raise ValueError('degenerate cut segment')
          cut_sets.append(frozenset(state_lib.crease_key(c) for c in cuts))
        sig = (
            tuple(
                (round(s[0], 6), round(s[1], 6), round(e[0], 6), round(e[1], 6))
                for s, e in lines
            ),
            tuple(cut_sets),
        )
        if sig in seen:
          continue
        seen.add(sig)
        variants = folded_region_fold.get_inverted_region_variants(
            st.pattern,
            cut_lines=lines,
            limiting_polygon=limiting,
            graph_data=graph_data,
            fold_percent=float(fold_percent),
            proximal_polygon=parent_polygon,
            target_polygon=polygon,
            spans=spans,
        )
      except (ValueError, AssertionError) as error:
        last_reason = f'{label}: {error}'
        continue

      for variant in variants:
        cand = variant['crease_pattern']
        if not allow_3d and not cand.local_flat_foldability():
          last_reason = f'{label} {variant["name"]}: not flat-foldable'
          continue
        accepted, reason, _ = state_lib.accept_pattern(
            st, cand, allow_3d_vertices=allow_3d, verify_fold=verify_fold
        )
        if accepted is None:
          last_reason = f'{label} {variant["name"]}: {reason}'
          continue
        return state_lib.ToolOutcome(
            state=accepted,
            note=(
                f'{element_name}: {kind} at {label} ({variant["name"]}),'
                f' +{variant["num_new_creases"]} creases,'
                f' {variant["num_flipped_creases"]} flipped.'
            ),
        )
    raise ValueError(f'{kind}({element_name}) failed: {last_reason}')

  return state_lib.run_with_3d_overlay(state, _impl, verify_fold=verify_fold)


def _reverse_3d_outcome(
    state: ShapingState,
    *,
    element_name: str,
    angle: float,
    depth: float = 0.5,
    fold_percent: float = 1.0,
    verify_fold: bool = True,
) -> state_lib.ToolOutcome:
  return _local_3d_cut(
      state,
      element_name=element_name,
      depth=depth,
      fold_percent=fold_percent,
      verify_fold=verify_fold,
      kind='reverse_fold(3d)',
      make_lines=lambda axis, d, pts: [
          folded_region_fold.reverse_fold_cut_line(
              axis, angle=angle, depth=d, snap_points=pts
          )
      ],
  )


def _crimp_3d_outcome(
    state: ShapingState,
    *,
    element_name: str,
    angle: float,
    depth: float = 0.4,
    gap: float | None = None,
    fold_percent: float = 1.0,
    verify_fold: bool = True,
) -> state_lib.ToolOutcome:
  return _local_3d_cut(
      state,
      element_name=element_name,
      depth=depth,
      fold_percent=fold_percent,
      verify_fold=verify_fold,
      kind='crimp_fold(3d)',
      make_lines=lambda axis, d, pts: (
          folded_region_fold.crimp_fold_cut_lines(
              axis, angle=angle, depth=d, gap=gap, snap_points=pts
          )
      ),
  )


# ------------------------------------------------------------------------------
# 1–4. Simple, crimp, reverse, and geometric simple folds.
# ------------------------------------------------------------------------------


def _simple_fold_outcome(
    state: ShapingState,
    *,
    element_name: str,
    angle: float | None = None,
    depth: float = 0.0,
    narrow: bool = False,
    fold_percent: float = 1.0,
    fold_order: str | None = None,
    verify_fold: bool = False,
) -> state_lib.ToolOutcome:
  """Unified 2-D simple fold: explicit angle, tree heuristic, and narrow."""

  def _impl(st: ShapingState) -> state_lib.ToolOutcome:
    allow_3d = not math.isclose(fold_percent, 1.0, abs_tol=1e-6)

    if angle is None and not narrow:
      _, rel_angles = heuristics.shaping_angles(
          st.tree, use_tree_root=st.config.use_tree_root
      )
      _, rel_az, rel_el = rel_angles.get(element_name, (None, 0.0, 0.0))
      target_deg = (
          math.degrees(float(rel_az))
          if not math.isclose(rel_az, 0.0, abs_tol=1e-4)
          else math.degrees(float(rel_el))
      )
      if math.isclose(target_deg, 0.0, abs_tol=1e-4):
        stick = st.tree[element_name] if element_name in st.tree else None
        fallback_deg = (
            stick.azimuth_angle
            if stick is not None and not math.isclose(stick.azimuth_angle, 0.0)
            else (stick.elevation_angle if stick is not None else 0.0)
        )
        if math.isclose(fallback_deg, 0.0, abs_tol=1e-4):
          raise ValueError(
              f'simple_fold({element_name}): stick has zero heuristic angle;'
              ' pass an explicit angle.'
          )
        target_deg = float(fallback_deg)
      out = _simple_fold_outcome(
          st,
          element_name=element_name,
          angle=target_deg,
          depth=depth,
          narrow=False,
          fold_percent=fold_percent,
          fold_order=fold_order,
          verify_fold=verify_fold,
      )
      resolved_args: dict[str, Any] = {
          'element_name': element_name,
          'angle': target_deg,
          'depth': depth,
          'narrow': False,
          'fold_percent': fold_percent,
          'verify_fold': verify_fold,
      }
      if fold_order is not None:
        resolved_args['fold_order'] = fold_order
      return dataclasses.replace(out, resolved_arguments=resolved_args)

    req_angle = 45.0 if angle is None else float(angle)
    if math.isclose(req_angle, 0.0):
      raise ValueError('angle must be non-zero.')

    elem = st.element(element_name)
    if narrow and isinstance(elem, shaping_elements.ShapingRiver):
      step_units = max(0.0, float(int(elem.width * depth)))
      clipped = relief.clip_pattern_fold(
          st,
          element_name=element_name,
          template_name='fold',
          step=step_units,
          enable_narrow=True,
          verify_fold=verify_fold,
      )
      return dataclasses.replace(
          clipped,
          state=clipped.state.with_narrowed(element_name),
          note=(
              f'{clipped.note} [routed river narrowing to'
              ' clip_pattern_fold(template_name="fold")]'
          ),
      )

    if narrow:
      trials: list[tuple[float, str | None]] = [(req_angle, fold_order)]
      if angle is None or verify_fold:
        trials.append((-req_angle, fold_order))
      for sgn in (req_angle, -req_angle) if verify_fold else (req_angle,):
        for explicit in ('MV', 'VM'):
          if fold_order is None and (sgn, explicit) not in trials:
            trials.append((sgn, explicit))
      last_reason = 'no narrowing creases produced'
      for cand_angle, cand_order in trials:
        try:
          el, param = _zigzag_params(
              st,
              element_name=element_name,
              angle=cand_angle,
              depth=depth,
              fold_percent=fold_percent,
              fold_order=cand_order,
              min_deg=_MIN_NARROW_ANGLE_DEG,
          )
          shaped = _apply_simple_zigzag(
              st,
              el,
              param,
              narrow=True,
              explicit_order=cand_order is not None,
              allow_small_angle_split=False,
          )
        except (ValueError, AssertionError, KeyError, IndexError) as error:
          last_reason = str(error)
          continue
        if len(shaped.creases) == len(st.pattern.creases):
          continue
        accepted, reason, _ = state_lib.accept_pattern(
            st, shaped, allow_3d_vertices=allow_3d, verify_fold=verify_fold
        )
        if accepted is not None:
          out = accepted.with_narrowed(element_name)
          delta = len(out.pattern.creases) - len(st.pattern.creases)
          return state_lib.ToolOutcome(
              state=out,
              note=(
                  f'{element_name}: narrowed simple_fold(angle={cand_angle:g},'
                  f' order={cand_order or "auto"}), +{delta} creases.'
              ),
          )
        last_reason = reason

      # Interior flap fallback via clip-pattern narrowing.
      if isinstance(elem, shaping_elements.ShapingFlap):
        step_units = max(0.0, float(elem.grid_unit_radius) * float(depth))
        try:
          clipped = relief.clip_pattern_fold(
              st,
              element_name=element_name,
              template_name='fold',
              step=step_units,
              enable_narrow=True,
              verify_fold=verify_fold,
          )
          return dataclasses.replace(
              clipped,
              state=clipped.state.with_narrowed(element_name),
              note=(
                  f'{clipped.note} [fallback from zigzag narrowing'
                  f' ({last_reason}) -> clip_pattern_fold]'
              ),
          )
        except ValueError as clip_err:
          last_reason = f'{last_reason}; clip fallback: {clip_err}'
      raise ValueError(
          f'simple_fold({element_name}, narrow=True) failed: {last_reason}'
      )

    el, param = _zigzag_params(
        st,
        element_name=element_name,
        angle=req_angle,
        depth=depth,
        fold_percent=fold_percent,
        fold_order=fold_order,
    )
    last_reason = 'no shaping creases generated'
    for split_small in (True, False):
      try:
        shaped = _apply_simple_zigzag(
            st,
            el,
            param,
            narrow=False,
            explicit_order=fold_order is not None,
            allow_small_angle_split=split_small,
        )
      except (ValueError, AssertionError) as error:
        last_reason = str(error)
        continue
      accepted, reason, _ = state_lib.accept_pattern(
          st, shaped, allow_3d_vertices=allow_3d, verify_fold=verify_fold
      )
      if accepted is not None:
        delta = len(accepted.pattern.creases) - len(st.pattern.creases)
        return state_lib.ToolOutcome(
            state=accepted,
            note=(
                f'{element_name}: simple_fold(angle={req_angle:g},'
                f' depth={depth:g}), +{delta} creases.'
            ),
        )
      last_reason = reason
    raise ValueError(f'simple_fold({element_name}) failed: {last_reason}')

  return state_lib.run_with_3d_overlay(state, _impl, verify_fold=verify_fold)


def simple_fold(
    state: ShapingState,
    *,
    element_name: str,
    angle: float | None = None,
    depth: float = 0.0,
    narrow: bool = False,
    fold_percent: float = 1.0,
    fold_order: str | None = None,
    verify_fold: bool = False,
) -> ShapingState:
  """Applies a 2-D zig-zag simple fold, narrowing, or tree-angle fold."""
  return _simple_fold_outcome(
      state,
      element_name=element_name,
      angle=angle,
      depth=depth,
      narrow=narrow,
      fold_percent=fold_percent,
      fold_order=fold_order,
      verify_fold=verify_fold,
  ).state


def _crimp_2d_outcome(
    state: ShapingState,
    *,
    element_name: str,
    angle: float,
    depth: float,
    fold_percent: float,
    verify_fold: bool,
) -> state_lib.ToolOutcome:
  """Runs a 2-D zig-zag crimp across mountain/valley assignment pairs."""
  elem, param = _zigzag_params(
      state,
      element_name=element_name,
      angle=angle,
      depth=depth,
      fold_percent=fold_percent,
  )
  zp = param.params
  mv = [crease_utils.Fold.MOUNTAIN, crease_utils.Fold.VALLEY]
  vm = [crease_utils.Fold.VALLEY, crease_utils.Fold.MOUNTAIN]
  crimp = fold_shaping.CrimpFoldShapingParams(
      element_name=element_name,
      crimp_fold_params=[
          fold_shaping.CrimpFoldParams(
              depth=int(zp.depth or 0),
              fold_order=mv,
              fold_percent=float(fold_percent),
              zigzag_flap_angle=zp.target_flap_angle or 0.0,
              zigzag_even_phase=zp.even_phase,
              zigzag_fold_order=vm,
              bank_index=getattr(zp, 'bank_index', None),
          )
      ],
  )
  flaps, rivers = state.shaping_elements
  inscribed, zigzags = fold_shaping.get_crimp_fold_creases(
      cp=state.pattern,
      grid_size=elem.grid_size,
      flaps=flaps,
      element_shaping_params=[crimp],
      rivers=rivers,
      central_crease_removed=state.central_crease_removed,
  )
  new_creases = list(inscribed) + list(zigzags)
  if not new_creases:
    raise ValueError(f'crimp_fold(2d): no creases produced on {element_name}.')
  base = state.pattern.creases
  tips, _ = crease_pattern_lib.get_tips_and_grid_creases(
      base + new_creases, grid_size=-1
  )
  split_ins = crease_pattern_lib.split_creases(
      inscribed, base + list(zigzags), tips
  )
  split_zig = crease_pattern_lib.split_creases(
      zigzags, base + list(inscribed) + list(zigzags), tips
  )
  split_base = crease_pattern_lib.split_creases(
      base, split_ins + split_zig, tips
  )
  allow_3d = not math.isclose(fold_percent, 1.0, abs_tol=1e-6)
  last_reason = 'not flat-foldable'
  for ins_order, zig_order in ((mv, vm), (vm, mv), (mv, mv), (vm, vm)):
    try:
      new_ins, new_zig = fold_shaping.assign_crimp_fold_orders(
          split_ins, split_zig, ins_order, zig_order
      )
      mod_base = fold_shaping.fix_crimp_flat_foldability(
          list(split_base),
          inscribed_creases=new_ins,
          zigzag_creases=new_zig,
      )
      added = new_ins + new_zig
      if allow_3d:
        added = [
            dataclasses.replace(c, fold_percent=float(fold_percent))
            for c in added
        ]
      cand = crease_pattern_lib.CreasePattern(
          creases=mod_base + added,
          grid_size=-1,
          light_mode=True,
          tolerance=state.pattern.tolerance,
      )
    except (ValueError, AssertionError) as error:
      last_reason = str(error)
      continue
    accepted, reason, _ = state_lib.accept_pattern(
        state, cand, allow_3d_vertices=allow_3d, verify_fold=verify_fold
    )
    if accepted is not None:
      delta = len(accepted.pattern.creases) - len(state.pattern.creases)
      return state_lib.ToolOutcome(
          state=accepted,
          note=f'{element_name}: crimp_fold(2d), +{delta} creases.',
      )
    last_reason = reason
  raise ValueError(f'crimp_fold(2d) on {element_name} failed: {last_reason}')


def _crimp_fold_outcome(
    state: ShapingState,
    *,
    element_name: str,
    angle: float,
    depth: float = 0.4,
    gap: float | None = None,
    fold_percent: float = 1.0,
    mode: str = 'auto',
    verify_fold: bool = True,
) -> state_lib.ToolOutcome:
  """Unified crimp fold supporting `'auto'`, `'2d'`, and `'3d'` modes."""
  if mode not in ('auto', '2d', '3d'):
    raise ValueError(f"mode must be 'auto', '2d', or '3d'; got {mode!r}.")
  if not 0.0 < depth < 1.0:
    raise ValueError(f'depth must be in (0, 1); got {depth:g}.')
  if math.isclose(angle, 0.0, abs_tol=1e-6):
    raise ValueError('angle must be non-zero.')
  wrapped = angle_utils.wrap_angle_centered(float(angle), 360.0)
  if math.isclose(abs(wrapped), 180.0, abs_tol=1e-5):
    raise ValueError('angle=180 is not allowed for crimp_fold.')
  if mode == '2d' and gap is not None:
    raise ValueError(
        "gap is only supported for mode='3d' or mode='auto', not mode='2d'."
    )
  if mode == '3d' or gap is not None:
    out = _crimp_3d_outcome(
        state,
        element_name=element_name,
        angle=angle,
        depth=depth,
        gap=gap,
        fold_percent=fold_percent,
        verify_fold=verify_fold,
    )
    if mode == 'auto' and gap is not None:
      resolved: dict[str, Any] = {
          'element_name': element_name,
          'angle': angle,
          'depth': depth,
          'gap': gap,
          'fold_percent': fold_percent,
          'mode': '3d',
          'verify_fold': verify_fold,
      }
      return dataclasses.replace(
          out,
          note=f'{out.note} [routed mode=auto with gap={gap:g} to 3d]',
          resolved_arguments=resolved,
      )
    return out
  if mode == '2d':
    return state_lib.run_with_3d_overlay(
        state,
        lambda st: _crimp_2d_outcome(
            st,
            element_name=element_name,
            angle=angle,
            depth=depth,
            fold_percent=fold_percent,
            verify_fold=verify_fold,
        ),
        verify_fold=verify_fold,
    )
  try:
    return state_lib.run_with_3d_overlay(
        state,
        lambda st: _crimp_2d_outcome(
            st,
            element_name=element_name,
            angle=angle,
            depth=depth,
            fold_percent=fold_percent,
            verify_fold=verify_fold,
        ),
        verify_fold=verify_fold,
    )
  except (ValueError, AssertionError) as err_2d:
    try:
      out = _crimp_3d_outcome(
          state,
          element_name=element_name,
          angle=angle,
          depth=depth,
          gap=gap,
          fold_percent=fold_percent,
          verify_fold=verify_fold,
      )
      resolved = {
          'element_name': element_name,
          'angle': angle,
          'depth': depth,
          'fold_percent': fold_percent,
          'mode': '3d',
          'verify_fold': verify_fold,
      }
      if gap is not None:
        resolved['gap'] = gap
      return dataclasses.replace(
          out,
          note=f'{out.note} [fallback from 2d: {err_2d}]',
          resolved_arguments=resolved,
      )
    except ValueError as err_3d:
      raise ValueError(f'{err_2d}; 3d fallback: {err_3d}') from err_3d


def crimp_fold(
    state: ShapingState,
    *,
    element_name: str,
    angle: float,
    depth: float = 0.4,
    gap: float | None = None,
    fold_percent: float = 1.0,
    mode: str = 'auto',
    verify_fold: bool = True,
) -> ShapingState:
  """Adds a crimp fold (`mode='auto'` tries 2-D zig-zag then local 3-D cut)."""
  return _crimp_fold_outcome(
      state,
      element_name=element_name,
      angle=angle,
      depth=depth,
      gap=gap,
      fold_percent=fold_percent,
      mode=mode,
      verify_fold=verify_fold,
  ).state


def _reverse_2d_outcome(
    state: ShapingState,
    *,
    element_name: str,
    angle: float,
    depth: float,
    fold_type: str,
    orientation: str,
    fold_percent: float,
    verify_fold: bool,
) -> state_lib.ToolOutcome:
  """Runs a 2-D zig-zag reverse fold honoring the caller's `fold_type`."""
  if orientation not in ('horizontal', 'vertical'):
    raise ValueError("orientation must be 'horizontal' or 'vertical'.")
  base_fold = _FOLD_TYPES.get(fold_type.lower())
  if base_fold is None:
    raise ValueError("fold_type must be 'valley' or 'mountain'.")

  elem, param = _zigzag_params(
      state,
      element_name=element_name,
      angle=angle,
      depth=depth,
      fold_percent=fold_percent,
  )
  zp = param.params
  rev_param = fold_shaping.make_reverse_fold_params(
      flap_angle=zp.target_flap_angle,
      fold_order=(base_fold, base_fold.flip()),
      even_phase=zp.even_phase,
      bank_index=getattr(zp, 'bank_index', None),
      depth=int(zp.depth or 0),
      fold_percent=float(fold_percent),
  )
  spec = fold_shaping.ReverseFoldShapingParams(
      element_name=element_name,
      reverse_fold_params=[rev_param],
      orientation=orientation,
      fold_type=base_fold,
  )
  flaps, rivers = state.shaping_elements
  base_creases, instructions = fold_shaping.get_reverse_fold_creases(
      cp=state.pattern,
      grid_size=elem.grid_size,
      rectangles=flaps,
      element_shaping_params=[spec],
      rivers=rivers,
  )
  zig_creases = [inst.crease for inst in instructions]
  if not zig_creases:
    raise ValueError(f'reverse_fold(2d): no creases on {element_name}.')
  tips, _ = crease_pattern_lib.get_tips_and_grid_creases(
      base_creases + zig_creases, grid_size=-1
  )
  split_base = crease_pattern_lib.split_creases(base_creases, zig_creases, tips)
  split_zigs = [
      crease_pattern_lib.split_creases(
          [inst.crease], base_creases + zig_creases, tips
      )
      for inst in instructions
  ]
  allow_3d = not math.isclose(fold_percent, 1.0, abs_tol=1e-6)
  other = 'vertical' if orientation == 'horizontal' else 'horizontal'
  last_reason = 'not flat-foldable'
  for cur_ori in (orientation, other):
    for fld, flip_base in (
        (base_fold, False),
        (base_fold, True),
        (base_fold.flip(), True),
        (base_fold.flip(), False),
    ):
      try:
        assigned = fold_shaping.assign_reverse_fold_orders(
            instructions, fld, cur_ori, flip_base
        )
        final_zigs: list[crease_utils.Crease] = []
        for asg, geoms in zip(assigned, split_zigs, strict=True):
          for geom in geoms:
            final_zigs.append(
                dataclasses.replace(
                    geom, fold=asg.fold, fold_percent=float(fold_percent)
                )
            )
        cand = crease_pattern_lib.CreasePattern(
            creases=list(split_base) + final_zigs,
            grid_size=-1,
            light_mode=True,
            tolerance=state.pattern.tolerance,
        )
      except (ValueError, AssertionError) as error:
        last_reason = str(error)
        continue
      accepted, reason, _ = state_lib.accept_pattern(
          state, cand, allow_3d_vertices=allow_3d, verify_fold=verify_fold
      )
      if accepted is not None:
        delta = len(accepted.pattern.creases) - len(state.pattern.creases)
        return state_lib.ToolOutcome(
            state=accepted,
            note=(
                f'{element_name}: reverse_fold(2d, {cur_ori},'
                f' {fld.name.lower()}), +{delta} creases.'
            ),
        )
      last_reason = reason
  raise ValueError(f'reverse_fold(2d) on {element_name} failed: {last_reason}')


def _reverse_fold_outcome(
    state: ShapingState,
    *,
    element_name: str,
    angle: float,
    depth: float | None = None,
    fold_type: str = 'valley',
    orientation: str = 'horizontal',
    fold_percent: float = 1.0,
    mode: str = 'auto',
    verify_fold: bool = True,
) -> state_lib.ToolOutcome:
  """Unified reverse fold supporting `'auto'`, `'2d'`, and `'3d'` modes."""
  if mode not in ('auto', '2d', '3d'):
    raise ValueError(f"mode must be 'auto', '2d', or '3d'; got {mode!r}.")
  if depth is not None and not 0.0 <= depth < 1.0:
    raise ValueError(f'depth must be in [0, 1); got {depth:g}.')
  if math.isclose(angle, 0.0, abs_tol=1e-6):
    raise ValueError('angle must be non-zero.')
  depth_2d = 0.0 if depth is None else float(depth)
  depth_3d = 0.5 if depth is None else float(depth)
  if mode == '3d':
    return _reverse_3d_outcome(
        state,
        element_name=element_name,
        angle=angle,
        depth=depth_3d,
        fold_percent=fold_percent,
        verify_fold=verify_fold,
    )
  if mode == '2d':
    return state_lib.run_with_3d_overlay(
        state,
        lambda st: _reverse_2d_outcome(
            st,
            element_name=element_name,
            angle=angle,
            depth=depth_2d,
            fold_type=fold_type,
            orientation=orientation,
            fold_percent=fold_percent,
            verify_fold=verify_fold,
        ),
        verify_fold=verify_fold,
    )
  try:
    return state_lib.run_with_3d_overlay(
        state,
        lambda st: _reverse_2d_outcome(
            st,
            element_name=element_name,
            angle=angle,
            depth=depth_2d,
            fold_type=fold_type,
            orientation=orientation,
            fold_percent=fold_percent,
            verify_fold=verify_fold,
        ),
        verify_fold=verify_fold,
    )
  except (ValueError, AssertionError) as err_2d:
    try:
      out = _reverse_3d_outcome(
          state,
          element_name=element_name,
          angle=angle,
          depth=depth_3d,
          fold_percent=fold_percent,
          verify_fold=verify_fold,
      )
      return dataclasses.replace(
          out,
          note=f'{out.note} [fallback from 2d: {err_2d}]',
          resolved_arguments={
              'element_name': element_name,
              'angle': angle,
              'depth': depth_3d,
              'fold_type': fold_type,
              'orientation': orientation,
              'fold_percent': fold_percent,
              'mode': '3d',
              'verify_fold': verify_fold,
          },
      )
    except ValueError as err_3d:
      raise ValueError(f'{err_2d}; 3d fallback: {err_3d}') from err_3d


def reverse_fold(
    state: ShapingState,
    *,
    element_name: str,
    angle: float,
    depth: float | None = None,
    fold_type: str = 'valley',
    orientation: str = 'horizontal',
    fold_percent: float = 1.0,
    mode: str = 'auto',
    verify_fold: bool = True,
) -> ShapingState:
  """Adds a reverse fold (`mode='auto'` tries 2-D zig-zag then 3-D cut)."""
  return _reverse_fold_outcome(
      state,
      element_name=element_name,
      angle=angle,
      depth=depth,
      fold_type=fold_type,
      orientation=orientation,
      fold_percent=fold_percent,
      mode=mode,
      verify_fold=verify_fold,
  ).state


def _geometric_simple_fold_outcome(
    state: ShapingState,
    *,
    folds: Sequence[Mapping[str, Any]],
    sequential: bool = False,
    verify_fold: bool = False,
) -> state_lib.ToolOutcome:
  """Applies cut-line simple folds specified in 2-D folded-form coordinates."""
  if not folds:
    raise ValueError('folds must not be empty.')

  def _impl(st: ShapingState) -> state_lib.ToolOutcome:
    specs: list[dict[str, Any]] = []
    for item in folds:
      names = list(item.get('element_names') or ())
      limiting = None
      if names:
        limiting = []
        for n in names:
          limiting.extend(st.element_polygon(str(n)))
      fold_type = _FOLD_TYPES.get(str(item['fold_type']).lower())
      if fold_type is None:
        raise ValueError("fold_type must be 'mountain' or 'valley'.")
      specs.append({
          'cut_line_start': (float(item['start_x']), float(item['start_y'])),
          'cut_line_end': (float(item['end_x']), float(item['end_y'])),
          'fold_assignment': fold_type,
          'limiting_polygon': limiting,
      })

    pattern = st.pattern
    if sequential:
      for idx, spec in enumerate(specs):
        gd = (
            st.graph_data
            if idx == 0
            else geometric_simple_fold_lib.prepare_graph_data(pattern)
        )
        creases = geometric_simple_fold_lib.get_geometric_fold_creases(
            crease_pattern=pattern, graph_data=gd, **spec
        )
        if creases:
          pattern = pattern.update_with_shaping_creases(creases)
    else:
      gd = st.graph_data
      new_creases: list[crease_utils.Crease] = []
      for spec in specs:
        new_creases.extend(
            geometric_simple_fold_lib.get_geometric_fold_creases(
                crease_pattern=pattern, graph_data=gd, **spec
            )
        )
      if not new_creases:
        raise ValueError('geometric_simple_fold produced no creases.')
      pattern = pattern.update_with_shaping_creases(new_creases)

    accepted, reason, _ = state_lib.accept_pattern(
        st, pattern, allow_3d_vertices=False, verify_fold=verify_fold
    )
    if accepted is None:
      raise ValueError(reason)
    delta = len(accepted.pattern.creases) - len(st.pattern.creases)
    return state_lib.ToolOutcome(
        state=accepted,
        note=f'geometric_simple_fold: {len(specs)} line(s), +{delta} creases.',
    )

  return state_lib.run_with_3d_overlay(state, _impl, verify_fold=verify_fold)


def geometric_simple_fold(
    state: ShapingState,
    *,
    folds: Sequence[Mapping[str, Any]],
    sequential: bool = False,
    verify_fold: bool = False,
) -> ShapingState:
  """Applies cut-line simple folds across elements in 2-D folded space."""
  return _geometric_simple_fold_outcome(
      state, folds=folds, sequential=sequential, verify_fold=verify_fold
  ).state


# ------------------------------------------------------------------------------
# 5–11. Public wrappers around `relief` and `fan_squash` modules.
# ------------------------------------------------------------------------------


def clip_pattern_fold(
    state: ShapingState,
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
) -> ShapingState:
  """Draws a layer-aware clip-pattern template (`fold`, `pleat_to_hide`, etc.)."""
  return relief.clip_pattern_fold(
      state,
      element_name=element_name,
      template_name=template_name,
      step=step,
      bend_angle=bend_angle,
      rotate_angle=rotate_angle,
      enable_narrow=enable_narrow,
      narrow_n_times=narrow_n_times,
      narrow_symmetric=narrow_symmetric,
      pleat_distance=pleat_distance,
      length=length,
      global_refine=global_refine,
      verify_fold=verify_fold,
  ).state


def narrow_central_flap(
    state: ShapingState,
    *,
    element_name: str,
    tip_width: float = 0.25,
    start_offset: float = 0.0,
    fold_type: str = 'auto',
    require_symmetric: bool = True,
    verify_fold: bool = True,
) -> ShapingState:
  """Symmetrically tapers both halves of a flap spanning the central crease."""
  return relief.narrow_central_flap(
      state,
      element_name=element_name,
      tip_width=tip_width,
      start_offset=start_offset,
      fold_type=fold_type,
      require_symmetric=require_symmetric,
      verify_fold=verify_fold,
  ).state


def fan_out_fold(
    state: ShapingState,
    *,
    element_name: str,
    move_delta: float = 1.0,
    orientation: str | None = None,
    fold_type: str | None = None,
    spread: bool = False,
    open_percent: float = 1.0,
    verify_fold: bool = True,
) -> ShapingState:
  """Fans border accordion pleats of a flap (auto-picks orientation if None)."""
  return fan_squash.fan_out_fold(
      state,
      element_name=element_name,
      move_delta=move_delta,
      orientation=orientation,
      fold_type=fold_type,
      spread=spread,
      open_percent=open_percent,
      verify_fold=verify_fold,
  ).state


def squash_fold(
    state: ShapingState,
    *,
    element_name: str,
    ridge_index: int | None = None,
    starting_position_index: int | None = None,
    ending_position_index: int | None = None,
    open_percent: float = 0.45,
    verify_fold: bool = True,
) -> ShapingState:
  """Squashes a flap along a ridge (or mirror ridge pair) to widen it in 3-D."""
  return fan_squash.squash_fold(
      state,
      element_name=element_name,
      ridge_index=ridge_index,
      starting_position_index=starting_position_index,
      ending_position_index=ending_position_index,
      open_percent=open_percent,
      verify_fold=verify_fold,
  ).state


def inscribed_flap(
    state: ShapingState,
    *,
    element_name: str,
    depths: int | Sequence[int] = (1,),
    fold_order: str = 'MV',
    fold_percent: float = 0.5,
    verify_fold: bool = True,
) -> ShapingState:
  """Adds concentric inscribed pleat steps inside a flap or river."""
  return relief.inscribed_flap(
      state,
      element_name=element_name,
      depths=depths,
      fold_order=fold_order,
      fold_percent=fold_percent,
      verify_fold=verify_fold,
  ).state


def open_flap(
    state: ShapingState,
    *,
    element_name: str,
    fold_percent: float = 0.35,
    ridge_only: bool = False,
    verify_fold: bool = True,
) -> ShapingState:
  """Opens interior pleats or ridges of a flap or river into a 3-D relief."""
  return relief.open_flap(
      state,
      element_name=element_name,
      fold_percent=fold_percent,
      ridge_only=ridge_only,
      verify_fold=verify_fold,
  ).state


def fold_spine(
    state: ShapingState,
    *,
    fold_percent: float = 0.65,
    element_name: str | None = None,
    fold_type: str = 'mountain',
    verify_fold: bool = True,
) -> ShapingState:
  """Partially folds the central `x = 0.5` spine to arch or tent the body."""
  return relief.fold_spine(
      state,
      fold_percent=fold_percent,
      element_name=element_name,
      fold_type=fold_type,
      verify_fold=verify_fold,
  ).state


# ------------------------------------------------------------------------------
# 12. Simultaneous 3-D spherical rotation & optional narrowing (`rotation_3d`).
# ------------------------------------------------------------------------------


def _rotation_3d_outcome(
    state: ShapingState,
    *,
    rotations: Sequence[Mapping[str, Any]],
    require_all: bool = False,
    allow_narrowed: bool = False,
    include_unlisted: bool = False,
    use_tree_root: bool | None = None,
) -> state_lib.ToolOutcome:
  """Rotates elements to target `(az, el)` angles, optionally narrowing them."""
  if not rotations:
    raise ValueError('rotations must not be empty.')

  def _compose_converter(
      st: ShapingState, step_conv: dict[str, str]
  ) -> dict[str, str]:
    base_conv = st.merge_converter
    if not base_conv:
      return step_conv
    return {k: base_conv.get(v, v) for k, v in step_conv.items()}

  def _impl(st: ShapingState) -> state_lib.ToolOutcome:
    effective_use_root = (
        st.config.use_tree_root
        if use_tree_root is None
        else bool(use_tree_root)
    )
    unknown = sorted({
        str(r['element_name'])
        for r in rotations
        if str(r['element_name']) not in st.tree
    })
    if unknown:
      raise ValueError(
          f'rotation_3d: unknown elements {unknown}; known:'
          f' {st.element_names()}.'
      )

    wants_narrow = [
        str(r['element_name']) for r in rotations if r.get('narrow', False)
    ]
    if wants_narrow or include_unlisted:
      requested = {str(r['element_name']) for r in rotations}
      tree_updates: list[dict[str, Any]] = []
      for item in rotations:
        name = str(item['element_name'])
        stick = st.tree[name]
        tree_updates.append({
            'element_name': name,
            'depth': float(item.get('depth', 0.0)),
            'azimuth': float(item.get('az', stick.azimuth_angle)),
            'elevation': float(item.get('el', stick.elevation_angle)),
            'width': (
                0.5 if item.get('narrow', False) else max(1.0, stick.width)
            ),
        })
      for stick in st.tree.children:
        if stick.name not in requested and stick.width < 1.0:
          tree_updates.append({
              'element_name': stick.name,
              'depth': 0.0,
              'azimuth': stick.azimuth_angle,
              'elevation': stick.elevation_angle,
              'width': 1.0,
          })
      tree, updated_names, converter = (
          stick_figure_lib.split_by_frac_depth_and_update_tree(
              st.tree, tree_updates
          )
      )
      subset = None if include_unlisted else updated_names[: len(rotations)]
      flaps, rivers = st.merged_shaping_elements
      shaped = combined_heuristics.shape_cp_with_combined_heuristics(
          base_cp=st.pattern,
          stick_figure=tree,
          packing_output=st.packing,
          solving_output=st.solving,
          solving_grid_size=st.solving_grid_size,
          central_crease_removed=st.central_crease_removed,
          merge_converter=_compose_converter(st, converter),
          hparams=combined_heuristics.CombinedHeuristicsHParams(
              geometric_fold_hparams=geometric_fold_3d_heuristics.GeometricFolding3DHParams(
                  default_inscribe_hparams=geometric_fold_3d_heuristics.InscribeHParams(
                      avoid_hinges=True,
                  ),
                  check_ff_violations=True,
                  strategy=heuristics_3d.FoldStrategy3D.SHALLOW_FOLDS,
                  shape_subset=subset,
                  use_tree_root=effective_use_root,
              )
          ),
          shaping_flaps=flaps,
          shaping_rivers=rivers,
          graph_data=st.base_graph_data,
      )
      out = st.with_pattern(shaped, allow_3d_vertices=True)
      if wants_narrow:
        out = out.with_narrowed(*wants_narrow)
      delta = len(out.pattern.creases) - len(st.pattern.creases)
      return state_lib.ToolOutcome(
          state=out,
          note=(
              f'rotation_3d: rotated/narrowed {len(rotations)} element(s),'
              f' +{delta} creases.'
          ),
      )

    if st.narrowed_elements and not allow_narrowed:
      _, merged_map = st.tree.merge_linear_chains()
      narrowed_set = set(st.narrowed_elements)
      blocked: list[str] = []
      for r in rotations:
        name = str(r['element_name'])
        if name in narrowed_set:
          blocked.append(name)
          continue
        for sub_names in merged_map.values():
          if name in sub_names and narrowed_set.intersection(sub_names):
            blocked.append(name)
            break
      if blocked:
        raise ValueError(
            f'rotation_3d: elements {sorted(set(blocked))} lie on narrowed'
            ' flaps; pass allow_narrowed=True, narrow after rotation_3d, or'
            ' use crimp_fold/reverse_fold/bend_to_direction.'
        )

    flaps, rivers = st.merged_shaping_elements

    def _run_attempt(
        items: Sequence[Mapping[str, Any]], priority: Sequence[str]
    ) -> tuple[Any, set[str], list[str], dict[str, str]]:
      updates: list[dict[str, Any]] = []
      for r in items:
        upd: dict[str, Any] = {
            'element_name': str(r['element_name']),
            'depth': float(r.get('depth', 0.0)),
        }
        if 'az' in r and r['az'] is not None:
          upd['azimuth'] = float(r['az'])
        if 'el' in r and r['el'] is not None:
          upd['elevation'] = float(r['el'])
        updates.append(upd)
      tree, elem_names, converter = (
          stick_figure_lib.split_by_frac_depth_and_update_tree(st.tree, updates)
      )
      res = geometric_fold_3d_heuristics.get_shaping_creases_with_3d_geometric_folding_heuristics(
          base_cp=st.pattern,
          stick_figure=tree,
          shaping_flaps=flaps,
          shaping_rivers=rivers,
          hinge_info_list=st.updated_hinge_info(),
          central_crease_removed=st.central_crease_removed,
          shape_with_unmerged_tree=True,
          merge_converter=_compose_converter(st, converter),
          hparams=geometric_fold_3d_heuristics.GeometricFolding3DHParams(
              shape_subset=elem_names,
              check_ff_violations=True,
              strategy=st.config.fold_strategy_3d,
              default_inscribe_hparams=geometric_fold_3d_heuristics.InscribeHParams(
                  avoid_hinges=True,
                  clip_slope_thresh=st.config.clip_slope_thresh,
                  max_slope_thresh=st.config.max_slope_thresh,
              ),
              priority=tuple(priority),
              direction_graph_data=st.base_graph_data,
              use_tree_root=effective_use_root,
          ),
          graph_data=st.graph_data,
          sacrificial_flaps=st.packing.sacrificial_flaps or [],
      )
      shaped_set = {
          name
          for name, infos in res['detailed_shaping_info'].items()
          if any(info['shaping_creases'] for info in infos)
      }
      reasons = dict(res.get('skip_reasons', {}))
      return res, shaped_set, list(elem_names), reasons

    def _ff_conflicts(
        attempt: tuple[Any, set[str], list[str], dict[str, str]],
    ) -> list[str]:
      _, shaped_set, elem_names, reasons = attempt
      return [
          n
          for n in elem_names
          if n not in shaped_set and reasons.get(n) == 'flat_foldability'
      ]

    best = _run_attempt(rotations, ())
    conflicts = _ff_conflicts(best)
    if conflicts:
      retry = _run_attempt(rotations, conflicts)
      if len(retry[1]) > len(best[1]):
        best = retry
    conflicts = _ff_conflicts(best)
    if conflicts:
      deeper = [
          dict(r, depth=_ROTATION_RETRY_DEPTH)
          if str(r['element_name']) in conflicts and not r.get('depth')
          else r
          for r in rotations
      ]
      if deeper != list(rotations):
        retry = _run_attempt(deeper, conflicts)
        if len(retry[1]) > len(best[1]):
          best = retry

    res, shaped_set, elem_names, reasons = best
    skipped = [
        f'{n} ({reasons.get(n, "no_creases")})'
        for n in elem_names
        if n not in shaped_set
    ]
    if not res['shaping_creases']:
      raise ValueError(
          'rotation_3d added no creases: ' + '; '.join(skipped or elem_names)
      )
    if skipped and require_all:
      raise ValueError(
          'rotation_3d could not rotate all elements (require_all=True): '
          + '; '.join(skipped)
      )
    shaped = st.pattern.update_with_shaping_creases(
        res['shaping_creases'],
        split_shaping_by_shaping=st.config.check_all_crease_intersections,
    )
    out = st.with_pattern(shaped, allow_3d_vertices=True)
    note = f'Rotated {sorted(shaped_set)}.'
    if skipped:
      note += f' Skipped: {"; ".join(skipped)}.'
    return state_lib.ToolOutcome(state=out, note=note)

  return state_lib.run_with_3d_overlay(state, _impl, verify_fold=False)


def rotation_3d(
    state: ShapingState,
    *,
    rotations: Sequence[Mapping[str, Any]],
    require_all: bool = False,
    allow_narrowed: bool = False,
    include_unlisted: bool = False,
    use_tree_root: bool | None = None,
) -> ShapingState:
  """Rotates elements to target 3-D `(az, el)` angles and optional `narrow`."""
  return _rotation_3d_outcome(
      state,
      rotations=rotations,
      require_all=require_all,
      allow_narrowed=allow_narrowed,
      include_unlisted=include_unlisted,
      use_tree_root=use_tree_root,
  ).state


# ------------------------------------------------------------------------------
# 13–14. Closed-loop 3-D aiming (`bend_to_direction`) and thin-limb search
#        (`thin_limbs_3d`).
# ------------------------------------------------------------------------------


def _spherical_unit(az_deg: float, el_deg: float) -> np.ndarray:
  az = math.radians(az_deg)
  el = math.radians(el_deg)
  c = math.cos(el)
  return np.array(
      [c * math.cos(az), c * math.sin(az), math.sin(el)], dtype=float
  )


def _angular_error_deg(a: np.ndarray, b: np.ndarray) -> float:
  na = float(np.linalg.norm(a))
  nb = float(np.linalg.norm(b))
  if na < 1e-9 or nb < 1e-9:
    return 180.0
  dot = max(-1.0, min(1.0, float(np.dot(a / na, b / nb))))
  return math.degrees(math.acos(dot))


def limb_direction_errors(
    state: ShapingState,
    model: folder_3d.FoldedModel | None = None,
) -> dict[str, float]:
  """Returns per-stick angular direction errors (degrees) in the root frame."""
  _, dirs = measured_directions(state, model=model)
  return {
      name: float(info['error'])
      for name, info in dirs.items()
      if info.get('error') is not None
  }


def _bend_to_direction_outcome(
    state: ShapingState,
    *,
    element_name: str,
    az: float | None = None,
    el: float | None = None,
    depth: float = 0.45,
    measure: str | None = None,
    tolerance: float = 12.0,
    max_candidates: int = 12,
    chosen: Mapping[str, Any] | None = None,
) -> state_lib.ToolOutcome:
  """Searches local 3-D crimp/reverse cuts steering `element_name` to target."""
  if element_name not in state.tree:
    raise ValueError(f'bend_to_direction: unknown element {element_name!r}.')
  measured_name = measure or element_name
  if measured_name not in state.tree:
    raise ValueError(f'bend_to_direction: unknown measure {measured_name!r}.')

  if chosen is not None:
    tool_kind = str(chosen['tool'])
    ang = float(chosen['angle'])
    dep = float(chosen['depth'])
    if tool_kind == 'reverse':
      out = _reverse_3d_outcome(
          state, element_name=element_name, angle=ang, depth=dep
      )
    elif tool_kind == 'crimp':
      out = _crimp_3d_outcome(
          state, element_name=element_name, angle=ang, depth=dep
      )
    else:
      out = _simple_fold_outcome(
          state,
          element_name=element_name,
          angle=ang,
          depth=dep,
          verify_fold=True,
      )
    return out

  stick = state.tree[measured_name]
  target_az = float(stick.azimuth_angle if az is None else az)
  target_el = float(stick.elevation_angle if el is None else el)
  goal = _spherical_unit(target_az, target_el)

  def _score(st: ShapingState) -> tuple[float, float] | None:
    ok, _, model = verify_3d_fold(st.pattern)
    if not ok or model is None:
      return None
    _, dirs = measured_directions(st, model=model)
    realized = dirs.get(measured_name, {}).get('realized')
    if realized is None:
      return None
    vec = _spherical_unit(float(realized[0]), float(realized[1]))
    err = _angular_error_deg(vec, goal)
    other_errs = [
        float(info['error'])
        for n, info in dirs.items()
        if n != measured_name and info.get('error') is not None
    ]
    mean_other = sum(other_errs) / len(other_errs) if other_errs else 0.0
    return err, mean_other

  base = _score(state)
  if base is None:
    raise ValueError('bend_to_direction: current crease pattern does not fold.')
  base_err, base_other = base
  if base_err <= tolerance:
    return state_lib.ToolOutcome(
        state=state,
        note=f'{measured_name}: already within {base_err:.1f} deg of target.',
    )

  angles = (30.0, 55.0, 85.0, 120.0)
  depths = tuple(
      dict.fromkeys((round(depth, 2), 0.25 if depth >= 0.35 else 0.55))
  )
  best_slot: list[tuple[float, float, ShapingState, str, float, float]] = []
  tried = 0

  def _try(kind: str, dep: float, ang: float) -> bool:
    nonlocal tried
    if tried >= max_candidates:
      return False
    tried += 1
    try:
      if kind == 'crimp':
        cand = _crimp_3d_outcome(
            state,
            element_name=element_name,
            angle=ang,
            depth=dep,
            verify_fold=False,
        ).state
      else:
        cand = _reverse_3d_outcome(
            state,
            element_name=element_name,
            angle=ang,
            depth=dep,
            verify_fold=False,
        ).state
    except ValueError:
      return False
    sc = _score(cand)
    if sc is None:
      return False
    err, other = sc
    cost = err + 0.5 * max(0.0, other - base_other)
    if not best_slot or cost < best_slot[0][0]:
      best_slot[:] = [(cost, err, cand, kind, ang, dep)]
    return err <= tolerance and other <= base_other + 5.0

  first_mag = min(angles, key=lambda a: abs(a - base_err))
  found = False
  for dep in depths:
    for kind in ('crimp', 'reverse'):
      for sign in (1.0, -1.0):
        if _try(kind, dep, sign * first_mag):
          found = True
          break
      if found:
        break
    if found:
      break

  if not found and best_slot:
    _, _, _, best_kind, best_ang, best_dep = best_slot[0]
    sgn = math.copysign(1.0, best_ang)
    for mag in sorted(angles, key=lambda a: abs(a - base_err)):
      if math.isclose(mag, abs(best_ang)):
        continue
      if _try(best_kind, best_dep, sgn * mag):
        break

  if not best_slot or best_slot[0][1] >= base_err - 1.0:
    raise ValueError(
        f'bend_to_direction({element_name}): no candidate improved error below'
        f' {base_err:.1f} deg ({tried} tried).'
    )
  _, err, cand, kind, ang, dep = best_slot[0]
  return state_lib.ToolOutcome(
      state=cand,
      note=(
          f'{measured_name}: {base_err:.1f}->{err:.1f} deg via'
          f' {kind}(angle={ang:g}, depth={dep:g}).'
      ),
      resolved_arguments={
          'element_name': element_name,
          'az': target_az,
          'el': target_el,
          'measure': measured_name,
          'chosen': {'tool': kind, 'angle': ang, 'depth': dep},
      },
  )


def bend_to_direction(
    state: ShapingState,
    *,
    element_name: str,
    az: float | None = None,
    el: float | None = None,
    depth: float = 0.45,
    measure: str | None = None,
    tolerance: float = 12.0,
    max_candidates: int = 12,
    chosen: Mapping[str, Any] | None = None,
) -> ShapingState:
  """Steers a flap to target root-frame `(az, el)` via closed-loop 3-D cuts."""
  return _bend_to_direction_outcome(
      state,
      element_name=element_name,
      az=az,
      el=el,
      depth=depth,
      measure=measure,
      tolerance=tolerance,
      max_candidates=max_candidates,
      chosen=chosen,
  ).state


def _apply_thin_spec(
    state: ShapingState, element_name: str, spec: Mapping[str, Any]
) -> ShapingState:
  """Applies one bend/knee action dictionary in `thin_limbs_3d`."""
  kind = str(spec.get('kind', 'simple'))
  angle = float(spec['angle'])
  depth = float(spec.get('depth', _THIN_BEND_DEPTH))
  pct = float(spec.get('fold_percent', 1.0))
  order = spec.get('fold_order')
  if kind == 'crimp':
    return _crimp_3d_outcome(
        state,
        element_name=element_name,
        angle=angle,
        depth=depth,
        fold_percent=pct,
        verify_fold=False,
    ).state
  if kind == 'reverse':
    return _reverse_3d_outcome(
        state,
        element_name=element_name,
        angle=angle,
        depth=depth,
        fold_percent=pct,
        verify_fold=False,
    ).state
  return _simple_fold_outcome(
      state,
      element_name=element_name,
      angle=angle,
      depth=depth,
      narrow=False,
      fold_percent=pct,
      fold_order=str(order) if order is not None else None,
      verify_fold=False,
  ).state


def _mirror_thin_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
  mirrored = dict(spec)
  if 'angle' in mirrored:
    mirrored['angle'] = -float(mirrored['angle'])
  if mirrored.get('fold_order') == 'MV':
    mirrored['fold_order'] = 'VM'
  elif mirrored.get('fold_order') == 'VM':
    mirrored['fold_order'] = 'MV'
  return mirrored


def _thin_limbs_3d_outcome(
    state: ShapingState,
    *,
    limbs: Sequence[Mapping[str, Any]],
    evaluate: Callable[[ShapingState], Mapping[str, float]] | None = None,
) -> state_lib.ToolOutcome:
  """Narrows thin limbs and searches batched 3-D bend + knee angles."""
  if not limbs:
    raise ValueError('limbs must not be empty.')
  plans: list[dict[str, Any]] = []
  for item in limbs:
    prox = str(item['proximal'])
    dist = str(item['distal']) if item.get('distal') is not None else None
    if prox not in state.tree:
      raise ValueError(f'thin_limbs_3d: unknown proximal stick {prox!r}.')
    if dist is not None and dist not in state.tree:
      raise ValueError(f'thin_limbs_3d: unknown distal stick {dist!r}.')
    plan: dict[str, Any] = {'proximal': prox, 'distal': dist}
    for key in ('narrow_angle', 'bend', 'knee'):
      if key in item:
        plan[key] = item[key]
    if dist is None:
      plan['knee'] = None
    plans.append(plan)

  def _eval(st: ShapingState) -> dict[str, float]:
    if evaluate is not None:
      return {str(k): float(v) for k, v in evaluate(st).items()}
    ok, reason, model = verify_3d_fold(st.pattern)
    if not ok or model is None:
      raise ValueError(reason)
    return limb_direction_errors(st, model=model)

  def _limb_err(errs: Mapping[str, float], idx: int, stage: str) -> float:
    prox = plans[idx]['proximal']
    dist = plans[idx]['distal']
    if stage == 'knee' and dist is not None:
      return float(errs.get(dist, errs.get(prox, 180.0)))
    vals = [errs[n] for n in (prox, dist) if n is not None and n in errs]
    return sum(vals) / len(vals) if vals else 180.0

  def _apply_stage(
      st: ShapingState, stage: str, choices: Mapping[int, Any]
  ) -> tuple[ShapingState, set[int]]:
    cur = st
    applied: set[int] = set()
    for idx, val in choices.items():
      if val is None:
        continue
      target = (
          plans[idx]['distal'] if stage == 'knee' else plans[idx]['proximal']
      )
      if target is None:
        continue
      try:
        if stage == 'narrow_angle':
          cur = _simple_fold_outcome(
              cur,
              element_name=target,
              angle=float(val),
              narrow=True,
              verify_fold=False,
          ).state
        else:
          cur = _apply_thin_spec(cur, target, val)
        applied.add(idx)
      except (ValueError, AssertionError, KeyError, IndexError):
        continue
    return cur, applied

  # Group bilateral partner indices so left/right limbs mirror one another.
  index_by_prox = {p['proximal']: i for i, p in enumerate(plans)}
  sym_partner: dict[int, int] = {}
  for i, p in enumerate(plans):
    if i in sym_partner:
      continue
    partner_name = resolve_symmetric_element(state, p['proximal'])
    j = index_by_prox.get(partner_name or '')
    if j is not None and j != i and j not in sym_partner:
      sym_partner[i] = j
      sym_partner[j] = i

  def _bend_specs(depth: float) -> list[dict[str, Any]]:
    return [
        {
            'kind': 'simple',
            'angle': 180.0,
            'depth': depth,
            'fold_order': 'MV',
            'fold_percent': 0.5,
        },
        {
            'kind': 'simple',
            'angle': 180.0,
            'depth': depth,
            'fold_order': 'VM',
            'fold_percent': 0.5,
        },
        {
            'kind': 'simple',
            'angle': 90.0,
            'depth': depth,
            'fold_order': 'MV',
            'fold_percent': 0.5,
        },
        {
            'kind': 'simple',
            'angle': -90.0,
            'depth': depth,
            'fold_order': 'VM',
            'fold_percent': 0.5,
        },
        {'kind': 'crimp', 'angle': 60.0, 'depth': depth, 'fold_percent': 1.0},
        {'kind': 'crimp', 'angle': -60.0, 'depth': depth, 'fold_percent': 1.0},
    ]

  stage_pools: tuple[tuple[str, Sequence[Any]], ...] = (
      ('narrow_angle', (45.0, -45.0, 90.0, -90.0)),
      ('bend', _bend_specs(_THIN_BEND_DEPTH)),
      ('knee', _bend_specs(_THIN_KNEE_DEPTH)),
  )

  current = state
  for stage, pool in stage_pools:
    pending = [i for i, p in enumerate(plans) if stage not in p]
    if pending:
      best: dict[int, tuple[float, Any]] = {
          i: (math.inf, None) for i in pending
      }
      if stage != 'narrow_angle':
        try:
          base_errs = _eval(current)
          for i in pending:
            cur_err = _limb_err(base_errs, i, stage)
            if cur_err < 15.0 and evaluate is None:
              plans[i][stage] = None
            else:
              best[i] = (cur_err - _THIN_MIN_GAIN_DEG, None)
          pending = [i for i in pending if stage not in plans[i]]
        except ValueError:
          pass

      # Batched parallel evaluation: apply k-th candidate to all pending limbs
      # simultaneously (mirroring right partners of paired limbs).
      for cand in pool:
        if not pending:
          break
        batch: dict[int, Any] = {}
        for i in pending:
          j = sym_partner.get(i)
          if j is not None and j in pending and i > j:
            batch[i] = (
                -float(cand)
                if stage == 'narrow_angle'
                else _mirror_thin_spec(cand)
            )
          else:
            batch[i] = cand
        trial, applied = _apply_stage(current, stage, batch)
        if not applied:
          continue
        try:
          errs = _eval(trial)
        except ValueError:
          continue
        for i in applied:
          score = _limb_err(errs, i, stage)
          j = sym_partner.get(i)
          if j is not None and j in applied:
            if i > j:
              continue
            pair_score = 0.5 * (score + _limb_err(errs, j, stage))
            prev = 0.5 * (best[i][0] + best[j][0])
            if pair_score < prev:
              best[i] = (pair_score, batch[i])
              best[j] = (pair_score, batch[j])
          elif score < best[i][0]:
            best[i] = (score, batch[i])

      for i in pending:
        plans[i][stage] = best[i][1]

    current, applied = _apply_stage(
        current, stage, {i: p.get(stage) for i, p in enumerate(plans)}
    )
    if stage == 'narrow_angle':
      if not applied:
        raise ValueError('thin_limbs_3d could not narrow any requested limb.')
      for i, p in enumerate(plans):
        if i not in applied:
          p.update(narrow_angle=None, bend=None, knee=None)

  final_errs = _eval(current)
  parts = [
      f'{p["proximal"]} ({final_errs.get(p["proximal"], 0.0):.0f} deg)'
      for p in plans
      if p.get('narrow_angle') is not None
  ]
  return state_lib.ToolOutcome(
      state=current,
      note='Posed thin limbs: ' + ', '.join(parts) + '.',
      resolved_arguments={'limbs': plans},
  )


def thin_limbs_3d(
    state: ShapingState,
    *,
    limbs: Sequence[Mapping[str, Any]],
    evaluate: Callable[[ShapingState], Mapping[str, float]] | None = None,
) -> ShapingState:
  """Narrows width-1 limbs and jointly searches 3-D bend and knee angles."""
  return _thin_limbs_3d_outcome(state, limbs=limbs, evaluate=evaluate).state


# ------------------------------------------------------------------------------
# 15–17. Multi-station curvature (`curve_flap`), fork spreading (`spread_fork`),
#        and circular crown fanning (`radial_fan`).
# ------------------------------------------------------------------------------


def _chain_suffix(state: ShapingState, element_name: str) -> list[str]:
  _, merged_map = state.tree.merge_linear_chains()
  for names in merged_map.values():
    seq = list(names)
    if element_name in seq:
      return seq[seq.index(element_name) :]
  return [element_name]


def _curve_flap_outcome(
    state: ShapingState,
    *,
    element_name: str,
    n_segments: int = 3,
    total_bend_deg: float = 90.0,
    direction: str = 'up',
    method: str = 'auto',
    fold_percent: float = 1.0,
    verify_fold: bool = True,
) -> state_lib.ToolOutcome:
  """Curves a flap or linear chain via graduated multi-station bends."""
  if n_segments < 1:
    raise ValueError(f'n_segments must be >= 1, got {n_segments}.')
  if math.isclose(total_bend_deg, 0.0):
    raise ValueError('total_bend_deg must be non-zero.')
  dir_norm = direction.lower()
  dir_map = {
      'up': np.array([0.0, 0.0, 1.0]),
      'down': np.array([0.0, 0.0, -1.0]),
      'left': np.array([0.0, 1.0, 0.0]),
      'right': np.array([0.0, -1.0, 0.0]),
      'forward': np.array([1.0, 0.0, 0.0]),
      'backward': np.array([-1.0, 0.0, 0.0]),
      'positive': None,
      'negative': None,
  }
  if dir_norm not in dir_map:
    raise ValueError(
        f'direction must be one of {sorted(dir_map)}, got {direction!r}.'
    )
  if method not in ('auto', 'crimp', 'reverse', 'tree_edit'):
    raise ValueError(
        "method must be 'auto', 'crimp', 'reverse', or 'tree_edit'; got"
        f' {method!r}.'
    )
  if element_name not in state.tree:
    raise ValueError(f'curve_flap: unknown element {element_name!r}.')

  if method == 'tree_edit':
    stick = state.tree[element_name]
    total_len = int(round(stick.length))
    if stick.is_river or total_len < n_segments:
      raise ValueError(
          f'{element_name!r} (length {total_len}) cannot subdivide into'
          f' {n_segments} integer segments.'
      )
    base_len, rem = divmod(total_len, n_segments)
    lengths = [base_len + (1 if i < rem else 0) for i in range(n_segments)]
    step_deg = float(total_bend_deg) / n_segments
    az_map: dict[str, float] = {}
    el_map: dict[str, float] = {}
    for i in range(n_segments):
      sub = f'{element_name}_{i}'
      frac = i + 1
      if dir_norm in ('left', 'right'):
        sgn = 1.0 if dir_norm == 'left' else -1.0
        az_map[sub] = angle_utils.wrap_angle_centered(
            stick.azimuth_angle + sgn * step_deg * frac, 360.0
        )
      else:
        sgn = -1.0 if dir_norm in ('down', 'negative') else 1.0
        el_map[sub] = max(
            -89.0, min(89.0, stick.elevation_angle + sgn * step_deg * frac)
        )
    return _tree_edit_outcome(
        state,
        subdivisions={element_name: lengths},
        azimuths=az_map or None,
        elevations=el_map or None,
        only_edited=True,
        verify_fold=verify_fold,
    )

  chain = _chain_suffix(state, element_name)
  if len(chain) >= n_segments:
    stations = [(name, 0.35) for name in chain[:n_segments]]
  else:
    stations = [
        (element_name, round(0.18 + 0.64 * i / max(1, n_segments - 1), 3))
        for i in range(n_segments)
    ]
  batches = [stations]
  if len(stations) > 2:
    batches.append(stations[:2])
  if len(stations) > 1:
    batches.append([(element_name, 0.45)])

  methods = ('crimp', 'reverse') if method == 'auto' else (method,)
  signs = (
      (-1.0,)
      if dir_norm == 'negative'
      else ((1.0,) if dir_norm == 'positive' else (1.0, -1.0))
  )
  goal_vec = dir_map[dir_norm]
  tip_stick = chain[-1]

  def _realized_vec(
      st: ShapingState, model: folder_3d.FoldedModel
  ) -> np.ndarray | None:
    _, dirs = measured_directions(st, model=model)
    real = dirs.get(tip_stick, {}).get('realized')
    if real is None:
      return None
    return _spherical_unit(float(real[0]), float(real[1]))

  base_tip: np.ndarray | None = None
  if verify_fold and goal_vec is not None:
    ok_b, _, m_b = verify_3d_fold(state.pattern)
    if ok_b and m_b is not None:
      base_tip = _realized_vec(state, m_b)

  for batch in batches:
    step_mag = min(135.0, max(15.0, abs(total_bend_deg) / len(batch)))
    scored: list[tuple[float, ShapingState, int]] = []
    for base_sign in signs:
      cur = state
      applied = 0
      rev_parity = 1.0
      for seg_name, dep in batch:
        for kind in methods:
          ang = base_sign * rev_parity * step_mag
          try:
            if kind == 'crimp':
              cur = _crimp_3d_outcome(
                  cur,
                  element_name=seg_name,
                  angle=ang,
                  depth=dep,
                  fold_percent=fold_percent,
                  verify_fold=False,
              ).state
            else:
              cur = _reverse_3d_outcome(
                  cur,
                  element_name=seg_name,
                  angle=ang,
                  depth=dep,
                  fold_percent=fold_percent,
                  verify_fold=False,
              ).state
              rev_parity *= -1.0
            applied += 1
            break
          except ValueError:
            continue
      if applied == 0:
        continue
      if verify_fold:
        ok_f, _, m_f = verify_3d_fold(cur.pattern)
        if not ok_f or m_f is None:
          continue
        alignment = 0.0
        if goal_vec is not None:
          after_tip = _realized_vec(cur, m_f)
          if after_tip is not None:
            diff = after_tip - base_tip if base_tip is not None else after_tip
            alignment = float(np.dot(diff, goal_vec))
        scored.append((alignment, cur, applied))
      else:
        return state_lib.ToolOutcome(
            state=cur,
            note=(
                f'{element_name}: curved {applied}/{n_segments} station(s)'
                f' (step={step_mag:.1f} deg).'
            ),
        )
    if scored:
      scored.sort(key=lambda item: item[0], reverse=True)
      _, best_st, count = scored[0]
      delta = len(best_st.pattern.creases) - len(state.pattern.creases)
      return state_lib.ToolOutcome(
          state=best_st,
          note=(
              f'{element_name}: curved {count}/{n_segments} station(s)'
              f' ({dir_norm}), +{delta} creases.'
          ),
      )
  raise ValueError(
      f'curve_flap({element_name}): no multi-station candidate folded cleanly.'
  )


def curve_flap(
    state: ShapingState,
    *,
    element_name: str,
    n_segments: int = 3,
    total_bend_deg: float = 90.0,
    direction: str = 'up',
    method: str = 'auto',
    fold_percent: float = 1.0,
    verify_fold: bool = True,
) -> ShapingState:
  """Curves a flap across multiple graduated bend stations or subdivisions."""
  return _curve_flap_outcome(
      state,
      element_name=element_name,
      n_segments=n_segments,
      total_bend_deg=total_bend_deg,
      direction=direction,
      method=method,
      fold_percent=fold_percent,
      verify_fold=verify_fold,
  ).state


def _tip_separation(
    state: ShapingState,
    model: folder_3d.FoldedModel,
    element_a: str,
    element_b: str,
) -> float | None:
  """Measures 3-D distance (tree units) between distal tips of `a` and `b`."""
  topo = paper_layout_lib.topology(state.tree)
  ja = topo.distal.get(_chain_suffix(state, element_a)[-1], '')
  jb = topo.distal.get(_chain_suffix(state, element_b)[-1], '')
  layout = state.paper_layout
  if ja not in layout.joints or jb not in layout.joints:
    return None
  means, _ = model.locate_groups({ja: layout.joints[ja], jb: layout.joints[jb]})
  if ja not in means or jb not in means:
    return None
  scale = layout.grid_size / max(layout.grid_units_per_tree_unit, 1e-6)
  return float(np.linalg.norm(means[ja] - means[jb]) * scale)


def _spread_fork_outcome(
    state: ShapingState,
    *,
    element_a: str,
    element_b: str,
    spread_angle: float = 35.0,
    depth: float = 0.25,
    fold_percent: float = 1.0,
    out_of_plane: bool = False,
    verify_fold: bool = True,
) -> state_lib.ToolOutcome:
  """Spreads two neighboring flaps apart sequentially to maximize tip gap."""
  if element_a == element_b:
    raise ValueError('element_a and element_b must be distinct.')
  for name in (element_a, element_b):
    if name not in state.tree:
      raise ValueError(f'spread_fork: unknown element {name!r}.')
  if math.isclose(spread_angle, 0.0):
    raise ValueError('spread_angle must be non-zero.')

  mag = abs(float(spread_angle))
  fp_a = (
      0.75
      if out_of_plane and math.isclose(fold_percent, 1.0)
      else float(fold_percent)
  )
  fp_b = float(fold_percent)

  def _apply_one(
      st: ShapingState, elem: str, ang: float, fp: float
  ) -> ShapingState | None:
    for fn in (_crimp_3d_outcome, _reverse_3d_outcome):
      try:
        return fn(
            st,
            element_name=elem,
            angle=ang,
            depth=depth,
            fold_percent=fp,
            verify_fold=False,
        ).state
      except ValueError:
        continue
    try:
      return _simple_fold_outcome(
          st,
          element_name=elem,
          angle=ang,
          depth=depth,
          fold_percent=fp,
          verify_fold=False,
      ).state
    except ValueError:
      return None

  base_gap = 0.0
  if verify_fold:
    ok_b, _, m_b = verify_3d_fold(state.pattern)
    if ok_b and m_b is not None:
      base_gap = _tip_separation(state, m_b, element_a, element_b) or 0.0

  best: tuple[float, ShapingState, float, float] | None = None
  for sa, sb in ((1.0, -1.0), (-1.0, 1.0), (1.0, 1.0), (-1.0, -1.0)):
    ang_a, ang_b = sa * mag, sb * mag
    st_a = _apply_one(state, element_a, ang_a, fp_a)
    if st_a is None:
      continue
    st_ab = _apply_one(st_a, element_b, ang_b, fp_b)
    if st_ab is None:
      continue
    if not verify_fold:
      return state_lib.ToolOutcome(
          state=st_ab,
          note=(
              f'spread_fork({element_a}, {element_b}): ({ang_a:g}, {ang_b:g}).'
          ),
      )
    ok_f, _, m_f = verify_3d_fold(st_ab.pattern)
    if not ok_f or m_f is None:
      continue
    gap = _tip_separation(st_ab, m_f, element_a, element_b) or 0.0
    if best is None or gap > best[0]:
      best = (gap, st_ab, ang_a, ang_b)

  if best is None:
    raise ValueError(
        f'spread_fork({element_a}, {element_b}): no symmetric pair folded'
        ' cleanly.'
    )
  gap, out, ang_a, ang_b = best
  return state_lib.ToolOutcome(
      state=out,
      note=(
          f'spread_fork({element_a}, {element_b}): tip gap'
          f' {base_gap:.2f}->{gap:.2f} at ({ang_a:g}, {ang_b:g}) deg.'
      ),
  )


def spread_fork(
    state: ShapingState,
    *,
    element_a: str,
    element_b: str,
    spread_angle: float = 35.0,
    depth: float = 0.25,
    fold_percent: float = 1.0,
    out_of_plane: bool = False,
    verify_fold: bool = True,
) -> ShapingState:
  """Spreads two parallel or forked flaps apart into a V-shaped gap."""
  return _spread_fork_outcome(
      state,
      element_a=element_a,
      element_b=element_b,
      spread_angle=spread_angle,
      depth=depth,
      fold_percent=fold_percent,
      out_of_plane=out_of_plane,
      verify_fold=verify_fold,
  ).state


def _radial_fan_outcome(
    state: ShapingState,
    *,
    elements: Sequence[str],
    arc_deg: float | None = None,
    center_az: float | None = None,
    base_el: float | None = None,
    dome_el: float = 0.0,
    fan_wide: bool = True,
    open_percent: float = 0.65,
    verify_fold: bool = True,
) -> state_lib.ToolOutcome:
  """Distributes `elements` around an angular arc and flares wide rays."""
  names = [str(e) for e in elements]
  if len(names) < 2:
    raise ValueError('radial_fan requires at least 2 elements.')
  missing = [n for n in names if n not in state.tree]
  if missing:
    raise ValueError(f'radial_fan: unknown elements {missing}.')

  n = len(names)
  rotations: list[dict[str, Any]] = []
  if arc_deg is None:
    for idx, name in enumerate(names):
      stick = state.tree[name]
      frac = 2.0 * idx / max(1, n - 1) - 1.0
      el = float(
          (stick.elevation_angle if base_el is None else base_el)
          + dome_el * (1.0 - frac * frac)
      )
      rotations.append({
          'element_name': name,
          'az': float(stick.azimuth_angle),
          'el': max(-89.0, min(89.0, el)),
      })
  else:
    span = float(arc_deg)
    full_circle = span >= 359.0
    if center_az is None:
      vecs = [
          (
              math.cos(math.radians(state.tree[nm].azimuth_angle)),
              math.sin(math.radians(state.tree[nm].azimuth_angle)),
          )
          for nm in names
      ]
      c_az = math.degrees(
          math.atan2(sum(y for _, y in vecs), sum(x for x, _ in vecs))
      )
    else:
      c_az = float(center_az)
    for idx, name in enumerate(names):
      stick = state.tree[name]
      offset = (
          -0.5 * span + span * idx / n
          if full_circle
          else -0.5 * span + span * idx / max(1, n - 1)
      )
      frac = 2.0 * idx / max(1, n - 1) - 1.0
      el = float(
          (stick.elevation_angle if base_el is None else base_el)
          + dome_el * (1.0 - frac * frac)
      )
      rotations.append({
          'element_name': name,
          'az': angle_utils.wrap_angle_centered(c_az + offset, 360.0),
          'el': max(-89.0, min(89.0, el)),
      })

  current = state
  rotated = False
  try:
    cand = _rotation_3d_outcome(current, rotations=rotations).state
    if not verify_fold or verify_3d_fold(cand.pattern)[0]:
      current = cand
      rotated = True
  except ValueError:
    pass

  flared: list[str] = []
  if fan_wide:
    for name in names:
      stick = current.tree[name]
      if stick.is_river or stick.width < 2.0:
        continue
      for fn in (
          lambda st, nm=name: fan_squash.squash_fold(
              st,
              element_name=nm,
              open_percent=open_percent,
              verify_fold=verify_fold,
          ).state,
          lambda st, nm=name: fan_squash.fan_out_fold(
              st,
              element_name=nm,
              move_delta=1.0,
              open_percent=open_percent,
              verify_fold=verify_fold,
          ).state,
      ):
        try:
          current = fn(current)
          flared.append(name)
          break
        except ValueError:
          continue

  if not rotated and not flared:
    raise ValueError(
        f'radial_fan({names}): could not rotate or flare any ray cleanly.'
    )
  return state_lib.ToolOutcome(
      state=current,
      note=(
          f'radial_fan: {len(names)} rays (rotated={rotated}, flared={flared}).'
      ),
  )


def radial_fan(
    state: ShapingState,
    *,
    elements: Sequence[str],
    arc_deg: float | None = None,
    center_az: float | None = None,
    base_el: float | None = None,
    dome_el: float = 0.0,
    fan_wide: bool = True,
    open_percent: float = 0.65,
    verify_fold: bool = True,
) -> ShapingState:
  """Arranges multiple petals, tentacles, or tail feathers into a 3-D arc."""
  return _radial_fan_outcome(
      state,
      elements=elements,
      arc_deg=arc_deg,
      center_az=center_az,
      base_el=base_el,
      dome_el=dome_el,
      fan_wide=fan_wide,
      open_percent=open_percent,
      verify_fold=verify_fold,
  ).state


# ------------------------------------------------------------------------------
# 18. Bilateral mirror wrapper (`apply_symmetric`).
# ------------------------------------------------------------------------------


def _partner_arg_variants(
    tool_name: str, base_args: Mapping[str, Any]
) -> list[dict[str, Any]]:
  """Builds mirrored-then-matched argument dicts for a symmetric partner."""
  mirrored = dict(base_args)
  matched = dict(base_args)
  for key in ('angle', 'rotate_angle', 'move_delta', 'az'):
    if key in mirrored and mirrored[key] is not None:
      mirrored[key] = -float(mirrored[key])
  if mirrored.get('direction') == 'left':
    mirrored['direction'] = 'right'
  elif mirrored.get('direction') == 'right':
    mirrored['direction'] = 'left'
  if mirrored.get('fold_order') == 'MV':
    mirrored['fold_order'] = 'VM'
  elif mirrored.get('fold_order') == 'VM':
    mirrored['fold_order'] = 'MV'
  if tool_name in ('rotation_3d', 'bend_to_direction'):
    return [mirrored]
  return [mirrored, matched] if mirrored != matched else [matched]


def _pair_asymmetry(
    state: ShapingState,
    model: folder_3d.FoldedModel,
    left_name: str,
    right_name: str,
) -> float:
  """Measures bilateral angle mismatch in `[0, 1]` (`0` = symmetric)."""
  _, dirs = measured_directions(state, model=model)
  rl = dirs.get(left_name, {}).get('realized')
  rr = dirs.get(right_name, {}).get('realized')
  if rl is None or rr is None:
    return 0.0
  vl = _spherical_unit(float(rl[0]), float(rl[1]))
  vr = _spherical_unit(float(rr[0]), float(rr[1]))
  vr_mirrored = np.array([vr[0], -vr[1], vr[2]], dtype=float)
  return _angular_error_deg(vl, vr_mirrored) / 180.0


def _apply_symmetric_outcome(
    state: ShapingState,
    *,
    tool: str,
    element_name: str,
    partner_name: str | None = None,
    arguments: Mapping[str, Any] | None = None,
    min_symmetry: float | None = None,
    left_arguments: Mapping[str, Any] | None = None,
    right_arguments: Mapping[str, Any] | None = None,
) -> state_lib.ToolOutcome:
  """Applies `tool` symmetrically to `element_name` and its mirror partner."""
  inner = normalize_tool_name(tool)
  if inner in (
      'apply_symmetric',
      'tree_edit',
      'geometric_simple_fold',
      'fold_spine',
      'radial_fan',
      'spread_fork',
  ):
    raise ValueError(
        f'apply_symmetric does not wrap global/multi-element tool {inner!r}.'
    )
  if inner not in _TOOL_OUTCOMES:
    raise ValueError(
        f'apply_symmetric: unknown tool {tool!r}; known: {TOOL_NAMES}.'
    )

  partner = partner_name or resolve_symmetric_element(state, element_name)
  if not partner or partner == element_name:
    raise ValueError(
        f'apply_symmetric: no bilateral partner found for {element_name!r};'
        ' pass partner_name explicitly.'
    )

  base = {k: v for k, v in (arguments or {}).items() if k != 'element_name'}

  # Simultaneous tools run once over both sticks.
  if inner == 'rotation_3d':
    l_args = dict(left_arguments or base)
    r_args = dict(right_arguments or _partner_arg_variants(inner, l_args)[0])
    l_item = {'element_name': element_name, **l_args}
    r_item = {'element_name': partner, **r_args}
    if 'az' not in l_item and element_name in state.tree:
      l_item['az'] = state.tree[element_name].azimuth_angle
      r_item['az'] = state.tree[partner].azimuth_angle
    if 'el' not in l_item and element_name in state.tree:
      l_item['el'] = state.tree[element_name].elevation_angle
      r_item['el'] = state.tree[partner].elevation_angle
    return _rotation_3d_outcome(
        state, rotations=[l_item, r_item], require_all=True
    )

  if inner == 'thin_limbs_3d':
    dist_l = base.get('distal')
    dist_r = (
        resolve_symmetric_element(state, str(dist_l))
        if dist_l is not None
        else None
    )
    return _thin_limbs_3d_outcome(
        state,
        limbs=[
            {'proximal': element_name, 'distal': dist_l},
            {'proximal': partner, 'distal': dist_r},
        ],
    )

  runner = _TOOL_OUTCOMES[inner]
  if left_arguments is not None and right_arguments is not None:
    mid = runner(state, element_name=element_name, **dict(left_arguments))
    fin = runner(mid.state, element_name=partner, **dict(right_arguments))
    return state_lib.ToolOutcome(
        state=fin.state,
        note=f'Symmetric {inner} replayed on ({element_name}, {partner}).',
    )

  left_out = runner(state, element_name=element_name, **base)
  eff_left = dict(left_out.resolved_arguments or base)
  eff_left.pop('element_name', None)
  eff_left.pop('measure', None)
  if 'chosen' in eff_left and isinstance(eff_left['chosen'], Mapping):
    ch = dict(eff_left['chosen'])
    ch['angle'] = -float(ch['angle'])
    right_variants = [{'chosen': ch}, base]
  else:
    right_variants = _partner_arg_variants(inner, base)

  verify = bool(base.get('verify_fold', True))
  best: tuple[float, ShapingState, dict[str, Any]] | None = None
  last_reason = 'partner call failed'
  for r_args in right_variants:
    try:
      r_out = runner(left_out.state, element_name=partner, **r_args)
    except ValueError as error:
      last_reason = str(error)
      continue
    eff_right = dict(r_out.resolved_arguments or r_args)
    eff_right.pop('element_name', None)
    if not verify:
      best = (0.0, r_out.state, eff_right)
      break
    ok_f, reason, model = verify_3d_fold(r_out.state.pattern)
    if not ok_f or model is None:
      last_reason = reason
      continue
    asym = _pair_asymmetry(r_out.state, model, element_name, partner)
    if best is None or asym < best[0]:
      best = (asym, r_out.state, eff_right)
    if asym <= 0.05:
      break

  if best is None:
    raise ValueError(
        f'apply_symmetric({inner}, {element_name}, {partner}) failed:'
        f' {last_reason}'
    )
  asym, out_state, chosen_right = best
  symmetry = 1.0 - asym
  if min_symmetry is not None and symmetry < float(min_symmetry):
    raise ValueError(
        f'apply_symmetric({inner}): bilateral symmetry {symmetry:.2f} <'
        f' min_symmetry={min_symmetry}.'
    )
  return state_lib.ToolOutcome(
      state=out_state,
      note=(
          f'Symmetric {inner}({element_name}, {partner}), symmetry'
          f' {symmetry:.2f}.'
      ),
      resolved_arguments={
          'tool': inner,
          'element_name': element_name,
          'partner_name': partner,
          'left_arguments': eff_left,
          'right_arguments': chosen_right,
      },
  )


def apply_symmetric(
    state: ShapingState,
    *,
    tool: str,
    element_name: str,
    partner_name: str | None = None,
    arguments: Mapping[str, Any] | None = None,
    min_symmetry: float | None = None,
    left_arguments: Mapping[str, Any] | None = None,
    right_arguments: Mapping[str, Any] | None = None,
) -> ShapingState:
  """Applies a single-element actuator symmetrically to a bilateral pair."""
  return _apply_symmetric_outcome(
      state,
      tool=tool,
      element_name=element_name,
      partner_name=partner_name,
      arguments=arguments,
      min_symmetry=min_symmetry,
      left_arguments=left_arguments,
      right_arguments=right_arguments,
  ).state


# ------------------------------------------------------------------------------
# 19. Packing-equivalent stick-figure editing (`tree_edit`).
# ------------------------------------------------------------------------------


def _build_edited_tree(
    base_tree: stick_figure_lib.StickFigure,
    *,
    subdivisions: Mapping[str, Sequence[float]] | None,
    azimuths: Mapping[str, float] | None,
    elevations: Mapping[str, float] | None,
    widths: Mapping[str, float] | None,
) -> tuple[stick_figure_lib.StickFigure, dict[str, str], list[str]]:
  """Constructs a packing-equivalent edited tree and sub-stick mapping."""
  _, base_merged_map = base_tree.merge_linear_chains()
  unmerged_to_merged: dict[str, str] = {}
  for m_name, subs in base_merged_map.items():
    for sub in subs:
      unmerged_to_merged[sub] = m_name
  az_map = {str(k): float(v) for k, v in (azimuths or {}).items()}
  el_map = {str(k): float(v) for k, v in (elevations or {}).items()}
  w_map = {str(k): float(v) for k, v in (widths or {}).items()}

  int_sub_map: dict[str, list[int]] = {}
  known_base = {s.name for s in base_tree.children}
  for raw_name, raw_parts in (subdivisions or {}).items():
    name = str(raw_name)
    if name not in known_base:
      raise ValueError(f'tree_edit: unknown stick {name!r} in subdivisions.')
    stick = base_tree[name]
    if stick.is_river:
      raise ValueError(f'Cannot subdivide river {name!r}.')
    parts = [float(x) for x in raw_parts]
    if len(parts) < 2 or any(x <= 0.0 for x in parts):
      raise ValueError(
          f'subdivisions[{name!r}] must have >= 2 positive lengths.'
      )
    int_parts = [int(round(x)) for x in parts]
    if any(
        not math.isclose(x, ix, abs_tol=1e-4) for x, ix in zip(parts, int_parts)
    ):
      raise ValueError(
          f'subdivisions[{name!r}] must use integer grid lengths; got {parts}.'
      )
    if sum(int_parts) != stick.length:
      raise ValueError(
          f'subdivisions[{name!r}] sums to {sum(int_parts)}, expected'
          f' {stick.length}.'
      )
    int_sub_map[name] = int_parts

  subdivided, split_names = base_tree.subdivide_sticks(int_sub_map)
  known_edited = {s.name for s in subdivided.children}
  for label, mapping in (
      ('azimuths', az_map),
      ('elevations', el_map),
      ('widths', w_map),
  ):
    extra = sorted(set(mapping) - known_edited)
    if extra:
      raise ValueError(f'tree_edit: unknown sticks in {label}: {extra}.')

  edited_tree = subdivided.update_tree(
      azimuth_updates=az_map,
      elevation_updates=el_map,
      width_updates=w_map,
  )
  # Refresh symmetry pairings when paired sticks subdivide identically.
  edited_tree = stick_figure_lib.StickFigure.from_dict(edited_tree.to_dict())

  sub_to_merged: dict[str, str] = {}
  edited_names: list[str] = []
  touched_updates = set(az_map) | set(el_map) | set(w_map)
  for stick in base_tree.children:
    origin = unmerged_to_merged.get(stick.name, stick.name)
    if stick.name in split_names:
      for sub_name in split_names[stick.name]:
        sub_to_merged[sub_name] = origin
        edited_names.append(sub_name)
    else:
      sub_to_merged[stick.name] = origin
      if stick.name in touched_updates:
        edited_names.append(stick.name)
  return edited_tree, sub_to_merged, edited_names


def _tree_edit_outcome(
    state: ShapingState,
    *,
    subdivisions: Mapping[str, Sequence[float]] | None = None,
    azimuths: Mapping[str, float] | None = None,
    elevations: Mapping[str, float] | None = None,
    widths: Mapping[str, float] | None = None,
    tree: Mapping[str, Any] | None = None,
    stick_name_mapping: Mapping[str, str] | None = None,
    narrow: bool = False,
    only_edited: bool = False,
    verify_fold: bool = False,
) -> state_lib.ToolOutcome:
  """Edits stick subdivisions/angles/widths while keeping the base packing."""

  def _impl(st: ShapingState) -> state_lib.ToolOutcome:
    if tree is not None:
      edited = stick_figure_lib.StickFigure.from_dict(dict(tree))
      if stick_name_mapping is None:
        raise ValueError(
            'stick_name_mapping is required when passing a raw tree dict.'
        )
      mapping = {str(k): str(v) for k, v in stick_name_mapping.items()}
      edited_names = [s.name for s in edited.children]
    else:
      if not any((subdivisions, azimuths, elevations, widths)):
        raise ValueError(
            'tree_edit requires at least one of subdivisions, azimuths,'
            ' elevations, widths, or tree.'
        )
      edited, mapping, edited_names = _build_edited_tree(
          st.tree,
          subdivisions=subdivisions,
          azimuths=azimuths,
          elevations=elevations,
          widths=widths,
      )

    base_merged, _ = st.tree.merge_linear_chains()
    if not utils_tree.check_shaped_tree_is_packing_equivalent(
        base_merged,
        edited,
        mapping,
        {},
        st.packing_config,
        check_traversal_order=False,
    ):
      raise ValueError(
          'Edited stick figure is not packing-equivalent to the base tree.'
      )
    step_conv = utils_tree.get_merge_converter_from_substick_map(
        edited, mapping
    )
    base_conv = st.merge_converter
    converter = (
        {k: base_conv.get(v, v) for k, v in step_conv.items()}
        if base_conv
        else step_conv
    )
    flaps, rivers = st.merged_shaping_elements
    subset = edited_names if only_edited and edited_names else None
    if narrow or widths:
      shaped = combined_heuristics.shape_cp_with_combined_heuristics(
          base_cp=st.pattern,
          stick_figure=edited,
          packing_output=st.packing,
          solving_output=st.solving,
          solving_grid_size=st.solving_grid_size,
          central_crease_removed=st.central_crease_removed,
          merge_converter=converter,
          hparams=combined_heuristics.CombinedHeuristicsHParams(
              geometric_fold_hparams=geometric_fold_3d_heuristics.GeometricFolding3DHParams(
                  default_inscribe_hparams=geometric_fold_3d_heuristics.InscribeHParams(
                      avoid_hinges=True
                  ),
                  check_ff_violations=True,
                  strategy=heuristics_3d.FoldStrategy3D.SHALLOW_FOLDS,
                  shape_subset=subset,
                  use_tree_root=st.config.use_tree_root,
              )
          ),
          shaping_flaps=flaps,
          shaping_rivers=rivers,
          graph_data=st.graph_data,
      )
    else:
      res = geometric_fold_3d_heuristics.get_shaping_creases_with_3d_geometric_folding_heuristics(
          base_cp=st.pattern,
          stick_figure=edited,
          shaping_flaps=flaps,
          shaping_rivers=rivers,
          hinge_info_list=st.updated_hinge_info(),
          central_crease_removed=st.central_crease_removed,
          shape_with_unmerged_tree=True,
          merge_converter=converter,
          hparams=geometric_fold_3d_heuristics.GeometricFolding3DHParams(
              shape_subset=subset,
              check_ff_violations=True,
              strategy=st.config.fold_strategy_3d,
              default_inscribe_hparams=geometric_fold_3d_heuristics.InscribeHParams(
                  avoid_hinges=True,
                  clip_slope_thresh=st.config.clip_slope_thresh,
                  max_slope_thresh=st.config.max_slope_thresh,
              ),
              direction_graph_data=st.base_graph_data,
              use_tree_root=st.config.use_tree_root,
          ),
          graph_data=st.graph_data,
          sacrificial_flaps=st.packing.sacrificial_flaps or [],
      )
      if not res['shaping_creases']:
        raise ValueError('tree_edit produced no shaping creases.')
      shaped = st.pattern.update_with_shaping_creases(res['shaping_creases'])

    accepted, reason, _ = state_lib.accept_pattern(
        st, shaped, allow_3d_vertices=True, verify_fold=verify_fold
    )
    if accepted is None:
      raise ValueError(reason)
    narrowed = tuple(
        s.name
        for s in edited.children
        if s.is_flap
        and s.width < 1.0
        and (narrow or (widths and s.name in widths))
    )
    out = accepted.replace(tree=edited)
    if narrowed:
      out = out.with_narrowed(*narrowed)
    delta = len(out.pattern.creases) - len(st.pattern.creases)
    return state_lib.ToolOutcome(
        state=out,
        note=(
            f'tree_edit: shaped {len(edited_names)} stick(s), +{delta} creases.'
        ),
    )

  return state_lib.run_with_3d_overlay(state, _impl, verify_fold=verify_fold)


def tree_edit(
    state: ShapingState,
    *,
    subdivisions: Mapping[str, Sequence[float]] | None = None,
    azimuths: Mapping[str, float] | None = None,
    elevations: Mapping[str, float] | None = None,
    widths: Mapping[str, float] | None = None,
    tree: Mapping[str, Any] | None = None,
    stick_name_mapping: Mapping[str, str] | None = None,
    narrow: bool = False,
    only_edited: bool = False,
    verify_fold: bool = False,
) -> ShapingState:
  """Subdivides or re-angles sticks in a packing-equivalent tree and shapes."""
  return _tree_edit_outcome(
      state,
      subdivisions=subdivisions,
      azimuths=azimuths,
      elevations=elevations,
      widths=widths,
      tree=tree,
      stick_name_mapping=stick_name_mapping,
      narrow=narrow,
      only_edited=only_edited,
      verify_fold=verify_fold,
  ).state


# ------------------------------------------------------------------------------
# Declarative registry, safe JSON dispatch, and non-mutating element probing.
# ------------------------------------------------------------------------------

_TOOL_OUTCOMES: dict[str, Callable[..., state_lib.ToolOutcome]] = {
    'simple_fold': _simple_fold_outcome,
    'crimp_fold': _crimp_fold_outcome,
    'reverse_fold': _reverse_fold_outcome,
    'geometric_simple_fold': _geometric_simple_fold_outcome,
    'clip_pattern_fold': relief.clip_pattern_fold,
    'narrow_central_flap': relief.narrow_central_flap,
    'fan_out_fold': fan_squash.fan_out_fold,
    'squash_fold': fan_squash.squash_fold,
    'inscribed_flap': relief.inscribed_flap,
    'open_flap': relief.open_flap,
    'fold_spine': relief.fold_spine,
    'rotation_3d': _rotation_3d_outcome,
    'bend_to_direction': _bend_to_direction_outcome,
    'thin_limbs_3d': _thin_limbs_3d_outcome,
    'curve_flap': _curve_flap_outcome,
    'spread_fork': _spread_fork_outcome,
    'radial_fan': _radial_fan_outcome,
    'apply_symmetric': _apply_symmetric_outcome,
    'tree_edit': _tree_edit_outcome,
}


def _format_error(error: BaseException) -> str:
  msg = str(error).strip()
  if msg and not isinstance(error, (AssertionError, KeyError, IndexError)):
    return msg
  for frame in reversed(traceback.extract_tb(error.__traceback__)):
    if 'bp_pipeline' in frame.filename:
      fname = os.path.basename(frame.filename)
      detail = f': {msg}' if msg else ''
      return f'{type(error).__name__}{detail} ({fname}:{frame.lineno})'
  return f'{type(error).__name__}: {msg}' if msg else type(error).__name__


def _append_history(state: ShapingState, result: ToolResult) -> ShapingState:
  return state.replace(history=state.history + (result.asdict(),))


def apply_tool(
    state: ShapingState,
    name: str,
    arguments: Mapping[str, Any] | None = None,
) -> tuple[ShapingState, ToolResult]:
  """Invokes a shaping tool by name without raising, appending to `history`."""
  raw_args = dict(arguments or {})
  canonical = normalize_tool_name(name)
  runner = _TOOL_OUTCOMES.get(canonical)
  if runner is None:
    result = ToolResult(
        tool=canonical,
        arguments=raw_args,
        ok=False,
        message=f'Unknown tool {name!r}; available: {", ".join(TOOL_NAMES)}.',
        num_creases=len(state.pattern.creases),
    )
    return _append_history(state, result), result

  try:
    outcome = runner(state, **raw_args)
  except (
      ValueError,
      AssertionError,
      KeyError,
      IndexError,
      TypeError,
      ZeroDivisionError,
  ) as error:
    result = ToolResult(
        tool=canonical,
        arguments=raw_args,
        ok=False,
        message=f'{canonical} failed: {_format_error(error)}',
        num_creases=len(state.pattern.creases),
    )
    return _append_history(state, result), result

  recorded = (
      dict(outcome.resolved_arguments)
      if outcome.resolved_arguments is not None
      else raw_args
  )
  delta = len(outcome.state.pattern.creases) - len(state.pattern.creases)
  base_msg = f'{canonical} ok ({delta:+d} creases).'
  message = f'{base_msg} {outcome.note}' if outcome.note else base_msg
  result = ToolResult(
      tool=canonical,
      arguments=recorded,
      ok=True,
      message=message,
      num_creases=len(outcome.state.pattern.creases),
  )
  return _append_history(outcome.state, result), result


def apply_tools(
    state: ShapingState,
    calls: Sequence[Mapping[str, Any]],
    *,
    stop_on_error: bool = False,
) -> tuple[ShapingState, list[ToolResult]]:
  """Applies a sequence of `{'tool': name, 'arguments': {...}}` calls."""
  current = state
  results: list[ToolResult] = []
  for call in calls:
    tool_name = str(call.get('tool') or call.get('name') or '')
    args = call.get('arguments')
    if args is None:
      args = {k: v for k, v in call.items() if k not in ('tool', 'name')}
    current, res = apply_tool(current, tool_name, args)
    results.append(res)
    if stop_on_error and not res.ok:
      break
  return current, results


def probe(
    state: ShapingState,
    element_names: Sequence[str] | None = None,
) -> dict[str, dict[str, Any]]:
  """Inspects per-element geometry and fast dry-run actuator applicability."""
  rows = {row['name']: row for row in element_table(state)}
  targets = (
      list(element_names) if element_names is not None else list(rows.keys())
  )
  root_name = heuristics.get_first_stick_for_shaping(
      state.tree, use_tree_root=state.config.use_tree_root
  ).name
  out: dict[str, dict[str, Any]] = {}
  for name in targets:
    elem = state.element(name)
    info = dict(rows.get(name, {'name': name}))
    tools_ok: list[str] = []
    if name != root_name:
      try:
        _rotation_3d_outcome(
            state,
            rotations=[{
                'element_name': name,
                'az': info.get('az', 0.0) or 30.0,
                'el': info.get('el', 0.0) or 30.0,
            }],
        )
        tools_ok.append('rotation_3d')
      except ValueError:
        pass
    try:
      _simple_fold_outcome(
          state, element_name=name, angle=45.0, verify_fold=False
      )
      tools_ok.append('simple_fold')
    except ValueError:
      pass
    if isinstance(elem, shaping_elements.ShapingFlap):
      try:
        _simple_fold_outcome(
            state, element_name=name, angle=45.0, narrow=True, verify_fold=False
        )
        tools_ok.append('simple_fold(narrow)')
      except ValueError:
        pass
      if name != root_name:
        try:
          _crimp_fold_outcome(
              state, element_name=name, angle=45.0, verify_fold=False
          )
          tools_ok.append('crimp_fold')
        except ValueError:
          pass
        try:
          _reverse_fold_outcome(
              state, element_name=name, angle=45.0, verify_fold=False
          )
          tools_ok.append('reverse_fold')
        except ValueError:
          pass
      counts = fan_squash.ridge_segment_counts(state.pattern, elem)
      info['ridge_segments'] = counts
      if any(c >= 2 for c in counts):
        tools_ok.append('squash_fold')
      if elem.grid_unit_radius >= 2:
        tools_ok.append('inscribed_flap')
      if elem.contains_central_crease and elem.grid_unit_radius >= 2:
        tools_ok.append('narrow_central_flap')
      tools_ok.append('open_flap')
    else:
      if elem.width >= 2:
        tools_ok.append('inscribed_flap')
      tools_ok.append('open_flap')
    info['available_tools'] = tools_ok
    out[name] = info
  return out
