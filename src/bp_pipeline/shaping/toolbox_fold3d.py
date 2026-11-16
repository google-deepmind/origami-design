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

"""Local 3-D reverse and crimp folds for the shaping toolbox.

`reverse_fold_3d` and `crimp_fold_3d` cut the target element's layers in the
folded state (the `rotation_3d` machinery: a cut line intersected with the
element's faces only) and invert the enclosed region's mountain/valley
assignment, so the creases stay inside the element, work on narrowed limbs,
and compose with `rotation_3d` before or after. See
`geometric_simple_fold.get_inverted_region_variants`.

The functions take a `shaping.ShapingState` and return a new one; they are
registered in `shaping._TOOLS` (this module does not import `shaping`).
"""

from collections.abc import Sequence
import dataclasses
import itertools
import math
import time
from typing import Any

from absl import logging
import numpy as np

from . import geometric_fold_3d_heuristics
from . import geometric_simple_fold
from . import shaping_elements
from ..crease_pattern.parser import svg as svg_parser
from . import folder_3d as folder
from . import fidelity

# Depth offsets tried after the requested depth (fractions of the element).
_DEPTH_FALLBACKS = (0.0, 0.15, -0.15, 0.3)


@dataclasses.dataclass(frozen=True)
class FoldAttempt:
  """One evaluated candidate (for messages and benchmarks)."""

  depth: str
  variant: str
  outcome: str


@dataclasses.dataclass
class FoldReport:
  """What the last call did (read by benchmarks; set on success/failure)."""

  element_name: str = ''
  attempts: list[FoldAttempt] = dataclasses.field(default_factory=list)
  chosen: FoldAttempt | None = None
  num_new_creases: int = 0
  num_flipped_creases: int = 0
  cut_lines: list[Any] = dataclasses.field(default_factory=list)


LAST_REPORT = FoldReport()


def element_polygon(state: Any, element_name: str) -> Any:
  """The limiting polygon (unfolded rectangles) of an element."""
  element = state.element(element_name)
  if isinstance(element, shaping_elements.ShapingFlap):
    return [element.to_legacy_rectangle()]
  return list(element.covering_legacy_rectangles)


def chain_polygon(state: Any, element_name: str) -> Any | None:
  """Rectangles of the merged chain flap holding `element_name`, if any."""
  _, merged_names = state.tree.merge_linear_chains()
  flaps, _ = state.merged_shaping_elements
  for merged_name, sub_names in merged_names.items():
    if element_name in sub_names or element_name == merged_name:
      for flap in flaps:
        if flap.name == merged_name:
          return [flap.to_legacy_rectangle()]
  return None


def _parent(state: Any, element_name: str) -> str:
  parent = fidelity.topology(state.tree).parent.get(element_name)
  if parent is None:
    raise ValueError(
        f'{element_name} is the root stick; reverse/crimp folds need a flap.'
    )
  return parent


def reference_point(
    state: Any, element_name: str, graph_data: Any
) -> np.ndarray:
  """Folded center of the parent element's footprint."""
  topo = fidelity.topology(state.tree)
  curr = element_name
  while True:
    parent = topo.parent.get(curr)
    if parent is None:
      break
    points = geometric_fold_3d_heuristics.element_folded_points(
        graph_data, element_polygon(state, parent)
    )
    if len(points):  # pylint: disable=g-explicit-length-test
      return points.mean(axis=0)
    curr = parent
  folded_v = graph_data.get('folded_vertices', [])
  if len(folded_v):  # pylint: disable=g-explicit-length-test
    return np.array(folded_v)[:, :2].mean(axis=0)
  return np.array([0.5, 0.5])


def folds(pattern: Any) -> tuple[bool, str]:
  """Whether the deterministic folder folds `pattern` (as `lib` renders)."""
  for c in pattern.creases:
    if math.hypot(c.p1.x - c.p2.x, c.p1.y - c.p2.y) < 5e-4:
      return False, 'degenerate segment (< 5e-4)'
  start = time.perf_counter()
  try:
    folder.get_fold(
        svg_parser.to_svg(pattern, ndigits=8),
        vert_tol=1e-3,
        simulation_tolerance=1.0,
    )
  except Exception as error:  # pylint: disable=broad-exception-caught
    return False, f'folder: {str(error)[:120]}'
  return True, f'folds ({time.perf_counter() - start:.1f}s)'


def _crease_key(crease: Any) -> tuple[float, float, float, float]:
  """Canonical rounded endpoints `(x1, y1, x2, y2)` of a crease segment."""
  p1 = (round(crease.p1.x, 6), round(crease.p1.y, 6))
  p2 = (round(crease.p2.x, 6), round(crease.p2.y, 6))
  return (*p1, *p2) if p1 <= p2 else (*p2, *p1)


def _apply(
    state: Any,
    *,
    element_name: str,
    depth: float,
    make_lines: Any,
    fold_percent: float,
    verify_fold: bool,
    kind: str,
) -> Any:
  """Tries depths x M/V variants; returns the first state that folds."""
  flat_backbone = getattr(state, 'flat_backbone', None)
  if flat_backbone is not None:
    flat_in = state.replace(pattern=flat_backbone, flat_backbone=None)
    flat_out = _apply(
        flat_in,
        element_name=element_name,
        depth=depth,
        make_lines=make_lines,
        fold_percent=fold_percent,
        verify_fold=verify_fold,
        kind=kind,
    )
    combined = fidelity.replay_3d_overlay(
        flat_backbone, state.pattern, flat_out.pattern
    )
    ok_ff, err_ff = fidelity.preserves_foldability(state.pattern, combined)
    if not ok_ff:
      raise ValueError(f'{kind} of {element_name} broke 3-D overlay: {err_ff}')
    if verify_fold:
      ok_fold, err_fold = folds(combined)
      if not ok_fold:
        raise ValueError(
            f'{kind} of {element_name} failed folder with 3-D overlay:'
            f' {err_fold}'
        )
    next_backbone = (
        fidelity.as_flat_backbone(flat_out.pattern)
        if fidelity.has_3d_overlay(combined)
        else None
    )
    return flat_out.replace(pattern=combined, flat_backbone=next_backbone)

  global LAST_REPORT
  report = FoldReport(element_name=element_name)
  LAST_REPORT = report
  if element_name not in state.tree:
    raise ValueError(
        f'{kind}: unknown element {element_name}; known:'
        f' {state.element_names()}'
    )
  if not 0 <= depth < 1:
    raise ValueError(f'depth must be in [0, 1); got {depth}.')
  if not 0 < fold_percent <= 1:
    raise ValueError(f'fold_percent must be in (0, 1]; got {fold_percent}.')
  if getattr(state, 'pattern', None) is getattr(state, 'base_pattern', None):
    graph_data = getattr(state, 'base_graph_data', None)
  else:
    graph_data = getattr(state, 'graph_data', None)
  if graph_data is None:
    graph_data = getattr(state, 'graph_data', None)
  if graph_data is None:
    raise ValueError(f'State has no graph_data; cannot fold {element_name}.')
  polygon = element_polygon(state, element_name)
  axis = geometric_fold_3d_heuristics.get_element_folded_axis(
      graph_data, polygon, reference_point(state, element_name, graph_data)
  )
  points = geometric_fold_3d_heuristics.element_folded_points(
      graph_data, polygon
  )
  # (name, limiting polygon, cut only near the element's footprint)
  parent_polygon = element_polygon(state, _parent(state, element_name))
  polygons = [('element', polygon, True)]
  chain = chain_polygon(state, element_name)
  if chain is not None:
    polygons.append(('chain', chain, True))
    # Chains packed as full-width strips (strip fallback): every layer of the
    # strip at the cut belongs to the flap, so cut the whole strip.
    polygons.append(('strip', chain, False))
  last_error = 'no candidate'
  candidates = []
  depth_offsets = (0.0,) if not verify_fold else _DEPTH_FALLBACKS
  for offset in depth_offsets:
    current = round(min(0.95, max(0.05, depth + offset)), 3)
    if current not in candidates:
      candidates.append(current)
  seen_cuts: set[Any] = set()
  for (polygon_name, limiting, near), current in itertools.product(
      polygons, candidates
  ):
    label = f'{current}/{polygon_name}'
    try:
      lines = make_lines(axis, current, points)
      spans = [
          geometric_fold_3d_heuristics.line_span(
              line, points, 0.25 * axis.width
          )
          if near
          else (-math.inf, math.inf)
          for line in lines
      ]
      cut_sets = []
      for index, (line, span) in enumerate(zip(lines, spans, strict=True)):
        cut_creases = geometric_simple_fold.get_segment_cut_creases(
            graph_data=graph_data,
            cut_line_start=line[0],
            cut_line_end=line[1],
            span=span,
            limiting_polygon=limiting,
        )
        if not cut_creases:
          raise ValueError(f'Cut line {index} does not cut the target flap.')
        if any(
            math.hypot(c.p1.x - c.p2.x, c.p1.y - c.p2.y) < 5e-4
            for c in cut_creases
        ):
          raise ValueError(
              'Cut line produces degenerate crease segments (< 5e-4).'
          )
        cut_sets.append(frozenset(_crease_key(c) for c in cut_creases))
      line_keys = tuple(
          (round(s[0], 6), round(s[1], 6), round(e[0], 6), round(e[1], 6))
          for s, e in lines
      )
      signature = (line_keys, tuple(cut_sets))
      if signature in seen_cuts:
        continue
      seen_cuts.add(signature)
      variants = geometric_simple_fold.get_inverted_region_variants(
          state.pattern,
          cut_lines=lines,
          limiting_polygon=limiting,
          graph_data=graph_data,
          fold_percent=fold_percent,
          proximal_polygon=parent_polygon,
          target_polygon=polygon,
          spans=spans,
      )
    except (ValueError, AssertionError) as error:
      last_error = f'depth {label}: {error}'
      report.attempts.append(FoldAttempt(label, '-', str(error)[:100]))
      continue
    for variant in variants:
      pattern = variant['crease_pattern']
      percentage, _, _ = pattern.vertex_flat_foldability_percentage()
      if not math.isclose(percentage, 1.0):
        report.attempts.append(
            FoldAttempt(label, variant['name'], f'not FF ({percentage:.1%})')
        )
        last_error = f'depth {label} {variant["name"]}: not flat-foldable'
        continue
      if verify_fold:
        ok, outcome = folds(pattern)
        report.attempts.append(FoldAttempt(label, variant['name'], outcome))
        if not ok:
          last_error = f'depth {label} {variant["name"]}: {outcome}'
          continue
      else:
        report.attempts.append(FoldAttempt(label, variant['name'], 'FF'))
      report.chosen = report.attempts[-1]
      report.num_new_creases = variant['num_new_creases']
      report.num_flipped_creases = variant['num_flipped_creases']
      report.cut_lines = lines
      logging.info(
          '%s %s: depth %s, %s, +%d creases, %d flipped',
          kind,
          element_name,
          label,
          variant['name'],
          variant['num_new_creases'],
          variant['num_flipped_creases'],
      )
      return state.with_pattern(pattern)
  raise ValueError(
      f'{kind} of {element_name} failed for every depth/assignment; last:'
      f" {last_error}. Try another angle sign, depth, or the element's"
      ' parent.'
  )


def reverse_fold_3d(
    state: Any,
    *,
    element_name: str,
    angle: float,
    depth: float = 0.5,
    fold_percent: float = 1.0,
    verify_fold: bool = True,
) -> Any:
  """Local reverse fold on one flap, cut in the folded state.

  The distal part of `element_name` (and everything beyond it) turns by
  `angle` degrees in the plane of the flap as it lies in the current folded
  form; only the flap's own layers get new creases. Works on narrowed flaps
  and after/before `rotation_3d`.

  Args:
    state: Current state.
    element_name: Flap (or sub-flap) to fold.
    angle: Turn in degrees, 0 < |angle| <= 180; the sign picks the side.
    depth: Where the cut crosses the flap's center line (0 = base, 1 = tip).
    fold_percent: Fold percent of the new creases (1 = flat).
    verify_fold: Keep only candidates the deterministic folder folds.

  Returns:
    The new state.
  """
  return _apply(
      state,
      element_name=element_name,
      depth=depth,
      make_lines=lambda axis, d, points: [
          geometric_fold_3d_heuristics.reverse_fold_cut_line(
              axis, angle=angle, depth=d, snap_points=points
          )
      ],
      fold_percent=fold_percent,
      verify_fold=verify_fold,
      kind='reverse_fold_3d',
  )


def crimp_fold_3d(
    state: Any,
    *,
    element_name: str,
    angle: float,
    depth: float = 0.4,
    gap: float | None = None,
    fold_percent: float = 1.0,
    verify_fold: bool = True,
) -> Any:
  """Local crimp (two opposite reverse folds) on one flap.

  Turns the part beyond the crimp by `angle` degrees in the flap's folded
  plane and shortens the flap by about the gap; creases stay on the flap.

  Args:
    state: Current state.
    element_name: Flap (or sub-flap) to crimp.
    angle: Turn in degrees, 0 < |angle| < 180; the sign picks the side.
    depth: Where the first cut crosses the center line (0 = base, 1 = tip).
    gap: Distance between the two cuts along the axis (folded units);
      default: they meet just outside the flap edge.
    fold_percent: Fold percent of the new creases (1 = flat).
    verify_fold: Keep only candidates the deterministic folder folds.

  Returns:
    The new state.
  """
  return _apply(
      state,
      element_name=element_name,
      depth=depth,
      make_lines=lambda axis, d, points: (
          geometric_fold_3d_heuristics.crimp_fold_cut_lines(
              axis, angle=angle, depth=d, gap=gap, snap_points=points
          )
      ),
      fold_percent=fold_percent,
      verify_fold=verify_fold,
      kind='crimp_fold_3d',
  )


def last_report_message() -> str:
  """One line describing the last call (for `ToolResult.message` notes)."""
  report = LAST_REPORT
  if report.chosen is None:
    return f'{report.element_name}: no candidate folded.'
  return (
      f'{report.element_name}: cut at depth/region {report.chosen.depth},'
      f' assignment {report.chosen.variant}; +{report.num_new_creases} creases'
      f' on the flap, {report.num_flipped_creases} flipped;'
      f' {len(report.attempts)} candidates tried.'
  )


TOOLS: dict[str, Any] = {
    'reverse_fold_3d': reverse_fold_3d,
    'crimp_fold_3d': crimp_fold_3d,
}


def tool_names() -> Sequence[str]:
  return tuple(TOOLS)
