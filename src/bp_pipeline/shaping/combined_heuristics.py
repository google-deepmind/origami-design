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
"""Combined 3D geometric-folding and half-grid flap-narrowing heuristics."""

import collections
from collections.abc import Mapping, Sequence
import dataclasses
import itertools
import math

from absl import logging

from .. import timeout as timeout_lib
from ..crease_pattern import crease as crease_utils
from ..crease_pattern import crease_pattern as cp_utils
from ..packing import packing_output as packing_output_utils
from ..packing import types as packing_types
from ..solving import hinges as hinge_utils
from ..stick_figures import stick_figure as sf_utils
from . import fold_shaping
from . import geometric_fold_3d_heuristics
from . import geometric_fold_3d_heuristics_utils
from . import geometric_simple_fold
from . import heuristics
from . import heuristics_3d
from . import heuristics_utils
from . import narrow_flap_shaping
from . import shaping_elements
from . import shaping_utils


@dataclasses.dataclass(frozen=True)
class CombinedHeuristicsHParams:
  """Configuration for combined 3D geometric folding and flap narrowing."""

  geometric_fold_hparams: (
      geometric_fold_3d_heuristics.GeometricFolding3DHParams
  ) = dataclasses.field(
      default_factory=geometric_fold_3d_heuristics.GeometricFolding3DHParams
  )
  use_narrowing: bool = True


def _group_by_narrowed_ancestor(
    results: Sequence[geometric_fold_3d_heuristics.ElementShapingResult],
    flap_info: Mapping[str, heuristics.Node],
    narrowed_roots: set[str],
) -> dict[str, list[geometric_fold_3d_heuristics.ElementShapingResult]]:
  """Groups element shaping results under their enclosing narrowed flap."""
  grouped: dict[
      str, list[geometric_fold_3d_heuristics.ElementShapingResult]
  ] = collections.defaultdict(list)
  for res in results:
    cur: str | None = res['shaping_element'].name
    while cur is not None and cur not in narrowed_roots:
      timeout_lib.maybe_timeout_cooperatively()
      node = flap_info.get(cur)
      cur = node.parent if node is not None else None
    if cur is None:
      raise ValueError(
          f'Element {res["shaping_element"].name!r} has no ancestor in'
          f' {sorted(narrowed_roots)}.'
      )
    grouped[cur].append(res)
  return grouped


def _infer_double_grid_creases(
    shaping_results: Sequence[
        geometric_fold_3d_heuristics.ElementShapingResult
    ],
    central_crease_removed: bool,
) -> list[crease_utils.Crease]:
  """Re-emits post-narrowing simple folds on a doubled (2x grid_size) grid."""
  creases: list[crease_utils.Crease] = []
  for res in shaping_results:
    element = res['shaping_element']
    if not isinstance(element, shaping_elements.ShapingFlap):
      continue
    target_creases = res['shaping_creases']
    if not target_creases:
      continue
    fold_info = res['simple_fold_info']
    angle_rad = abs(math.radians(fold_info.rotation_angle))
    zz_req = fold_shaping.infer_fold_order_phase_from_creases(
        target_creases=target_creases,
        element=element,
        target_flap_angle=angle_rad,
        inscribe_depth=0.0,
        fold_percent=target_creases[0].fold_percent,
        central_crease_removed=central_crease_removed,
    )
    if zz_req is None:
      logging.warning(
          'Could not infer zigzag params for narrowed flap %s; skipping.',
          element.name,
      )
      continue
    try:
      creases.extend(
          fold_shaping.get_simple_fold_shaping_creases(
              grid_size=2 * element.grid_size,
              flaps=(element,),
              rivers=(),
              element_shaping_params=(zz_req,),
              central_crease_removed=central_crease_removed,
          )
      )
    except ValueError as err:
      logging.warning(
          'Failed to emit narrowed zigzag creases for %s: %s',
          element.name,
          err,
      )
  return creases


def compute_narrowed_flaps(
    *,
    stick_figure: sf_utils.StickFigure,
    shaping_data: heuristics_utils.ShapingData,
    use_narrowing: bool,
) -> tuple[set[str], set[str], dict[str, heuristics_3d.FoldStrategy3D]]:
  """Identifies `(narrowed_parents, narrowed_children, per_flap_strategies)`."""
  if not use_narrowing:
    return set(), set(), {}
  flap_names = {flap.name for flap in shaping_data['shaping_flaps']}
  parents: set[str] = set()
  children: set[str] = set()
  strategies: dict[str, heuristics_3d.FoldStrategy3D] = {}

  for stick in stick_figure.children:
    if stick.width < 1.0 and stick.name in flap_names:
      parents.add(stick.name)
      descendants = heuristics.get_descendants_names(
          shaping_data['flap_info'], stick.name
      )
      children.update(descendants)
      for name in (stick.name, *descendants):
        strategies[name] = heuristics_3d.FoldStrategy3D.ROTATE_BEND
  return parents - children, children, strategies


def _partition_for_narrowing(
    stick_figure: sf_utils.StickFigure,
    narrowed_parents: set[str],
    narrowed_children: set[str],
    detailed_info: Mapping[
        str, Sequence[geometric_fold_3d_heuristics.ElementShapingResult]
    ],
) -> tuple[
    list[geometric_fold_3d_heuristics.ElementShapingResult],
    list[shaping_utils.Rectangle],
    list[geometric_fold_3d_heuristics.ElementShapingResult],
]:
  """Splits 3D fold results into pre-narrow, target rects, and post-narrow."""
  pre_narrow: list[geometric_fold_3d_heuristics.ElementShapingResult] = []
  post_narrow: list[geometric_fold_3d_heuristics.ElementShapingResult] = []
  narrow_rects: list[shaping_utils.Rectangle] = []

  for stick in stick_figure.children:
    if (
        stick.name not in narrowed_parents
        and stick.name not in narrowed_children
    ):
      pre_narrow.extend(detailed_info.get(stick.name, ()))

  for parent_name in narrowed_parents:
    entries = detailed_info.get(parent_name)
    if not entries:
      continue
    initial = entries[0]
    rot = initial['simple_fold_info'].rotation_angle
    if not math.isclose(abs(rot), 180.0, abs_tol=1e-3):
      raise ValueError(
          f'Narrowed flap {parent_name!r} requires initial 180-degree fold,'
          f' got {rot}.'
      )
    element = initial['shaping_element']
    if not isinstance(element, shaping_elements.ShapingFlap):
      raise ValueError(
          f'Narrowing target {parent_name!r} must be a ShapingFlap.'
      )
    pre_narrow.append(initial)
    narrow_rects.append(element.to_legacy_rectangle())
    post_narrow.extend(entries[1:])

  for child_name in narrowed_children:
    post_narrow.extend(detailed_info.get(child_name, ()))

  return pre_narrow, narrow_rects, post_narrow


def shape_cp_with_combined_heuristics(
    *,
    base_cp: cp_utils.CreasePattern,
    stick_figure: sf_utils.StickFigure,
    packing_output: packing_output_utils.PackingOutput | None = None,
    solving_output: hinge_utils.SolverResult | None = None,
    hinge_info_list: Sequence[packing_types.HingeInfo] | None = None,
    solving_grid_size: int,
    central_crease_removed: bool,
    shape_with_unmerged_tree: bool = True,
    merge_converter: Mapping[str, str] | None = None,
    hparams: CombinedHeuristicsHParams = CombinedHeuristicsHParams(),
    shaping_flaps: Sequence[shaping_elements.ShapingFlap] | None = None,
    shaping_rivers: Sequence[shaping_elements.ShapingRiver] | None = None,
    graph_data: geometric_simple_fold.GraphData | None = None,
    shaping_data: heuristics_utils.ShapingData | None = None,
) -> cp_utils.CreasePattern:
  """Shapes `base_cp` with 3D geometric folds plus narrowing for thin sticks."""
  if shaping_flaps is None or shaping_rivers is None:
    if packing_output is None or packing_output.packing_cp is None:
      raise ValueError(
          'packing_output with packing_cp is required when shaping_flaps or'
          ' shaping_rivers is None.'
      )
    shaping_flaps, shaping_rivers = (
        shaping_elements.convert_packing_elements_to_shaping_elements(
            packing_rectangles=packing_output.flap_rectangles,
            packing_rivers=packing_output.packing_rivers,
            packing_grid_size=packing_output.packing_cp.grid_size,
            solving_grid_size=solving_grid_size,
            base_cp=base_cp,
        )
    )
  if hinge_info_list is None:
    if packing_output is None or solving_output is None:
      raise ValueError(
          'packing_output and solving_output are required when hinge_info_list'
          ' is None.'
      )
    hinge_info_list = heuristics.update_hinge_info_from_solver_result(
        packing_output.hinges, solving_output
    )

  if shaping_data is None:
    shaping_data = heuristics_utils.get_heuristics_shaping_data(
        stick_figure=stick_figure,
        shaping_flaps=list(shaping_flaps),
        shaping_rivers=list(shaping_rivers),
        hinge_info_list=list(hinge_info_list),
        central_crease_removed=central_crease_removed,
        shape_with_unmerged_tree=shape_with_unmerged_tree,
        merge_converter=merge_converter,
        use_tree_root=hparams.geometric_fold_hparams.use_tree_root,
    )

  narrowed_parents, narrowed_children, flap_strategies = compute_narrowed_flaps(
      stick_figure=stick_figure,
      shaping_data=shaping_data,
      use_narrowing=hparams.use_narrowing,
  )
  per_flap_inscribe = {
      name: (
          1.0
          if strategy == heuristics_3d.FoldStrategy3D.ROTATE_BEND
          else geometric_fold_3d_heuristics_utils.INSCRIBE_FRACTION
      )
      for name, strategy in flap_strategies.items()
  }
  geo_hparams = dataclasses.replace(
      hparams.geometric_fold_hparams,
      strategy=heuristics_3d.FoldStrategy3D.SHALLOW_FOLDS,
      per_flap_strategy=flap_strategies,
      per_flap_hinge_inscribe_depth=per_flap_inscribe,
  )

  geo_result = geometric_fold_3d_heuristics.get_shaping_creases_with_3d_geometric_folding_heuristics(
      base_cp=base_cp,
      stick_figure=stick_figure,
      shaping_flaps=shaping_flaps,
      shaping_rivers=shaping_rivers,
      hinge_info_list=hinge_info_list,
      central_crease_removed=central_crease_removed,
      shape_with_unmerged_tree=shape_with_unmerged_tree,
      merge_converter=dict(merge_converter) if merge_converter else None,
      hparams=geo_hparams,
      graph_data=graph_data,
      shaping_data=shaping_data,
  )

  pre_narrow, narrow_rects, post_narrow = _partition_for_narrowing(
      stick_figure,
      narrowed_parents,
      narrowed_children,
      geo_result['detailed_shaping_info'],
  )
  pre_creases = list(
      itertools.chain.from_iterable(
          info['shaping_creases'] for info in pre_narrow
      )
  )
  cp = base_cp.update_with_shaping_creases(pre_creases)

  target_names = {r.name for r in narrow_rects}
  flap_info = shaping_data['flap_info']
  seed_by_flap = _group_by_narrowed_ancestor(
      [
          info
          for info in pre_narrow
          if info['shaping_element'].name in target_names
      ],
      flap_info,
      target_names,
  )
  post_by_flap = _group_by_narrowed_ancestor(
      post_narrow, flap_info, target_names
  )

  narrowed_ok: list[geometric_fold_3d_heuristics.ElementShapingResult] = []
  fallback_unnarrowed: list[
      geometric_fold_3d_heuristics.ElementShapingResult
  ] = []

  for rect in narrow_rects:
    seeds = seed_by_flap.get(rect.name, ())
    if len(seeds) != 1:
      raise ValueError(
          f'Expected exactly one seed fold for narrowed flap {rect.name!r},'
          f' got {len(seeds)}.'
      )
    seed_creases = seeds[0]['shaping_creases']
    try:
      if not seed_creases:
        raise ValueError('No seed zigzag creases to narrow with.')
      if not math.isclose(seed_creases[0].fold_percent, 1.0, abs_tol=1e-3):
        raise ValueError('Seed creases have fold_percent < 1.0.')
      cp = narrow_flap_shaping.narrow(
          cp=cp,
          zigzag_creases=list(seed_creases),
          rectangles=(rect,),
          grid_size=base_cp.grid_size,
          raise_error_on_ff_violation=True,
      )
      narrowed_ok.extend(post_by_flap.get(rect.name, ()))
    except ValueError as err:
      logging.warning('Narrowing skipped for flap %s: %s', rect.name, err)
      # Seed creases are already in `cp`; only add the remaining child folds.
      fallback_unnarrowed.extend(post_by_flap.get(rect.name, ()))

  if narrowed_ok:
    doubled = _infer_double_grid_creases(narrowed_ok, central_crease_removed)
    if doubled:
      cp = cp.update_with_shaping_creases(doubled)
  if fallback_unnarrowed:
    fallback_creases = list(
        itertools.chain.from_iterable(
            info['shaping_creases'] for info in fallback_unnarrowed
        )
    )
    if fallback_creases:
      cp = cp.update_with_shaping_creases(fallback_creases)
  return cp
