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

"""Adapter converting immutable bus records into legacy shaping inputs."""

import copy
import dataclasses
import logging
import types
from typing import Any

from .. import bus
from .. import rectangle as rectangle_lib
from ..crease_pattern import crease
from ..packing import compute_ridges
from ..packing import level_shifter as packing_ls
from ..packing import packing_output as packing_output_lib
from ..packing import types as packing_types
from ..packing import utils_packing_svg
from ..solving import central_fold_utils
from ..solving import hinges as hinge_utils
from ..stick_figures import stick as stick_lib
from ..stick_figures import stick_figure as stick_figure_lib


def unclipped_flap_rectangle(
    flap: bus.Flap,
    grid_size: int | None = None,
) -> rectangle_lib.Rectangle:
  """Reconstructs unclipped packing Rectangle around `flap.circle_centers`."""
  return flap.to_unclipped_rectangle(grid_size)


def ensure_unmerged_flap_shapes(
    unmerged_tree: stick_figure_lib.StickFigure,
    merged_tree: stick_figure_lib.StickFigure | None,
) -> None:
  """Ensures `flap_tip_shape` is populated on `unmerged_tree` from `merged_tree`."""
  if merged_tree is None:
    return
  if not any(
      s.is_flap and s.flap_tip_shape in (None, stick_lib.FlapTipShape.UNSET)
      for s in unmerged_tree.children
  ):
    return
  flap_shapes = {
      s.name: s.flap_tip_shape
      for s in merged_tree.children
      if s.is_flap
      and s.flap_tip_shape not in (None, stick_lib.FlapTipShape.UNSET)
  }
  if flap_shapes:
    try:
      compute_ridges.assign_flap_shapes_to_unmerged_tree(
          unmerged_tree, flap_shapes
      )
    except (KeyError, ValueError) as err:
      logging.warning('Could not assign flap shapes to unmerged tree: %s', err)


def legacy_inputs(
    solved: bus.SolvedCreasePattern,
) -> tuple[packing_output_lib.PackingOutput, hinge_utils.SolverResult]:
  """Builds legacy `(PackingOutput, SolverResult)` from `SolvedCreasePattern` bus data."""
  packed = solved.packed
  unmerged_sf = (
      packed.design.source if packed.design is not None else packed.packed_tree
  )
  merged_sf = (
      packed.packed_tree
      if packed.packed_tree is not None
      else (packed.design.merged if packed.design is not None else None)
  )
  if unmerged_sf is not None:
    unmerged_sf = copy.deepcopy(unmerged_sf)
    ensure_unmerged_flap_shapes(unmerged_sf, merged_sf)

  sfs: list[packing_types.SacrificialFlap] | None = None
  ls_rect_by_id: dict[str, rectangle_lib.Rectangle] = {}
  if packed.level_shift_reservations:
    sfs, ls_rect_by_id = packing_ls.reconstruct_pocket_packing_data(
        grid_size=packed.grid_size,
        flaps=packed.flaps,
        rivers=packed.rivers,
        reservations=packed.reservations,
        level_shift_reservations=packed.level_shift_reservations,
    )

  prefix = packing_types.AXIS_FILLER_PREFIX
  flap_rectangles = [
      unclipped_flap_rectangle(f, grid_size=packed.grid_size)
      for f in packed.flaps.values()
  ]
  for r in packed.reservations:
    if r.id in ls_rect_by_id:
      flap_rectangles.append(ls_rect_by_id[r.id])
    else:
      res_name = r.id if r.id.startswith(prefix) else f'{prefix}_{r.id}'
      flap_rectangles.append(r.rect.to_rectangle(name=res_name))
  packing_rivers = {
      r_id: r.to_packing_river(packed.grid_size)
      for r_id, r in packed.rivers.items()
  }
  if flap_rectangles or packing_rivers:
    poly_map = {
        f.id: types.SimpleNamespace(
            offset=(f.rect.x, f.rect.y),
            size=(f.rect.width, f.rect.height),
        )
        for f in packed.flaps.values()
    }
    for r in packed.reservations:
      poly_id = (
          ls_rect_by_id[r.id].name
          if r.id in ls_rect_by_id
          else (r.id if r.id.startswith(prefix) else f'{prefix}_{r.id}')
      )
      poly_map[poly_id] = types.SimpleNamespace(
          offset=(r.rect.x, r.rect.y),
          size=(r.rect.width, r.rect.height),
      )
    tracker_proxy: Any = types.SimpleNamespace(
        packed_polygons=poly_map,
        packed_rivers=packing_rivers,
    )
    hinge_infos = utils_packing_svg.extract_hinges_info(
        tracker_proxy, packed.grid_size
    )
  else:
    hinge_infos = [h.to_hinge_info(packed.grid_size) for h in packed.hinges]

  tiling_details: Any = packed.diagnostics.to_tiling_details()
  packing_output = packing_output_lib.PackingOutput(
      success=True,
      packing_cp=packed.creases,
      flap_rectangles=flap_rectangles,
      packing_rivers=packing_rivers,
      stick_figure=merged_sf,
      unmerged_stick_figure=unmerged_sf,
      hinges=hinge_infos,
      sacrificial_flaps=sfs,
      tracker=None,
      details=tiling_details,
      config=packed.config,
      error_message=None,
  )

  if solved.assigned_hinges or solved.assigned_pleats:
    hinges = list(solved.assigned_hinges)
    pleats = list(solved.assigned_pleats)
    ridges = [
        c
        for c in solved.creases.creases
        if c.fold != crease.Fold.BORDER and c.is_diagonal
    ]
  else:
    # Heuristic hinge/pleat classification fallback only covers
    # from_crease_pattern inputs (e.g. SVG uploads lacking solver state).
    packed_hinge_creases = [
        c for c in packed.creases.creases if c.fold == crease.Fold.HINGE
    ]
    hinges = []
    pleats = []
    ridges = []
    for c in solved.creases.creases:
      if c.fold == crease.Fold.BORDER:
        continue
      if c.is_diagonal:
        ridges.append(c)
      elif any(c.is_overlapping(ph) for ph in packed_hinge_creases):
        hinges.append(c)
      else:
        pleats.append(c)

  cfg = dataclasses.replace(
      solved.config,
      grid_size=solved.grid_size,
      remove_central_crease=solved.central_crease_removed,
  )
  state = hinge_utils.SolverStateSnapshot(
      pleats=pleats,
      unassigned_pleats=[],
      hinges=hinges,
      unsplit_hinges=[],
      unvisited_hinges=[],
      state_score=solved.stats.state_score,
      ridges=ridges,
      expanded=True,
      solver_config=cfg,
      transform=solved.symmetry,
      partition_progress=solved.stats.partition_progress,
  )
  solver_result = hinge_utils.SolverResult(
      crease_pattern=solved.creases,
      final_state=state,
      is_globally_flat_foldable=solved.is_globally_flat_foldable,
      time_taken=solved.stats.time_taken,
      base_type=solved.base_type,
  )

  if solved.central_crease_removed and packing_output.packing_cp is not None:
    try:
      repacked = central_fold_utils.repack_cp_with_shortened_flaps(
          packing_output,
          solved.creases.grid_size,
      )
      packing_output = repacked.model_copy(
          update={
              'unmerged_stick_figure': packing_output.unmerged_stick_figure,
              'sacrificial_flaps': packing_output.sacrificial_flaps,
              'config': packing_output.config,
          }
      )
    except Exception as err:
      logging.warning(
          'Failed to repack CP with shortened flaps for central crease'
          ' removal: %s',
          err,
      )
      raise

  return packing_output, solver_result
