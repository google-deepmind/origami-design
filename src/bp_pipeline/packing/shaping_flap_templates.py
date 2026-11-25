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

"""Composable `KneeTemplate` and `TailTemplate` block detection and packing."""

from collections.abc import Sequence
import copy
import dataclasses
import itertools
import math
from typing import Literal

import numpy as np

from ..stick_figures import packing_utils
from ..stick_figures import shaping_templates as sf_templates
from ..stick_figures import stick as stick_lib
from ..stick_figures import stick_figure as stick_figure_lib
from . import config as packing_config_lib
from . import river as river_lib
from . import scoring
from . import tiling_solver
from . import types as packing_types
from . import utils_packing_flaps
from . import utils_packing_svg
from . import utils_river_solver
from . import utils_tracker
from .debug import debug_types

Orientation = Literal['Wide', 'Tall']
CornerName = Literal['BottomLeft', 'BottomRight', 'TopLeft', 'TopRight']
AxisEdge = Literal['bottom', 'top']

_CORNER_NAMES: tuple[CornerName, ...] = (
    'BottomLeft',
    'BottomRight',
    'TopLeft',
    'TopRight',
)
_LEFT_CORNERS: tuple[CornerName, ...] = ('BottomLeft', 'TopLeft')

has_shaping_flaps = sf_templates.has_shaping_flaps
validate_shaping_flap_templates = sf_templates.validate_shaping_flap_templates


@dataclasses.dataclass(frozen=True)
class KneeTemplate:
  """3-stick joint template (`river` + `normal_flap` + `shaping_flap`).

  Attributes:
    shaping_flap: Auxiliary `shaping_flap=True` leaf stick at the joint.
    normal_flap: Primary distal leaf stick (`shaping_flap=False`) at the joint.
    other: Proximal river stick connecting the joint to the rest of the figure.
    joint_name: Shared degree-3 joint name.
  """

  shaping_flap: stick_lib.Stick
  normal_flap: stick_lib.Stick
  other: stick_lib.Stick
  joint_name: str

  def inner_size(self, orientation: Orientation) -> tuple[int, int]:
    """Returns on-paper `(width, height)` of the flap pair before river wrap."""
    span_along = int(self.normal_flap.length) + 2 * int(
        self.shaping_flap.length
    )
    extra_n = 2 * max(0, int(math.ceil(self.normal_flap.width)) - 1)
    span_perp = max(
        int(self.normal_flap.length) + extra_n,
        int(self.shaping_flap.length),
    )
    if orientation == 'Wide':
      return (span_along, span_perp)
    return (span_perp, span_along)

  def total_size(self, orientation: Orientation) -> tuple[int, int]:
    """Returns on-paper `(width, height)` of the full corner block + river."""
    inner_w, inner_h = self.inner_size(orientation)
    river_w = int(self.other.length)
    return (inner_w + river_w, inner_h + river_w)

  def pack_polygons(
      self,
      orientation: Orientation,
      corner: CornerName,
      grid_size: int,
      existing_occupancy: np.ndarray | None = None,
  ) -> tuple[dict[str, packing_types.PolygonPlacement], river_lib.HuggingRiver]:
    """Places `normal_flap`, `shaping_flap`, and `other` river in `corner`."""
    ln = int(self.normal_flap.length)
    ls = int(self.shaping_flap.length)
    rw = int(self.other.length)
    extra_n = 2 * max(0, int(math.ceil(self.normal_flap.width)) - 1)
    h_inner = max(ln + extra_n, ls)
    total_w, total_h = self.total_size(orientation)
    if total_w > grid_size or total_h > grid_size:
      raise ValueError(
          f'KneeTemplate ({total_w}x{total_h}) exceeds grid_size={grid_size}.'
      )

    on_right = corner in ('BottomRight', 'TopRight')
    on_top = corner in ('TopLeft', 'TopRight')

    if orientation == 'Wide':
      norm_size = (2 * ln, 2 * h_inner)
      sf_size = (2 * ls, 2 * h_inner)
      norm_x = grid_size - ln if on_right else -ln
      norm_y = grid_size - h_inner if on_top else -h_inner
      sf_x = grid_size - ln - 2 * ls if on_right else ln
      sf_y = norm_y
      inner_x0 = grid_size - (ln + 2 * ls) if on_right else 0
      inner_y0 = grid_size - h_inner if on_top else 0
      inner_w, inner_h = ln + 2 * ls, h_inner
      if corner == 'BottomLeft':
        start_cell = (inner_h, 0)
        hug_side = river_lib.HugSide.LEFT
      elif corner == 'BottomRight':
        start_cell = (inner_h, grid_size - 1)
        hug_side = river_lib.HugSide.RIGHT
      elif corner == 'TopLeft':
        start_cell = (inner_y0 - 1, 0)
        hug_side = river_lib.HugSide.RIGHT
      else:
        start_cell = (inner_y0 - 1, grid_size - 1)
        hug_side = river_lib.HugSide.LEFT
    else:
      norm_size = (2 * h_inner, 2 * ln)
      sf_size = (2 * h_inner, 2 * ls)
      norm_x = grid_size - h_inner if on_right else -h_inner
      norm_y = grid_size - ln if on_top else -ln
      sf_x = norm_x
      sf_y = grid_size - ln - 2 * ls if on_top else ln
      inner_x0 = grid_size - h_inner if on_right else 0
      inner_y0 = grid_size - (ln + 2 * ls) if on_top else 0
      inner_w, inner_h = h_inner, ln + 2 * ls
      if corner == 'BottomLeft':
        start_cell = (0, inner_w)
        hug_side = river_lib.HugSide.RIGHT
      elif corner == 'BottomRight':
        start_cell = (0, inner_x0 - 1)
        hug_side = river_lib.HugSide.LEFT
      elif corner == 'TopLeft':
        start_cell = (grid_size - 1, inner_w)
        hug_side = river_lib.HugSide.LEFT
      else:
        start_cell = (grid_size - 1, inner_x0 - 1)
        hug_side = river_lib.HugSide.RIGHT

    block_x0 = grid_size - total_w if on_right else 0
    block_y0 = grid_size - total_h if on_top else 0
    if existing_occupancy is not None:
      if existing_occupancy[
          block_y0 : block_y0 + total_h, block_x0 : block_x0 + total_w
      ].any():
        raise ValueError(
            f'KneeTemplate block at {corner} overlaps occupied grid cells.'
        )

    placements = {
        self.normal_flap.name: packing_types.PolygonPlacement.from_grid_coords(
            size=norm_size,
            offset=(norm_x, norm_y),
            grid_size=grid_size,
            require_on_paper=False,
        ),
        self.shaping_flap.name: packing_types.PolygonPlacement.from_grid_coords(
            size=sf_size,
            offset=(sf_x, sf_y),
            grid_size=grid_size,
            require_on_paper=False,
        ),
    }

    local_occ = np.zeros((grid_size, grid_size), dtype=bool)
    local_occ[inner_y0 : inner_y0 + inner_h, inner_x0 : inner_x0 + inner_w] = (
        True
    )
    river = river_lib.HuggingRiver.from_contour(
        grid_size=grid_size,
        width=rw,
        occupancy_grid=local_occ,
        start_cell=start_cell,
        hug_side=hug_side,
    )
    return placements, river


@dataclasses.dataclass(frozen=True)
class TailTemplate:
  """Alternating `SF_0 - R_0 - SF_1 - R_1 - ...` segmented tail template.

  Attributes:
    shaping_flaps: Ordered tuple of `shaping_flap=True` sticks from tip to base.
    connecting_rivers: Ordered tuple of connecting river sticks from tip to base
      (`len == len(shaping_flaps)`).
    tip_flap: Optional non-`shaping_flap` terminal tail-tip stick sharing the
      innermost tail joint with `shaping_flaps[0]` and `connecting_rivers[0]`.
  """

  shaping_flaps: tuple[stick_lib.Stick, ...]
  connecting_rivers: tuple[stick_lib.Stick, ...]
  tip_flap: stick_lib.Stick | None = None

  def __post_init__(self) -> None:
    if not self.shaping_flaps:
      raise ValueError('TailTemplate requires at least 1 shaping flap.')
    if len(self.shaping_flaps) != len(self.connecting_rivers):
      raise ValueError(
          'TailTemplate requires equal counts of shaping_flaps and'
          f' connecting_rivers: {len(self.shaping_flaps)} vs'
          f' {len(self.connecting_rivers)}.'
      )

  def total_size(self, *, min_total_width: int = 0) -> tuple[int, int]:
    """Returns `(total_width, total_height)` of the packed tail stack."""
    widths, heights = self._band_layout(min_total_width=min_total_width)
    tip_h = int(self.tip_flap.length) if self.tip_flap is not None else 0
    total_w = widths[-1] + 2 * int(self.connecting_rivers[-1].length)
    total_h = (
        tip_h
        + sum(heights)
        + sum(int(r.length) for r in self.connecting_rivers)
    )
    return (total_w, total_h)

  def _band_layout(
      self, *, min_total_width: int = 0
  ) -> tuple[list[int], list[int]]:
    """Computes `(flap_widths, on_paper_flap_heights)` from tip `0` outward."""
    w0 = 2
    if self.tip_flap is not None:
      lt = int(self.tip_flap.length)
      wt_decl = max(2, int(math.ceil(self.tip_flap.width)))
      if wt_decl % 2 == 1:
        wt_decl += 1
      w0 = max(w0, 2 * lt, wt_decl)
    cum_rw = 0
    for idx, sf in enumerate(self.shaping_flaps):
      lf = int(sf.length)
      wf_decl = max(2, int(math.ceil(sf.width)))
      if wf_decl % 2 == 1:
        wf_decl += 1
      needed = max(2 * lf, wf_decl)
      w0 = max(w0, needed - 2 * cum_rw)
      cum_rw += int(self.connecting_rivers[idx].length)
    if min_total_width > 0:
      w0 = max(w0, int(min_total_width) - 2 * cum_rw)
    if w0 % 2 == 1:
      w0 += 1

    widths: list[int] = []
    on_paper_heights: list[int] = []
    for idx, sf in enumerate(self.shaping_flaps):
      lf = int(sf.length)
      if idx == 0:
        widths.append(w0)
        on_paper_heights.append(2 * lf if self.tip_flap is not None else lf)
      else:
        widths.append(
            widths[-1] + 2 * int(self.connecting_rivers[idx - 1].length)
        )
        on_paper_heights.append(2 * lf)
    return widths, on_paper_heights

  def pack_polygons(
      self,
      tracker: utils_tracker.CreasePatternTracker,
      grid_size: int,
      *,
      edge: AxisEdge = 'bottom',
      min_total_width: int = 0,
      debug: debug_types.DebugConfig | None = None,
  ) -> tuple[
      dict[str, packing_types.PolygonPlacement],
      utils_tracker.CreasePatternTracker,
      river_lib.HuggingRiver,
  ]:
    """Packs concentric shaping flaps and hugging rivers on the symmetry axis."""
    total_w, total_h = self.total_size(min_total_width=min_total_width)
    if total_w > grid_size or total_h > grid_size:
      raise ValueError(
          f'TailTemplate ({total_w}x{total_h}) exceeds grid_size={grid_size}.'
      )

    center_x = grid_size // 2
    widths, _ = self._band_layout(min_total_width=min_total_width)
    placements: dict[str, packing_types.PolygonPlacement] = {}
    local_tracker = utils_tracker.CreasePatternTracker(grid_size)
    outer_river: river_lib.HuggingRiver | None = None

    y_cursor = 0 if edge == 'bottom' else grid_size
    if self.tip_flap is not None:
      l_tip = int(self.tip_flap.length)
      tip_w = widths[0]
      tip_left_x = center_x - tip_w // 2
      tip_offset_y = -l_tip if edge == 'bottom' else grid_size - l_tip
      tip_placement = packing_types.PolygonPlacement.from_grid_coords(
          size=(tip_w, 2 * l_tip),
          offset=(tip_left_x, tip_offset_y),
          grid_size=grid_size,
          require_on_paper=False,
      )
      placements[self.tip_flap.name] = tip_placement
      tracker.add_polygon(self.tip_flap.name, tip_placement)
      local_tracker.add_polygon(self.tip_flap.name, tip_placement)
      if debug is not None and debug.is_active():
        debug.record_step(
            tracker, 'INFO', f'Placed TailTemplate tip {self.tip_flap.name}'
        )
      y_cursor = l_tip if edge == 'bottom' else grid_size - l_tip

    for idx, (sf, rv) in enumerate(
        zip(self.shaping_flaps, self.connecting_rivers, strict=True)
    ):
      lf = int(sf.length)
      rw = int(rv.length)
      flap_w = widths[idx]
      flap_h = 2 * lf
      left_x = center_x - flap_w // 2
      edge_anchored = idx == 0 and self.tip_flap is None

      if edge == 'bottom':
        offset_y = -lf if edge_anchored else y_cursor
        next_y_after_flap = lf if edge_anchored else y_cursor + flap_h
        start_cell = (0, left_x - 1)
        hug_side = river_lib.HugSide.LEFT
      else:
        offset_y = grid_size - lf if edge_anchored else y_cursor - flap_h
        next_y_after_flap = (
            grid_size - lf if edge_anchored else y_cursor - flap_h
        )
        start_cell = (grid_size - 1, left_x - 1)
        hug_side = river_lib.HugSide.RIGHT

      placement = packing_types.PolygonPlacement.from_grid_coords(
          size=(flap_w, flap_h),
          offset=(left_x, offset_y),
          grid_size=grid_size,
          require_on_paper=False,
      )
      placements[sf.name] = placement
      tracker.add_polygon(sf.name, placement)
      local_tracker.add_polygon(sf.name, placement)
      if debug is not None and debug.is_active():
        debug.record_step(
            tracker, 'INFO', f'Placed TailTemplate flap {sf.name}'
        )

      river = river_lib.HuggingRiver.from_contour(
          grid_size=grid_size,
          width=rw,
          occupancy_grid=local_tracker.occupancy_grid,
          start_cell=start_cell,
          hug_side=hug_side,
      )
      tracker.add_river(rv.name, river)
      local_tracker.add_river(rv.name, river)
      outer_river = river
      if debug is not None and debug.is_active():
        debug.record_step(
            tracker, 'INFO', f'Placed TailTemplate river {rv.name}'
        )

      if edge == 'bottom':
        y_cursor = next_y_after_flap + rw
      else:
        y_cursor = next_y_after_flap - rw

    if outer_river is None:
      raise ValueError('TailTemplate placed no rivers.')
    return placements, tracker, outer_river


def find_tail_templates(
    tree: stick_figure_lib.StickFigure,
    *,
    merge_chains: bool = True,
) -> tuple[list[TailTemplate], set[str]]:
  """Finds all `TailTemplate` chains in `tree` (merged or unmerged)."""
  tail_tuples, _, claimed = sf_templates.discover_shaping_template_tuples(
      tree, merge_chains=merge_chains
  )
  templates = [
      TailTemplate(
          shaping_flaps=flaps,
          connecting_rivers=rivers,
          tip_flap=tip,
      )
      for flaps, rivers, tip in tail_tuples
  ]
  return templates, claimed


def find_knee_templates(
    tree: stick_figure_lib.StickFigure,
    exclude_sticks: set[str] | None = None,
    *,
    merge_chains: bool = True,
) -> list[KneeTemplate]:
  """Finds all degree-3 `KneeTemplate` joints (`river + normal_flap + SF`)."""
  merged_tree = tree.merge_linear_chains()[0] if merge_chains else tree
  excluded = exclude_sticks or set()
  templates: list[KneeTemplate] = []
  used_joints: set[str] = set()

  for joint_name, joint in merged_tree.joints.items():
    if joint_name in used_joints or joint.degree != 3:
      continue
    sticks = joint.sticks
    if any(s.name in excluded for s in sticks):
      continue
    sf_sticks = [
        s
        for s in sticks
        if s.shaping_flap
        and (
            merged_tree.joints[s.start].degree == 1
            or merged_tree.joints[s.end].degree == 1
        )
    ]
    normal_flaps = [
        s for s in sticks if not s.shaping_flap and not s.is_river and s.is_flap
    ]
    rivers = [s for s in sticks if not s.shaping_flap and s.is_river]
    if len(sf_sticks) == 1 and len(normal_flaps) == 1 and len(rivers) == 1:
      templates.append(
          KneeTemplate(
              shaping_flap=sf_sticks[0],
              normal_flap=normal_flaps[0],
              other=rivers[0],
              joint_name=joint_name,
          )
      )
      for s in sticks:
        used_joints.add(s.start)
        used_joints.add(s.end)
  return templates


def describe_shaping_templates(tree: stick_figure_lib.StickFigure) -> str:
  """Returns a concise summary naming the discovered templates in `tree`."""
  tails, tail_claimed = find_tail_templates(tree, merge_chains=False)
  knees = find_knee_templates(
      tree, exclude_sticks=tail_claimed, merge_chains=False
  )
  parts: list[str] = []
  for tail in tails:
    flaps = [s.name for s in tail.shaping_flaps]
    rivers = [r.name for r in tail.connecting_rivers]
    if tail.tip_flap is not None:
      parts.append(
          f'TailTemplate(tip={tail.tip_flap.name!r}, flaps={flaps},'
          f' rivers={rivers})'
      )
    else:
      parts.append(f'TailTemplate(flaps={flaps}, rivers={rivers})')
  for kt in knees:
    parts.append(
        f'KneeTemplate(joint={kt.joint_name!r},'
        f' shaping_flap={kt.shaping_flap.name!r},'
        f' normal_flap={kt.normal_flap.name!r})'
    )
  return ', '.join(parts) if parts else 'shaping-flap templates'


def estimate_template_min_grid_size(
    merged_tree: stick_figure_lib.StickFigure,
    is_symmetric: bool,
) -> int:
  """Estimates a starting grid size for the builtin template packer.

  This value is a heuristic estimate rather than a strict lower bound, and the
  packing grid-size sweep only steps upward from it.

  Args:
    merged_tree: Merged stick figure containing shaping-flap templates.
    is_symmetric: Whether bilateral symmetry is enforced.

  Returns:
    Estimated starting integer grid size for the sweep.
  """
  base_est = merged_tree.get_packing_gridsize()
  tails, tail_claimed = find_tail_templates(merged_tree, merge_chains=False)
  knees = find_knee_templates(
      merged_tree, exclude_sticks=tail_claimed, merge_chains=False
  )

  min_w = 4
  min_h = 4
  for t in tails:
    tw, th = t.total_size()
    min_w = max(min_w, tw)
    min_h = max(min_h, th + 2)

  seen_knees: set[str] = set()
  unique_knees: list[KneeTemplate] = []
  for kt in knees:
    if kt.shaping_flap.name in seen_knees:
      continue
    seen_knees.add(kt.shaping_flap.name)
    if kt.shaping_flap.symmetric_stick_name:
      seen_knees.add(kt.shaping_flap.symmetric_stick_name)
    unique_knees.append(kt)

  if unique_knees:
    first_kt = unique_knees[0]
    kw, kh = first_kt.total_size('Wide')
    k_min_span = min(kw, kh)
    k_max_span = max(kw, kh)
    if is_symmetric and first_kt.shaping_flap.symmetric_stick_name:
      min_w = max(min_w, 2 * k_min_span)
    else:
      min_w = max(min_w, k_max_span)
    if tails:
      min_w = max(min_w, 2 * k_min_span if is_symmetric else k_max_span)
      min_h = max(min_h, k_min_span + tails[0].total_size()[1] + 2)
    else:
      min_h = max(min_h, k_max_span)

  covered_sticks = set(tail_claimed)
  min_cells = 0
  for t in tails:
    tw, th = t.total_size()
    min_cells += tw * th
  used_corners = 0
  for kt in knees:
    covered_sticks.update(
        (kt.shaping_flap.name, kt.normal_flap.name, kt.other.name)
    )
    kw, kh = kt.total_size('Wide')
    min_cells += kw * kh
    used_corners += 1
  free_corners = max(0, 4 - used_corners)
  rem_leaf_lengths: list[int] = []
  rem_river_cells = 0
  for stick in merged_tree.children:
    if stick.name in covered_sticks:
      continue
    if stick.is_river:
      rem_river_cells += 4 * max(1, int(stick.length))
    else:
      rem_leaf_lengths.append(max(1, int(stick.length)))
  rem_leaf_lengths.sort(reverse=True)
  for idx_leaf, length_val in enumerate(rem_leaf_lengths):
    factor = 1 if idx_leaf < free_corners else 2
    min_cells += factor * length_val * length_val
  min_cells += rem_river_cells
  area_side = int(math.ceil(math.sqrt(max(16, min_cells))))

  span_bound = max(min_w, min_h, area_side)
  est = max(span_bound, min(base_est, span_bound + 4))
  if is_symmetric and est % 2 == 1:
    est += 1
  return est


def _place_templates_on_tracker(
    half_or_full_tree: stick_figure_lib.StickFigure,
    full_tree: stick_figure_lib.StickFigure,
    grid_size: int,
    is_symmetric: bool,
    tails: Sequence[TailTemplate],
    knees: Sequence[KneeTemplate],
    tail_edges: Sequence[AxisEdge],
    knee_corners: Sequence[CornerName],
    knee_orientations: Sequence[Orientation],
    tail_min_widths: Sequence[int] | None = None,
) -> (
    tuple[
        utils_tracker.CreasePatternTracker,
        set[str],
        dict[str, dict[str, str]],
    ]
    | None
):
  """Places all `TailTemplate`s and `KneeTemplate`s onto one fresh tracker."""
  tracker = utils_tracker.CreasePatternTracker(grid_size)
  debug = debug_types.DebugConfig(live_plot=False, save_history=False)
  ignored_in_half: set[str] = set()
  template_joint_to_side: dict[str, dict[str, str]] = {}
  half_names = {s.name for s in half_or_full_tree.children}
  req_widths = (
      tail_min_widths if tail_min_widths is not None else (0,) * len(tails)
  )

  deferred_river_adj: list[
      tuple[str, river_lib.HuggingRiver, dict[str, str]]
  ] = []

  for tail, edge, req_min_w in zip(tails, tail_edges, req_widths, strict=True):
    outer_stick = tail.connecting_rivers[-1]
    inner_sf = tail.shaping_flaps[-1]
    shared = outer_stick.shared_joint_name(inner_sf)
    inner_joint = shared[0] if shared else outer_stick.start
    outer_joint = outer_stick.other_joint_name(inner_joint)

    cojoint_min_w = int(req_min_w)
    same_edge_kws: list[int] = []
    for kt, corner, orient in zip(
        knees, knee_corners, knee_orientations, strict=True
    ):
      same_edge = (
          edge == 'bottom' and corner in ('BottomLeft', 'BottomRight')
      ) or (edge == 'top' and corner in ('TopLeft', 'TopRight'))
      if not same_edge:
        continue
      kt_w, _ = kt.total_size(orient)
      same_edge_kws.append(kt_w)
      kt_outer_joint = kt.other.other_joint_name(kt.joint_name)
      if kt_outer_joint == outer_joint:
        target_w = grid_size - (2 * kt_w if is_symmetric else kt_w)
        if target_w < tail.total_size()[0]:
          return None
        cojoint_min_w = max(cojoint_min_w, target_w)
    if same_edge_kws:
      reserved_w = (
          2 * max(same_edge_kws) if is_symmetric else sum(same_edge_kws)
      )
      if tail.total_size(min_total_width=cojoint_min_w)[0] > (
          grid_size - reserved_w
      ):
        return None

    try:
      _, tracker, outer_river = tail.pack_polygons(
          tracker,
          grid_size,
          edge=edge,
          min_total_width=cojoint_min_w,
          debug=debug,
      )
    except ValueError:
      return None
    j2s_tail = {inner_joint: 'hugging', outer_joint: 'opposite'}
    template_joint_to_side[outer_stick.name] = j2s_tail
    deferred_river_adj.append((outer_stick.name, outer_river, j2s_tail))
    if tail.tip_flap is not None and tail.tip_flap.name in half_names:
      ignored_in_half.add(tail.tip_flap.name)
    for s in tail.shaping_flaps:
      if s.name in half_names:
        ignored_in_half.add(s.name)
    for r in tail.connecting_rivers:
      if r.name in half_names:
        ignored_in_half.add(r.name)

  for kt, corner, orient in zip(
      knees, knee_corners, knee_orientations, strict=True
  ):
    if is_symmetric and corner not in _LEFT_CORNERS:
      return None
    try:
      placements, river = kt.pack_polygons(
          orient,
          corner,
          grid_size,
          existing_occupancy=tracker.occupancy_grid,
      )
      for s_obj in (kt.normal_flap, kt.shaping_flap):
        tracker.add_polygon(s_obj.name, placements[s_obj.name])
        if s_obj.name in half_names:
          ignored_in_half.add(s_obj.name)
      tracker.add_river(kt.other.name, river)
      if kt.other.name in half_names:
        ignored_in_half.add(kt.other.name)
      other_joint = kt.other.other_joint_name(kt.joint_name)
      j2s_knee = {kt.joint_name: 'hugging', other_joint: 'opposite'}
      template_joint_to_side[kt.other.name] = j2s_knee
      deferred_river_adj.append((kt.other.name, river, j2s_knee))

      if is_symmetric:
        sym_norm = full_tree[kt.normal_flap.name].symmetric_stick_name
        sym_sf = full_tree[kt.shaping_flap.name].symmetric_stick_name
        sym_rv = full_tree[kt.other.name].symmetric_stick_name
        if sym_norm and sym_norm != kt.normal_flap.name:
          tracker.add_polygon(
              sym_norm,
              utils_packing_svg.get_mirrored_polygon_placement(
                  placements[kt.normal_flap.name], grid_size
              ),
          )
          if sym_norm in half_names:
            ignored_in_half.add(sym_norm)
        if sym_sf and sym_sf != kt.shaping_flap.name:
          tracker.add_polygon(
              sym_sf,
              utils_packing_svg.get_mirrored_polygon_placement(
                  placements[kt.shaping_flap.name], grid_size
              ),
          )
          if sym_sf in half_names:
            ignored_in_half.add(sym_sf)
        if sym_rv and sym_rv != kt.other.name:
          tracker.add_river(sym_rv, river.mirror(grid_size))
          if sym_rv in half_names:
            ignored_in_half.add(sym_rv)
    except ValueError:
      return None

  for rv_name, rv_obj, j2s in deferred_river_adj:
    utils_river_solver.add_river_adjacent_cells(tracker, rv_name, rv_obj, j2s)

  return tracker, ignored_in_half, template_joint_to_side


def solve_trackers_with_shaping_templates(
    full_tree: stick_figure_lib.StickFigure,
    tree_to_pack: stick_figure_lib.StickFigure,
    *,
    grid_size: int,
    is_symmetric: bool,
    symmetry_axes: dict[str, bool] | None,
    pockets_info: list[packing_utils.FlapPocket] | None,
    default_traversal_plan: list[packing_types.TraversalStep],
    debug: debug_types.DebugConfig,
    scoring_config: scoring.ScoringConfig,
    max_solutions: int,
    config: packing_config_lib.PackingConfig,
) -> list[utils_tracker.CreasePatternTracker]:
  """Packs `tree_to_pack` using composable `TailTemplate` + `KneeTemplate`s.

  Args:
    full_tree: Full merged stick figure (with `shaping_flap=True` preserved).
    tree_to_pack: Half-figure (when `is_symmetric`) or `full_tree`.
    grid_size: Target integer grid size.
    is_symmetric: Whether bilateral symmetry is enforced.
    symmetry_axes: Optional map from stick name to `is_on_symmetry_axis`.
    pockets_info: Precomputed intermediate flap pockets.
    default_traversal_plan: Traversal plan over `tree_to_pack`.
    debug: Packing debug configuration.
    scoring_config: Tiling scoring configuration.
    max_solutions: Maximum number of tiled trackers to return.
    config: Packing configuration.

  Returns:
    List of fully tiled `CreasePatternTracker`s (empty when `grid_size` cannot
    accommodate the figure).

  Raises:
    ValueError: If any `shaping_flap=True` stick does not belong to a valid
      `TailTemplate` or `KneeTemplate`, or if template counts exceed the builtin
      packer limits (`<= 2` tails and `<= 4` corner knees / `<= 2` symmetric
      pairs).
  """
  validation_errors = sf_templates.validate_shaping_flap_templates(full_tree)
  if validation_errors:
    raise ValueError(' '.join(validation_errors))

  tails, tail_claimed = find_tail_templates(tree_to_pack, merge_chains=False)
  knees = find_knee_templates(
      tree_to_pack, exclude_sticks=tail_claimed, merge_chains=False
  )
  if is_symmetric:
    deduped_knees: list[KneeTemplate] = []
    seen_knee_sf: set[str] = set()
    for kt in knees:
      sf_name = kt.shaping_flap.name
      if sf_name in seen_knee_sf:
        continue
      seen_knee_sf.add(sf_name)
      sym_sf = full_tree[sf_name].symmetric_stick_name
      if sym_sf:
        seen_knee_sf.add(sym_sf)
      deduped_knees.append(kt)
    knees = deduped_knees
  if len(tails) > sf_templates.MAX_TAIL_TEMPLATES:
    raise ValueError(
        f'At most {sf_templates.MAX_TAIL_TEMPLATES} TailTemplates are'
        f' supported, got {len(tails)}.'
    )
  avail_corners = _LEFT_CORNERS if is_symmetric else _CORNER_NAMES
  if len(knees) > len(avail_corners):
    raise ValueError(
        f'At most {len(avail_corners)}'
        f' {"mirrored KneeTemplate pairs" if is_symmetric else "corner KneeTemplates"}'
        f' are supported, got {len(knees)}.'
    )
  if not tails and not knees:
    raise ValueError(
        'Figure contains shaping_flap sticks that do not form any valid'
        ' TailTemplate or KneeTemplate.'
    )

  if is_symmetric and grid_size % 2 == 1:
    return []

  if not tails:
    edge_assignments: list[tuple[AxisEdge, ...]] = [()]
  elif len(tails) == 1:
    edge_assignments = [('bottom',), ('top',)]
  else:
    edge_assignments = [('bottom', 'top'), ('top', 'bottom')]

  corner_perms = list(itertools.permutations(avail_corners, len(knees)))
  orient_options: tuple[Orientation, Orientation] = ('Wide', 'Tall')
  orient_combos = list(itertools.product(orient_options, repeat=len(knees)))

  template_stick_names = set(tail_claimed)
  for kt in knees:
    for s_obj in (kt.shaping_flap, kt.normal_flap, kt.other):
      template_stick_names.add(s_obj.name)
      sym_s = full_tree[s_obj.name].symmetric_stick_name
      if sym_s:
        template_stick_names.add(sym_s)
  has_non_template_rivers = any(
      s.name not in template_stick_names and not s.is_flap
      for s in tree_to_pack.children
  )
  per_tail_w_opts: list[tuple[int, ...]] = []
  for tail in tails:
    nat_w, _ = tail.total_size()
    if nat_w < grid_size:
      per_tail_w_opts.append(
          (0, grid_size) if has_non_template_rivers else (grid_size, 0)
      )
    else:
      per_tail_w_opts.append((0,))
  tail_width_combos: list[tuple[int, ...]] = (
      list(itertools.product(*per_tail_w_opts)) if tails else [()]
  )

  cand_limit = (
      max(4, max_solutions * 2) if config.river_backtracking else max_solutions
  )
  tiled_trackers: list[utils_tracker.CreasePatternTracker] = []
  for t_edges in edge_assignments:
    for t_min_widths in tail_width_combos:
      for k_corners in corner_perms:
        for k_orients in orient_combos:
          placed = _place_templates_on_tracker(
              tree_to_pack,
              full_tree,
              grid_size,
              is_symmetric,
              tails,
              knees,
              t_edges,
              k_corners,
              k_orients,
              tail_min_widths=t_min_widths,
          )
          if placed is None:
            continue
          init_tracker, ignored_names, template_joint_to_side = placed

          rem_rivers = [
              s
              for s in tree_to_pack.children
              if s.name not in ignored_names and not s.is_flap
          ]
          if rem_rivers:
            template_options = packing_types.TemplateRiverOptions(
                preplaced_joint_to_side=template_joint_to_side,
                full_tree=full_tree,
                mirror_pocket_flaps=is_symmetric,
                infer_zero_area_hugging_sides=True,
                retry_opposite_hug_side=True,
                retry_without_pocket_start_cell=True,
                reorder_pocket_flaps=True,
            )
            river_trackers = utils_river_solver.solve_rivers_iteratively(
                tree_to_pack,
                debug=debug,
                grid_size=grid_size,
                is_symmetric=is_symmetric,
                symmetry_axes=symmetry_axes,
                scoring_config=scoring_config,
                max_solutions=max(1, max_solutions),
                initial_tracker=init_tracker,
                ignore_sticks=ignored_names,
                pockets_info=(
                    copy.deepcopy(pockets_info)
                    if pockets_info is not None
                    else None
                ),
                config=config,
                template_options=template_options,
            )
            if not river_trackers:
              continue
          else:
            river_trackers = [init_tracker]

          for trk in river_trackers:
            work_trk = copy.deepcopy(trk)
            already_packed = set(work_trk.packed_polygons) | set(
                work_trk.packed_rivers
            )
            rem_flaps = [
                step['stick']
                for step in default_traversal_plan
                if step['stick'].name not in already_packed
                and step['stick'].is_flap
            ]
            if rem_flaps:
              if not rem_rivers and tails:
                min_flap_half_w = min(
                    max(
                        1,
                        utils_packing_flaps.get_flap_dim(
                            f, config.use_level_shifter
                        )
                        // 2,
                    )
                    for f in rem_flaps
                )
                impossible_side_strip = False
                for tail, edge, req_w in zip(
                    tails, t_edges, t_min_widths, strict=True
                ):
                  has_same_edge_knee = any(
                      (
                          edge == 'bottom'
                          and c in ('BottomLeft', 'BottomRight')
                      )
                      or (edge == 'top' and c in ('TopLeft', 'TopRight'))
                      for c in k_corners
                  )
                  if not has_same_edge_knee:
                    actual_tw, _ = tail.total_size(min_total_width=req_w)
                    side_w = (grid_size - actual_tw) // 2
                    if 0 < side_w < min_flap_half_w:
                      impossible_side_strip = True
                      break
                if impossible_side_strip:
                  continue

              rem_names = {f.name for f in rem_flaps}
              rem_trav = [
                  step
                  for step in default_traversal_plan
                  if step['stick'].name in rem_names
              ]
              first_flap = rem_trav[0]['stick']
              conn_joint = (
                  first_flap.start
                  if full_tree.joints[first_flap.start].degree > 1
                  else first_flap.end
              )
              adj_cells: list[tuple[int, int]] = []
              for r in range(grid_size):
                for c in range(grid_size):
                  if work_trk.occupancy_grid[r, c]:
                    continue
                  cell_state = work_trk.grid[r, c]
                  if (
                      isinstance(cell_state, utils_packing_flaps.GridCellState)
                      and cell_state.type
                      == utils_packing_flaps.GridCellType.RIVER_ADJACENT
                      and cell_state.source_id == conn_joint
                  ):
                    adj_cells.append((r, c))
              if (
                  any(
                      s.name in work_trk.packed_rivers
                      for s in full_tree.joints[conn_joint].sticks
                      if s.is_river
                  )
                  and not adj_cells
              ):
                continue

              sac_polys = {
                  n: p
                  for n, p in work_trk.packed_polygons.items()
                  if n.endswith(packing_types.RIVER_SF1_SUFFIX)
                  or n.endswith(packing_types.RIVER_SF2_SUFFIX)
              }
              flap_configs = [config]
              if not rem_rivers and config.enforce_proximity_constraints:
                flap_configs.append(
                    dataclasses.replace(
                        config, enforce_proximity_constraints=False
                    )
                )
              packed_flaps = None
              for flap_cfg in flap_configs:
                try:
                  packed_flaps = tiling_solver.pack_flaps(
                      tracker=work_trk,
                      flaps_to_pack=rem_flaps,
                      traversal_plan=rem_trav,
                      debug=debug,
                      is_symmetric=is_symmetric,
                      symmetry_axes=symmetry_axes,
                      scoring_config=scoring_config,
                      config=flap_cfg,
                      river_adjacent_cells=adj_cells or None,
                      sacrificial_polygons=sac_polys or None,
                  )
                except (ValueError, packing_types.NoPackingFoundError):
                  packed_flaps = None
                if packed_flaps is not None:
                  break
              if packed_flaps is None:
                continue
              for flap_name, placement in packed_flaps.items():
                if flap_name not in work_trk.packed_polygons:
                  work_trk.add_polygon(flap_name, placement)
              if is_symmetric:
                utils_packing_svg.mirror_symmetric_placements(
                    work_trk, grid_size, full_tree
                )
              tiled_trackers.append(work_trk)
            else:
              if is_symmetric:
                utils_packing_svg.mirror_symmetric_placements(
                    work_trk, grid_size, full_tree
                )
              unfilled = grid_size * grid_size - int(
                  np.count_nonzero(work_trk.occupancy_grid)
              )
              if unfilled == 0:
                tiled_trackers.append(work_trk)

            if len(tiled_trackers) >= cand_limit:
              return tiled_trackers

  return tiled_trackers
