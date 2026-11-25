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

"""Sacrificial-pocket level-shifter repacking and river slot placement."""

import collections
from collections.abc import Callable, Iterator, Mapping, Sequence
import copy
import dataclasses
import enum
import math
from typing import Any

from absl import logging
import numpy as np

from .. import rectangle as rectangle_lib
from .. import utils_tree
from ..crease_pattern import point as point_lib
from ..crease_pattern.parser import svg as svg_parser
from ..stick_figures import stick as stick_lib
from ..stick_figures import stick_figure as stick_figure_lib
from . import flap_placement
from . import packing_output
from . import river as river_lib
from . import types
from . import utils_packing_flaps
from . import utils_packing_svg
from . import utils_tracker

VERTICAL_SACRIFICIAL_FLAP_SUFFIX = '__sacrificial_vertical'
HORIZONTAL_SACRIFICIAL_FLAP_SUFFIX = '__sacrificial_horizontal'
CORNER_SUFFIX = '__corner'
SACRIFICIAL_FLAP_LENGTH = 4
SACRIFICIAL_FLAP_WIDTH = 2

pocket_depth = utils_packing_flaps.level_shifter_pocket_depth
pocket_margin = utils_packing_flaps.level_shifter_center_margin


class Corner(enum.Enum):
  """Corner choice when decomposing a wide flap into an L-shaped pocket."""

  TOP_LEFT = 1
  TOP_RIGHT = 2
  BOTTOM_LEFT = 3
  BOTTOM_RIGHT = 4

  @property
  def is_left(self) -> bool:
    """Whether this corner sits on the left (`x_start`) side of the box."""
    return self in (Corner.TOP_LEFT, Corner.BOTTOM_LEFT)

  @property
  def is_top(self) -> bool:
    """Whether this corner sits on the top (`y_start`) side of the box."""
    return self in (Corner.TOP_LEFT, Corner.TOP_RIGHT)

  def get_coordinates(
      self, placement: types.PolygonPlacement, corner_size: int = 2
  ) -> tuple[int, int]:
    """Returns the `(x, y)` grid offset of a `corner_size` square corner."""
    x, y = placement.offset
    w, h = placement.size
    match self:
      case Corner.TOP_LEFT:
        return x, y
      case Corner.TOP_RIGHT:
        return x + w - corner_size, y
      case Corner.BOTTOM_LEFT:
        return x, y + h - corner_size
      case Corner.BOTTOM_RIGHT:
        return x + w - corner_size, y + h - corner_size


def validate_supported_level_shifter_widths(
    stick_figure: stick_figure_lib.StickFigure,
) -> None:
  """Validates that all level-shifted sticks in `stick_figure` have width 2."""
  for stick in stick_figure.children:
    if stick.requires_level_shifter and round(stick.width) != 2:
      raise ValueError(
          'Sacrificial-pocket level shifters currently support width == 2;'
          f' stick {stick.name!r} has width={stick.width}.'
      )


def find_valid_repack_corner(
    placement: types.PolygonPlacement,
    grid_size: int,
    corner_size: int,
) -> Corner:
  """Finds an on-paper corner of size `corner_size` inside `placement`."""
  for corner in Corner:
    x, y = corner.get_coordinates(placement, corner_size=corner_size)
    if flap_placement.is_valid_level_shifter_corner(
        off_x=x, off_y=y, grid_size=grid_size, size=corner_size
    ):
      return corner
  raise ValueError(
      f'No valid on-paper corner of size {corner_size} found for {placement}.'
  )


def _repack_box_components(
    stick_name: str,
    offset: tuple[int, int],
    size: tuple[int, int],
    corner_size: int,
    corner: Corner,
    grid_size: int,
) -> dict[str, types.PolygonPlacement]:
  """Splits box `(offset, size)` into 4 L-shaped pocket placements."""
  x, y = offset
  dim_w, dim_h = size
  rem_w = dim_w - corner_size
  rem_h = dim_h - corner_size
  is_left = corner.is_left
  is_top = corner.is_top
  corner_x = x if is_left else x + rem_w
  corner_y = y if is_top else y + rem_h
  vert_x = corner_x
  vert_y = y + corner_size if is_top else y
  horiz_x = x + corner_size if is_left else x
  horiz_y = corner_y
  main_x = horiz_x
  main_y = vert_y
  return {
      f'{stick_name}{CORNER_SUFFIX}': types.PolygonPlacement.from_grid_coords(
          offset=(corner_x, corner_y),
          size=(corner_size, corner_size),
          grid_size=grid_size,
      ),
      f'{stick_name}{VERTICAL_SACRIFICIAL_FLAP_SUFFIX}': (
          types.PolygonPlacement.from_grid_coords(
              offset=(vert_x, vert_y),
              size=(corner_size, rem_h),
              grid_size=grid_size,
              require_on_paper=False,
          )
      ),
      f'{stick_name}{HORIZONTAL_SACRIFICIAL_FLAP_SUFFIX}': (
          types.PolygonPlacement.from_grid_coords(
              offset=(horiz_x, horiz_y),
              size=(rem_w, corner_size),
              grid_size=grid_size,
              require_on_paper=False,
          )
      ),
      stick_name: types.PolygonPlacement.from_grid_coords(
          offset=(main_x, main_y),
          size=(rem_w, rem_h),
          grid_size=grid_size,
          require_on_paper=False,
      ),
  }


def compute_exact_ls_components(
    stick_name: str,
    placement: types.PolygonPlacement,
    stick: stick_lib.Stick,
    corner: Corner,
    grid_size: int,
) -> dict[str, types.PolygonPlacement]:
  """Decomposes a width-2 flap placement into main + corner + 2 pockets."""
  if round(stick.width) < 2:
    raise ValueError(
        f'Level-shifter repacking requires width == 2, got {stick.width}.'
    )
  corner_size = pocket_depth(round(stick.width))
  dim_w, dim_h = placement.size
  if dim_w <= corner_size or dim_h <= corner_size:
    raise ValueError(
        f'Placement {placement.size} is too small for corner_size={corner_size}'
        f' on stick {stick_name} (width={stick.width}).'
    )
  return _repack_box_components(
      stick_name,
      offset=placement.offset,
      size=placement.size,
      corner_size=corner_size,
      corner=corner,
      grid_size=grid_size,
  )


def identify_sacrificial_flaps(
    tracker: utils_tracker.CreasePatternTracker,
) -> list[types.SacrificialFlap]:
  """Identifies all level-shifter sacrificial flaps in `tracker`."""
  sacrificial_flaps: list[types.SacrificialFlap] = []
  suffixes: tuple[tuple[str, types.Orientation], ...] = (
      (VERTICAL_SACRIFICIAL_FLAP_SUFFIX, 'vertical'),
      (HORIZONTAL_SACRIFICIAL_FLAP_SUFFIX, 'horizontal'),
      (CORNER_SUFFIX, 'corner'),
  )

  for polygon_name, placement in tracker.packed_polygons.items():
    for suffix, orientation in suffixes:
      if polygon_name.endswith(suffix):
        shifted_element_name = polygon_name.removesuffix(suffix)
        main_placement = tracker.packed_polygons.get(shifted_element_name)
        if main_placement is not None:
          sacrificial_flaps.append(
              types.SacrificialFlap.from_flap_placement(
                  polygon_name=polygon_name,
                  placement=placement,
                  shifted_element_name=shifted_element_name,
                  orientation=orientation,
                  main_placement=main_placement,
              )
          )
        break

  river_sf_pairs: dict[str, dict[str, types.PolygonPlacement]] = (
      collections.defaultdict(dict)
  )
  for polygon_name, placement in tracker.packed_polygons.items():
    if placement.river_sf_info is not None:
      river_name = placement.river_sf_info.river_name
      copy_suffix = utils_packing_svg.PolygonIDSuffix.COPY.value
      group_key = (
          river_name + copy_suffix
          if polygon_name.endswith(copy_suffix)
          else river_name
      )
      river_sf_pairs[group_key][polygon_name] = placement

  for group_key, sf_placements in river_sf_pairs.items():
    river_name = group_key.removesuffix(
        utils_packing_svg.PolygonIDSuffix.COPY.value
    )
    if len(sf_placements) != 2:
      raise ValueError(
          f'Expected 2 sacrificial flaps for river {river_name}, got'
          f' {len(sf_placements)}.'
      )
    sf1, sf2 = tuple(sf_placements.values())
    is_horiz = sf1.size[0] + sf2.size[0] > sf1.size[1] + sf2.size[1]
    orientation = 'horizontal' if is_horiz else 'vertical'
    if is_horiz:
      crimp_directions: list[types.SidePosition] = ['bottom', 'top']
      if sf1.offset[1] > sf2.offset[1]:
        crimp_directions.reverse()
    else:
      crimp_directions = ['right', 'left']
      if sf1.offset[0] > sf2.offset[0]:
        crimp_directions.reverse()

    for (polygon_name, placement), crimp_direction in zip(
        sf_placements.items(), crimp_directions, strict=True
    ):
      sacrificial_flaps.append(
          types.SacrificialFlap(
              polygon_name=polygon_name,
              placement=placement,
              shifted_element_name=river_name,
              orientation=orientation,
              shifted_element=tracker.packed_rivers[river_name],
              crimp_direction=crimp_direction,
          )
      )
  return sacrificial_flaps


def _pocket_rectangle(
    name: str,
    rect: Any,
    orientation: types.Orientation,
    grid_size: int,
) -> rectangle_lib.Rectangle:
  """Builds an unclipped pocket `Rectangle` with central-ridge circle centers."""
  if orientation == 'corner':
    r_half = rect.width * 0.5
    centers = [
        point_lib.Point(
            x=rect.x + rect.width * 0.5,
            y=rect.y + rect.height * 0.5,
            require_on_paper=False,
        )
    ]
  elif orientation == 'horizontal':
    cy = rect.y + rect.height * 0.5
    r_half = rect.height * 0.5
    c1_x = 0.0 if rect.x == 0 else rect.x + r_half
    c2_x = (
        float(grid_size)
        if rect.x + rect.width == grid_size
        else rect.x + rect.width - r_half
    )
    if math.isclose(c1_x, c2_x):
      centers = [point_lib.Point(x=c1_x, y=cy, require_on_paper=False)]
    else:
      centers = [
          point_lib.Point(x=c1_x, y=cy, require_on_paper=False),
          point_lib.Point(x=c2_x, y=cy, require_on_paper=False),
      ]
  else:
    cx = rect.x + rect.width * 0.5
    r_half = rect.width * 0.5
    c1_y = 0.0 if rect.y == 0 else rect.y + r_half
    c2_y = (
        float(grid_size)
        if rect.y + rect.height == grid_size
        else rect.y + rect.height - r_half
    )
    if math.isclose(c1_y, c2_y):
      centers = [point_lib.Point(x=cx, y=c1_y, require_on_paper=False)]
    else:
      centers = [
          point_lib.Point(x=cx, y=c1_y, require_on_paper=False),
          point_lib.Point(x=cx, y=c2_y, require_on_paper=False),
      ]
  return rectangle_lib.Rectangle.from_centers(
      circle_centers=centers,
      radius=r_half * math.sqrt(2.0),
      name=name,
  )


def _classify_flap_pocket(
    rect: Any,
    mrect: Any,
) -> tuple[types.Orientation, types.MainFlapPosition] | None:
  """Derives `(orientation, crimp_direction)` from flap-pocket adjacency."""
  sx, sy, sw, sh = rect.x, rect.y, rect.width, rect.height
  mx, my, mw, mh = mrect.x, mrect.y, mrect.width, mrect.height
  dx_ov = min(sx + sw, mx + mw) - max(sx, mx)
  dy_ov = min(sy + sh, my + mh) - max(sy, my)
  if dy_ov > 0 and (sx + sw == mx or mx + mw == sx):
    return ('vertical', 'right' if sx + sw == mx else 'left')
  if dx_ov > 0 and (sy + sh == my or my + mh == sy):
    return ('horizontal', 'bottom' if sy + sh == my else 'top')
  if (
      dx_ov == 0
      and dy_ov == 0
      and (sx + sw == mx or mx + mw == sx)
      and (sy + sh == my or my + mh == sy)
  ):
    if sx + sw == mx and sy + sh == my:
      return ('corner', 'bottom_right')
    if sx == mx + mw and sy + sh == my:
      return ('corner', 'bottom_left')
    if sx + sw == mx and sy == my + mh:
      return ('corner', 'top_right')
    return ('corner', 'top_left')
  return None


def reconstruct_pocket_packing_data(
    *,
    grid_size: int,
    flaps: Mapping[str, Any],
    rivers: Mapping[str, Any],
    reservations: Sequence[Any],
    level_shift_reservations: Sequence[Any],
) -> tuple[list[types.SacrificialFlap], dict[str, rectangle_lib.Rectangle]]:
  """Reconstructs `SacrificialFlap`s and pocket `Rectangle`s from reservations."""
  inv_g = 1.0 / grid_size
  res_by_id = {r.id: r.rect for r in reservations}
  out_sfs: list[types.SacrificialFlap] = []
  rect_by_id: dict[str, rectangle_lib.Rectangle] = {}

  for lsr in level_shift_reservations:
    stick = lsr.stick
    missing = [rid for rid in lsr.region_ids if rid not in res_by_id]
    if missing:
      raise ValueError(
          f'LevelShiftReservation for stick {stick!r} references unknown'
          f' reservation ID(s) {missing}.'
      )
    flap_cands = [
        flaps[k] for k in (stick, f'{stick}{types.COPY_SUFFIX}') if k in flaps
    ]
    if flap_cands:
      for rid in lsr.region_ids:
        rect = res_by_id[rid]
        matched: (
            tuple[Any, types.Orientation, types.MainFlapPosition] | None
        ) = None
        for mf in flap_cands:
          cls_res = _classify_flap_pocket(rect, mf.rect)
          if cls_res is not None:
            matched = (mf, cls_res[0], cls_res[1])
            break
        if matched is None:
          raise ValueError(
              f'Reserved pocket {rid!r} ({rect}) is not adjacent to flap'
              f' {stick!r}.'
          )
        mf, orient, crimp_dir = matched
        if mf.circle_centers:
          mcx = mf.circle_centers[0].x * inv_g
          mcy = mf.circle_centers[0].y * inv_g
        else:
          mcx = (mf.rect.x + mf.rect.width * 0.5) * inv_g
          mcy = (mf.rect.y + mf.rect.height * 0.5) * inv_g
        main_place = types.PolygonPlacement(
            offset=(mf.rect.x, mf.rect.y),
            size=(mf.rect.width, mf.rect.height),
            center=point_lib.Point(x=mcx, y=mcy, require_on_paper=False),
            coords=None,
        )
        sf_place = types.PolygonPlacement(
            offset=(rect.x, rect.y),
            size=(rect.width, rect.height),
            center=point_lib.Point(
                x=(rect.x + rect.width * 0.5) * inv_g,
                y=(rect.y + rect.height * 0.5) * inv_g,
                require_on_paper=False,
            ),
            coords=None,
        )
        out_sfs.append(
            types.SacrificialFlap(
                polygon_name=rid,
                placement=sf_place,
                shifted_element_name=mf.id,
                orientation=orient,
                shifted_element=main_place,
                crimp_direction=crimp_dir,
            )
        )
        rect_by_id[rid] = _pocket_rectangle(rid, rect, orient, grid_size)
      continue

    if stick not in rivers:
      raise ValueError(
          f'Reserved stick {stick!r} not found in flaps or rivers.'
      )
    items = [(rid, res_by_id[rid]) for rid in lsr.region_ids]
    if len(items) == 2:
      pairs = [(items[0], items[1])]
    elif len(items) == 4:
      non_copy = [
          it for it in items if not it[0].endswith(types.COPY_SUFFIX)
      ]
      copy_it = [it for it in items if it[0].endswith(types.COPY_SUFFIX)]
      if len(non_copy) == 2 and len(copy_it) == 2:
        pairs = [(non_copy[0], non_copy[1]), (copy_it[0], copy_it[1])]
      else:
        by_x = sorted(
            items,
            key=lambda it: (
                it[1].x + it[1].width * 0.5,
                it[1].y + it[1].height * 0.5,
            ),
        )
        pairs = [(by_x[0], by_x[1]), (by_x[2], by_x[3])]
    else:
      raise ValueError(
          f'Expected 2 or 4 river pocket reservations for {stick!r}, got'
          f' {len(items)}.'
      )

    rv_main = rivers[stick]
    rv_copy = rivers.get(f'{stick}{types.COPY_SUFFIX}')
    for pair_idx, ((id1, r1), (id2, r2)) in enumerate(pairs):
      dx_ov = min(r1.x + r1.width, r2.x + r2.width) - max(r1.x, r2.x)
      dy_ov = min(r1.y + r1.height, r2.y + r2.height) - max(r1.y, r2.y)
      if dx_ov > 0 and dy_ov <= 0 and r1.y != r2.y:
        orient = 'horizontal'
        crimp_dirs: tuple[types.SidePosition, types.SidePosition] = (
            ('bottom', 'top') if r1.y < r2.y else ('top', 'bottom')
        )
      elif dy_ov > 0 and dx_ov <= 0 and r1.x != r2.x:
        orient = 'vertical'
        crimp_dirs = ('right', 'left') if r1.x < r2.x else ('left', 'right')
      else:
        raise ValueError(
            f'River pockets {id1!r} ({r1}) and {id2!r} ({r2}) do not face'
            f' across river {stick!r}.'
        )
      target_rv = (
          rv_copy if (pair_idx == 1 and rv_copy is not None) else rv_main
      )
      packing_rv = target_rv.to_packing_river(grid_size)
      need_copy_tag = len(pairs) == 2 and pair_idx == 1 and rv_copy is None
      for (rid, rect), crimp_dir in zip(
          ((id1, r1), (id2, r2)), crimp_dirs, strict=True
      ):
        poly_name = (
            f'{rid}{types.COPY_SUFFIX}'
            if need_copy_tag and not rid.endswith(types.COPY_SUFFIX)
            else rid
        )
        r_info = types.RiverSfInfo(
            river_name=stick, river_side='', parent_joint=''
        )
        sf_place = types.PolygonPlacement(
            offset=(rect.x, rect.y),
            size=(rect.width, rect.height),
            center=point_lib.Point(
                x=(rect.x + rect.width * 0.5) * inv_g,
                y=(rect.y + rect.height * 0.5) * inv_g,
                require_on_paper=False,
            ),
            coords=None,
            river_sf_info=r_info,
        )
        out_sfs.append(
            types.SacrificialFlap(
                polygon_name=poly_name,
                placement=sf_place,
                shifted_element_name=stick,
                orientation=orient,
                shifted_element=packing_rv,
                crimp_direction=crimp_dir,
            )
        )
        rect_by_id[rid] = _pocket_rectangle(poly_name, rect, orient, grid_size)

  return out_sfs, rect_by_id


def get_corner_map_for_repacking(
    packing: packing_output.PackingOutput,
) -> dict[str, Corner]:
  """Determines valid repacking corners for all wide flaps in `packing`."""
  tracker = packing.tracker
  stick_figure = packing.stick_figure
  if not tracker or not stick_figure:
    raise ValueError(
        'Packing output must have a tracker and a stick figure to repack level'
        ' shifter flaps.'
    )

  corner_map: dict[str, Corner] = {}
  for stick_name, placement in tracker.packed_polygons.items():
    if (
        stick_name in stick_figure
        and stick_figure[stick_name].requires_level_shifter
    ):
      stick = stick_figure[stick_name]
      c_size = pocket_depth(round(stick.width))
      if placement.size[0] > c_size and placement.size[1] > c_size:
        corner_map[stick_name] = find_valid_repack_corner(
            placement, tracker.grid_size, c_size
        )
  return corner_map


def _repack_tracker(
    tracker: utils_tracker.CreasePatternTracker,
    stick_figure: stick_figure_lib.StickFigure,
    corner_map: Mapping[str, Corner],
) -> utils_tracker.CreasePatternTracker:
  """Repacks level-shifter flaps in `tracker` using `corner_map`."""
  new_tracker = copy.deepcopy(tracker)
  grid_size = new_tracker.grid_size
  polygons_to_repack = []
  for stick_name, placement in tracker.packed_polygons.items():
    if stick_name not in stick_figure:
      continue
    stick = stick_figure[stick_name]
    if stick.requires_level_shifter and stick_name in corner_map:
      polygons_to_repack.append((stick_name, placement, stick))

  for stick_name, placement, stick in polygons_to_repack:
    new_placements = compute_exact_ls_components(
        stick_name,
        placement,
        stick,
        corner_map[stick_name],
        grid_size,
    )
    new_tracker.remove_by_id(stick_name)
    for name, p in new_placements.items():
      new_tracker.add_polygon(name, p)
  return new_tracker


def repack_tracker_polygons(
    packing: packing_output.PackingOutput,
    corner_map: Mapping[str, Corner],
) -> packing_output.PackingOutput:
  """Repacks level-shifter flaps in `packing` into 4 L-shaped components."""
  if not packing.success or not packing.tracker or not packing.stick_figure:
    raise ValueError('Packing output must be successful.')

  new_tracker = _repack_tracker(
      packing.tracker,
      packing.stick_figure,
      corner_map,
  )
  sacrificial_flaps = identify_sacrificial_flaps(new_tracker)
  reason = 'Repacked level shifter flaps.'
  if packing.details and packing.details.get('reason'):
    reason = f'{packing.details["reason"]} with level shifter repacking.'
  details = copy.deepcopy(packing.details) if packing.details else None
  if details is not None:
    details['reason'] = reason

  return packing.model_copy(
      update={
          'tracker': new_tracker,
          'packing_cp': None,
          'hinges': None,
          'flap_rectangles': None,
          'details': details,
          'sacrificial_flaps': sacrificial_flaps,
      }
  )


def update_stick_figure_with_level_shifters(
    stick_figure_data: stick_figure_lib.StickFigure,
    corner_map: Mapping[str, Corner],
) -> stick_figure_lib.StickFigure:
  """Appends synthetic level-shifter corner and sacrificial sticks to tree."""
  new_children = list(stick_figure_data.children)
  joints = stick_figure_data.joints

  for stick_name in corner_map:
    original_stick = stick_figure_data[stick_name]
    is_leaf = len(joints[original_stick.start].sticks) == 1
    parent_joint = original_stick.end if is_leaf else original_stick.start
    target_up = utils_tree.is_stick_up(original_stick, parent_joint)

    for suffix in (
        CORNER_SUFFIX,
        VERTICAL_SACRIFICIAL_FLAP_SUFFIX,
        HORIZONTAL_SACRIFICIAL_FLAP_SUFFIX,
    ):
      new_stick_name = f'{stick_name}{suffix}'
      new_stick = copy.deepcopy(original_stick)
      new_stick.name = new_stick_name
      new_stick.start = parent_joint
      new_stick.end = f'{new_stick_name}_end'
      new_stick.up_at_start_joint = target_up
      new_stick.width = 1
      new_stick.length = 1
      new_children.append(new_stick)

  return dataclasses.replace(stick_figure_data, children=new_children)


def add_river_level_shifters(
    tree: stick_figure_lib.StickFigure,
    tracker: utils_tracker.CreasePatternTracker,
) -> stick_figure_lib.StickFigure:
  """Appends sacrificial flap sticks for rivers using `RiverSfInfo` metadata."""
  existing_names = {s.name for s in tree.children}
  new_children = list(tree.children)
  for poly_name, placement in tracker.packed_polygons.items():
    if placement.river_sf_info is None:
      continue
    if poly_name in existing_names:
      continue
    new_stick = stick_lib.Stick(
        name=poly_name,
        start=placement.river_sf_info.parent_joint,
        end=f'{poly_name}_end',
        length=1,
        width=1,
        azimuth_angle=0,
        elevation_angle=0,
    )
    new_stick.is_flap = True
    new_children.append(new_stick)
    existing_names.add(poly_name)
  return dataclasses.replace(tree, children=new_children)


def update_packing_cp_and_hinges(
    repacked_output: packing_output.PackingOutput,
) -> packing_output.PackingOutput:
  """Regenerates `packing_cp`, `hinges`, and `flap_rectangles` from tracker."""
  if not repacked_output.tracker:
    return repacked_output

  grid_size = repacked_output.tracker.grid_size
  cfg = repacked_output.config
  y_ridge_at_border = cfg.y_ridge_at_border if cfg is not None else True
  recovery = cfg.recovery if cfg is not None else False
  try:
    repacked_cp, flap_rectangles = (
        utils_packing_svg.build_packing_crease_pattern(
            grid_size,
            repacked_output.tracker,
            y_ridge_at_border=y_ridge_at_border,
            recovery=recovery,
        )
    )
  except svg_parser.SVGParseError as err:
    logging.exception('Failed to parse repacked SVG on grid %d', grid_size)
    details = copy.deepcopy(repacked_output.details) or {}
    details['reason'] = 'SVG parsing failed after repacking'
    return packing_output.PackingOutput.from_error(
        error_message=(
            f'Successfully repacked, but failed to parse generated SVG: {err}'
        ),
        details=details,  # pyrefly: ignore[bad-argument-type]
        tracker=repacked_output.tracker,
        config=repacked_output.config,
    )

  hinges_info = utils_packing_svg.extract_hinges_info(
      repacked_output.tracker, grid_size
  )
  return repacked_output.model_copy(
      update={
          'packing_cp': repacked_cp,
          'hinges': hinges_info,
          'flap_rectangles': flap_rectangles,
      }
  )


def scan_segment_for_aligned_pairs(
    occupancy_grid: np.ndarray,
    grid_size: int,
    flap_size: tuple[int, int],
    bank1_offset_fn: Callable[[int], tuple[float, float]],
    bank2_offset_fn: Callable[[int], tuple[float, float]],
    dim_min: int,
    dim_max: int,
    alignment: int,
    is_symmetric: bool,
) -> Iterator[tuple[types.PolygonPlacement, types.PolygonPlacement]]:
  """Scans two banks of a river segment for aligned sacrificial flap pairs."""
  sac_len = max(flap_size)

  def _try_placement(
      offset: tuple[float, float], name: str
  ) -> types.PolygonPlacement | None:
    rect = rectangle_lib.Rectangle(
        name=name,
        x_start=offset[0],
        y_start=offset[1],
        x_len=flap_size[0],
        y_len=flap_size[1],
    )
    if not utils_packing_flaps.is_unoccupied(occupancy_grid, rect):
      return None
    cand = types.PolygonPlacement.from_grid_coords(
        offset=(int(offset[0]), int(offset[1])),
        size=flap_size,
        grid_size=grid_size,
        require_on_paper=False,
    )
    if is_symmetric and cand.offset[0] + cand.size[0] > grid_size / 2:
      return None
    return cand

  for i in range(dim_min, dim_max + 1):
    p1 = _try_placement(bank1_offset_fn(i), '_sf1')
    if p1 is None:
      continue
    for j in range(i - sac_len + alignment, i + sac_len - alignment + 1):
      if dim_min <= j <= dim_max:
        p2 = _try_placement(bank2_offset_fn(j), '_sf2')
        if p2 is not None:
          yield p1, p2


def get_segment_scan_params(
    is_horizontal: bool,
    bank1_pos: float,
    bank2_pos: float,
    dim_min: int,
    dim_max: int,
    sac_len: int = SACRIFICIAL_FLAP_LENGTH,
    sac_wid: int = SACRIFICIAL_FLAP_WIDTH,
) -> tuple[
    tuple[int, int],
    Callable[[int], tuple[float, float]],
    Callable[[int], tuple[float, float]],
    int,
    int,
]:
  """Builds scan parameters for a straight river segment."""
  if is_horizontal:
    flap_size = (sac_len, sac_wid)
    bank1_fn = lambda idx: (float(idx), bank1_pos)
    bank2_fn = lambda idx: (float(idx), bank2_pos)
  else:
    flap_size = (sac_wid, sac_len)
    bank1_fn = lambda idx: (bank1_pos, float(idx))
    bank2_fn = lambda idx: (bank2_pos, float(idx))
  return (flap_size, bank1_fn, bank2_fn, dim_min, dim_max)


def extract_straight_segments(
    river: river_lib.HuggingRiver,
) -> list[tuple[bool, float, float, int, int]]:
  """Extracts maximal straight horizontal/vertical runs from a HuggingRiver."""
  path = river.subriver_paths[0]
  flap_length = SACRIFICIAL_FLAP_LENGTH
  flap_width = SACRIFICIAL_FLAP_WIDTH
  river_width = river.width
  segments = []

  index = 0
  while index < len(path) - 1:
    row_current, col_current = path[index]
    row_next, col_next = path[index + 1]
    delta_row = row_next - row_current
    delta_col = col_next - col_current
    if abs(delta_row) + abs(delta_col) != 1:
      index += 1
      continue

    is_horizontal = delta_row == 0
    run_start = index
    while index < len(path) - 1:
      row_ahead, col_ahead = path[index + 1]
      row_here, col_here = path[index]
      if (row_ahead - row_here, col_ahead - col_here) != (delta_row, delta_col):
        break
      index += 1
    run_end = index
    if run_end - run_start + 1 < flap_length:
      index += 1
      continue

    if is_horizontal:
      run_col_min = min(path[run_start][1], path[run_end][1])
      run_col_max = max(path[run_start][1], path[run_end][1])
      bank1_pos = float(row_current - flap_width)
      bank2_pos = float(row_current + river_width)
      dim_min = run_col_min
      dim_max = run_col_max - flap_length + 1
    else:
      run_row_min = min(path[run_start][0], path[run_end][0])
      run_row_max = max(path[run_start][0], path[run_end][0])
      bank1_pos = float(col_current - flap_width)
      bank2_pos = float(col_current + river_width)
      dim_min = run_row_min
      dim_max = run_row_max - flap_length + 1

    if dim_max >= dim_min:
      segments.append((is_horizontal, bank1_pos, bank2_pos, dim_min, dim_max))
    index += 1
  return segments


def border_cell_from_index(idx: int, grid_size: int) -> tuple[int, int]:
  """Converts a counter-clockwise border cell index to `(row, col)`."""
  n = grid_size - 1
  if idx < n:
    return (idx, 0)
  if idx < 2 * n:
    return (n, idx - n)
  if idx < 3 * n:
    return (3 * n - idx, n)
  return (0, 4 * n - idx)


def get_sf_candidate_for_border_cell(
    row: int,
    col: int,
    grid_size: int,
    sac_len: int,
    sac_wid: int,
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]] | None:
  """Returns `(offset, size, next_border_cell)` for a border sacrificial flap."""
  if col == 0:
    offset = (col, row)
    size = (sac_len, sac_wid)
    new_start = (row + sac_wid, col)
  elif col == grid_size - 1:
    offset = (col - sac_len + 1, row)
    size = (sac_len, sac_wid)
    new_start = (row + sac_wid, col)
  elif row == 0:
    offset = (col, row)
    size = (sac_wid, sac_len)
    new_start = (row, col + sac_wid)
  elif row == grid_size - 1:
    offset = (col, row - sac_len + 1)
    size = (sac_wid, sac_len)
    new_start = (row, col + sac_wid)
  else:
    return None

  if offset[0] < 0 or offset[1] < 0:
    return None
  if offset[0] + size[0] > grid_size or offset[1] + size[1] > grid_size:
    return None
  if new_start[0] >= grid_size or new_start[1] >= grid_size:
    return None
  return offset, size, new_start


def place_hugging_side_sacrificial_flap(
    start_cell: tuple[int, int],
    grid_size: int,
    occupancy_grid: np.ndarray,
) -> tuple[types.PolygonPlacement, tuple[int, int]] | None:
  """Places a `2x4` sacrificial flap near a hugging river's border start cell."""
  num_border_cells = 4 * grid_size - 4
  if num_border_cells <= 0:
    return None

  start_idx = river_lib.get_border_cell_counter_clockwise_index(
      start_cell, grid_size
  )
  for step in range(num_border_cells):
    idx = (start_idx + step) % num_border_cells
    row, col = border_cell_from_index(idx, grid_size)
    candidate = get_sf_candidate_for_border_cell(
        row, col, grid_size, SACRIFICIAL_FLAP_LENGTH, SACRIFICIAL_FLAP_WIDTH
    )
    if candidate is None:
      continue
    offset, size, new_start = candidate
    rect = rectangle_lib.Rectangle(
        name='_hugging_sf',
        x_start=offset[0],
        y_start=offset[1],
        x_len=size[0],
        y_len=size[1],
    )
    if not utils_packing_flaps.is_unoccupied(occupancy_grid, rect):
      continue
    try:
      sf_placement = types.PolygonPlacement.from_grid_coords(
          offset=(int(offset[0]), int(offset[1])),
          size=size,
          grid_size=grid_size,
          require_on_paper=False,
      )
    except ValueError:
      continue
    return sf_placement, new_start
  return None


def find_opposite_sacrificial_slot(
    hugging_sf: types.PolygonPlacement,
    river: river_lib.HuggingRiver,
    occupancy_grid: np.ndarray,
    alignment: int = 4,
    is_symmetric: bool = False,
) -> Iterator[types.PolygonPlacement]:
  """Yields valid opposite-bank sacrificial flap slots aligned to `hugging_sf`."""
  grid_size = river.grid_size
  sac_len = SACRIFICIAL_FLAP_LENGTH
  sac_wid = SACRIFICIAL_FLAP_WIDTH
  hx, hy = hugging_sf.offset
  hw, hh = hugging_sf.size
  is_h = hw > hh

  for (
      seg_is_h,
      bank1_pos,
      bank2_pos,
      dim_min,
      dim_max,
  ) in extract_straight_segments(river):
    if seg_is_h != is_h:
      continue
    hugging_dim_pos = hx if is_h else hy
    hugging_bank_pos = hy if is_h else hx
    if abs(hugging_bank_pos - bank1_pos) < abs(hugging_bank_pos - bank2_pos):
      opp_bank_pos = bank2_pos
    else:
      opp_bank_pos = bank1_pos

    flap_size = (sac_len, sac_wid) if is_h else (sac_wid, sac_len)
    for j in range(
        hugging_dim_pos - sac_len + alignment,
        hugging_dim_pos + sac_len - alignment + 1,
    ):
      if not dim_min <= j <= dim_max:
        continue
      offset = (float(j), opp_bank_pos) if is_h else (opp_bank_pos, float(j))
      if (
          offset[0] < 0
          or offset[1] < 0
          or offset[0] + flap_size[0] > grid_size
          or offset[1] + flap_size[1] > grid_size
      ):
        continue
      rect = rectangle_lib.Rectangle(
          name='_opp_sf',
          x_start=offset[0],
          y_start=offset[1],
          x_len=flap_size[0],
          y_len=flap_size[1],
      )
      if not utils_packing_flaps.is_unoccupied(occupancy_grid, rect):
        continue
      try:
        cand = types.PolygonPlacement.from_grid_coords(
            offset=(int(offset[0]), int(offset[1])),
            size=flap_size,
            grid_size=grid_size,
            require_on_paper=False,
        )
      except ValueError:
        continue
      if is_symmetric and cand.offset[0] + cand.size[0] > grid_size / 2:
        continue
      yield cand


def find_aligned_sacrificial_slots(
    river: river_lib.PackingRiver,
    occupancy_grid: np.ndarray,
    alignment: int = 4,
    is_symmetric: bool = False,
) -> Iterator[tuple[types.PolygonPlacement, types.PolygonPlacement]]:
  """Yields aligned `2x4` sacrificial flap pairs straddling `river`."""
  grid_size = river.grid_size
  sac_len = SACRIFICIAL_FLAP_LENGTH
  sac_wid = SACRIFICIAL_FLAP_WIDTH

  if isinstance(river, river_lib.LShapedRiver):
    found_pairs: set[tuple[tuple[int, int], ...]] = set()
    for rect in river.rectangles:
      is_h = rect.y_len == river.width and rect.x_len >= sac_len
      is_v = rect.x_len == river.width and rect.y_len >= sac_len
      if is_h:
        params = get_segment_scan_params(
            True,
            rect.y_start - sac_wid,
            rect.y_start + river.width,
            int(rect.x_start),
            int(rect.x_start + rect.x_len - sac_len),
        )
      elif is_v:
        params = get_segment_scan_params(
            False,
            rect.x_start - sac_wid,
            rect.x_start + river.width,
            int(rect.y_start),
            int(rect.y_start + rect.y_len - sac_len),
        )
      else:
        continue
      for p1, p2 in scan_segment_for_aligned_pairs(
          occupancy_grid, grid_size, *params, alignment, is_symmetric
      ):
        pair = tuple(sorted((p1.offset, p2.offset)))
        if pair not in found_pairs:
          yield p1, p2
          found_pairs.add(pair)
    return

  if isinstance(river, river_lib.River):
    is_h = river.orientation == river_lib.RiverOrientation.HORIZONTAL
    params = get_segment_scan_params(
        is_h,
        river.offset - sac_wid,
        river.offset + river.width,
        0,
        grid_size - sac_len,
    )
    yield from scan_segment_for_aligned_pairs(
        occupancy_grid, grid_size, *params, alignment, is_symmetric
    )
    return

  if isinstance(river, river_lib.HuggingRiver):
    found_pairs = set()
    for (
        is_h,
        bank1_pos,
        bank2_pos,
        dim_min,
        dim_max,
    ) in extract_straight_segments(river):
      params = get_segment_scan_params(
          is_h, bank1_pos, bank2_pos, dim_min, dim_max
      )
      for p1, p2 in scan_segment_for_aligned_pairs(
          occupancy_grid, grid_size, *params, alignment, is_symmetric
      ):
        pair = tuple(sorted((p1.offset, p2.offset)))
        if pair not in found_pairs:
          yield p1, p2
          found_pairs.add(pair)
    return

  raise NotImplementedError(f'River type {type(river)} not supported.')


def _clipped_span(offset: int, length: int, grid_size: int) -> tuple[int, int]:
  return max(0, int(offset)), min(int(grid_size), int(offset) + int(length))


def has_consistent_sacrificial_parity(
    output: packing_output.PackingOutput,
) -> bool:
  """Returns whether paired border-touching sacrificial pockets match parity."""
  if not output.sacrificial_flaps or output.packing_cp is None:
    return True
  grid_size = output.packing_cp.grid_size
  by_group: dict[str, list[types.SacrificialFlap]] = collections.defaultdict(
      list
  )
  for sf in output.sacrificial_flaps:
    if sf.is_corner:
      continue
    group_key = sf.shifted_element_name
    if sf.polygon_name.endswith(types.COPY_SUFFIX):
      group_key += types.COPY_SUFFIX
    by_group[group_key].append(sf)

  for s_flaps in by_group.values():
    if len(s_flaps) != 2:
      continue
    sf_a, sf_b = s_flaps[0], s_flaps[1]
    dirs = {sf_a.crimp_direction, sf_b.crimp_direction}
    if dirs <= {'top', 'bottom'}:
      ax0, ax1 = _clipped_span(
          sf_a.placement.offset[0], sf_a.placement.size[0], grid_size
      )
      bx0, bx1 = _clipped_span(
          sf_b.placement.offset[0], sf_b.placement.size[0], grid_size
      )
      touches = (ax0 == 0 or ax1 == grid_size) and (
          bx0 == 0 or bx1 == grid_size
      )
      if touches and (ax1 - ax0) % 2 != (bx1 - bx0) % 2:
        return False
    elif dirs <= {'left', 'right'}:
      ay0, ay1 = _clipped_span(
          sf_a.placement.offset[1], sf_a.placement.size[1], grid_size
      )
      by0, by1 = _clipped_span(
          sf_b.placement.offset[1], sf_b.placement.size[1], grid_size
      )
      touches = (ay0 == 0 or ay1 == grid_size) and (
          by0 == 0 or by1 == grid_size
      )
      if touches and (ay1 - ay0) % 2 != (by1 - by0) % 2:
        return False
  return True


def _rebalance_sacrificial_flap_parities(
    output: packing_output.PackingOutput,
) -> packing_output.PackingOutput:
  """Rebalances mirrored river sacrificial pockets with mismatched parity."""
  if (
      has_consistent_sacrificial_parity(output)
      or output.tracker is None
      or output.packing_cp is None
  ):
    return output
  tracker = output.tracker
  grid_size = tracker.grid_size
  half_g = grid_size // 2
  by_elem: dict[str, list[types.SacrificialFlap]] = collections.defaultdict(
      list
  )
  for sf in output.sacrificial_flaps or ():
    if not sf.is_corner and not sf.polygon_name.endswith(types.COPY_SUFFIX):
      by_elem[sf.shifted_element_name].append(sf)

  new_tracker = copy.deepcopy(tracker)
  changed = False
  for s_flaps in by_elem.values():
    if len(s_flaps) != 2:
      continue
    sf_a, sf_b = s_flaps[0], s_flaps[1]
    if not {sf_a.crimp_direction, sf_b.crimp_direction} <= {'top', 'bottom'}:
      continue
    ax0, ax1 = _clipped_span(
        sf_a.placement.offset[0], sf_a.placement.size[0], grid_size
    )
    bx0, bx1 = _clipped_span(
        sf_b.placement.offset[0], sf_b.placement.size[0], grid_size
    )
    aw, bw = ax1 - ax0, bx1 - bx0
    if aw % 2 == bw % 2:
      continue
    sf_short = sf_a if aw < bw else sf_b
    sx0, sx1 = _clipped_span(
        sf_short.placement.offset[0], sf_short.placement.size[0], grid_size
    )
    sy0, sy1 = _clipped_span(
        sf_short.placement.offset[1], sf_short.placement.size[1], grid_size
    )
    sh_h = sy1 - sy0
    copy_name = sf_short.polygon_name + types.COPY_SUFFIX
    copy_place = new_tracker.packed_polygons.get(copy_name)
    if (
        sx0 != 0
        or sx1 >= half_g
        or sh_h <= 0
        or sh_h % 2 != 0
        or copy_place is None
    ):
      continue
    cx0, cx1 = _clipped_span(
        copy_place.offset[0], copy_place.size[0], grid_size
    )
    cy0, cy1 = _clipped_span(
        copy_place.offset[1], copy_place.size[1], grid_size
    )
    gap_w = grid_size - 2 * sx1
    if (cx0, cx1) != (grid_size - sx1, grid_size) or (cy0, cy1) != (sy0, sy1):
      continue
    blocker_name: str | None = None
    new_blocker_place: types.PolygonPlacement | None = None
    for p_name, p_place in new_tracker.packed_polygons.items():
      if p_name in (sf_short.polygon_name, copy_name):
        continue
      if p_place.offset[0] != sx1 or p_place.size[0] != gap_w:
        continue
      py0 = int(p_place.offset[1])
      py1 = int(p_place.offset[1] + p_place.size[1])
      min_h = max(2, int(p_place.size[0]))
      if output.stick_figure is not None and p_name in output.stick_figure:
        min_h = max(min_h, 2 * int(output.stick_figure[p_name].length))
      if (
          py0 < sy0
          and py1 == sy1
          and (sy0 - py0) >= min_h
          and (sy0 - py0) % 2 == 0
          and (sy0 + py0) * 0.5 >= 0.0
      ):
        blocker_name = p_name
        new_blocker_place = types.PolygonPlacement.from_grid_coords(
            offset=(p_place.offset[0], py0),
            size=(p_place.size[0], sy0 - py0),
            grid_size=grid_size,
            require_on_paper=False,
            river_sf_info=p_place.river_sf_info,
        )
        break
      if (
          py0 == sy0
          and py1 > sy1
          and (py1 - sy1) >= min_h
          and (py1 - sy1) % 2 == 0
          and (py1 + sy1) * 0.5 <= grid_size
      ):
        blocker_name = p_name
        new_blocker_place = types.PolygonPlacement.from_grid_coords(
            offset=(p_place.offset[0], sy1),
            size=(p_place.size[0], py1 - sy1),
            grid_size=grid_size,
            require_on_paper=False,
            river_sf_info=p_place.river_sf_info,
        )
        break
    if blocker_name is None or new_blocker_place is None:
      continue
    even_w = half_g if half_g % 2 == 0 else half_g + 1
    short_off_x = 0 if half_g % 2 == 0 else -1
    new_short = types.PolygonPlacement.from_grid_coords(
        offset=(short_off_x, sy0),
        size=(even_w, sh_h),
        grid_size=grid_size,
        require_on_paper=False,
        river_sf_info=sf_short.placement.river_sf_info,
    )
    new_copy = types.PolygonPlacement.from_grid_coords(
        offset=(half_g, sy0),
        size=(even_w, sh_h),
        grid_size=grid_size,
        require_on_paper=False,
        river_sf_info=copy_place.river_sf_info,
    )
    new_tracker.remove_by_id(blocker_name)
    new_tracker.remove_by_id(sf_short.polygon_name)
    new_tracker.remove_by_id(copy_name)
    new_tracker.add_polygon(blocker_name, new_blocker_place)
    new_tracker.add_polygon(sf_short.polygon_name, new_short)
    new_tracker.add_polygon(copy_name, new_copy)
    changed = True

  if not changed:
    return output
  try:
    new_sfs = identify_sacrificial_flaps(new_tracker)
    repacked = output.model_copy(
        update={
            'tracker': new_tracker,
            'sacrificial_flaps': new_sfs,
            'packing_cp': None,
            'hinges': None,
            'flap_rectangles': None,
        }
    )
    updated = update_packing_cp_and_hinges(repacked)
    if updated.packing_cp is not None and has_consistent_sacrificial_parity(
        updated
    ):
      return updated
  except (ValueError, KeyError, RuntimeError):
    return output
  return output


def _mirror_corner_x(corner: Corner) -> Corner:
  match corner:
    case Corner.TOP_LEFT:
      return Corner.TOP_RIGHT
    case Corner.TOP_RIGHT:
      return Corner.TOP_LEFT
    case Corner.BOTTOM_LEFT:
      return Corner.BOTTOM_RIGHT
    case Corner.BOTTOM_RIGHT:
      return Corner.BOTTOM_LEFT


def _symmetrize_mirrored_flap_pockets(
    output: packing_output.PackingOutput,
) -> packing_output.PackingOutput:
  """Reflects right-half L-shaped flap corner pockets across `x = G / 2`."""
  if (
      not output.sacrificial_flaps
      or output.tracker is None
      or output.packing_cp is None
  ):
    return output
  tracker = output.tracker
  grid_size = tracker.grid_size
  if grid_size <= 0 or grid_size % 2 != 0:
    return output

  repacked_info: dict[str, tuple[int, int, int, int, int, Corner]] = {}
  for poly_name, p_corner in tracker.packed_polygons.items():
    if not poly_name.endswith(CORNER_SUFFIX):
      continue
    s_name = poly_name.removesuffix(CORNER_SUFFIX)
    v_name = f'{s_name}{VERTICAL_SACRIFICIAL_FLAP_SUFFIX}'
    h_name = f'{s_name}{HORIZONTAL_SACRIFICIAL_FLAP_SUFFIX}'
    p_main = tracker.packed_polygons.get(s_name)
    p_vert = tracker.packed_polygons.get(v_name)
    p_horiz = tracker.packed_polygons.get(h_name)
    if p_main is None or p_vert is None or p_horiz is None:
      continue
    parts = (p_main, p_corner, p_vert, p_horiz)
    ux0 = min(p.offset[0] for p in parts)
    uy0 = min(p.offset[1] for p in parts)
    ux1 = max(p.offset[0] + p.size[0] for p in parts)
    uy1 = max(p.offset[1] + p.size[1] for p in parts)
    c_size = int(p_corner.size[0])
    is_left = p_corner.offset[0] == ux0
    is_top = p_corner.offset[1] == uy0
    if is_left and is_top:
      cur_corner = Corner.TOP_LEFT
    elif not is_left and is_top:
      cur_corner = Corner.TOP_RIGHT
    elif is_left:
      cur_corner = Corner.BOTTOM_LEFT
    else:
      cur_corner = Corner.BOTTOM_RIGHT
    repacked_info[s_name] = (ux0, uy0, ux1, uy1, c_size, cur_corner)

  if len(repacked_info) < 2:
    return output

  new_tracker = copy.deepcopy(tracker)
  changed = False
  half_g = 0.5 * grid_size
  names = sorted(repacked_info)
  for s_left in names:
    lx0, ly0, lx1, ly1, lc_size, l_corner = repacked_info[s_left]
    if 0.5 * (lx0 + lx1) >= half_g:
      continue
    for s_right in names:
      if s_right == s_left:
        continue
      rx0, ry0, rx1, ry1, rc_size, r_corner = repacked_info[s_right]
      if (
          lx0 + rx1 != grid_size
          or lx1 + rx0 != grid_size
          or ly0 != ry0
          or ly1 != ry1
          or lc_size != rc_size
      ):
        continue
      want_r_corner = _mirror_corner_x(l_corner)
      if r_corner == want_r_corner:
        break
      union_r = types.PolygonPlacement.from_grid_coords(
          offset=(rx0, ry0),
          size=(rx1 - rx0, ry1 - ry0),
          grid_size=grid_size,
          require_on_paper=False,
      )
      mx, my = want_r_corner.get_coordinates(union_r, corner_size=rc_size)
      if not flap_placement.is_valid_level_shifter_corner(
          off_x=mx, off_y=my, grid_size=grid_size, size=rc_size
      ):
        break
      new_parts = _repack_box_components(
          s_right,
          offset=(rx0, ry0),
          size=(rx1 - rx0, ry1 - ry0),
          corner_size=rc_size,
          corner=want_r_corner,
          grid_size=grid_size,
      )
      for comp_name in new_parts:
        new_tracker.remove_by_id(comp_name)
      for comp_name, comp_place in new_parts.items():
        new_tracker.add_polygon(comp_name, comp_place)
      changed = True
      break

  if not changed:
    return output
  try:
    new_sfs = identify_sacrificial_flaps(new_tracker)
    repacked = output.model_copy(
        update={
            'tracker': new_tracker,
            'sacrificial_flaps': new_sfs,
            'packing_cp': None,
            'hinges': None,
            'flap_rectangles': None,
        }
    )
    updated = update_packing_cp_and_hinges(repacked)
    if updated.packing_cp is not None:
      return updated
  except (ValueError, KeyError, RuntimeError):
    return output
  return output


def _has_river_sf_placements(
    tracker: utils_tracker.CreasePatternTracker,
) -> bool:
  return any(
      p.river_sf_info is not None for p in tracker.packed_polygons.values()
  )


def apply_level_shifters(
    packing_output_obj: packing_output.PackingOutput,
) -> packing_output.PackingOutput:
  """Repacks width-2 flaps and river sacrificial slots on `packing_output_obj`."""
  if not packing_output_obj.success or not packing_output_obj.config:
    return packing_output_obj
  if not packing_output_obj.config.use_level_shifter:
    raise ValueError(
        'apply_level_shifters should not be called when use_level_shifter is'
        ' False.'
    )

  corner_map = get_corner_map_for_repacking(packing_output_obj)
  if not corner_map:
    tracker = packing_output_obj.tracker
    if tracker and _has_river_sf_placements(tracker):
      sfs = identify_sacrificial_flaps(tracker)
      new_stick_figure = packing_output_obj.stick_figure
      if new_stick_figure:
        new_stick_figure = add_river_level_shifters(new_stick_figure, tracker)
      out = packing_output_obj.model_copy(
          update={
              'sacrificial_flaps': sfs,
              'stick_figure': new_stick_figure,
          }
      )
      return _rebalance_sacrificial_flap_parities(out)
    return packing_output_obj

  repacked_output = repack_tracker_polygons(packing_output_obj, corner_map)
  if not repacked_output.stick_figure or not repacked_output.tracker:
    return repacked_output

  new_stick_figure = update_stick_figure_with_level_shifters(
      repacked_output.stick_figure, corner_map
  )
  new_stick_figure = add_river_level_shifters(
      new_stick_figure, repacked_output.tracker
  )
  repacked_output = repacked_output.model_copy(
      update={'stick_figure': new_stick_figure}
  )
  updated = update_packing_cp_and_hinges(repacked_output)
  return _rebalance_sacrificial_flap_parities(
      _symmetrize_mirrored_flap_pockets(updated)
  )
