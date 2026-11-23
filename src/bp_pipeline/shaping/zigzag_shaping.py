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
"""Zigzag and inscribed-polygon shaping primitives for flaps and rivers."""

from collections.abc import Sequence
import itertools
import math

from absl import logging

from .. import timeout as timeout_lib
from ..crease_pattern import crease as crease_utils
from ..crease_pattern import crease_pattern as cp_utils
from ..crease_pattern import point as point_utils
from . import narrow_flap_shaping
from . import shaping_elements
from . import shaping_utils
from . import types as shaping_types
from . import vector

type FlapLike = shaping_utils.Rectangle | shaping_elements.ShapingFlap
type AnchorPoint = tuple[point_utils.Point, int, int]

DEFAULT_TOLERANCE = 1e-5
_CENTRAL_CREASE_X = 0.5


# ---------------------------------------------------------------------------
# Zigzag shaping
# ---------------------------------------------------------------------------


def _resolve_zigzag_depth(
    target_flap_angle: float | None,
    target_zigzag_depth: float | None,
) -> float:
  """Returns the zigzag depth in grid units from angle or explicit depth."""
  if target_zigzag_depth is not None:
    return target_zigzag_depth
  if target_flap_angle is None:
    raise ValueError(
        'Either target_flap_angle or target_zigzag_depth must be provided.'
    )
  return shaping_utils.zigzag_depth_from_fold_angle(target_flap_angle)


def get_closest_ridge_intersection(
    start: point_utils.Point,
    end: point_utils.Point,
    ridges: Sequence[crease_utils.Crease],
    require_on_paper: bool = True,
    angle: float | None = None,
) -> tuple[point_utils.Point | None, int | None]:
  """Returns the nearest valid intersection `(point, index)` with `ridges`."""
  seg = crease_utils.Crease(start, end, fold=crease_utils.Fold.MOUNTAIN)
  seg_slope = abs(
      vector.Vector(x=seg.normed_vector[0], y=seg.normed_vector[1]).slope
  )
  is_diagonal = abs(seg_slope - 1.0) < shaping_utils.EPS

  best_point: point_utils.Point | None = None
  best_idx: int | None = None
  best_dist = math.inf

  for idx, ridge in enumerate(ridges):
    hit = ridge.vertex_with_crease(seg, include_external=not require_on_paper)
    if hit is None or start.isclose(hit):
      continue
    if (hit.isclose(ridge.p1) or hit.isclose(ridge.p2)) and (
        angle is None or angle < math.pi / 2
    ):
      continue
    if hit.isclose(end) and not is_diagonal:
      continue
    dist = point_utils.distance(start, hit)
    if dist < best_dist:
      best_dist = dist
      best_point = hit
      best_idx = idx
  return best_point, best_idx


def draw_zigzag_from_anchor(
    start: point_utils.Point,
    on_edge: bool,
    edge_index: int,
    edge_vectors: Sequence[vector.Vector],
    depth_vectors: Sequence[vector.Vector],
    zigzag_depth: float,
) -> point_utils.Point:
  """Steps one zigzag segment away from (`on_edge`) or back to the boundary."""
  sign = 1.0 if on_edge else -1.0
  step = (
      edge_vectors[edge_index] + sign * zigzag_depth * depth_vectors[edge_index]
  )
  return step.point_add(start)


def _snap_point(point: point_utils.Point, grid_size: int) -> point_utils.Point:
  """Snaps each coordinate of `point` independently when close to a gridline."""
  for snap_x, snap_y in ((True, False), (False, True)):
    try:
      point = shaping_utils.snap_point_to_grid(
          point, grid_size, snap_x=snap_x, snap_y=snap_y
      )
    except ValueError:
      pass
  return point


def shape_boundary_with_zigzags(
    anchor_points: Sequence[AnchorPoint],
    boundary: shaping_elements.ShapingBoundary,
    grid_size: int,
    target_flap_angle: float | None,
    target_zigzag_depth: float | None,
    fold_order: Sequence[crease_utils.Fold],
    even_phase: bool,
    fold_percent: float,
    ridges: Sequence[crease_utils.Crease],
    snap_to_grid: bool,
    require_on_paper: bool = True,
    fold_step: int = 0,
) -> list[crease_utils.Crease]:
  """Traces ridge-reflected zigzag creases around `boundary` from anchors."""
  if not fold_order:
    raise ValueError('fold_order must be non-empty.')
  depth = _resolve_zigzag_depth(target_flap_angle, target_zigzag_depth)

  remaining = list(anchor_points)
  visited: list[point_utils.Point] = []
  zigzags: list[crease_utils.Crease] = []
  num_edges = len(boundary.creases)

  while remaining:
    timeout_lib.maybe_timeout_cooperatively()
    anchor, edge_index, grid_index = remaining[0]
    start = anchor
    fold_index = grid_index % len(fold_order)
    on_edge = True
    cur_ridge: crease_utils.Crease | None = None
    prev_zigzag: crease_utils.Crease | None = None

    while prev_zigzag is None or (
        not any(start.isclose(v) for v in visited)
        and not (start.is_border and require_on_paper)
    ):
      timeout_lib.maybe_timeout_cooperatively()
      if cur_ridge is not None:
        reflected = shaping_utils.reflect_crease_over_ridge(
            prev_zigzag, cur_ridge, require_on_paper=require_on_paper
        )
        try:
          end = reflected.point_add(start)
        except ValueError:
          break
      else:
        end = draw_zigzag_from_anchor(
            start,
            on_edge,
            edge_index,
            boundary.edge_unit_vectors,
            boundary.depth_unit_vectors,
            depth,
        )

      hit, ridge_idx = get_closest_ridge_intersection(
          start,
          end,
          ridges,
          require_on_paper=require_on_paper,
          angle=target_flap_angle,
      )
      if hit is not None and ridge_idx is not None:
        end = hit
        cur_ridge = ridges[ridge_idx]
      else:
        cur_ridge = None

      if shaping_utils.get_crease_ridges_containment(start, end, ridges):
        edge_index = (edge_index + 1) % num_edges
      else:
        if snap_to_grid:
          end = _snap_point(end, grid_size)
        zigzag = crease_utils.Crease(
            start,
            end,
            fold=fold_order[fold_index],
            fold_percent=fold_percent,
            fold_step=fold_step,
        )
        if prev_zigzag is not None and prev_zigzag.isclose(
            zigzag, atol=DEFAULT_TOLERANCE
        ):
          raise ValueError(
              f'Backtracking detected while tracing zigzag at {start}.'
          )
        zigzags.append(zigzag)
        if on_edge:
          visited.append(start)
          remaining = [item for item in remaining if not item[0].isclose(start)]
        prev_zigzag = zigzag
        start = end
        on_edge = not on_edge
        if on_edge:
          if cur_ridge is None:
            next_edge = boundary.get_edge_index(end)
            if next_edge is None:
              raise ValueError(f'Point {end} is not on boundary {boundary}.')
            edge_index = next_edge
          else:
            edge_index = (edge_index + 1) % num_edges

      fold_index = (fold_index + 1) % len(fold_order)

    if not boundary.closed and not even_phase:
      reversed_edges = [-1 * vec for vec in boundary.edge_unit_vectors]
      end = draw_zigzag_from_anchor(
          anchor, True, 0, reversed_edges, boundary.depth_unit_vectors, depth
      )
      zigzags.append(
          crease_utils.Crease(
              anchor,
              end,
              fold=fold_order[-1],
              fold_percent=fold_percent,
              fold_step=fold_step,
          )
      )

  return zigzags


def compute_flap_anchor_points(
    rect: FlapLike,
    grid_size: int,
    even_phase: bool,
) -> list[AnchorPoint]:
  """Computes clockwise non-corner grid anchor points around a flap."""
  midpoint = point_utils.midpoint(rect.corners[0], rect.corners[1])
  start = shaping_utils.compute_anchor_points_start(
      midpoint, grid_size, even_phase, rect.get_unit_edge_vectors(grid_size)[0]
  )
  edge_index = shaping_utils.get_edge_index(start, rect.corners, rect.edges)
  cur_p = start
  grid_idx = 0
  width = int(shaping_utils.crease_grid_length(rect.edges[0], grid_size))
  height = int(shaping_utils.crease_grid_length(rect.edges[1], grid_size))
  upper_bound = (2 * (width + 1) + 2 * (height + 1)) // 2 + 1
  boundary = shaping_elements.get_flap_boundary(rect, grid_size)

  anchors: list[AnchorPoint] = []
  for _ in range(upper_bound):
    timeout_lib.maybe_timeout_cooperatively()
    if any(cur_p.isclose(pt) for pt, _, _ in anchors):
      break
    if not cur_p.isclose(rect.corners[edge_index]):
      anchors.append((cur_p, edge_index, grid_idx))
    cur_p, edge_index = boundary.walk(cur_p, edge_index, grid_size)
    cur_p, edge_index = boundary.walk(cur_p, edge_index, grid_size)
    grid_idx += 2
  return anchors


def shape_flap_with_zigzags(
    *,
    rect: FlapLike,
    grid_size: int,
    target_flap_angle: float | None,
    target_zigzag_depth: float | None,
    fold_order: Sequence[crease_utils.Fold],
    even_phase: bool,
    fold_percent: float,
    snap_to_grid: bool,
    fold_step: int = 0,
) -> list[crease_utils.Crease]:
  """Draws zigzag creases on `rect` and clips them to the unit paper."""
  if not rect.circle_centers:
    raise ValueError(
        f'Rectangle {rect} must have circle centers to apply zigzag shaping.'
    )
  if not rect.check_center_alignment():
    raise ValueError(
        f'Rectangle {rect} must be centered around its circle centers.'
    )
  anchors = compute_flap_anchor_points(rect, grid_size, even_phase)
  if not anchors:
    return []

  boundary = shaping_elements.get_flap_boundary(rect, grid_size)
  creases = shape_boundary_with_zigzags(
      anchors,
      boundary,
      grid_size,
      target_flap_angle,
      target_zigzag_depth,
      fold_order,
      even_phase,
      fold_percent,
      rect.get_ridges(),
      snap_to_grid=snap_to_grid,
      require_on_paper=False,
      fold_step=fold_step,
  )
  return shaping_utils.restrict_creases_to_paper(creases)


def compute_river_bank_anchor_points(
    boundary: shaping_elements.ShapingBoundary,
    grid_size: int,
    even_phase: bool,
) -> list[AnchorPoint]:
  """Computes alternating non-corner anchor points along a river bank."""
  start = boundary.creases[0].p1_init
  if not even_phase:
    start = boundary.edge_unit_vectors[0].point_add(start)
  edge_index = boundary.get_edge_index(start)
  if edge_index is None:
    raise ValueError(f'Start point {start} is not on river bank boundary.')

  cur_p = start
  end = boundary.creases[-1].p2_init
  upper_bound = sum(
      int(shaping_utils.get_grid_unit_length(edge.length, grid_size)) + 1
      for edge in boundary.creases
  )
  anchors: list[AnchorPoint] = []
  for grid_idx in range(upper_bound):
    timeout_lib.maybe_timeout_cooperatively()
    if anchors and cur_p.isclose(end):
      break
    if grid_idx % 2 == 0 and not cur_p.isclose(boundary.tips[edge_index]):
      anchors.append((cur_p, edge_index, grid_idx))
    cur_p, edge_index = boundary.walk(cur_p, edge_index, grid_size)

  if not boundary.closed:
    anchors.append((start, 0, 0))
  return anchors


def shape_river_with_zigzags(
    *,
    river: shaping_elements.ShapingRiver,
    grid_size: int,
    target_flap_angle: float | None,
    target_zigzag_depth: float | None,
    fold_order: Sequence[crease_utils.Fold],
    even_phase: bool,
    fold_percent: float,
    bank_index: shaping_types.BankIndex,
    snap_to_grid: bool,
    fold_step: int = 0,
) -> list[crease_utils.Crease]:
  """Draws zigzag creases along one bank of `river`."""
  depth = _resolve_zigzag_depth(target_flap_angle, target_zigzag_depth)
  if depth > river.width:
    raise ValueError(
        f'River {river.name} (width {river.width}) cannot fit zigzag depth'
        f' {depth}.'
    )
  boundary = river.get_boundary_from_index(bank_index)
  anchors = compute_river_bank_anchor_points(boundary, grid_size, even_phase)
  if not anchors:
    return []
  return shape_boundary_with_zigzags(
      anchors,
      boundary,
      grid_size,
      target_flap_angle,
      depth,
      fold_order,
      even_phase,
      fold_percent,
      river.ridges,
      snap_to_grid=snap_to_grid,
      require_on_paper=True,
      fold_step=fold_step,
  )


def _generate_zigzag_creases_split(
    grid_size: int,
    rectangles: Sequence[FlapLike],
    element_shaping_params: Sequence[shaping_types.ElementZigzagParams],
    *,
    rivers: Sequence[shaping_elements.ShapingRiver] = (),
    snap_to_grid: bool = False,
    fold_step: int = 0,
) -> tuple[list[crease_utils.Crease], list[crease_utils.Crease]]:
  """Generates `(all_creases, flap_only_creases)` for zigzag requests."""
  elements: dict[str, FlapLike | shaping_elements.ShapingRiver] = {
      **{rect.name: rect for rect in rectangles},
      **{river.name: river for river in rivers},
  }
  all_creases: list[crease_utils.Crease] = []
  flap_creases: list[crease_utils.Crease] = []

  for i, req in enumerate(element_shaping_params):
    timeout_lib.maybe_timeout_cooperatively()
    element = elements.get(req.element_name)
    if element is None:
      raise ValueError(f'Element {req.element_name!r} not found in CP.')
    zp = req.params
    if not (zp.target_flap_angle or zp.target_zigzag_depth):
      continue

    if isinstance(
        element, (shaping_utils.Rectangle, shaping_elements.ShapingFlap)
    ):
      if zp.bank_index is not None:
        raise ValueError(
            f'Flap {req.element_name!r} cannot specify river bank_index'
            f' {zp.bank_index!r}.'
        )
      flap = element
      if zp.depth > 0:
        if not isinstance(flap, shaping_elements.ShapingFlap):
          raise ValueError(
              f'Cannot inscribe zigzags (depth={zp.depth}) into plain'
              f' Rectangle {req.element_name!r}; pass a ShapingFlap.'
          )
        flap = flap.shrink(zp.depth)
      created = shape_flap_with_zigzags(
          rect=flap,
          grid_size=grid_size,
          target_flap_angle=zp.target_flap_angle,
          target_zigzag_depth=zp.target_zigzag_depth,
          fold_order=zp.fold_order,
          even_phase=zp.even_phase,
          fold_percent=zp.fold_percent,
          snap_to_grid=snap_to_grid,
          fold_step=fold_step + i,
      )
      all_creases.extend(created)
      flap_creases.extend(created)
    elif isinstance(element, shaping_elements.ShapingRiver):
      if zp.bank_index is None:
        raise ValueError(
            f'River {req.element_name!r} requires a non-None bank_index.'
        )
      target_depth = _resolve_zigzag_depth(
          zp.target_flap_angle, zp.target_zigzag_depth
      )
      if target_depth > element.width:
        logging.warning(
            'River %s is too narrow (width %s) for zigzag depth %s; skipping.',
            element.name,
            element.width,
            target_depth,
        )
        continue
      river = (
          element.shrink_along_bank(zp.bank_index, zp.depth)
          if zp.depth > 0
          else element
      )
      all_creases.extend(
          shape_river_with_zigzags(
              river=river,
              grid_size=grid_size,
              target_flap_angle=zp.target_flap_angle,
              target_zigzag_depth=zp.target_zigzag_depth,
              fold_order=zp.fold_order,
              even_phase=zp.even_phase,
              fold_percent=zp.fold_percent,
              bank_index=zp.bank_index,
              snap_to_grid=snap_to_grid,
              fold_step=fold_step + i,
          )
      )
    else:
      raise ValueError(f'Unsupported shaping element: {element!r}.')

  return all_creases, flap_creases


def get_zigzag_shaping_creases(
    grid_size: int,
    rectangles: Sequence[FlapLike],
    element_shaping_params: Sequence[shaping_types.ElementZigzagParams],
    *,
    rivers: Sequence[shaping_elements.ShapingRiver] = (),
    snap_to_grid: bool = False,
    fold_step: int = 0,
) -> list[crease_utils.Crease]:
  """Generates zigzag shaping creases for the requested flaps and rivers."""
  creases, _ = _generate_zigzag_creases_split(
      grid_size,
      rectangles,
      element_shaping_params,
      rivers=rivers,
      snap_to_grid=snap_to_grid,
      fold_step=fold_step,
  )
  return creases


def shape_cp_with_zigzags(
    cp: cp_utils.CreasePattern,
    grid_size: int,
    rectangles: Sequence[FlapLike],
    element_shaping_params: Sequence[shaping_types.ElementZigzagParams],
    *,
    rivers: Sequence[shaping_elements.ShapingRiver] = (),
    light_mode: bool = False,
    snap_to_grid: bool = False,
    narrow: bool = False,
    narrow_flap_names: Sequence[str] | None = None,
    raise_error_on_ff_violation: bool = False,
) -> cp_utils.CreasePattern:
  """Shapes `cp` with zigzag creases and optional flap narrowing."""
  zigzag_creases, flap_creases = _generate_zigzag_creases_split(
      grid_size,
      rectangles,
      element_shaping_params,
      rivers=rivers,
      snap_to_grid=snap_to_grid,
      fold_step=cp.num_fold_steps,
  )
  shaped_cp = cp.update_with_shaping_creases(
      zigzag_creases,
      split_shaping_by_shaping=not light_mode,
  )
  if raise_error_on_ff_violation and not shaped_cp.local_flat_foldability():
    raise ValueError('Zigzag shaped CP is not flat-foldable.')

  if narrow and flap_creases and rectangles:
    if narrow_flap_names is None:
      targets = rectangles
    else:
      targets = [
          r
          for r in rectangles
          if any(name in r.name for name in narrow_flap_names)
      ]
    if targets:
      shaped_cp = narrow_flap_shaping.narrow(
          cp=shaped_cp,
          zigzag_creases=flap_creases,
          rectangles=targets,
          grid_size=grid_size,
          light_mode=light_mode,
          raise_error_on_ff_violation=raise_error_on_ff_violation,
      )
  return shaped_cp


# ---------------------------------------------------------------------------
# Inscribed polygon shaping (flaps and rivers)
# ---------------------------------------------------------------------------


def find_inscribed_start_corner(
    corners: Sequence[point_utils.Point],
    edges: Sequence[crease_utils.Crease],
) -> int:
  """Returns the corner index where a border edge meets an interior edge."""
  for idx, corner in enumerate(corners):
    if shaping_utils.check_corner_between_border_edge_and_interior_edge(
        corner, corners, edges
    ):
      return idx
  return 0


def get_inscribed_rectangle(
    corners: Sequence[point_utils.Point],
    circle_centers: Sequence[point_utils.Point],
    grid_size: int,
    inscribe_depth: float,
) -> shaping_utils.Rectangle:
  """Shrinks `corners` inward toward `circle_centers` by `inscribe_depth`."""
  if not circle_centers:
    raise ValueError('circle_centers must be non-empty.')
  grid_delta = 1.0 / grid_size
  new_corners: list[point_utils.Point] = []

  for corner in corners:
    timeout_lib.maybe_timeout_cooperatively()
    closest = min(
        circle_centers,
        key=lambda c, pt=corner: point_utils.distance(c, pt),
    )
    to_center = vector.Vector.from_point(closest) - vector.Vector.from_point(
        corner
    )
    unit_step = grid_delta * to_center.inf_normalize
    slope = abs(unit_step.slope)
    if not (
        slope == math.inf
        or math.isclose(slope, 0.0)
        or math.isclose(slope, 1.0)
    ):
      raise ValueError(
          f'Expected axis-aligned or 45-degree corner step, got {unit_step}'
          f' for corner {corner} and centers {circle_centers}.'
      )
    displacement = inscribe_depth * unit_step
    if (
        displacement.norm >= to_center.norm - shaping_utils.EPS
        and not corner.isclose(closest)
    ):
      raise ValueError(
          f'inscribe_depth {inscribe_depth} exceeds corner-to-center distance'
          f' ({displacement.norm} >= {to_center.norm}).'
      )
    new_corners.append(displacement.point_add(corner))

  return shaping_utils.Rectangle.from_corners(
      new_corners, 'inscribed_rectangle', list(circle_centers)[:2]
  )


def _crosses_removed_centerline(
    cur_point: point_utils.Point,
    next_point: point_utils.Point,
    central_crease_removed: bool,
) -> bool:
  """Returns whether stepping `cur_point -> next_point` lands on `x = 0.5`."""
  return (
      central_crease_removed
      and not math.isclose(
          cur_point.x, _CENTRAL_CREASE_X, abs_tol=shaping_utils.EPS
      )
      and math.isclose(
          next_point.x, _CENTRAL_CREASE_X, abs_tol=shaping_utils.EPS
      )
  )


def get_inscribed_rectangle_creases(
    *,
    grid_size: int,
    start_index: int,
    corners: Sequence[point_utils.Point],
    fold_order: Sequence[crease_utils.Fold],
    fold_percent: float,
    central_crease_removed: bool = True,
    fold_step: int = 0,
) -> list[crease_utils.Crease]:
  """Walks clockwise along `corners`, emitting alternating-MV unit creases."""
  creases: list[crease_utils.Crease] = []
  start = corners[start_index]
  cur_point = start
  cur_edge = start_index
  fold_cycler = itertools.cycle(fold_order)
  boundary = shaping_elements.get_flap_boundary(
      shaping_utils.Rectangle.from_corners(corners, 'inscribed_rectangle'),
      grid_size,
  )

  while not (creases and (start.isclose(cur_point) or cur_point.is_border)):
    timeout_lib.maybe_timeout_cooperatively()
    next_point, next_edge = boundary.walk(cur_point, cur_edge, grid_size)
    if next_edge == cur_edge and _crosses_removed_centerline(
        cur_point, next_point, central_crease_removed
    ):
      next_point, next_edge = boundary.walk(next_point, next_edge, grid_size)
    creases.append(
        crease_utils.Crease(
            cur_point,
            next_point,
            fold=next(fold_cycler),
            fold_percent=fold_percent,
            fold_step=fold_step,
        )
    )
    cur_point, cur_edge = next_point, next_edge
  return creases


def shape_flap_with_inscribed_rectangles(
    flap: FlapLike,
    grid_size: int,
    shaping_params: Sequence[shaping_types.InscribedParams],
    central_crease_removed: bool = True,
    fold_step: int = 0,
) -> tuple[list[crease_utils.Crease], list[shaping_utils.Rectangle]]:
  """Inscribes concentric rectangles in `flap`, returning creases and rects."""
  rect = (
      flap.to_legacy_rectangle()
      if isinstance(flap, shaping_elements.ShapingFlap)
      else flap
  )
  if not rect.circle_centers:
    raise ValueError(
        f'Flap {rect.name!r} must have circle_centers to inscribe rectangles.'
    )

  flap_corners, _, _, _ = shaping_utils.preprocess_rect(
      rect, grid_size, None, 1
  )
  inscribed_rects: list[shaping_utils.Rectangle] = []
  creases: list[crease_utils.Crease] = []

  for param in sorted(shaping_params, key=lambda p: p.depth):
    timeout_lib.maybe_timeout_cooperatively()
    if param.bank_index is not None:
      raise ValueError(
          f'Flap {rect.name!r} cannot specify river bank_index'
          f' {param.bank_index!r}.'
      )
    inscribed = get_inscribed_rectangle(
        flap_corners, rect.circle_centers, grid_size, param.depth
    )
    clipped = inscribed.clip_to_paper()
    if clipped is None:
      continue
    inscribed_rects.append(inscribed)
    if math.isclose(param.fold_percent, 0.0):
      continue
    ins_corners, ins_edges, _, _ = shaping_utils.preprocess_rect(
        clipped, grid_size, None, 1
    )
    start_idx = find_inscribed_start_corner(ins_corners, ins_edges)
    creases.extend(
        get_inscribed_rectangle_creases(
            grid_size=grid_size,
            start_index=start_idx,
            corners=ins_corners,
            fold_order=param.fold_order,
            fold_percent=param.fold_percent,
            central_crease_removed=central_crease_removed,
            fold_step=fold_step,
        )
    )
  return creases, inscribed_rects


def get_inscribed_river_creases(
    river: shaping_elements.ShapingRiver,
    param: shaping_types.InscribedParams,
    central_crease_removed: bool = True,
    fold_step: int = 0,
) -> list[crease_utils.Crease]:
  """Walks along an inscribed bank of `river`, emitting alternating creases."""
  if param.bank_index is None:
    raise ValueError(
        f'River {river.name!r} requires a non-None bank_index for inscribing.'
    )
  segments = river.get_inscribed_river_bank(param.bank_index, param.depth)
  fold_cycler = itertools.cycle(param.fold_order)
  creases: list[crease_utils.Crease] = []

  for segment, edge_vec in zip(segments, river.edge_unit_vectors, strict=True):
    timeout_lib.maybe_timeout_cooperatively()
    num_steps = shaping_utils.integer_crease_grid_length(
        segment, river.grid_size
    )
    start = segment.p1_init
    step = 0
    while step < num_steps:
      end = edge_vec.point_add(start)
      if step + 1 < num_steps and _crosses_removed_centerline(
          start, end, central_crease_removed
      ):
        end = edge_vec.point_add(end)
        step += 1
      creases.append(
          crease_utils.Crease(
              start,
              end,
              fold=next(fold_cycler),
              fold_percent=param.fold_percent,
              fold_step=fold_step,
          )
      )
      start = end
      step += 1
  return creases


def shape_river_with_inscribed_rivers(
    river: shaping_elements.ShapingRiver,
    shaping_params: Sequence[shaping_types.InscribedParams],
    central_crease_removed: bool = True,
    fold_step: int = 0,
) -> list[crease_utils.Crease]:
  """Inscribes parallel river-bank contours inside `river`."""
  creases: list[crease_utils.Crease] = []
  for param in sorted(shaping_params, key=lambda p: p.depth):
    timeout_lib.maybe_timeout_cooperatively()
    if math.isclose(param.fold_percent, 0.0):
      continue
    creases.extend(
        get_inscribed_river_creases(
            river, param, central_crease_removed, fold_step=fold_step
        )
    )
  return shaping_utils.restrict_creases_to_paper(creases)


def get_inscribed_shaping_creases(
    grid_size: int,
    rectangles: Sequence[FlapLike],
    element_shaping_params: Sequence[shaping_types.ElementInscribedParams],
    *,
    rivers: Sequence[shaping_elements.ShapingRiver] = (),
    central_crease_removed: bool = True,
    fold_step: int | None = None,
) -> list[crease_utils.Crease]:
  """Generates inscribed polygon creases for the requested flaps and rivers."""
  elements: dict[str, FlapLike | shaping_elements.ShapingRiver] = {
      **{rect.name: rect for rect in rectangles},
      **{river.name: river for river in rivers},
  }
  creases: list[crease_utils.Crease] = []

  for i, req in enumerate(element_shaping_params):
    timeout_lib.maybe_timeout_cooperatively()
    element = elements.get(req.element_name)
    if element is None:
      raise ValueError(f'Element {req.element_name!r} not found in CP.')
    if not req.params:
      continue
    elem_fold_step = 0 if fold_step is None else fold_step + i
    if isinstance(
        element, (shaping_utils.Rectangle, shaping_elements.ShapingFlap)
    ):
      flap_creases, _ = shape_flap_with_inscribed_rectangles(
          element,
          grid_size,
          req.params,
          central_crease_removed=central_crease_removed,
          fold_step=elem_fold_step,
      )
      creases.extend(flap_creases)
    elif isinstance(element, shaping_elements.ShapingRiver):
      creases.extend(
          shape_river_with_inscribed_rivers(
              element,
              req.params,
              central_crease_removed=central_crease_removed,
              fold_step=elem_fold_step,
          )
      )
    else:
      raise ValueError(f'Unsupported shaping element: {element!r}.')
  return creases


def shape_cp_with_inscribed_elements(
    cp: cp_utils.CreasePattern,
    grid_size: int,
    rectangles: Sequence[FlapLike],
    element_shaping_params: Sequence[shaping_types.ElementInscribedParams],
    *,
    rivers: Sequence[shaping_elements.ShapingRiver] = (),
    central_crease_removed: bool = True,
    light_mode: bool = False,
    raise_error_on_ff_violation: bool = False,
) -> cp_utils.CreasePattern:
  """Shapes `cp` with inscribed flap rectangles and/or inscribed river banks."""
  creases = get_inscribed_shaping_creases(
      grid_size,
      rectangles,
      element_shaping_params,
      rivers=rivers,
      central_crease_removed=central_crease_removed,
  )
  shaped_cp = cp.update_with_shaping_creases(
      creases,
      split_shaping_by_shaping=not light_mode,
  )
  if raise_error_on_ff_violation and not shaped_cp.local_flat_foldability():
    raise ValueError('Inscribed polygon shaped CP is not flat-foldable.')
  return shaped_cp
