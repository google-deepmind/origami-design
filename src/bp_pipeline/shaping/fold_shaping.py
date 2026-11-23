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
"""Simple-fold, reverse-fold, and crimp-fold shaping operations."""

from collections.abc import Mapping, Sequence
import dataclasses
import enum
import itertools
import math

from .. import timeout as timeout_lib
from ..crease_pattern import crease as crease_utils
from ..crease_pattern import crease_pattern as cp_utils
from ..crease_pattern import point as point_utils
from . import heuristics
from . import heuristics_utils
from . import narrow_flap_shaping
from . import shaping_elements
from . import shaping_utils
from . import types as shaping_types
from . import zigzag_shaping

PI_THRESHOLD = heuristics_utils.PI_THRESHOLD


# ---------------------------------------------------------------------------
# Simple folds (unified zigzag + 180-degree inscribed dispatch)
# ---------------------------------------------------------------------------


def _is_pi_angle(angle: float) -> bool:
  """Returns whether `|angle|` is within `PI_THRESHOLD` of `pi`."""
  return math.isclose(abs(angle), math.pi, abs_tol=PI_THRESHOLD)


def _partition_simple_fold_requests(
    requests: Sequence[shaping_types.ElementZigzagParams],
) -> tuple[
    list[shaping_types.ElementZigzagParams],
    list[shaping_types.ElementInscribedParams],
]:
  """Splits simple folds into `(zigzag_requests, inscribed_requests)`."""
  zigzag_reqs: list[shaping_types.ElementZigzagParams] = []
  inscribed_reqs: list[shaping_types.ElementInscribedParams] = []

  for req in requests:
    timeout_lib.maybe_timeout_cooperatively()
    zp = req.params
    angle = (
        zp.target_flap_angle
        if zp.target_flap_angle is not None
        else shaping_utils.fold_angle_from_zigzag_depth(zp.target_zigzag_depth)
    )
    if _is_pi_angle(angle):
      inscribed_reqs.append(
          shaping_types.ElementShapingParams(
              element_name=req.element_name,
              params=(
                  shaping_types.InscribedParams(
                      depth=int(zp.depth),
                      fold_order=zp.fold_order,
                      fold_percent=zp.fold_percent,
                      bank_index=zp.bank_index,
                  ),
              ),
          )
      )
    else:
      zigzag_reqs.append(req)
  return zigzag_reqs, inscribed_reqs


def get_simple_fold_shaping_creases(
    grid_size: int,
    flaps: Sequence[zigzag_shaping.FlapLike],
    rivers: Sequence[shaping_elements.ShapingRiver],
    element_shaping_params: Sequence[shaping_types.ElementZigzagParams],
    central_crease_removed: bool,
    *,
    fold_step: int = 0,
) -> list[crease_utils.Crease]:
  """Generates simple-fold creases (inscribed at pi, zigzag otherwise)."""
  zigzag_reqs, inscribed_reqs = _partition_simple_fold_requests(
      element_shaping_params
  )
  creases: list[crease_utils.Crease] = []
  if inscribed_reqs:
    creases.extend(
        zigzag_shaping.get_inscribed_shaping_creases(
            grid_size=grid_size,
            rectangles=flaps,
            element_shaping_params=inscribed_reqs,
            rivers=rivers,
            central_crease_removed=central_crease_removed,
            fold_step=fold_step,
        )
    )
  if zigzag_reqs:
    creases.extend(
        zigzag_shaping.get_zigzag_shaping_creases(
            grid_size=grid_size,
            rectangles=flaps,
            element_shaping_params=zigzag_reqs,
            rivers=rivers,
            fold_step=fold_step + len(inscribed_reqs),
        )
    )
  return creases


def shape_cp_with_simple_folds(
    cp: cp_utils.CreasePattern,
    grid_size: int,
    flaps: Sequence[zigzag_shaping.FlapLike],
    rivers: Sequence[shaping_elements.ShapingRiver],
    element_shaping_params: Sequence[shaping_types.ElementZigzagParams],
    central_crease_removed: bool,
    light_mode: bool = True,
    narrow: bool = False,
    raise_error_on_ff_violation: bool = False,
) -> cp_utils.CreasePattern:
  """Applies simple folds (inscribed at pi, zigzag otherwise) to `cp`."""
  zigzag_reqs, inscribed_reqs = _partition_simple_fold_requests(
      element_shaping_params
  )
  inscribed_flap_creases: list[crease_utils.Crease] = []
  if inscribed_reqs:
    if narrow and not zigzag_reqs and flaps:
      flap_ins = [
          r
          for r in inscribed_reqs
          if r.params and r.params[0].bank_index is None
      ]
      if flap_ins:
        inscribed_flap_creases = zigzag_shaping.get_inscribed_shaping_creases(
            grid_size=grid_size,
            rectangles=flaps,
            element_shaping_params=flap_ins,
            central_crease_removed=central_crease_removed,
            fold_step=cp.num_fold_steps,
        )
    cp = zigzag_shaping.shape_cp_with_inscribed_elements(
        cp=cp,
        grid_size=grid_size,
        rectangles=flaps,
        element_shaping_params=inscribed_reqs,
        rivers=rivers,
        central_crease_removed=central_crease_removed,
        light_mode=light_mode,
        raise_error_on_ff_violation=raise_error_on_ff_violation,
    )
  if zigzag_reqs:
    cp = zigzag_shaping.shape_cp_with_zigzags(
        cp=cp,
        grid_size=grid_size,
        rectangles=flaps,
        element_shaping_params=zigzag_reqs,
        rivers=rivers,
        light_mode=light_mode,
        narrow=narrow,
        raise_error_on_ff_violation=raise_error_on_ff_violation,
    )
  elif narrow and inscribed_flap_creases and flaps:
    cp = narrow_flap_shaping.narrow(
        cp=cp,
        zigzag_creases=inscribed_flap_creases,
        rectangles=flaps,
        grid_size=grid_size,
        light_mode=light_mode,
        raise_error_on_ff_violation=raise_error_on_ff_violation,
    )
  cp.grid_size = grid_size
  return cp


def apply_heuristics_to_shaping_params(
    *,
    grid_size: int,
    flaps: Sequence[shaping_elements.ShapingFlap],
    rivers: Sequence[shaping_elements.ShapingRiver],
    element_orientations: Mapping[str, heuristics.AbsFlapDirection],
    hinge_segments: Sequence[crease_utils.Crease],
    element_shaping_params: Sequence[shaping_types.ElementZigzagParams],
    use_heuristics_for_fold_order: bool,
    use_heuristics_for_depth: bool,
    use_heuristics_for_phase: bool,
    convert_small_angles: bool,
) -> list[shaping_types.ElementZigzagParams]:
  """Refines phase, fold order, depth, and acute angles via 2D heuristics."""
  elements: dict[str, shaping_elements.ShapingElement] = {
      **{f.name: f for f in flaps},
      **{r.name: r for r in rivers},
  }
  resolved: list[shaping_types.ElementZigzagParams] = []

  for req in element_shaping_params:
    timeout_lib.maybe_timeout_cooperatively()
    name = req.element_name
    element = elements[name]
    zp = req.params
    if zp.target_flap_angle is None:
      raise ValueError(
          f'Request for {name!r} must set target_flap_angle for heuristics.'
      )
    angle = zp.target_flap_angle

    if isinstance(element, shaping_elements.ShapingFlap):
      if use_heuristics_for_phase:
        phase = heuristics_utils.convert_flap_angle_to_global_phase(
            angle, element_orientations[name], element
        )
      else:
        phase = zp.even_phase
      if use_heuristics_for_fold_order:
        fold_order, is_valid = heuristics_utils.get_fold_order(
            phase, grid_size, element, hinge_segments
        )
      else:
        fold_order, is_valid = zp.fold_order, True
    elif isinstance(element, shaping_elements.ShapingRiver):
      if zp.bank_index is None:
        raise ValueError(f'River {name!r} requires a non-None bank_index.')
      if use_heuristics_for_phase:
        phase = heuristics_utils.convert_river_angle_to_global_phase(
            angle, element_orientations[name], zp.bank_index, element
        )
      else:
        phase = zp.even_phase
      fold_order, is_valid = zp.fold_order, True
    else:
      raise ValueError(f'Unsupported element type: {type(element)}.')

    updated = shaping_types.ElementShapingParams(
        element_name=name,
        params=dataclasses.replace(
            zp,
            target_flap_angle=abs(angle),
            fold_order=fold_order,
            even_phase=phase,
        ),
    )
    if convert_small_angles:
      batch = heuristics_utils.split_acute_zigzag_params(
          updated, inscribe_by_epsilon=False
      )
    else:
      batch = [updated]
    if (
        isinstance(element, shaping_elements.ShapingFlap)
        and use_heuristics_for_depth
        and not is_valid
    ):
      batch = [heuristics_utils.shift_zigzag_depth(item) for item in batch]
    resolved.extend(batch)
  return resolved


def _creases_match(c1: crease_utils.Crease, c2: crease_utils.Crease) -> bool:
  """Returns whether `c1` and `c2` share geometry, fold type, and percent."""
  return (
      c1.isclose(c2)
      and c1.fold == c2.fold
      and math.isclose(c1.fold_percent, c2.fold_percent, abs_tol=1e-3)
  )


def infer_fold_order_phase_from_creases(
    target_creases: Sequence[crease_utils.Crease],
    element: shaping_elements.ShapingElement,
    target_flap_angle: float,
    inscribe_depth: float,
    fold_percent: float,
    central_crease_removed: bool,
) -> shaping_types.ElementZigzagParams | None:
  """Finds `(even_phase, fold_order)` reproducing `target_creases`, if any."""
  is_river = isinstance(element, shaping_elements.ShapingRiver)
  bank = element.get_outer_bank_index() if is_river else None
  flaps = () if is_river else (element,)
  rivers = (element,) if is_river else ()

  for even_phase, fold_order in itertools.product(
      (True, False),
      (shaping_types.MOUNTAIN_TO_VALLEY, shaping_types.VALLEY_TO_MOUNTAIN),
  ):
    candidate_req = shaping_types.ElementShapingParams(
        element_name=element.name,
        params=shaping_types.ZigzagParams(
            target_flap_angle=target_flap_angle,
            fold_order=fold_order,
            even_phase=even_phase,
            depth=inscribe_depth,
            fold_percent=fold_percent,
            bank_index=bank,
        ),
    )
    candidate_creases = get_simple_fold_shaping_creases(
        grid_size=element.grid_size,
        flaps=flaps,
        rivers=rivers,
        element_shaping_params=(candidate_req,),
        central_crease_removed=central_crease_removed,
    )
    if len(candidate_creases) == len(target_creases) and all(
        any(_creases_match(target, cand) for cand in candidate_creases)
        for target in target_creases
    ):
      return candidate_req
  return None


# ---------------------------------------------------------------------------
# Reverse folds (inside/outside beak fold + split/flipped base ridge)
# ---------------------------------------------------------------------------


class ReverseFoldOrientation(enum.StrEnum):
  """Symmetry axis along which a flap reverse fold flips half of its zigzags."""

  VERTICAL = 'vertical'
  HORIZONTAL = 'horizontal'


@dataclasses.dataclass(frozen=True, kw_only=True)
class ReverseFoldParams:
  """Parameters for a single reverse fold on a flap or river bank."""

  flap_angle: float
  fold_order: Sequence[crease_utils.Fold]
  even_phase: bool
  bank_index: shaping_types.BankIndex | None = None
  fold_percent: float = 1.0
  depth: int = 0

  def __post_init__(self) -> None:
    normalized = tuple(self.fold_order)
    if not normalized:
      raise ValueError('fold_order must be non-empty.')
    object.__setattr__(self, 'fold_order', normalized)


@dataclasses.dataclass(frozen=True, kw_only=True)
class ReverseFoldShapingParams:
  """Reverse-fold configuration for one named flap or river."""

  element_name: str
  reverse_fold_params: Sequence[ReverseFoldParams]
  orientation: ReverseFoldOrientation | str = ReverseFoldOrientation.VERTICAL
  fold_type: crease_utils.Fold = crease_utils.Fold.VALLEY


def make_reverse_fold_params(
    *,
    flap_angle: float,
    even_phase: bool,
    fold_order: Sequence[crease_utils.Fold],
    fold_percent: float = 1.0,
    bank_index: shaping_types.BankIndex | None = None,
    depth: int | None = None,
) -> ReverseFoldParams:
  """Builds a `ReverseFoldParams` with normalized tuple `fold_order`."""
  return ReverseFoldParams(
      flap_angle=flap_angle,
      fold_order=fold_order,
      even_phase=even_phase,
      bank_index=bank_index,
      fold_percent=fold_percent,
      depth=depth or 0,
  )


@dataclasses.dataclass(frozen=True, kw_only=True)
class ZigzagInstruction:
  """Annotated reverse-fold crease with beak flag and per-element metadata."""

  crease: crease_utils.Crease
  is_beak: bool
  original_index: int
  flap_rectangle: zigzag_shaping.FlapLike | None
  fold_type: crease_utils.Fold = crease_utils.Fold.VALLEY
  orientation: str | None = ReverseFoldOrientation.VERTICAL


def _classify_reverse_fold_zigzags(
    flap: zigzag_shaping.FlapLike,
    zigzag_creases: Sequence[crease_utils.Crease],
    *,
    fold_type: crease_utils.Fold,
    orientation: str,
) -> tuple[list[ZigzagInstruction], list[crease_utils.Crease]]:
  """Identifies the central boundary-sharing beak pair in `zigzag_creases`."""
  x_min, y_min = flap.x_start, flap.y_start
  x_max, y_max = x_min + flap.x_len, y_min + flap.y_len
  cx, cy = flap.center
  center_pt = point_utils.Point(x=cx, y=cy, require_on_paper=False)
  tol = point_utils.DEFAULT_TOLERANCE

  def on_rect_border(pt: point_utils.Point) -> bool:
    return (
        math.isclose(pt.x, x_min, abs_tol=tol)
        or math.isclose(pt.x, x_max, abs_tol=tol)
        or math.isclose(pt.y, y_min, abs_tol=tol)
        or math.isclose(pt.y, y_max, abs_tol=tol)
    )

  best_pair: tuple[int, int] | None = None
  best_dist = math.inf
  for i in range(len(zigzag_creases)):
    timeout_lib.maybe_timeout_cooperatively()
    for j in range(i + 1, len(zigzag_creases)):
      shared = zigzag_creases[i].vertex_with_crease(zigzag_creases[j])
      if shared is not None and on_rect_border(shared):
        dist = point_utils.distance(shared, center_pt)
        if dist < best_dist:
          best_dist = dist
          best_pair = (i, j)

  beak_set = set(best_pair) if best_pair is not None else set()
  beaks = (
      [zigzag_creases[best_pair[0]], zigzag_creases[best_pair[1]]]
      if best_pair is not None
      else []
  )
  instructions = [
      ZigzagInstruction(
          crease=c,
          is_beak=idx in beak_set,
          original_index=idx,
          flap_rectangle=None if idx in beak_set else flap,
          fold_type=fold_type,
          orientation=orientation,
      )
      for idx, c in enumerate(zigzag_creases)
  ]
  return instructions, beaks


def _split_and_flip_reverse_fold_base(
    flap: zigzag_shaping.FlapLike,
    existing_creases: Sequence[crease_utils.Crease],
    beak_creases: Sequence[crease_utils.Crease],
    zigzag_creases: Sequence[crease_utils.Crease],
) -> list[crease_utils.Crease]:
  """Splits base creases spanning `beak_creases` and flips the inner segment."""
  if len(beak_creases) < 2:
    raise ValueError(
        f'Expected two reverse-fold beak creases, found {len(beak_creases)}.'
    )
  to_split: list[crease_utils.Crease] = []
  untouched: list[crease_utils.Crease] = []
  for c in existing_creases:
    if (
        c.vertex_with_crease(beak_creases[0]) is not None
        and c.vertex_with_crease(beak_creases[1]) is not None
    ):
      to_split.append(c)
    else:
      untouched.append(c)
  if not to_split:
    return list(existing_creases)

  temp_cp = cp_utils.CreasePattern(creases=[*zigzag_creases, *to_split])
  result = list(untouched)
  for c in temp_cp.creases:
    timeout_lib.maybe_timeout_cooperatively()
    if any(c.isclose(zz) for zz in zigzag_creases):
      continue
    if (
        flap.crease_is_within(c)
        and c.fold != crease_utils.Fold.BORDER
        and c.vertex_with_crease(beak_creases[0]) is not None
        and c.vertex_with_crease(beak_creases[1]) is not None
    ):
      result.append(c.flip_fold())
    else:
      result.append(c)
  return result


def get_reverse_fold_creases(
    cp: cp_utils.CreasePattern,
    grid_size: int,
    rectangles: Sequence[zigzag_shaping.FlapLike],
    element_shaping_params: Sequence[ReverseFoldShapingParams],
    *,
    rivers: Sequence[shaping_elements.ShapingRiver] = (),
    central_crease_removed: bool = True,
    snap_to_grid: bool = False,
) -> tuple[list[crease_utils.Crease], list[ZigzagInstruction]]:
  """Returns `(modified_creases, zigzag_instructions)` for reverse folds."""
  base_creases = list(cp.creases)
  instructions: list[ZigzagInstruction] = []
  elements: dict[
      str, zigzag_shaping.FlapLike | shaping_elements.ShapingRiver
  ] = {
      **{r.name: r for r in rectangles},
      **{r.name: r for r in rivers},
  }

  for req in element_shaping_params:
    element = elements.get(req.element_name)
    if element is None:
      raise ValueError(f'Element {req.element_name!r} not found in CP.')
    is_flap = isinstance(
        element, (shaping_utils.Rectangle, shaping_elements.ShapingFlap)
    )
    orientation_str = str(req.orientation)

    for rp in req.reverse_fold_params:
      timeout_lib.maybe_timeout_cooperatively()
      if math.isclose(rp.flap_angle, 0.0):
        continue
      is_inscribed = _is_pi_angle(rp.flap_angle)
      if is_inscribed:
        created = zigzag_shaping.get_inscribed_shaping_creases(
            grid_size=grid_size,
            rectangles=(element,) if is_flap else (),
            element_shaping_params=(
                shaping_types.ElementShapingParams(
                    element_name=element.name,
                    params=(
                        shaping_types.InscribedParams(
                            depth=rp.depth,
                            fold_order=rp.fold_order,
                            fold_percent=rp.fold_percent,
                            bank_index=rp.bank_index,
                        ),
                    ),
                ),
            ),
            rivers=() if is_flap else (element,),
            central_crease_removed=central_crease_removed,
            fold_step=cp.num_fold_steps,
        )
      else:
        created = zigzag_shaping.get_zigzag_shaping_creases(
            grid_size=grid_size,
            rectangles=(element,) if is_flap else (),
            element_shaping_params=(
                shaping_types.ElementShapingParams(
                    element_name=element.name,
                    params=shaping_types.ZigzagParams(
                        target_flap_angle=rp.flap_angle,
                        fold_order=rp.fold_order,
                        even_phase=rp.even_phase,
                        fold_percent=rp.fold_percent,
                        bank_index=rp.bank_index,
                    ),
                ),
            ),
            rivers=() if is_flap else (element,),
            snap_to_grid=snap_to_grid,
            fold_step=cp.num_fold_steps,
        )
      if not created:
        raise ValueError(
            f'No reverse-fold creases generated for {req.element_name!r} with'
            f' params {rp}.'
        )
      if is_flap and not is_inscribed:
        flap_instrs, beaks = _classify_reverse_fold_zigzags(
            element,
            created,
            fold_type=req.fold_type,
            orientation=orientation_str,
        )
        base_creases = _split_and_flip_reverse_fold_base(
            element, base_creases, beaks, created
        )
        instructions.extend(flap_instrs)
      else:
        instructions.extend(
            ZigzagInstruction(
                crease=c,
                is_beak=False,
                original_index=idx,
                flap_rectangle=None,
                fold_type=req.fold_type,
                orientation=orientation_str,
            )
            for idx, c in enumerate(created)
        )

  return base_creases, instructions


def assign_reverse_fold_orders(
    instructions: Sequence[ZigzagInstruction],
    fold_type: crease_utils.Fold | None = None,
    orientation: str | None = None,
    flip_base_folds: bool = False,
) -> list[crease_utils.Crease]:
  """Assigns M/V folds to reverse-fold `instructions` using beak symmetry."""
  assigned: list[crease_utils.Crease] = []
  for instr in instructions:
    active_fold_type = instr.fold_type if fold_type is None else fold_type
    if instr.is_beak:
      assigned.append(dataclasses.replace(instr.crease, fold=active_fold_type))
      continue

    fold = instr.crease.fold
    active_orient = instr.orientation if orientation is None else orientation
    flap = instr.flap_rectangle
    if active_orient and flap is not None:
      c = instr.crease
      needs_flip = False
      if active_orient == ReverseFoldOrientation.VERTICAL:
        needs_flip = (
            c.top_point.y > flap.center[1]
            if flap.is_bottom_of_central_crease
            else c.bottom_point.y < flap.center[1]
        )
      elif active_orient == ReverseFoldOrientation.HORIZONTAL:
        needs_flip = (
            c.left_point.x > flap.center[0]
            if flap.is_right_of_central_crease
            else c.right_point.x < flap.center[0]
        )
      if needs_flip:
        fold = fold.flip()
    if flip_base_folds:
      fold = fold.flip()
    assigned.append(dataclasses.replace(instr.crease, fold=fold))
  return assigned


def shape_cp_with_reverse_fold(
    cp: cp_utils.CreasePattern,
    grid_size: int,
    rectangles: Sequence[zigzag_shaping.FlapLike],
    element_shaping_params: Sequence[ReverseFoldShapingParams],
    *,
    rivers: Sequence[shaping_elements.ShapingRiver] = (),
    central_crease_removed: bool = True,
    snap_to_grid: bool = False,
) -> cp_utils.CreasePattern:
  """Shapes `cp` with inside or outside reverse folds using per-flap opts."""
  base_creases, instructions = get_reverse_fold_creases(
      cp=cp,
      grid_size=grid_size,
      rectangles=rectangles,
      element_shaping_params=element_shaping_params,
      rivers=rivers,
      central_crease_removed=central_crease_removed,
      snap_to_grid=snap_to_grid,
  )
  # Pass None so each instruction uses its own element's fold_type/orientation.
  zigzag_creases = assign_reverse_fold_orders(instructions)
  return cp_utils.CreasePattern(creases=[*base_creases, *zigzag_creases])


# ---------------------------------------------------------------------------
# Crimp folds (inscribed polygon + MM/VV zigzags + bridge-crease M/V repair)
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, kw_only=True)
class CrimpFoldParams:
  """Parameters for a single crimp fold (inscribed contour + zigzag pleat)."""

  depth: int
  fold_order: Sequence[crease_utils.Fold]
  fold_percent: float
  zigzag_flap_angle: float
  zigzag_fold_order: Sequence[crease_utils.Fold]
  zigzag_even_phase: bool
  bank_index: shaping_types.BankIndex | None = None

  def __post_init__(self) -> None:
    ins_order = tuple(self.fold_order)
    zz_order = tuple(self.zigzag_fold_order)
    if self.fold_percent != 0.0 and not ins_order:
      raise ValueError('fold_order must be non-empty when fold_percent != 0.')
    if not math.isclose(self.zigzag_flap_angle, 0.0) and not zz_order:
      raise ValueError(
          'zigzag_fold_order must be non-empty when zigzag_flap_angle != 0.'
      )
    object.__setattr__(self, 'fold_order', ins_order)
    object.__setattr__(self, 'zigzag_fold_order', zz_order)


@dataclasses.dataclass(frozen=True, kw_only=True)
class CrimpFoldShapingParams:
  """Crimp-fold configuration for one named flap or river."""

  element_name: str
  crimp_fold_params: Sequence[CrimpFoldParams]


def fix_crimp_flat_foldability(
    processed_creases: Sequence[crease_utils.Crease],
    inscribed_creases: Sequence[crease_utils.Crease],
    zigzag_creases: Sequence[crease_utils.Crease],
) -> list[crease_utils.Crease]:
  """Flips base segments bridging MM/VV zigzag apices to inscribed corners."""
  inscribed_pts = set(
      itertools.chain.from_iterable((c.p1, c.p2) for c in inscribed_creases)
  )
  zigzag_pts = set(
      itertools.chain.from_iterable((c.p1, c.p2) for c in zigzag_creases)
  )

  shared_ins: set[point_utils.Point] = set()
  shared_zz: set[point_utils.Point] = set()
  for ip in inscribed_pts:
    match = next((zp for zp in zigzag_pts if ip.isclose(zp)), None)
    if match is not None:
      shared_ins.add(ip)
      shared_zz.add(match)
  inscribed_only = inscribed_pts - shared_ins
  zigzag_only = zigzag_pts - shared_zz

  fold_counts: list[tuple[point_utils.Point, dict[crease_utils.Fold, int]]] = []
  for c in zigzag_creases:
    timeout_lib.maybe_timeout_cooperatively()
    for pt in (c.p1, c.p2):
      bucket = next((cnt for key, cnt in fold_counts if pt.isclose(key)), None)
      if bucket is None:
        fold_counts.append((pt, {c.fold: 1}))
      else:
        bucket[c.fold] = bucket.get(c.fold, 0) + 1

  repaired: list[crease_utils.Crease] = []
  for c in processed_creases:
    timeout_lib.maybe_timeout_cooperatively()
    if not c.mountain_or_valley():
      repaired.append(c)
      continue
    p1_zz = any(p.isclose(c.p1) for p in zigzag_only)
    p2_zz = any(p.isclose(c.p2) for p in zigzag_only)
    p1_ins = any(p.isclose(c.p1) for p in inscribed_only)
    p2_ins = any(p.isclose(c.p2) for p in inscribed_only)
    apex = (
        c.p1 if (p1_zz and p2_ins) else (c.p2 if (p2_zz and p1_ins) else None)
    )
    if apex is not None:
      counts = next(
          (cnt for key, cnt in fold_counts if apex.isclose(key)), None
      )
      if counts is not None and counts.get(c.fold, 0) > 1:
        c = dataclasses.replace(c, fold=c.fold.flip())
    repaired.append(c)
  return repaired


def assign_crimp_fold_orders(
    inscribed_creases: Sequence[crease_utils.Crease],
    zigzag_creases: Sequence[crease_utils.Crease],
    fold_order: Sequence[crease_utils.Fold],
    zigzag_fold_order: Sequence[crease_utils.Fold],
) -> tuple[list[crease_utils.Crease], list[crease_utils.Crease]]:
  """Reassigns cyclic fold orders onto crimp inscribed and zigzag creases."""
  new_ins = [
      dataclasses.replace(c, fold=fold_order[i % len(fold_order)])
      for i, c in enumerate(inscribed_creases)
  ]
  new_zz = [
      dataclasses.replace(c, fold=zigzag_fold_order[i % len(zigzag_fold_order)])
      for i, c in enumerate(zigzag_creases)
  ]
  return new_ins, new_zz


def get_crimp_fold_creases(
    cp: cp_utils.CreasePattern,
    grid_size: int,
    flaps: Sequence[zigzag_shaping.FlapLike],
    element_shaping_params: Sequence[CrimpFoldShapingParams],
    *,
    rivers: Sequence[shaping_elements.ShapingRiver] = (),
    central_crease_removed: bool = True,
    snap_to_grid: bool = False,
) -> tuple[list[crease_utils.Crease], list[crease_utils.Crease]]:
  """Returns `(inscribed_creases, zigzag_creases)` for crimp-fold requests."""
  flaps_by_name = {f.name: f for f in flaps}
  rivers_by_name = {r.name: r for r in rivers}
  inscribed_out: list[crease_utils.Crease] = []
  zigzag_out: list[crease_utils.Crease] = []

  for req in element_shaping_params:
    flap = flaps_by_name.get(req.element_name)
    river = rivers_by_name.get(req.element_name)
    if flap is None and river is None:
      raise ValueError(f'Element {req.element_name!r} not found in CP.')

    for cp_param in req.crimp_fold_params:
      ins_param = shaping_types.InscribedParams(
          depth=cp_param.depth,
          fold_order=cp_param.fold_order,
          fold_percent=cp_param.fold_percent,
          bank_index=cp_param.bank_index,
      )
      if flap is not None:
        ins_creases, ins_rects = (
            zigzag_shaping.shape_flap_with_inscribed_rectangles(
                flap,
                grid_size,
                (ins_param,),
                central_crease_removed=central_crease_removed,
                fold_step=cp.num_fold_steps,
            )
        )
        shrunk_flaps: Sequence[zigzag_shaping.FlapLike] = ins_rects
        shrunk_rivers: Sequence[shaping_elements.ShapingRiver] = ()
        target_name = ins_rects[0].name if ins_rects else ''
      else:
        if river is None or cp_param.bank_index is None:
          raise ValueError(
              f'River {req.element_name!r} crimp fold requires a bank_index.'
          )
        ins_creases = zigzag_shaping.shape_river_with_inscribed_rivers(
            river,
            (ins_param,),
            central_crease_removed=central_crease_removed,
            fold_step=cp.num_fold_steps,
        )
        shrunk_river = river.shrink_along_bank(
            cp_param.bank_index, cp_param.depth
        )
        shrunk_flaps = ()
        shrunk_rivers = (shrunk_river,)
        target_name = shrunk_river.name

      if not math.isclose(cp_param.fold_percent, 0.0):
        inscribed_out.extend(ins_creases)

      if not math.isclose(cp_param.zigzag_flap_angle, 0.0):
        if not target_name:
          raise ValueError(
              f'Inscribed region for {req.element_name!r} clipped off paper.'
          )
        raw_zz = zigzag_shaping.get_zigzag_shaping_creases(
            grid_size=grid_size,
            rectangles=shrunk_flaps,
            element_shaping_params=(
                shaping_types.ElementShapingParams(
                    element_name=target_name,
                    params=shaping_types.ZigzagParams(
                        target_flap_angle=cp_param.zigzag_flap_angle,
                        fold_order=shaping_types.MOUNTAIN_TO_VALLEY,
                        even_phase=cp_param.zigzag_even_phase,
                        bank_index=cp_param.bank_index,
                    ),
                ),
            ),
            rivers=shrunk_rivers,
            snap_to_grid=snap_to_grid,
            fold_step=cp.num_fold_steps,
        )
        if not raw_zz:
          raise ValueError(
              f'No crimp zigzag creases generated for {req.element_name!r}'
              f' with params {cp_param}.'
          )
        zz_order = cp_param.zigzag_fold_order
        zigzag_out.extend(
            dataclasses.replace(c, fold=zz_order[idx % len(zz_order)])
            for idx, c in enumerate(raw_zz)
        )

  return inscribed_out, zigzag_out


def shape_cp_with_crimp_fold(
    cp: cp_utils.CreasePattern,
    grid_size: int,
    flaps: Sequence[zigzag_shaping.FlapLike],
    element_shaping_params: Sequence[CrimpFoldShapingParams],
    *,
    rivers: Sequence[shaping_elements.ShapingRiver] = (),
    central_crease_removed: bool = True,
    snap_to_grid: bool = False,
) -> cp_utils.CreasePattern:
  """Shapes `cp` with crimp folds and repairs bridge-crease fold parity."""
  inscribed_creases, zigzag_creases = get_crimp_fold_creases(
      cp=cp,
      grid_size=grid_size,
      flaps=flaps,
      element_shaping_params=element_shaping_params,
      rivers=rivers,
      central_crease_removed=central_crease_removed,
      snap_to_grid=snap_to_grid,
  )
  added = [*inscribed_creases, *zigzag_creases]
  tips, _ = cp_utils.get_tips_and_grid_creases(
      [*cp.creases, *added], grid_size=-1
  )
  split_base = cp_utils.split_creases(cp.creases, added, tips)
  repaired_base = fix_crimp_flat_foldability(
      split_base, inscribed_creases, zigzag_creases
  )
  return cp_utils.CreasePattern(
      creases=[*repaired_base, *added], light_mode=True
  )
