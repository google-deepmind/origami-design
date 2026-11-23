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
"""Pure-data crease templates for folded-flap and river clip-pattern shaping.

Each template returns `PatternSegment` entries in unit-flap coordinates, where
`x` spans `[0, 1]` across the flap or river hinge and `y >= 0` advances away
from the hinge along the folded strip.
"""

from collections.abc import Sequence
import dataclasses
import enum
import math

from ...crease_pattern import crease

_M = crease.Fold.MOUNTAIN
_V = crease.Fold.VALLEY


class TemplateName(enum.StrEnum):
  """Named high-level clip-pattern shaping templates."""

  FOLD = 'fold'
  TURN_AND_NARROW = 'turn_and_narrow'
  BEND_WHEN_TURN_AND_NARROW = 'bend_when_turn_and_narrow'
  PLEAT_TO_HIDE = 'pleat_to_hide'


@dataclasses.dataclass(frozen=True, slots=True)
class PatternSegment:
  """Single crease segment in unit-flap template coordinates.

  Attributes:
    p1: Segment start `(x, y)` in template coordinates.
    p2: Segment end `(x, y)` in template coordinates.
    fold: Fold assignment for an unflipped (face-up) paper layer.
    angle_deg: Target fold angle in `[0, 180]` degrees.
  """

  p1: tuple[float, float]
  p2: tuple[float, float]
  fold: crease.Fold
  angle_deg: float = 180.0

  def __post_init__(self) -> None:
    if not 0.0 <= self.angle_deg <= 180.0:
      raise ValueError(
          f'Fold angle must lie in [0, 180], got {self.angle_deg}.'
      )
    if self.fold not in (_M, _V, crease.Fold.BORDER):
      raise ValueError(f'Unsupported clip-pattern fold kind: {self.fold}.')


_NARROW_ADAPTER: tuple[PatternSegment, ...] = (
    PatternSegment((0.0, 0.0), (0.5, 0.5), _V),
    PatternSegment((0.0, 1.0), (0.5, 0.5), _V),
    PatternSegment((0.5, 0.5), (1.0, 0.5), _V),
    PatternSegment((0.0, 1.0), (0.5, 1.0), _M),
    PatternSegment((0.5, 1.0), (1.0, 1.0), _M),
    PatternSegment((0.5, 1.0), (0.5, 0.5), _M),
)

_SYM_NARROW_BASE_HALF: tuple[PatternSegment, ...] = (
    PatternSegment((0.0, 0.0), (1.0, 1.0), _M),
    PatternSegment((0.0, 1.0), (1.0, 1.0), _V),
    PatternSegment((1.0, 1.0), (0.5, 1.5), _M),
    PatternSegment((0.5, 1.5), (0.0, 2.0), _M),
    PatternSegment((0.0, 2.0), (0.5, 2.0), _V),
    PatternSegment((0.5, 2.0), (1.0, 2.0), _V),
)

_TURN_AND_NARROW_BASE_6X: tuple[PatternSegment, ...] = (
    PatternSegment((0.0, 0.0), (3.0, 3.0), _M),
    PatternSegment((3.0, 3.0), (6.0, 0.0), _M),
    PatternSegment((2.0, 4.0), (3.0, 3.0), _M),
    PatternSegment((1.0, 5.0), (2.0, 4.0), _M),
    PatternSegment((0.0, 6.0), (1.0, 5.0), _M),
    PatternSegment((3.0, 3.0), (4.0, 4.0), _M),
    PatternSegment((4.0, 4.0), (5.0, 5.0), _M),
    PatternSegment((5.0, 5.0), (6.0, 6.0), _M),
    PatternSegment((5.0, 5.0), (6.0, 5.0), _M),
    PatternSegment((4.0, 4.0), (4.0, 6.0), _M),
    PatternSegment((2.0, 4.0), (2.0, 6.0), _M),
    PatternSegment((0.0, 5.0), (1.0, 5.0), _M),
    PatternSegment((1.0, 5.0), (1.0, 6.0), _V),
    PatternSegment((5.0, 6.0), (5.0, 5.0), _V),
    PatternSegment((4.0, 4.0), (6.0, 4.0), _V),
    PatternSegment((0.0, 4.0), (2.0, 4.0), _V),
    PatternSegment((0.0, 3.0), (3.0, 3.0), _V, 90.0),
    PatternSegment((3.0, 3.0), (6.0, 3.0), _V, 90.0),
    PatternSegment((0.0, 6.0), (1.0, 6.0), _V),
    PatternSegment((1.0, 6.0), (2.0, 6.0), _V),
    PatternSegment((3.0, 3.0), (3.0, 6.0), _V),
    PatternSegment((2.0, 6.0), (3.0, 6.0), _V),
    PatternSegment((3.0, 6.0), (4.0, 6.0), _V),
    PatternSegment((4.0, 6.0), (5.0, 6.0), _V),
    PatternSegment((5.0, 6.0), (6.0, 6.0), _V),
)


def _affine_segments(
    segments: Sequence[PatternSegment],
    *,
    tx: float = 0.0,
    ty: float = 0.0,
    sx: float = 1.0,
    sy: float = 1.0,
) -> list[PatternSegment]:
  """Applies `(x, y) -> (tx + sx * x, ty + sy * y)` to template segments."""
  return [
      PatternSegment(
          p1=(tx + sx * seg.p1[0], ty + sy * seg.p1[1]),
          p2=(tx + sx * seg.p2[0], ty + sy * seg.p2[1]),
          fold=seg.fold,
          angle_deg=seg.angle_deg,
      )
      for seg in segments
  ]


def normalize_angle(angle_deg: float) -> float:
  """Normalizes an angle in degrees to `[-180, 180)`."""
  return (angle_deg + 180.0) % 360.0 - 180.0


def bend(angle: float, *, offset_y: float = 0.0) -> list[PatternSegment]:
  """Returns a horizontal 3D bend crease for `angle` degrees."""
  theta = normalize_angle(angle)
  if math.isclose(theta, 0.0, abs_tol=1e-8):
    return []
  fold = _M if theta > 0.0 else _V
  return [PatternSegment((0.0, offset_y), (1.0, offset_y), fold, abs(theta))]


def rotate(angle: float, *, offset_y: float = 0.0) -> list[PatternSegment]:
  """Returns one or two flat creases in `[0, 1]^2` rotating a flap by `angle`.

  Positive angles rotate clockwise; negative angles rotate counter-clockwise.
  Angles strictly inside `(0, 90)` or `(-90, 0)` compose a 90-degree crease
  with a counter-rotation crease so both fit inside a unit `[0, 1]^2` cell.

  Args:
    angle: Desired in-plane flap rotation in degrees.
    offset_y: Longitudinal offset from the hinge in grid units.

  Returns:
    Zero, one, or two 180-degree `PatternSegment` entries.
  """
  alpha = normalize_angle(angle)
  if math.isclose(alpha, 0.0, abs_tol=1e-8):
    return []
  if math.isclose(abs(alpha), 180.0, abs_tol=1e-5):
    y = offset_y + 0.05
    return [PatternSegment((0.0, y), (1.0, y), _V)]
  if math.isclose(alpha, 90.0, abs_tol=1e-5):
    return [PatternSegment((1.0, offset_y), (0.0, offset_y + 1.0), _V)]
  if math.isclose(alpha, -90.0, abs_tol=1e-5):
    return [PatternSegment((0.0, offset_y), (1.0, offset_y + 1.0), _V)]
  if 0.0 < alpha < 90.0:
    dy = math.tan(math.radians(45.0 - 0.5 * alpha))
    return [
        PatternSegment((1.0, offset_y), (0.0, offset_y + 1.0), _M),
        PatternSegment((1.0, offset_y), (0.0, offset_y + dy), _V),
    ]
  if 90.0 < alpha < 180.0:
    dy = math.tan(math.radians(0.5 * (180.0 - alpha)))
    return [PatternSegment((1.0, offset_y), (0.0, offset_y + dy), _V)]
  if -90.0 < alpha < 0.0:
    dy = math.tan(math.radians(45.0 + 0.5 * alpha))
    return [
        PatternSegment((0.0, offset_y), (1.0, offset_y + 1.0), _M),
        PatternSegment((0.0, offset_y), (1.0, offset_y + dy), _V),
    ]
  dy = math.tan(math.radians(0.5 * (180.0 + alpha)))
  return [PatternSegment((0.0, offset_y), (1.0, offset_y + dy), _V)]


def _narrow_symmetric(
    length: float | None, n_times: int, offset_y: float
) -> list[PatternSegment]:
  """Builds a symmetric flap or river narrowing gadget."""
  eff_len = 200.0 if length is None else length
  if eff_len < 2.0 - 1e-6:
    return []
  main_len = 2.0 * (eff_len - 2.0)
  main_half: list[PatternSegment] = []
  for i in range(1, n_times):
    t = i / n_times
    y_hi = 2.0 + main_len
    main_half.extend((
        PatternSegment((0.0, 2.0 - t), (t, 2.0 - t), _M),
        PatternSegment((t, 2.0 - t), (t, 2.0), _V),
        PatternSegment((t, 2.0), (t, y_hi), _M),
        PatternSegment((t, y_hi), (t, y_hi + t), _V),
        PatternSegment((t, y_hi + t), (0.0, y_hi + t), _M),
    ))
  entry = (*_SYM_NARROW_BASE_HALF, *main_half)
  out: list[PatternSegment] = []
  for tx, sx in ((0.0, 0.5), (1.0, -0.5)):
    out.extend(_affine_segments(entry, tx=tx, ty=offset_y, sx=sx, sy=0.5))
  for tx, sx in ((0.0, 0.5), (1.0, -0.5)):
    out.extend(
        _affine_segments(
            _SYM_NARROW_BASE_HALF,
            tx=tx,
            ty=offset_y + eff_len,
            sx=sx,
            sy=-0.5,
        )
    )
  return out


def narrow(
    length: float | None = None,
    *,
    symmetric: bool = True,
    n_times: int = 2,
    offset_y: float = 0.0,
) -> list[PatternSegment]:
  """Returns creases narrowing a flap (`length=None`) or river (`length>=2`).

  Args:
    length: River length in grid units, or `None` for an open-ended flap.
    symmetric: Whether to emit a bilaterally symmetric narrowing gadget.
    n_times: Integer narrowing factor (`>= 2`); target width is `1 / n_times`.
    offset_y: Longitudinal offset from the hinge in grid units.

  Returns:
    Template segments, or `[]` when `length` is shorter than 2 grid units.
  """
  if n_times < 2:
    raise ValueError(f'n_times must be >= 2, got {n_times}.')
  if symmetric:
    return _narrow_symmetric(length, n_times, offset_y)
  if length is None:
    out = _affine_segments(_NARROW_ADAPTER, ty=offset_y)
    out.append(
        PatternSegment((0.5, offset_y + 100.0), (0.5, offset_y + 1.0), _V)
    )
    return out
  if length < 2.0:
    return []
  out = _affine_segments(_NARROW_ADAPTER, ty=offset_y)
  if not math.isclose(length, 2.0, rel_tol=1e-5, abs_tol=1e-8):
    out.append(
        PatternSegment(
            (0.5, offset_y + length - 1.0), (0.5, offset_y + 1.0), _V
        )
    )
  out.extend(_affine_segments(_NARROW_ADAPTER, ty=offset_y + length, sy=-1.0))
  return out


def turn_and_narrow(
    length: float | None = None, *, offset_y: float = 0.0
) -> list[PatternSegment]:
  """Returns a 90-degree 3D turn combined with 6-pleat narrowing."""
  end_y = 600.0 if length is None else float(max(6, round(6.0 + 6.0 * length)))
  tails = (
      PatternSegment((2.0, 6.0), (2.0, end_y), _V),
      PatternSegment((4.0, 6.0), (4.0, end_y), _V),
      PatternSegment((1.0, 6.0), (1.0, end_y), _M),
      PatternSegment((5.0, 6.0), (5.0, end_y), _M),
      PatternSegment((3.0, 6.0), (3.0, end_y), _M),
  )
  scale = 1.0 / 6.0
  return _affine_segments(
      (*_TURN_AND_NARROW_BASE_6X, *tails), ty=offset_y, sx=scale, sy=scale
  )


def bend_when_turn_and_narrow(
    bend_angle: float, *, offset_y: float = 0.0
) -> list[PatternSegment]:
  """Returns alternating 6-pleat bend creases for a turned-and-narrowed flap."""
  if math.isclose(bend_angle, 0.0, abs_tol=1e-8):
    return []
  abs_angle = abs(bend_angle)
  out: list[PatternSegment] = []
  for i in range(6):
    fold = _M if i % 2 == 0 else _V
    if bend_angle < 0.0:
      fold = fold.flip()
    out.append(
        PatternSegment(
            (i / 6.0, offset_y), ((i + 1) / 6.0, offset_y), fold, abs_angle
        )
    )
  return out


def pleat_to_hide(
    length: float, distance: float, *, offset_y: float = 0.0
) -> list[PatternSegment]:
  """Returns an even count of alternating horizontal pleats inside `length`."""
  if distance <= 0.0:
    raise ValueError(f'Pleat distance must be positive, got {distance}.')
  eps = 1e-9
  num_lines = math.ceil(length / distance - eps)
  if num_lines % 2 != 0:
    num_lines -= 1
  if num_lines <= 0:
    return []
  slack = length - (num_lines - 1) * distance
  start_y = distance if distance < slack - eps else 0.5 * slack
  return [
      PatternSegment(
          p1=(0.0, offset_y + start_y + i * distance),
          p2=(1.0, offset_y + start_y + i * distance),
          fold=_M if i % 2 == 0 else _V,
      )
      for i in range(num_lines)
  ]


def build_fold_pattern(
    *,
    margin: float,
    step: float,
    bend_angle: float = 0.0,
    rotate_angle: float = 0.0,
    enable_narrow: bool = False,
    narrow_symmetric: bool = True,
    narrow_n_times: int | None = None,
    narrow_length: float | None = None,
) -> tuple[PatternSegment, ...]:
  """Composes sequential bend, in-plane rotate, and narrow sub-templates."""
  cursor = step
  if math.isclose(step, 0.0, abs_tol=1e-8):
    cursor += margin
  segments: list[PatternSegment] = []
  if not math.isclose(bend_angle, 0.0, abs_tol=1e-8):
    segments.extend(bend(bend_angle, offset_y=cursor))
    cursor += margin
  if not math.isclose(rotate_angle, 0.0, abs_tol=1e-8):
    segments.extend(rotate(rotate_angle, offset_y=cursor))
    cursor += 1.0 + margin
  if enable_narrow:
    remaining = (
        round(narrow_length - cursor, 8) if narrow_length is not None else None
    )
    factor = (
        2 if narrow_n_times is None or narrow_n_times < 2 else narrow_n_times
    )
    segments.extend(
        narrow(
            remaining,
            symmetric=narrow_symmetric,
            n_times=factor,
            offset_y=cursor,
        )
    )
  return tuple(segments)


def build_template_pattern(
    *,
    template_name: TemplateName,
    margin: float,
    step: float,
    bend_angle: float = 0.0,
    length: float | None = None,
    pleat_distance: float = 0.25,
) -> tuple[PatternSegment, ...]:
  """Builds one of the specialized named clip-pattern templates."""
  offset_y = margin + step
  match template_name:
    case TemplateName.TURN_AND_NARROW:
      return tuple(turn_and_narrow(length=length, offset_y=offset_y))
    case TemplateName.BEND_WHEN_TURN_AND_NARROW:
      return tuple(bend_when_turn_and_narrow(bend_angle, offset_y=offset_y))
    case TemplateName.PLEAT_TO_HIDE:
      base_length = 10.0 if length is None else length
      return tuple(
          pleat_to_hide(
              base_length - step - margin,
              pleat_distance,
              offset_y=offset_y,
          )
      )
    case _:
      raise ValueError(f'Unsupported template name: {template_name}.')
