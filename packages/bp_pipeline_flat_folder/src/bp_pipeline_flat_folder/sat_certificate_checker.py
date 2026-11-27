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

"""Independent exact-rational SAT certificate checker for flat-foldability.

Verifies a candidate flat-folded state (`cell_faces_ordered` or `face_orders`
plus `face_orientations`) directly against a raw `.cp` crease pattern using only
exact integer/rational arithmetic (zero C++ dependencies, zero imports from
`bp_pipeline.third_party.flat_folder`).

Checked invariants:
1. Exact planar subdivision of the `.cp` crease pattern and exact isometric
   reflection of all faces across active (`M`, `V`, `U`) creases (verifying
   vertex closure / Kawasaki consistency and face-orientation parity).
2. Per-cell stack coverage, cross-cell pairwise order consistency, and per-cell
   acyclicity (strict total order on the exact set of faces covering each cell).
3. Mountain/Valley (`M`/`V`) local relative layer-order consistency across every
   assigned interior crease edge.
4. Non-crossing along every elementary folded segment:
   - no taco-taco interleaving on the same side of a folded line,
   - no taco-tortilla piercing (for both uncreased spanning faces and `F` flat
     creases),
   - no tortilla-tortilla inversion across a folded line.
"""

import collections
from collections.abc import Mapping, Sequence
import dataclasses
import enum
import fractions
import math
import numbers
import re
import sys
from typing import Any

Point2DInt = tuple[int, int]
Transform2D = tuple[
    int, int, int, int, int, int
]  # (a11, a12, tx, a21, a22, ty)

# Maximum denominator allowed for each rational token and for the cumulative
# common denominator `lcm_denom`.
# Derivation (IEEE-754 binary64, 53-bit significand): solver coordinates arrive
# as binary64 floats and are scaled by `u_scale = 4 * lcm_denom` and
# `f_scale_total = 16 * lcm_denom`. Power-of-two scaling shifts only the
# exponent, so capping both token and cumulative denominators at `2^50` keeps
# `u_scale` and `f_scale_total` bit-exact in binary64 (`<= 2^54 << 2^1024`) and
# caps intermediate `math.lcm` products at `2^100` (no exponential bignum
# blowup across distinct prime denominators).
_MAX_RATIONAL_DENOM_BITS = 50
_MAX_RATIONAL_DENOM = 1 << _MAX_RATIONAL_DENOM_BITS
_MAX_LCM_DENOM = _MAX_RATIONAL_DENOM

# Maximum coordinate magnitude (`10^100`, matching C++ ingress bound).
# With `f_scale_total <= 16 * 2^50 < 2e16`, both `|x| * f_scale_total < 2e116`
# and `(2 * |x|)^2 <= 4e200` stay well below `sys.float_info.max` (~1.797e308),
# preventing float64 overflow while bounding exact integers to <= 388 bits.
_MAX_COORD_MAGNITUDE_EXP10 = 100
_MAX_COORD_MAGNITUDE = 10**_MAX_COORD_MAGNITUDE_EXP10

# Maximum scientific-notation exponent `|exp|` (`sys.float_info.max_10_exp ==
# 308`). Checked before `fractions.Fraction` so `10 ** abs(exp)` is <= 309
# digits and cannot stall on `"1e100000000"` or `"1e-100000000"`.
_MAX_DECIMAL_EXPONENT = sys.float_info.max_10_exp

_RATIONAL_TOKEN_RE = re.compile(r"^[+-]?(\d+)/(\d+)$")
_DECIMAL_TOKEN_RE = re.compile(
    r"^[+-]?(?:(?P<int>\d+)(?:\.(?P<frac>\d*))?|\.(?P<frac_only>\d+))"
    r"(?:[eE](?P<exp>[+-]?\d+))?$"
)


class CertificateRejectionKind(enum.StrEnum):
  """Three-way classification of failed SAT certificate checks."""

  PATTERN = "pattern"
  ENGINE_CLAIM = "engine_claim"
  CANNOT_CHECK = "cannot_check"


@dataclasses.dataclass(frozen=True, kw_only=True)
class CertificateRejectionRule:
  """Classification rule in `REJECTION_REASON_TABLE`."""

  pattern: str
  kind: CertificateRejectionKind
  prefix_match: bool = False
  rationale: str = ""


# Single source of truth classifying every rejection reason emitted by this
# module:
# - `PATTERN`: exact proof that no flat folding of this pattern exists
#   (every row carries one sentence stating why).
# - `ENGINE_CLAIM`: proof only that the engine's claimed layer order or fields
#   are wrong (plus any reason not found in this table by default).
# - `CANNOT_CHECK`: checker parse, geometric, or scope limits.
REJECTION_REASON_TABLE: tuple[CertificateRejectionRule, ...] = (
    # --- Kind 1: exact proof about the pattern (`NOT_FOLDABLE`) ---
    CertificateRejectionRule(
        pattern="orientation_parity_conflict",
        kind=CertificateRejectionKind.PATTERN,
        rationale=(
            "Active creases toggle face orientation parity while flat and"
            " internal boundary edges preserve it, so an odd-parity face cycle"
            " proves no flat folding exists regardless of layer order."
        ),
    ),
    CertificateRejectionRule(
        pattern="transform_holonomy_conflict",
        kind=CertificateRejectionKind.PATTERN,
        rationale=(
            "Each face's 2D orthogonal folded transform is uniquely fixed by"
            " reflections across creases, so conflicting transforms along a"
            " cycle prove no flat folding exists regardless of layer order."
        ),
    ),
    CertificateRejectionRule(
        pattern="folded_vertex_closure_mismatch",
        kind=CertificateRejectionKind.PATTERN,
        rationale=(
            "Folded vertex positions are fixed by the creases alone, so two"
            " faces placing a shared vertex at distinct exact points prove no"
            " flat folding exists whatever the layer order."
        ),
    ),
    # --- Kind 3: checker parse, scope, or representation limits ---
    CertificateRejectionRule(
        pattern="cp_parse_error",
        kind=CertificateRejectionKind.CANNOT_CHECK,
        rationale="Token invalid or outside the checker's numeric bounds.",
    ),
    CertificateRejectionRule(
        pattern="zero_length_segment_at_line_",
        kind=CertificateRejectionKind.CANNOT_CHECK,
        prefix_match=True,
        rationale=(
            "Input contains a degenerate zero-length segment rejected at"
            " checker ingress."
        ),
    ),
    CertificateRejectionRule(
        pattern="non_octilinear_cp",
        kind=CertificateRejectionKind.CANNOT_CHECK,
        rationale="Crease direction is off the exact 45-degree lines.",
    ),
    CertificateRejectionRule(
        pattern="excessive_common_denominator",
        kind=CertificateRejectionKind.CANNOT_CHECK,
        rationale="Cumulative coordinate LCM denominator exceeds 2^50.",
    ),
    CertificateRejectionRule(
        pattern="empty_cp",
        kind=CertificateRejectionKind.CANNOT_CHECK,
        rationale="No crease segments were parsed from the input document.",
    ),
    CertificateRejectionRule(
        pattern="empty_arrangement",
        kind=CertificateRejectionKind.CANNOT_CHECK,
        rationale="Planar subdivision produced no segments for checking.",
    ),
    CertificateRejectionRule(
        pattern="empty_folded_arrangement",
        kind=CertificateRejectionKind.CANNOT_CHECK,
        rationale="Exact folded arrangement produced no bounded cells.",
    ),
    CertificateRejectionRule(
        pattern="conflicting_collinear_crease_codes",
        kind=CertificateRejectionKind.CANNOT_CHECK,
        rationale=(
            "Overlapping collinear segments carry distinct non-flat codes,"
            " preventing construction of a unique unfolded crease map."
        ),
    ),
    CertificateRejectionRule(
        pattern="no_unfolded_faces",
        kind=CertificateRejectionKind.CANNOT_CHECK,
        rationale="Input segments bound no interior unfolded face.",
    ),
    CertificateRejectionRule(
        pattern="non_manifold_or_orphan_edge",
        kind=CertificateRejectionKind.CANNOT_CHECK,
        rationale="Loose, orphan, or >2-face edge is outside 2-manifold BFS.",
    ),
    CertificateRejectionRule(
        pattern="slit_or_dangling_edge",
        kind=CertificateRejectionKind.CANNOT_CHECK,
        rationale="Slit or dangling edge has the same face on both sides.",
    ),
    CertificateRejectionRule(
        pattern="disconnected_unfolded_faces",
        kind=CertificateRejectionKind.CANNOT_CHECK,
        rationale="Unfolded mesh has multiple edge-disconnected components.",
    ),
    CertificateRejectionRule(
        pattern="unvisited_vertex",
        kind=CertificateRejectionKind.CANNOT_CHECK,
        rationale="Unfolded vertex lies outside all bounded checker faces.",
    ),
    CertificateRejectionRule(
        pattern="cell_sample_failed_",
        kind=CertificateRejectionKind.CANNOT_CHECK,
        prefix_match=True,
        rationale="Checker found no exact integer interior point in the cell.",
    ),
    CertificateRejectionRule(
        pattern="zero_folded_normal_cross",
        kind=CertificateRejectionKind.CANNOT_CHECK,
        rationale="Degenerate folded edge normal yields zero cross product.",
    ),
    CertificateRejectionRule(
        pattern="mv_edge_faces_never_overlap_",
        kind=CertificateRejectionKind.CANNOT_CHECK,
        prefix_match=True,
        rationale="Checker found no folded sample cell shared by M/V faces.",
    ),
    CertificateRejectionRule(
        pattern="taco_face_missing_from_adjacent_cell_",
        kind=CertificateRejectionKind.CANNOT_CHECK,
        prefix_match=True,
        rationale="Adjacent folded cell covering omits a taco crease face.",
    ),
    CertificateRejectionRule(
        pattern="tortilla_face_missing_from_cell_",
        kind=CertificateRejectionKind.CANNOT_CHECK,
        prefix_match=True,
        rationale="Side folded cell covering omits a spanning tortilla face.",
    ),
    CertificateRejectionRule(
        pattern="tortilla_missing_side_cell",
        kind=CertificateRejectionKind.CANNOT_CHECK,
        rationale="Tortilla segment lacks an adjacent cell covering map.",
    ),
    CertificateRejectionRule(
        pattern="tortilla_face_not_in_side_cell",
        kind=CertificateRejectionKind.CANNOT_CHECK,
        rationale="Tortilla face pair misses the adjacent cell covering.",
    ),
    CertificateRejectionRule(
        pattern="degenerate_shared_tortilla",
        kind=CertificateRejectionKind.CANNOT_CHECK,
        rationale="Two tortilla pairs share a face across the same segment.",
    ),
    # --- Kind 2: engine claim (wrong layer order or handed-over fields) ---
    CertificateRejectionRule(
        pattern="result_not_sat",
        kind=CertificateRejectionKind.ENGINE_CLAIM,
    ),
    CertificateRejectionRule(
        pattern="missing_solver_result_fields",
        kind=CertificateRejectionKind.ENGINE_CLAIM,
    ),
    CertificateRejectionRule(
        pattern="invalid_folded_state_fold",
        kind=CertificateRejectionKind.ENGINE_CLAIM,
    ),
    CertificateRejectionRule(
        pattern="missing_faceOrders",
        kind=CertificateRejectionKind.ENGINE_CLAIM,
    ),
    CertificateRejectionRule(
        pattern="missing_faceOrientations",
        kind=CertificateRejectionKind.ENGINE_CLAIM,
    ),
    CertificateRejectionRule(
        pattern="face_count_mismatch:",
        kind=CertificateRejectionKind.ENGINE_CLAIM,
        prefix_match=True,
    ),
    CertificateRejectionRule(
        pattern="invalid_solver_unfolded_vertex",
        kind=CertificateRejectionKind.ENGINE_CLAIM,
    ),
    CertificateRejectionRule(
        pattern="non_finite_solver_unfolded_vertex",
        kind=CertificateRejectionKind.ENGINE_CLAIM,
    ),
    CertificateRejectionRule(
        pattern="solver_unfolded_vertex_mismatch",
        kind=CertificateRejectionKind.ENGINE_CLAIM,
    ),
    CertificateRejectionRule(
        pattern="solver_face_vertex_index_out_of_range_",
        kind=CertificateRejectionKind.ENGINE_CLAIM,
        prefix_match=True,
    ),
    CertificateRejectionRule(
        pattern="degenerate_solver_face",
        kind=CertificateRejectionKind.ENGINE_CLAIM,
    ),
    CertificateRejectionRule(
        pattern="face_polygon_mismatch_at_",
        kind=CertificateRejectionKind.ENGINE_CLAIM,
        prefix_match=True,
    ),
    CertificateRejectionRule(
        pattern="face_orientation_mismatch_at_",
        kind=CertificateRejectionKind.ENGINE_CLAIM,
        prefix_match=True,
    ),
    CertificateRejectionRule(
        pattern="invalid_solver_folded_vertex",
        kind=CertificateRejectionKind.ENGINE_CLAIM,
    ),
    CertificateRejectionRule(
        pattern="non_finite_solver_folded_vertex",
        kind=CertificateRejectionKind.ENGINE_CLAIM,
    ),
    CertificateRejectionRule(
        pattern="folded_vertex_count_mismatch",
        kind=CertificateRejectionKind.ENGINE_CLAIM,
    ),
    CertificateRejectionRule(
        pattern="solver_folded_vertex_mismatch",
        kind=CertificateRejectionKind.ENGINE_CLAIM,
    ),
    CertificateRejectionRule(
        pattern="missing_solver_cell_geometry",
        kind=CertificateRejectionKind.ENGINE_CLAIM,
    ),
    CertificateRejectionRule(
        pattern="cell_geometry_length_mismatch",
        kind=CertificateRejectionKind.ENGINE_CLAIM,
    ),
    CertificateRejectionRule(
        pattern="invalid_solver_cell_point",
        kind=CertificateRejectionKind.ENGINE_CLAIM,
    ),
    CertificateRejectionRule(
        pattern="non_finite_solver_cell_point",
        kind=CertificateRejectionKind.ENGINE_CLAIM,
    ),
    CertificateRejectionRule(
        pattern="cell_face_index_out_of_range_",
        kind=CertificateRejectionKind.ENGINE_CLAIM,
        prefix_match=True,
    ),
    CertificateRejectionRule(
        pattern="cell_vertex_index_out_of_range_",
        kind=CertificateRejectionKind.ENGINE_CLAIM,
        prefix_match=True,
    ),
    CertificateRejectionRule(
        pattern="degenerate_solver_cell_",
        kind=CertificateRejectionKind.ENGINE_CLAIM,
        prefix_match=True,
    ),
    CertificateRejectionRule(
        pattern="unmatched_folded_cell_",
        kind=CertificateRejectionKind.ENGINE_CLAIM,
        prefix_match=True,
    ),
    CertificateRejectionRule(
        pattern="face_orders_length_mismatch",
        kind=CertificateRejectionKind.ENGINE_CLAIM,
    ),
    CertificateRejectionRule(
        pattern="invalid_solver_face_orders",
        kind=CertificateRejectionKind.ENGINE_CLAIM,
    ),
    CertificateRejectionRule(
        pattern="duplicate_face_order_rank_in_cell_",
        kind=CertificateRejectionKind.ENGINE_CLAIM,
        prefix_match=True,
    ),
    CertificateRejectionRule(
        pattern="no_order_certificate_supplied",
        kind=CertificateRejectionKind.ENGINE_CLAIM,
    ),
    CertificateRejectionRule(
        pattern="nonempty_stack_on_empty_cell_",
        kind=CertificateRejectionKind.ENGINE_CLAIM,
        prefix_match=True,
    ),
    CertificateRejectionRule(
        pattern="cell_stack_coverage_mismatch_",
        kind=CertificateRejectionKind.ENGINE_CLAIM,
        prefix_match=True,
    ),
    CertificateRejectionRule(
        pattern="cross_cell_order_cycle_on_faces_",
        kind=CertificateRejectionKind.ENGINE_CLAIM,
        prefix_match=True,
    ),
    CertificateRejectionRule(
        pattern="mv_order_violation_on_faces_",
        kind=CertificateRejectionKind.ENGINE_CLAIM,
        prefix_match=True,
    ),
    CertificateRejectionRule(
        pattern="taco_taco_crossing",
        kind=CertificateRejectionKind.ENGINE_CLAIM,
    ),
    CertificateRejectionRule(
        pattern="taco_tortilla_crossing",
        kind=CertificateRejectionKind.ENGINE_CLAIM,
    ),
    CertificateRejectionRule(
        pattern="tortilla_tortilla_crossing",
        kind=CertificateRejectionKind.ENGINE_CLAIM,
    ),
)

_EXACT_REJECTION_RULES: dict[str, CertificateRejectionKind] = {
    rule.pattern: rule.kind
    for rule in REJECTION_REASON_TABLE
    if not rule.prefix_match
}
_PREFIX_REJECTION_RULES: tuple[tuple[str, CertificateRejectionKind], ...] = (
    tuple(
        (rule.pattern, rule.kind)
        for rule in REJECTION_REASON_TABLE
        if rule.prefix_match
    )
)


def classify_rejection_reason(reason: str) -> CertificateRejectionKind:
  """Classifies an exact-checker failure reason into one of three kinds.

  Reasons not matched by `REJECTION_REASON_TABLE` default to
  `CertificateRejectionKind.ENGINE_CLAIM` so unrecognised failures never refute
  a crease pattern.

  Args:
    reason: Exact reason string emitted on a failed `CertificateCheckResult`.

  Returns:
    Matching `CertificateRejectionKind` (`PATTERN`, `ENGINE_CLAIM`, or
    `CANNOT_CHECK`).
  """
  exact = _EXACT_REJECTION_RULES.get(reason)
  if exact is not None:
    return exact
  for prefix, kind in _PREFIX_REJECTION_RULES:
    if reason.startswith(prefix):
      return kind
  return CertificateRejectionKind.ENGINE_CLAIM


@dataclasses.dataclass(frozen=True)
class CertificateCheckResult:
  """Outcome of verifying a flat-foldability SAT certificate."""

  ok: bool
  reason: str = ""
  num_faces: int = 0
  num_nonempty_cells: int = 0
  checked_mv_edges: int = 0
  checked_taco_taco_pairs: int = 0
  checked_taco_tortilla_pairs: int = 0
  checked_tortilla_tortilla_pairs: int = 0

  @property
  def rejection_kind(self) -> CertificateRejectionKind | None:
    """Kind of certificate rejection, or None when `ok` is True."""
    if self.ok:
      return None
    return classify_rejection_reason(self.reason)


def _parse_rational(token: str) -> fractions.Fraction:
  """Parses a `.cp` token to a bounded exact rational (fails closed)."""
  max_str_digits = sys.get_int_max_str_digits()
  if max_str_digits > 0 and len(token) > max_str_digits:
    raise ValueError(f"Oversized coordinate token (len={len(token)})")

  rat_match = _RATIONAL_TOKEN_RE.match(token)
  if rat_match is not None:
    num = int(rat_match.group(1))
    den = int(rat_match.group(2))
    if den == 0:
      raise ValueError(f"Zero denominator in rational token: {token!r}")
    if token.startswith("-"):
      num = -num
    frac = fractions.Fraction(num, den)
  else:
    dec_match = _DECIMAL_TOKEN_RE.match(token)
    if dec_match is None:
      raise ValueError(f"Invalid coordinate token: {token!r}")
    exp_str = dec_match.group("exp")
    exp = 0
    if exp_str is not None:
      exp_digits = exp_str.lstrip("+-").lstrip("0")
      if len(exp_digits) > 4:
        raise ValueError(f"Excessive exponent in coordinate token: {token!r}")
      exp = int(exp_str)
      if abs(exp) > _MAX_DECIMAL_EXPONENT:
        raise ValueError(f"Excessive exponent in coordinate token: {token!r}")

    int_part = dec_match.group("int") or ""
    frac_part = dec_match.group("frac")
    if frac_part is None:
      frac_part = dec_match.group("frac_only") or ""
    frac_stripped = frac_part.rstrip("0")
    all_digits = (int_part + frac_stripped).lstrip("0")
    if not all_digits:
      return fractions.Fraction(0)

    # Write non-zero decimal as `M * 10^(-p)` where `M` has no trailing zero
    # (not divisible by 10). Then `gcd(M, 10^p)` is a pure power of 2 or 5, so
    # the reduced denominator is at least `2^p`; hence `p > 50` guarantees
    # `denom > 2^50`. Likewise `|x| >= 10^(len(sig) - 1 - p)`, so
    # `len(sig) - 1 - p > 100` guarantees `|x| > 10^100`.
    sig_digits = all_digits.rstrip("0")
    trailing_zeros = len(all_digits) - len(sig_digits)
    neg_pow10 = len(frac_stripped) - exp - trailing_zeros
    if neg_pow10 > _MAX_RATIONAL_DENOM_BITS:
      raise ValueError(f"Excessive denominator in decimal token: {token!r}")
    if len(sig_digits) - 1 - neg_pow10 > _MAX_COORD_MAGNITUDE_EXP10:
      raise ValueError(f"Coordinate magnitude exceeds bound: {token!r}")

    frac = fractions.Fraction(token)

  if frac.denominator > _MAX_RATIONAL_DENOM:
    raise ValueError(f"Excessive denominator in rational token: {token!r}")
  if abs(frac) > _MAX_COORD_MAGNITUDE:
    raise ValueError(f"Coordinate magnitude exceeds bound: {token!r}")
  return frac


def _as_strict_int(value: Any) -> int | None:
  """Converts a non-bool integral value to `int`, or `None` if invalid."""
  if isinstance(value, bool) or not isinstance(value, numbers.Integral):
    return None
  return int(value)


def _checked_index_seq(raw_seq: Any, upper_bound: int) -> list[int] | None:
  """Validates a sequence of integer indices in `[0, upper_bound)`, or `None`."""
  if isinstance(raw_seq, (str, bytes, Mapping)):
    return None
  try:
    iterator = iter(raw_seq)
  except TypeError:
    return None
  out: list[int] = []
  for raw_idx in iterator:
    idx = _as_strict_int(raw_idx)
    if idx is None or idx < 0 or idx >= upper_bound:
      return None
    out.append(idx)
  return out


def _parse_float_pair(item: Any) -> tuple[tuple[float, float] | None, str]:
  """Parses a 2D real coordinate into finite `(x, y)`, or `(None, kind)`."""
  if isinstance(item, (str, bytes, Mapping)):
    return None, "invalid"
  try:
    if len(item) != 2:
      return None, "invalid"
    raw_x, raw_y = item[0], item[1]
  except (TypeError, IndexError, KeyError):
    return None, "invalid"
  if (
      isinstance(raw_x, bool)
      or isinstance(raw_y, bool)
      or not isinstance(raw_x, numbers.Real)
      or not isinstance(raw_y, numbers.Real)
  ):
    return None, "invalid"
  try:
    fx = float(raw_x)
    fy = float(raw_y)
  except (TypeError, ValueError, OverflowError):
    return None, "non_finite"
  if not (math.isfinite(fx) and math.isfinite(fy)):
    return None, "non_finite"
  return (fx, fy), ""


def _parse_float_pairs(
    items: Any, field_tag: str
) -> tuple[list[tuple[float, float]] | None, str]:
  """Parses finite 2D float pairs or returns `(None, diagnostic_reason)`."""
  if isinstance(items, (str, bytes, Mapping)):
    return None, f"invalid_{field_tag}"
  try:
    iterator = iter(items)
  except TypeError:
    return None, f"invalid_{field_tag}"
  parsed: list[tuple[float, float]] = []
  for item in iterator:
    pair, err_kind = _parse_float_pair(item)
    if pair is None:
      return None, f"{err_kind}_{field_tag}"
    parsed.append(pair)
  return parsed, ""


def _octilinear_dir_code(dx: int, dy: int) -> int:
  """Returns CCW octilinear direction index `0..7` without floating point."""
  if dy == 0:
    return 0 if dx > 0 else 4
  if dx == 0:
    return 2 if dy > 0 else 6
  if dx == dy:
    return 1 if dx > 0 else 5
  if dx == -dy:
    return 7 if dx > 0 else 3
  raise ValueError(f"Non-octilinear vector ({dx}, {dy}).")


def _line_key_and_param(
    p1: Point2DInt, p2: Point2DInt
) -> tuple[int, int, int, int]:
  """Maps an octilinear segment to `(family, line_const, lo_param, hi_param)`."""
  x1, y1 = p1
  x2, y2 = p2
  dx = x2 - x1
  dy = y2 - y1
  if dy == 0:
    return (0, y1, min(x1, x2), max(x1, x2))
  if dx == 0:
    return (1, x1, min(y1, y2), max(y1, y2))
  if dx == dy:
    return (2, y1 - x1, min(x1, x2), max(x1, x2))
  if dx == -dy:
    return (3, y1 + x1, min(x1, x2), max(x1, x2))
  raise ValueError("Non-octilinear segment.")


def _point_on_line(family: int, c: int, param: int) -> Point2DInt:
  if family == 0:
    return (param, c)
  if family == 1:
    return (c, param)
  if family == 2:
    return (param, param + c)
  return (param, c - param)


def _param_on_line(family: int, pt: Point2DInt) -> int:
  return pt[1] if family == 1 else pt[0]


def _intersect_lines(
    fam1: int, c1: int, fam2: int, c2: int
) -> Point2DInt | None:
  """Intersects two octilinear lines from distinct families `(fam1 < fam2)`."""
  if fam1 == fam2:
    return None
  if fam1 > fam2:
    fam1, c1, fam2, c2 = fam2, c2, fam1, c1
  if fam1 == 0 and fam2 == 1:  # y = c1, x = c2
    return (c2, c1)
  if fam1 == 0 and fam2 == 2:  # y = c1, y - x = c2 -> x = c1 - c2
    return (c1 - c2, c1)
  if fam1 == 0 and fam2 == 3:  # y = c1, y + x = c2 -> x = c2 - c1
    return (c2 - c1, c1)
  if fam1 == 1 and fam2 == 2:  # x = c1, y - x = c2 -> y = c1 + c2
    return (c1, c1 + c2)
  if fam1 == 1 and fam2 == 3:  # x = c1, y + x = c2 -> y = c2 - c1
    return (c1, c2 - c1)
  # fam1 == 2 (y - x = c1) and fam2 == 3 (y + x = c2):
  # x = (c2 - c1) / 2, y = (c2 + c1) / 2 (exact int when coords are even).
  if (c2 - c1) % 2 != 0:
    return None
  return ((c2 - c1) // 2, (c2 + c1) // 2)


def _param_in_intervals(
    param: int, intervals: Sequence[tuple[int, int]]
) -> bool:
  for lo, hi in intervals:
    if lo <= param <= hi:
      return True
  return False


def _build_octilinear_arrangement(
    segments: Sequence[tuple[Point2DInt, Point2DInt, int]],
    *,
    check_mv_conflict: bool,
    contract_collinear_endpoints: bool = False,
) -> tuple[
    list[Point2DInt],
    list[tuple[int, int, int]],
    list[list[int]],
    str,
]:
  """Splits octilinear segments and extracts bounded CCW interior faces."""
  by_line: dict[tuple[int, int], list[tuple[int, int, int]]] = (
      collections.defaultdict(list)
  )
  for p1, p2, code in segments:
    if p1 == p2:
      continue
    fam, c, lo, hi = _line_key_and_param(p1, p2)
    by_line[(fam, c)].append((lo, hi, code))

  if not by_line:
    return [], [], [], "empty_arrangement"

  merged_spans: dict[tuple[int, int], list[tuple[int, int]]] = {}
  split_params: dict[tuple[int, int], set[int]] = {}
  for key, seg_list in by_line.items():
    pts: set[int] = set()
    raw_ivs = sorted((lo, hi) for lo, hi, _ in seg_list)
    merged: list[tuple[int, int]] = []
    for lo, hi in raw_ivs:
      if not contract_collinear_endpoints:
        pts.add(lo)
        pts.add(hi)
      if merged and lo <= merged[-1][1]:
        merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
      else:
        merged.append((lo, hi))
    if contract_collinear_endpoints:
      for m_lo, m_hi in merged:
        pts.add(m_lo)
        pts.add(m_hi)
    merged_spans[key] = merged
    split_params[key] = pts

  line_keys = sorted(by_line.keys())
  for i, k1 in enumerate(line_keys):
    fam1, c1 = k1
    spans1 = merged_spans[k1]
    for j in range(i + 1, len(line_keys)):
      k2 = line_keys[j]
      fam2, c2 = k2
      if fam1 == fam2:
        continue
      ipt = _intersect_lines(fam1, c1, fam2, c2)
      if ipt is None:
        continue
      p1 = _param_on_line(fam1, ipt)
      if not _param_in_intervals(p1, spans1):
        continue
      p2 = _param_on_line(fam2, ipt)
      if not _param_in_intervals(p2, merged_spans[k2]):
        continue
      split_params[k1].add(p1)
      split_params[k2].add(p2)

  vert_index: dict[Point2DInt, int] = {}
  verts: list[Point2DInt] = []

  def _get_vid(pt: Point2DInt) -> int:
    vid = vert_index.get(pt)
    if vid is None:
      vid = len(verts)
      vert_index[pt] = vid
      verts.append(pt)
    return vid

  elem_edges: list[tuple[int, int, int]] = []
  for key in line_keys:
    fam, c = key
    s_pts = sorted(split_params[key])
    seg_list = by_line[key]
    spans = merged_spans[key]
    for a, b in zip(s_pts, s_pts[1:]):
      if contract_collinear_endpoints:
        if not any(m_lo <= a and b <= m_hi for m_lo, m_hi in spans):
          continue
        chosen = 1
      else:
        codes = [code for lo, hi, code in seg_list if lo <= a and b <= hi]
        if not codes:
          continue
        non_flat = [cd for cd in codes if cd != 4]
        if check_mv_conflict and len(set(non_flat)) > 1:
          return [], [], [], "conflicting_collinear_crease_codes"
        chosen = non_flat[0] if non_flat else 4
      u = _get_vid(_point_on_line(fam, c, a))
      v = _get_vid(_point_on_line(fam, c, b))
      if u != v:
        elem_edges.append((u, v, chosen))

  adj: list[list[int]] = [[] for _ in range(len(verts))]
  for u, v, _ in elem_edges:
    adj[u].append(v)
    adj[v].append(u)

  pos_in_adj: dict[tuple[int, int], int] = {}
  for u, nbrs in enumerate(adj):
    ux, uy = verts[u]
    nbrs.sort(
        key=lambda w, ox=ux, oy=uy: _octilinear_dir_code(
            verts[w][0] - ox, verts[w][1] - oy
        )
    )
    for idx, w in enumerate(nbrs):
      pos_in_adj[(u, w)] = idx

  visited_he: set[tuple[int, int]] = set()
  faces: list[list[int]] = []
  for u, nbrs in enumerate(adj):
    for v in nbrs:
      if (u, v) in visited_he:
        continue
      cur = (u, v)
      cycle: list[int] = []
      while cur not in visited_he:
        visited_he.add(cur)
        cycle.append(cur[0])
        cu, cv = cur
        k = pos_in_adj[(cv, cu)]
        nbrs_cv = adj[cv]
        nxt_w = nbrs_cv[(k - 1) % len(nbrs_cv)]
        cur = (cv, nxt_w)
      if len(cycle) < 3:
        continue
      area2 = 0
      m = len(cycle)
      for i in range(m):
        x1, y1 = verts[cycle[i]]
        x2, y2 = verts[cycle[(i + 1) % m]]
        area2 += x1 * y2 - x2 * y1
      if area2 > 0:
        faces.append(cycle)

  return verts, elem_edges, faces, ""


def _canonical_face_key(
    face: Sequence[int], coords: Sequence[Point2DInt]
) -> tuple[Point2DInt, ...]:
  """Returns canonical CCW vertex tuple from min `(x, y)`, or `()`."""
  pts: list[Point2DInt] = []
  raw = [coords[vi] for vi in face]
  m = len(raw)
  if m < 3:
    return ()
  # Strip collinear intermediate vertices so polygon keys are split-invariant.
  for i in range(m):
    p_prev = raw[(i - 1) % m]
    p_curr = raw[i]
    p_next = raw[(i + 1) % m]
    dx1, dy1 = p_curr[0] - p_prev[0], p_curr[1] - p_prev[1]
    dx2, dy2 = p_next[0] - p_curr[0], p_next[1] - p_curr[1]
    if dx1 * dy2 - dy1 * dx2 != 0:
      pts.append(p_curr)
  if len(pts) < 3:
    return ()
  area2 = sum(
      pts[i][0] * pts[(i + 1) % len(pts)][1]
      - pts[(i + 1) % len(pts)][0] * pts[i][1]
      for i in range(len(pts))
  )
  if area2 == 0:
    return ()
  if area2 < 0:
    pts.reverse()
  min_idx = min(range(len(pts)), key=pts.__getitem__)
  return tuple(pts[min_idx:] + pts[:min_idx])


def _point_strictly_in_polygon(
    pt: Point2DInt, poly: Sequence[Point2DInt]
) -> bool:
  """Exact integer horizontal ray-cast test (`pt` assumed off boundary)."""
  px, py = pt
  inside = False
  m = len(poly)
  for i in range(m):
    x1, y1 = poly[i]
    x2, y2 = poly[(i + 1) % m]
    if (y1 > py) != (y2 > py):
      # Check whether edge crossing x_cross > px in exact integer arithmetic.
      lhs = (x2 - x1) * (py - y1)
      rhs = (px - x1) * (y2 - y1)
      if (lhs > rhs) if (y2 > y1) else (lhs < rhs):
        inside = not inside
  return inside


def _cell_interior_sample_point(
    cell_poly: Sequence[Point2DInt],
    all_verts_y_sorted: Sequence[int],
    all_segs: Sequence[tuple[Point2DInt, Point2DInt]],
) -> Point2DInt | None:
  """Finds an exact integer point strictly inside a simple polygon cell.

  Assumes all arrangement vertex coordinates are multiples of 4, so midpoints
  `y_mid` between consecutive distinct vertex `y`s are multiples of 2,
  octilinear segment crossings at `y_mid` are multiples of 2, and `x_mid` is an
  exact integer.

  Args:
    cell_poly: Ordered polygon vertices of the candidate cell.
    all_verts_y_sorted: Sorted unique y-coordinates of all arrangement vertices.
    all_segs: All elementary segments in the arrangement.

  Returns:
    An exact integer point strictly inside `cell_poly`, or `None` if none found.
  """
  y_min = min(p[1] for p in cell_poly)
  y_next = None
  for yv in all_verts_y_sorted:
    if yv > y_min:
      y_next = yv
      break
  if y_next is None:
    return None
  y_mid = (y_min + y_next) // 2

  # Find the leftmost two crossings of THIS cell's boundary with `y = y_mid`.
  cell_xs: list[int] = []
  m = len(cell_poly)
  for i in range(m):
    x1, y1 = cell_poly[i]
    x2, y2 = cell_poly[(i + 1) % m]
    if (y1 < y_mid < y2) or (y2 < y_mid < y1):
      xc = x1 + (x2 - x1) * (y_mid - y1) // (y2 - y1)
      cell_xs.append(xc)
  cell_xs.sort()
  if len(cell_xs) < 2:
    return None
  x_lo, x_hi = cell_xs[0], cell_xs[1]

  # Guard against any island/hole segment crossing `(x_lo, x_hi)` at `y_mid`.
  inner_xs = [x_lo, x_hi]
  for (sx1, sy1), (sx2, sy2) in all_segs:
    if (sy1 < y_mid < sy2) or (sy2 < y_mid < sy1):
      xc = sx1 + (sx2 - sx1) * (y_mid - sy1) // (sy2 - sy1)
      if x_lo < xc < x_hi:
        inner_xs.append(xc)
  inner_xs.sort()
  for xa, xb in zip(inner_xs, inner_xs[1:]):
    cand = ((xa + xb) // 2, y_mid)
    if _point_strictly_in_polygon(cand, cell_poly):
      return cand
  return None


def _reflect_across_octilinear_edge(
    p1: Point2DInt, p2: Point2DInt
) -> Transform2D:
  """Exact integer rigid reflection `(a11, a12, tx, a21, a22, ty)` across `p1->p2`."""
  x1, y1 = p1
  x2, y2 = p2
  dx = x2 - x1
  dy = y2 - y1
  if dy == 0:  # Horizontal line y = y1: (x, 2*y1 - y)
    return (1, 0, 0, 0, -1, 2 * y1)
  if dx == 0:  # Vertical line x = x1: (2*x1 - x, y)
    return (-1, 0, 2 * x1, 0, 1, 0)
  if dx == dy:  # +45 line y - x = c: (y - c, x + c)
    c = y1 - x1
    return (0, 1, -c, 1, 0, c)
  if dx == -dy:  # -45 line y + x = c: (c - y, c - x)
    c = y1 + x1
    return (0, -1, c, -1, 0, c)
  raise ValueError("Non-octilinear reflection axis.")


def _compose_transforms(
    t_outer: Transform2D, t_inner: Transform2D
) -> Transform2D:
  a11, a12, tx, a21, a22, ty = t_outer
  b11, b12, ux, b21, b22, uy = t_inner
  return (
      a11 * b11 + a12 * b21,
      a11 * b12 + a12 * b22,
      a11 * ux + a12 * uy + tx,
      a21 * b11 + a22 * b21,
      a21 * b12 + a22 * b22,
      a21 * ux + a22 * uy + ty,
  )


def _apply_transform(t: Transform2D, pt: Point2DInt) -> Point2DInt:
  a11, a12, tx, a21, a22, ty = t
  return (a11 * pt[0] + a12 * pt[1] + tx, a21 * pt[0] + a22 * pt[1] + ty)


@dataclasses.dataclass(frozen=True)
class _ParsedOctilinearCp:
  """Exact integer unfolded arrangement and manifold topology of a `.cp` doc."""

  u_scale: int
  u_verts: list[Point2DInt]
  u_edges: list[tuple[int, int, int]]
  u_faces: list[list[int]]
  edge_code: dict[tuple[int, int], int]
  edge_faces: dict[tuple[int, int], list[int]]
  face_dir_on_edge: dict[tuple[int, tuple[int, int]], tuple[int, int]]


def _parse_octilinear_unfolded_cp(
    cp_doc: str,
    *,
    flip_to_top_face_up: bool,
) -> tuple[_ParsedOctilinearCp | None, str]:
  """Parses a `.cp` string into an exact integer manifold unfolded mesh."""
  raw_segs: list[
      tuple[
          int,
          fractions.Fraction,
          fractions.Fraction,
          fractions.Fraction,
          fractions.Fraction,
      ]
  ] = []
  lcm_denom = 1
  for line_no, line in enumerate(cp_doc.splitlines(), start=1):
    parts = line.split()
    if len(parts) < 5 or parts[0].startswith("#"):
      continue
    try:
      raw_code = int(parts[0])
      x1 = _parse_rational(parts[1])
      y1 = _parse_rational(parts[2])
      x2 = _parse_rational(parts[3])
      y2 = _parse_rational(parts[4])
    except ValueError:
      return None, "cp_parse_error"
    if (x1, y1) == (x2, y2):
      return None, f"zero_length_segment_at_line_{line_no}"
    norm_code = raw_code if 1 <= raw_code <= 4 else 5
    if not flip_to_top_face_up:
      y1 = -y1 + 1
      y2 = -y2 + 1
      as_code = norm_code
    else:
      as_code = 3 if norm_code == 2 else (2 if norm_code == 3 else norm_code)
    dx = x2 - x1
    dy = y2 - y1
    if dx != 0 and dy != 0 and abs(dx) != abs(dy):
      return None, "non_octilinear_cp"
    for q in (x1, y1, x2, y2):
      lcm_denom = math.lcm(lcm_denom, q.denominator)
      if lcm_denom > _MAX_LCM_DENOM:
        return None, "excessive_common_denominator"
    raw_segs.append((as_code, x1, y1, x2, y2))

  if not raw_segs:
    return None, "empty_cp"

  # Scale by `4 * lcm_denom` so all unfolded split vertices and all folded
  # reflected vertices remain exact multiples of 2 (no Fractions needed).
  u_scale = 4 * lcm_denom
  scaled_u_segs: list[tuple[Point2DInt, Point2DInt, int]] = [
      (
          (int(x1 * u_scale), int(y1 * u_scale)),
          (int(x2 * u_scale), int(y2 * u_scale)),
          code,
      )
      for code, x1, y1, x2, y2 in raw_segs
  ]

  u_verts, u_edges, raw_u_faces, err = _build_octilinear_arrangement(
      scaled_u_segs, check_mv_conflict=True
  )
  if err or not raw_u_faces:
    return None, err or "no_unfolded_faces"

  edge_code: dict[tuple[int, int], int] = {}
  for u, v, cd in u_edges:
    edge_code[(u, v) if u < v else (v, u)] = cd

  # Filter interior cutout holes bounded solely by boundary (`code == 1`) edges.
  if len(raw_u_faces) > 1:
    u_faces = [
        cyc
        for cyc in raw_u_faces
        if any(
            edge_code[
                (cyc[i], cyc[(i + 1) % len(cyc)])
                if cyc[i] < cyc[(i + 1) % len(cyc)]
                else (cyc[(i + 1) % len(cyc)], cyc[i])
            ]
            != 1
            for i in range(len(cyc))
        )
    ]
  else:
    u_faces = raw_u_faces
  if not u_faces:
    return None, "no_unfolded_faces"

  edge_faces: dict[tuple[int, int], list[int]] = collections.defaultdict(list)
  face_dir_on_edge: dict[tuple[int, tuple[int, int]], tuple[int, int]] = {}
  for u_fi, cyc in enumerate(u_faces):
    m = len(cyc)
    for i in range(m):
      a, b = cyc[i], cyc[(i + 1) % m]
      ek = (a, b) if a < b else (b, a)
      edge_faces[ek].append(u_fi)
      face_dir_on_edge[(u_fi, ek)] = (a, b)

  # Enforce 2-manifold edge incidence on every arrangement edge (rejecting
  # orphan edges `len == 0` and dangling/slit edges `[fi, fi]`).
  for ek_u in edge_code:
    faces_inc = edge_faces.get(ek_u, [])
    if len(faces_inc) not in (1, 2):
      return None, "non_manifold_or_orphan_edge"
    if len(faces_inc) == 2 and faces_inc[0] == faces_inc[1]:
      return None, "slit_or_dangling_edge"

  return (
      _ParsedOctilinearCp(
          u_scale=u_scale,
          u_verts=u_verts,
          u_edges=u_edges,
          u_faces=u_faces,
          edge_code=edge_code,
          edge_faces=edge_faces,
          face_dir_on_edge=face_dir_on_edge,
      ),
      "",
  )


def _fold_octilinear_mesh(
    parsed: _ParsedOctilinearCp,
    *,
    root_face: int,
    root_flipped: bool,
) -> tuple[
    tuple[list[Transform2D], list[bool], list[Point2DInt]] | None,
    str,
]:
  """Folds `parsed` by exact reflections and checks holonomy + closure."""
  num_f = len(parsed.u_faces)
  face_tf: list[Transform2D | None] = [None] * num_f
  face_flipped: list[bool] = [False] * num_f
  face_tf[root_face] = (1, 0, 0, 0, 1, 0)
  face_flipped[root_face] = root_flipped
  bfs_q: collections.deque[int] = collections.deque([root_face])

  while bfs_q:
    fi = bfs_q.popleft()
    tf = face_tf[fi]
    assert tf is not None
    cyc = parsed.u_faces[fi]
    m = len(cyc)
    for i in range(m):
      a, b = cyc[i], cyc[(i + 1) % m]
      ek = (a, b) if a < b else (b, a)
      faces_inc = parsed.edge_faces[ek]
      if len(faces_inc) == 1:
        continue
      cd = parsed.edge_code[ek]
      if cd in (2, 3, 5):
        refl_tf = _compose_transforms(
            tf,
            _reflect_across_octilinear_edge(
                parsed.u_verts[a], parsed.u_verts[b]
            ),
        )
        nxt_flip = not face_flipped[fi]
      else:
        # Both `F` (`4`) and 2-face internal `B` (`1`) remain coplanar.
        refl_tf = tf
        nxt_flip = face_flipped[fi]
      for nb in faces_inc:
        if nb == fi:
          continue
        if face_tf[nb] is None:
          face_tf[nb] = refl_tf
          face_flipped[nb] = nxt_flip
          bfs_q.append(nb)
        else:
          if face_flipped[nb] != nxt_flip:
            return None, "orientation_parity_conflict"
          if face_tf[nb] != refl_tf:
            return None, "transform_holonomy_conflict"

  if any(tf is None for tf in face_tf):
    return None, "disconnected_unfolded_faces"

  folded_u_verts: list[Point2DInt | None] = [None] * len(parsed.u_verts)
  for u_fi, cyc in enumerate(parsed.u_faces):
    tf = face_tf[u_fi]
    assert tf is not None
    for vi in cyc:
      fpt = _apply_transform(tf, parsed.u_verts[vi])
      prev = folded_u_verts[vi]
      if prev is None:
        folded_u_verts[vi] = fpt
      elif prev != fpt:
        return None, "folded_vertex_closure_mismatch"

  f_verts = [pt for pt in folded_u_verts if pt is not None]
  if len(f_verts) != len(parsed.u_verts):
    return None, "unvisited_vertex"
  tfs_checked = [tf for tf in face_tf if tf is not None]
  return (tfs_checked, face_flipped, f_verts), ""


def _verify_core(
    cp_doc: str,
    *,
    solver_unfolded_vertices: Sequence[tuple[float, float]],
    solver_face_vertices: Sequence[Sequence[int]],
    solver_face_orientations: Sequence[bool | int],
    solver_folded_vertices: Sequence[tuple[float, float]] | None,
    solver_cell_points: Sequence[tuple[float, float]] | None,
    solver_cell_vertices: Sequence[Sequence[int]] | None,
    solver_cell_faces_ordered: Sequence[Sequence[int]] | None,
    solver_face_orders: Sequence[int] | None,
    flip_to_top_face_up: bool,
) -> CertificateCheckResult:
  """Core exact verification shared by C++ cell-stack and `face_orders` modes."""
  parsed, parse_err = _parse_octilinear_unfolded_cp(
      cp_doc, flip_to_top_face_up=flip_to_top_face_up
  )
  if parsed is None:
    return CertificateCheckResult(False, parse_err)

  u_scale = parsed.u_scale
  u_verts = parsed.u_verts
  u_edges = parsed.u_edges
  u_faces = parsed.u_faces
  edge_code = parsed.edge_code
  edge_faces = parsed.edge_faces
  face_dir_on_edge = parsed.face_dir_on_edge

  num_f = len(u_faces)
  if (
      len(solver_face_vertices) != num_f
      or len(solver_face_orientations) != num_f
  ):
    return CertificateCheckResult(
        False,
        f"face_count_mismatch:exact={num_f},solver={len(solver_face_vertices)}",
    )

  # Bijectively match solver unfolded faces `s_fi` -> exact `u_fi`.
  u_key_to_fi: dict[tuple[Point2DInt, ...], int] = {}
  for u_fi, cyc in enumerate(u_faces):
    u_key_to_fi[_canonical_face_key(cyc, u_verts)] = u_fi

  s_u_pairs, u_pair_err = _parse_float_pairs(
      solver_unfolded_vertices, "solver_unfolded_vertex"
  )
  if s_u_pairs is None:
    return CertificateCheckResult(False, u_pair_err)

  tol_u = 1e-3 * max(1, u_scale)
  s_u_verts: list[Point2DInt] = []
  for vx, vy in s_u_pairs:
    scaled_x = vx * u_scale
    scaled_y = vy * u_scale
    if not (math.isfinite(scaled_x) and math.isfinite(scaled_y)):
      return CertificateCheckResult(False, "non_finite_solver_unfolded_vertex")
    sx = round(scaled_x)
    sy = round(scaled_y)
    if abs(scaled_x - sx) > tol_u or abs(scaled_y - sy) > tol_u:
      return CertificateCheckResult(False, "solver_unfolded_vertex_mismatch")
    s_u_verts.append((sx, sy))

  num_s_verts = len(s_u_verts)
  s_to_u: list[int] = [-1] * num_f
  u_to_s: list[int] = [-1] * num_f
  for s_fi, s_cyc in enumerate(solver_face_vertices):
    checked_cyc = _checked_index_seq(s_cyc, num_s_verts)
    if checked_cyc is None:
      return CertificateCheckResult(
          False, f"solver_face_vertex_index_out_of_range_{s_fi}"
      )
    if len(checked_cyc) < 3:
      return CertificateCheckResult(False, "degenerate_solver_face")
    key = _canonical_face_key(checked_cyc, s_u_verts)
    if not key:
      return CertificateCheckResult(False, "degenerate_solver_face")
    u_fi = u_key_to_fi.get(key)
    if u_fi is None or u_to_s[u_fi] != -1:
      return CertificateCheckResult(False, f"face_polygon_mismatch_at_{s_fi}")
    s_to_u[s_fi] = u_fi
    u_to_s[u_fi] = s_fi

  # Fold exact faces from `u_root = s_to_u[0]` using the solver's root parity.
  folded_res, fold_err = _fold_octilinear_mesh(
      parsed,
      root_face=s_to_u[0],
      root_flipped=bool(solver_face_orientations[0]),
  )
  if folded_res is None:
    return CertificateCheckResult(False, fold_err)
  face_tf, face_flipped, folded_u_verts = folded_res

  for u_fi in range(num_f):
    s_fi = u_to_s[u_fi]
    if bool(solver_face_orientations[s_fi]) != face_flipped[u_fi]:
      return CertificateCheckResult(
          False, f"face_orientation_mismatch_at_{s_fi}"
      )

  if solver_folded_vertices is not None:
    s_f_pairs, f_pair_err = _parse_float_pairs(
        solver_folded_vertices, "solver_folded_vertex"
    )
    if s_f_pairs is None:
      return CertificateCheckResult(False, f_pair_err)
    if len(s_f_pairs) != len(s_u_verts):
      return CertificateCheckResult(False, "folded_vertex_count_mismatch")
    for sfx, sfy in s_f_pairs:
      if not (math.isfinite(sfx * u_scale) and math.isfinite(sfy * u_scale)):
        return CertificateCheckResult(False, "non_finite_solver_folded_vertex")
    u_pt_to_vid = {pt: idx for idx, pt in enumerate(u_verts)}
    for s_vi, up in enumerate(s_u_verts):
      u_vi = u_pt_to_vid.get(up)
      if u_vi is not None:
        efx, efy = folded_u_verts[u_vi]
        sfx, sfy = s_f_pairs[s_vi]
        if abs(sfx * u_scale - efx) > tol_u or abs(sfy * u_scale - efy) > tol_u:
          return CertificateCheckResult(False, "solver_folded_vertex_mismatch")

  # Build exact folded arrangement, scaled by 4 so cell sample points are ints.
  f_scale_mult = 4
  f_scale_total = u_scale * f_scale_mult
  f_verts_scaled: list[Point2DInt] = [
      (pt[0] * f_scale_mult, pt[1] * f_scale_mult) for pt in folded_u_verts
  ]
  folded_input_segs: list[tuple[Point2DInt, Point2DInt, int]] = [
      (f_verts_scaled[u], f_verts_scaled[v], 1) for u, v, _ in u_edges
  ]
  c_pts, c_segs, c_polys, f_err = _build_octilinear_arrangement(
      folded_input_segs, check_mv_conflict=False
  )
  if f_err or not c_polys:
    return CertificateCheckResult(False, f_err or "empty_folded_arrangement")

  face_folded_polys: list[list[Point2DInt]] = []
  for u_fi, cyc in enumerate(u_faces):
    poly = [f_verts_scaled[vi] for vi in cyc]
    # Ensure CCW in folded plane for point-in-polygon tests.
    area2 = sum(
        poly[i][0] * poly[(i + 1) % len(poly)][1]
        - poly[(i + 1) % len(poly)][0] * poly[i][1]
        for i in range(len(poly))
    )
    if area2 < 0:
      poly.reverse()
    face_folded_polys.append(poly)

  all_cy_sorted = sorted({p[1] for p in c_pts})
  raw_c_segs = [(c_pts[u], c_pts[v]) for u, v, _ in c_segs]

  cell_samples: list[Point2DInt] = []
  cell_covering: list[list[int]] = []
  for ci, c_cyc in enumerate(c_polys):
    c_poly = [c_pts[vi] for vi in c_cyc]
    spt = _cell_interior_sample_point(c_poly, all_cy_sorted, raw_c_segs)
    if spt is None:
      return CertificateCheckResult(False, f"cell_sample_failed_{ci}")
    cell_samples.append(spt)
    cov = [
        u_fi
        for u_fi, fpoly in enumerate(face_folded_polys)
        if _point_strictly_in_polygon(spt, fpoly)
    ]
    cell_covering.append(cov)

  # Map solver per-cell stacks or `face_orders` onto exact cells `0..n_cells-1`.
  exact_cell_stacks: list[list[int]] = [[] for _ in range(len(c_polys))]
  if solver_cell_faces_ordered is not None:
    if solver_cell_points is None or solver_cell_vertices is None:
      return CertificateCheckResult(False, "missing_solver_cell_geometry")
    if len(solver_cell_faces_ordered) != len(solver_cell_vertices):
      return CertificateCheckResult(False, "cell_geometry_length_mismatch")
    s_c_pairs, c_pair_err = _parse_float_pairs(
        solver_cell_points, "solver_cell_point"
    )
    if s_c_pairs is None:
      return CertificateCheckResult(False, c_pair_err)
    s_c_pts: list[Point2DInt] = []
    for px, py in s_c_pairs:
      scaled_x = px * f_scale_total
      scaled_y = py * f_scale_total
      if not (math.isfinite(scaled_x) and math.isfinite(scaled_y)):
        return CertificateCheckResult(False, "non_finite_solver_cell_point")
      s_c_pts.append((round(scaled_x), round(scaled_y)))
    num_s_c_pts = len(s_c_pts)
    s_cell_by_key: dict[tuple[Point2DInt, ...], int] = {}
    s_nonempty_polys: list[tuple[int, list[Point2DInt]]] = []
    checked_cell_stacks: list[list[int]] = []
    for s_ci, s_cyc in enumerate(solver_cell_vertices):
      checked_stack = _checked_index_seq(solver_cell_faces_ordered[s_ci], num_f)
      if checked_stack is None:
        return CertificateCheckResult(
            False, f"cell_face_index_out_of_range_{s_ci}"
        )
      checked_cell_stacks.append(checked_stack)
      if not checked_stack:
        continue
      checked_cyc = _checked_index_seq(s_cyc, num_s_c_pts)
      if checked_cyc is None:
        return CertificateCheckResult(
            False, f"cell_vertex_index_out_of_range_{s_ci}"
        )
      if len(checked_cyc) < 3:
        return CertificateCheckResult(False, f"degenerate_solver_cell_{s_ci}")
      ckey = _canonical_face_key(checked_cyc, s_c_pts)
      if not ckey:
        return CertificateCheckResult(False, f"degenerate_solver_cell_{s_ci}")
      s_cell_by_key[ckey] = s_ci
      spoly = [s_c_pts[vi] for vi in checked_cyc]
      a2 = sum(
          spoly[i][0] * spoly[(i + 1) % len(spoly)][1]
          - spoly[(i + 1) % len(spoly)][0] * spoly[i][1]
          for i in range(len(spoly))
      )
      if a2 < 0:
        spoly.reverse()
      s_nonempty_polys.append((s_ci, spoly))

    for ci, cov in enumerate(cell_covering):
      if not cov:
        continue
      ckey = _canonical_face_key(c_polys[ci], c_pts)
      s_ci = s_cell_by_key.get(ckey)
      if s_ci is None:
        spt = cell_samples[ci]
        matches = [
            idx
            for idx, spoly in s_nonempty_polys
            if _point_strictly_in_polygon(spt, spoly)
        ]
        if len(matches) != 1:
          return CertificateCheckResult(False, f"unmatched_folded_cell_{ci}")
        s_ci = matches[0]
      exact_cell_stacks[ci] = [
          s_to_u[s_fi] for s_fi in checked_cell_stacks[s_ci]
      ]
  elif solver_face_orders is not None:
    if len(solver_face_orders) != num_f:
      return CertificateCheckResult(False, "face_orders_length_mismatch")
    checked_orders: list[int] = []
    for raw_rank in solver_face_orders:
      rank = _as_strict_int(raw_rank)
      if rank is None:
        return CertificateCheckResult(False, "invalid_solver_face_orders")
      checked_orders.append(rank)
    for ci, cov in enumerate(cell_covering):
      if not cov:
        continue
      ranks = [checked_orders[u_to_s[u_fi]] for u_fi in cov]
      if len(set(ranks)) != len(cov):
        return CertificateCheckResult(
            False, f"duplicate_face_order_rank_in_cell_{ci}"
        )
      exact_cell_stacks[ci] = [
          u_fi for _, u_fi in sorted(zip(ranks, cov), key=lambda t: t[0])
      ]
  else:
    return CertificateCheckResult(False, "no_order_certificate_supplied")

  # 1. Per-cell coverage, per-cell acyclicity (strict total order), and
  #    cross-cell pairwise relative-order consistency.
  cell_pos: list[dict[int, int]] = []
  pair_order: dict[tuple[int, int], bool] = {}  # (a<b) -> True iff a below b
  nonempty_cells = 0
  for ci, cov in enumerate(cell_covering):
    st = exact_cell_stacks[ci]
    if not cov:
      if st:
        return CertificateCheckResult(
            False, f"nonempty_stack_on_empty_cell_{ci}"
        )
      cell_pos.append({})
      continue
    nonempty_cells += 1
    if len(st) != len(cov) or len(set(st)) != len(cov) or set(st) != set(cov):
      return CertificateCheckResult(False, f"cell_stack_coverage_mismatch_{ci}")
    pos = {u_fi: rk for rk, u_fi in enumerate(st)}
    cell_pos.append(pos)
    for i in range(len(st)):
      fi = st[i]
      for j in range(i + 1, len(st)):
        fj = st[j]  # fi is strictly below fj in cell `ci`.
        pk = (fi, fj) if fi < fj else (fj, fi)
        low_is_first = fi < fj
        prev = pair_order.get(pk)
        if prev is None:
          pair_order[pk] = low_is_first
        elif prev != low_is_first:
          return CertificateCheckResult(
              False, f"cross_cell_order_cycle_on_faces_{pk[0]}_{pk[1]}"
          )

  # Map each elementary folded segment `(cu, cv)` (`cu < cv`) to
  # `(c_left, c_right)` relative to directed segment `c_pts[cu] -> c_pts[cv]`.
  seg_left_cell: dict[tuple[int, int], int] = {}
  seg_right_cell: dict[tuple[int, int], int] = {}
  for ci, c_cyc in enumerate(c_polys):
    m = len(c_cyc)
    for i in range(m):
      a, b = c_cyc[i], c_cyc[(i + 1) % m]
      # Interior of CCW `c_polys[ci]` lies to the LEFT of `a -> b`.
      if a < b:
        seg_left_cell[(a, b)] = ci
      else:
        seg_right_cell[(b, a)] = ci

  # Index elementary folded segments by line `(fam, c)` for edge projection.
  f_segs_by_line: dict[
      tuple[int, int], list[tuple[int, int, tuple[int, int]]]
  ] = collections.defaultdict(list)
  for cu, cv, _ in c_segs:
    ek = (cu, cv) if cu < cv else (cv, cu)
    fam, c, lo, hi = _line_key_and_param(c_pts[ek[0]], c_pts[ek[1]])
    f_segs_by_line[(fam, c)].append((lo, hi, ek))
  for lst in f_segs_by_line.values():
    lst.sort()

  # 2. M/V consistency & classify tacos / F-crease tortillas per elementary seg.
  tacos_left: dict[tuple[int, int], list[tuple[int, int]]] = (
      collections.defaultdict(list)
  )
  tacos_right: dict[tuple[int, int], list[tuple[int, int]]] = (
      collections.defaultdict(list)
  )
  flat_tortillas: dict[tuple[int, int], list[tuple[int, int]]] = (
      collections.defaultdict(list)
  )
  checked_mv = 0

  for ek_u, faces_inc in edge_faces.items():
    if len(faces_inc) != 2:
      continue
    cd = edge_code[ek_u]
    fa, fb = (faces_inc[0], faces_inc[1])
    if fb < fa:
      fa, fb = fb, fa

    if cd in (2, 3):
      checked_mv += 1
      fa_flip = face_flipped[fa]
      # Matches `RunStage1CreasesAndTaco` + `SortFacesInCellsByOrder`:
      # `(!fa_flip && cd==2) || (fa_flip && cd==3)` => `fa` below `fb`;
      # otherwise `fb` below `fa`.
      fa_below_fb = (not fa_flip and cd == 2) or (fa_flip and cd == 3)
      pk = (fa, fb)
      if pk not in pair_order:
        return CertificateCheckResult(
            False, f"mv_edge_faces_never_overlap_{fa}_{fb}"
        )
      if pair_order[pk] != fa_below_fb:
        return CertificateCheckResult(
            False, f"mv_order_violation_on_faces_{fa}_{fb}"
        )

    # Determine local folded normal for `fa` along unfolded edge `ek_u`.
    da, db = face_dir_on_edge[(fa, ek_u)]
    # Inward normal to CCW `da -> db` in unfolded space: `(-(yb-ya), xb-xa)`.
    n_ux = -(u_verts[db][1] - u_verts[da][1])
    n_uy = u_verts[db][0] - u_verts[da][0]
    tfa = face_tf[fa]
    n_fx = tfa[0] * n_ux + tfa[1] * n_uy
    n_fy = tfa[3] * n_ux + tfa[4] * n_uy

    fp1 = f_verts_scaled[ek_u[0]]
    fp2 = f_verts_scaled[ek_u[1]]
    fam, c, elo, ehi = _line_key_and_param(fp1, fp2)
    for slo, shi, f_ek in f_segs_by_line[(fam, c)]:
      if shi <= elo:
        continue
      if slo >= ehi:
        break
      if elo <= slo and shi <= ehi:
        p_pt = c_pts[f_ek[0]]
        q_pt = c_pts[f_ek[1]]
        sdx = q_pt[0] - p_pt[0]
        sdy = q_pt[1] - p_pt[1]
        cross_side = sdx * n_fy - sdy * n_fx
        if cross_side == 0:
          return CertificateCheckResult(False, "zero_folded_normal_cross")
        if cd in (2, 3, 5):
          if cross_side > 0:
            tacos_left[f_ek].append((fa, fb))
          else:
            tacos_right[f_ek].append((fa, fb))
        elif cd in (1, 4):
          if cross_side > 0:
            flat_tortillas[f_ek].append((fa, fb))
          else:
            flat_tortillas[f_ek].append((fb, fa))

  # 3. Check Taco-Taco, Taco-Tortilla, and Tortilla-Tortilla on all folded segs.
  tt_count = 0
  t_tort_count = 0
  tort_tort_count = 0

  all_f_eks = [(cu, cv) if cu < cv else (cv, cu) for cu, cv, _ in c_segs]
  for f_ek in all_f_eks:
    c_l = seg_left_cell.get(f_ek)
    c_r = seg_right_cell.get(f_ek)
    pos_l = cell_pos[c_l] if c_l is not None else {}
    pos_r = cell_pos[c_r] if c_r is not None else {}

    # Gather all tortillas `(f_left, f_right)` crossing `f_ek`:
    # (a) explicit `F` creases + (b) uncreased faces spanning `c_l` and `c_r`.
    creased_left_faces: set[int] = set()
    creased_right_faces: set[int] = set()
    for a, b in tacos_left.get(f_ek, ()):
      creased_left_faces.update((a, b))
    for a, b in tacos_right.get(f_ek, ()):
      creased_right_faces.update((a, b))
    tortillas: list[tuple[int, int]] = []
    for fl, fr in flat_tortillas.get(f_ek, ()):
      creased_left_faces.add(fl)
      creased_right_faces.add(fr)
      tortillas.append((fl, fr))
    if c_l is not None and c_r is not None:
      for fc in cell_covering[c_l]:
        if (
            fc in pos_r
            and fc not in creased_left_faces
            and fc not in creased_right_faces
        ):
          tortillas.append((fc, fc))

    # Check Taco-Taco and Taco-Tortilla on both left and right sides of `f_ek`.
    for side_idx, (tacos, pos_s) in enumerate(
        ((tacos_left.get(f_ek, ()), pos_l), (tacos_right.get(f_ek, ()), pos_r))
    ):
      if not tacos:
        continue
      intervals: list[tuple[int, int]] = []
      for fa, fb in tacos:
        if fa not in pos_s or fb not in pos_s:
          return CertificateCheckResult(
              False, f"taco_face_missing_from_adjacent_cell_{fa}_{fb}"
          )
        ra, rb = pos_s[fa], pos_s[fb]
        intervals.append((min(ra, rb), max(ra, rb)))

      for i in range(len(intervals)):
        lo1, hi1 = intervals[i]
        for j in range(i + 1, len(intervals)):
          lo2, hi2 = intervals[j]
          tt_count += 1
          if (lo1 < lo2 < hi1 < hi2) or (lo2 < lo1 < hi2 < hi1):
            return CertificateCheckResult(False, "taco_taco_crossing")
        for tl, tr in tortillas:
          tf_side = tl if side_idx == 0 else tr
          if tf_side not in pos_s:
            return CertificateCheckResult(
                False, f"tortilla_face_missing_from_cell_{tf_side}"
            )
          rt = pos_s[tf_side]
          t_tort_count += 1
          if lo1 < rt < hi1:
            return CertificateCheckResult(False, "taco_tortilla_crossing")

    # Check Tortilla-Tortilla across `f_ek`.
    if len(tortillas) >= 2:
      if not pos_l or not pos_r:
        return CertificateCheckResult(False, "tortilla_missing_side_cell")
      for i in range(len(tortillas)):
        al, ar = tortillas[i]
        if al not in pos_l or ar not in pos_r:
          return CertificateCheckResult(False, "tortilla_face_not_in_side_cell")
        for j in range(i + 1, len(tortillas)):
          bl, br = tortillas[j]
          if bl not in pos_l or br not in pos_r:
            return CertificateCheckResult(
                False, "tortilla_face_not_in_side_cell"
            )
          if al == bl or ar == br:
            return CertificateCheckResult(False, "degenerate_shared_tortilla")
          tort_tort_count += 1
          if (pos_l[al] < pos_l[bl]) != (pos_r[ar] < pos_r[br]):
            return CertificateCheckResult(False, "tortilla_tortilla_crossing")

  return CertificateCheckResult(
      ok=True,
      reason="",
      num_faces=num_f,
      num_nonempty_cells=nonempty_cells,
      checked_mv_edges=checked_mv,
      checked_taco_taco_pairs=tt_count,
      checked_taco_tortilla_pairs=t_tort_count,
      checked_tortilla_tortilla_pairs=tort_tort_count,
  )


def verify_cpp_sat_certificate(
    cp_doc: str,
    cpp_res: Any,
    *,
    flip_to_top_face_up: bool = True,
) -> CertificateCheckResult:
  """Verifies a C++ `_fast_flat_folder.SolveResult` SAT certificate."""
  if not getattr(cpp_res, "flat_foldable", False):
    return CertificateCheckResult(False, "result_not_sat")
  vertex_coords = getattr(cpp_res, "vertex_coords", None)
  face_vertices = getattr(cpp_res, "face_vertices", None)
  face_orientations = getattr(cpp_res, "face_orientations", None)
  folded_vertices = getattr(cpp_res, "folded_vertices", None)
  cell_points = getattr(cpp_res, "cell_points", None)
  cell_vertices = getattr(cpp_res, "cell_vertices", None)
  cell_faces_ordered = getattr(cpp_res, "cell_faces_ordered", None)
  if (
      vertex_coords is None
      or face_vertices is None
      or face_orientations is None
      or folded_vertices is None
      or cell_points is None
      or cell_vertices is None
      or cell_faces_ordered is None
  ):
    return CertificateCheckResult(False, "missing_solver_result_fields")
  return _verify_core(
      cp_doc,
      solver_unfolded_vertices=vertex_coords,
      solver_face_vertices=face_vertices,
      solver_face_orientations=face_orientations,
      solver_folded_vertices=folded_vertices,
      solver_cell_points=cell_points,
      solver_cell_vertices=cell_vertices,
      solver_cell_faces_ordered=cell_faces_ordered,
      solver_face_orders=None,
      flip_to_top_face_up=flip_to_top_face_up,
  )


def verify_reference_sat_certificate(
    cp_doc: str,
    *,
    unfolded_vertices: Sequence[Sequence[float]],
    face_vertices: Sequence[Sequence[int]],
    folded_state_fold: Mapping[str, Any],
    flip_to_top_face_up: bool = True,
) -> CertificateCheckResult:
  """Verifies a Python `fold_cp` `folded_state_fold` dictionary on SAT."""
  if not isinstance(folded_state_fold, Mapping):
    return CertificateCheckResult(False, "invalid_folded_state_fold")
  face_orders_outer = folded_state_fold.get("faceOrders") or []
  if not face_orders_outer or isinstance(face_orders_outer[0], (str, bytes)):
    return CertificateCheckResult(False, "missing_faceOrders")
  face_orientations = folded_state_fold.get("faceOrientations")
  if face_orientations is None:
    return CertificateCheckResult(False, "missing_faceOrientations")
  parsed_unfolded, err = _parse_float_pairs(
      unfolded_vertices, "solver_unfolded_vertex"
  )
  if parsed_unfolded is None:
    return CertificateCheckResult(False, err)
  folded_verts = folded_state_fold.get("vertices_coords")
  parsed_folded: list[tuple[float, float]] | None = None
  if folded_verts is not None:
    parsed_folded, err = _parse_float_pairs(
        folded_verts, "solver_folded_vertex"
    )
    if parsed_folded is None:
      return CertificateCheckResult(False, err)
  return _verify_core(
      cp_doc,
      solver_unfolded_vertices=parsed_unfolded,
      solver_face_vertices=face_vertices,
      solver_face_orientations=face_orientations,
      solver_folded_vertices=parsed_folded,
      solver_cell_points=None,
      solver_cell_vertices=None,
      solver_cell_faces_ordered=None,
      solver_face_orders=face_orders_outer[0],
      flip_to_top_face_up=flip_to_top_face_up,
  )


def verify_shaping_record_sat_certificate(
    cp_doc: str,
    *,
    unfolded_vertices: Sequence[Sequence[float]],
    face_vertices: Sequence[Sequence[int]],
    face_top_up: Sequence[bool | int],
    face_orders: Sequence[int],
    flip_to_top_face_up: bool = True,
) -> CertificateCheckResult:
  """Verifies a recorded shaping `solve_flat_folded_state` SAT trace entry."""
  parsed_unfolded, err = _parse_float_pairs(
      unfolded_vertices, "solver_unfolded_vertex"
  )
  if parsed_unfolded is None:
    return CertificateCheckResult(False, err)
  return _verify_core(
      cp_doc,
      solver_unfolded_vertices=parsed_unfolded,
      solver_face_vertices=face_vertices,
      solver_face_orientations=[not bool(up) for up in face_top_up],
      solver_folded_vertices=None,
      solver_cell_points=None,
      solver_cell_vertices=None,
      solver_cell_faces_ordered=None,
      solver_face_orders=face_orders,
      flip_to_top_face_up=flip_to_top_face_up,
  )


def audit_exact_folded_arrangement(
    cp_doc: str,
    *,
    flip_to_top_face_up: bool = True,
) -> dict[str, Any]:
  """Computes exact integer folded arrangement stats & 45-deg overlap diagnostics."""
  parsed, parse_err = _parse_octilinear_unfolded_cp(
      cp_doc, flip_to_top_face_up=flip_to_top_face_up
  )
  if parsed is None:
    return {"locally_closed": False, "reason": parse_err}

  folded_res, fold_err = _fold_octilinear_mesh(
      parsed, root_face=0, root_flipped=False
  )
  if folded_res is None:
    return {"locally_closed": False, "reason": fold_err}
  _, _, f_verts = folded_res

  # Inspect folded 45-degree (`family in (2, 3)`) collinear overlaps.
  diag_all: dict[tuple[int, int], list[tuple[int, int]]] = (
      collections.defaultdict(list)
  )
  diag_dedup_set: dict[tuple[int, int], set[tuple[int, int]]] = (
      collections.defaultdict(set)
  )
  for u, v, _ in parsed.u_edges:
    p1, p2 = f_verts[u], f_verts[v]
    if p1 == p2:
      continue
    fam, c, lo, hi = _line_key_and_param(p1, p2)
    if fam in (2, 3):
      diag_all[(fam, c)].append((lo, hi))
      diag_dedup_set[(fam, c)].add((lo, hi))

  def _count_overlapping_lines(
      line_map: Mapping[tuple[int, int], Sequence[tuple[int, int]]],
  ) -> int:
    count = 0
    for ivs in line_map.values():
      if len(ivs) < 2:
        continue
      ordered = sorted(ivs)
      max_hi = ordered[0][1]
      for lo, hi in ordered[1:]:
        if lo < max_hi:
          count += 1
          break
        if hi > max_hi:
          max_hi = hi
    return count

  coll45_any_lines = _count_overlapping_lines(diag_all)
  coll45_distinct_lines = _count_overlapping_lines(
      {k: list(v) for k, v in diag_dedup_set.items()}
  )

  folded_input_segs: list[tuple[Point2DInt, Point2DInt, int]] = [
      (f_verts[u], f_verts[v], 1) for u, v, _ in parsed.u_edges
  ]
  c_pts, c_segs, c_polys, f_err = _build_octilinear_arrangement(
      folded_input_segs,
      check_mv_conflict=False,
      contract_collinear_endpoints=True,
  )
  if f_err or not c_polys:
    return {"locally_closed": False, "reason": f_err or "empty_folded"}

  return {
      "locally_closed": True,
      "exact_folded_v": len(c_pts),
      "exact_folded_e": len(c_segs),
      "exact_folded_ccw_cells": len(c_polys),
      "has_collinear_45_any": coll45_any_lines > 0,
      "collinear_45_any_lines": coll45_any_lines,
      "has_collinear_45_distinct_endpoints": coll45_distinct_lines > 0,
      "collinear_45_distinct_endpoint_lines": coll45_distinct_lines,
  }
