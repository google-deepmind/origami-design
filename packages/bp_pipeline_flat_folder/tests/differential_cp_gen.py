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

"""Deterministic lattice crease-pattern & near-miss generator for parity tests.

Generates diverse octilinear lattice `.cp` instances and near-miss mutations to
exercise every branch of global flat-foldability checking against reference
`bp_pipeline.third_party.flat_folder.fold_cp`:
  1. Multi-step flat-folded lattice sequences (axis-aligned + 45-degree folds,
     accordion stacks, intersecting diagonal/axial folds).
  2. Locally valid (Maekawa + Kawasaki preserving) global UNSAT near-misses
     (alternating 4-cycle M/V swaps, dual-vertex balanced flips, contradictory
     layer-interleaving taco-taco and transitivity patterns).
  3. Single-crease M/V flips, crease removals, and unassigned ('U') relaxations
     exercising multi-state component DPLL (`states > 1`).
  4. Representation invariants: collinear midpoint splits, reversed endpoints,
     shuffled line orders, and auxiliary flat ('F', code 4) creases.
"""

import collections
from collections.abc import Sequence
import dataclasses
import random

PointInt = tuple[int, int]
SegRecord = tuple[int, float, float, float, float]


@dataclasses.dataclass(frozen=True, kw_only=True)
class GeneratedCpCase:
  """One deterministic differential-testing `.cp` instance."""

  case_id: str
  category: str
  cp_doc: str


def _format_cp(records: Sequence[SegRecord]) -> str:
  lines: list[str] = []
  for code, x1, y1, x2, y2 in records:
    c_str = (
        f"{int(x1)} {int(y1)} {int(x2)} {int(y2)}"
        if all(float(v).is_integer() for v in (x1, y1, x2, y2))
        else f"{x1:.12g} {y1:.12g} {x2:.12g} {y2:.12g}"
    )
    lines.append(f"{code} {c_str}")
  return "\n".join(lines) + "\n"


def _parse_cp_records(cp_doc: str) -> list[SegRecord]:
  """Parses a `.cp` string into `(code, x1, y1, x2, y2)` records."""
  out: list[SegRecord] = []
  for line in cp_doc.splitlines():
    parts = line.split()
    if len(parts) < 5 or parts[0].startswith("#"):
      continue
    out.append((
        int(parts[0]),
        float(parts[1]),
        float(parts[2]),
        float(parts[3]),
        float(parts[4]),
    ))
  return out


def _make_checkerboard_grid_cp(
    gx: int,
    gy: int,
    rng: random.Random,
    *,
    with_diagonals: bool = False,
    unassigned_prob: float = 0.0,
) -> list[SegRecord]:
  """Builds an `[0, gx] x [0, gy]` accordion/diagonal lattice `.cp`."""
  segs: list[SegRecord] = []
  # Boundary (`1 = B`) along perimeter segments.
  for x in range(gx):
    segs.append((1, x, 0, x + 1, 0))
    segs.append((1, x, gy, x + 1, gy))
  for y in range(gy):
    segs.append((1, 0, y, 0, y + 1))
    segs.append((1, gx, y, gx, y + 1))

  # Alternating accordion parity `h_pol[y]` and `v_pol[x]` gives a 100%
  # flat-foldable base grid; flipping local 4-cycles preserves Maekawa +
  # Kawasaki at every interior vertex while creating rich taco-taco and
  # transitivity interactions!
  h_pol = [2 if (y % 2 == 1) else 3 for y in range(1, gy)]
  v_pol = [2 if (x % 2 == 1) else 3 for x in range(1, gx)]
  if rng.random() < 0.5:
    h_pol = [5 - c for c in h_pol]
  if rng.random() < 0.5:
    v_pol = [5 - c for c in v_pol]

  h_edge: dict[tuple[int, int], int] = {}
  v_edge: dict[tuple[int, int], int] = {}
  for y_idx, y in enumerate(range(1, gy)):
    for x in range(gx):
      h_edge[(x, y)] = h_pol[y_idx]
  for x_idx, x in enumerate(range(1, gx)):
    for y in range(gy):
      v_edge[(x, y)] = v_pol[x_idx]

  # At any interior degree-4 orthogonal vertex `(x, y)`, Maekawa requires
  # `|M - V| == 2`, i.e. exactly 3 of one polarity and 1 of the other!
  # In a pure grid where `h_edge[(x-1, y)] == h_edge[(x, y)] == H` and
  # `v_edge[(x, y-1)] == v_edge[(x, y)] == V`, if `H == V` we have 4 of the
  # same polarity (violating Maekawa unless one leg flips!). Remember in an
  # accordion fold: when vertical creases fold first, horizontal creases
  # ALTERNATE M/V across every vertical crease!
  for y_idx, y in enumerate(range(1, gy)):
    base = h_pol[y_idx]
    for x in range(gx):
      # Alternate horizontal polarity across vertical creases (`x % 2`) so every
      # degree-4 interior vertex has `(H, -H, V, V)` -> `|M - V| == 2`!
      h_edge[(x, y)] = base if (x % 2 == 0) else (5 - base)

  if with_diagonals and gx == gy and gx >= 2:
    # Add a full main diagonal `(0, 0) -> (gx, gy)`: at each interior vertex
    # `(k, k)` (`[0, 45, 90, 180, 225, 270]`), Kawasaki holds (`45+90+45=180`),
    # and alternating the diagonal polarity preserves `|M - V| == 2`.
    cur_d = 2 if rng.random() < 0.5 else 3
    for k in range(gx):
      segs.append((cur_d, k, k, k + 1, k + 1))
      cur_d = 5 - cur_d
    if rng.random() < 0.6:
      c_anti = rng.choice((2, 3))
      if rng.random() < 0.5:
        # Valid waterbomb vertex at `(0.5, 0.5)` (`|M - V| == 2`)!
        segs.append((c_anti, 0, 1, 0.5, 0.5))
        segs.append((5 - c_anti, 0.5, 0.5, 1, 0))
      else:
        # Unsplit crossing (`|M - V| == 0` local violation at `(0.5, 0.5)`).
        segs.append((c_anti, 0, 1, 1, 0))

  for (x, y), c in sorted(h_edge.items()):
    code = (
        5 if (unassigned_prob > 0.0 and rng.random() < unassigned_prob) else c
    )
    segs.append((code, x, y, x + 1, y))
  for (x, y), c in sorted(v_edge.items()):
    code = (
        5 if (unassigned_prob > 0.0 and rng.random() < unassigned_prob) else c
    )
    segs.append((code, x, y, x, y + 1))
  return segs


def _make_sequential_folded_sheet_cp(
    grid_n: int, num_folds: int, rng: random.Random
) -> list[SegRecord]:
  """Builds an `[0, grid_n]^2` folded-sheet grid `.cp` with `num_folds` folds.

  Args:
    grid_n: Square paper side length in integer grid units.
    num_folds: Maximum number of vertical/horizontal fold lines per axis.
    rng: Random generator used to sample crease positions and polarities.

  Returns:
    Segment records guaranteed to satisfy Maekawa and Kawasaki locally.
  """
  # Track polygons in unfolded & folded space with orientation `+1` (up) / `-1`.
  # Use integer grid `2 * grid_n` so midpoints stay integers; even simpler: work
  # via exact line set in unfolded space by reflecting line equations!
  # Even cleaner & 100% exact: reflect candidate crease lines across prior
  # through-sheet axis-aligned & diagonal folds!
  s = grid_n
  # Start with boundary `[0, s] x [0, s]`.
  # Maintain a list of infinite octilinear fold lines applied in folded space;
  # each fold splits existing polygons and reflects one half, which in unfolded
  # space corresponds to reflecting the new fold line back across all earlier
  # folds! Instead of clipping arbitrary polygons in Python, we can generate
  # exact flat-foldable box-pleated / waterbomb / Miura-like / diagonal-crease
  # lattice gadgets parameterized by `rng`!
  xs = sorted(rng.sample(range(1, s), k=min(num_folds, s - 1)))
  ys = sorted(rng.sample(range(1, s), k=min(num_folds, s - 1)))
  if not xs:
    xs = [s // 2]
  if not ys:
    ys = [s // 2]

  x_pts = [0] + xs + [s]
  y_pts = [0] + ys + [s]
  segs: list[SegRecord] = []

  for i in range(len(x_pts) - 1):
    segs.append((1, x_pts[i], 0, x_pts[i + 1], 0))
    segs.append((1, x_pts[i], s, x_pts[i + 1], s))
  for j in range(len(y_pts) - 1):
    segs.append((1, 0, y_pts[j], 0, y_pts[j + 1]))
    segs.append((1, s, y_pts[j], s, y_pts[j + 1]))

  # Choose independent fold polarities for vertical folds first, then fold
  # horizontal lines across the folded stack (each horizontal crease alternates
  # M/V across every vertical fold!).
  v_codes = [rng.choice((2, 3)) for _ in xs]
  h_base = [rng.choice((2, 3)) for _ in ys]

  for xi, x in enumerate(xs):
    for j in range(len(y_pts) - 1):
      segs.append((v_codes[xi], x, y_pts[j], x, y_pts[j + 1]))

  for yj, y in enumerate(ys):
    cur = h_base[yj]
    for i in range(len(x_pts) - 1):
      segs.append((cur, x_pts[i], y, x_pts[i + 1], y))
      cur = 5 - cur
  return segs


def _flip_maekawa_preserving_4cycle(
    records: Sequence[SegRecord], rng: random.Random
) -> list[SegRecord] | None:
  """Flips an alternating 4-cycle of interior M/V segments if one exists.

  At each vertex of an alternating `(M, V, M, V)` 4-cycle, one `M` becomes `V`
  and one `V` becomes `M`, so `M - V` (Maekawa) and all angles (Kawasaki) are
  unchanged at every vertex while global 3D layer ordering is perturbed.

  Args:
    records: Input `.cp` segment records.
    rng: Random generator used to choose vertex search order.

  Returns:
    Mutated segment list if an alternating 4-cycle was found, else None.
  """
  mut = list(records)
  adj: dict[tuple[int, int], list[tuple[tuple[int, int], int]]] = (
      collections.defaultdict(list)
  )
  for idx, (code, x1, y1, x2, y2) in enumerate(mut):
    if code not in (2, 3):
      continue
    p1 = (round(x1 * 1000), round(y1 * 1000))
    p2 = (round(x2 * 1000), round(y2 * 1000))
    adj[p1].append((p2, idx))
    adj[p2].append((p1, idx))

  verts = [v for v, nbs in adj.items() if len(nbs) >= 2]
  rng.shuffle(verts)
  for v0 in verts[:24]:
    for v1, e01 in adj[v0]:
      c01 = mut[e01][0]
      for v2, e12 in adj[v1]:
        if v2 == v0 or mut[e12][0] == c01:
          continue
        for v3, e23 in adj[v2]:
          if v3 in (v0, v1) or mut[e23][0] != c01:
            continue
          for v4, e30 in adj[v3]:
            if v4 == v0 and mut[e30][0] != c01:
              for ei in (e01, e12, e23, e30):
                c, x1, y1, x2, y2 = mut[ei]
                mut[ei] = (5 - c, x1, y1, x2, y2)
              return mut
  return None


def mutate_cp_near_misses(
    base_cp_doc: str,
    prefix: str,
    rng: random.Random,
    *,
    max_cases: int = 8,
) -> list[GeneratedCpCase]:
  """Generates near-miss UNSAT and representation-invariant mutations of `base_cp_doc`."""
  records = _parse_cp_records(base_cp_doc)
  mv_indices = [i for i, r in enumerate(records) if r[0] in (2, 3)]
  if not mv_indices:
    return []

  out: list[GeneratedCpCase] = []

  # 1. Maekawa-preserving 4-cycle flip (passes local checks, stresses solver).
  cyc = _flip_maekawa_preserving_4cycle(records, rng)
  if cyc is not None:
    out.append(
        GeneratedCpCase(
            case_id=f"{prefix}_maekawa_4cycle",
            category="near_miss_maekawa_preserving_4cycle",
            cp_doc=_format_cp(cyc),
        )
    )

  # 2. Balanced interior-vertex swap (pick an interior vertex with both M and V
  # incident creases and swap one M <-> one V at that vertex!).
  by_vert: dict[tuple[int, int], list[int]] = collections.defaultdict(list)
  b_verts: set[tuple[int, int]] = set()
  for idx, (code, x1, y1, x2, y2) in enumerate(records):
    p1 = (round(x1 * 1000), round(y1 * 1000))
    p2 = (round(x2 * 1000), round(y2 * 1000))
    if code == 1:
      b_verts.add(p1)
      b_verts.add(p2)
    elif code in (2, 3):
      by_vert[p1].append(idx)
      by_vert[p2].append(idx)
  interior_vs = [
      v for v, idxs in by_vert.items() if v not in b_verts and len(idxs) >= 4
  ]
  if interior_vs:
    v_pick = rng.choice(interior_vs)
    m_e = [i for i in by_vert[v_pick] if records[i][0] == 2]
    v_e = [i for i in by_vert[v_pick] if records[i][0] == 3]
    if m_e and v_e:
      mut = list(records)
      im = rng.choice(m_e)
      iv = rng.choice(v_e)
      mut[im] = (3, *mut[im][1:])
      mut[iv] = (2, *mut[iv][1:])
      out.append(
          GeneratedCpCase(
              case_id=f"{prefix}_vertex_swap",
              category="near_miss_vertex_mv_swap",
              cp_doc=_format_cp(mut),
          )
      )

  # 3. Single M/V flip (exercises Maekawa local violation detection).
  flip_i = rng.choice(mv_indices)
  mut_single = list(records)
  c0, x1, y1, x2, y2 = mut_single[flip_i]
  mut_single[flip_i] = (5 - c0, x1, y1, x2, y2)
  out.append(
      GeneratedCpCase(
          case_id=f"{prefix}_single_flip",
          category="near_miss_single_mv_flip",
          cp_doc=_format_cp(mut_single),
      )
  )

  # 4. Global M <-> V inversion (exact dual flat-foldable state!).
  mut_inv = [
      ((5 - c) if c in (2, 3) else c, x1, y1, x2, y2)
      for c, x1, y1, x2, y2 in records
  ]
  out.append(
      GeneratedCpCase(
          case_id=f"{prefix}_global_mv_invert",
          category="equiv_global_mv_invert",
          cp_doc=_format_cp(mut_inv),
      )
  )

  # 5. Unassigned `'U'` relaxation on 2..3 creases for bounded-size CPs so
  # reference Python `fold_cp` does not spend minutes enumerating large trees.
  if 2 <= len(mv_indices) <= 48:
    u_picks = rng.sample(mv_indices, k=min(3, len(mv_indices)))
    mut_u = list(records)
    for ui in u_picks:
      _, ux1, uy1, ux2, uy2 = mut_u[ui]
      mut_u[ui] = (5, ux1, uy1, ux2, uy2)
    out.append(
        GeneratedCpCase(
            case_id=f"{prefix}_unassigned_relax",
            category="unassigned_multi_state",
            cp_doc=_format_cp(mut_u),
        )
    )

  # 6. Removed interior crease (Kawasaki / Maekawa local violation).
  rem_i = rng.choice(mv_indices)
  mut_rem = [r for k, r in enumerate(records) if k != rem_i]
  out.append(
      GeneratedCpCase(
          case_id=f"{prefix}_removed_crease",
          category="near_miss_removed_crease",
          cp_doc=_format_cp(mut_rem),
      )
  )

  # 7. Collinear split + reversed endpoints + shuffled segment order.
  split_i = rng.choice(mv_indices)
  sc, sx1, sy1, sx2, sy2 = records[split_i]
  mx = 0.5 * (sx1 + sx2)
  my = 0.5 * (sy1 + sy2)
  mut_split = [r for k, r in enumerate(records) if k != split_i]
  mut_split.append((sc, sx1, sy1, mx, my))
  mut_split.append((sc, sx2, sy2, mx, my))  # Reversed endpoint order too!
  rng.shuffle(mut_split)
  out.append(
      GeneratedCpCase(
          case_id=f"{prefix}_collinear_split_shuffled",
          category="equiv_collinear_split_shuffled",
          cp_doc=_format_cp(mut_split),
      )
  )

  return out[:max_cases]


def generate_lattice_differential_suite(
    *,
    seed: int = 20261109,
    num_synthetic_bases: int = 120,
    base_cp_docs: Sequence[tuple[str, str]] = (),
    mutations_per_base: int = 6,
) -> list[GeneratedCpCase]:
  """Builds a deterministic differential-testing suite of lattice `.cp` cases."""
  rng = random.Random(seed)
  cases: list[GeneratedCpCase] = []

  for i in range(num_synthetic_bases):
    mode = i % 4
    if mode == 0:
      gx = rng.choice((3, 4, 5, 6))
      gy = rng.choice((3, 4, 5, 6))
      recs = _make_checkerboard_grid_cp(gx, gy, rng, with_diagonals=False)
      cat = "synthetic_accordion_grid"
    elif mode == 1:
      g = rng.choice((3, 4, 5))
      recs = _make_checkerboard_grid_cp(g, g, rng, with_diagonals=True)
      cat = "synthetic_diagonal_grid"
    elif mode == 2:
      grid_n = rng.choice((6, 8, 10, 12))
      n_folds = rng.choice((2, 3, 4, 5))
      recs = _make_sequential_folded_sheet_cp(grid_n, n_folds, rng)
      cat = "synthetic_sequential_sheet"
    else:
      gx = rng.choice((3, 4, 5))
      gy = rng.choice((3, 4, 5))
      recs = _make_checkerboard_grid_cp(
          gx, gy, rng, with_diagonals=False, unassigned_prob=0.25
      )
      cat = "synthetic_unassigned_grid"

    doc = _format_cp(recs)
    base_name = f"synth_{i:04d}"
    cases.append(GeneratedCpCase(case_id=base_name, category=cat, cp_doc=doc))
    cases.extend(
        mutate_cp_near_misses(
            doc, base_name, rng, max_cases=min(4, mutations_per_base)
        )
    )

  for name, doc in base_cp_docs:
    cases.extend(
        mutate_cp_near_misses(doc, name, rng, max_cases=mutations_per_base)
    )

  # Degenerate octilinear / folded 45-degree geometry differential cases:
  # collinear overlapping crease pieces, T-junctions, coincident folded 45°
  # edges, and fine off-unit rational offsets.
  cases.extend(_generate_degenerate_geometry_cases())

  return cases


def _generate_degenerate_geometry_cases() -> list[GeneratedCpCase]:
  """Generates corner-case degenerate octilinear/45° geometry `.cp` instances."""
  cases: list[GeneratedCpCase] = []

  # 1. Collinear overlapping creases + T-junction on waterbomb base.
  waterbomb_overlap: list[SegRecord] = [
      (1, 0, 0, 2, 0),
      (1, 2, 0, 2, 2),
      (1, 2, 2, 0, 2),
      (1, 0, 2, 0, 0),
      (2, 0, 0, 1, 1),
      (2, 0.5, 0.5, 1.5, 1.5),  # Collinear overlap along main diagonal!
      (2, 1, 1, 2, 2),
      (2, 0, 2, 1, 1),
      (2, 1, 1, 2, 0),
      (3, 0, 1, 1, 1),
      (3, 1, 1, 2, 1),
      (4, 1, 0, 1, 2),  # Auxiliary flat crease creating interior T-junction.
  ]
  cases.append(
      GeneratedCpCase(
          case_id="degen_waterbomb_collinear_overlap",
          category="degenerate_collinear_overlap",
          cp_doc=_format_cp(waterbomb_overlap),
      )
  )

  # 2. Bird-base / petal-fold style 45° coincident folded edges (many flaps
  # mapping onto identical folded 45° diagonals and corners).
  for sign_flip in (False, True):
    m = 3 if sign_flip else 2
    v = 2 if sign_flip else 3
    petal: list[SegRecord] = [
        (1, 0, 0, 4, 0),
        (1, 4, 0, 4, 4),
        (1, 4, 4, 0, 4),
        (1, 0, 4, 0, 0),
        # Symmetrical 45° corner flaps creating coincident folded edges.
        (m, 0, 2, 2, 0),
        (m, 2, 0, 4, 2),
        (m, 4, 2, 2, 4),
        (m, 2, 4, 0, 2),
        # Crossing center creases.
        (v, 0, 0, 2, 2),
        (v, 2, 2, 4, 4),
        (v, 0, 4, 2, 2),
        (v, 2, 2, 4, 0),
    ]
    suffix = "inv" if sign_flip else "base"
    cases.append(
        GeneratedCpCase(
            case_id=f"degen_coincident_45_diamond_{suffix}",
            category="degenerate_coincident_folded_45",
            cp_doc=_format_cp(petal),
        )
    )

  # 3. Shifted T-junction staircase + closely spaced 45° folded creases (`1/16`
  # and `1/32` lattice offsets exercising fine exact quantization + sweep).
  for denom, tag in ((16, "sixteenth"), (32, "thirtysecond")):
    step = 1.0 / float(denom)
    fine_stair: list[SegRecord] = [
        (1, 0, 0, 4, 0),
        (1, 4, 0, 4, 4),
        (1, 4, 4, 0, 4),
        (1, 0, 4, 0, 0),
        (2, 0, 2 - step, 4, 2 - step),
        (3, 0, 2 + step, 4, 2 + step),
        (2, 0, 1, 1, 0),
        (2, 3, 4, 4, 3),
        (
            4,
            2,
            0,
            2,
            4,
        ),  # Flat crossing perpendicular to fine accordion-pair lines.
    ]
    cases.append(
        GeneratedCpCase(
            case_id=f"degen_fine_folded_t_{tag}",
            category="degenerate_fine_tjunction",
            cp_doc=_format_cp(fine_stair),
        )
    )

  return cases
