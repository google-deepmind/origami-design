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

from collections.abc import Sequence
from concurrent import futures
import hashlib
from importlib import metadata as importlib_metadata
import json
import random
import sys
import xml.etree.ElementTree as ET

from absl.testing import absltest
from absl.testing import parameterized

# pylint: disable=g-bad-import-order
from bp_pipeline import plugins
from bp_pipeline.crease_pattern.parser import cp_format
from bp_pipeline.third_party.flat_folder import fold_cp
import bp_pipeline_flat_folder
from bp_pipeline_flat_folder import _fast_flat_folder
from bp_pipeline_flat_folder import fast_flat_folder
from bp_pipeline_flat_folder import sat_certificate_checker
import cp_test_fixtures
import differential_cp_gen

# pylint: enable=g-bad-import-order


def _compute_cp_corpus_sha256(cp_docs: Sequence[str]) -> str:
  """Computes a deterministic order-preserving SHA-256 digest over `.cp` docs."""
  hasher = hashlib.sha256()
  for doc in cp_docs:
    encoded = doc.strip().encode('utf-8')
    hasher.update(len(encoded).to_bytes(8, 'big'))
    hasher.update(encoded)
  return hasher.hexdigest()


_WATERBOMB_SAT_CP = '\n'.join([
    '1 0 0 2 0',
    '1 2 0 2 2',
    '1 2 2 0 2',
    '1 0 2 0 0',
    '1 0 0 0 1',
    '1 0 1 0 2',
    '1 2 0 2 1',
    '1 2 1 2 2',
    '2 0 0 1 1',
    '2 1 1 2 2',
    '2 0 2 1 1',
    '2 1 1 2 0',
    '3 0 1 1 1',
    '3 1 1 2 1',
])

_WATERBOMB_INVERTED_SAT_CP = '\n'.join([
    '1 0 0 2 0',
    '1 2 0 2 2',
    '1 2 2 0 2',
    '1 0 2 0 0',
    '1 0 0 0 1',
    '1 0 1 0 2',
    '1 2 0 2 1',
    '1 2 1 2 2',
    '3 0 0 1 1',
    '3 1 1 2 2',
    '3 0 2 1 1',
    '3 1 1 2 0',
    '2 0 1 1 1',
    '2 1 1 2 1',
])

# Real sub-CP taco-taco conflicts (`spike` call_idx=0 and call_idx=1).
_SPIKE_TACO_TACO_UNSAT_CP_0 = '\n'.join([
    '1 4 12 6 10',
    '2 4 12 6 12',
    '1 4 12 6 14',
    '2 6 10 6 12',
    '1 6 10 8 8',
    '2 6 12 6 14',
    '3 6 12 8 12',
    '1 6 14 8 16',
    '3 8 8 8 12',
    '1 8 8 10 8',
    '3 8 12 8 16',
    '2 8 12 10 12',
    '1 8 16 8 18',
    '2 8 16 10 14',
    '3 8 16 12 16',
    '1 8 18 8 20',
    '2 8 18 12 18',
    '1 8 20 8 22',
    '3 8 20 12 20',
    '1 8 22 8 24',
    '2 8 22 12 22',
    '1 8 24 12 24',
    '2 10 8 10 12',
    '1 10 8 12 8',
    '2 10 12 10 14',
    '3 10 12 12 12',
    '3 10 14 12 12',
    '2 10 14 12 14',
    '3 12 8 12 12',
    '1 12 8 14 8',
    '2 12 12 12 14',
    '3 12 12 14 12',
    '2 12 12 14 14',
    '2 12 14 12 16',
    '3 12 14 14 14',
    '3 12 16 12 18',
    '3 12 16 16 16',
    '2 12 18 12 20',
    '2 12 18 16 18',
    '2 12 20 12 22',
    '2 12 20 16 20',
    '2 12 22 12 24',
    '3 12 22 16 22',
    '1 12 24 16 24',
    '2 14 8 14 12',
    '1 14 8 16 8',
    '2 14 12 14 14',
    '2 14 12 16 12',
    '2 14 14 16 16',
    '3 16 8 16 12',
    '1 16 8 18 10',
    '3 16 12 16 16',
    '3 16 12 18 12',
    '1 16 16 16 18',
    '1 16 16 18 14',
    '1 16 18 16 20',
    '1 16 20 16 22',
    '1 16 22 16 24',
    '2 18 10 18 12',
    '1 18 10 20 12',
    '2 18 12 18 14',
    '2 18 12 20 12',
    '1 18 14 20 12',
])

_SPIKE_TACO_TACO_UNSAT_CP_1 = '\n'.join([
    '1 4 12 6 10',
    '2 4 12 6 12',
    '1 4 12 6 14',
    '2 6 10 6 12',
    '1 6 10 8 8',
    '2 6 12 6 14',
    '3 6 12 8 12',
    '1 6 14 8 16',
    '3 8 8 8 12',
    '1 8 8 10 8',
    '3 8 12 8 16',
    '2 8 12 10 12',
    '1 8 16 8 18',
    '2 8 16 10 14',
    '3 8 16 12 16',
    '1 8 18 8 20',
    '2 8 18 12 18',
    '1 8 20 8 22',
    '3 8 20 12 20',
    '1 8 22 8 24',
    '2 8 22 12 22',
    '1 8 24 12 24',
    '2 10 8 10 12',
    '1 10 8 12 8',
    '2 10 12 10 14',
    '3 10 12 12 12',
    '3 10 14 12 12',
    '2 10 14 12 14',
    '3 12 8 12 12',
    '1 12 8 14 8',
    '2 12 12 12 14',
    '3 12 12 14 12',
    '2 12 12 14 14',
    '3 12 14 12 16',
    '2 12 14 14 14',
    '2 12 16 12 18',
    '3 12 16 16 16',
    '2 12 18 12 20',
    '3 12 18 16 18',
    '2 12 20 12 22',
    '2 12 20 16 20',
    '2 12 22 12 24',
    '3 12 22 16 22',
    '1 12 24 16 24',
    '3 14 8 14 12',
    '1 14 8 16 8',
    '3 14 12 14 14',
    '2 14 12 16 12',
    '2 14 14 16 16',
    '3 16 8 16 12',
    '1 16 8 18 10',
    '3 16 12 16 16',
    '3 16 12 18 12',
    '1 16 16 16 18',
    '1 16 16 18 14',
    '1 16 18 16 20',
    '1 16 20 16 22',
    '1 16 22 16 24',
    '2 18 10 18 12',
    '1 18 10 20 12',
    '2 18 12 18 14',
    '2 18 12 20 12',
    '1 18 14 20 12',
])

# Local-foldability-before-folding regression (`hang2.cp`): 5x6 integer grid
# with local Maekawa violations plus one non-octilinear segment `(0, 4)-(3, 0)`.
# Folding before checking local foldability previously stalled the float sweep.
_HANG2_CP = '\n'.join([
    '1 0 0 1 0',
    '1 0 6 1 6',
    '1 1 0 2 0',
    '1 1 6 2 6',
    '1 2 0 3 0',
    '1 2 6 3 6',
    '1 3 0 4 0',
    '1 3 6 4 6',
    '1 4 0 5 0',
    '1 4 6 5 6',
    '1 0 0 0 1',
    '3 4 5 5 5',
    '1 5 0 5 1',
    '1 0 1 0 2',
    '1 5 1 5 2',
    '1 0 2 0 3',
    '1 5 2 5 3',
    '1 0 3 0 4',
    '1 5 3 5 4',
    '1 0 4 0 5',
    '1 5 4 5 5',
    '1 0 5 0 6',
    '1 5 5 5 6',
    '3 0 1 1 1',
    '2 0 2 1 2',
    '3 0 3 1 3',
    '2 0 4 1 4',
    '3 0 5 1 5',
    '2 1 1 2 1',
    '3 1 2 2 2',
    '2 1 3 2 3',
    '3 1 4 2 4',
    '2 1 5 2 5',
    '3 2 1 3 1',
    '2 2 2 3 2',
    '3 2 3 3 3',
    '2 2 4 3 4',
    '3 2 5 3 5',
    '4 3 1 4 1',
    '3 3 2 4 2',
    '2 3 3 4 3',
    '3 3 4 4 4',
    '2 3 5 4 5',
    '3 4 1 5 1',
    '2 4 2 5 2',
    '3 4 3 5 3',
    '2 4 4 5 4',
    '3 4 5 5 5',
    '3 1 0 1 1',
    '3 1 1 1 2',
    '3 1 2 1 3',
    '3 1 3 1 4',
    '3 1 4 1 5',
    '3 1 5 1 6',
    '2 2 0 2 1',
    '2 2 1 2 2',
    '2 2 2 2 3',
    '2 2 3 2 4',
    '2 2 4 2 5',
    '2 2 5 2 6',
    '3 3 0 3 1',
    '3 3 1 3 2',
    '3 3 2 3 3',
    '3 3 3 3 4',
    '3 3 4 3 5',
    '3 3 5 3 6',
    '2 4 0 4 1',
    '2 4 1 4 2',
    '2 4 2 4 3',
    '2 4 3 4 4',
    '2 4 4 4 5',
    '2 4 5 4 6',
    '5 0 4 3 0',
])


def _mutate_cp_doc(rng: random.Random, doc: str) -> str:
  """Applies 1-3 deterministic mutations to a `.cp` document."""
  lines = doc.splitlines()
  for _ in range(rng.randint(1, 3)):
    if not lines:
      break
    idx = rng.randrange(len(lines))
    parts = lines[idx].split()
    op = rng.random()
    if op < 0.15:
      del lines[idx]
    elif op < 0.30:
      lines.insert(rng.randrange(len(lines) + 1), lines[idx])
    elif op < 0.60 and len(parts) == 5:
      parts[0] = str(rng.choice((1, 2, 3, 4, 5)))
      lines[idx] = ' '.join(parts)
    elif op < 0.75 and len(parts) == 5:
      lines[idx] = ' '.join([parts[0], parts[3], parts[4], parts[1], parts[2]])
    elif op < 0.90 and len(parts) == 5:
      other_idx = rng.randrange(len(lines))
      other = lines[other_idx].split()
      if len(other) == 5:
        lines.append(
            ' '.join([
                str(rng.choice((2, 3, 5))),
                parts[1],
                parts[2],
                other[3],
                other[4],
            ])
        )
    else:
      rng.shuffle(lines)
  return '\n'.join(lines)


class FastFlatFolderTest(parameterized.TestCase):

  def test_drop_in_analyze_fold_file_state_data_svg_and_fold_json_parity(self):
    checked = 0
    for ex in cp_test_fixtures.list_cp_fixtures():
      if ex.grid_size > 20 or not ex.base_svg or not ex.base_svg.strip():
        continue
      try:
        base_cp = ex.base_cp
      except Exception:  # pylint: disable=broad-exception-caught
        continue
      if base_cp is None:
        continue

      cp_doc = cp_format.to_cp(base_cp, scale=1, ndigits=8, grid=ex.grid_size)
      with self.subTest(example=ex.name):
        ref = fold_cp.analyze_fold_file(
            ex.name, cp_doc, '.cp', checked < 6
        ).flat_folder_fold
        got = fast_flat_folder.analyze_fold_file(
            ex.name, cp_doc, '.cp', checked < 6
        ).flat_folder_fold

        self.assertEqual(got.flat_foldable, ref.flat_foldable)
        self.assertEqual(got.states, ref.states)
        self.assertEqual(bool(got.local_violations), bool(ref.local_violations))
        if checked < 6 and ref.flat_foldable:
          assert got.folded_state_fold is not None
          assert ref.folded_state_fold is not None
          self.assertEqual(
              got.folded_state_fold['faceOrientations'],
              ref.folded_state_fold['faceOrientations'],
          )
          self.assertEqual(
              got.folded_state_fold['faceOrders'],
              ref.folded_state_fold['faceOrders'],
          )
          self.assertEqual(
              got.folded_state_fold['faces_vertices'],
              ref.folded_state_fold['faces_vertices'],
          )
          self._assert_svg_geometrically_equal(
              got.folded_state_svg, ref.folded_state_svg
          )
          self._assert_svg_geometrically_equal(
              got.folded_state_svg_back, ref.folded_state_svg_back
          )
      checked += 1
    self.assertGreater(checked, 0)

    # Also check `.svg` (both unassigned and CSS `style=...`) and `.fold`
    # formats directly against `fold_cp`.
    svg_doc = (
        '<svg xmlns="http://www.w3.org/2000/svg">'
        '<line x1="0" y1="0" x2="20" y2="0" stroke="black"/>'
        '<line x1="20" y1="0" x2="20" y2="20" stroke="black"/>'
        '<line x1="20" y1="20" x2="0" y2="20" stroke="black"/>'
        '<line x1="0" y1="20" x2="0" y2="0" stroke="black"/>'
        '<line x1="10" y1="0" x2="10" y2="20" stroke="red"/>'
        '<line x1="0" y1="10" x2="10" y2="10" stroke="red"/>'
        '<line x1="10" y1="10" x2="20" y2="10" stroke="blue"/>'
        '</svg>'
    )
    ref_svg = fold_cp.analyze_fold_file(
        'bifold.svg', svg_doc, '.svg', True
    ).flat_folder_fold
    got_svg = fast_flat_folder.analyze_fold_file(
        'bifold.svg', svg_doc, '.svg', True
    ).flat_folder_fold
    self.assertTrue(ref_svg.flat_foldable)
    self.assertEqual(got_svg.flat_foldable, ref_svg.flat_foldable)
    self.assertEqual(got_svg.states, ref_svg.states)
    self.assertEqual(
        got_svg.folded_state_fold['faceOrders'],
        ref_svg.folded_state_fold['faceOrders'],
    )
    self.assertEqual(got_svg.folded_state_svg, ref_svg.folded_state_svg)

    svg_styled_doc = (
        '<svg xmlns="http://www.w3.org/2000/svg">'
        '<line x1="0" y1="0" x2="20" y2="0" style="stroke:black"/>'
        '<line x1="20" y1="0" x2="20" y2="20" style="stroke:black"/>'
        '<line x1="20" y1="20" x2="0" y2="20" style="stroke:black"/>'
        '<line x1="0" y1="20" x2="0" y2="0" style="stroke:black"/>'
        '<line x1="10" y1="0" x2="10" y2="20" style="stroke:red"/>'
        '<line x1="0" y1="10" x2="10" y2="10" style="stroke:red"/>'
        '<line x1="10" y1="10" x2="20" y2="10" style="stroke:blue"/>'
        '</svg>'
    )
    ref_styled = fold_cp.analyze_fold_file(
        'bifold_styled.svg', svg_styled_doc, '.svg', True
    ).flat_folder_fold
    got_styled = fast_flat_folder.analyze_fold_file(
        'bifold_styled.svg', svg_styled_doc, '.svg', True
    ).flat_folder_fold
    self.assertTrue(ref_styled.flat_foldable)
    self.assertEqual(got_styled.flat_foldable, ref_styled.flat_foldable)
    self.assertEqual(got_styled.states, ref_styled.states)
    self.assertEqual(
        got_styled.folded_state_fold['faceOrders'],
        ref_styled.folded_state_fold['faceOrders'],
    )
    self.assertEqual(got_styled.folded_state_svg, ref_styled.folded_state_svg)

    fold_doc = json.dumps({
        'vertices_coords': [[0, 0], [1, 0], [2, 0], [0, 2], [1, 2], [2, 2]],
        'edges_vertices': [
            [0, 1],
            [1, 2],
            [2, 5],
            [5, 4],
            [4, 3],
            [3, 0],
            [1, 4],
        ],
        'edges_assignment': ['B', 'B', 'B', 'B', 'B', 'B', 'M'],
    })
    ref_fold = fold_cp.analyze_fold_file(
        'rect.fold', fold_doc, '.fold', True
    ).flat_folder_fold
    got_fold = fast_flat_folder.analyze_fold_file(
        'rect.fold', fold_doc, '.fold', True
    ).flat_folder_fold
    self.assertEqual(got_fold.flat_foldable, ref_fold.flat_foldable)
    self.assertEqual(got_fold.states, ref_fold.states)
    self.assertEqual(
        got_fold.folded_state_fold['faceOrders'],
        ref_fold.folded_state_fold['faceOrders'],
    )

  def test_fixed_seed_differential_lattice_parity_suite(self):
    example_bases: list[tuple[str, str]] = []
    for ex in cp_test_fixtures.list_cp_fixtures():
      if ex.grid_size > 20 or not ex.base_svg or not ex.base_svg.strip():
        continue
      try:
        base_cp = ex.base_cp
      except Exception:  # pylint: disable=broad-exception-caught
        continue
      if base_cp is None:
        continue
      cp_doc = cp_format.to_cp(base_cp, scale=1, ndigits=8, grid=ex.grid_size)
      example_bases.append((ex.name, cp_doc))
      if len(example_bases) >= 12:
        break

    suite = differential_cp_gen.generate_lattice_differential_suite(
        seed=20261109,
        num_synthetic_bases=50,
        base_cp_docs=example_bases,
        mutations_per_base=5,
    )
    self.assertGreaterEqual(len(suite), 250)

    # Deterministic SHA-256 digest ensures the fixed-seed suite is reproducible.
    digest_a = _compute_cp_corpus_sha256([c.cp_doc for c in suite])
    digest_b = _compute_cp_corpus_sha256([c.cp_doc for c in suite])
    self.assertEqual(digest_a, digest_b)
    self.assertLen(suite, 280)
    self.assertEqual(
        digest_a,
        'c2db17d45605b2a7f99b4c7ca1ea6fa69d3acaaacc3b13e65b06da4ef9855851',
    )

    cold_session = fast_flat_folder.FlatFolderSession(
        use_memo_cache=False,
        use_support_cache=False,
        use_component_cache=False,
        auto_seed_deltas=False,
    )
    inc_session = fast_flat_folder.FlatFolderSession(
        use_memo_cache=False,
        use_support_cache=True,
        use_component_cache=True,
        auto_seed_deltas=True,
    )
    from bp_pipeline.third_party.flat_folder import io_utils as ff_io_utils  # pylint: disable=g-import-not-at-top

    sat_count = 0
    unsat_count = 0
    unassigned_sat_count = 0
    multi_state_count = 0
    refuted_ref_sat_cases: list[tuple[str, str]] = []

    def _group_fn(poly_rh):
      cx = sum(float(pt[0]) for pt in poly_rh) / len(poly_rh)
      cy = sum(float(pt[1]) for pt in poly_rh) / len(poly_rh)
      return int(round(cx * 5.0 + 0.13)) * 31 + int(round(cy * 5.0 + 0.27))

    for case in suite:
      ref = fold_cp.analyze_fold_file(
          case.case_id, case.cp_doc, '.cp', True
      ).flat_folder_fold
      got_cold = cold_session.analyze_fold_file(
          case.case_id, case.cp_doc, '.cp', True
      ).flat_folder_fold
      got_inc = inc_session.analyze_fold_file(
          case.case_id, case.cp_doc, '.cp', True, stream_id='diff_suite'
      ).flat_folder_fold

      with self.subTest(case_id=case.case_id, category=case.category):
        self.assertEqual(got_inc.flat_foldable, got_cold.flat_foldable)
        self.assertEqual(got_inc.states, got_cold.states)
        self.assertEqual(
            bool(got_cold.local_violations), bool(ref.local_violations)
        )
        self.assertEqual(got_inc.local_violations, got_cold.local_violations)

        # A C++ rejection against Python SAT is accepted only when the exact
        # checker also refutes Python's SAT.
        ref_sat_refuted = False
        if not got_cold.flat_foldable and ref.flat_foldable:
          v_coords, _, _, _, _, f_verts, _ = ff_io_utils.parse_document(
              case.cp_doc, 'cp', True
          )
          ref_chk = sat_certificate_checker.verify_reference_sat_certificate(
              case.cp_doc,
              unfolded_vertices=v_coords,
              face_vertices=f_verts,
              folded_state_fold=ref.folded_state_fold,
          )
          self.assertFalse(
              ref_chk.ok,
              msg=(
                  f'{case.case_id}: C++ rejected ({got_cold.conflict_reason}) '
                  'but Python SAT passed exact certificate check'
              ),
          )
          self.assertNotIn(
              ref_chk.reason, ('non_octilinear_cp', 'cp_parse_error')
          )
          refuted_ref_sat_cases.append((case.case_id, ref_chk.reason))
          ref_sat_refuted = True
        else:
          self.assertEqual(got_cold.flat_foldable, ref.flat_foldable)
          self.assertEqual(got_cold.states, ref.states)

        cold_multi = cold_session.check_cp(case.cp_doc, solution_limit=100)
        inc_multi = inc_session.check_cp(
            case.cp_doc, stream_id='diff_multi', solution_limit=100
        )
        self.assertEqual(inc_multi.states, cold_multi.states)
        if cold_multi.flat_foldable:
          self.assertTrue(cold_multi.certificate_verified)
          self.assertTrue(inc_multi.certificate_verified)

        if ref.flat_foldable and not ref_sat_refuted:
          sat_count += 1
          if case.category == 'unassigned_multi_state':
            unassigned_sat_count += 1
          if cold_multi.states > 1:
            multi_state_count += 1

          assert got_cold.folded_state_fold is not None
          assert got_inc.folded_state_fold is not None
          assert ref.folded_state_fold is not None
          self.assertEqual(
              got_inc.folded_state_fold, got_cold.folded_state_fold
          )
          self.assertEqual(got_inc.folded_state_svg, got_cold.folded_state_svg)
          self.assertEqual(
              got_inc.folded_state_svg_back, got_cold.folded_state_svg_back
          )
          self.assertEqual(
              got_cold.folded_state_fold['faceOrientations'],
              ref.folded_state_fold['faceOrientations'],
          )
          self.assertEqual(
              got_cold.folded_state_fold['faces_vertices'],
              ref.folded_state_fold['faces_vertices'],
          )
          self.assertEqual(
              got_cold.folded_state_fold['edges_vertices'],
              ref.folded_state_fold['edges_vertices'],
          )
          self.assertEqual(
              got_cold.folded_state_fold['faceOrders'],
              ref.folded_state_fold['faceOrders'],
          )
          self.assertLen(
              got_cold.folded_state_fold['vertices_coords'],
              len(ref.folded_state_fold['vertices_coords']),
          )
          for (gx, gy), (rx, ry) in zip(
              got_cold.folded_state_fold['vertices_coords'],
              ref.folded_state_fold['vertices_coords'],
          ):
            self.assertAlmostEqual(gx, rx, delta=1e-9)
            self.assertAlmostEqual(gy, ry, delta=1e-9)
          self._assert_svg_geometrically_equal(
              got_cold.folded_state_svg, ref.folded_state_svg
          )
          self._assert_svg_geometrically_equal(
              got_cold.folded_state_svg_back, ref.folded_state_svg_back
          )

          sol_cold = cold_session.solve_flat_folded_state(
              None, grid_size=8, face_group_fn=_group_fn, cp_str=case.cp_doc
          )
          sol_inc = inc_session.solve_flat_folded_state(
              None,
              grid_size=8,
              face_group_fn=_group_fn,
              cp_str=case.cp_doc,
              stream_id='diff_shape',
          )
          self.assertEqual(sol_inc, sol_cold)
        else:
          unsat_count += 1
          self.assertEqual(
              bool(got_inc.conflict_reason), bool(got_cold.conflict_reason)
          )
          self.assertEqual(
              bool(got_inc.conflict_faces), bool(got_cold.conflict_faces)
          )
          if ref.conflict_faces:
            self.assertNotEmpty(got_cold.conflict_faces)
            self.assertNotEmpty(got_inc.conflict_faces)

    self.assertGreater(sat_count, 50)
    self.assertGreater(unsat_count, 50)
    self.assertGreater(unassigned_sat_count, 10)
    self.assertGreater(multi_state_count, 5)
    self.assertGreater(inc_session.cache_stats().support_hits, 50)
    self.assertEmpty(refuted_ref_sat_cases)

  def test_incremental_session_memo_support_component_and_delta_seeding(self):
    session = fast_flat_folder.FlatFolderSession(
        use_memo_cache=True,
        use_support_cache=True,
        use_component_cache=True,
        auto_seed_deltas=True,
    )

    # 1. Initial call populates Tier-2 support and Tier-1 memo.
    r0 = session.check_cp(_WATERBOMB_SAT_CP, stream_id='wb')
    self.assertTrue(r0.flat_foldable)
    self.assertFalse(r0.memo_cache_hit)
    self.assertFalse(r0.support_cache_hit)
    self.assertTrue(r0.used_exact_octilinear)
    self.assertTrue(r0.used_exact_folded_octilinear)

    # 2. Repeat identical `.cp` hits Tier-1 memo.
    r0_memo = session.check_cp(_WATERBOMB_SAT_CP, stream_id='wb')
    self.assertTrue(r0_memo.flat_foldable)
    self.assertTrue(r0_memo.memo_cache_hit)

    # 3. Inverted M/V on identical crease geometry hits Tier-2 support cache.
    r1 = session.check_cp(_WATERBOMB_INVERTED_SAT_CP, stream_id='wb')
    self.assertTrue(r1.flat_foldable)
    self.assertFalse(r1.memo_cache_hit)
    self.assertTrue(r1.support_cache_hit)

    # 4. Sequential `spike` UNSAT sub-CPs hit Tier-2 support + Wave-1 delta.
    u0 = session.check_cp(_SPIKE_TACO_TACO_UNSAT_CP_0, stream_id='spike')
    u1 = session.check_cp(_SPIKE_TACO_TACO_UNSAT_CP_1, stream_id='spike')
    self.assertFalse(u0.flat_foldable)
    self.assertFalse(u1.flat_foldable)
    self.assertTrue(u1.support_cache_hit)
    self.assertTrue(u1.seeded_hit)
    self.assertNotEmpty(u1.conflict_faces)

    # 5. Multi-component unassigned `'U'` crease pattern hits Tier-3 cache!
    # `[0, 5] x [0, 1]` with `U` at `x=1` (folds `[0, 1]` onto `[1, 2]`) and
    # `M`/`V` at `x=4` (folds `[4, 5]` onto `[3, 4]`, disjoint from `[1, 2]`).
    unassigned_cp_a = '\n'.join([
        '1 0 0 5 0',
        '1 5 0 5 1',
        '1 5 1 0 1',
        '1 0 1 0 0',
        '5 1 0 1 1',
        '2 4 0 4 1',
    ])
    unassigned_cp_b = '\n'.join([
        '1 0 0 5 0',
        '1 5 0 5 1',
        '1 5 1 0 1',
        '1 0 1 0 0',
        '5 1 0 1 1',
        '3 4 0 4 1',
    ])
    ca = session.check_cp(unassigned_cp_a, stream_id='u_comp', solution_limit=0)
    cb = session.check_cp(unassigned_cp_b, stream_id='u_comp', solution_limit=0)
    self.assertTrue(ca.flat_foldable)
    self.assertTrue(ca.used_exact_octilinear)
    self.assertTrue(ca.used_exact_folded_octilinear)
    self.assertEqual(ca.states, 2)
    self.assertTrue(cb.flat_foldable)
    self.assertEqual(cb.states, 2)
    self.assertEqual(cb.component_cache_hits, 1)

    stats = session.cache_stats()
    self.assertGreaterEqual(stats.memo_hits, 1)
    self.assertGreaterEqual(stats.support_hits, 2)
    self.assertGreaterEqual(stats.component_hits, 1)

  def test_solve_flat_folded_state_and_coherent_face_group_fn(self):
    # Inward trifold (`[0, 3] x [0, 2]` with `M` at `x=1` and `M` at `x=2`):
    # left flap `[0, 1]` and right flap `[2, 3]` both fold onto `[1, 2]` on the
    # same side of the center panel, so their relative stacking order is
    # unconstrained by hinges and deterministically steered by `face_group_fn`.
    trifold_cp = '\n'.join([
        '1 0 0 3 0',
        '1 3 0 3 2',
        '1 3 2 0 2',
        '1 0 2 0 0',
        '2 1 0 1 2',
        '2 2 0 2 2',
    ])
    session = fast_flat_folder.FlatFolderSession(use_memo_cache=False)

    sol_asc = session.solve_flat_folded_state(
        None,
        grid_size=3,
        face_group_fn=lambda poly_rh: int(
            round(sum(p[0] * 6 for p in poly_rh) / len(poly_rh))
        ),
        cp_str=trifold_cp,
    )
    sol_desc = session.solve_flat_folded_state(
        None,
        grid_size=3,
        face_group_fn=lambda poly_rh: -int(
            round(sum(p[0] * 6 for p in poly_rh) / len(poly_rh))
        ),
        cp_str=trifold_cp,
    )

    self.assertLen(sol_asc.face_orders, 3)
    self.assertLen(sol_asc.face_top_up, 3)
    self.assertNotEqual(sol_asc.face_orders, sol_desc.face_orders)
    for cell_stack in sol_asc.cell_faces_ordered:
      orders = [sol_asc.face_orders[fi] for fi in cell_stack]
      self.assertEqual(orders, sorted(orders))

  def test_order_independent_session_shadow_audit_and_python_ref_flag(self):
    # Verify order-independent session settings disable support/component/delta
    # caches while preserving exact-CP memoization and exact SAT certificates.
    default_sess = fast_flat_folder.FlatFolderSession(
        use_memo_cache=True,
        use_support_cache=False,
        use_component_cache=False,
        auto_seed_deltas=False,
    )
    wb_res_0 = default_sess.check_cp(
        _WATERBOMB_SAT_CP, stream_id=None, verify_certificate=True
    )
    wb_res_1 = default_sess.check_cp(
        _WATERBOMB_INVERTED_SAT_CP, stream_id=None, verify_certificate=True
    )
    wb_res_0_again = default_sess.check_cp(
        _WATERBOMB_SAT_CP, stream_id=None, verify_certificate=True
    )
    self.assertTrue(wb_res_0.flat_foldable)
    self.assertTrue(wb_res_0.certificate_verified)
    self.assertTrue(wb_res_1.flat_foldable)
    self.assertTrue(wb_res_1.certificate_verified)
    self.assertTrue(wb_res_0_again.flat_foldable)
    default_stats = default_sess.cache_stats()
    self.assertEqual(default_stats.support_hits, 0)
    self.assertEqual(default_stats.support_entries, 0)
    self.assertEqual(default_stats.component_entries, 0)
    self.assertGreaterEqual(default_stats.memo_hits, 1)

    session = fast_flat_folder.FlatFolderSession(shadow_audit_every_n=1)
    audited_res = session.check_cp(
        _WATERBOMB_SAT_CP, stream_id=None, verify_certificate=True
    )
    self.assertTrue(audited_res.flat_foldable)
    self.assertTrue(audited_res.certificate_verified)
    self.assertEqual(audited_res.states, 1)
    self.assertEqual(session.shadow_audit_stats().shadow_audits_run, 1)
    self.assertEqual(session.shadow_audit_stats().mismatches, 0)

    # Verify reference flag (`backend='python_ref'`) also works cleanly.
    ref_res = fast_flat_folder.check_cp(_WATERBOMB_SAT_CP, backend='python_ref')
    self.assertTrue(ref_res.flat_foldable)
    self.assertEqual(ref_res.exit_stage, 'python_ref')

    # Verify runtime shadow auditor catches any injected divergence!
    with self.assertRaisesRegex(
        fast_flat_folder.FastFlatFolderParityError, 'Shadow audit mismatch'
    ):
      session._maybe_shadow_audit_cp(_WATERBOMB_SAT_CP, False, 0, 5e-2)  # pylint: disable=protected-access

  def test_public_boundary_geometry_apis_and_import_firewall(self):
    # Package boundary: no private `.engine` property, no unverifiable corpus
    # constants, trimmed root exports, zero `ortools` imports, and zero
    # side-effect registrations on `bp_pipeline.plugins`.
    session = bp_pipeline_flat_folder.FlatFolderSession()
    self.assertFalse(hasattr(session, 'engine'))
    self.assertFalse(hasattr(fast_flat_folder, 'VERIFIED_PARITY_CORPUS_ID'))
    self.assertFalse(hasattr(fast_flat_folder, 'VERIFIED_PARITY_CP_COUNT'))
    self.assertFalse(hasattr(fast_flat_folder, 'build_parity_certificate'))
    self.assertFalse(hasattr(fast_flat_folder, 'make_trusted_gate3_oracle'))
    self.assertFalse(hasattr(fast_flat_folder, 'compute_cp_corpus_sha256'))
    expected_root_exports = {
        'BackendName',
        'FastFlatFolderParityError',
        'FastFoldResult',
        'FlatFoldSolution',
        'FlatFolderOutput',
        'FlatFolderSession',
        'FoldVerdict',
        'analyze_fold_file',
        'check_cp',
        'compute_folded_vertices_cp',
        'fast_flat_folder',
        'sat_certificate_checker',
        'solve_flat_folded_state',
    }
    self.assertEqual(
        set(bp_pipeline_flat_folder.__all__), expected_root_exports
    )
    for trimmed_name in (
        'compute_cp_corpus_sha256',
        'GUI_COLORS',
        'visualize_folded_state',
        'get_absolute_face_orders_from_edges',
        'Point2DFloat',
        'SegmentCoords',
        'ShadowAuditStats',
        'diff_cp_changed_segments',
        'verify_cpp_sat_certificate',
        'verify_reference_sat_certificate',
        'CertificateCheckResult',
        'FlatFolderFOLD',
        'MeshLayerOrderResult',
    ):
      self.assertFalse(
          hasattr(bp_pipeline_flat_folder, trimmed_name),
          msg=f'Trimmed symbol unexpectedly exported at root: {trimmed_name}',
      )
    self.assertTrue(hasattr(fast_flat_folder, 'MeshLayerOrderResult'))
    self.assertTrue(hasattr(session, 'solve_segments'))
    self.assertTrue(hasattr(session, 'solve_graph'))
    # Submodule exports on `sat_certificate_checker` remain intact.
    self.assertTrue(hasattr(sat_certificate_checker, 'CertificateCheckResult'))
    self.assertTrue(
        hasattr(sat_certificate_checker, 'CertificateRejectionKind')
    )
    self.assertTrue(hasattr(sat_certificate_checker, 'REJECTION_REASON_TABLE'))
    self.assertTrue(
        hasattr(sat_certificate_checker, 'verify_cpp_sat_certificate')
    )
    self.assertTrue(
        hasattr(sat_certificate_checker, 'verify_reference_sat_certificate')
    )

    for mod_name in sys.modules:
      self.assertFalse(
          mod_name == 'ortools' or mod_name.startswith('ortools.'),
          msg=f'Unexpected ortools import: {mod_name}',
      )
    installed_plugin_eps = list(
        importlib_metadata.entry_points(group='bp_pipeline.plugins')
    )
    for ep in installed_plugin_eps:
      self.assertFalse(
          ep.value.startswith('bp_pipeline_flat_folder'),
          msg=f'Unexpected bp_pipeline.plugins entry point: {ep}',
      )
    if not installed_plugin_eps:
      self.assertEqual(plugins.list_packers(), ('default', 'builtin'))
      self.assertEqual(plugins.list_solvers(), ('default', 'builtin'))
      self.assertEqual(plugins.resolve_packer('default'), 'builtin')
      self.assertEqual(plugins.resolve_solver('default'), 'builtin')

    # Public `FlatFolderSession.extract_unfolded_faces_cp`.
    v_coords, f_verts = session.extract_unfolded_faces_cp(
        _WATERBOMB_SAT_CP, flip_to_top_face_up=True
    )
    self.assertNotEmpty(v_coords)
    self.assertNotEmpty(f_verts)
    with self.assertRaisesRegex(ValueError, 'line 2'):
      session.extract_unfolded_faces_cp('1 0 0 1 0\n1 0 0\n')

    # Public `compute_folded_vertices_cp` (stateless and session-backed).
    stateless_pairs = bp_pipeline_flat_folder.compute_folded_vertices_cp(
        _WATERBOMB_SAT_CP
    )
    self.assertIsNotNone(stateless_pairs)
    assert stateless_pairs is not None
    self.assertLen(stateless_pairs[0], len(stateless_pairs[1]))
    self.assertEqual(
        bp_pipeline_flat_folder.compute_folded_vertices_cp(
            _WATERBOMB_SAT_CP, session=session
        ),
        stateless_pairs,
    )
    self.assertIsNone(
        bp_pipeline_flat_folder.compute_folded_vertices_cp(
            _SPIKE_TACO_TACO_UNSAT_CP_0
        )
    )
    self.assertIsNone(
        bp_pipeline_flat_folder.compute_folded_vertices_cp('1 0 0 1 0\n1 0 0\n')
    )

  def _assert_svg_geometrically_equal(
      self, got_svg: str | None, ref_svg: str | None, atol: float = 1e-6
  ) -> None:
    self.assertIsNotNone(got_svg)
    self.assertIsNotNone(ref_svg)
    got_root = ET.fromstring(got_svg)
    ref_root = ET.fromstring(ref_svg)
    self.assertEqual(got_root.attrib, ref_root.attrib)

    def _canon_polys(root: ET.Element):
      polys = []
      for el in root.iter():
        if el.tag.endswith('polygon'):
          nums = [
              float(tok)
              for tok in el.attrib['points'].replace(',', ' ').split()
          ]
          pts = [(nums[i], nums[i + 1]) for i in range(0, len(nums), 2)]
          min_idx = min(
              (round(pt[1], 4), round(pt[0], 4), idx)
              for idx, pt in enumerate(pts)
          )[2]
          rot = pts[min_idx:] + pts[:min_idx]
          key = tuple((round(p[1], 4), round(p[0], 4)) for p in rot)
          style = {k: v for k, v in el.attrib.items() if k != 'points'}
          polys.append((key, rot, style))
      polys.sort(key=lambda item: item[0])
      return polys

    def _canon_lines(root: ET.Element):
      lines = []
      for el in root.iter():
        if el.tag.endswith('line'):
          p1 = (float(el.attrib['x1']), float(el.attrib['y1']))
          p2 = (float(el.attrib['x2']), float(el.attrib['y2']))
          if (round(p2[1], 4), round(p2[0], 4)) < (
              round(p1[1], 4),
              round(p1[0], 4),
          ):
            p1, p2 = p2, p1
          key = (
              round(p1[1], 4),
              round(p1[0], 4),
              round(p2[1], 4),
              round(p2[0], 4),
              el.attrib.get('stroke', ''),
          )
          style = {
              k: v
              for k, v in el.attrib.items()
              if k not in ('x1', 'y1', 'x2', 'y2')
          }
          lines.append((key, (p1, p2), style))
      lines.sort(key=lambda item: item[0])
      return lines

    g_polys = _canon_polys(got_root)
    r_polys = _canon_polys(ref_root)
    self.assertLen(g_polys, len(r_polys))
    for (_, g_pts, g_st), (_, r_pts, r_st) in zip(g_polys, r_polys):
      self.assertEqual(g_st, r_st)
      self.assertLen(g_pts, len(r_pts))
      for (gx, gy), (rx, ry) in zip(g_pts, r_pts):
        self.assertAlmostEqual(gx, rx, delta=atol)
        self.assertAlmostEqual(gy, ry, delta=atol)

    g_lines = _canon_lines(got_root)
    r_lines = _canon_lines(ref_root)
    self.assertLen(g_lines, len(r_lines))
    for (_, (gp1, gp2), g_st), (_, (rp1, rp2), r_st) in zip(g_lines, r_lines):
      self.assertEqual(g_st, r_st)
      for gv, rv in (*zip(gp1, rp1), *zip(gp2, rp2)):
        self.assertAlmostEqual(gv, rv, delta=atol)

  def test_sat_certificate_checker_soundness_and_corruptions(self):
    """Verifies exact SAT-certificate checker accepts valid SATs and rejects corruptions."""
    from bp_pipeline.third_party.flat_folder import io_utils as ff_io_utils  # pylint: disable=g-import-not-at-top

    engine = _fast_flat_folder.FastFlatFolderEngine(64, 64, 64)
    opts = _fast_flat_folder.SolveOptions()
    opts.flip_to_top_face_up = True
    opts.compute_layer_order = True

    class _FakeCert:
      """Mutable proxy around `SolveResult` for negative soundness tests."""

      def __init__(self, base):
        self.flat_foldable = bool(base.flat_foldable)
        self.vertex_coords = list(base.vertex_coords)
        self.face_vertices = [list(f) for f in base.face_vertices]
        self.face_orientations = list(base.face_orientations)
        self.folded_vertices = list(base.folded_vertices)
        self.cell_points = list(base.cell_points)
        self.cell_vertices = [list(c) for c in base.cell_vertices]
        self.cell_faces_ordered = [list(st) for st in base.cell_faces_ordered]

    # 1. Verify C++ and Python `fold_cp` SAT certificates across differential
    #    suite.
    cases = differential_cp_gen.generate_lattice_differential_suite(
        seed=20261109,
        num_synthetic_bases=50,
        base_cp_docs=[],
        mutations_per_base=3,
    )
    sat_checked = 0
    for case in cases:
      cpp_res = engine.analyze_cp(case.cp_doc, opts)
      if cpp_res.flat_foldable:
        chk = sat_certificate_checker.verify_cpp_sat_certificate(
            case.cp_doc, cpp_res
        )
        self.assertTrue(chk.ok, msg=f'{case.case_id}: {chk.reason}')
        ref_out = fold_cp.analyze_fold_file(
            case.case_id, case.cp_doc, '.cp', True
        ).flat_folder_fold
        if ref_out.flat_foldable:
          v_coords, _, _, _, _, f_verts, _ = ff_io_utils.parse_document(
              case.cp_doc, 'cp', True
          )
          ref_chk = sat_certificate_checker.verify_reference_sat_certificate(
              case.cp_doc,
              unfolded_vertices=v_coords,
              face_vertices=f_verts,
              folded_state_fold=ref_out.folded_state_fold,
          )
          self.assertTrue(
              ref_chk.ok, msg=f'{case.case_id} ref: {ref_chk.reason}'
          )
        sat_checked += 1
    self.assertGreater(sat_checked, 40)

    # 2. Corrupt M/V order (reverse stack on a 1-crease fold).
    one_crease_cp = '\n'.join([
        '1 0 0 2 0',
        '1 2 0 2 2',
        '1 2 2 0 2',
        '1 0 2 0 0',
        '2 1 0 1 2',
    ])
    c1 = engine.analyze_cp(one_crease_cp, opts)
    self.assertTrue(c1.flat_foldable)
    self.assertTrue(
        sat_certificate_checker.verify_cpp_sat_certificate(one_crease_cp, c1).ok
    )
    bad_mv = _FakeCert(c1)
    for st in bad_mv.cell_faces_ordered:
      st.reverse()
    res_mv = sat_certificate_checker.verify_cpp_sat_certificate(
        one_crease_cp, bad_mv
    )
    self.assertFalse(res_mv.ok)
    self.assertIn('mv_order_violation', res_mv.reason)

    # 3. Corrupt Taco-Taco (interleave two unassigned folded tacos on same
    #    line).
    # 4 vertical strips [0,1],[1,2],[2,3],[3,4] folded at x=1 (U), x=2 (F? no:
    # fold at x=1, x=2, x=3 so x=1 and x=3 both land on folded x=1!).
    taco_taco_cp = '\n'.join([
        '1 0 0 4 0',
        '1 4 0 4 1',
        '1 4 1 0 1',
        '1 0 1 0 0',
        '5 1 0 1 1',
        '5 2 0 2 1',
        '5 3 0 3 1',
    ])
    ct = engine.analyze_cp(taco_taco_cp, opts)
    self.assertTrue(ct.flat_foldable)
    self.assertTrue(
        sat_certificate_checker.verify_cpp_sat_certificate(taco_taco_cp, ct).ok
    )
    # Find the non-empty cell of 4 faces; interleave taco (0,1) and taco (2,3).
    bad_tt = _FakeCert(ct)
    for idx, st in enumerate(bad_tt.cell_faces_ordered):
      if len(st) == 4:
        # Force interleaved order [f0, f2, f1, f3] using current face IDs:
        # valid stack nests or separates (e.g. [a, b, c, d]); swapping middle
        # with outer interleaves the two tacos on x=1!
        bad_tt.cell_faces_ordered[idx] = [st[0], st[2], st[1], st[3]]
    res_tt = sat_certificate_checker.verify_cpp_sat_certificate(
        taco_taco_cp, bad_tt
    )
    self.assertFalse(res_tt.ok)
    self.assertEqual(res_tt.reason, 'taco_taco_crossing')

    # 4. Corrupt Taco-Tortilla (pierce a fold at x=1 with an uncreased sheet).
    # Strips [0,2] (face A, width 2 spanning x=1!), [2,3] (folded at x=2 onto
    # [1,2]), [3,4] (folded at x=3 onto [1,2], so crease x=3 lands on x=1!).
    taco_tort_cp = '\n'.join([
        '1 0 0 4 0',
        '1 4 0 4 1',
        '1 4 1 0 1',
        '1 0 1 0 0',
        '5 2 0 2 1',
        '5 3 0 3 1',
    ])
    ctt = engine.analyze_cp(taco_tort_cp, opts)
    self.assertTrue(ctt.flat_foldable)
    self.assertTrue(
        sat_certificate_checker.verify_cpp_sat_certificate(taco_tort_cp, ctt).ok
    )
    bad_ttort = _FakeCert(ctt)
    # Put the wide tortilla face strictly between the two faces meeting at x=3!
    for idx, st in enumerate(bad_ttort.cell_faces_ordered):
      if len(st) == 3:
        # In the valid 3-layer stack on [1,2], the wide face is at top or bottom
        # (`st[0]` or `st[2]`). Moving that extreme face into the middle `st[1]`
        # pierces the taco!
        # Identify wide face as the one also in the 1-layer cell [0,1].
        single_faces = {
            s[0] for s in bad_ttort.cell_faces_ordered if len(s) == 1
        }
        wide_f = next(iter(single_faces))
        others = [f for f in st if f != wide_f]
        bad_ttort.cell_faces_ordered[idx] = [others[0], wide_f, others[1]]
    res_ttort = sat_certificate_checker.verify_cpp_sat_certificate(
        taco_tort_cp, bad_ttort
    )
    self.assertFalse(res_ttort.ok)
    self.assertEqual(res_ttort.reason, 'taco_tortilla_crossing')

    # 5. Corrupt Tortilla-Tortilla across an `F` flat crease (`code 4`).
    # Fold [0,2]x[0,1] in half at x=1 (`U`), with flat crease `F` at x=0.5 and
    # x=1.5 landing on the SAME folded line x=0.5!
    tort_tort_cp = '\n'.join([
        '1 0 0 2 0',
        '1 2 0 2 1',
        '1 2 1 0 1',
        '1 0 1 0 0',
        '4 0.5 0 0.5 1',
        '5 1 0 1 1',
        '4 1.5 0 1.5 1',
    ])
    c_ff = engine.analyze_cp(tort_tort_cp, opts)
    self.assertTrue(c_ff.flat_foldable)
    self.assertTrue(
        sat_certificate_checker.verify_cpp_sat_certificate(
            tort_tort_cp, c_ff
        ).ok
    )
    bad_ff = _FakeCert(c_ff)
    nonempty_idxs = [
        i for i, st in enumerate(bad_ff.cell_faces_ordered) if len(st) == 2
    ]
    self.assertLen(nonempty_idxs, 2)
    # Reverse ONLY one of the two cells across the flat crease!
    bad_ff.cell_faces_ordered[nonempty_idxs[0]].reverse()
    res_ff = sat_certificate_checker.verify_cpp_sat_certificate(
        tort_tort_cp, bad_ff
    )
    self.assertFalse(res_ff.ok)
    self.assertEqual(res_ff.reason, 'tortilla_tortilla_crossing')

    # 6. Corrupt per-cell coverage / cross-cell cycle.
    bad_dup = _FakeCert(ct)
    for idx, st in enumerate(bad_dup.cell_faces_ordered):
      if len(st) >= 2:
        bad_dup.cell_faces_ordered[idx] = [st[0], st[0]] + st[2:]
    res_dup = sat_certificate_checker.verify_cpp_sat_certificate(
        taco_taco_cp, bad_dup
    )
    self.assertFalse(res_dup.ok)
    self.assertIn('cell_stack_coverage_mismatch', res_dup.reason)

    # 7. Verify `check_cp(..., verify_certificate=True)` checks exact SAT
    #    certificates on valid patterns.
    ok_gate3 = fast_flat_folder.check_cp(
        _WATERBOMB_SAT_CP, verify_certificate=True
    )
    self.assertTrue(ok_gate3.flat_foldable)
    self.assertTrue(ok_gate3.certificate_verified)

    # 8. Fail-closed rational parser: reject excessive denominator (`> 2^50`)
    #    without falling back to float approximation.
    huge_denom_cp = one_crease_cp.replace(
        '2 1 0 1 2', f'2 1/{(1 << 50) + 1} 0 1 2'
    )
    res_denom = sat_certificate_checker.verify_cpp_sat_certificate(
        huge_denom_cp, c1
    )
    self.assertFalse(res_denom.ok)
    self.assertEqual(res_denom.reason, 'cp_parse_error')

    # 9. Manifold incidence guards: reject dangling slit inside a face and
    #    orphan segment outside the sheet.
    slit_cp = one_crease_cp + '\n2 0 1 0.5 1'
    res_slit = sat_certificate_checker.verify_cpp_sat_certificate(slit_cp, c1)
    self.assertFalse(res_slit.ok)
    self.assertEqual(res_slit.reason, 'slit_or_dangling_edge')

    orphan_cp = one_crease_cp + '\n2 5 5 6 5'
    res_orphan = sat_certificate_checker.verify_cpp_sat_certificate(
        orphan_cp, c1
    )
    self.assertFalse(res_orphan.ok)
    self.assertEqual(res_orphan.reason, 'non_manifold_or_orphan_edge')

    # 10. Internal 2-face `B` (`code == 1`) tortilla crossing:
    #     `1 0.5 0 0.5 1` splits `[0, 1] x [0, 1]` into two non-hole faces (both
    #     touch `4 0 0.5 2 0.5`), while `[1, 2] x [0, 1]` has NO crease at
    #     `x = 1.5`. Inverting the stack on `x < 0.5` crosses the uncreased
    #     right-half sheet across the 2-face `B` segment at `x = 0.5`.
    internal_b_cp = '\n'.join([
        '1 0 0 2 0',
        '1 2 0 2 1',
        '1 2 1 0 1',
        '1 0 1 0 0',
        '4 0 0.5 2 0.5',
        '1 0.5 0 0.5 1',
        '5 1 0 1 1',
    ])
    c_ib = engine.analyze_cp(internal_b_cp, opts)
    self.assertTrue(c_ib.flat_foldable)
    self.assertTrue(
        sat_certificate_checker.verify_cpp_sat_certificate(
            internal_b_cp, c_ib
        ).ok
    )
    bad_ib = _FakeCert(c_ib)
    xs_ib = [pt[0] for pt in bad_ib.cell_points]
    mid_x = 0.5 * (min(xs_ib) + max(xs_ib))
    reversed_cells = 0
    for idx, st in enumerate(bad_ib.cell_faces_ordered):
      if len(st) == 2:
        cyc = bad_ib.cell_vertices[idx]
        mean_x = sum(bad_ib.cell_points[vi][0] for vi in cyc) / len(cyc)
        if mean_x < mid_x:
          st.reverse()
          reversed_cells += 1
    self.assertEqual(reversed_cells, 2)
    res_ib = sat_certificate_checker.verify_cpp_sat_certificate(
        internal_b_cp, bad_ib
    )
    self.assertFalse(res_ib.ok)
    self.assertEqual(res_ib.reason, 'tortilla_tortilla_crossing')

    # 11. Non-tree rigid transform holonomy conflict around a cutout hole:
    #     Punctured annulus `[0, 6] x [0, 4]` minus `[2, 4] x [1, 3]` with two
    #     parallel active creases `M` (`x = 2`) and `V` (`x = 3`) across the top
    #     bridge (`y in [3, 4]`). Orientation parity (`face_flipped`) is even
    #     around the hole, and there are 0 interior vertices, but rigid
    #     reflection composition has non-zero horizontal translation holonomy.
    holonomy_cp = '\n'.join([
        '1 0 0 6 0',
        '1 6 0 6 4',
        '1 6 4 0 4',
        '1 0 4 0 0',
        '1 2 1 4 1',
        '1 4 1 4 3',
        '1 4 3 2 3',
        '1 2 3 2 1',
        '4 0 1 2 1',
        '4 4 1 6 1',
        '4 0 3 2 3',
        '4 4 3 6 3',
        '2 2 3 2 4',
        '3 3 3 3 4',
    ])
    holo_audit = sat_certificate_checker.audit_exact_folded_arrangement(
        holonomy_cp
    )
    self.assertFalse(holo_audit['locally_closed'])
    self.assertEqual(holo_audit['reason'], 'transform_holonomy_conflict')

    class _FakeHolonomyCert:
      flat_foldable = True
      vertex_coords = [
          (0.0, 0.0),
          (6.0, 0.0),
          (0.0, 1.0),
          (2.0, 1.0),
          (4.0, 1.0),
          (6.0, 1.0),
          (0.0, 3.0),
          (2.0, 3.0),
          (3.0, 3.0),
          (4.0, 3.0),
          (6.0, 3.0),
          (0.0, 4.0),
          (2.0, 4.0),
          (3.0, 4.0),
          (6.0, 4.0),
      ]
      face_vertices = [
          [0, 1, 5, 2],
          [2, 3, 7, 6],
          [4, 5, 10, 9],
          [6, 7, 12, 11],
          [7, 8, 13, 12],
          [8, 10, 14, 13],
      ]
      face_orientations = [False, False, False, False, True, False]
      folded_vertices = vertex_coords
      cell_points = vertex_coords
      cell_vertices = face_vertices
      cell_faces_ordered = [[0], [1], [2], [3], [4], [5]]

    res_holo = sat_certificate_checker.verify_cpp_sat_certificate(
        holonomy_cp, _FakeHolonomyCert()
    )
    self.assertFalse(res_holo.ok)
    self.assertEqual(res_holo.reason, 'transform_holonomy_conflict')

  def test_reference_fold_cp_sweep_line_defect_minimal_repro(self):
    """Minimal 22-line integer `.cp` repro for reference `_SweepLineRunner` bug.

    Constructed by sequential flat folds on the `[0, 16]^2` integer grid
    (3 interior vertices at `(1, 5)`, `(1, 11)`, `(8, 11)`, 7 faces, octilinear
    creases only). Float64 `1/sqrt(2)` reflection drift in
    `folded_state.compute_folded_state` perturbs folded vertex coordinates by
    `O(1e-15)`, causing
    `graph_creation._SweepLineRunner._add_intersection_to_queue`
    to re-queue a past horizontal vertex and corrupt `cell_faces` flood-fill so
    `face_constraints.add_constraint_to_table` raises
    `ValueError('Constraint variable missing')`. Exact `Z[1/16]` octilinear
    geometry in `fast_flat_folder` + `sat_certificate_checker` proves the CP is
    flat-foldable (`states=1`).
    """
    repro_cp = '\n'.join([
        '1 0 0 0 5',
        '1 0 0 6 0',
        '1 0 5 0 6',
        '2 0 5 1 5',
        '1 0 6 0 11',
        '3 0 6 1 5',
        '1 0 11 0 16',
        '3 0 11 1 11',
        '1 0 16 1 16',
        '3 1 5 1 11',
        '3 1 5 6 0',
        '2 1 11 1 16',
        '3 1 11 8 11',
        '1 1 16 13 16',
        '1 6 0 16 0',
        '2 8 11 13 16',
        '3 8 11 16 3',
        '3 8 11 16 11',
        '1 13 16 16 16',
        '1 16 0 16 3',
        '1 16 3 16 11',
        '1 16 11 16 16',
    ])

    # 1. Reference `fold_cp` falsely rejects with `Constraint variable missing`.
    ref_out = fold_cp.analyze_fold_file(
        'repro_sweep_defect.cp', repro_cp, '.cp', True
    ).flat_folder_fold
    self.assertFalse(ref_out.flat_foldable)
    self.assertEqual(ref_out.conflict_reason, 'Constraint variable missing')

    # 2. Exact C++ solver + standalone exact SAT-certificate checker prove SAT.
    got_res = fast_flat_folder.check_cp(repro_cp, verify_certificate=True)
    self.assertTrue(got_res.flat_foldable)
    self.assertTrue(got_res.certificate_verified)
    self.assertTrue(got_res.used_exact_octilinear)
    self.assertTrue(got_res.used_exact_folded_octilinear)
    self.assertEqual(got_res.states, 1)

    got_fold = fast_flat_folder.analyze_fold_file(
        'repro_sweep_defect.cp', repro_cp, '.cp', True
    ).flat_folder_fold
    self.assertTrue(got_fold.flat_foldable)
    self.assertEqual(got_fold.states, 1)
    self.assertIsNotNone(got_fold.folded_state_fold)

    exact_topo = sat_certificate_checker.audit_exact_folded_arrangement(
        repro_cp
    )
    self.assertTrue(exact_topo['locally_closed'])

    # 3. Runtime shadow auditor classifies proven reference false negative as
    #    `reference_incorrect` without raising `FastFlatFolderParityError`.
    audit_sess = fast_flat_folder.FlatFolderSession(shadow_audit_every_n=1)
    audited_res = audit_sess.check_cp(repro_cp, verify_certificate=True)
    self.assertTrue(audited_res.flat_foldable)
    self.assertTrue(audited_res.certificate_verified)
    self.assertTrue(audited_res.shadow_audited)
    stats = audit_sess.shadow_audit_stats()
    self.assertEqual(stats.shadow_audits_run, 1)
    self.assertEqual(stats.reference_incorrect, 1)
    self.assertEqual(stats.mismatches, 0)

  def test_multithreaded_concurrent_session_and_stateless_calls(self):
    """Stresses thread-safe C++/Python caches, stateless APIs, and CoordBits."""
    # Top-level stateless helpers must not retain a hidden global session.
    self.assertFalse(hasattr(fast_flat_folder, '_DEFAULT_SESSION'))

    # Verify IEEE-754 `CoordBits` / `diff_cp_changed_segments` canonicalization
    # handles `-0.0` vs `+0.0` and `1e12` scale (beyond `llround(v*1e9)`).
    neg_zero_cp = _WATERBOMB_SAT_CP.replace(' 0 ', ' -0.0 ')
    self.assertEmpty(
        fast_flat_folder.diff_cp_changed_segments(
            _WATERBOMB_SAT_CP, neg_zero_cp
        )
    )
    large_cp_m = '\n'.join([
        '1 0 0 2e12 0',
        '1 2e12 0 2e12 1e12',
        '1 2e12 1e12 0 1e12',
        '1 0 1e12 0 0',
        '2 1e12 0 1e12 1e12',
    ])
    large_cp_v = large_cp_m.replace('2 1e12 0 1e12 1e12', '3 1e12 0 1e12 1e12')
    self.assertLen(
        fast_flat_folder.diff_cp_changed_segments(large_cp_m, large_cp_v), 1
    )
    coord_sess = fast_flat_folder.FlatFolderSession()
    self.assertTrue(coord_sess.check_cp(_WATERBOMB_SAT_CP).flat_foldable)
    self.assertTrue(coord_sess.check_cp(neg_zero_cp).memo_cache_hit)
    self.assertTrue(coord_sess.check_cp(large_cp_m).flat_foldable)
    large_v_res = coord_sess.check_cp(large_cp_v)
    self.assertTrue(large_v_res.flat_foldable)
    self.assertTrue(large_v_res.support_cache_hit)

    # Small LRU bounds force concurrent insert/evict/splice-to-front under
    # `absl::Mutex` across 8 worker threads.
    shared_session = fast_flat_folder.FlatFolderSession(
        use_memo_cache=True,
        use_support_cache=True,
        use_component_cache=True,
        auto_seed_deltas=True,
        shadow_audit_every_n=5,
        max_memo_entries=24,
        max_support_entries=12,
        max_component_entries=48,
    )
    trifold_cp = '\n'.join([
        '1 0 0 3 0',
        '1 3 0 3 2',
        '1 3 2 0 2',
        '1 0 2 0 0',
        '2 1 0 1 2',
        '2 2 0 2 2',
    ])
    unassigned_cp = '\n'.join([
        '1 0 0 5 0',
        '1 5 0 5 1',
        '1 5 1 0 1',
        '1 0 1 0 0',
        '5 1 0 1 1',
        '2 4 0 4 1',
    ])
    work_items = [
        (_WATERBOMB_SAT_CP, True),
        (_WATERBOMB_INVERTED_SAT_CP, True),
        (_SPIKE_TACO_TACO_UNSAT_CP_0, False),
        (_SPIKE_TACO_TACO_UNSAT_CP_1, False),
        (trifold_cp, True),
        (unassigned_cp, True),
    ]
    suite = differential_cp_gen.generate_lattice_differential_suite(
        seed=20261109,
        num_synthetic_bases=16,
        base_cp_docs=[],
        mutations_per_base=2,
    )
    for case in suite[:24]:
      expected = bool(fast_flat_folder.check_cp(case.cp_doc).flat_foldable)
      work_items.append((case.cp_doc, expected))

    def _worker(task_id: int) -> tuple[bool, int]:
      cp_doc, expected_ff = work_items[task_id % len(work_items)]
      # Mix shared stream IDs (contending on `_prev_cp_by_stream`) and per-lane
      # stream IDs, plus stateless top-level entrypoints.
      stream = 'shared_stream' if (task_id % 3 == 0) else f'lane_{task_id % 4}'
      res_shared = shared_session.check_cp(
          cp_doc,
          stream_id=stream,
          verify_certificate=(task_id % 4 == 0),
      )
      if bool(res_shared.flat_foldable) != expected_ff:
        raise AssertionError(
            f'task {task_id}: got {res_shared.flat_foldable}, want'
            f' {expected_ff}'
        )
      if task_id % 5 == 0:
        stateless_res = fast_flat_folder.check_cp(
            cp_doc, verify_certificate=expected_ff
        )
        if bool(stateless_res.flat_foldable) != expected_ff:
          raise AssertionError(f'stateless mismatch on task {task_id}')
      if task_id % 7 == 0:
        fold_out = fast_flat_folder.analyze_fold_file(
            f't_{task_id}.cp', cp_doc, '.cp', expected_ff
        ).flat_folder_fold
        if bool(fold_out.flat_foldable) != expected_ff:
          raise AssertionError(f'analyze_fold_file mismatch on task {task_id}')
      if task_id % 11 == 0:
        sol = fast_flat_folder.solve_flat_folded_state(
            None, grid_size=3, cp_str=trifold_cp
        )
        if len(sol.face_orders) != 3:
          raise AssertionError('unexpected trifold face_orders length')
      c_stats = shared_session.cache_stats()
      s_stats = shared_session.shadow_audit_stats()
      return (
          bool(res_shared.flat_foldable),
          c_stats.memo_hits + s_stats.total_calls,
      )

    num_tasks = 192
    with futures.ThreadPoolExecutor(max_workers=8) as pool:
      results = list(pool.map(_worker, range(num_tasks)))

    self.assertLen(results, num_tasks)
    audit_stats = shared_session.shadow_audit_stats()
    self.assertEqual(audit_stats.mismatches, 0)
    self.assertGreater(audit_stats.shadow_audits_run, 0)
    cache_stats = shared_session.cache_stats()
    self.assertLessEqual(cache_stats.memo_entries, 24)
    self.assertLessEqual(cache_stats.support_entries, 12)
    self.assertLessEqual(cache_stats.component_entries, 48)

  def test_full_folded_form_and_multi_state_tie_break_parity(self):
    """Verifies all pairwise face orders, cells, multi-state counts, and tie-breaks."""
    from bp_pipeline.third_party.flat_folder import constraints as ff_constraints  # pylint: disable=g-import-not-at-top
    from bp_pipeline.third_party.flat_folder import face_constraints as ff_fc  # pylint: disable=g-import-not-at-top
    from bp_pipeline.third_party.flat_folder import folded_state as ff_fs  # pylint: disable=g-import-not-at-top
    from bp_pipeline.third_party.flat_folder import graph_creation as ff_gc  # pylint: disable=g-import-not-at-top
    from bp_pipeline.third_party.flat_folder import graph_topology as ff_gt  # pylint: disable=g-import-not-at-top
    from bp_pipeline.third_party.flat_folder import io_utils as ff_io  # pylint: disable=g-import-not-at-top
    from bp_pipeline.third_party.flat_folder import math_utils as ff_mu  # pylint: disable=g-import-not-at-top
    from bp_pipeline.third_party.flat_folder import solver as ff_solver  # pylint: disable=g-import-not-at-top

    ff_constraints.Constraints.build_implication_tables()

    def _ref_ordered_pairs_and_cells(cp_doc: str):
      v_coords, _, ev_map, e_assign, e_faces, f_verts, _ = ff_io.parse_document(
          cp_doc, 'cp', True
      )
      f_coords, f_orient = ff_fs.compute_folded_state(
          v_coords, f_verts, ev_map, e_assign
      )
      lines = [tuple(ff_mu.expand(edge, f_coords)) for edge in ev_map]
      pts, seg_pts, seg_edges, _ = ff_gc.create_graph_from_lines(lines)
      _, cell_verts = ff_gt.graph_to_adjacency_and_faces(pts, seg_pts)
      seg_cells, _ = ff_gt.build_edge_face_maps(seg_pts, cell_verts)
      cell_faces, face_cells = ff_fc.compute_cell_face_mappings(
          e_faces, f_verts, seg_pts, seg_edges, cell_verts, seg_cells
      )
      face_pairs = ff_fc.compute_transitive_face_pairs(
          e_faces, seg_pts, seg_edges, cell_verts, cell_faces
      )
      fp_to_idx = {pair: idx for idx, pair in enumerate(face_pairs)}
      ctable = ff_fc.build_face_pair_constraint_table(
          face_pairs, fp_to_idx, e_faces, seg_edges, cell_faces, seg_cells
      )
      f_comps = ff_fc.build_face_graph_components(face_cells, ctable)
      tc = {'all': 0, 'reduced': 0}
      init_a = ff_solver.get_initial_assignments_from_creases(
          e_faces, e_assign, f_orient, face_pairs, fp_to_idx
      )
      prop_a = ff_solver.propagate_initial_assignments(
          init_a,
          face_pairs,
          ctable,
          fp_to_idx,
          face_cells,
          cell_faces,
          f_comps,
          tc,
      )
      acomps = ff_solver.get_assignment_components(
          fp_to_idx,
          face_pairs,
          ctable,
          prop_a,
          face_cells,
          cell_faces,
          f_comps,
          tc,
      )
      solved = ff_solver.solve_all_components(
          fp_to_idx,
          face_pairs,
          ctable,
          list(prop_a),
          acomps,
          face_cells,
          cell_faces,
          f_comps,
          1,
      )
      ord_pairs = ff_fc.get_ordered_face_pairs_from_assignments(
          face_pairs, acomps, solved, [0] * len(solved)
      )
      c_ordered = ff_fc.sort_faces_in_cells_by_order(cell_faces, ord_pairs)
      kept_stacks = [[int(f) for f in st] for st in c_ordered if st]
      return sorted((int(a), int(b)) for a, b in ord_pairs), kept_stacks

    one_dof_cp = '\n'.join([
        '1 0 0 3 0',
        '1 3 0 3 1',
        '1 3 1 0 1',
        '1 0 1 0 0',
        '5 1 0 1 1',
    ])
    two_dof_cp = '\n'.join([
        '1 0 0 5 0',
        '1 5 0 5 1',
        '1 5 1 0 1',
        '1 0 1 0 0',
        '5 1 0 1 1',
        '5 3 0 3 1',
    ])
    trifold_cp = '\n'.join([
        '1 0 0 3 0',
        '1 3 0 3 2',
        '1 3 2 0 2',
        '1 0 2 0 0',
        '2 1 0 1 2',
        '2 2 0 2 2',
    ])

    cold_engine = _fast_flat_folder.FastFlatFolderEngine(0, 0, 0)
    inc_engine = _fast_flat_folder.FastFlatFolderEngine(0, 64, 64)
    opts = _fast_flat_folder.SolveOptions()
    opts.solution_limit = 1
    opts.compute_layer_order = True
    opts.flip_to_top_face_up = True

    for name, cp_doc, expected_multi in (
        ('one_dof', one_dof_cp, 2),
        ('two_dof', two_dof_cp, 6),
        ('trifold', trifold_cp, 2),
    ):
      with self.subTest(case=name):
        ref_fold = fold_cp.analyze_fold_file(
            name, cp_doc, '.cp', True
        ).flat_folder_fold
        ref_pairs, ref_stacks = _ref_ordered_pairs_and_cells(cp_doc)

        cpp_cold = cold_engine.analyze_cp(cp_doc, opts)
        cpp_inc_0 = inc_engine.analyze_cp(cp_doc, opts)
        cpp_inc_1 = inc_engine.analyze_cp(cp_doc, opts)
        self.assertTrue(cpp_inc_1.support_cache_hit)

        self.assertEqual(
            sorted((int(a), int(b)) for a, b in cpp_cold.ordered_face_pairs),
            ref_pairs,
        )
        self.assertEqual(
            list(cpp_inc_1.ordered_face_pairs),
            list(cpp_cold.ordered_face_pairs),
        )
        cpp_stacks = [
            [int(f) for f in st] for st in cpp_cold.cell_faces_ordered if st
        ]
        self.assertCountEqual(cpp_stacks, ref_stacks)
        self.assertEqual(
            list(cpp_inc_0.cell_faces_ordered),
            list(cpp_cold.cell_faces_ordered),
        )
        self.assertEqual(
            list(cpp_inc_1.cell_faces_ordered),
            list(cpp_cold.cell_faces_ordered),
        )

        got_fold = fast_flat_folder.analyze_fold_file(
            name, cp_doc, '.cp', True
        ).flat_folder_fold
        self.assertEqual(
            got_fold.folded_state_fold['faceOrders'],
            ref_fold.folded_state_fold['faceOrders'],
        )
        self.assertEqual(
            got_fold.folded_state_fold['faceOrientations'],
            ref_fold.folded_state_fold['faceOrientations'],
        )

        multi_res = fast_flat_folder.check_cp(cp_doc, solution_limit=100)
        self.assertEqual(multi_res.states, expected_multi)

  def test_fold_graph_bounds_and_malformed_cp_rejection(self):
    engine = _fast_flat_folder.FastFlatFolderEngine()
    vertices = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    assignments = [1, 1, 1, 1, 3]

    oob_edge = engine.analyze_fold_graph(
        vertices, [(0, 1), (1, 99), (2, 3), (3, 0), (0, 2)], assignments
    )
    self.assertFalse(oob_edge.ok_parse)
    self.assertFalse(oob_edge.flat_foldable)

    oob_face = engine.analyze_fold_graph(
        vertices,
        [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)],
        assignments,
        [[0, 1, 2], [0, 2, -1]],
    )
    self.assertFalse(oob_face.ok_parse)
    self.assertFalse(oob_face.flat_foldable)

    bad_cp = engine.analyze_cp('1 0 0 1 0\n1 0 0\n')
    self.assertFalse(bad_cp.ok_parse)
    self.assertFalse(bad_cp.flat_foldable)
    self.assertIn('line 2', bad_cp.conflict_reason)

    parsed_segs = _fast_flat_folder.parse_cp_document(_WATERBOMB_SAT_CP)
    self.assertGreaterEqual(len(parsed_segs), 8)
    with self.assertRaisesRegex(ValueError, 'line 2'):
      _fast_flat_folder.parse_cp_document('1 0 0 1 0\n1 0 0\n')

    v_coords, f_verts = engine.extract_unfolded_faces_cp(_WATERBOMB_SAT_CP)
    self.assertNotEmpty(v_coords)
    self.assertNotEmpty(f_verts)
    with self.assertRaisesRegex(ValueError, 'line 2'):
      engine.extract_unfolded_faces_cp('1 0 0 1 0\n1 0 0\n')

  def _make_one_crease_valid_cert(self):
    one_crease_cp = '\n'.join([
        '1 0 0 2 0',
        '1 2 0 2 2',
        '1 2 2 0 2',
        '1 0 2 0 0',
        '2 1 0 1 2',
    ])
    engine = _fast_flat_folder.FastFlatFolderEngine(0, 0, 0)
    opts = _fast_flat_folder.SolveOptions()
    opts.flip_to_top_face_up = True
    opts.compute_layer_order = True
    base = engine.analyze_cp(one_crease_cp, opts)
    self.assertTrue(base.flat_foldable)

    class _MutableCert:

      def __init__(self, res):
        self.flat_foldable = bool(res.flat_foldable)
        self.vertex_coords = list(res.vertex_coords)
        self.face_vertices = [list(f) for f in res.face_vertices]
        self.face_orientations = list(res.face_orientations)
        self.folded_vertices = list(res.folded_vertices)
        self.cell_points = list(res.cell_points)
        self.cell_vertices = [list(c) for c in res.cell_vertices]
        self.cell_faces_ordered = [list(st) for st in res.cell_faces_ordered]

    cert = _MutableCert(base)
    self.assertTrue(
        sat_certificate_checker.verify_cpp_sat_certificate(
            one_crease_cp, cert
        ).ok
    )
    return one_crease_cp, cert

  @parameterized.named_parameters(
      dict(
          testcase_name='small_primes_exceed_2_pow_50',
          primes=(3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53),
      ),
      dict(
          testcase_name='thirty_37bit_primes_exceed_float64_max',
          primes=(
              100000000003,
              100000000019,
              100000000057,
              100000000063,
              100000000069,
              100000000073,
              100000000091,
              100000000103,
              100000000129,
              100000000147,
              100000000163,
              100000000183,
              100000000187,
              100000000193,
              100000000199,
              100000000201,
              100000000211,
              100000000229,
              100000000241,
              100000000243,
              100000000253,
              100000000279,
              100000000297,
              100000000309,
              100000000333,
              100000000337,
              100000000339,
              100000000357,
              100000000367,
              100000000373,
          ),
      ),
  )
  def test_sat_certificate_checker_rejects_lcm_denom_blowup(self, primes):
    lines = [
        '1 0 0 2 0',
        '1 2 0 2 2',
        '1 2 2 0 2',
        '1 0 2 0 0',
    ]
    for p in primes:
      lines.append(f'4 1/{p} 0 1/{p} 2')
    cp_doc = '\n'.join(lines)
    num_faces = len(primes) + 1

    class _MinimalCert:
      flat_foldable = True
      vertex_coords = [(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)]
      face_vertices = [[0, 1, 2, 3]] * num_faces
      face_orientations = [False] * num_faces
      folded_vertices = vertex_coords
      cell_points = vertex_coords
      cell_vertices = [[0, 1, 2, 3]]
      cell_faces_ordered = [list(range(num_faces))]

    chk = sat_certificate_checker.verify_cpp_sat_certificate(
        cp_doc, _MinimalCert()
    )
    self.assertFalse(chk.ok)
    self.assertEqual(chk.reason, 'excessive_common_denominator')

    audit = sat_certificate_checker.audit_exact_folded_arrangement(cp_doc)
    self.assertFalse(audit['locally_closed'])
    self.assertEqual(audit['reason'], 'excessive_common_denominator')

  @parameterized.named_parameters(
      dict(
          testcase_name='zero_denominator',
          cp_doc='\n'.join([
              '1 0 0 2 0',
              '1 2 0 2 2',
              '1 2 2 0 2',
              '1 0 2 0 0',
              '2 1/0 0 1 2',
          ]),
      ),
      dict(
          testcase_name='positive_exponent_above_float64_max',
          cp_doc='\n'.join([
              '1 0 0 2e500 0',
              '1 2e500 0 2e500 2e500',
              '1 2e500 2e500 0 2e500',
              '1 0 2e500 0 0',
              '2 1e500 0 1e500 2e500',
          ]),
      ),
      dict(
          testcase_name='magnitude_above_1e100',
          cp_doc='\n'.join([
              '1 0 0 2e101 0',
              '1 2e101 0 2e101 2e101',
              '1 2e101 2e101 0 2e101',
              '1 0 2e101 0 0',
              '2 1e101 0 1e101 2e101',
          ]),
      ),
      dict(
          testcase_name='excessive_exponent_on_zero',
          cp_doc='\n'.join([
              '1 0e100000 0 2 0',
              '1 2 0 2 2',
              '1 2 2 0 2',
              '1 0 2 0 0',
              '2 1 0 1 2',
          ]),
      ),
      dict(
          testcase_name='huge_positive_exponent_no_stall',
          cp_doc='\n'.join([
              '1 0 0 1e100000000 0',
              '1 2 0 2 2',
              '1 2 2 0 2',
              '1 0 2 0 0',
              '2 1 0 1 2',
          ]),
      ),
      dict(
          testcase_name='huge_negative_exponent_no_stall',
          cp_doc='\n'.join([
              '1 0 0 1e-100000000 0',
              '1 2 0 2 2',
              '1 2 2 0 2',
              '1 0 2 0 0',
              '2 1 0 1 2',
          ]),
      ),
      dict(
          testcase_name='oversized_decimal_token_no_stall',
          cp_doc='\n'.join([
              f'1 0.{"0" * 5000}1 0 2 0',
              '1 2 0 2 2',
              '1 2 2 0 2',
              '1 0 2 0 0',
              '2 1 0 1 2',
          ]),
      ),
  )
  def test_sat_certificate_checker_rejects_adversarial_cp_tokens(self, cp_doc):
    _, cert = self._make_one_crease_valid_cert()
    chk = sat_certificate_checker.verify_cpp_sat_certificate(cp_doc, cert)
    self.assertFalse(chk.ok)
    self.assertEqual(chk.reason, 'cp_parse_error')

    audit = sat_certificate_checker.audit_exact_folded_arrangement(cp_doc)
    self.assertFalse(audit['locally_closed'])
    self.assertEqual(audit['reason'], 'cp_parse_error')

  @parameterized.named_parameters(
      dict(
          testcase_name='unfolded_vertex_nan',
          mutate='unfolded_vertex_nan',
          expected_reason='non_finite_solver_unfolded_vertex',
      ),
      dict(
          testcase_name='unfolded_vertex_inf',
          mutate='unfolded_vertex_inf',
          expected_reason='non_finite_solver_unfolded_vertex',
      ),
      dict(
          testcase_name='unfolded_vertex_scale_overflow',
          mutate='unfolded_vertex_scale_overflow',
          expected_reason='non_finite_solver_unfolded_vertex',
      ),
      dict(
          testcase_name='folded_vertex_nan',
          mutate='folded_vertex_nan',
          expected_reason='non_finite_solver_folded_vertex',
      ),
      dict(
          testcase_name='folded_vertex_inf',
          mutate='folded_vertex_inf',
          expected_reason='non_finite_solver_folded_vertex',
      ),
      dict(
          testcase_name='cell_point_nan',
          mutate='cell_point_nan',
          expected_reason='non_finite_solver_cell_point',
      ),
      dict(
          testcase_name='cell_point_neg_inf',
          mutate='cell_point_neg_inf',
          expected_reason='non_finite_solver_cell_point',
      ),
      dict(
          testcase_name='cell_point_scale_overflow',
          mutate='cell_point_scale_overflow',
          expected_reason='non_finite_solver_cell_point',
      ),
      dict(
          testcase_name='cell_faces_ordered_shorter',
          mutate='cell_faces_ordered_shorter',
          expected_reason='cell_geometry_length_mismatch',
      ),
      dict(
          testcase_name='cell_faces_ordered_longer',
          mutate='cell_faces_ordered_longer',
          expected_reason='cell_geometry_length_mismatch',
      ),
      dict(
          testcase_name='cell_face_index_oob',
          mutate='cell_face_index_oob',
          expected_reason='cell_face_index_out_of_range',
      ),
      dict(
          testcase_name='cell_face_index_negative',
          mutate='cell_face_index_negative',
          expected_reason='cell_face_index_out_of_range',
      ),
      dict(
          testcase_name='face_vertex_index_oob',
          mutate='face_vertex_index_oob',
          expected_reason='solver_face_vertex_index_out_of_range',
      ),
      dict(
          testcase_name='face_vertex_index_negative',
          mutate='face_vertex_index_negative',
          expected_reason='solver_face_vertex_index_out_of_range',
      ),
      dict(
          testcase_name='cell_vertex_index_oob',
          mutate='cell_vertex_index_oob',
          expected_reason='cell_vertex_index_out_of_range',
      ),
      dict(
          testcase_name='cell_vertex_index_negative',
          mutate='cell_vertex_index_negative',
          expected_reason='cell_vertex_index_out_of_range',
      ),
      dict(
          testcase_name='all_collinear_solver_face',
          mutate='all_collinear_solver_face',
          expected_reason='degenerate_solver_face',
      ),
      dict(
          testcase_name='all_collinear_solver_cell',
          mutate='all_collinear_solver_cell',
          expected_reason='degenerate_solver_cell',
      ),
  )
  def test_sat_certificate_checker_rejects_malformed_cpp_cert(
      self, mutate, expected_reason
  ):
    one_crease_cp, cert = self._make_one_crease_valid_cert()
    nonempty_ci = next(i for i, st in enumerate(cert.cell_faces_ordered) if st)
    if mutate == 'unfolded_vertex_nan':
      cert.vertex_coords[0] = (float('nan'), 0.0)
    elif mutate == 'unfolded_vertex_inf':
      cert.vertex_coords[0] = (float('inf'), 0.0)
    elif mutate == 'unfolded_vertex_scale_overflow':
      cert.vertex_coords[0] = (1e308, 0.0)
    elif mutate == 'folded_vertex_nan':
      cert.folded_vertices[0] = (float('nan'), float('nan'))
    elif mutate == 'folded_vertex_inf':
      cert.folded_vertices[0] = (float('inf'), 0.0)
    elif mutate == 'cell_point_nan':
      cert.cell_points[0] = (float('nan'), 0.0)
    elif mutate == 'cell_point_neg_inf':
      cert.cell_points[0] = (float('-inf'), 0.0)
    elif mutate == 'cell_point_scale_overflow':
      cert.cell_points[0] = (1e308, 0.0)
    elif mutate == 'cell_faces_ordered_shorter':
      cert.cell_faces_ordered.pop()
    elif mutate == 'cell_faces_ordered_longer':
      cert.cell_faces_ordered.append([0])
    elif mutate == 'cell_face_index_oob':
      cert.cell_faces_ordered[nonempty_ci][0] = 999
    elif mutate == 'cell_face_index_negative':
      last_fi = len(cert.face_vertices) - 1
      cert.cell_faces_ordered[nonempty_ci] = [
          -1 if fi == last_fi else fi
          for fi in cert.cell_faces_ordered[nonempty_ci]
      ]
    elif mutate == 'face_vertex_index_oob':
      cert.face_vertices[0][0] = 999
    elif mutate == 'face_vertex_index_negative':
      last_vi = len(cert.vertex_coords) - 1
      cert.face_vertices = [
          [-1 if vi == last_vi else vi for vi in cyc]
          for cyc in cert.face_vertices
      ]
    elif mutate == 'cell_vertex_index_oob':
      cert.cell_vertices[nonempty_ci][0] = 999
    elif mutate == 'cell_vertex_index_negative':
      last_cvi = len(cert.cell_points) - 1
      cert.cell_vertices[nonempty_ci] = [
          -1 if vi == last_cvi else vi for vi in cert.cell_vertices[nonempty_ci]
      ]
    elif mutate == 'all_collinear_solver_face':
      v0 = cert.face_vertices[0][0]
      v1 = cert.face_vertices[0][1]
      p0 = cert.vertex_coords[v0]
      p1 = cert.vertex_coords[v1]
      mid_vi = len(cert.vertex_coords)
      cert.vertex_coords.append(((p0[0] + p1[0]) * 0.5, (p0[1] + p1[1]) * 0.5))
      cert.folded_vertices.append(cert.folded_vertices[v0])
      cert.face_vertices[0] = [v0, mid_vi, v1]
    elif mutate == 'all_collinear_solver_cell':
      c0 = cert.cell_vertices[nonempty_ci][0]
      cert.cell_vertices[nonempty_ci] = [c0, c0, c0]
    else:
      self.fail(f'Unknown mutation: {mutate}')

    chk = sat_certificate_checker.verify_cpp_sat_certificate(
        one_crease_cp, cert
    )
    self.assertFalse(chk.ok)
    self.assertIn(expected_reason, chk.reason)

  def test_sat_certificate_checker_rejects_incomplete_and_malformed_wrappers(
      self,
  ):
    one_crease_cp, cert = self._make_one_crease_valid_cert()

    class _IncompleteCppRes:
      flat_foldable = True

    res_cpp = sat_certificate_checker.verify_cpp_sat_certificate(
        one_crease_cp, _IncompleteCppRes()
    )
    self.assertFalse(res_cpp.ok)
    self.assertEqual(res_cpp.reason, 'missing_solver_result_fields')

    res_missing_orient = (
        sat_certificate_checker.verify_reference_sat_certificate(
            one_crease_cp,
            unfolded_vertices=cert.vertex_coords,
            face_vertices=cert.face_vertices,
            folded_state_fold={'faceOrders': [[0, 1]]},
        )
    )
    self.assertFalse(res_missing_orient.ok)
    self.assertEqual(res_missing_orient.reason, 'missing_faceOrientations')

    res_nan_order = sat_certificate_checker.verify_reference_sat_certificate(
        one_crease_cp,
        unfolded_vertices=cert.vertex_coords,
        face_vertices=cert.face_vertices,
        folded_state_fold={
            'faceOrders': [[float('nan'), 1]],
            'faceOrientations': cert.face_orientations,
        },
    )
    self.assertFalse(res_nan_order.ok)
    self.assertEqual(res_nan_order.reason, 'invalid_solver_face_orders')

    res_bad_ref_pt = sat_certificate_checker.verify_reference_sat_certificate(
        one_crease_cp,
        unfolded_vertices=[(0.0,)],
        face_vertices=cert.face_vertices,
        folded_state_fold={
            'faceOrders': [[0, 1]],
            'faceOrientations': cert.face_orientations,
        },
    )
    self.assertFalse(res_bad_ref_pt.ok)
    self.assertEqual(res_bad_ref_pt.reason, 'invalid_solver_unfolded_vertex')

    res_bad_shaping_pt = (
        sat_certificate_checker.verify_shaping_record_sat_certificate(
            one_crease_cp,
            unfolded_vertices=[(0.0,)],
            face_vertices=cert.face_vertices,
            face_top_up=[True, False],
            face_orders=[0, 1],
        )
    )
    self.assertFalse(res_bad_shaping_pt.ok)
    self.assertEqual(
        res_bad_shaping_pt.reason, 'invalid_solver_unfolded_vertex'
    )

  def test_sat_certificate_checker_docstring_names_solve_result(self):
    doc = sat_certificate_checker.verify_cpp_sat_certificate.__doc__ or ''
    self.assertIn('SolveResult', doc)
    self.assertNotIn('AnalyzeResult', doc)

  def test_sat_certificate_checker_accepts_boundary_rationals(self):
    # `1 / 2^50` equals `5^50 * 1e-50`; both hit the exact derived denominator
    # ceiling without exceeding it, and `2e12` verifies scientific notation.
    exact_pow2_cp = '\n'.join([
        '1 0 0 2 0',
        '1 2 0 2 2',
        '1 2 2 0 2',
        '1 0 2 0 0',
        f'2 1/{1 << 50} 0 1/{1 << 50} 2',
    ])
    self.assertTrue(
        sat_certificate_checker.audit_exact_folded_arrangement(exact_pow2_cp)[
            'locally_closed'
        ]
    )
    sci_cp = '\n'.join([
        '1 0 0 2e12 0',
        '1 2e12 0 2e12 2e12',
        '1 2e12 2e12 0 2e12',
        '1 0 2e12 0 0',
        '2 1e12 0 1e12 2e12',
    ])
    self.assertTrue(
        sat_certificate_checker.audit_exact_folded_arrangement(sci_cp)[
            'locally_closed'
        ]
    )

  def test_j1a_hang2_returns_local_violation_within_time_limit(self):
    # `hang2.cp` must reject at local validation; the test target's timeout
    # enforces termination without flaky wall-clock assertions under sanitizers.
    session = fast_flat_folder.FlatFolderSession(
        use_memo_cache=True,
        use_support_cache=True,
        use_component_cache=True,
    )
    res = session.check_cp(_HANG2_CP)
    self.assertFalse(res.flat_foldable)
    self.assertEqual(res.exit_stage, 'local_violation')
    self.assertNotEmpty(res.local_violations)
    self.assertEqual(res.num_cells, 0)
    self.assertEqual(res.num_face_pairs, 0)
    self.assertGreater(res.num_faces, 0)

    stats = session.cache_stats()
    self.assertEqual(stats.support_entries, 0)
    self.assertEqual(stats.memo_entries, 1)

    cached = session.check_cp(_HANG2_CP)
    self.assertTrue(cached.memo_cache_hit)
    self.assertEqual(cached.exit_stage, 'local_violation')

    v_coords, f_verts = session.extract_unfolded_faces_cp(_HANG2_CP)
    self.assertNotEmpty(v_coords)
    self.assertNotEmpty(f_verts)
    self.assertEqual(session.cache_stats().support_entries, 0)

  def test_mutated_cp_fuzz_cold_warm_and_certificate_oracle(self):
    rng = random.Random(4242)
    base_suite = differential_cp_gen.generate_lattice_differential_suite(
        seed=20261125,
        num_synthetic_bases=20,
        base_cp_docs=[],
        mutations_per_base=2,
    )
    base_docs = [case.cp_doc for case in base_suite]

    cold = _fast_flat_folder.FastFlatFolderEngine(0, 0, 0)
    warm = _fast_flat_folder.FastFlatFolderEngine(64, 64, 256)
    opts = _fast_flat_folder.SolveOptions()
    opts.compute_layer_order = True
    opts.use_memo_cache = False

    mutated_docs = [_HANG2_CP]
    while len(mutated_docs) < 150:
      candidate = _mutate_cp_doc(rng, rng.choice(base_docs))
      if candidate.strip():
        mutated_docs.append(candidate)

    cert_ok = 0
    local_violation_count = 0
    unexpected_cert_failures: list[str] = []

    for idx, doc in enumerate(mutated_docs):
      res_cold = cold.analyze_cp(doc, opts)
      res_warm = warm.analyze_cp(doc, opts)

      cold_key = (
          res_cold.flat_foldable,
          res_cold.states,
          list(res_cold.face_orders),
          [list(stack) for stack in res_cold.cell_faces_ordered],
          bool(res_cold.local_violations),
          res_cold.num_cells,
          res_cold.num_face_pairs,
      )
      warm_key = (
          res_warm.flat_foldable,
          res_warm.states,
          list(res_warm.face_orders),
          [list(stack) for stack in res_warm.cell_faces_ordered],
          bool(res_warm.local_violations),
          res_warm.num_cells,
          res_warm.num_face_pairs,
      )
      self.assertEqual(
          warm_key,
          cold_key,
          msg=f'Cold/warm mismatch on mutated case {idx}',
      )

      if res_cold.exit_stage == 'local_violation':
        local_violation_count += 1
        self.assertEqual(res_cold.num_cells, 0)
        self.assertEqual(res_cold.num_face_pairs, 0)

      if res_cold.flat_foldable:
        chk = sat_certificate_checker.verify_cpp_sat_certificate(doc, res_cold)
        if chk.ok:
          cert_ok += 1
        else:
          unexpected_cert_failures.append(f'case {idx}: {chk.reason}')

    self.assertEmpty(unexpected_cert_failures)
    self.assertGreaterEqual(cert_ok, 15)
    self.assertGreaterEqual(local_violation_count, 20)

  def test_j2_exact_path_rejects_holonomy_and_interior_boundary_conflicts(self):
    from bp_pipeline.third_party.flat_folder import io_utils as ff_io_utils  # pylint: disable=g-import-not-at-top

    # 1. Punctured annulus with even orientation parity and non-zero rigid
    #    translation holonomy (`holonomy_cp`).
    holonomy_cp = '\n'.join([
        '1 0 0 6 0',
        '1 6 0 6 4',
        '1 6 4 0 4',
        '1 0 4 0 0',
        '1 2 1 4 1',
        '1 4 1 4 3',
        '1 4 3 2 3',
        '1 2 3 2 1',
        '4 0 1 2 1',
        '4 4 1 6 1',
        '4 0 3 2 3',
        '4 4 3 6 3',
        '2 2 3 2 4',
        '3 3 3 3 4',
    ])
    # Must be rejected by C++ itself even when `verify_certificate=False`.
    holo_res = fast_flat_folder.check_cp(holonomy_cp, verify_certificate=False)
    self.assertFalse(holo_res.flat_foldable)
    self.assertTrue(holo_res.used_exact_octilinear)
    self.assertEqual(holo_res.exit_stage, 'folded_geometry_error')
    self.assertEqual(holo_res.states, 0)

    holo_fold = fast_flat_folder.analyze_fold_file(
        'holonomy.cp', holonomy_cp, '.cp', True
    ).flat_folder_fold
    self.assertFalse(holo_fold.flat_foldable)
    self.assertEqual(holo_fold.states, 0)

    # 2. Smallest 3x3 square with one interior `B`-coded segment (`1 1 2 1 3`):
    #    C++ exact path rejects on orientation parity conflict, while Python
    #    reference `fold_cp` falsely returns SAT (refuted by exact checker).
    valid_3x3_cp = '\n'.join([
        '1 0 0 1 0',
        '1 1 0 2 0',
        '1 2 0 3 0',
        '1 0 3 1 3',
        '1 1 3 2 3',
        '1 2 3 3 3',
        '1 0 0 0 1',
        '1 0 1 0 2',
        '1 0 2 0 3',
        '1 3 0 3 1',
        '1 3 1 3 2',
        '1 3 2 3 3',
        '2 1 0 1 1',
        '2 1 1 1 2',
        '2 1 2 1 3',
        '3 2 0 2 1',
        '3 2 1 2 2',
        '3 2 2 2 3',
        '2 0 1 1 1',
        '3 1 1 2 1',
        '2 2 1 3 1',
        '3 0 2 1 2',
        '2 1 2 2 2',
        '3 2 2 3 2',
    ])
    interior_b_3x3_cp = valid_3x3_cp.replace('2 1 2 1 3', '1 1 2 1 3', 1)
    even_parity_closure_3x3_cp = interior_b_3x3_cp.replace(
        '2 1 2 2 2', '1 1 2 2 2', 1
    )

    valid_res = fast_flat_folder.check_cp(valid_3x3_cp)
    self.assertTrue(valid_res.flat_foldable)
    self.assertTrue(valid_res.certificate_verified)

    odd_res = fast_flat_folder.check_cp(
        interior_b_3x3_cp, verify_certificate=False
    )
    self.assertFalse(odd_res.flat_foldable)
    self.assertTrue(odd_res.used_exact_octilinear)
    self.assertEqual(odd_res.exit_stage, 'folded_geometry_error')
    self.assertEqual(odd_res.states, 0)

    even_res = fast_flat_folder.check_cp(
        even_parity_closure_3x3_cp, verify_certificate=False
    )
    self.assertFalse(even_res.flat_foldable)
    self.assertTrue(even_res.used_exact_octilinear)
    self.assertEqual(even_res.exit_stage, 'folded_geometry_error')
    self.assertEqual(even_res.states, 0)

    # Confirm Python reference falsely says SAT and exact checker refutes it.
    ref_b = fold_cp.analyze_fold_file(
        'interior_b_3x3.cp', interior_b_3x3_cp, '.cp', True
    ).flat_folder_fold
    self.assertTrue(ref_b.flat_foldable)
    v_coords, _, _, _, _, f_verts, _ = ff_io_utils.parse_document(
        interior_b_3x3_cp, 'cp', True
    )
    ref_chk = sat_certificate_checker.verify_reference_sat_certificate(
        interior_b_3x3_cp,
        unfolded_vertices=v_coords,
        face_vertices=f_verts,
        folded_state_fold=ref_b.folded_state_fold,
    )
    self.assertFalse(ref_chk.ok)
    self.assertEqual(ref_chk.reason, 'orientation_parity_conflict')

    # Runtime shadow auditor classifies Python's refuted false SAT as
    # `reference_incorrect` without raising `FastFlatFolderParityError`.
    audit_sess = fast_flat_folder.FlatFolderSession(shadow_audit_every_n=1)
    audit_res = audit_sess.check_cp(interior_b_3x3_cp)
    self.assertFalse(audit_res.flat_foldable)
    self.assertTrue(audit_res.shadow_audited)
    audit_stats = audit_sess.shadow_audit_stats()
    self.assertEqual(audit_stats.reference_incorrect, 1)
    self.assertEqual(audit_stats.mismatches, 0)

    # 3. Mutated-pattern fuzz oracle: fixed seed verifying zero second-arrival
    #    false SATs in raw C++ and zero checker-refuted SATs under default
    #    `check_cp`.
    rng = random.Random(4242)
    base_docs = [
        c.cp_doc
        for c in differential_cp_gen.generate_lattice_differential_suite(
            seed=20261125,
            num_synthetic_bases=20,
            base_cp_docs=[],
            mutations_per_base=2,
        )
    ]
    engine = _fast_flat_folder.FastFlatFolderEngine(0, 0, 0)
    opts = _fast_flat_folder.SolveOptions()
    opts.compute_layer_order = True
    j2_reasons = frozenset({
        'orientation_parity_conflict',
        'transform_holonomy_conflict',
        'folded_vertex_closure_mismatch',
    })
    mutated_sat_checked = 0
    for _ in range(300):
      lines = rng.choice(base_docs).splitlines()
      for _ in range(rng.randint(1, 2)):
        idx = rng.randrange(len(lines))
        parts = lines[idx].split()
        if len(parts) == 5:
          parts[0] = str(rng.choice((1, 2, 3, 4, 5)))
          lines[idx] = ' '.join(parts)
      doc = '\n'.join(lines)
      cpp_res = engine.analyze_cp(doc, opts)
      if cpp_res.flat_foldable:
        chk = sat_certificate_checker.verify_cpp_sat_certificate(doc, cpp_res)
        self.assertNotIn(chk.reason, j2_reasons, msg=doc)
        self.assertFalse(
            chk.reason.startswith('face_orientation_mismatch'), msg=doc
        )
      # Public `check_cp` with default arguments must never return SAT on a
      # pattern refuted by `sat_certificate_checker`.
      pub_res = fast_flat_folder.check_cp(doc)
      if pub_res.flat_foldable:
        self.assertTrue(pub_res.certificate_verified)
        mutated_sat_checked += 1
    self.assertGreater(mutated_sat_checked, 10)

  def test_public_check_cp_defaults_to_verify_certificate_true(self):
    # Top-level `check_cp` and `FlatFolderSession.check_cp` verify SAT
    # certificates by default.
    top_res = fast_flat_folder.check_cp(_WATERBOMB_SAT_CP)
    self.assertTrue(top_res.flat_foldable)
    self.assertTrue(top_res.certificate_verified)

    session = fast_flat_folder.FlatFolderSession()
    sess_res = session.check_cp(_WATERBOMB_SAT_CP, stream_id=None)
    self.assertTrue(sess_res.flat_foldable)
    self.assertTrue(sess_res.certificate_verified)

    # Explicit opt-out skips exact certificate checking (used by search loops).
    unverified_res = session.check_cp(
        _WATERBOMB_SAT_CP, stream_id=None, verify_certificate=False
    )
    self.assertTrue(unverified_res.flat_foldable)
    self.assertFalse(unverified_res.certificate_verified)

    # Float-path holonomy pattern: `2.333333` does not quantize onto C++'s
    # exact `kBaseScales`, so C++ takes the float path (`used_exact_octilinear
    # == False`) where exact second-arrival checks do not run; default
    # `verify_certificate=True` still protects public callers.
    float_holonomy_cp = '\n'.join([
        '1 0 0 6 0',
        '1 6 0 6 4',
        '1 6 4 0 4',
        '1 0 4 0 0',
        '1 2 1 4 1',
        '1 4 1 4 3',
        '1 4 3 2 3',
        '1 2 3 2 1',
        '4 0 1 2 1',
        '4 4 1 6 1',
        '4 0 3 2 3',
        '4 4 3 6 3',
        '2 2 3 2 4',
        '3 2.333333 3 2.333333 4',
    ])
    raw_float = fast_flat_folder.check_cp(
        float_holonomy_cp, verify_certificate=False
    )
    self.assertFalse(raw_float.used_exact_octilinear)
    self.assertTrue(raw_float.flat_foldable)

    protected_top = fast_flat_folder.check_cp(float_holonomy_cp)
    self.assertEqual(
        protected_top.verdict, fast_flat_folder.FoldVerdict.NOT_FOLDABLE
    )
    self.assertFalse(protected_top.flat_foldable)
    self.assertFalse(protected_top.certificate_verified)
    self.assertEqual(protected_top.exit_stage, 'certificate_rejected')
    self.assertIn('transform_holonomy_conflict', protected_top.conflict_reason)

    protected_sess = session.check_cp(float_holonomy_cp, stream_id=None)
    self.assertEqual(
        protected_sess.verdict, fast_flat_folder.FoldVerdict.NOT_FOLDABLE
    )
    self.assertFalse(protected_sess.flat_foldable)
    self.assertEqual(protected_sess.exit_stage, 'certificate_rejected')

    # Foldable pattern off the 45-degree lines (single valley crease along
    # direction (2, 1), ~26.6 deg): the engine finds it foldable, while default
    # `verify_certificate=True` cannot check it with the octilinear exact
    # checker (`non_octilinear_cp`) and returns `FoldVerdict.COULD_NOT_DECIDE`
    # (`flat_foldable=None`, `exit_stage='could_not_decide'`), NOT
    # `NOT_FOLDABLE`.
    foldable_non_octilinear_cp = '\n'.join([
        '1 0 0 2 0',
        '1 2 0 2 2',
        '1 2 2 0 2',
        '1 0 2 0 0',
        '3 0 0 2 1',
    ])
    raw_non_oct = fast_flat_folder.check_cp(
        foldable_non_octilinear_cp, verify_certificate=False
    )
    self.assertFalse(raw_non_oct.used_exact_octilinear)
    self.assertEqual(raw_non_oct.verdict, fast_flat_folder.FoldVerdict.FOLDABLE)
    self.assertTrue(raw_non_oct.flat_foldable)
    self.assertFalse(raw_non_oct.certificate_verified)

    for backend in ('cpp', 'python_ref'):
      default_non_oct = fast_flat_folder.check_cp(
          foldable_non_octilinear_cp, backend=backend
      )
      self.assertEqual(
          default_non_oct.verdict,
          fast_flat_folder.FoldVerdict.COULD_NOT_DECIDE,
          msg=f'backend={backend}',
      )
      self.assertIsNone(default_non_oct.flat_foldable, msg=f'backend={backend}')
      self.assertFalse(default_non_oct.certificate_verified)
      self.assertEqual(default_non_oct.exit_stage, 'could_not_decide')
      self.assertEqual(default_non_oct.states, 0)
      self.assertIn('non_octilinear_cp', default_non_oct.conflict_reason or '')

    # `taco_only=True` and `ignore_flat_creases=True` skip exact confirmation,
    # and `certificate_verified` reports False.
    taco_res = session.check_cp(
        _WATERBOMB_SAT_CP, stream_id=None, taco_only=True
    )
    self.assertEqual(
        taco_res.verdict, fast_flat_folder.FoldVerdict.COULD_NOT_DECIDE
    )
    self.assertIsNone(taco_res.flat_foldable)
    self.assertTrue(taco_res.passed_taco_stage)
    self.assertFalse(taco_res.certificate_verified)
    no_flat_res = session.check_cp(
        _WATERBOMB_SAT_CP, stream_id=None, ignore_flat_creases=True
    )
    self.assertEqual(no_flat_res.verdict, fast_flat_folder.FoldVerdict.FOLDABLE)
    self.assertTrue(no_flat_res.flat_foldable)
    self.assertFalse(no_flat_res.certificate_verified)

  def test_j1b_hang_rows_and_three_valued_verdict(self):
    # Near-lattice float-jitter sweep-cap regression: 9 near-lattice
    # segments with ~1e-12 float jitter that previously caused an infinite loop
    # and unbounded memory growth in `GeneralSweepRunner`. Termination is
    # enforced by this test target's timeout (no flaky wall-clock assertions).
    hang_rows_cp = '\n'.join([
        '5 2.0000000000006684 3.0 3.0 1.0000000000005347',
        '0 -7.695333816428548e-13 2.0 1.0 0.9999999999996401',
        '4 1.0000000000006535 0.0 2.0 8.786379660612406e-13',
        '1 2.9999999999999836 3.0 0.0 3.0000000000006963',
        '2 7.575194486189778e-13 3.0 1.0 1.9999999999999345',
        '1 -1.4783135079967974e-13 3.0 0.0 -2.0446541825234597e-13',
        '1 2.9999999999991513 0.0 3.0 2.999999999999662',
        '5 1.9999999999999172 1.0 1.0 3.000000000000856',
        '1 -9.994966226387114e-13 0.0 3.0 -3.153723961478634e-13',
    ])

    for verify_cert in (True, False):
      res = fast_flat_folder.check_cp(
          hang_rows_cp, verify_certificate=verify_cert
      )
      self.assertEqual(
          res.verdict, fast_flat_folder.FoldVerdict.COULD_NOT_DECIDE
      )
      self.assertIsNone(res.flat_foldable)
      self.assertEqual(res.exit_stage, 'could_not_decide')
      self.assertEqual(res.states, 0)
      self.assertFalse(res.certificate_verified)
      self.assertIn('Sweep event limit exceeded', res.conflict_reason or '')

    # Shadow audit enabled must safely skip `COULD_NOT_DECIDE` without raising.
    audit_session = fast_flat_folder.FlatFolderSession(shadow_audit_every_n=1)
    sess_res = audit_session.check_cp(hang_rows_cp, stream_id=None)
    self.assertEqual(
        sess_res.verdict, fast_flat_folder.FoldVerdict.COULD_NOT_DECIDE
    )
    self.assertIsNone(sess_res.flat_foldable)
    self.assertFalse(sess_res.shadow_audited)

    # `solve_flat_folded_state` surfaces `could not decide` distinctly.
    with self.assertRaisesRegex(ValueError, 'could not decide'):
      audit_session.solve_flat_folded_state(
          cp=None, grid_size=3, cp_str=hang_rows_cp, stream_id=None
      )

    # Raw pybind11 engine binding exposes `FoldVerdict.COULD_NOT_DECIDE` and
    # `ExitStage.COULD_NOT_DECIDE` directly.
    raw_engine = _fast_flat_folder.FastFlatFolderEngine(0, 0, 0, 0)
    raw_res = raw_engine.analyze_cp(
        hang_rows_cp, _fast_flat_folder.SolveOptions()
    )
    self.assertTrue(raw_res.ok_parse)
    self.assertEqual(
        raw_res.verdict, _fast_flat_folder.FoldVerdict.COULD_NOT_DECIDE
    )
    self.assertIsNone(raw_res.flat_foldable)
    self.assertEqual(
        raw_res.stage, _fast_flat_folder.ExitStage.COULD_NOT_DECIDE
    )
    self.assertEqual(raw_res.exit_stage, 'could_not_decide')

    # Disconnected `.fold` graph (two separate folded squares, each split by a
    # valley diagonal) is also classified as `COULD_NOT_DECIDE` (engine
    # limitation, not a refutation).
    disc_res = raw_engine.analyze_fold_graph(
        [
            (0.0, 0.0),
            (1.0, 0.0),
            (1.0, 1.0),
            (0.0, 1.0),
            (3.0, 0.0),
            (4.0, 0.0),
            (4.0, 1.0),
            (3.0, 1.0),
        ],
        [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),
            (0, 2),
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 4),
            (4, 6),
        ],
        [1, 1, 1, 1, 3, 1, 1, 1, 1, 3],
        [[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]],
        _fast_flat_folder.SolveOptions(),
    )
    self.assertTrue(disc_res.ok_parse)
    self.assertEqual(
        disc_res.verdict, _fast_flat_folder.FoldVerdict.COULD_NOT_DECIDE
    )
    self.assertIsNone(disc_res.flat_foldable)
    self.assertEqual(disc_res.exit_stage, 'could_not_decide')
    self.assertIn('Disconnected face', disc_res.conflict_reason)

    # Folded-stage `InvalidArgumentError` on valid `<= 1e100` ingress coords:
    # `[0, 1e100] x [0, 1e99]` folded across `x = 0.9e100` reflects face 1 to
    # `x = 1.8e100 > 1e100`, so `BuildFoldedCellGraph` returns
    # `InvalidArgumentError`. Not an exact proof, so must answer
    # `COULD_NOT_DECIDE`, never `NOT_FOLDABLE`.
    fold_overflow_res = raw_engine.analyze_fold_graph(
        [
            (0.0, 0.0),
            (0.9e100, 0.0),
            (1e100, 0.0),
            (1e100, 1e99),
            (0.9e100, 1e99),
            (0.0, 1e99),
        ],
        [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0), (1, 4)],
        [1, 1, 1, 1, 1, 1, 3],
        [[1, 2, 3, 4], [0, 1, 4, 5]],
        _fast_flat_folder.SolveOptions(),
    )
    self.assertTrue(fold_overflow_res.ok_parse)
    self.assertEqual(
        fold_overflow_res.verdict,
        _fast_flat_folder.FoldVerdict.COULD_NOT_DECIDE,
    )
    self.assertIsNone(fold_overflow_res.flat_foldable)
    self.assertEqual(fold_overflow_res.exit_stage, 'could_not_decide')
    self.assertIn('1e100', fold_overflow_res.conflict_reason)

    # `FastFoldResult.verdict` is derived directly from `flat_foldable`.
    for ff_val, expected_v in (
        (True, fast_flat_folder.FoldVerdict.FOLDABLE),
        (False, fast_flat_folder.FoldVerdict.NOT_FOLDABLE),
        (None, fast_flat_folder.FoldVerdict.COULD_NOT_DECIDE),
    ):
      derived = fast_flat_folder.FastFoldResult(
          flat_foldable=ff_val,
          passed_taco_stage=bool(ff_val),
          states=1 if ff_val else 0,
          exit_stage='sat' if ff_val else 'could_not_decide',
      )
      self.assertEqual(derived.verdict, expected_v)

  @parameterized.named_parameters(
      dict(
          testcase_name='duplicate_undirected_edge',
          vertices=[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
          edges=[(0, 1), (1, 2), (2, 3), (3, 0), (0, 2), (2, 0)],
          assignments=[1, 1, 1, 1, 3, 2],
          faces=[[0, 1, 2], [0, 2, 3]],
          expected_substr='Duplicate edge',
      ),
      dict(
          testcase_name='repeated_vertex_in_face',
          vertices=[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
          edges=[(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)],
          assignments=[1, 1, 1, 1, 3],
          faces=[[0, 1, 1, 2], [0, 2, 3]],
          expected_substr='Repeated vertex',
      ),
      dict(
          testcase_name='undeclared_face_side',
          vertices=[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
          edges=[(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)],
          assignments=[1, 1, 1, 1, 3],
          faces=[[0, 1, 2], [0, 2, 3], [0, 3, 1]],
          expected_substr='undeclared edge',
      ),
      dict(
          testcase_name='duplicate_face_reuses_half_edge',
          vertices=[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
          edges=[(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)],
          assignments=[1, 1, 1, 1, 3],
          faces=[[0, 1, 2], [0, 1, 2], [0, 2, 3]],
          expected_substr='half-edge',
      ),
      dict(
          testcase_name='three_faces_share_edge',
          vertices=[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.5, 0.5)],
          edges=[(0, 1), (1, 2), (2, 3), (3, 0), (0, 2), (0, 4), (4, 2)],
          assignments=[1, 1, 1, 1, 3, 1, 1],
          faces=[[0, 1, 2], [0, 2, 3], [0, 4, 2]],
          expected_substr='half-edge',
      ),
      dict(
          testcase_name='mixed_winding_reuses_directed_half_edge',
          vertices=[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
          edges=[(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)],
          assignments=[1, 1, 1, 1, 3],
          faces=[[0, 1, 2], [3, 2, 0]],
          expected_substr='half-edge',
      ),
  )
  def test_j3_analyze_fold_graph_rejects_malformed_topology(
      self, vertices, edges, assignments, faces, expected_substr
  ):
    engine = _fast_flat_folder.FastFlatFolderEngine()
    res = engine.analyze_fold_graph(vertices, edges, assignments, faces)
    self.assertFalse(res.ok_parse)
    self.assertFalse(res.flat_foldable)
    self.assertEqual(res.exit_stage, 'parse_error')
    self.assertIn(expected_substr, res.conflict_reason)

  def test_j4_conflicting_collinear_crease_codes_and_flat_under_crease(self):
    square = '1 0 0 4 0\n1 4 0 4 4\n1 4 4 0 4\n1 0 4 0 0\n'
    base_valid_cp = square + '2 0 1.9375 4 1.9375\n'
    base_res = fast_flat_folder.check_cp(base_valid_cp, verify_certificate=True)
    self.assertTrue(base_res.flat_foldable)
    self.assertTrue(base_res.certificate_verified)

    cert_engine = _fast_flat_folder.FastFlatFolderEngine()
    cert_opts = _fast_flat_folder.SolveOptions()
    cert_opts.compute_layer_order = True
    base_cpp_res = cert_engine.analyze_cp(base_valid_cp, cert_opts)
    self.assertTrue(base_cpp_res.flat_foldable)

    # Conflicting M+V, M+U, and B+M must be rejected by BOTH C++ (even with
    # `verify_certificate=False`) and `sat_certificate_checker`.
    conflict_cases = [
        (
            'M_vs_V_exact_repro',
            square + '2 0 1.9375 4 1.9375\n3 0 1.9375 4 1.9375\n',
            'lines 5 and 6',
        ),
        (
            'M_vs_U',
            square + '2 0 2 4 2\n5 0 2 4 2\n',
            'lines 5 and 6',
        ),
        (
            'B_vs_M_exterior',
            square + '2 0 0 4 0\n2 0 2 4 2\n',
            'lines 1 and 5',
        ),
    ]
    for name, cp_doc, expected_lines in conflict_cases:
      with self.subTest(conflict=name):
        res_no_cert = fast_flat_folder.check_cp(
            cp_doc, verify_certificate=False
        )
        self.assertFalse(res_no_cert.flat_foldable)
        self.assertEqual(res_no_cert.exit_stage, 'parse_error')
        self.assertIn(
            'Conflicting collinear crease codes', res_no_cert.conflict_reason
        )
        self.assertIn(expected_lines, res_no_cert.conflict_reason)

        chk = sat_certificate_checker.verify_cpp_sat_certificate(
            cp_doc, base_cpp_res
        )
        self.assertFalse(chk.ok)
        self.assertEqual(chk.reason, 'conflicting_collinear_crease_codes')

    # Flat auxiliary `F` (code 4) under M, V, or B (in either order) is NOT a
    # conflict: the crease's code stands in both C++ and the exact checker.
    valid_flat_overlay_cases = [
        ('F_then_M', square + '4 0 1.9375 4 1.9375\n2 0 1.9375 4 1.9375\n'),
        ('M_then_F', square + '2 0 1.9375 4 1.9375\n4 0 1.9375 4 1.9375\n'),
        ('F_under_V', square + '4 0 1.9375 4 1.9375\n3 0 1.9375 4 1.9375\n'),
        ('F_under_B', square + '4 0 0 4 0\n2 0 1.9375 4 1.9375\n'),
    ]
    for name, cp_doc in valid_flat_overlay_cases:
      with self.subTest(valid_flat_overlay=name):
        res = fast_flat_folder.check_cp(cp_doc, verify_certificate=True)
        self.assertTrue(res.flat_foldable)
        self.assertEqual(res.exit_stage, 'sat')
        self.assertTrue(res.certificate_verified)

    # Warm `support_cache` hit on `FastFlatFolderEngine`: M+M populates the
    # cache, then M+V hits the same support key and must still fail parse.
    engine = _fast_flat_folder.FastFlatFolderEngine()
    opts = _fast_flat_folder.SolveOptions()
    opts.use_support_cache = True
    opts.use_memo_cache = False
    mm_doc = square + '2 0 1.9375 4 1.9375\n2 0 1.9375 4 1.9375\n'
    mv_doc = square + '2 0 1.9375 4 1.9375\n3 0 1.9375 4 1.9375\n'
    self.assertTrue(engine.analyze_cp(mm_doc, opts).flat_foldable)
    warm_mv = engine.analyze_cp(mv_doc, opts)
    self.assertFalse(warm_mv.ok_parse)
    self.assertFalse(warm_mv.flat_foldable)
    self.assertEqual(warm_mv.exit_stage, 'parse_error')
    self.assertIn('lines 5 and 6', warm_mv.conflict_reason)

  def test_j5_crease_codes_accept_0_to_5_and_reject_out_of_range(self):
    # Code 0 maps to unassigned (5), matching Python `io_utils.cp_to_lines`.
    seg_zero = _fast_flat_folder.InputSegment(0.0, 0.0, 1.0, 0.0, 0)
    self.assertEqual(seg_zero.assignment, 5)
    seg_zero.assignment = 0
    self.assertEqual(seg_zero.assignment, 5)

    zero_code_cp = '\n'.join([
        '1 0 0 2 0',
        '1 2 0 2 2',
        '1 2 2 0 2',
        '1 0 2 0 0',
        '0 1 0 1 2',
    ])
    parsed = _fast_flat_folder.parse_cp_document(zero_code_cp)
    self.assertLen(parsed, 5)
    self.assertEqual(parsed[4].assignment, 5)
    zero_res = fast_flat_folder.check_cp(zero_code_cp)
    self.assertTrue(zero_res.flat_foldable)
    self.assertTrue(zero_res.certificate_verified)

    engine = _fast_flat_folder.FastFlatFolderEngine()
    vertices = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)]
    fg_zero = engine.analyze_fold_graph(vertices, edges, [1, 1, 1, 1, 0])
    self.assertTrue(fg_zero.ok_parse)
    self.assertTrue(fg_zero.flat_foldable)

    for bad_code in (-4, -1, 6, 7, 99):
      with self.subTest(bad_code=bad_code):
        with self.assertRaisesRegex(ValueError, 'crease'):
          _fast_flat_folder.InputSegment(0.0, 0.0, 1.0, 0.0, bad_code)
        with self.assertRaisesRegex(ValueError, 'crease'):
          seg_zero.assignment = bad_code

        bad_cp = zero_code_cp.replace('0 1 0 1 2', f'{bad_code} 1 0 1 2')
        with self.assertRaisesRegex(ValueError, 'line 5'):
          _fast_flat_folder.parse_cp_document(bad_cp)

        res = fast_flat_folder.check_cp(bad_cp)
        self.assertFalse(res.flat_foldable)
        self.assertEqual(res.exit_stage, 'parse_error')
        self.assertIn('line 5', res.conflict_reason or '')

        fg_bad = engine.analyze_fold_graph(
            vertices, edges, [1, 1, 1, 1, bad_code]
        )
        self.assertFalse(fg_bad.ok_parse)
        self.assertFalse(fg_bad.flat_foldable)
        self.assertEqual(fg_bad.exit_stage, 'parse_error')

  @parameterized.named_parameters(
      dict(testcase_name='at_vertex', extra_line='3 1 0 1 0'),
      dict(testcase_name='on_crease', extra_line='3 1 1 1 1'),
      dict(testcase_name='v_in_face', extra_line='3 0.5 0.5 0.5 0.5'),
      dict(testcase_name='f_in_face', extra_line='4 0.5 0.5 0.5 0.5'),
      dict(testcase_name='outside_paper', extra_line='3 5 5 5 5'),
  )
  def test_j6_zero_length_segments_rejected_at_ingress(self, extra_line):
    cp_doc = '\n'.join([
        '1 0 0 2 0',
        '1 2 0 2 2',
        '1 2 2 0 2',
        '1 0 2 0 0',
        '3 1 0 1 2',
        extra_line,
    ])
    with self.assertRaisesRegex(ValueError, 'line 6'):
      _fast_flat_folder.parse_cp_document(cp_doc)
    res = fast_flat_folder.check_cp(cp_doc)
    self.assertFalse(res.flat_foldable)
    self.assertEqual(res.exit_stage, 'parse_error')
    self.assertIn('line 6', res.conflict_reason or '')
    self.assertIn('zero-length', (res.conflict_reason or '').lower())

    audit = sat_certificate_checker.audit_exact_folded_arrangement(cp_doc)
    self.assertFalse(audit['locally_closed'])
    self.assertEqual(audit['reason'], 'zero_length_segment_at_line_6')

  @parameterized.named_parameters(
      dict(
          testcase_name='minimal_49_byte_eps_default',
          cp_doc='\n'.join([
              '1 0 0 4 0',
              '1 4 0 4 4',
              '1 4 4 0 4',
              '1 0 4 0 0',
              '1 2 2 2 2',
          ]),
          kawasaki_epsilon=0.05,
      ),
      dict(
          testcase_name='minimal_49_byte_eps_zero',
          cp_doc='\n'.join([
              '1 0 0 4 0',
              '1 4 0 4 4',
              '1 4 4 0 4',
              '1 0 4 0 0',
              '1 2 2 2 2',
          ]),
          kawasaki_epsilon=0.0,
      ),
      dict(
          testcase_name='minimal_f_code_eps_zero',
          cp_doc='\n'.join([
              '1 0 0 3 0',
              '1 3 0 3 3',
              '1 3 3 0 3',
              '1 0 3 0 0',
              '4 2 2 2 2',
          ]),
          kawasaki_epsilon=0.0,
      ),
      dict(
          testcase_name='fractional_f_code_eps_1e4',
          cp_doc='\n'.join([
              '1 0 0 4 0',
              '1 4 0 4 4',
              '1 4 4 0 4',
              '1 0 4 0 0',
              '4 2.5 3.5 2.5 3.5',
          ]),
          kawasaki_epsilon=1e-4,
      ),
      dict(
          testcase_name='fractional_f_code_eps_default',
          cp_doc='\n'.join([
              '1 0 0 4 0',
              '1 4 0 4 4',
              '1 4 4 0 4',
              '1 0 4 0 0',
              '4 2.5 3.5 2.5 3.5',
          ]),
          kawasaki_epsilon=0.05,
      ),
      dict(
          testcase_name='multi_zero_length_f_eps_default',
          cp_doc='\n'.join([
              '1 0 0 2 0',
              '1 2 0 2 2',
              '1 2 2 0 2',
              '1 0 2 0 0',
              '4 1 0 1 0',
              '4 2 0 2 2',
              '4 1 1 1 1',
          ]),
          kawasaki_epsilon=0.05,
      ),
  )
  def test_j6_fuzz_zero_length_cases_rejected_by_cpp_and_exact_checker(
      self, cp_doc, kawasaki_epsilon
  ):
    res = fast_flat_folder.check_cp(
        cp_doc, kawasaki_epsilon=kawasaki_epsilon, verify_certificate=True
    )
    self.assertFalse(res.flat_foldable)
    self.assertEqual(res.exit_stage, 'parse_error')
    self.assertIn('line 5', res.conflict_reason or '')
    self.assertIn('zero-length', (res.conflict_reason or '').lower())

    audit = sat_certificate_checker.audit_exact_folded_arrangement(cp_doc)
    self.assertFalse(audit['locally_closed'])
    self.assertEqual(audit['reason'], 'zero_length_segment_at_line_5')

    _, cert = self._make_one_crease_valid_cert()
    chk = sat_certificate_checker.verify_cpp_sat_certificate(cp_doc, cert)
    self.assertFalse(chk.ok)
    self.assertEqual(chk.reason, 'zero_length_segment_at_line_5')

  def test_j7_coordinate_magnitude_bound_rejected_at_ingress(self):
    over_cp = '\n'.join([
        '1 0 0 1e101 0',
        '1 1e101 0 1e101 1e101',
        '1 1e101 1e101 0 1e101',
        '1 0 1e101 0 0',
        '3 0 0 1e101 1e101',
    ])
    with self.assertRaisesRegex(ValueError, 'line 1'):
      _fast_flat_folder.parse_cp_document(over_cp)
    res = fast_flat_folder.check_cp(over_cp, verify_certificate=False)
    self.assertFalse(res.flat_foldable)
    self.assertEqual(res.exit_stage, 'parse_error')
    self.assertIn('line 1', res.conflict_reason or '')

    engine = _fast_flat_folder.FastFlatFolderEngine()
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)]
    for huge_v in (
        [(0.0, 0.0), (1e200, 0.0), (1e200, 1e200), (0.0, 1e200)],
        [(-1e308, 0.0), (1e308, 0.0), (1e308, 1e308), (-1e308, 1e308)],
    ):
      fg_res = engine.analyze_fold_graph(huge_v, edges, [1, 1, 1, 1, 3])
      self.assertFalse(fg_res.ok_parse)
      self.assertFalse(fg_res.flat_foldable)
      self.assertEqual(fg_res.exit_stage, 'parse_error')

  def test_j8_rejects_negative_solution_limit_and_wrong_length_ranks(self):
    trifold_cp = '\n'.join([
        '1 0 0 3 0',
        '1 3 0 3 2',
        '1 3 2 0 2',
        '1 0 2 0 0',
        '2 1 0 1 2',
        '2 2 0 2 2',
    ])
    # `solution_limit=0` is documented as unlimited.
    unlim = fast_flat_folder.check_cp(trifold_cp, solution_limit=0)
    self.assertTrue(unlim.flat_foldable)
    self.assertEqual(unlim.states, 2)

    neg = fast_flat_folder.check_cp(trifold_cp, solution_limit=-1)
    self.assertFalse(neg.flat_foldable)
    self.assertEqual(neg.exit_stage, 'parse_error')
    self.assertIn('solution_limit', neg.conflict_reason or '')

    engine = _fast_flat_folder.FastFlatFolderEngine()
    for bad_ranks in ([0], [0, 1, 2, 3]):
      opts = _fast_flat_folder.SolveOptions()
      opts.face_group_ranks = bad_ranks
      res = engine.analyze_cp(trifold_cp, opts)
      self.assertFalse(res.ok_parse)
      self.assertFalse(res.flat_foldable)
      self.assertEqual(res.exit_stage, 'parse_error')
      self.assertIn('face_group_ranks', res.conflict_reason)

    with self.assertRaisesRegex(ValueError, 'face_group_ranks'):
      fast_flat_folder.solve_flat_folded_state(
          None, grid_size=3, cp_str=trifold_cp, face_group_ranks=[0]
      )

  def test_j9_memo_cache_includes_seed_segments_and_resets_telemetry(self):
    deltas = fast_flat_folder.diff_cp_changed_segments(
        _SPIKE_TACO_TACO_UNSAT_CP_0, _SPIKE_TACO_TACO_UNSAT_CP_1
    )
    self.assertNotEmpty(deltas)

    session = fast_flat_folder.FlatFolderSession(
        use_memo_cache=True,
        use_support_cache=True,
        use_component_cache=True,
        auto_seed_deltas=False,
    )
    r_unseeded = session.check_cp(
        _SPIKE_TACO_TACO_UNSAT_CP_1,
        changed_segments=(),
        stream_id=None,
    )
    self.assertFalse(r_unseeded.flat_foldable)
    self.assertFalse(r_unseeded.seeded_hit)
    self.assertEqual(r_unseeded.exit_stage, 'stage1_taco_taco')

    r_seeded = session.check_cp(
        _SPIKE_TACO_TACO_UNSAT_CP_1,
        changed_segments=deltas,
        stream_id=None,
    )
    self.assertFalse(r_seeded.memo_cache_hit)
    self.assertTrue(r_seeded.seeded_hit)
    self.assertEqual(r_seeded.exit_stage, 'stage1_taco_taco_seeded')

    # Memo hit resets per-call support/component cache telemetry.
    unassigned_a = '\n'.join([
        '1 0 0 5 0',
        '1 5 0 5 1',
        '1 5 1 0 1',
        '1 0 1 0 0',
        '5 1 0 1 1',
        '2 4 0 4 1',
    ])
    unassigned_b = unassigned_a.replace('2 4 0 4 1', '3 4 0 4 1')
    session.check_cp(unassigned_a, stream_id=None, solution_limit=0)
    rb_miss = session.check_cp(unassigned_b, stream_id=None, solution_limit=0)
    self.assertFalse(rb_miss.memo_cache_hit)
    self.assertTrue(rb_miss.support_cache_hit)
    self.assertEqual(rb_miss.component_cache_hits, 1)

    rb_memo = session.check_cp(unassigned_b, stream_id=None, solution_limit=0)
    self.assertTrue(rb_memo.memo_cache_hit)
    self.assertFalse(rb_memo.support_cache_hit)
    self.assertEqual(rb_memo.component_cache_hits, 0)
    self.assertEqual(rb_memo.component_cache_misses, 0)

  def test_j13_compute_absolute_face_orders_enforces_max_faces_ceiling(self):
    # Boundary `8192` succeeds; `8193` and `5_000_000` fail cleanly.
    at_limit = _fast_flat_folder.compute_absolute_face_orders([], 8192)
    self.assertLen(at_limit, 8192)
    for bad_n in (8193, 5_000_000):
      with self.assertRaisesRegex(ValueError, '8192'):
        _fast_flat_folder.compute_absolute_face_orders([], bad_n)

  def test_j10_diff_cp_changed_segments_endpoint_order_invariant(self):
    # Horizontal segments with `x > y`, `x < y`, and `x == y` must canonicalize
    # identically when endpoints are reversed.
    for seg_fwd, seg_rev in (
        ('2 3 1 5 1', '2 5 1 3 1'),
        ('2 0.25 1 0.75 1', '2 0.75 1 0.25 1'),
        ('2 1 1 3 1', '2 3 1 1 1'),
        ('2 1 0 1 2', '2 1 2 1 0'),
        ('2 0 0 2 2', '2 2 2 0 0'),
    ):
      with self.subTest(seg_fwd=seg_fwd):
        self.assertEmpty(
            fast_flat_folder.diff_cp_changed_segments(seg_fwd, seg_rev)
        )

    # An actual code flip with reversed horizontal endpoints reports exactly 1
    # canonical changed segment, not 2 phantom add/delete segments.
    changed = fast_flat_folder.diff_cp_changed_segments(
        '2 3 1 5 1', '3 5 1 3 1'
    )
    self.assertEqual(changed, ((3.0, 1.0, 5.0, 1.0),))

  def test_j11_analyze_fold_file_catches_malformed_fold_parse_errors(self):
    malformed_folds = (
        ('invalid_json', '{not valid json'),
        ('empty_obj', '{}'),
        ('missing_edges', '{"vertices_coords": [[0, 0], [1, 0]]}'),
        ('json_list', '[]'),
        (
            'oob_edge_vertex',
            '{"vertices_coords": [[0, 0]], "edges_vertices": [[0, 9]]}',
        ),
        (
            'non_numeric_coord',
            (
                '{"vertices_coords": [["bad", 0], [1, 0]], '
                '"edges_vertices": [[0, 1]], "faces_vertices": [[0, 1, 0]]}'
            ),
        ),
    )
    for name, doc in malformed_folds:
      with self.subTest(case=name):
        out = fast_flat_folder.analyze_fold_file(
            f'{name}.fold', doc, '.fold', False
        ).flat_folder_fold
        self.assertFalse(out.flat_foldable)
        self.assertEqual(out.states, 0)
        self.assertEqual(out.conflict_reason, 'IO Parser failed')

    # Valid `.fold` still parses and solves normally.
    valid_fold = json.dumps({
        'vertices_coords': [[0, 0], [1, 0], [1, 1], [0, 1]],
        'edges_vertices': [[0, 1], [1, 2], [2, 3], [3, 0], [0, 2]],
        'edges_assignment': ['B', 'B', 'B', 'B', 'V'],
        'faces_vertices': [[0, 1, 2], [0, 2, 3]],
    })
    ok_out = fast_flat_folder.analyze_fold_file(
        'valid.fold', valid_fold, '.fold', False
    ).flat_folder_fold
    self.assertTrue(ok_out.flat_foldable)
    self.assertEqual(ok_out.states, 1)

  def test_j14_stream_history_is_lru_capped(self):
    session = fast_flat_folder.FlatFolderSession(
        use_memo_cache=False,
        use_support_cache=True,
        use_component_cache=True,
        auto_seed_deltas=True,
    )
    # Prime `pinned_stream`, flood 300 distinct ephemeral streams, then prime
    # `hot_stream` as MRU.
    session.check_cp(_SPIKE_TACO_TACO_UNSAT_CP_0, stream_id='pinned_stream')
    for i in range(300):
      session.check_cp(
          _WATERBOMB_SAT_CP,
          stream_id=f'ephemeral_{i}',
          verify_certificate=False,
      )
    session.check_cp(_SPIKE_TACO_TACO_UNSAT_CP_0, stream_id='hot_stream')

    self.assertLessEqual(
        len(session._prev_cp_by_stream), 256  # pylint: disable=protected-access
    )

    # `pinned_stream` was LRU-evicted, so delta seeding does not fire;
    # `hot_stream` remains MRU and still gets a seeded Wave-1 hit.
    res_evicted = session.check_cp(
        _SPIKE_TACO_TACO_UNSAT_CP_1, stream_id='pinned_stream'
    )
    self.assertFalse(res_evicted.flat_foldable)
    self.assertFalse(res_evicted.seeded_hit)

    res_hot = session.check_cp(
        _SPIKE_TACO_TACO_UNSAT_CP_1, stream_id='hot_stream'
    )
    self.assertFalse(res_hot.flat_foldable)
    self.assertTrue(res_hot.seeded_hit)

    # Custom `max_stream_history` cap, zero-cap disablement, and validation.
    with self.assertRaisesRegex(ValueError, 'max_stream_history'):
      fast_flat_folder.FlatFolderSession(max_stream_history=-1)

    disabled_session = fast_flat_folder.FlatFolderSession(
        use_memo_cache=False, max_stream_history=0
    )
    disabled_session.check_cp(_SPIKE_TACO_TACO_UNSAT_CP_0, stream_id='s0')
    self.assertEmpty(
        disabled_session._prev_cp_by_stream  # pylint: disable=protected-access
    )
    res_disabled = disabled_session.check_cp(
        _SPIKE_TACO_TACO_UNSAT_CP_1, stream_id='s0'
    )
    self.assertFalse(res_disabled.seeded_hit)

    tiny_session = fast_flat_folder.FlatFolderSession(
        use_memo_cache=False, max_stream_history=2
    )
    tiny_session.check_cp(_SPIKE_TACO_TACO_UNSAT_CP_0, stream_id='s_a')
    tiny_session.check_cp(_WATERBOMB_SAT_CP, stream_id='s_b')
    # Touch `s_a` again so `s_b` becomes LRU and gets evicted by `s_c`.
    tiny_session.check_cp(_SPIKE_TACO_TACO_UNSAT_CP_0, stream_id='s_a')
    tiny_session.check_cp(_WATERBOMB_SAT_CP, stream_id='s_c')
    self.assertEqual(
        list(tiny_session._prev_cp_by_stream),  # pylint: disable=protected-access
        ['s_a', 's_c'],
    )
    self.assertTrue(
        tiny_session.check_cp(
            _SPIKE_TACO_TACO_UNSAT_CP_1, stream_id='s_a'
        ).seeded_hit
    )

  def test_rejection_reason_table_covers_all_checker_reasons(self):
    import ast  # pylint: disable=g-import-not-at-top
    import pathlib  # pylint: disable=g-import-not-at-top

    table = sat_certificate_checker.REJECTION_REASON_TABLE
    patterns = [rule.pattern for rule in table]
    self.assertLen(patterns, len(set(patterns)))

    # Every `PATTERN` row must explain why no flat folding exists regardless of
    # layer order; every `CANNOT_CHECK` row must state the checker limitation.
    for rule in table:
      if rule.kind in (
          sat_certificate_checker.CertificateRejectionKind.PATTERN,
          sat_certificate_checker.CertificateRejectionKind.CANNOT_CHECK,
      ):
        self.assertNotEmpty(rule.rationale.strip(), msg=rule.pattern)

    # Extract every reason literal or joined-string prefix returned by
    # `sat_certificate_checker.py` and verify it classifies explicitly inside
    # `REJECTION_REASON_TABLE` (not merely via the `ENGINE_CLAIM` fallback).
    checker_src = pathlib.Path(sat_certificate_checker.__file__).read_text(
        encoding='utf-8'
    )
    tree = ast.parse(checker_src)

    def _matched_rule(probe: str):
      for rule in table:
        if rule.prefix_match:
          if probe.startswith(rule.pattern):
            return rule
        elif probe == rule.pattern:
          return rule
      return None

    # Internal helper sub-tokens from `_parse_float_pair` (`"invalid"`,
    # `"non_finite"`) are formatted with `field_tag` by `_parse_float_pairs`
    # before leaving the checker (`{invalid,non_finite}_solver_*`).
    internal_subtokens = frozenset({'invalid', 'non_finite', 'invalid_'})
    emitted_probes: set[str] = set()

    def _collect_reason_expr(expr: ast.expr) -> None:
      if (
          isinstance(expr, ast.Constant)
          and isinstance(expr.value, str)
          and expr.value
          and '_' in expr.value
          and expr.value not in internal_subtokens
      ):
        emitted_probes.add(expr.value)
      elif isinstance(expr, ast.JoinedStr) and expr.values:
        first_val = expr.values[0]
        if (
            isinstance(first_val, ast.Constant)
            and isinstance(first_val.value, str)
            and first_val.value
            and '_' in first_val.value
            and first_val.value not in internal_subtokens
        ):
          emitted_probes.add(first_val.value)

    for node in ast.walk(tree):
      if isinstance(node, ast.Call) and (
          isinstance(node.func, ast.Name)
          and node.func.id == 'CertificateCheckResult'
      ):
        if len(node.args) >= 2 and isinstance(node.args[0], ast.Constant):
          if isinstance(node.args[0].value, bool) and not node.args[0].value:
            _collect_reason_expr(node.args[1])
      elif isinstance(node, ast.Return) and isinstance(node.value, ast.Tuple):
        if len(node.value.elts) >= 2:
          _collect_reason_expr(node.value.elts[-1])

    self.assertGreaterEqual(len(emitted_probes), 30)
    self.assertIn('zero_length_segment_at_line_', emitted_probes)
    for probe in sorted(emitted_probes):
      self.assertIsNotNone(
          _matched_rule(probe),
          msg=f'Reason {probe!r} missing from REJECTION_REASON_TABLE',
      )

    # Unknown reasons default cleanly to `ENGINE_CLAIM`, never `PATTERN`.
    for unknown_reason in ('', 'mystery_future_reason', 'unexpected_42'):
      self.assertEqual(
          sat_certificate_checker.classify_rejection_reason(unknown_reason),
          sat_certificate_checker.CertificateRejectionKind.ENGINE_CLAIM,
      )

  @parameterized.named_parameters(
      dict(
          testcase_name='cpp_taco_taco_crossing',
          backend='cpp',
          reason='taco_taco_crossing',
          expected_prefix='C++ flat-folder engine claim rejected:',
      ),
      dict(
          testcase_name='cpp_folded_vertex_mismatch',
          backend='cpp',
          reason='solver_folded_vertex_mismatch',
          expected_prefix='C++ flat-folder engine claim rejected:',
      ),
      dict(
          testcase_name='cpp_unknown_reason_defaults_to_engine_claim',
          backend='cpp',
          reason='unregistered_future_checker_reason',
          expected_prefix='C++ flat-folder engine claim rejected:',
      ),
      dict(
          testcase_name='pyref_taco_taco_crossing',
          backend='python_ref',
          reason='taco_taco_crossing',
          expected_prefix='Python fold_cp engine claim rejected:',
      ),
      dict(
          testcase_name='pyref_unknown_reason_defaults_to_engine_claim',
          backend='python_ref',
          reason='unregistered_future_checker_reason',
          expected_prefix='Python fold_cp engine claim rejected:',
      ),
  )
  def test_engine_claim_and_unknown_rejection_returns_could_not_decide(
      self, backend: str, reason: str, expected_prefix: str
  ):
    from unittest import mock  # pylint: disable=g-import-not-at-top

    fake_chk = sat_certificate_checker.CertificateCheckResult(False, reason)
    session = fast_flat_folder.FlatFolderSession(backend=backend)  # type: ignore[arg-type]
    before_global = fast_flat_folder.engine_claim_rejections()
    before_sess = session.engine_claim_rejections

    patch_attr = (
        'verify_cpp_sat_certificate'
        if backend == 'cpp'
        else 'verify_reference_sat_certificate'
    )
    with mock.patch.object(
        fast_flat_folder, patch_attr, return_value=fake_chk, autospec=True
    ):
      sess_res = session.check_cp(_WATERBOMB_SAT_CP, stream_id=None)
      top_res = fast_flat_folder.check_cp(_WATERBOMB_SAT_CP, backend=backend)  # type: ignore[arg-type]

    for res in (sess_res, top_res):
      self.assertEqual(
          res.verdict, fast_flat_folder.FoldVerdict.COULD_NOT_DECIDE
      )
      self.assertIsNone(res.flat_foldable)
      self.assertEqual(res.states, 0)
      self.assertEqual(res.exit_stage, 'could_not_decide')
      self.assertFalse(res.certificate_verified)
      self.assertTrue(res.engine_claim_rejected)
      self.assertIn(expected_prefix, res.conflict_reason or '')
      self.assertIn(reason, res.conflict_reason or '')

    self.assertEqual(session.engine_claim_rejections, before_sess + 1)
    self.assertEqual(
        session.shadow_audit_stats().engine_claim_rejections, before_sess + 1
    )
    self.assertEqual(
        fast_flat_folder.engine_claim_rejections(), before_global + 2
    )

  @parameterized.named_parameters(
      dict(
          testcase_name='cannot_check_boundary_slit',
          cp_doc='\n'.join([
              '1 0 0 3 0',
              '1 3 0 3 3',
              '1 3 3 0 3',
              '1 0 3 0 0',
              '1 3 3 1 1',
          ]),
          expected_verdict=fast_flat_folder.FoldVerdict.COULD_NOT_DECIDE,
          expected_stage='could_not_decide',
          expected_substr='slit_or_dangling_edge',
      ),
      dict(
          testcase_name='cannot_check_dangling_flat_line',
          cp_doc='\n'.join([
              '1 0 0 4 0',
              '1 4 0 4 4',
              '1 4 4 0 4',
              '1 0 4 0 0',
              '4 0 2 1 1',
          ]),
          expected_verdict=fast_flat_folder.FoldVerdict.COULD_NOT_DECIDE,
          expected_stage='could_not_decide',
          expected_substr='slit_or_dangling_edge',
      ),
      dict(
          testcase_name='cannot_check_disconnected_squares',
          cp_doc='\n'.join([
              '1 0 0 1 0',
              '1 1 0 1 1',
              '1 1 1 0 1',
              '1 0 1 0 0',
              '3 0 0 1 1',
              '1 3 0 4 0',
              '1 4 0 4 1',
              '1 4 1 3 1',
              '1 3 1 3 0',
              '3 3 0 4 1',
          ]),
          expected_verdict=fast_flat_folder.FoldVerdict.COULD_NOT_DECIDE,
          expected_stage='could_not_decide',
          expected_substr='Disconnected',
      ),
      dict(
          testcase_name='pattern_proof_float_holonomy',
          cp_doc='\n'.join([
              '1 0 0 6 0',
              '1 6 0 6 4',
              '1 6 4 0 4',
              '1 0 4 0 0',
              '1 2 1 4 1',
              '1 4 1 4 3',
              '1 4 3 2 3',
              '1 2 3 2 1',
              '4 0 1 2 1',
              '4 4 1 6 1',
              '4 0 3 2 3',
              '4 4 3 6 3',
              '2 2 3 2 4',
              '3 2.333333 3 2.333333 4',
          ]),
          expected_verdict=fast_flat_folder.FoldVerdict.NOT_FOLDABLE,
          expected_stage='certificate_rejected',
          expected_substr='transform_holonomy_conflict',
      ),
  )
  def test_three_way_certificate_classification_real_inputs(
      self,
      cp_doc: str,
      expected_verdict: fast_flat_folder.FoldVerdict,
      expected_stage: str,
      expected_substr: str,
  ):
    before_global = fast_flat_folder.engine_claim_rejections()
    res = fast_flat_folder.check_cp(cp_doc)
    self.assertEqual(res.verdict, expected_verdict)
    self.assertEqual(res.exit_stage, expected_stage)
    self.assertFalse(res.engine_claim_rejected)
    self.assertIn(expected_substr, res.conflict_reason or '')
    self.assertEqual(fast_flat_folder.engine_claim_rejections(), before_global)

  def test_default_check_cp_fuzz_not_foldable_only_with_pattern_proof(self):
    # Across mutated-CP fuzz (150 + 300), C++ fuzz (including zero-length
    # segments), and float-path holonomy with ZERO excused reasons:
    # default `check_cp` returns `NOT_FOLDABLE` on certificate rejection ONLY
    # with an exact `PATTERN` proof and records 0 `engine_claim` failures.
    rng = random.Random(4242)
    base_docs = [
        c.cp_doc
        for c in differential_cp_gen.generate_lattice_differential_suite(
            seed=20261125,
            num_synthetic_bases=20,
            base_cp_docs=[],
            mutations_per_base=2,
        )
    ]
    fuzz_docs = [_HANG2_CP]
    while len(fuzz_docs) < 150:
      cand = _mutate_cp_doc(rng, rng.choice(base_docs))
      if cand.strip():
        fuzz_docs.append(cand)

    rng_codes = random.Random(4242)
    for _ in range(300):
      lines = rng_codes.choice(base_docs).splitlines()
      for _ in range(rng_codes.randint(1, 2)):
        idx = rng_codes.randrange(len(lines))
        parts = lines[idx].split()
        if len(parts) == 5:
          parts[0] = str(rng_codes.choice((1, 2, 3, 4, 5)))
          lines[idx] = ' '.join(parts)
      fuzz_docs.append('\n'.join(lines))

    rng_cc = random.Random(20261126)
    for _ in range(200):
      g = rng_cc.choice((2, 3, 4, 6))
      lines = [
          f'1 0 0 {g} 0',
          f'1 {g} 0 {g} {g}',
          f'1 {g} {g} 0 {g}',
          f'1 0 {g} 0 0',
      ]
      for _ in range(rng_cc.randint(0, 9)):
        x1 = rng_cc.randint(0, g)
        y1 = rng_cc.randint(0, g)
        dx, dy = rng_cc.choice(((1, 0), (0, 1), (1, 1), (1, -1)))
        step = rng_cc.randint(0, g)
        x2, y2 = x1 + dx * step, y1 + dy * step
        code = rng_cc.choice((1, 2, 3, 4, 5))
        lines.append(f'{code} {x1} {y1} {x2} {y2}')
      fuzz_docs.append('\n'.join(lines))

    # Include the 49-byte zero-length square and float-path holonomy case.
    fuzz_docs.append('1 0 0 4 0\n1 4 0 4 4\n1 4 4 0 4\n1 0 4 0 0\n1 2 2 2 2')
    fuzz_docs.append(
        '\n'.join([
            '1 0 0 6 0',
            '1 6 0 6 4',
            '1 6 4 0 4',
            '1 0 4 0 0',
            '1 2 1 4 1',
            '1 4 1 4 3',
            '1 4 3 2 3',
            '1 2 3 2 1',
            '4 0 1 2 1',
            '4 4 1 6 1',
            '4 0 3 2 3',
            '4 4 3 6 3',
            '2 2 3 2 4',
            '3 2.333333 3 2.333333 4',
        ])
    )

    before_claims = fast_flat_folder.engine_claim_rejections()
    session = fast_flat_folder.FlatFolderSession()
    engine = _fast_flat_folder.FastFlatFolderEngine(0, 0, 0)
    opts = _fast_flat_folder.SolveOptions()
    opts.compute_layer_order = True
    cert_rejected_pattern = 0
    for idx, doc in enumerate(fuzz_docs):
      res = session.check_cp(doc, stream_id=None)
      self.assertFalse(res.engine_claim_rejected, msg=f'case {idx}')
      if res.verdict == fast_flat_folder.FoldVerdict.NOT_FOLDABLE:
        if res.exit_stage == 'certificate_rejected':
          chk = sat_certificate_checker.verify_cpp_sat_certificate(
              doc, engine.analyze_cp(doc, opts)
          )
          self.assertFalse(chk.ok)
          self.assertEqual(
              chk.rejection_kind,
              sat_certificate_checker.CertificateRejectionKind.PATTERN,
              msg=f'case {idx}: {chk.reason}',
          )
          cert_rejected_pattern += 1
        else:
          self.assertFalse(
              engine.analyze_cp(doc, opts).flat_foldable,
              msg=f'case {idx}: NOT_FOLDABLE without PATTERN cert proof',
          )
      elif res.flat_foldable:
        self.assertTrue(res.certificate_verified)
    self.assertEqual(session.engine_claim_rejections, 0)
    self.assertEqual(fast_flat_folder.engine_claim_rejections(), before_claims)
    self.assertGreaterEqual(cert_rejected_pattern, 1)

  def test_solve_segments_and_solve_graph(self):
    session = fast_flat_folder.FlatFolderSession(use_memo_cache=False)
    segs = [
        (0.0, 0.0, 4.0, 0.0, 'B'),
        (4.0, 0.0, 4.0, 4.0, 'B'),
        (4.0, 4.0, 0.0, 4.0, 'B'),
        (0.0, 4.0, 0.0, 0.0, 'B'),
        (1.0, 0.0, 1.0, 4.0, 'V'),
        (2.0, 0.0, 2.0, 4.0, 'M'),
        (3.0, 0.0, 3.0, 4.0, 'V'),
    ]
    seg_res = session.solve_segments(segs)
    self.assertTrue(seg_res.flat_foldable)
    self.assertEqual(seg_res.verdict, fast_flat_folder.FoldVerdict.FOLDABLE)
    self.assertLen(seg_res.face_orders, 4)
    self.assertLen(set(seg_res.face_orders), 4)
    self.assertNotEmpty(seg_res.ordered_face_pairs)
    self.assertNotEmpty(seg_res.cell_faces_ordered)

    no_flip_res = session.solve_segments(segs, flip_to_top_face_up=False)
    self.assertTrue(no_flip_res.flat_foldable)
    self.assertLen(no_flip_res.face_orders, 4)

    # Unassigned interior crease: `face_group_ranks` picks which face sits on
    # top (`smaller rank` -> preferred toward top of stack).
    unassigned_segs = [
        (0.0, 0.0, 2.0, 0.0, 'B'),
        (2.0, 0.0, 2.0, 2.0, 'B'),
        (2.0, 2.0, 0.0, 2.0, 'B'),
        (0.0, 2.0, 0.0, 0.0, 'B'),
        (1.0, 0.0, 1.0, 2.0, 'U'),
    ]
    rank_0_top = session.solve_segments(
        unassigned_segs, face_group_ranks=[0, 10]
    )
    rank_1_top = session.solve_segments(
        unassigned_segs, face_group_ranks=[10, 0]
    )
    self.assertTrue(rank_0_top.flat_foldable)
    self.assertTrue(rank_1_top.flat_foldable)
    self.assertNotEqual(rank_0_top.face_orders, rank_1_top.face_orders)
    self.assertGreater(rank_0_top.face_orders[0], rank_0_top.face_orders[1])
    self.assertGreater(rank_1_top.face_orders[1], rank_1_top.face_orders[0])

    # Wrong-length `face_group_ranks` -> C++ `parse_error` / `NOT_FOLDABLE`.
    bad_ranks_res = session.solve_segments(segs, face_group_ranks=[0])
    self.assertFalse(bad_ranks_res.flat_foldable)
    self.assertEqual(bad_ranks_res.exit_stage, 'parse_error')

    # UNSAT locked C-roll (`V` at x=3 and `V` at x=4 on [0, 8]^2).
    unsat_res = session.solve_segments([
        (0.0, 0.0, 8.0, 0.0, 'B'),
        (8.0, 0.0, 8.0, 8.0, 'B'),
        (8.0, 8.0, 0.0, 8.0, 'B'),
        (0.0, 8.0, 0.0, 0.0, 'B'),
        (3.0, 0.0, 3.0, 8.0, 'V'),
        (4.0, 0.0, 4.0, 8.0, 'V'),
    ])
    self.assertFalse(unsat_res.flat_foldable)
    self.assertEqual(
        unsat_res.verdict, fast_flat_folder.FoldVerdict.NOT_FOLDABLE
    )

    graph_verts = [
        (0.0, 0.0),
        (2.0, 0.0),
        (4.0, 0.0),
        (4.0, 4.0),
        (2.0, 4.0),
        (0.0, 4.0),
    ]
    graph_edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 5),
        (5, 0),
        (1, 4),
    ]
    graph_asgns = ['B', 'B', 'B', 'B', 'B', 'B', 'V']
    cw_faces = [[0, 5, 4, 1], [1, 4, 3, 2]]
    graph_res = session.solve_graph(
        graph_verts,
        graph_edges,
        graph_asgns,
        face_vertices=cw_faces,
    )
    self.assertTrue(graph_res.flat_foldable)
    self.assertLen(graph_res.face_orders, 2)
    self.assertNotEqual(graph_res.face_orders[0], graph_res.face_orders[1])
    self.assertEqual(
        [list(f) for f in graph_res.face_vertices],
        [list(reversed(f)) for f in cw_faces],
    )

    no_flip_graph_res = session.solve_graph(
        graph_verts,
        graph_edges,
        graph_asgns,
        face_vertices=cw_faces,
        flip_to_top_face_up=False,
    )
    self.assertTrue(no_flip_graph_res.flat_foldable)
    self.assertEqual(
        [list(f) for f in no_flip_graph_res.face_vertices], cw_faces
    )

    # `face_vertices=None` extracts planar faces automatically in graph mode.
    auto_face_res = session.solve_graph(
        graph_verts,
        graph_edges,
        graph_asgns,
        face_vertices=None,
        flip_to_top_face_up=False,
    )
    self.assertTrue(auto_face_res.flat_foldable)
    self.assertLen(auto_face_res.face_orders, 2)

    # Input validation: empty inputs, `python_ref` backend, and bad codes.
    with self.assertRaisesRegex(ValueError, 'non-empty'):
      session.solve_segments([])
    with self.assertRaisesRegex(ValueError, 'non-empty'):
      session.solve_graph([], [], [])
    with self.assertRaisesRegex(ValueError, 'same length'):
      session.solve_graph(graph_verts, graph_edges, ['B'])

    py_session = fast_flat_folder.FlatFolderSession(backend='python_ref')
    with self.assertRaisesRegex(ValueError, "backend='cpp'"):
      py_session.solve_segments(segs)
    with self.assertRaisesRegex(ValueError, "backend='cpp'"):
      py_session.solve_graph(graph_verts, graph_edges, graph_asgns)

    for bad_code in ('X', 'm', -1, 6, True):
      with self.assertRaisesRegex(ValueError, 'Unsupported crease assignment'):
        session.solve_segments([(0.0, 0.0, 1.0, 0.0, bad_code)])  # type: ignore[arg-type]
      with self.assertRaisesRegex(ValueError, 'Unsupported crease assignment'):
        session.solve_graph(
            graph_verts,
            graph_edges,
            ['B', 'B', 'B', 'B', 'B', 'B', bad_code],  # type: ignore[list-item]
        )


if __name__ == '__main__':
  absltest.main()
