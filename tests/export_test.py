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

"""Unit tests for BoxPleatingTreeData (.bps.json), SVG (.svg), and FOLD (.fold) export utilities."""

import copy
import io
import json
import math
import pathlib
import tempfile

from absl.testing import absltest
from absl.testing import parameterized
import bp_pipeline as bp
from bp_pipeline.crease_pattern.parser import svg as svg_parser
from bp_pipeline.stick_figures import converters as stick_converters
import cp_test_fixtures as cp_examples


class ExportTest(parameterized.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.cat_result = bp.fold('cat', preset='demo')
    cls.human_rcc_packed = bp.pack(
        bp.prepare(bp.load_example('human')),
        bp.PackingConfig(remove_central_crease=True),
    )

  def test_bp_studio_tree_and_round_trip(self):
    sf = bp.load_example('cat')
    design = bp.prepare(sf)

    bps_from_sf = bp.to_bp_studio(sf)
    bps_from_dt = bp.to_bp_studio(design)
    self.assertEqual(bps_from_sf, bps_from_dt)
    self.assertEqual(bps_from_dt['version'], '0.7')
    self.assertEqual(bps_from_dt['design']['title'], 'cat')
    self.assertEqual(bps_from_dt['design']['mode'], 'tree')

    tree = bps_from_dt['design']['tree']
    self.assertLen(tree['edges'], len(sf.children))
    self.assertNotEmpty(tree['nodes'])

    roundtripped = bp.from_bp_studio(bps_from_dt)
    self.assertEqual(roundtripped.name, 'cat')
    self.assertLen(roundtripped.children, len(sf.children))

    orig_pairs = sorted(
        (round(s.length), round(s.azimuth_angle % 360.0, 3))
        for s in sf.children
    )
    rt_pairs = sorted(
        (round(s.length), round(s.azimuth_angle % 360.0, 3))
        for s in roundtripped.children
    )
    for (len1, ang1), (len2, ang2) in zip(orig_pairs, rt_pairs, strict=True):
      self.assertEqual(len1, len2)
      self.assertAlmostEqual(ang1, ang2, places=2)

  def test_bp_studio_example_tree_data_round_trip(self):
    mini_human_bps = cp_examples.MINI_HUMAN.tree_data
    self.assertIsNotNone(mini_human_bps)
    sf = stick_converters.bps_to_stick_figure(mini_human_bps)
    exported = bp.to_bp_studio(sf)
    re_parsed = bp.from_bp_studio(exported)

    self.assertEqual(re_parsed.name, sf.name)
    self.assertLen(re_parsed.children, len(sf.children))
    for s_orig, s_rt in zip(sf.children, re_parsed.children, strict=True):
      self.assertEqual(s_orig.length, s_rt.length)
      self.assertAlmostEqual(
          s_orig.azimuth_angle % 360.0,
          s_rt.azimuth_angle % 360.0,
          places=3,
      )

  def test_bp_studio_packed_and_solved_layout(self):
    packed = self.cat_result.packed
    solved = self.cat_result.solved

    self.assertIsNotNone(packed.flap_for_stick('tail'))

    bps_packed = bp.to_bp_studio(packed)
    bps_solved = bp.to_bp_studio(solved)
    bps_pipeline = self.cat_result.to_bp_studio()

    self.assertEqual(bps_packed['design']['mode'], 'layout')
    self.assertEqual(bps_solved, bps_pipeline)

    layout = bps_packed['design']['layout']
    self.assertEqual(layout['sheet']['width'], packed.grid_size)
    self.assertEqual(layout['sheet']['height'], packed.grid_size)
    self.assertNotEmpty(layout['flaps'])
    self.assertNotEmpty(layout['stretches'])

    for flap in layout['flaps']:
      self.assertIn('id', flap)
      self.assertGreaterEqual(flap['x'], 0)
      self.assertGreaterEqual(flap['y'], 0)
      self.assertLessEqual(flap['x'] + flap['width'], packed.grid_size)
      self.assertLessEqual(flap['y'] + flap['height'], packed.grid_size)

    for stretch in layout['stretches']:
      self.assertIn('id', stretch)
      self.assertIn('stick', stretch)
      self.assertGreater(stretch['width'], 0)
      self.assertNotEmpty(stretch['rects'])

    with tempfile.TemporaryDirectory() as tmp_dir_str:
      tmp_dir = pathlib.Path(tmp_dir_str)
      out_file = tmp_dir / 'cat.bps.json'
      ret = self.cat_result.save_bp_studio(out_file)
      self.assertIsNone(ret)
      self.assertTrue(out_file.is_file())
      loaded_json = json.loads(out_file.read_text(encoding='utf-8'))
      self.assertEqual(bps_pipeline, loaded_json)
      self.assertEqual(bp.from_bp_studio(out_file.read_text()).name, 'cat')

  def test_svg_export_options_and_round_trip(self):
    packed = self.cat_result.packed
    solved = self.cat_result.solved

    svg_default = solved.to_svg()
    self.assertTrue(svg_parser.valid_svg(svg_default))
    self.assertEqual(svg_default, self.cat_result.to_svg())
    self.assertEqual(svg_default, bp.to_svg(solved))

    svg_with_grid = solved.to_svg(
        include_grid=True,
        dashed_valley=True,
        stroke_width=0.1,
    )
    self.assertIn('bp-grid-lines', svg_with_grid)
    self.assertIn('stroke-dasharray', svg_with_grid)
    self.assertTrue(svg_parser.valid_svg(svg_with_grid))

    parsed_cp = svg_parser.parse_svg(svg_with_grid, grid_size=solved.grid_size)
    self.assertTrue(parsed_cp.local_flat_foldability())

    svg_custom_colors = bp.to_svg(
        packed,
        colors={'hinge': '#00ff00', 'mountain': '#ff0000'},
    )
    self.assertIn('stroke="#00ff00"', svg_custom_colors)
    self.assertTrue(svg_parser.valid_svg(svg_custom_colors))
    parsed_packed_cp = svg_parser.parse_svg(
        svg_custom_colors, grid_size=packed.grid_size
    )
    self.assertEqual(
        len(parsed_packed_cp.creases),
        len(packed.creases.creases),
    )

    buf = io.StringIO()
    ret = bp.save_svg(
        solved.creases, buf, grid_size=solved.grid_size, include_grid=True
    )
    self.assertIsNone(ret)
    self.assertIn('<svg', buf.getvalue())

  def test_fold_export_schema_and_round_trip(self):
    packed = self.cat_result.packed
    solved = self.cat_result.solved

    packed_fold = bp.to_fold(packed)
    self.assertEqual(packed_fold['file_spec'], 1.2)
    self.assertEqual(packed_fold['file_creator'], 'bp_pipeline')
    self.assertEqual(packed_fold['frame_classes'], ['creasePattern'])
    self.assertEqual(packed_fold['file_title'], 'cat')
    self.assertIn('U', packed_fold['edges_assignment'])

    solved_fold = bp.to_fold(solved, include_faces=True)
    self.assertEqual(solved_fold, self.cat_result.to_fold(include_faces=True))
    self.assertEqual(solved_fold['file_spec'], 1.2)
    self.assertEqual(solved_fold['frame_classes'], ['creasePattern'])
    self.assertNotEmpty(solved_fold['vertices_coords'])
    self.assertNotEmpty(solved_fold['edges_vertices'])
    self.assertLen(
        solved_fold['edges_assignment'],
        len(solved_fold['edges_vertices']),
    )
    self.assertLen(
        solved_fold['edges_foldAngle'],
        len(solved_fold['edges_vertices']),
    )
    self.assertTrue(
        set(solved_fold['edges_assignment']).issubset({'B', 'M', 'V', 'F', 'U'})
    )
    self.assertIn('M', solved_fold['edges_assignment'])
    self.assertIn('V', solved_fold['edges_assignment'])
    self.assertIn('B', solved_fold['edges_assignment'])
    self.assertNotEmpty(solved_fold['faces_vertices'])

    for assign, angle in zip(
        solved_fold['edges_assignment'],
        solved_fold['edges_foldAngle'],
        strict=True,
    ):
      if assign == 'M':
        self.assertLess(angle, 0.0)
        self.assertTrue(math.isclose(angle, -180.0, abs_tol=1e-3))
      elif assign == 'V':
        self.assertGreater(angle, 0.0)
        self.assertTrue(math.isclose(angle, 180.0, abs_tol=1e-3))
      else:
        self.assertEqual(angle, 0.0)

    svg_from_fold = stick_converters.fold_to_svg(solved_fold)
    self.assertIsNotNone(svg_from_fold)
    self.assertIn('<svg', svg_from_fold)

    reconstructed_cp = bp.from_fold(solved_fold, grid_size=solved.grid_size)
    self.assertEqual(
        reconstructed_cp.hash_value,
        solved.creases.hash_value,
    )
    self.assertTrue(reconstructed_cp.local_flat_foldability())

    with tempfile.TemporaryDirectory() as tmp_dir_str:
      tmp_dir = pathlib.Path(tmp_dir_str)
      fold_path = tmp_dir / 'cat.fold'
      ret = self.cat_result.save_fold(fold_path)
      self.assertIsNone(ret)
      self.assertTrue(fold_path.is_file())
      reloaded = bp.from_fold(
          fold_path.read_text(encoding='utf-8'),
          grid_size=solved.grid_size,
      )
      self.assertEqual(reloaded.hash_value, solved.creases.hash_value)

  def test_import_bps_round_trip_valid_packing(self):
    packed = self.cat_result.packed
    bps_doc = bp.to_bp_studio(packed)
    result = bp.import_bps(bps_doc)

    self.assertEmpty(result.validation_errors)
    self.assertIsNotNone(result.packed)
    self.assertIsNotNone(result.first_passing_config)
    self.assertEqual(result.packed.grid_size, packed.grid_size)
    self.assertEqual(result.design.source.name, 'cat')

    solved = bp.solve_best(result.packed)
    self.assertIsNotNone(solved)
    self.assertEqual(solved.grid_size, packed.grid_size)
    self.assertTrue(solved.creases.local_flat_foldability())

  def test_import_bps_round_trip_giraffe_degree2_chain(self):
    sf = bp.load_example('giraffe')
    design = bp.prepare(sf)
    packed = bp.pack(design)
    bps_doc = bp.to_bp_studio(packed)
    result = bp.import_bps(bps_doc)

    self.assertEmpty(result.validation_errors)
    self.assertIsNotNone(result.packed)
    self.assertIsNotNone(result.first_passing_config)
    self.assertEqual(result.packed.grid_size, packed.grid_size)
    self.assertEqual(result.design.source.name, 'giraffe')

    solved = bp.solve_best(result.packed)
    self.assertIsNotNone(solved)
    self.assertEqual(solved.grid_size, packed.grid_size)
    self.assertTrue(solved.creases.local_flat_foldability())

  def test_to_bp_studio_stacked_internal_flaps_rejected(self):
    sf = bp.load_example('giraffe')
    design = bp.prepare(sf)
    packed = bp.stack(design)
    with self.assertRaisesRegex(ValueError, 'Cannot export stacked packing'):
      bp.to_bp_studio(packed)

  def test_import_bps_tree_only(self):
    sf = bp.load_example('cat')
    bps_doc = bp.to_bp_studio(sf)
    result = bp.import_bps(bps_doc)

    self.assertEmpty(result.validation_errors)
    self.assertIsNone(result.packed)
    self.assertEqual(result.design.source.name, 'cat')
    self.assertEqual(len(result.design.source.children), len(sf.children))

  def test_import_bps_invalid_overlapping_flaps(self):
    packed = self.cat_result.packed
    bps_doc = copy.deepcopy(bp.to_bp_studio(packed))
    layout = bps_doc['design']['layout']
    for flap in layout['flaps']:
      flap['x'] = 10
      flap['y'] = 10
    result = bp.import_bps(bps_doc)

    self.assertIsNone(result.packed)
    self.assertNotEmpty(result.validation_errors)
    self.assertTrue(
        any('overlaps with' in err for err in result.validation_errors)
    )

  def test_import_bps_boundary_violation(self):
    packed = self.cat_result.packed
    bps_doc = copy.deepcopy(bp.to_bp_studio(packed))
    layout = bps_doc['design']['layout']
    layout['flaps'][0]['x'] = packed.grid_size + 10
    result = bp.import_bps(bps_doc)
    self.assertIsNone(result.packed)
    self.assertNotEmpty(result.validation_errors)
    self.assertTrue(
        any(
            'exceed sheet boundaries' in err for err in result.validation_errors
        )
    )

  def test_import_bps_tree_metric_violation(self):
    packed = self.cat_result.packed
    bps_doc = copy.deepcopy(bp.to_bp_studio(packed))
    layout = bps_doc['design']['layout']
    layout['flaps'][0]['x'] = 5
    layout['flaps'][0]['y'] = 5
    layout['flaps'][1]['x'] = 6
    layout['flaps'][1]['y'] = 5
    result = bp.import_bps(bps_doc)

    self.assertIsNone(result.packed)
    self.assertNotEmpty(result.validation_errors)
    self.assertTrue(
        any('violate tree metric' in err for err in result.validation_errors)
    )

  def test_import_bps_uncovered_hole_rejected(self):
    packed = self.cat_result.packed
    bps_doc = copy.deepcopy(bp.to_bp_studio(packed))
    layout = bps_doc['design']['layout']
    # Enlarge sheet by 2 so cells along border are left uncovered
    layout['sheet']['width'] = packed.grid_size + 2
    layout['sheet']['height'] = packed.grid_size + 2
    result = bp.import_bps(bps_doc)

    self.assertIsNone(result.packed)
    self.assertNotEmpty(result.validation_errors)
    self.assertTrue(
        any('uncovered cells' in err for err in result.validation_errors)
    )

  def test_import_bps_idempotent_no_mutation(self):
    packed = self.cat_result.packed
    bps_doc = bp.to_bp_studio(packed)
    result1 = bp.import_bps(bps_doc)
    self.assertEmpty(result1.validation_errors)
    self.assertIsNotNone(result1.packed)

    orig_tip_shapes = [s.flap_tip_shape for s in result1.design.source.children]

    result2 = bp.import_bps(bps_doc)
    self.assertEmpty(result2.validation_errors)
    self.assertIsNotNone(result2.packed)

    after_tip_shapes = [
        s.flap_tip_shape for s in result1.design.source.children
    ]
    self.assertEqual(orig_tip_shapes, after_tip_shapes)
    self.assertEqual(
        result1.design.source.to_dict(), result2.design.source.to_dict()
    )

  def test_import_bps_scale_factor_indivisible_rejected(self):
    packed = self.cat_result.packed
    bps_doc = copy.deepcopy(bp.to_bp_studio(packed))
    layout = bps_doc['design']['layout']
    layout['scale_factor'] = 2
    # Make flap coordinate odd (not divisible by 2)
    layout['flaps'][0]['x'] = 3
    result = bp.import_bps(bps_doc)

    self.assertIsNone(result.packed)
    self.assertNotEmpty(result.validation_errors)
    self.assertTrue(
        any(
            'not divisible by scale_factor' in err
            for err in result.validation_errors
        )
    )

  def test_import_bps_invalid_stretch_stick_rejected(self):
    packed = self.cat_result.packed
    bps_doc = copy.deepcopy(bp.to_bp_studio(packed))
    layout = bps_doc['design']['layout']
    # cat has an internal stretch for 'body'. Mislabeled stick should fail.
    if layout.get('stretches'):
      layout['stretches'][0]['stick'] = 'nonexistent_branch'
      result = bp.import_bps(bps_doc)
      self.assertIsNone(result.packed)
      self.assertNotEmpty(result.validation_errors)
      self.assertTrue(
          any('unknown stick' in err for err in result.validation_errors)
      )

  def test_remove_central_crease_human_export_and_import(self):
    packed = self.human_rcc_packed
    self.assertNotEmpty(packed.reservations)
    self.assertFalse(
        any(fid.startswith('axis_filler') for fid in packed.flaps)
    )
    bps_doc = bp.to_bp_studio(packed)
    self.assertNotEmpty(bps_doc['design']['layout'].get('reservations', ()))
    result = bp.import_bps(bps_doc)
    self.assertEmpty(result.validation_errors)
    self.assertIsNotNone(result.packed)
    self.assertEqual(result.packed.reservations, packed.reservations)
    self.assertEqual(
        result.packed.creases.hash_value, packed.creases.hash_value
    )
    solved = bp.solve_best(result.packed)
    self.assertTrue(solved.creases.local_flat_foldability())
    solved_rcc = bp.solve_best(
        result.packed, bp.SolverConfig(remove_central_crease=True)
    )
    self.assertTrue(solved_rcc.creases.local_flat_foldability())
    self.assertTrue(solved_rcc.central_crease_removed)

  def test_to_bp_studio_omits_reservations_when_empty(self):
    packed = self.cat_result.packed
    self.assertEmpty(packed.reservations)
    bps_doc = bp.to_bp_studio(packed)
    layout = bps_doc['design']['layout']
    self.assertNotIn('reservations', layout)
    self.assertEqual(
        list(layout.keys()), ['sheet', 'scale_factor', 'flaps', 'stretches']
    )
    without_reservations = copy.deepcopy(bps_doc)
    without_reservations['design']['layout'].pop('reservations', None)
    self.assertEqual(
        json.dumps(bps_doc, indent=2),
        json.dumps(without_reservations, indent=2),
    )

  def test_import_bps_reservations_scale_factor_and_custom_id_round_trip(self):
    packed = self.human_rcc_packed
    bps_doc = copy.deepcopy(bp.to_bp_studio(packed))
    layout = bps_doc['design']['layout']
    layout['sheet']['width'] *= 2
    layout['sheet']['height'] *= 2
    layout['scale_factor'] = 2
    for flap in layout['flaps']:
      for k in ('x', 'y', 'width', 'height'):
        flap[k] *= 2
    for stretch in layout['stretches']:
      stretch['width'] *= 2
      for rect in stretch['rects']:
        for k in ('x', 'y', 'width', 'height'):
          rect[k] *= 2
    for i, res in enumerate(layout['reservations']):
      res['id'] = f'custom_strip_{i}'
      for k in ('x', 'y', 'width', 'height'):
        res[k] *= 2

    result = bp.import_bps(bps_doc)
    self.assertEmpty(result.validation_errors)
    self.assertIsNotNone(result.packed)
    self.assertEqual(
        [r.id for r in result.packed.reservations],
        [f'custom_strip_{i}' for i in range(len(packed.reservations))],
    )
    self.assertEqual(
        [r.rect for r in result.packed.reservations],
        [r.rect for r in packed.reservations],
    )

  @parameterized.named_parameters(
      dict(
          testcase_name='out_of_bounds',
          mutate=lambda layout: layout['reservations'][0].update(
              {'x': layout['sheet']['width']}
          ),
          expected_error='exceeds sheet bounds',
      ),
      dict(
          testcase_name='non_positive_dimension',
          mutate=lambda layout: layout['reservations'][0].update({'width': 0}),
          expected_error='invalid or exceeds sheet bounds',
      ),
      dict(
          testcase_name='overlaps_flap',
          mutate=lambda layout: layout['reservations'][0].update(
              {'x': 0, 'y': 0, 'width': 2, 'height': 2}
          ),
          expected_error='overlaps with',
      ),
      dict(
          testcase_name='overlaps_river',
          mutate=lambda layout: layout['reservations'].append({
              'id': 'dup_river',
              **layout['stretches'][0]['rects'][0],
          }),
          expected_error='overlaps with',
      ),
      dict(
          testcase_name='overlaps_other_reservation',
          mutate=lambda layout: layout['reservations'].append(
              {**layout['reservations'][0], 'id': 'dup_res'}
          ),
          expected_error='overlaps with',
      ),
      dict(
          testcase_name='indivisible_scale_factor',
          mutate=lambda layout: (
              layout.update({'scale_factor': 2}),
              layout['reservations'][0].update({'x': 3}),
          ),
          expected_error='not divisible by scale_factor',
      ),
      dict(
          testcase_name='indivisible_scale_factor_null_id_uses_fallback',
          mutate=lambda layout: (
              layout.update({'scale_factor': 2}),
              layout['reservations'][0].update({'id': None, 'x': 3}),
          ),
          expected_error="Reservation 'axis_filler_0' coordinate 'x'=3",
      ),
  )
  def test_import_bps_invalid_reservations_rejected(
      self, mutate, expected_error
  ):
    bps_doc = copy.deepcopy(bp.to_bp_studio(self.human_rcc_packed))
    mutate(bps_doc['design']['layout'])
    result = bp.import_bps(bps_doc)
    self.assertIsNone(result.packed)
    self.assertNotEmpty(result.validation_errors)
    self.assertTrue(
        any(expected_error in err for err in result.validation_errors),
        msg=f'Expected {expected_error!r} in {result.validation_errors}',
    )


if __name__ == '__main__':
  absltest.main()
