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

"""Tests for the FOLD export of 3D folded meshes."""

from absl.testing import absltest
import bp_pipeline as bp
from bp_pipeline import folder_3d as fold_3d
import fold_checks
import numpy as np


class MeshToFoldDictTest(absltest.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.result = bp.fold('cat', preset='demo', shape=True)
    cls.fold = bp.build_folded_mesh_3d(cls.result.shaped).to_fold_dict('cat')

  def test_flat_frame_is_the_unit_sheet_with_counter_clockwise_faces(self):
    flat, _ = fold_checks.frames(self.fold)
    areas = fold_checks.face_areas(self.fold)

    np.testing.assert_allclose(flat.min(axis=0), [0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(flat.max(axis=0), [1.0, 1.0], atol=1e-6)
    self.assertGreater(areas.min(), 0.0)
    self.assertAlmostEqual(areas.sum(), 1.0, places=5)

  def test_folded_frame_folds_the_flat_frame_by_the_fold_angles(self):
    errors = fold_checks.fold_angle_errors(self.fold)

    # Without partly folded creases, a mirrored folded frame would pass too.
    self.assertTrue(fold_checks.partly_folded_creases(self.fold).any())
    self.assertLess(np.nanmax(errors), 1.0)
    self.assertLess(fold_checks.mean_strain(self.fold), 1e-4)

  def test_regions_are_the_flaps_and_rivers_of_the_packing(self):
    regions = self.fold['file_bp:regions']
    faces_region = self.fold['faces_bp:region']

    self.assertEqual(
        self.fold['file_bp:gridSize'], self.result.packed.grid_size
    )
    region_sticks = []
    for region in regions:
      region_sticks.extend(region['sticks'])
    self.assertCountEqual(
        region_sticks,
        [stick.name for stick in bp.load_example('cat').children],
    )
    self.assertContainsSubset(
        {region['kind'] for region in regions}, {'flap', 'river'}
    )
    self.assertLen(faces_region, len(self.fold['faces_vertices']))
    # The flaps and rivers tile the sheet, so every face lies in one of them
    # and none of them is left without faces.
    self.assertCountEqual(set(faces_region), range(len(regions)))

  def test_shaping_marks_folded_creases_that_the_solved_base_lacks(self):
    shaping = np.asarray(self.fold['edges_bp:shaping'])
    assignments = np.asarray(self.fold['edges_assignment'])
    solved = bp.build_folded_mesh_3d(self.result.solved).to_fold_dict('cat')

    self.assertLen(shaping, len(assignments))
    self.assertTrue(np.isin(assignments[shaping], ['M', 'V']).all())
    self.assertTrue(shaping.any())
    # The creases kept from the base lie on creases of the base, so together
    # they cannot be longer.
    folded_creases = np.isin(assignments, ['M', 'V'])
    self.assertLessEqual(
        _total_length(self.fold, folded_creases & ~shaping),
        _total_length(solved, np.isin(solved['edges_assignment'], ['M', 'V']))
        + 1e-6,
    )
    self.assertNotIn('edges_bp:shaping', solved)

  def test_mesh_of_a_bare_crease_pattern_has_no_pipeline_properties(self):
    mesh = bp.build_folded_mesh_3d(self.result.shaped.creases)

    fold = mesh.to_fold_dict('cat')

    fold_checks.frames(fold)
    self.assertEmpty([key for key in fold if 'bp:' in key])

  def test_folded_frame_shares_proper_so3_handedness_with_folder_3d(self):
    model = fold_3d.fold_3d(
        self.result.shaped.creases,
        vert_tol=0.001,
        add_border=True,
        merge_collinear=False,
        strict=False,
    )
    src = model.vertices_3d - model.vertices_3d.mean(axis=0)
    dst = np.asarray(self.fold['vertices_coords'], dtype=float)
    dst = dst - dst.mean(axis=0)
    self.assertEqual(src.shape, dst.shape)
    u, _, vt = np.linalg.svd(src.T @ dst)
    rot = u @ vt
    self.assertAlmostEqual(float(np.linalg.det(rot)), 1.0, places=5)
    np.testing.assert_allclose(src @ rot, dst, atol=2e-5)


def _total_length(fold, edge_mask):
  """Returns the summed flat length of the edges selected by `edge_mask`."""
  flat, _ = fold_checks.frames(fold)
  start, end = np.asarray(fold['edges_vertices'])[edge_mask].T
  return np.linalg.norm(flat[start] - flat[end], axis=1).sum()


if __name__ == '__main__':
  absltest.main()
