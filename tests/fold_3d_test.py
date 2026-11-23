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

"""Tests for the pure-NumPy 3D origami folder and planar face extractor."""

from absl.testing import absltest
from absl.testing import parameterized
import bp_pipeline as bp
from bp_pipeline import folder_3d as fold_3d
from bp_pipeline.crease_pattern import crease as crease_lib
from bp_pipeline.crease_pattern import crease_pattern as cp_lib
from bp_pipeline.crease_pattern import point as point_lib
import numpy as np


def _seg(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    fold: crease_lib.Fold,
    fold_percent: float = 1.0,
) -> crease_lib.Crease:
  return crease_lib.Crease(
      point_lib.Point(x=x1, y=y1),
      point_lib.Point(x=x2, y=y2),
      fold=fold,
      fold_percent=fold_percent,
  )


def _border_creases() -> list[crease_lib.Crease]:
  b = crease_lib.Fold.BORDER
  return [
      _seg(0.0, 0.0, 1.0, 0.0, b),
      _seg(1.0, 0.0, 1.0, 1.0, b),
      _seg(1.0, 1.0, 0.0, 1.0, b),
      _seg(0.0, 1.0, 0.0, 0.0, b),
  ]


class Fold3DTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('valley_lifts_to_plus_z', crease_lib.Fold.VALLEY, 0.25),
      ('mountain_drops_to_minus_z', crease_lib.Fold.MOUNTAIN, -0.25),
  )
  def test_ninety_degree_hinge_polarity_and_se3_matrices(
      self, fold, expected_z
  ):
    cp = cp_lib.CreasePattern(
        [*_border_creases(), _seg(0.5, 0.0, 0.5, 1.0, fold, fold_percent=0.5)],
        light_mode=True,
    )
    model = fold_3d.fold_3d(cp)
    self.assertLen(model.faces, 2)
    self.assertAlmostEqual(model.vertex_error, 0.0, places=6)

    # Every face transform must be a proper SE(3) rigid motion.
    for mat in model.face_matrices:
      rot = mat[:3, :3]
      np.testing.assert_allclose(rot.T @ rot, np.eye(3), atol=1e-12)
      self.assertAlmostEqual(float(np.linalg.det(rot)), 1.0, places=12)

    left = model.locate_point([0.25, 0.5])
    right = model.locate_point([0.75, 0.5])
    self.assertIsNotNone(left)
    self.assertIsNotNone(right)
    # Anchor sits at z = 0; the opposite half sits at x = 0.5, z = expected_z.
    moving = right if abs(float(left[2])) < 1e-9 else left
    np.testing.assert_allclose(moving, [0.5, 0.5, expected_z], atol=1e-9)

  def test_splits_diagonal_x_crossing_into_four_flat_foldable_faces(self):
    # Classical 4-crease vertex: 3 mountains + 1 valley meeting at (0.5, 0.5).
    cp = cp_lib.CreasePattern(
        [
            *_border_creases(),
            _seg(0.0, 0.0, 1.0, 1.0, crease_lib.Fold.MOUNTAIN),
            _seg(0.0, 1.0, 0.5, 0.5, crease_lib.Fold.MOUNTAIN),
            _seg(0.5, 0.5, 1.0, 0.0, crease_lib.Fold.VALLEY),
        ],
        light_mode=True,
    )
    model = fold_3d.fold_3d(cp)
    self.assertLen(model.faces, 4)
    self.assertAlmostEqual(model.vertex_error, 0.0, places=6)

    # Full unsplit X-crossing splits cleanly into 4 triangles.
    x_cp = cp_lib.CreasePattern(
        [
            *_border_creases(),
            _seg(0.0, 0.0, 1.0, 1.0, crease_lib.Fold.HINGE),
            _seg(0.0, 1.0, 1.0, 0.0, crease_lib.Fold.HINGE),
        ],
        light_mode=True,
    )
    x_graph = fold_3d.build_fold_graph(x_cp)
    self.assertLen(x_graph.faces, 4)

  def test_merges_redundant_collinear_degree2_vertices(self):
    cp = cp_lib.CreasePattern(
        [
            _seg(0.0, 0.0, 0.5, 0.0, crease_lib.Fold.BORDER),
            _seg(0.5, 0.0, 1.0, 0.0, crease_lib.Fold.BORDER),
            _seg(1.0, 0.0, 1.0, 1.0, crease_lib.Fold.BORDER),
            _seg(1.0, 1.0, 0.0, 1.0, crease_lib.Fold.BORDER),
            _seg(0.0, 1.0, 0.0, 0.0, crease_lib.Fold.BORDER),
        ],
        light_mode=True,
    )
    merged = fold_3d.build_fold_graph(cp, merge_collinear=True)
    unmerged = fold_3d.build_fold_graph(cp, merge_collinear=False)
    self.assertLen(merged.vertices_2d, 4)
    self.assertLen(unmerged.vertices_2d, 5)

  def test_locates_points_accurately_on_nonconvex_face(self):
    # Unit square with a recessed inner top-right square creates a concave L.
    b = crease_lib.Fold.BORDER
    f = crease_lib.Fold.HINGE
    cp = cp_lib.CreasePattern(
        [
            _seg(0.0, 0.0, 1.0, 0.0, b),
            _seg(1.0, 0.0, 1.0, 0.5, b),
            _seg(1.0, 0.5, 1.0, 1.0, b),
            _seg(1.0, 1.0, 0.5, 1.0, b),
            _seg(0.5, 1.0, 0.0, 1.0, b),
            _seg(0.0, 1.0, 0.0, 0.0, b),
            _seg(0.5, 1.0, 0.5, 0.5, f),
            _seg(0.5, 0.5, 1.0, 0.5, f),
        ],
        light_mode=True,
    )
    model = fold_3d.fold_3d(cp)
    self.assertLen(model.faces, 2)
    faces = model.locate_faces(
        np.array([[0.25, 0.25], [0.75, 0.75], [1.5, 0.5]])
    )
    self.assertNotEqual(int(faces[0]), int(faces[1]))
    self.assertGreaterEqual(int(faces[0]), 0)
    self.assertGreaterEqual(int(faces[1]), 0)
    self.assertEqual(int(faces[2]), -1)

    means, spreads = model.locate_groups({
        'segment': np.array([[0.1, 0.2], [0.3, 0.2]]),
        'outside': np.array([[2.0, 2.0]]),
    })
    self.assertNotIn('outside', means)
    np.testing.assert_allclose(means['segment'], [0.2, 0.2, 0.0], atol=1e-9)
    self.assertAlmostEqual(spreads['segment'], 0.1, places=9)

  def test_rejects_disconnected_and_incompatible_folds_in_strict_mode(self):
    b = crease_lib.Fold.BORDER
    disconnected = cp_lib.CreasePattern(
        [
            _seg(0.0, 0.0, 0.3, 0.0, b),
            _seg(0.3, 0.0, 0.3, 0.3, b),
            _seg(0.3, 0.3, 0.0, 0.3, b),
            _seg(0.0, 0.3, 0.0, 0.0, b),
            _seg(0.6, 0.6, 0.9, 0.6, b),
            _seg(0.9, 0.6, 0.9, 0.9, b),
            _seg(0.9, 0.9, 0.6, 0.9, b),
            _seg(0.6, 0.9, 0.6, 0.6, b),
        ],
        light_mode=True,
    )
    with self.assertRaisesRegex(ValueError, 'disconnected'):
      fold_3d.fold_3d(disconnected, strict=True)

    # Four 180-degree creases with unequal alternating sectors violate Kawasaki.
    m = crease_lib.Fold.MOUNTAIN
    v = crease_lib.Fold.VALLEY
    incompatible = cp_lib.CreasePattern(
        [
            *_border_creases(),
            _seg(0.5, 0.5, 0.0, 0.0, m),
            _seg(0.5, 0.5, 1.0, 0.0, m),
            _seg(0.5, 0.5, 1.0, 1.0, m),
            _seg(0.5, 0.5, 0.2, 1.0, v),
        ],
        light_mode=False,
    )
    with self.assertRaisesRegex(ValueError, 'strain'):
      fold_3d.fold_3d(incompatible, strict=True, max_vertex_error=1.0)
    unconstrained = fold_3d.fold_3d(incompatible, strict=False)
    self.assertGreater(unconstrained.vertex_error, 1.0)

  def test_fold_svg_and_public_fold_3d_round_trip(self):
    res = bp.fold('cat', preset='demo', shape=True)
    model = bp.fold_3d(res)
    self.assertLess(model.vertex_error, 0.05)
    from_svg = fold_3d.fold_svg(bp.to_svg(res))
    self.assertLess(from_svg.vertex_error, 0.05)
    fold_dict = model.to_fold_dict('cat')
    self.assertEqual(fold_dict['file_spec'], 1.2)
    self.assertLen(fold_dict['faces_vertices'], len(model.faces))

  def test_locate_faces_contains_border_and_crease_points_on_cw_faces(self):
    # Diagonal ridge meeting the top border creates CW triangle and quad faces
    # with overlapping bounding boxes sharing vertices on the 180-degree crease.
    # Boundary points on y=0 and x=1 must stay inside the true triangle face
    # rather than jumping across the fold to the adjacent quad.
    m = crease_lib.Fold.MOUNTAIN
    v = crease_lib.Fold.VALLEY
    cp = cp_lib.CreasePattern(
        [
            *_border_creases(),
            _seg(0.0, 0.0, 0.5, 0.5, m),
            _seg(0.5, 0.5, 1.0, 0.0, m),
            _seg(0.5, 0.5, 0.0, 1.0, m),
            _seg(0.5, 0.5, 1.0, 1.0, v),
        ],
        light_mode=True,
    )
    model = fold_3d.fold_3d(cp)
    self.assertLen(model.faces, 4)
    border_pts = np.array([
        [0.5, 0.0],
        [0.25, 0.0],
        [0.75, 0.0],
        [1.0, 0.5],
        [0.0, 0.5],
        [0.5, 1.0],
        [0.5, 0.5],
    ])
    faces = model.locate_faces(border_pts)
    self.assertTrue(bool(np.all(faces >= 0)))
    for pt, f_idx in zip(border_pts, faces, strict=True):
      poly = model.vertices_2d[model.faces[int(f_idx)]]
      # Point must lie within 1e-9 of an edge of its assigned face.
      seg_start = poly
      seg_end = np.roll(poly, -1, axis=0)
      diff = seg_end - seg_start
      len_sq = np.sum(diff * diff, axis=1)
      t = np.clip(
          np.sum((pt[None, :] - seg_start) * diff, axis=1)
          / np.where(len_sq > 1e-24, len_sq, 1.0),
          0.0,
          1.0,
      )
      proj = seg_start + t[:, None] * diff
      dist = float(np.min(np.linalg.norm(pt[None, :] - proj, axis=1)))
      self.assertLessEqual(dist, 1e-9, msg=f'pt={pt} mapped outside face')

    # Symmetric pair (0.25, 0.0) and (0.75, 0.0) must fold symmetrically to the
    # same distance from the center vertex (0.5, 0.5).
    locs = model.locate(np.array([[0.25, 0.0], [0.75, 0.0], [0.5, 0.5]]))
    d_left = float(np.linalg.norm(locs[0] - locs[2]))
    d_right = float(np.linalg.norm(locs[1] - locs[2]))
    expected = float(np.hypot(0.25, 0.5))
    self.assertAlmostEqual(d_left, expected, places=6)
    self.assertAlmostEqual(d_right, expected, places=6)


if __name__ == '__main__':
  absltest.main()
