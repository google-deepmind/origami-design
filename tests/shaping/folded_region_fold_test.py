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

"""Tests for folded-state reverse/crimp folds and planar cut utilities."""

import math
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from bp_pipeline.crease_pattern import crease as crease_utils
from bp_pipeline.crease_pattern import crease_pattern as cp_utils
from bp_pipeline.crease_pattern import point as point_utils
from bp_pipeline.shaping import folded_region_fold
from bp_pipeline.shaping import geometric_simple_fold
from bp_pipeline.shaping import shaping_utils
from bp_pipeline.third_party.flat_folder import graph_creation
import numpy as np


def _pt(x: float, y: float) -> point_utils.Point:
  return point_utils.Point(x=x, y=y)


def _border_creases() -> list[crease_utils.Crease]:
  b = crease_utils.Fold.BORDER
  return [
      crease_utils.Crease(_pt(0.0, 0.0), _pt(1.0, 0.0), fold=b),
      crease_utils.Crease(_pt(1.0, 0.0), _pt(1.0, 1.0), fold=b),
      crease_utils.Crease(_pt(1.0, 1.0), _pt(0.0, 1.0), fold=b),
      crease_utils.Crease(_pt(0.0, 1.0), _pt(0.0, 0.0), fold=b),
  ]


def _book_fold_cp() -> cp_utils.CreasePattern:
  """Two-layer vertical valley fold along x = 0.5."""
  return cp_utils.CreasePattern(
      creases=[
          *_border_creases(),
          crease_utils.Crease(
              _pt(0.5, 0.0), _pt(0.5, 1.0), fold=crease_utils.Fold.VALLEY
          ),
      ],
      grid_size=8,
      light_mode=False,
  )


def _double_pleat_cp() -> cp_utils.CreasePattern:
  """Non-alternating narrowed-style vertical pleat stack (V, V, M, M)."""
  v = crease_utils.Fold.VALLEY
  m = crease_utils.Fold.MOUNTAIN
  xs = (0.2, 0.4, 0.6, 0.8)
  folds = (v, v, m, m)
  creases = list(_border_creases())
  for x, fold in zip(xs, folds, strict=True):
    creases.append(crease_utils.Crease(_pt(x, 0.0), _pt(x, 1.0), fold=fold))
  return cp_utils.CreasePattern(creases=creases, grid_size=10, light_mode=False)


class GeometricCutUtilsTest(parameterized.TestCase):

  def test_fit_2d_isometry_rotation_and_reflection(self):
    source = [(0.0, 0.0), (2.0, 0.0), (0.0, 1.0)]
    angle = math.radians(37.0)
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    rotated = [
        (cos_a * x - sin_a * y + 0.3, sin_a * x + cos_a * y - 0.7)
        for x, y in source
    ]
    rot, s_mean, t_mean = geometric_simple_fold.fit_2d_isometry(source, rotated)
    for src, expected in zip(source, rotated, strict=True):
      mapped = geometric_simple_fold.apply_2d_isometry(src, rot, s_mean, t_mean)
      self.assertAlmostEqual(mapped[0], expected[0], places=9)
      self.assertAlmostEqual(mapped[1], expected[1], places=9)

    reflected = [(-x + 1.0, y + 0.5) for x, y in source]
    rot, s_mean, t_mean = geometric_simple_fold.fit_2d_isometry(
        source, reflected
    )
    for src, expected in zip(source, reflected, strict=True):
      mapped = geometric_simple_fold.apply_2d_isometry(src, rot, s_mean, t_mean)
      self.assertAlmostEqual(mapped[0], expected[0], places=9)
      self.assertAlmostEqual(mapped[1], expected[1], places=9)

    with self.assertRaises(ValueError):
      geometric_simple_fold.fit_2d_isometry([], [])
    with self.assertRaises(ValueError):
      geometric_simple_fold.fit_2d_isometry([(0.0, 0.0)], [])

  def test_fast_create_graph_fallback_and_planarity_guards(self):
    # Split border at (0.5, 0) and (0.5, 1) so lines are pre-split planar edges,
    # including a decimal-boundary perturbation of 3e-10.
    planar_lines = [
        ((0.0, 0.0), (0.5, 0.0)),
        ((0.5 + 3e-10, 0.0), (1.0, 0.0)),
        ((1.0, 0.0), (1.0, 1.0)),
        ((1.0, 1.0), (0.5, 1.0)),
        ((0.5, 1.0), (0.0, 1.0)),
        ((0.0, 1.0), (0.0, 0.0)),
        ((0.5, 0.0), (0.5, 1.0)),
    ]
    with mock.patch.object(
        graph_creation,
        '_create_graph_from_lines_eps',
        side_effect=ValueError('Missing Endpoint'),
    ):
      vertices, edges, _ = geometric_simple_fold.fast_create_graph(planar_lines)
    self.assertLen(vertices, 6)
    self.assertLen(edges, 7)

    # T-junction (unsplit border edge) must be rejected by planar fallback.
    unsplit = [
        ((0.0, 0.0), (1.0, 0.0)),
        ((1.0, 0.0), (1.0, 1.0)),
        ((1.0, 1.0), (0.0, 1.0)),
        ((0.0, 1.0), (0.0, 0.0)),
        ((0.5, 0.0), (0.5, 1.0)),
    ]
    with mock.patch.object(
        graph_creation,
        '_create_graph_from_lines_eps',
        side_effect=ValueError('Missing Endpoint'),
    ):
      v_t, e_t, _ = geometric_simple_fold.fast_create_graph(unsplit)
    self.assertEmpty(v_t)
    self.assertEmpty(e_t)

    # Proper X-crossing must also be rejected by planar fallback.
    x_crossing = [
        ((0.0, 0.0), (1.0, 1.0)),
        ((0.0, 1.0), (1.0, 0.0)),
    ]
    with mock.patch.object(
        graph_creation,
        '_create_graph_from_lines_eps',
        side_effect=ValueError('Missing Endpoint'),
    ):
      v_x, e_x, _ = geometric_simple_fold.fast_create_graph(x_crossing)
    self.assertEmpty(v_x)
    self.assertEmpty(e_x)

  def test_clipped_folded_points_and_segment_cut_creases(self):
    cp = _book_fold_cp()
    graph_data = geometric_simple_fold.prepare_graph_data(cp)
    # Sub-rectangle strictly inside top half (no face centroid inside if tall).
    sub_rect = shaping_utils.Rectangle(
        name='distal', x_start=0.0, y_start=0.6, x_len=1.0, y_len=0.4
    )
    pts = geometric_simple_fold.clipped_folded_points(graph_data, sub_rect)
    self.assertGreaterEqual(len(pts), 6)
    self.assertTrue(
        geometric_simple_fold.check_inside_limiting_polygon(
            (0.5, 0.8), sub_rect
        )
    )
    self.assertFalse(
        geometric_simple_fold.check_inside_limiting_polygon(
            (0.5, 0.2), sub_rect
        )
    )

    # Cut across both layers inside the folded x-span vs outside the span.
    pts_arr = np.array(pts)
    y_mid = float(pts_arr[:, 1].mean())
    x_lo, x_hi = float(pts_arr[:, 0].min()), float(pts_arr[:, 0].max())
    hit_creases = geometric_simple_fold.get_segment_cut_creases(
        graph_data=graph_data,
        cut_line_start=(0.0, y_mid),
        cut_line_end=(1.0, y_mid),
        span=(x_lo - 0.1, x_hi + 0.1),
        limiting_polygon=sub_rect,
    )
    self.assertLen(hit_creases, 2)
    miss_creases = geometric_simple_fold.get_segment_cut_creases(
        graph_data=graph_data,
        cut_line_start=(0.0, y_mid),
        cut_line_end=(1.0, y_mid),
        span=(x_hi + 5.0, x_hi + 6.0),
        limiting_polygon=sub_rect,
    )
    self.assertEmpty(miss_creases)

    with self.assertRaises(ValueError):
      geometric_simple_fold.get_segment_cut_creases(
          graph_data=graph_data,
          cut_line_start=(0.2, 0.2),
          cut_line_end=(0.2, 0.2),
          span=(-1.0, 1.0),
          limiting_polygon=sub_rect,
      )


class FoldedRegionFoldTest(parameterized.TestCase):

  def test_folded_axis_snap_and_cut_line_builders(self):
    cp = _book_fold_cp()
    graph_data = geometric_simple_fold.prepare_graph_data(cp)
    full_paper = shaping_utils.Rectangle(
        name='flap', x_start=0.0, y_start=0.0, x_len=1.0, y_len=1.0
    )
    pts = folded_region_fold.element_folded_points(graph_data, full_paper)
    self.assertEqual(pts.shape[1], 2)

    # Book fold has footprint width 0.5 x length 1.0 (aspect 2.0).
    axis = folded_region_fold.get_element_folded_axis(
        graph_data, full_paper, reference_point=np.array([0.25, -1.0])
    )
    self.assertAlmostEqual(axis.length, 1.0, places=5)
    self.assertAlmostEqual(axis.width, 0.5, places=5)
    self.assertGreater(axis.tip[1], axis.base[1])

    # snap_to_points snaps within threshold and leaves point unchanged outside.
    candidates = np.array([[0.0, 0.0], [1.0, 0.0]])
    np.testing.assert_allclose(
        folded_region_fold.snap_to_points(
            np.array([0.05, 0.0]), candidates, 0.1
        ),
        [0.0, 0.0],
    )
    np.testing.assert_allclose(
        folded_region_fold.snap_to_points(
            np.array([0.5, 0.0]), candidates, 0.1
        ),
        [0.5, 0.0],
    )
    np.testing.assert_allclose(
        folded_region_fold.snap_to_points(np.array([0.5, 0.0]), None, 0.1),
        [0.5, 0.0],
    )

    rev_line = folded_region_fold.reverse_fold_cut_line(
        axis, angle=60.0, depth=0.5
    )
    lo, hi = folded_region_fold.line_span(rev_line, pts, margin=0.05)
    self.assertLess(lo, hi)

    crimp_lines = folded_region_fold.crimp_fold_cut_lines(
        axis, angle=40.0, depth=0.4
    )
    self.assertLen(crimp_lines, 2)

    with self.assertRaises(ValueError):
      folded_region_fold.reverse_fold_cut_line(axis, angle=0.0, depth=0.5)
    with self.assertRaises(ValueError):
      folded_region_fold.crimp_fold_cut_lines(axis, angle=180.0, depth=0.5)

  def test_reverse_fold_and_crimp_on_book_fold(self):
    cp = _book_fold_cp()
    graph_data = geometric_simple_fold.prepare_graph_data(cp)
    full_paper = shaping_utils.Rectangle(
        name='flap', x_start=0.0, y_start=0.0, x_len=1.0, y_len=1.0
    )
    axis = folded_region_fold.get_element_folded_axis(
        graph_data, full_paper, reference_point=np.array([0.25, -1.0])
    )

    # Reverse fold at depth 0.5 turning by 60 degrees produces flat-foldable
    # inside and outside reverse fold variants (flipping the distal center
    # crease).
    rev_line = folded_region_fold.reverse_fold_cut_line(
        axis, angle=60.0, depth=0.5
    )
    rev_variants = folded_region_fold.get_inverted_region_variants(
        cp,
        cut_lines=[rev_line],
        limiting_polygon=full_paper,
        graph_data=graph_data,
    )
    ff_rev = [v for v in rev_variants if v['is_flat_foldable']]
    self.assertNotEmpty(ff_rev)
    for variant in ff_rev:
      self.assertEqual(variant['num_new_creases'], 2)
      self.assertEqual(variant['num_flipped_creases'], 1)
      self.assertTrue(variant['crease_pattern'].local_flat_foldability())

    # Crimp fold at depth 0.35 turning by 40 degrees flips only the central
    # strip segment of the spine and preserves local flat-foldability.
    crimp_lines = folded_region_fold.crimp_fold_cut_lines(
        axis, angle=40.0, depth=0.35, gap=0.2
    )
    crimp_variants = folded_region_fold.get_inverted_region_variants(
        cp,
        cut_lines=crimp_lines,
        limiting_polygon=full_paper,
        graph_data=graph_data,
    )
    ff_crimp = [v for v in crimp_variants if v['is_flat_foldable']]
    self.assertNotEmpty(ff_crimp)
    for variant in ff_crimp:
      self.assertEqual(variant['num_new_creases'], 4)
      self.assertEqual(variant['num_flipped_creases'], 1)
      self.assertTrue(variant['crease_pattern'].local_flat_foldability())

  def test_multi_layer_pleat_crimp_and_proximal_guard(self):
    cp = _double_pleat_cp()
    graph_data = geometric_simple_fold.prepare_graph_data(cp)
    full_paper = shaping_utils.Rectangle(
        name='pleat_flap', x_start=0.0, y_start=0.0, x_len=1.0, y_len=1.0
    )
    axis = folded_region_fold.get_element_folded_axis(
        graph_data, full_paper, reference_point=np.array([0.0, -1.0])
    )
    crimp_lines = folded_region_fold.crimp_fold_cut_lines(
        axis, angle=30.0, depth=0.35, gap=0.2
    )
    variants = folded_region_fold.get_inverted_region_variants(
        cp,
        cut_lines=crimp_lines,
        limiting_polygon=full_paper,
        graph_data=graph_data,
    )
    self.assertTrue(any(v['is_flat_foldable'] for v in variants))

    # Proximal polygon covering the rightmost pleat layer triggers a separation
    # error when flood-filling from left layers across x = 0.6.
    right_layer_guard = shaping_utils.Rectangle(
        name='blocked', x_start=0.6, y_start=0.0, x_len=0.4, y_len=1.0
    )
    rev_line = folded_region_fold.reverse_fold_cut_line(
        axis, angle=45.0, depth=0.5
    )
    with self.assertRaises(ValueError):
      folded_region_fold.get_inverted_region_variants(
          cp,
          cut_lines=[rev_line],
          limiting_polygon=full_paper,
          graph_data=graph_data,
          proximal_polygon=right_layer_guard,
      )


if __name__ == '__main__':
  absltest.main()
