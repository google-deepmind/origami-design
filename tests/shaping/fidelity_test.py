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

"""Tests for paper layout, pose alignment, 3D overlays, and fidelity scoring."""

import dataclasses
import functools
import math
from typing import Any

from absl.testing import absltest
from absl.testing import parameterized
import bp_pipeline as bp
from bp_pipeline import folder_3d as fold_3d
from bp_pipeline.crease_pattern import crease as crease_lib
from bp_pipeline.crease_pattern import crease_pattern as cp_lib
from bp_pipeline.crease_pattern import point as point_lib
from bp_pipeline.shaping import fidelity
from bp_pipeline.shaping import overlay_3d
from bp_pipeline.shaping import paper_layout
from bp_pipeline.shaping import pose_alignment
import numpy as np


@functools.lru_cache(maxsize=4)
def _load_solved(example_name: str) -> bp.SolvedCreasePattern:
  """Packs and solves a built-in example keeping the central crease intact."""
  design = bp.prepare(example_name)
  packed = bp.pack(
      design,
      bp.PackingTaskConfig(
          packing_config=bp.PackingConfig(use_level_shifter=False),
          max_retries=5,
          max_cps_per_grid_size=1,
          max_number_of_cps=1,
      ),
  )
  try:
    return bp.solve_best(
        packed,
        bp.SolverConfig(
            max_number_of_cps=1,
            max_steps=4000,
            hinges_from_stick_figure=True,
            remove_central_crease=False,
            up_down_method='flaps_down',
        ),
    )
  except bp.SolvingFailedError:
    for cand in bp.solve_sweep(packed):
      if not cand.central_crease_removed:
        return cand
    raise


def _rotation(axis: list[float], angle: float) -> np.ndarray:
  unit = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
  skew = np.array([
      [0.0, -unit[2], unit[1]],
      [unit[2], 0.0, -unit[0]],
      [-unit[1], unit[0], 0.0],
  ])
  return (
      np.eye(3) + math.sin(angle) * skew + (1.0 - math.cos(angle)) * skew @ skew
  )


def _synthetic_geometry(
    fold_transform: Any,
    overrides: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
  """Builds a synthetic two-legged `fidelity.measure` dictionary."""
  targets = {
      'hip': [0.0, 0.0, 0.0],
      'neck': [2.0, 0.0, 0.0],
      'knee_l': [0.0, 1.0, -0.5],
      'foot_l': [0.5, 1.5, -1.5],
      'knee_r': [0.0, -1.0, -0.5],
      'foot_r': [0.5, -1.5, -1.5],
      'head': [3.0, 0.0, 1.0],
  }
  sticks = {
      'body': ('hip', 'neck', None, 2.0, 1.0, None),
      'head': ('neck', 'head', 'body', 1.0, 1.0, None),
      'thigh_l': ('hip', 'knee_l', 'body', 1.0, 0.5, 'thigh_r'),
      'shin_l': ('knee_l', 'foot_l', 'thigh_l', 1.0, 0.5, 'shin_r'),
      'thigh_r': ('hip', 'knee_r', 'body', 1.0, 0.5, 'thigh_l'),
      'shin_r': ('knee_r', 'foot_r', 'thigh_r', 1.0, 0.5, 'shin_l'),
  }
  folded = {
      name: np.asarray(
          fold_transform(np.asarray(pos, dtype=float)), dtype=float
      )
      for name, pos in targets.items()
  }
  folded.update(overrides or {})
  sections: dict[str, dict[str, Any]] = {}
  for name, (near, far, _, _, width, _) in sticks.items():
    for frac in paper_layout.SECTION_FRACTIONS:
      pt = (1.0 - frac) * folded[near] + frac * folded[far]
      sections[f'{name}@{frac:g}'] = {'folded': list(pt), 'width': width}
  return {
      'root': 'body',
      'sticks': {
          name: {
              'length': length,
              'width': width,
              'proximal': near,
              'distal': far,
              'parent': parent,
              'is_river': False,
              'symmetric': sym,
          }
          for name, (near, far, parent, length, width, sym) in sticks.items()
      },
      'chains': {
          'head': ['head'],
          'leg_l': ['thigh_l', 'shin_l'],
          'leg_r': ['thigh_r', 'shin_r'],
      },
      'joints': {
          name: {'folded': list(folded[name]), 'target': pos}
          for name, pos in targets.items()
      },
      'sections': sections,
      'model_extents': [1.0, 0.8, 0.6],
      'body_extents': [1.0, 0.5, 0.0],
  }


class SimilarityTest(parameterized.TestCase):

  @parameterized.parameters(False, True)
  def test_recovers_similarity(self, mirrored: bool):
    rng = np.random.default_rng(0)
    source = rng.normal(size=(12, 3))
    rot = _rotation([1.0, 2.0, 0.5], 0.7)
    if mirrored:
      rot = rot @ np.diag([1.0, -1.0, 1.0])
    target = 2.5 * source @ rot.T + np.array([1.0, -2.0, 0.5])
    fit = pose_alignment.fit_similarity(source, target)
    np.testing.assert_allclose(fit.apply(source), target, atol=1e-8)
    self.assertAlmostEqual(fit.scale, 2.5)
    self.assertEqual(fit.reflected, mirrored)

  def test_proper_fit_never_reflects(self):
    rng = np.random.default_rng(1)
    source = rng.normal(size=(10, 3))
    target = source * np.array([1.0, -1.0, 1.0])
    fit = pose_alignment.fit_similarity(source, target, allow_reflection=False)
    self.assertFalse(fit.reflected)
    self.assertAlmostEqual(float(np.linalg.det(fit.rotation)), 1.0)

  def test_robust_fit_ignores_outlier(self):
    rng = np.random.default_rng(2)
    source = rng.normal(size=(15, 3))
    target = source @ _rotation([0.0, 0.0, 1.0], 1.1).T
    target[0] += np.array([5.0, 5.0, 5.0])
    fit, weights = pose_alignment.fit_similarity_robust(
        source, target, size=1.0
    )
    np.testing.assert_allclose(fit.apply(source[1:]), target[1:], atol=1e-2)
    self.assertLess(float(weights[0]), 0.01)


class ComponentsTest(absltest.TestCase):

  def test_rigidly_moved_figure_scores_perfectly(self):
    rot = _rotation([0.3, -1.0, 0.2], 2.0)
    scores, details, alignment = fidelity.components(
        _synthetic_geometry(lambda p: 0.1 * rot @ p + 7.0)
    )
    for name in (
        'pose',
        'direction',
        'separation',
        'width',
        'depth',
        'symmetry',
    ):
      self.assertAlmostEqual(scores[name], 1.0, places=3, msg=name)
    self.assertEqual(details['limb_articulation'], {'leg_l': 1.0, 'leg_r': 1.0})
    self.assertAlmostEqual(
        float(np.linalg.det(alignment.render_rotation)), 1.0, places=4
    )
    self.assertEqual(alignment.render_frame, 'stick_figure')

  def test_mirror_image_scores_perfectly(self):
    scores, details, _ = fidelity.components(
        _synthetic_geometry(lambda p: p * np.array([1.0, 1.0, -1.0]))
    )
    self.assertTrue(details['mirrored'])
    self.assertAlmostEqual(scores['pose'], 1.0)

  def test_stacked_legs_and_flat_model_are_penalized(self):
    good_scores, _, _ = fidelity.components(_synthetic_geometry(lambda p: p))
    stacked_scores, details, _ = fidelity.components(
        _synthetic_geometry(
            lambda p: np.array([p[0], 0.0, 0.1 * abs(p[1]) + p[2]])
        )
    )
    self.assertIn('thigh_l/thigh_r', details['stacked_pairs'])
    self.assertLess(stacked_scores['separation'], 0.5)
    self.assertLess(stacked_scores['depth'], 0.5)
    self.assertLess(
        fidelity.combine(stacked_scores), fidelity.combine(good_scores) - 3.0
    )

  def test_wide_and_asymmetric_flaps_are_penalized(self):
    geom = _synthetic_geometry(lambda p: p)
    for name, sec in geom['sections'].items():
      if name.startswith(('thigh', 'shin')):
        sec['width'] = 1.0
    scores, _, _ = fidelity.components(geom)
    self.assertLess(scores['width'], 0.8)

    asym_geom = _synthetic_geometry(lambda p: p)
    asym_geom['sections']['thigh_r@0.5']['width'] = 2.2
    asym_scores, asym_details, _ = fidelity.components(asym_geom)
    self.assertIn('thigh_l/thigh_r', asym_details['asymmetric_width_pairs'])
    self.assertLess(asym_scores['symmetry'], 0.85)

  def test_planar_figure_folded_flat_has_full_depth(self):
    flatten = np.array([1.0, 1.0, 0.0])
    geom = _synthetic_geometry(lambda p: p * flatten)
    for joint in geom['joints'].values():
      joint['target'] = list(np.asarray(joint['target']) * flatten)
    scores, details, _ = fidelity.components(geom)
    self.assertEqual(details['target_depth_ratio'], 0.0)
    self.assertAlmostEqual(scores['depth'], 1.0)

    flat_scores, _, _ = fidelity.components(
        _synthetic_geometry(lambda p: p * flatten)
    )
    self.assertLess(flat_scores['depth'], 0.5)

  def test_straight_limbs_get_zero_articulation(self):
    _, details, _ = fidelity.components(
        _synthetic_geometry(
            lambda p: p,
            overrides={
                'knee_l': np.array([0.25, 0.75, -0.75]),
                'knee_r': np.array([0.25, -0.75, -0.75]),
            },
        )
    )
    self.assertEqual(details['limb_articulation'], {'leg_l': 0.0, 'leg_r': 0.0})

  def test_collapsed_bar_falls_back_to_root_frame(self):
    geom = _synthetic_geometry(
        lambda p: np.array([p[0] + 0.1 * p[1] + 0.1 * p[2], 0.0, 0.0])
    )
    geom['root_normal'] = [0.0, 1.0, 0.0]
    _, _, alignment = fidelity.components(geom)
    self.assertEqual(alignment.render_frame, 'root')
    self.assertIn('collinear', alignment.render_frame_reason)
    rot = alignment.render_rotation
    self.assertAlmostEqual(float(np.linalg.det(rot)), 1.0, places=4)
    np.testing.assert_allclose(
        rot @ [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], atol=1e-4
    )
    self.assertAlmostEqual(
        abs(float((rot @ [0.0, 1.0, 0.0])[2])), 1.0, places=4
    )

  def test_cleanliness_and_clean_gate(self):
    self.assertTrue(fidelity.clean_gate(0.05, 0).passed)
    self.assertFalse(fidelity.clean_gate(0.05, 2).passed)
    self.assertFalse(fidelity.clean_gate(0.89, 0).passed)
    clean, _, _ = fidelity.components(_synthetic_geometry(lambda p: p))
    messy, _, _ = fidelity.components(
        _synthetic_geometry(lambda p: p), vertex_error=0.89, non_ff_vertices=2
    )
    self.assertAlmostEqual(clean['cleanliness'], 1.0)
    self.assertLess(messy['cleanliness'], 0.25)
    self.assertLess(fidelity.combine(messy), fidelity.combine(clean) - 1.2)

  def test_silhouette_penalizes_thin_wire_on_wide_body(self):
    geom = _synthetic_geometry(lambda p: p)
    geom['sticks']['body']['width'] = 4.0
    geom['sticks']['body']['is_river'] = True
    layout = paper_layout.PaperLayout(
        joints={},
        sections={},
        chains={},
        grid_size=1,
        grid_units_per_tree_unit=1.0,
        flap_rectangles=(),
    )

    def _make_mesh(half_span: float) -> fold_3d.FoldedModel:
      verts_3d = np.array([
          [0.0, -half_span, -half_span],
          [3.0, -half_span, -half_span],
          [3.0, half_span, half_span],
          [0.0, half_span, half_span],
          [0.0, -half_span, half_span],
          [3.0, -half_span, half_span],
          [3.0, half_span, -half_span],
          [0.0, half_span, -half_span],
      ])
      return fold_3d.FoldedModel(
          vertices_2d=np.zeros((8, 2)),
          vertices_3d=verts_3d,
          faces=(np.array([0, 1, 2, 3]), np.array([4, 5, 6, 7])),
          edges=np.zeros((0, 2), dtype=int),
          assignments=(),
          fold_angles=np.zeros(0),
          face_matrices=np.broadcast_to(np.eye(4), (2, 4, 4)).copy(),
      )

    _, wire, _ = fidelity.components(
        geom, layout=layout, model=_make_mesh(0.08)
    )
    _, plate, _ = fidelity.components(
        geom, layout=layout, model=_make_mesh(1.4)
    )
    self.assertLess(wire['wide_element_coverage'], 0.45)
    self.assertLess(wire['local_body_volume'], 0.35)
    self.assertGreater(plate['wide_element_coverage'], 0.80)
    self.assertGreater(plate['local_body_volume'], 0.80)

  def test_geometric_mean_combine_and_floor(self):
    ones = {name: 1.0 for name in fidelity.DEFAULT_WEIGHTS}
    self.assertAlmostEqual(fidelity.combine(ones), 10.0)
    halves = {name: 0.5 for name in fidelity.DEFAULT_WEIGHTS}
    self.assertAlmostEqual(fidelity.combine(halves), 5.0)
    zero_depth = dict(ones, depth=0.0)
    self.assertGreater(fidelity.combine(zero_depth), 0.0)
    self.assertLess(fidelity.combine(zero_depth), 6.0)


class SideCrossingsAndPoseGateTest(absltest.TestCase):

  def test_crossed_and_midline_feet_fail_pose_gate(self):
    scores, details, _ = fidelity.components(
        _synthetic_geometry(
            lambda p: p,
            overrides={
                'foot_l': np.array([0.5, -1.0, -1.5]),
                'foot_r': np.array([0.5, 0.0, -1.5]),
            },
        )
    )
    crossings = details['side_crossings']
    by_pt = {(c['stick'], c['point']): c for c in crossings}
    self.assertEqual(by_pt['shin_l', 'tip']['kind'], 'crossed')
    self.assertEqual(by_pt['shin_r', 'tip']['kind'], 'midline')
    self.assertEqual(crossings[0]['stick'], 'shin_l')
    gate = fidelity.pose_gate(scores, details)
    self.assertFalse(gate.passed)
    self.assertCountEqual(gate.crossing, ['shin_l', 'shin_r'])

  def test_midline_origin_and_proximal_hip_crossings(self):
    sticks = {
        'body': {'proximal': 'hips', 'distal': 'shoulders'},
        'leg_L': {'proximal': 'hips', 'distal': 'ankle_L'},
        'leg_R': {'proximal': 'hips', 'distal': 'ankle_R'},
    }
    design = {
        'hips': [0.0, 0.0, 0.0],
        'shoulders': [3.0, 0.0, 0.0],
        'ankle_L': [0.0, 1.12, -1.9],
        'ankle_R': [0.0, -1.12, -1.9],
    }
    realized = {
        'hips': [0.0, 0.0, 0.0],
        'shoulders': [3.0, 0.0, 0.0],
        'ankle_L': [0.0, 0.63, -1.9],
        'ankle_R': [0.0, -0.53, -1.9],
    }
    crossings = pose_alignment.find_side_crossings(
        sticks,
        'body',
        design,
        realized,
        realized_sections={
            'leg_L@0': [0.0, -0.46, 0.0],
            'leg_L@0.5': [0.0, 0.08, -0.9],
            'leg_R@0': [0.0, 0.61, 0.0],
            'leg_R@0.5': [0.0, -0.05, -0.9],
        },
    )
    self.assertCountEqual(
        pose_alignment.crossing_sticks(crossings), ['leg_L', 'leg_R']
    )
    self.assertTrue(all(c.point == 'proximal' for c in crossings))

  def test_pose_gate_edge_cases(self):
    self.assertTrue(
        fidelity.pose_gate({'direction': 0.9}, {'stray_flaps': ['a']}).passed
    )
    self.assertFalse(
        fidelity.pose_gate({'direction': 0.6}, {'stray_flaps': []}).passed
    )
    stray_gate = fidelity.pose_gate(
        {'direction': 0.95}, {'stray_flaps': list('abc')}
    )
    self.assertFalse(stray_gate.passed)
    self.assertLen(stray_gate.stray, 3)
    crossing = {
        'stick': 'shin_l',
        'point': 'tip',
        'kind': 'midline',
        'design_side': 'left',
        'design_offset': 1.5,
        'realized_offset': 0.0,
    }
    dup_gate = fidelity.pose_gate(
        {'direction': 0.95},
        {'stray_flaps': [], 'side_crossings': [crossing, dict(crossing)]},
    )
    self.assertFalse(dup_gate.passed)
    self.assertEqual(dup_gate.crossing, ('shin_l',))
    self.assertEqual(dup_gate.to_dict()['num_crossing'], 1)

  def test_points_near_axis_by_design_are_not_checked(self):
    sticks = {
        'body': {'proximal': 'a', 'distal': 'b'},
        'whisker': {'proximal': 'b', 'distal': 'c'},
        'arm': {'proximal': 'b', 'distal': 'd'},
    }
    design = {
        'a': [0.0, 0.0, 0.0],
        'b': [1.0, 0.0, 0.0],
        'c': [1.5, 0.3, 0.0],  # Within SIDE_MIN_DESIGN_OFFSET of midline.
        'd': [1.0, 2.0, 0.0],
    }
    realized = dict(design, c=[1.5, -0.3, 0.0], d=[1.0, -2.0, 0.0])
    crossings = pose_alignment.find_side_crossings(
        sticks, 'body', design, realized
    )
    self.assertEqual(pose_alignment.crossing_sticks(crossings), ['arm'])
    shifted = {k: np.asarray(v) + 5.0 for k, v in design.items()}
    self.assertEmpty(
        pose_alignment.find_side_crossings(sticks, 'body', design, shifted)
    )
    self.assertEmpty(
        pose_alignment.find_side_crossings(
            sticks, 'body', design, dict(design, d=None)
        )
    )

  def test_midline_rooted_stick_mid_section_tolerates_hub_origin(self):
    sticks = {
        'body': {'proximal': 'a', 'distal': 'b'},
        'horn_l': {'proximal': 'b', 'distal': 'c'},
        'shin_l': {'proximal': 'c', 'distal': 'd'},
    }
    design = {
        'a': [0.0, 0.0, 0.0],
        'b': [1.0, 0.0, 0.0],
        'c': [1.5, 1.4, 0.0],
        'd': [1.5, 1.8, -1.0],
    }
    realized = dict(design)
    crossings = pose_alignment.find_side_crossings(
        sticks,
        'body',
        design,
        realized,
        realized_sections={
            'horn_l@0.5': [1.25, -0.06, 0.0],
            'shin_l@0.5': [1.5, 0.0, -0.5],
        },
    )
    self.assertEqual(pose_alignment.crossing_sticks(crossings), ['shin_l'])
    crossed = pose_alignment.find_side_crossings(
        sticks,
        'body',
        design,
        realized,
        realized_sections={'horn_l@0.5': [1.25, -0.4, 0.0]},
    )
    self.assertEqual(pose_alignment.crossing_sticks(crossed), ['horn_l'])
    self.assertEqual(crossed[0].kind, 'crossed')


class Overlay3DTest(absltest.TestCase):

  def test_detects_strip_and_replay_partial_fold_overlay(self):
    b = crease_lib.Fold.BORDER
    v = crease_lib.Fold.VALLEY

    def _pt(x: float, y: float) -> point_lib.Point:
      return point_lib.Point(x=x, y=y)

    base = cp_lib.CreasePattern(
        [
            crease_lib.Crease(_pt(0.0, 0.0), _pt(1.0, 0.0), fold=b),
            crease_lib.Crease(_pt(1.0, 0.0), _pt(1.0, 1.0), fold=b),
            crease_lib.Crease(_pt(1.0, 1.0), _pt(0.0, 1.0), fold=b),
            crease_lib.Crease(_pt(0.0, 1.0), _pt(0.0, 0.0), fold=b),
            crease_lib.Crease(
                _pt(0.5, 0.0), _pt(0.5, 1.0), fold=v, fold_percent=1.0
            ),
        ],
        light_mode=False,
    )
    self.assertFalse(overlay_3d.has_3d_overlay(base))
    partial_creases = [
        dataclasses.replace(c, fold_percent=0.4) if c.fold == v else c
        for c in base.creases
    ]
    partial = cp_lib.CreasePattern(partial_creases, light_mode=True)
    self.assertTrue(overlay_3d.has_3d_overlay(partial))

    stripped = overlay_3d.as_flat_backbone(partial)
    self.assertFalse(overlay_3d.has_3d_overlay(stripped))

    replayed = overlay_3d.replay_3d_overlay(base, partial, stripped)
    mv = [c for c in replayed.creases if c.mountain_or_valley()]
    self.assertLen(mv, 1)
    self.assertAlmostEqual(mv[0].fold_percent, 0.4)


class EndToEndFidelityTest(parameterized.TestCase):

  @parameterized.parameters('cat', 'human')
  def test_unshaped_base_folds_every_stick_to_target_length(self, name: str):
    solved = _load_solved(name)
    assert solved.packed.design is not None
    tree = solved.packed.design.source
    layout = paper_layout.build_paper_layout(tree, solved.packed)
    model = fold_3d.fold_3d(solved)
    geom = fidelity.measure(tree, layout, model)
    checked = 0
    for stick_name, stick in geom['sticks'].items():
      near = geom['joints'][stick['proximal']]['folded']
      far = geom['joints'][stick['distal']]['folded']
      if near is None or far is None:
        continue
      length = float(np.linalg.norm(np.asarray(far) - np.asarray(near)))
      self.assertAlmostEqual(
          length, stick['length'], delta=0.15, msg=stick_name
      )
      checked += 1
    self.assertGreater(checked, len(geom['sticks']) // 2)

  def test_unshaped_base_scores_low_and_shaped_improves(self):
    res = bp.fold('cat', preset='demo', shape=True)
    solved_rep = bp.evaluate_fidelity(res.solved)
    self.assertIsNotNone(res.shaped)
    shaped_rep = bp.evaluate_fidelity(res.shaped)
    pipe_rep = bp.evaluate_fidelity(res)

    self.assertEqual(shaped_rep.fidelity, pipe_rep.fidelity)
    self.assertLess(solved_rep.components['depth'], 0.05)
    self.assertLess(solved_rep.fidelity, 4.5)
    self.assertGreater(shaped_rep.fidelity, solved_rep.fidelity)
    self.assertTrue(shaped_rep.clean_gate.locally_flat_foldable)
    self.assertLess(shaped_rep.fold_vertex_error_percent, 0.05)
    self.assertEqual(shaped_rep.render_rotation.shape, (3, 3))

    as_dict = shaped_rep.to_dict()
    for key in (
        'fidelity',
        'components',
        'pose_gate',
        'clean_gate',
        'alignment',
    ):
      self.assertIn(key, as_dict)

    assert res.packed.design is not None
    tree = res.packed.design.source
    base_model = bp.fold_3d(res.solved)
    shaped_model = bp.fold_3d(res.shaped)
    layout = paper_layout.build_paper_layout(tree, res.packed)
    _, dirs = fidelity.stick_directions(
        tree, res.packed, shaped_model, layout=layout
    )
    self.assertIn(tree.root, dirs)

    # Verify root_frame_directions handles true SE(3) face_matrices cleanly.
    root_flap = res.packed.flaps.get(tree.root)
    if root_flap is not None:
      g = float(res.packed.grid_size)
      rx = (root_flap.rect.x + 0.5 * root_flap.rect.width) / g
      ry = (root_flap.rect.y + 0.5 * root_flap.rect.height) / g
      root_dirs = pose_alignment.root_frame_directions(
          tree,
          layout,
          shaped_model,
          base_model,
          root_name=tree.root,
          root_points=np.array([[rx, ry]]),
          up_reference=np.array([0.0, 0.0, 1.0]),
      )
      self.assertIsNotNone(root_dirs)
      self.assertIn('directions', root_dirs)

  def test_shaping_flap_sticks_carry_no_pose_target_or_fidelity_term(self):
    geom_base = _synthetic_geometry(lambda p: p)
    scores_base, details_base, _ = fidelity.components(geom_base)
    self.assertEmpty(details_base['stray_flaps'])

    # Add an auxiliary `shaping_flap: True` stick pointing 90 degrees off its
    # nominal stick-figure angle and with mismatched width: all fidelity terms,
    # joint counts, and `stray_flaps` must stay identical to `geom_base`.
    geom_with_sf = _synthetic_geometry(lambda p: p)
    geom_with_sf['joints']['sf_tip'] = {
        'folded': [0.0, 1.0, 0.0],
        'target': [1.0, 0.0, 0.5],
    }
    geom_with_sf['sticks']['aux_sf'] = {
        'length': 1.0,
        'width': 1.0,
        'proximal': 'hip',
        'distal': 'sf_tip',
        'parent': 'body',
        'is_river': False,
        'shaping_flap': True,
        'symmetric': None,
    }
    geom_with_sf['chains']['aux_sf'] = ['aux_sf']
    geom_with_sf['sections']['aux_sf@0.5'] = {
        'folded': [0.0, 0.5, 0.0],
        'width': 0.25,
    }
    # Add a mirrored shaping-flap pair whose folded tips sit stacked on `foot_l`
    # with straight collinear sections so articulation, tip-clumping, and
    # mirrored-pair separation/symmetry exclusions are all exercised.
    clumped_pt = [0.5, 1.5, -1.5]
    geom_with_sf['joints']['sf_tip_l'] = {
        'folded': list(clumped_pt),
        'target': [0.5, 1.2, 0.5],
    }
    geom_with_sf['joints']['sf_tip_r'] = {
        'folded': list(clumped_pt),
        'target': [0.5, -1.2, 0.5],
    }
    for side, partner, tip_j in (
        ('aux_sf_l', 'aux_sf_r', 'sf_tip_l'),
        ('aux_sf_r', 'aux_sf_l', 'sf_tip_r'),
    ):
      geom_with_sf['sticks'][side] = {
          'length': 2.0,
          'width': 1.0,
          'proximal': 'hip',
          'distal': tip_j,
          'parent': 'body',
          'is_river': False,
          'shaping_flap': True,
          'symmetric': partner,
      }
      geom_with_sf['chains'][side] = [side]
      for frac in paper_layout.SECTION_FRACTIONS:
        geom_with_sf['sections'][f'{side}@{frac:g}'] = {
            'folded': [frac * c for c in clumped_pt],
            'width': 0.2,
        }
    scores_sf, details_sf, _ = fidelity.components(geom_with_sf)
    self.assertEmpty(details_sf['stray_flaps'])
    self.assertEmpty(details_sf['clumped_tips'])
    self.assertEmpty(details_sf['stacked_pairs'])
    self.assertNotIn('aux_sf', details_sf['direction_error_deg'])
    self.assertNotIn('aux_sf_l', details_sf['direction_error_deg'])
    self.assertNotIn('aux_sf', details_sf['widths'])
    for comp in (
        'direction',
        'articulation',
        'separation',
        'symmetry',
        'width',
        'pose',
    ):
      self.assertAlmostEqual(scores_sf[comp], scores_base[comp], places=6)
    self.assertEqual(
        details_sf['joints_located'], details_base['joints_located']
    )
    self.assertEqual(details_sf['joints_total'], details_base['joints_total'])

    # Verify `_silhouette_and_body_metrics` skips `shaping_flap` sticks from
    # recall while keeping their capsule in `union_loose` so opened shaping
    # paper does not penalize `silhouette_precision`.
    layout = paper_layout.PaperLayout(
        joints={},
        sections={},
        chains={},
        grid_size=1,
        grid_units_per_tree_unit=1.0,
        flap_rectangles=(),
    )
    body_mesh = fold_3d.FoldedModel(
        vertices_2d=np.zeros((4, 2)),
        vertices_3d=np.array([
            [0.0, -0.5, 0.0],
            [2.0, -0.5, 0.0],
            [2.0, 0.5, 0.0],
            [0.0, 0.5, 0.0],
        ]),
        faces=(np.array([0, 1, 2, 3]),),
        edges=np.zeros((0, 2), dtype=int),
        assignments=(),
        fold_angles=np.zeros(0),
        face_matrices=np.broadcast_to(np.eye(4), (1, 4, 4)).copy(),
    )
    sil_scores_base, sil_details_base, _ = fidelity.components(
        geom_base, layout=layout, model=body_mesh
    )
    sil_scores_sf, sil_details_sf, _ = fidelity.components(
        geom_with_sf, layout=layout, model=body_mesh
    )
    self.assertNotIn('aux_sf', sil_details_sf['stick_silhouette_coverage'])
    self.assertAlmostEqual(
        sil_details_sf['silhouette_recall'],
        sil_details_base['silhouette_recall'],
        places=4,
    )
    self.assertAlmostEqual(
        sil_scores_sf['silhouette'], sil_scores_base['silhouette'], places=4
    )

    sf_paper_mesh = fold_3d.FoldedModel(
        vertices_2d=np.zeros((8, 2)),
        vertices_3d=np.array([
            [0.0, -0.5, 0.0],
            [2.0, -0.5, 0.0],
            [2.0, 0.5, 0.0],
            [0.0, 0.5, 0.0],
            [0.0, -0.2, 0.0],
            [1.0, -0.2, 0.5],
            [1.0, 0.2, 0.5],
            [0.0, 0.2, 0.0],
        ]),
        faces=(np.array([0, 1, 2, 3]), np.array([4, 5, 6, 7])),
        edges=np.zeros((0, 2), dtype=int),
        assignments=(),
        fold_angles=np.zeros(0),
        face_matrices=np.broadcast_to(np.eye(4), (2, 4, 4)).copy(),
    )
    _, sf_paper_details, _ = fidelity.components(
        geom_with_sf, layout=layout, model=sf_paper_mesh
    )
    self.assertAlmostEqual(
        sf_paper_details['silhouette_precision'], 1.0, places=4
    )

    # Mirrored `stick_directions` re-fit also excludes `shaping_flap` joints.
    cat_res = bp.fold('cat', preset='demo', shape=True)
    assert cat_res.packed.design is not None
    assert cat_res.shaped is not None
    cat_tree = cat_res.packed.design.source
    cat_layout = paper_layout.build_paper_layout(cat_tree, cat_res.packed)
    cat_model = fold_3d.fold_3d(cat_res.shaped)
    mirror_model = dataclasses.replace(
        cat_model,
        vertices_3d=cat_model.vertices_3d * np.array([1.0, 1.0, -1.0]),
    )
    rep_base, dirs_base = fidelity.stick_directions(
        cat_tree, cat_res.packed, mirror_model, layout=cat_layout
    )
    self.assertTrue(rep_base.mirrored)
    cat_dict = cat_tree.to_dict()
    cat_root = cat_tree[cat_tree.root]
    cat_dict['children'].append({
        'name': 'aux_mirror_sf',
        'start': cat_root.start,
        'end': 'aux_mirror_sf_tip',
        'length': 6,
        'width': 1,
        'azimuth_angle': 90.0,
        'elevation_angle': 85.0,
        'shaping_flap': True,
    })
    cat_with_sf = bp.StickFigure.from_dict(cat_dict)
    cat_sf_joints = dict(cat_layout.joints)
    cat_sf_joints['aux_mirror_sf_tip'] = np.array([0.0, 0.0])
    cat_sf_layout = dataclasses.replace(cat_layout, joints=cat_sf_joints)
    rep_sf, dirs_sf = fidelity.stick_directions(
        cat_with_sf, cat_res.packed, mirror_model, layout=cat_sf_layout
    )
    self.assertTrue(rep_sf.mirrored)
    np.testing.assert_allclose(
        dirs_sf[cat_tree.root]['realized'],
        dirs_base[cat_tree.root]['realized'],
        atol=1e-4,
    )

    # Also verify direction, articulation, tip-clumping, and mirrored-pair
    # separation/symmetry skip `shaping_flap` sticks even when fewer than 3
    # non-shaping joints exist (`located = located_all` includes `sf_tip*`).
    minimal_sections: dict[str, dict[str, Any]] = {
        'body@0.5': {'folded': [1.0, 0.0, 0.0], 'width': 1.0},
        'aux_sf@0.5': {'folded': [0.0, 0.0, 1.0], 'width': 1.0},
    }
    for side in ('aux_sf_l', 'aux_sf_r'):
      for frac in paper_layout.SECTION_FRACTIONS:
        minimal_sections[f'{side}@{frac:g}'] = {
            'folded': [2.0 * frac, 0.0, 0.0],
            'width': 1.0,
        }
    minimal_sf_geom = {
        'root': 'body',
        'sticks': {
            'body': {
                'length': 2.0,
                'width': 1.0,
                'proximal': 'hip',
                'distal': 'neck',
                'parent': None,
                'is_river': False,
                'shaping_flap': False,
                'symmetric': None,
            },
            'aux_sf': {
                'length': 2.0,
                'width': 1.0,
                'proximal': 'hip',
                'distal': 'sf_tip',
                'parent': 'body',
                'is_river': False,
                'shaping_flap': True,
                'symmetric': None,
            },
            'aux_sf_l': {
                'length': 2.0,
                'width': 1.0,
                'proximal': 'hip',
                'distal': 'sf_tip_l',
                'parent': 'body',
                'is_river': False,
                'shaping_flap': True,
                'symmetric': 'aux_sf_r',
            },
            'aux_sf_r': {
                'length': 2.0,
                'width': 1.0,
                'proximal': 'hip',
                'distal': 'sf_tip_r',
                'parent': 'body',
                'is_river': False,
                'shaping_flap': True,
                'symmetric': 'aux_sf_l',
            },
        },
        'chains': {
            'body': ['body'],
            'aux_sf': ['aux_sf'],
            'aux_sf_l': ['aux_sf_l'],
            'aux_sf_r': ['aux_sf_r'],
        },
        'joints': {
            'hip': {'folded': [0.0, 0.0, 0.0], 'target': [0.0, 0.0, 0.0]},
            'neck': {'folded': [2.0, 0.0, 0.0], 'target': [2.0, 0.0, 0.0]},
            'sf_tip': {'folded': [0.0, 0.0, 2.0], 'target': [0.0, 0.0, 2.0]},
            'sf_tip_l': {'folded': [2.0, 0.0, 0.0], 'target': [2.0, 2.5, 0.0]},
            'sf_tip_r': {'folded': [2.0, 0.0, 0.0], 'target': [2.0, -2.5, 0.0]},
        },
        'sections': minimal_sections,
        'model_extents': [2.0, 2.0, 2.0],
        'body_extents': [2.0, 1.0, 0.2],
    }
    min_scores, min_details, _ = fidelity.components(minimal_sf_geom)
    self.assertEmpty(min_details['stray_flaps'])
    self.assertEmpty(min_details['clumped_tips'])
    self.assertEmpty(min_details['stacked_pairs'])
    self.assertNotIn('aux_sf', min_details['direction_error_deg'])
    self.assertNotIn('aux_sf_l', min_details['direction_error_deg'])
    for comp in ('direction', 'articulation', 'separation', 'symmetry'):
      self.assertAlmostEqual(min_scores[comp], 1.0, places=6)

    # Control: when `shaping_flap` is False, the 90-degree error IS flagged.
    geom_with_sf['sticks']['aux_sf']['shaping_flap'] = False
    scores_regular, details_regular, _ = fidelity.components(geom_with_sf)
    self.assertIn('aux_sf', details_regular['stray_flaps'])
    self.assertLess(scores_regular['direction'], scores_base['direction'])
    self.assertLess(scores_regular['width'], scores_base['width'])


if __name__ == '__main__':
  absltest.main()
