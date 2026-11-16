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

"""Tests for the objective fidelity score."""

import functools
import json
import math
import pathlib

from absl.testing import absltest
from absl.testing import parameterized
import bp_pipeline as bp
from bp_pipeline.shaping import bus_adapter
from bp_pipeline.shaping import fidelity
import numpy as np

_EXAMPLES_DIR = (
    pathlib.Path(__file__).parents[2]
    / '.agents'
    / 'skills'
    / 'stick-figure-design'
    / 'examples'
)


@functools.lru_cache(maxsize=4)
def _load_solved(filename: str) -> bp.SolvedCreasePattern:
  raw = json.loads((_EXAMPLES_DIR / filename).read_text(encoding='utf-8'))
  design = bp.prepare(raw)
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


def _load(name: str):
  mapped = {
      'spider_simple': 'cat.json',
      'scorpion_p3': 'human.json',
      'human_simple': 'human.json',
  }.get(name, f'{name}.json')
  solved = _load_solved(mapped)
  packing, _ = bus_adapter.legacy_inputs(solved)
  assert packing.unmerged_stick_figure is not None
  return packing.unmerged_stick_figure, packing, solved.creases


def _rotation(axis, angle: float) -> np.ndarray:
  axis = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
  k = np.array([
      [0.0, -axis[2], axis[1]],
      [axis[2], 0.0, -axis[0]],
      [-axis[1], axis[0], 0.0],
  ])
  return np.eye(3) + math.sin(angle) * k + (1 - math.cos(angle)) * k @ k


def _synthetic_geometry(fold_transform, overrides=None) -> dict:
  """A two-legged figure; folded joints = fold_transform(target joints).

  Args:
    fold_transform: Maps a target joint position to its folded position.
    overrides: Folded positions for some joints (after the transform).

  Returns:
    `fidelity.measure`-style geometry; sections interpolate folded joints.
  """
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
      name: np.asarray(fold_transform(np.asarray(position)), dtype=float)
      for name, position in targets.items()
  }
  folded.update(overrides or {})
  joints = {
      name: {'folded': list(folded[name]), 'target': position}
      for name, position in targets.items()
  }
  sections = {}
  for name, (near, far, *_rest) in sticks.items():
    for fraction in fidelity.SECTION_FRACTIONS:
      point = (1 - fraction) * folded[near] + fraction * folded[far]
      sections[f'{name}@{fraction:g}'] = {
          'folded': list(point),
          'width': sticks[name][4],
      }
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
              'symmetric': symmetric,
          }
          for name, (
              near,
              far,
              parent,
              length,
              width,
              symmetric,
          ) in sticks.items()
      },
      'chains': {
          'head': ['head'],
          'leg_l': ['thigh_l', 'shin_l'],
          'leg_r': ['thigh_r', 'shin_r'],
      },
      'joints': joints,
      'sections': sections,
      'model_extents': [1.0, 0.8, 0.6],
      'body_extents': [1.0, 0.5, 0.0],
  }


class SimilarityTest(parameterized.TestCase):

  @parameterized.parameters(False, True)
  def test_recovers_similarity(self, mirrored):
    rng = np.random.default_rng(0)
    source = rng.normal(size=(12, 3))
    rotation = _rotation([1.0, 2.0, 0.5], 0.7)
    if mirrored:
      rotation = rotation @ np.diag([1.0, -1.0, 1.0])
    target = 2.5 * source @ rotation.T + np.array([1.0, -2.0, 0.5])
    fit = fidelity.fit_similarity(source, target)
    np.testing.assert_allclose(fit.apply(source), target, atol=1e-8)
    self.assertAlmostEqual(fit.scale, 2.5)
    self.assertEqual(fit.reflected, mirrored)

  def test_proper_fit_never_reflects(self):
    rng = np.random.default_rng(1)
    source = rng.normal(size=(10, 3))
    target = source * np.array([1.0, -1.0, 1.0])
    fit = fidelity.fit_similarity(source, target, allow_reflection=False)
    self.assertFalse(fit.reflected)
    self.assertAlmostEqual(np.linalg.det(fit.rotation), 1.0)

  def test_robust_fit_ignores_an_outlier(self):
    rng = np.random.default_rng(2)
    source = rng.normal(size=(15, 3))
    rotation = _rotation([0.0, 0.0, 1.0], 1.1)
    target = source @ rotation.T
    target[0] += np.array([5.0, 5.0, 5.0])
    fit, weights = fidelity.fit_similarity_robust(source, target, size=1.0)
    np.testing.assert_allclose(fit.apply(source[1:]), target[1:], atol=1e-2)
    self.assertLess(weights[0], 0.01)


class ComponentsTest(absltest.TestCase):

  def test_rigidly_moved_figure_scores_perfectly(self):
    rotation = _rotation([0.3, -1.0, 0.2], 2.0)
    geometry = _synthetic_geometry(lambda p: 0.1 * rotation @ p + 7.0)
    result = fidelity.components(geometry)
    for name in (
        'pose',
        'direction',
        'separation',
        'width',
        'depth',
        'symmetry',
    ):
      self.assertAlmostEqual(
          result['components'][name], 1.0, places=3, msg=name
      )
    # The knees bend like the figure's.
    self.assertEqual(
        result['details']['limb_articulation'], {'leg_l': 1.0, 'leg_r': 1.0}
    )
    self.assertAlmostEqual(
        np.linalg.det(
            fidelity.render_rotation({'alignment': result['alignment']})
        ),
        1.0,
        places=4,
    )
    self.assertEqual(result['alignment']['render_frame'], 'stick_figure')

  def test_mirror_image_scores_perfectly(self):
    geometry = _synthetic_geometry(lambda p: p * np.array([1.0, 1.0, -1.0]))
    result = fidelity.components(geometry)
    self.assertTrue(result['details']['mirrored'])
    self.assertAlmostEqual(result['components']['pose'], 1.0)

  def test_stacked_legs_and_flat_model_are_penalized(self):
    def collapse(point):
      # Left and right legs fold onto each other in the body plane.
      return np.array([point[0], 0.0, 0.1 * abs(point[1]) + point[2]])

    good = fidelity.components(_synthetic_geometry(lambda p: p))
    stacked = fidelity.components(_synthetic_geometry(collapse))
    self.assertIn('thigh_l/thigh_r', stacked['details']['stacked_pairs'])
    self.assertLess(stacked['components']['separation'], 0.5)
    self.assertLess(stacked['components']['depth'], 0.5)
    self.assertLess(
        fidelity.combine(stacked['components']),
        fidelity.combine(good['components']) - 3.0,
    )

  def test_wide_flaps_lower_the_width_component(self):
    geometry = _synthetic_geometry(lambda p: p)
    for name, section in geometry['sections'].items():
      if name.startswith(('thigh', 'shin')):
        section['width'] = 1.0  # Figure says 0.5: unnarrowed planks.
    result = fidelity.components(geometry)
    self.assertLess(result['components']['width'], 0.8)

  def test_planar_figure_folded_flat_has_full_depth(self):
    # A stick figure with every elevation 0 (gallery Dragonfly) folded flat:
    # there is no 3-D extent to reproduce, so depth must not be penalized.
    flatten = np.array([1.0, 1.0, 0.0])
    geometry = _synthetic_geometry(lambda p: p * flatten)
    for joint in geometry['joints'].values():
      joint['target'] = list(np.asarray(joint['target']) * flatten)
    result = fidelity.components(geometry)
    self.assertEqual(result['details']['target_depth_ratio'], 0.0)
    self.assertAlmostEqual(result['components']['depth'], 1.0)
    # A figure with height still requires it.
    flat_fold = fidelity.components(_synthetic_geometry(lambda p: p * flatten))
    self.assertLess(flat_fold['components']['depth'], 0.5)

  def test_straight_limbs_get_no_articulation(self):
    # Knees on the straight line from hip to foot.
    geometry = _synthetic_geometry(
        lambda p: p,
        overrides={
            'knee_l': np.array([0.25, 0.75, -0.75]),
            'knee_r': np.array([0.25, -0.75, -0.75]),
        },
    )
    result = fidelity.components(geometry)
    self.assertEqual(
        result['details']['limb_articulation'], {'leg_l': 0.0, 'leg_r': 0.0}
    )

  def test_collapsed_bar_renders_in_root_frame(self):
    # Every joint folded onto one line: the similarity rotation is ill-posed.
    geometry = _synthetic_geometry(
        lambda p: np.array([p[0] + 0.1 * p[1] + 0.1 * p[2], 0.0, 0.0])
    )
    geometry['root_normal'] = [0.0, 1.0, 0.0]
    result = fidelity.components(geometry)
    alignment = result['alignment']
    self.assertEqual(alignment['render_frame'], 'root')
    self.assertIn('collinear', alignment['render_frame_reason'])
    rotation = fidelity.render_rotation({'alignment': alignment})
    self.assertAlmostEqual(np.linalg.det(rotation), 1.0, places=4)
    # The folded root axis (+x) stays the figure's root axis (+x) and the
    # paper normal at the root becomes vertical.
    np.testing.assert_allclose(
        rotation @ [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], atol=1e-4
    )
    self.assertAlmostEqual(abs((rotation @ [0.0, 1.0, 0.0])[2]), 1.0, places=4)

  def test_asymmetric_flap_width_blowout_lowers_symmetry(self):
    geometry = _synthetic_geometry(lambda p: p)
    geometry['sections']['thigh_r@0.5']['width'] = 2.2
    result = fidelity.components(geometry)
    self.assertIn(
        'thigh_l/thigh_r', result['details']['asymmetric_width_pairs']
    )
    self.assertLess(result['components']['symmetry'], 0.85)

  def test_cleanliness_and_clean_gate_penalize_broken_kawasaki_and_strain(self):
    self.assertTrue(fidelity.clean_gate(0.05, 0)['passed'])
    self.assertFalse(fidelity.clean_gate(0.05, 2)['passed'])
    self.assertFalse(fidelity.clean_gate(0.89, 0)['passed'])
    clean = fidelity.components(_synthetic_geometry(lambda p: p))
    messy = fidelity.components(
        _synthetic_geometry(lambda p: p),
        vertex_error=0.89,
        non_ff_vertices=2,
    )
    self.assertAlmostEqual(clean['components']['cleanliness'], 1.0)
    self.assertLess(messy['components']['cleanliness'], 0.25)
    self.assertLess(
        fidelity.combine(messy['components']),
        fidelity.combine(clean['components']) - 1.2,
    )

  def test_silhouette_penalizes_thin_wire_on_wide_target_body(self):
    geometry = _synthetic_geometry(lambda p: p)
    geometry['sticks']['body']['width'] = 4.0
    geometry['sticks']['body']['is_river'] = True
    identity_fit = fidelity.Similarity(
        rotation=np.eye(3),
        scale=1.0,
        translation=np.zeros(3),
        reflected=False,
    )
    aligned = {
        name: np.asarray(val['folded'], dtype=float)
        for name, val in geometry['joints'].items()
    }
    target = {
        name: np.asarray(val['target'], dtype=float)
        for name, val in geometry['joints'].items()
    }
    # Build two meshes: a thin wire along x in [0, 3] vs a broad body plate.
    def _make_mesh(half_span: float) -> fidelity.FoldedModel:
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
      faces = (
          np.array([0, 1, 2, 3], dtype=int),
          np.array([4, 5, 6, 7], dtype=int),
      )
      return fidelity.FoldedModel(
          vertices_2d=np.zeros((8, 2)),
          vertices_3d=verts_3d,
          faces=faces,
          matrices=np.broadcast_to(np.eye(4), (2, 4, 4)).copy(),
          vertex_error=0.0,
          seconds=0.0,
      )

    wire_metrics = fidelity._silhouette_and_body_metrics(
        geometry, aligned, target, identity_fit, _make_mesh(0.08), 1.0
    )
    plate_metrics = fidelity._silhouette_and_body_metrics(
        geometry, aligned, target, identity_fit, _make_mesh(1.4), 1.0
    )
    self.assertLess(wire_metrics['wide_element_coverage'], 0.45)
    self.assertLess(wire_metrics['local_body_volume'], 0.35)
    self.assertGreater(plate_metrics['wide_element_coverage'], 0.80)
    self.assertGreater(plate_metrics['local_body_volume'], 0.80)


class CombineTest(absltest.TestCase):

  def test_geometric_mean_and_floor(self):
    ones = {name: 1.0 for name in fidelity.WEIGHTS}
    self.assertAlmostEqual(fidelity.combine(ones), 10.0)
    halves = {name: 0.5 for name in fidelity.WEIGHTS}
    self.assertAlmostEqual(fidelity.combine(halves), 5.0)
    zero_depth = dict(ones, depth=0.0)
    self.assertGreater(fidelity.combine(zero_depth), 0.0)
    self.assertLess(fidelity.combine(zero_depth), 6.0)


class FoldedBaseTest(parameterized.TestCase):

  @parameterized.parameters('spider_simple', 'scorpion_p3')
  def test_unshaped_base_folds_every_stick_to_its_length(self, name):
    tree, packing, base = _load(name)
    layout = fidelity.paper_layout(tree, packing)
    model = fidelity.FoldedModel.from_pattern(base)
    geometry = fidelity.measure(tree, layout, model)
    checked = 0
    for stick_name, stick in geometry['sticks'].items():
      near = geometry['joints'][stick['proximal']]['folded']
      far = geometry['joints'][stick['distal']]['folded']
      if near is None or far is None:
        continue
      length = np.linalg.norm(np.asarray(far) - np.asarray(near))
      self.assertAlmostEqual(
          length, stick['length'], delta=0.15, msg=stick_name
      )
      checked += 1
    self.assertGreater(checked, len(geometry['sticks']) // 2)

  def test_unshaped_base_scores_low_and_reports(self):
    tree, packing, base = _load('scorpion_p3')
    report = fidelity.score_pattern(tree, packing, base)
    # The uniaxial base is one flat strip: no depth, stacked pairs.
    self.assertLess(report['components']['depth'], 0.05)
    self.assertNotEmpty(report['stacked_pairs'])
    self.assertLess(report['fidelity'], 4.5)
    for key in (
        'components',
        'alignment',
        'seconds',
        'widths',
        'direction_error_deg',
        'joint_error',
    ):
      self.assertIn(key, report)
    self.assertEqual(
        np.asarray(report['alignment']['render_rotation']).shape, (3, 3)
    )

  def test_pose_gate(self):
    self.assertTrue(
        fidelity.pose_gate({'direction': 0.9}, {'stray_flaps': ['a']})['passed']
    )
    self.assertFalse(
        fidelity.pose_gate({'direction': 0.6}, {'stray_flaps': []})['passed']
    )
    gate = fidelity.pose_gate({'direction': 0.95}, {'stray_flaps': list('abc')})
    self.assertFalse(gate['passed'])
    self.assertEqual(gate['num_stray'], 3)
    crossing = {
        'stick': 'shin_l',
        'point': 'tip',
        'kind': 'midline',
        'design_side': 'left',
        'design_offset': 1.5,
        'realized_offset': 0.0,
    }
    gate = fidelity.pose_gate(
        {'direction': 0.95},
        {'stray_flaps': [], 'side_crossings': [crossing, dict(crossing)]},
    )
    self.assertFalse(gate['passed'])
    self.assertEqual(gate['crossing'], ['shin_l'])
    self.assertEqual(gate['num_crossing'], 1)

  def test_folded_model_cache_checks_identity(self):
    tree, packing, base = _load('human_simple')
    model = fidelity.FoldedModel.from_pattern(base)
    self.assertIs(fidelity.FoldedModel.from_pattern(base), model)
    key = (id(base), len(base.creases))
    fidelity._MODEL_PATTERN_CACHE[key] = (object(), 'stale')
    self.assertIsInstance(
        fidelity.FoldedModel.from_pattern(base), fidelity.FoldedModel
    )
    del tree, packing

  def test_score_cache_checks_identity_not_just_id(self):
    # A freed pattern's id can be reused by a new pattern with the same crease
    # count; the cache must not serve the old pattern's score then.
    tree, packing, base = _load('human_simple')
    report = fidelity.score_pattern(tree, packing, base)
    self.assertIs(fidelity.score_pattern(tree, packing, base), report)
    key = (id(tree), id(packing), id(base), len(base.creases))
    stale = {'fidelity': -1.0}
    fidelity._PATTERN_SCORE_CACHE[key] = ((tree, packing, object()), stale)
    fresh = fidelity.score_pattern(tree, packing, base)
    self.assertIsNot(fresh, stale)
    self.assertEqual(fresh['fidelity'], report['fidelity'])


class SideCrossingsTest(absltest.TestCase):

  def test_rigid_and_mirrored_models_keep_their_sides(self):
    rotation = _rotation([0.3, -1.0, 0.2], 2.0)
    for transform in (
        lambda p: 0.1 * rotation @ p + 7.0,
        lambda p: p * np.array([1.0, 1.0, -1.0]),
    ):
      result = fidelity.components(_synthetic_geometry(transform))
      self.assertEmpty(result['details']['side_crossings'])

  def test_crossed_and_midline_feet_fail_the_gate(self):
    # The left foot is folded across to the right; the right foot ends on
    # the mid-plane (where the two feet would cross each other).
    geometry = _synthetic_geometry(
        lambda p: p,
        overrides={
            'foot_l': np.array([0.5, -1.0, -1.5]),
            'foot_r': np.array([0.5, 0.0, -1.5]),
        },
    )
    result = fidelity.components(geometry)
    crossings = result['details']['side_crossings']
    by_point = {(c['stick'], c['point']): c for c in crossings}
    self.assertEqual(by_point['shin_l', 'tip']['kind'], 'crossed')
    self.assertEqual(by_point['shin_l', 'tip']['design_side'], 'left')
    self.assertEqual(by_point['shin_r', 'tip']['kind'], 'midline')
    # The worst crossing comes first.
    self.assertEqual(crossings[0]['stick'], 'shin_l')
    # Thighs stay on their sides; the head and body are on the axis.
    self.assertNotIn('thigh_l', fidelity.crossing_sticks(crossings))
    self.assertNotIn('head', fidelity.crossing_sticks(crossings))
    gate = fidelity.pose_gate(result['components'], result['details'])
    self.assertFalse(gate['passed'])
    self.assertCountEqual(gate['crossing'], ['shin_l', 'shin_r'])

  def test_points_near_the_axis_by_design_are_not_checked(self):
    sticks = {
        'body': {'proximal': 'a', 'distal': 'b'},
        'whisker': {'proximal': 'b', 'distal': 'c'},
        'arm': {'proximal': 'b', 'distal': 'd'},
    }
    design = {
        'a': [0.0, 0.0, 0.0],
        'b': [1.0, 0.0, 0.0],
        'c': [1.5, 0.3, 0.0],  # Less than SIDE_MIN_DESIGN_OFFSET.
        'd': [1.0, 2.0, 0.0],
    }
    realized = dict(design, c=[1.5, -0.3, 0.0], d=[1.0, -2.0, 0.0])
    crossings = fidelity.side_crossings(sticks, 'body', design, realized)
    self.assertEqual(fidelity.crossing_sticks(crossings), ['arm'])
    # Offsets are measured from the root stick, wherever the model sits.
    shifted = {k: np.asarray(v) + 5.0 for k, v in design.items()}
    self.assertEmpty(fidelity.side_crossings(sticks, 'body', design, shifted))
    # Unlocated points are skipped.
    self.assertEmpty(
        fidelity.side_crossings(sticks, 'body', design, dict(design, d=None))
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
    # horn_l starts on the midline (b at y=0) and splays to y=1.4: a mid-section
    # near the central hub (y=-0.06) is not a crossing, whereas shin_l (both
    # joints off-axis) dropping to y=0.0 at its mid-section IS a midline hit.
    crossings = fidelity.side_crossings(
        sticks,
        'body',
        design,
        realized,
        realized_sections={
            'horn_l@0.5': [1.25, -0.06, 0.0],
            'shin_l@0.5': [1.5, 0.0, -0.5],
        },
    )
    self.assertEqual(fidelity.crossing_sticks(crossings), ['shin_l'])
    # If horn_l's mid-section crosses past -margin, it is still caught.
    crossed = fidelity.side_crossings(
        sticks,
        'body',
        design,
        realized,
        realized_sections={'horn_l@0.5': [1.25, -0.4, 0.0]},
    )
    self.assertEqual(fidelity.crossing_sticks(crossed), ['horn_l'])
    self.assertEqual(crossed[0]['kind'], 'crossed')

  def test_proximal_to_distal_hip_crossing_is_caught(self):
    # A pair of legs whose ankles end on their target sides (y = +1.1, -1.1)
    # but whose proximal hip attachments on the folded base sit on the opposite
    # sides (y = -0.46, +0.61) physically cross each other under the body.
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
    crossings = fidelity.side_crossings(
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
        fidelity.crossing_sticks(crossings), ['leg_L', 'leg_R']
    )
    by_point = {(c['stick'], c['point']): c for c in crossings}
    self.assertEqual(by_point['leg_L', 'proximal']['kind'], 'crossed')
    self.assertEqual(by_point['leg_R', 'proximal']['kind'], 'crossed')



class PublicEvaluateFidelityTest(absltest.TestCase):

  def test_evaluate_fidelity_on_solved_shaped_and_pipeline_result(self):
    res = bp.fold('cat', preset='demo', shape=True)
    solved_rep = bp.evaluate_fidelity(res.solved)
    self.assertIsNotNone(res.shaped)
    shaped_rep = bp.evaluate_fidelity(res.shaped)
    pipe_rep = bp.evaluate_fidelity(res)
    self.assertEqual(shaped_rep['fidelity'], pipe_rep['fidelity'])
    self.assertGreater(shaped_rep['fidelity'], solved_rep['fidelity'])
    self.assertTrue(shaped_rep['clean_gate']['locally_flat_foldable'])
    self.assertLess(shaped_rep['fold_vertex_error_percent'], 0.05)


if __name__ == '__main__':
  absltest.main()
