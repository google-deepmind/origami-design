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

"""Tests for the showcase models and the studio page under `docs/`."""

import json
import os
import pathlib
import re
import shutil
import subprocess
import tempfile

from absl.testing import absltest
from absl.testing import parameterized
from bp_pipeline.stick_figures import stick_figure as sf_lib
import fold_checks
import numpy as np

_DOCS_DIR = pathlib.Path(__file__).parents[1] / 'docs'
_MODELS_DIR = _DOCS_DIR / 'models'
_CATALOG = json.loads(
    (_MODELS_DIR / 'catalog.json').read_text(encoding='utf-8')
)

_PNG_SIGNATURE = b'\x89PNG\r\n\x1a\n'

_MODELS_WITH_BASE_FOLD_ANGLE = frozenset({
    'bounding-red-kangaroo',
    'boxer-crab',
    'bull-moose',
    'castor-and-pollux',
    'crested-hoopoe',
    'emperor-scorpion',
    'golden-eagle',
    'indian-peacock',
    'king-cobra',
    'spinosaurus',
    'triceratops',
    'winged-pegasus',
    'wyrmlord-dragon',
})

_MODELS_WITH_UNMODIFIED_BASE_TOPOLOGY = frozenset({
    'bounding-red-kangaroo',
    'boxer-crab',
    'bull-moose',
    'castor-and-pollux',
    'crested-hoopoe',
    'emperor-scorpion',
    'indian-peacock',
    'spinosaurus',
})


def _each_model():
  """Yields one named test case per catalog entry."""
  for entry in _CATALOG:
    yield entry['slug'].replace('-', '_'), entry


def _read_json(slug: str, name: str):
  return json.loads((_MODELS_DIR / slug / name).read_text(encoding='utf-8'))


class ShowcaseTest(parameterized.TestCase):

  def test_studio_page_refers_to_files_that_exist(self):
    page = (_DOCS_DIR / 'index.html').read_text(encoding='utf-8')
    # Relative URLs are those without a scheme.
    urls = re.findall(r'(?:src|href)="([^":]+)"', page)
    self.assertContainsSubset({'style.css', 'js/main.js'}, urls)
    for url in urls:
      self.assertTrue((_DOCS_DIR / url).exists(), url)

    self.assertTrue((_DOCS_DIR / 'js' / 'video.js').is_file())
    self.assertIn('id="record-video"', page)
    self.assertIn('id="video-preset"', page)
    self.assertIn('<input id="creases" type="checkbox">', page)

    for script in (_DOCS_DIR / 'js').glob('*.js'):
      source = script.read_text(encoding='utf-8')
      for module in re.findall(r"from '(\./[^']+)'", source):
        self.assertTrue((script.parent / module).is_file(), module)

  def test_catalog_lists_the_model_directories(self):
    self.assertCountEqual(
        [entry['slug'] for entry in _CATALOG],
        [path.name for path in _MODELS_DIR.iterdir() if path.is_dir()],
    )

  @parameterized.named_parameters(_each_model())
  def test_catalog_entry_has_what_the_studio_reads(self, entry):
    for key in ('title', 'subtitle', 'summary'):
      self.assertNotEmpty(entry[key], key)
    self.assertNotEmpty(entry['techniques'])
    for key in ('colorTop', 'colorBottom'):
      self.assertRegex(entry[key], r'^#[0-9a-fA-F]{6}$')
    self.assertBetween(entry['camera']['azimuth'], -180, 180)
    self.assertBetween(entry['camera']['elevation'], -90, 90)

    preview = (_MODELS_DIR / entry['slug'] / 'preview.png').read_bytes()
    self.assertStartsWith(preview, _PNG_SIGNATURE)

    if (_MODELS_DIR / entry['slug'] / 'shaping_history.json').is_file():
      self.assertIn('pipeline_args', entry)
      self.assertIsInstance(entry['pipeline_args'], list)
      for arg in entry['pipeline_args']:
        self.assertIsInstance(arg, str)

  @parameterized.named_parameters(_each_model())
  def test_fold_file_folds_its_crease_pattern(self, entry):
    fold = _read_json(entry['slug'], 'shaped_3d.fold')

    flat, _ = fold_checks.frames(fold)
    areas = fold_checks.face_areas(fold)
    angle_errors = fold_checks.fold_angle_errors(fold)
    partly_folded = fold_checks.partly_folded_creases(fold)

    np.testing.assert_allclose(flat.min(axis=0), [0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(flat.max(axis=0), [1.0, 1.0], atol=1e-6)
    self.assertGreater(areas.min(), 0.0)
    self.assertAlmostEqual(areas.sum(), 1.0, places=4)
    # Shaping leaves some of these models slightly strained, so their folded
    # frame is not quite a rigid folding of the crease pattern. A mirrored
    # frame would get nearly all of the partly folded creases wrong.
    self.assertLess(fold_checks.mean_strain(fold), 0.05)
    self.assertGreater(np.mean(angle_errors[partly_folded] < 5.0), 0.9)

  @parameterized.named_parameters(_each_model())
  def test_fold_file_refers_to_its_stick_figure(self, entry):
    fold = _read_json(entry['slug'], 'shaped_3d.fold')
    stick_figure = sf_lib.StickFigure.from_dict(
        _read_json(entry['slug'], 'stick_figure.json')
    )
    regions = fold['file_bp:regions']
    faces_region = fold['faces_bp:region']

    self.assertGreater(fold['file_bp:gridSize'], 0)
    region_sticks = set()
    for region in regions:
      region_sticks.update(region['sticks'])
    self.assertContainsSubset(
        region_sticks,
        {stick.name for stick in stick_figure.children},
    )
    self.assertContainsSubset(
        {region['kind'] for region in regions}, {'flap', 'river'}
    )
    self.assertLen(faces_region, len(fold['faces_vertices']))
    self.assertContainsSubset(set(faces_region) - {None}, range(len(regions)))
    self.assertLen(fold['edges_bp:shaping'], len(fold['edges_vertices']))
    if entry['slug'] in _MODELS_WITH_BASE_FOLD_ANGLE:
      self.assertIn('edges_bp:baseFoldAngle', fold)
    if 'edges_bp:baseFoldAngle' not in fold:
      return
    self.assertLen(fold['edges_bp:baseFoldAngle'], len(fold['edges_vertices']))

    base_angles = np.asarray(fold['edges_bp:baseFoldAngle'], dtype=float)
    shaping = np.asarray(fold['edges_bp:shaping'], dtype=bool)
    assignments = np.asarray(fold['edges_assignment'])
    mv = np.isin(assignments, ['M', 'V'])
    self.assertTrue(np.isin(base_angles, [-180.0, 0.0, 180.0]).all())
    np.testing.assert_array_equal(base_angles[shaping], 0.0)
    np.testing.assert_array_equal(shaping[mv], base_angles[mv] == 0.0)
    np.testing.assert_array_equal(base_angles[assignments == 'B'], 0.0)

    # Every interior base vertex outside shaped squash/fan edits must satisfy
    # Maekawa's theorem (|M - V| == 2) on `edges_bp:baseFoldAngle` (and across
    # the entire sheet when shaping did not delete base segments).
    flat, _ = fold_checks.frames(fold)
    num_verts = len(flat)
    border_verts = set()
    touches_shaping = np.zeros(num_verts, dtype=bool)
    m_counts = np.zeros(num_verts, dtype=int)
    v_counts = np.zeros(num_verts, dtype=int)
    for v_idx, (x, y) in enumerate(flat):
      if (
          x <= 1e-5
          or x >= 1.0 - 1e-5
          or y <= 1e-5
          or y >= 1.0 - 1e-5
      ):
        border_verts.add(v_idx)
    for e_idx, (u, v) in enumerate(fold['edges_vertices']):
      if assignments[e_idx] == 'B':
        border_verts.add(u)
        border_verts.add(v)
        continue
      if shaping[e_idx]:
        touches_shaping[u] = True
        touches_shaping[v] = True
      if base_angles[e_idx] == -180.0:
        m_counts[u] += 1
        m_counts[v] += 1
      elif base_angles[e_idx] == 180.0:
        v_counts[u] += 1
        v_counts[v] += 1

    base_degree = m_counts + v_counts
    violations = [
        v_idx
        for v_idx in range(num_verts)
        if v_idx not in border_verts
        and base_degree[v_idx] > 0
        and abs(int(m_counts[v_idx]) - int(v_counts[v_idx])) != 2
    ]
    unshaped_violations = [v for v in violations if not touches_shaping[v]]
    self.assertEmpty(unshaped_violations)
    if entry['slug'] in _MODELS_WITH_UNMODIFIED_BASE_TOPOLOGY:
      self.assertEmpty(violations)

  def test_video_timeline_and_live_schedule_under_node(self):
    node = os.environ.get('NODE_BIN') or shutil.which('node')
    if node and not os.path.isabs(node) and not os.path.exists(node):
      node = os.path.join(
          os.environ.get('TEST_SRCDIR', ''),
          os.environ.get('TEST_WORKSPACE', ''),
          node,
      )
    if os.environ.get('NODE_BIN'):
      self.assertTrue(os.path.exists(node), node)
    if not node or not (os.path.exists(node) or shutil.which(node)):
      self.skipTest('node runtime not available')

    eagle_fold = (_MODELS_DIR / 'golden-eagle' / 'shaped_3d.fold').resolve()
    eagle_sf = (_MODELS_DIR / 'golden-eagle' / 'stick_figure.json').resolve()
    with tempfile.TemporaryDirectory() as tmpdir:
      tmp = pathlib.Path(tmpdir)
      (tmp / 'package.json').write_text(
          '{"type": "module"}\n', encoding='utf-8'
      )
      for js_file in (_DOCS_DIR / 'js').glob('*.js'):
        shutil.copy2(js_file, tmp / js_file.name)
      script_path = tmp / 'check.mjs'
      script_path.write_text(
          f"""
      import fs from 'node:fs';
      import assert from 'node:assert/strict';
      import {{
        DEFAULT_BASE_COMPLETE_T,
        DEFAULT_SHAPING_START_T,
        DEFAULT_STANDUP_START_T,
        buildFoldingSchedule,
      }} from './schedule.js';
      import {{evaluateVideoTimeline}} from './video.js';
      import {{Folder, parseFold}} from './fold.js';

      for (const staged of [true, false]) {{
        const cfg = {{
          staged,
          baseCompleteT: DEFAULT_BASE_COMPLETE_T,
          shapingStartT: DEFAULT_SHAPING_START_T,
          standupStartT: DEFAULT_STANDUP_START_T,
          heroAzimuth: 36,
          heroElevation: 20,
          initialFold: 1,
        }};
        for (const preset of ['guided', 'hero', 'fold']) {{
          const total = evaluateVideoTimeline(preset, 0, cfg).totalSec;
          let prev = -1;
          let sawBaseDwell = false;
          for (let i = 0; i <= 2000; i++) {{
            const t = (total * i) / 2000;
            const step = evaluateVideoTimeline(preset, t, cfg);
            assert(
                step.fold >= prev - 1e-12,
                `${{preset}} snapped backwards at t=${{t}}`);
            if (Math.abs(step.fold - DEFAULT_BASE_COMPLETE_T) < 1e-9) {{
              sawBaseDwell = true;
            }}
            prev = step.fold;
          }}
          if (staged) assert(sawBaseDwell, `${{preset}} missed base dwell`);
        }}
      }}

      const eagle = parseFold(
          JSON.parse(fs.readFileSync({json.dumps(str(eagle_fold))}, 'utf8')));
      const sf = JSON.parse(
          fs.readFileSync({json.dumps(str(eagle_sf))}, 'utf8'));
      const sched = buildFoldingSchedule(eagle, sf);
      const folder = new Folder(eagle);
      folder.setSchedule(sched);
      let peakStrain = 0;
      let transitions = 0;
      let lastKey = null;
      for (let i = 0; i <= 480; i++) {{
        folder.positionsAt(i / 480, true);
        peakStrain = Math.max(peakStrain, folder.lastState.strainPct);
        if (folder.lastState.diagramKey !== lastKey) {{
          transitions++;
          lastKey = folder.lastState.diagramKey;
        }}
      }}
      assert(
          peakStrain < 105,
          `golden-eagle live peak strain=${{peakStrain}}`);
      assert(
          transitions <= 2 * sched.phases.length + 4,
          `excessive diagramKey transitions=${{transitions}}`);
      """,
          encoding='utf-8',
      )
      res = subprocess.run(
          [node, str(script_path)],
          capture_output=True,
          text=True,
          check=False,
      )
      self.assertEqual(res.returncode, 0, msg=res.stderr or res.stdout)

  @parameterized.named_parameters(_each_model())
  def test_recorded_history_schema(self, entry):
    slug = entry['slug']
    history_path = _MODELS_DIR / slug / 'shaping_history.json'
    if not history_path.is_file():
      return

    history = json.loads(history_path.read_text(encoding='utf-8'))
    self.assertIsInstance(history, list)
    self.assertNotEmpty(history)
    for call in history:
      self.assertIsInstance(call.get('tool'), str)
      self.assertIsInstance(call.get('arguments'), dict)


if __name__ == '__main__':
  absltest.main()
