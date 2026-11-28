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
    'southern-white-rhino',
    'spinosaurus',
    'triceratops',
    'winged-pegasus',
    'wyrmlord-dragon',
})

_MODELS_WITH_UNMODIFIED_BASE_TOPOLOGY = frozenset({
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
        {region['kind'] for region in regions}, {'flap', 'river', 'reserved'}
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

  def _run_node_script(self, script_name: str, script_body: str) -> None:
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

    tmp = pathlib.Path(self.create_tempdir().full_path)
    (tmp / 'package.json').write_text('{"type": "module"}\n', encoding='utf-8')
    for js_file in (_DOCS_DIR / 'js').glob('*.js'):
      shutil.copy2(js_file, tmp / js_file.name)
    script_path = tmp / script_name
    script_path.write_text(script_body, encoding='utf-8')
    completed = subprocess.run(
        [node, str(script_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    self.assertEqual(
        completed.returncode, 0, msg=completed.stderr or completed.stdout
    )

  def test_video_timeline_and_live_schedule_under_node(self):
    eagle_fold = (_MODELS_DIR / 'golden-eagle' / 'shaped_3d.fold').resolve()
    eagle_sf = (_MODELS_DIR / 'golden-eagle' / 'stick_figure.json').resolve()
    self._run_node_script(
        'check.mjs',
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
      const tFast0 = performance.now();
      const fastSched = buildFoldingSchedule(eagle, sf);
      const fastMs = performance.now() - tFast0;
      assert.equal(fastSched.act1Keyframes, null);
      assert(fastMs < 150, `non-XPBD schedule took ${{fastMs}} ms`);

      const sched = buildFoldingSchedule(eagle, sf, {{solveXpbd: true}});
      assert(Array.isArray(sched.act1Keyframes) && sched.act1Keyframes.length >= 15);
      const folder = new Folder(eagle);
      folder.setSchedule(sched);
      let act1PeakStrain = 0;
      let peakStrain = 0;
      let transitions = 0;
      let lastKey = null;
      for (let i = 0; i <= 480; i++) {{
        const p = i / 480;
        folder.positionsAt(p, true);
        if (p <= sched.baseCompleteT) {{
          act1PeakStrain = Math.max(
              act1PeakStrain, folder.lastState.strainPct);
        }}
        peakStrain = Math.max(peakStrain, folder.lastState.strainPct);
        if (folder.lastState.diagramKey !== lastKey) {{
          transitions++;
          lastKey = folder.lastState.diagramKey;
        }}
      }}
      assert(
          act1PeakStrain < 8,
          `golden-eagle live Act I peak strain=${{act1PeakStrain}}`);
      assert(
          peakStrain < 20,
          `golden-eagle live peak strain=${{peakStrain}}`);
      assert(
          transitions <= 2 * sched.phases.length + 4,
          `excessive diagramKey transitions=${{transitions}}`);
      """,
    )

  def test_bundled_schedules_conserve_area_under_node(self):
    models_dir_str = str(_MODELS_DIR.resolve())
    self._run_node_script(
        'verify_schedules.mjs',
        f"""
      import fs from 'node:fs';
      import path from 'node:path';
      import assert from 'node:assert/strict';
      import {{
        Folder,
        computeModelMeshHash,
        hydrateSchedule,
        parseFold,
      }} from './fold.js';
      import {{
        buildFoldingSchedule,
        unwarpScheduleProgress,
        warpScheduleProgress,
      }} from './schedule.js';
      import {{earClipFace}} from './xpbd.js';

      function meshSurfaceArea(tris, pos) {{
        let area = 0;
        for (let t = 0; t < tris.length; t++) {{
          const [a, b, c] = tris[t];
          const i0 = 3 * a;
          const i1 = 3 * b;
          const i2 = 3 * c;
          const ux = pos[i1] - pos[i0];
          const uy = pos[i1 + 1] - pos[i0 + 1];
          const uz = pos[i1 + 2] - pos[i0 + 2];
          const vx = pos[i2] - pos[i0];
          const vy = pos[i2 + 1] - pos[i0 + 1];
          const vz = pos[i2 + 2] - pos[i0 + 2];
          area += 0.5 * Math.hypot(
              uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx);
        }}
        return area;
      }}

      const modelsDir = {json.dumps(models_dir_str)};
      const catalog = JSON.parse(
          fs.readFileSync(path.join(modelsDir, 'catalog.json'), 'utf8'));
      for (const entry of catalog) {{
        const dir = path.join(modelsDir, entry.slug);
        const foldJson = JSON.parse(
            fs.readFileSync(path.join(dir, 'shaped_3d.fold'), 'utf8'));
        const model = parseFold(foldJson);
        const tris = model.faces.flatMap((fc) => earClipFace(fc, model.flat));
        const sf = JSON.parse(
            fs.readFileSync(path.join(dir, 'stick_figure.json'), 'utf8'));
        const schedPath = path.join(dir, 'schedule.json');
        const rawSched = fs.existsSync(schedPath) ?
            JSON.parse(fs.readFileSync(schedPath, 'utf8')) :
            null;
        const hydrated = hydrateSchedule(rawSched, model);
        if (rawSched) {{
          assert.notEqual(
              hydrated, null, `${{entry.slug}} schedule.json failed hydration`);
          assert.equal(typeof rawSched.act1DeltaQ, 'string');
          assert.equal(rawSched.meshHash, computeModelMeshHash(model));
          assert.equal(
              hydrateSchedule({{...rawSched, numVertices: null}}, model), null);
          assert.equal(
              hydrateSchedule(
                  {{...rawSched, rootFace: model.faces.length}}, model),
              null);
          assert.equal(
              hydrateSchedule({{...rawSched, shapingHopCosts: [1, 2]}}, model),
              null);
          const movedModel = {{...model, flat: Float64Array.from(model.flat)}};
          movedModel.flat[0] += 1e-3;
          assert.equal(hydrateSchedule(rawSched, movedModel), null);
        }}
        const sched =
            hydrated ?? buildFoldingSchedule(model, sf, {{solveXpbd: true}});
        const folder = new Folder(model);
        folder.setSchedule(sched);

        for (let k = 0; k <= 40; k++) {{
          const p = k / 40;
          const w = warpScheduleProgress(sched, p);
          const u = unwarpScheduleProgress(sched, w);
          assert(
              Math.abs(u - p) < 1e-9,
              `${{entry.slug}} warp roundtrip drift=${{Math.abs(u - p)}}`);
        }}

        const a0 = meshSurfaceArea(tris, folder.positionsAt(0, true));
        assert(Math.abs(a0 - 1.0) < 1e-3, `${{entry.slug}} flat area=${{a0}}`);

        const nSteps = 200;
        let maxAct1AreaDevPct = 0;
        let prevPos = Float32Array.from(folder.positionsAt(0, true));
        const maxVertSteps = new Float64Array(nSteps);
        for (let k = 1; k <= nSteps; k++) {{
          const p = k / nSteps;
          const pos = folder.positionsAt(p, true);
          const wt = folder.lastState.warpedT;
          if (wt <= sched.shapingStartT + 1e-9) {{
            const areaNow = meshSurfaceArea(tris, pos);
            const dev = 100 * Math.abs(areaNow - a0) / a0;
            if (dev > maxAct1AreaDevPct) maxAct1AreaDevPct = dev;
          }}
          let stepMax = 0;
          for (let i = 0; i < pos.length; i += 3) {{
            const d = Math.hypot(
                pos[i] - prevPos[i],
                pos[i + 1] - prevPos[i + 1],
                pos[i + 2] - prevPos[i + 2]);
            if (d > stepMax) stepMax = d;
          }}
          maxVertSteps[k - 1] = stepMax;
          prevPos.set(pos);
        }}
        const peakVertStep = Math.max(...maxVertSteps);
        assert(
            peakVertStep < 0.35,
            `${{entry.slug}} discontinuous frame jump=${{peakVertStep}}`);
        for (let k = 1; k < nSteps - 1; k++) {{
          const neighbor = Math.max(
              1e-3, 0.5 * (maxVertSteps[k - 1] + maxVertSteps[k + 1]));
          assert(
              maxVertSteps[k] / neighbor < 4.5,
              `${{entry.slug}} step spike ratio=${{maxVertSteps[k] / neighbor}} at k=${{k}}`);
        }}
        assert(
            maxAct1AreaDevPct < 12.0,
            `${{entry.slug}} Act I area deviation=${{maxAct1AreaDevPct}}%`);
        folder.positionsAt(
            unwarpScheduleProgress(sched, sched.baseCompleteT), true);
        const maxBaseStrain =
            (!hydrated || rawSched?.includesBaseFrame) ? 2.5 : 0.1;
        assert(
            folder.lastState.strainPct < maxBaseStrain,
            `${{entry.slug}} base strain=${{folder.lastState.strainPct}}%`);
      }}
      """,
    )

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
