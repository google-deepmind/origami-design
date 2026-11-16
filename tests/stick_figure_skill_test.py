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

"""Tests for the `bp-pipeline` CLI, visual API, and bundled design skills."""

import contextlib
import io
import json
import pathlib
import re
import tempfile
import textwrap

from absl.testing import absltest
from absl.testing import parameterized
import bp_pipeline as bp
from bp_pipeline import cli as cli_lib
from bp_pipeline.stick_figures import graph_checks
from bp_pipeline.stick_figures import visualize as sf_visualize
import matplotlib.pyplot as plt

_REPO_ROOT = pathlib.Path(__file__).parents[1]
_SKILLS_DIR = _REPO_ROOT / '.agents' / 'skills'
_EXAMPLES_DIR = _SKILLS_DIR / 'stick-figure-design' / 'examples'


class StickFigureSkillTest(parameterized.TestCase):

  def test_no_top_level_examples_submodule_collision(self):
    self.assertFalse(hasattr(bp, 'examples'))
    self.assertNotIn('examples', dir(bp))
    self.assertIsInstance(bp.list_examples(), list)
    self.assertGreaterEqual(len(bp.list_examples()), 47)

  @parameterized.parameters(
      ('stick-figure-design', 'stick-figure-design', False),
      ('stick-figure-generation', 'stick-figure-generation', True),
      ('origami-shaping', 'origami-shaping', True),
      ('origami-solving-pipeline', 'origami-solving-pipeline', True),
      ('origami-critique', 'origami-critique', True),
  )
  def test_llm_skills_frontmatter_and_python_snippets(
      self,
      folder_name: str,
      expected_name: str,
      expect_python_snippets: bool,
  ):
    skill_path = _SKILLS_DIR / folder_name / 'SKILL.md'
    self.assertTrue(skill_path.exists(), f'Missing skill file: {skill_path}')
    content = skill_path.read_text(encoding='utf-8')
    self.assertTrue(
        content.startswith('---\n'),
        f'{skill_path} must start with YAML frontmatter',
    )
    self.assertIn(f'name: {expected_name}', content)
    self.assertIn('description:', content)
    self.assertNotIn('validate_and_solve.py', content)
    self.assertNotIn('scripts/render_3d.py', content)
    self.assertNotIn('scripts/render.py', content)

    stale_bps_refs = set(
        re.findall(r'\bbps\.([A-Za-z_][A-Za-z0-9_]*)\b', content)
    ) - {'json'}
    self.assertEmpty(
        stale_bps_refs,
        f'{folder_name}/SKILL.md contains stale bps.* references:'
        f' {stale_bps_refs}',
    )
    for attr in set(re.findall(r'\bbp\.([A-Za-z_][A-Za-z0-9_]*)\b', content)):
      self.assertTrue(
          hasattr(bp, attr),
          f'{folder_name}/SKILL.md references missing bp.{attr}',
      )
    for attr in set(
        re.findall(r'\bgraph_checks\.([A-Za-z_][A-Za-z0-9_]*)\b', content)
    ):
      self.assertTrue(
          hasattr(graph_checks, attr),
          f'{folder_name}/SKILL.md references missing graph_checks.{attr}',
      )
    for attr in set(re.findall(r'\bFold\.([A-Z_][A-Z0-9_]*)\b', content)):
      self.assertTrue(
          hasattr(bp.Fold, attr),
          f'{folder_name}/SKILL.md references missing Fold.{attr}',
      )

    snippets = re.findall(r'```python\n(.*?)```', content, flags=re.DOTALL)
    if expect_python_snippets:
      self.assertNotEmpty(snippets)
    exec_ns: dict[str, object] = {
        'stick_figure_dict': bp.load_example('human').to_dict(),
    }
    for idx, snippet in enumerate(snippets):
      dedented = textwrap.dedent(snippet)
      code_obj = compile(
          dedented, f'{folder_name}/SKILL.md#snippet_{idx}', 'exec'
      )
      exec(code_obj, exec_ns)  # pylint: disable=exec-used

    json_blocks = re.findall(r'```json\n(.*?)```', content, flags=re.DOTALL)
    for json_block in json_blocks:
      parsed = json.loads(json_block)
      errors = graph_checks.prevalidate(parsed, check_symmetry=True)
      self.assertEmpty(
          errors, f'Invalid JSON example in {folder_name}: {errors}'
      )
      sf = bp.StickFigure.from_dict(parsed)
      self.assertTrue(graph_checks.is_valid_stick_figure(sf))
      design = bp.prepare(parsed)
      self.assertTrue(design.is_symmetric)

  @parameterized.parameters(
      'cat.json', 'human.json', 'spider.json', 'palm_tree.json'
  )
  def test_all_example_jsons_validate_and_pack(self, filename: str):
    path = _EXAMPLES_DIR / filename
    self.assertTrue(path.exists(), f'Missing example JSON: {path}')
    report = cli_lib.run_pipeline(
        path,
        skip_solve=True,
    )
    self.assertTrue(report['valid'], f'Validation failed: {report}')
    self.assertEmpty(report['validation_errors'])
    self.assertIsNotNone(report['packing'])
    self.assertTrue(report['packing']['success'])

  def test_run_pipeline_end_to_end_with_default_visual_artifacts(self):
    path = _EXAMPLES_DIR / 'human.json'
    with tempfile.TemporaryDirectory() as tmp_dir:
      out_dir = pathlib.Path(tmp_dir)
      report = cli_lib.validate_and_solve(
          path,
          output_dir=out_dir,
          check_symmetry=True,
          max_retries=5,
          max_steps=4000,
          skip_solve=False,
      )
      self.assertTrue(report['valid'])
      self.assertTrue(report['is_symmetric_tree'])
      self.assertTrue(report['packing']['success'])
      self.assertTrue(report['solving']['success'])
      self.assertTrue(report['solving']['is_locally_flat_foldable'])
      self.assertTrue(report['solving']['is_globally_flat_foldable'])
      self.assertTrue(report['shaping']['success'])
      self.assertIsNone(report['shaping']['shaped_flat_fold_method_error'])
      self.assertTrue((out_dir / 'stick_figure_views.png').exists())
      self.assertTrue((out_dir / 'pipeline_stages.png').exists())
      self.assertTrue((out_dir / 'shaped_3d_views.png').exists())
      self.assertTrue((out_dir / 'contact_sheet.html').exists())
      self.assertTrue((out_dir / 'packing_cp.svg').exists())
      self.assertTrue((out_dir / 'solving_cp.svg').exists())
      self.assertTrue((out_dir / 'shaped_cp.svg').exists())
      self.assertTrue((out_dir / 'layout.bps.json').exists())
      self.assertTrue((out_dir / 'solved.fold').exists())
      self.assertTrue((out_dir / 'shaped.fold').exists())
      self.assertTrue((out_dir / 'shaped_3d.fold').exists())
      self.assertTrue((out_dir / 'report.json').exists())

  def test_cli_builtin_example_target_no_viz_and_public_visual_api(self):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
      cli_lib.main(['--list-examples'])
    listed = buf.getvalue().splitlines()
    self.assertIn('cat', listed)
    self.assertIn('human', listed)

    with tempfile.TemporaryDirectory() as tmp_dir_str:
      tmp_dir = pathlib.Path(tmp_dir_str)
      cli_out = tmp_dir / 'cat_cli'
      with contextlib.redirect_stdout(io.StringIO()):
        cli_lib.main([
            'cat',
            '--output-dir',
            str(cli_out),
            '--skip-solve',
        ])
      self.assertTrue((cli_out / 'stick_figure_views.png').exists())
      self.assertTrue((cli_out / 'pipeline_stages.png').exists())
      self.assertTrue((cli_out / 'contact_sheet.html').exists())
      self.assertTrue((cli_out / 'layout.bps.json').exists())

      no_viz_out = tmp_dir / 'cat_no_viz'
      with contextlib.redirect_stdout(io.StringIO()):
        cli_lib.main([
            'cat',
            '--output-dir',
            str(no_viz_out),
            '--skip-solve',
            '--no-viz',
        ])
      self.assertFalse((no_viz_out / 'stick_figure_views.png').exists())
      self.assertFalse((no_viz_out / 'contact_sheet.html').exists())
      self.assertTrue((no_viz_out / 'layout.bps.json').exists())

      api_out = tmp_dir / 'api_viz'
      res = bp.fold('cat', preset='demo', shape=True)
      views_path = bp.save_stick_figure_views(
          res.design, api_out / 'views.png'
      )
      mesh = res.save_3d_turnaround(api_out / 'turnaround.png')
      stages_path = res.save_pipeline_summary(api_out / 'stages.png')
      self.assertTrue(views_path.exists())
      self.assertTrue((api_out / 'turnaround.png').exists())
      self.assertTrue(stages_path.exists())
      self.assertIsInstance(mesh, bp.FoldedMesh3D)
      self.assertIn('faces_vertices', mesh.to_fold_dict('cat'))

  def test_invalid_stick_figure_returns_structured_validation_errors(self):
    bad_data = {
        'name': 'broken_tree',
        'category': 'fantasy',
        'complexity': 1,
        'root': 'missing_root',
        'children': [],
    }
    report = cli_lib.run_pipeline(bad_data)
    self.assertFalse(report['valid'])
    self.assertNotEmpty(report['validation_errors'])
    self.assertIsNone(report['packing'])

  def test_missing_start_end_and_asymmetric_diagnostics(self):
    missing_joints = {
        'name': 'missing_joints',
        'category': 'animals',
        'complexity': 1,
        'color_top': [200, 200, 200],
        'color_bottom': [200, 200, 200],
        'root': 'body',
        'children': [
            {
                'name': 'body',
                'parent_joint': 'hips',
                'child_joint': 'shoulders',
                'length': 4,
                'width': 1,
                'azimuth_angle': 0,
                'elevation_angle': 0,
                'shaping_flap': False,
            },
        ],
    }
    errors = graph_checks.prevalidate(missing_joints, check_symmetry=True)
    self.assertNotEmpty(errors)
    self.assertTrue(any("is missing 'start'" in e for e in errors))

    asymmetric_data = json.loads(
        (_EXAMPLES_DIR / 'human.json').read_text(encoding='utf-8')
    )
    for stick in asymmetric_data['children']:
      if stick['name'] == 'left_arm':
        stick['azimuth_angle'] = 140.0
    sym_errors = graph_checks.prevalidate(asymmetric_data, check_symmetry=True)
    self.assertIn('Figure is not symmetric.', sym_errors)
    self.assertTrue(
        any("Stick 'left_arm'" in e for e in sym_errors),
        f'Expected per-stick symmetry error in {sym_errors}',
    )

  def test_batch_dir_skips_non_design_jsons_and_writes_incremental_reports(
      self,
  ):
    with tempfile.TemporaryDirectory() as tmp_dir_str:
      tmp_dir = pathlib.Path(tmp_dir_str)
      batch_in = tmp_dir / 'batch_in'
      batch_out = tmp_dir / 'batch_out'
      batch_in.mkdir()
      cat_text = (_EXAMPLES_DIR / 'cat.json').read_text(encoding='utf-8')
      (batch_in / 'cat.json').write_text(cat_text, encoding='utf-8')
      # Non-design JSON files in the same directory should be skipped cleanly:
      (batch_in / 'batch_summary.json').write_text('{}', encoding='utf-8')
      (batch_in / 'cat.bps.json').write_text('{"design": {}}', encoding='utf-8')
      (batch_in / 'run.metadata.json').write_text('{"id": 1}', encoding='utf-8')
      (batch_in / 'misc_notes.json').write_text(
          '{"hello": "world"}', encoding='utf-8'
      )

      with contextlib.redirect_stdout(io.StringIO()):
        cli_lib.main([
            '--batch_dir',
            str(batch_in),
            '--output_dir',
            str(batch_out),
            '--skip_solve',
            '--no-viz',
            '--pack_timeout',
            '30.0',
        ])

      summary = json.loads(
          (batch_out / 'batch_summary.json').read_text(encoding='utf-8')
      )
      self.assertEqual(list(summary.keys()), ['cat.json'])
      self.assertTrue(summary['cat.json']['packing']['success'])
      per_design_report = json.loads(
          (batch_out / 'cat' / 'report.json').read_text(encoding='utf-8')
      )
      self.assertTrue(per_design_report['packing']['success'])

  def test_high_stick_count_visualization_declutters_labels(self):
    small_sf = bp.load_example('cat')
    fig_small, ax_small = sf_visualize.visualize_stick_figure(small_sf)
    try:
      small_red_labels = [
          t.get_text() for t in ax_small.texts if t.get_color() == 'red'
      ]
      self.assertLen(small_red_labels, len(small_sf.joints))
    finally:
      plt.close(fig_small)

    # Subdivide a tree past 16 sticks and verify auto-decluttering labels only
    # degree>=3 hubs while keeping explicit show_joint_label=True available.
    spider_sf = bp.StickFigure.from_dict(
        json.loads((_EXAMPLES_DIR / 'spider.json').read_text(encoding='utf-8'))
    )
    dense_sf, _ = spider_sf.subdivide_sticks({
        s.name: [1, s.length - 1]
        for s in spider_sf.children
        if s.name != 'abdomen'
    })
    self.assertGreater(len(dense_sf.children), 16)
    fig_auto, ax_auto = sf_visualize.visualize_stick_figure(dense_sf)
    try:
      auto_red_labels = {
          t.get_text() for t in ax_auto.texts if t.get_color() == 'red'
      }
      expected_hubs = {
          j.name for j in dense_sf.joints.values() if len(j.sticks) >= 3
      }
      self.assertEqual(auto_red_labels, expected_hubs)
    finally:
      plt.close(fig_auto)

  def test_origami_designer_agent_spec_and_references(self):
    agent_path = _REPO_ROOT / '.agents' / 'agents' / 'origami-designer.md'
    self.assertTrue(agent_path.exists(), f'Missing agent file: {agent_path}')
    agent_text = agent_path.read_text(encoding='utf-8')
    self.assertTrue(agent_text.startswith('---\n'))
    self.assertIn('name: origami-designer', agent_text)
    self.assertIn('description:', agent_text)
    for skill_ref in (
        'stick-figure-design',
        'stick-figure-generation',
        'origami-solving-pipeline',
        'origami-shaping',
        'origami-critique',
    ):
      self.assertIn(skill_ref, agent_text)
      self.assertTrue((_SKILLS_DIR / skill_ref / 'SKILL.md').exists())

    refs_dir = _SKILLS_DIR / 'origami-shaping' / 'references'
    tools_md = (refs_dir / 'tools.md').read_text(encoding='utf-8')
    pose_md = (refs_dir / 'pose.md').read_text(encoding='utf-8')
    for tool_name in bp.list_shaping_tools():
      self.assertIn(tool_name, tools_md)
    for doc_text in (agent_text, tools_md, pose_md):
      for attr in set(
          re.findall(r'\bbp\.([A-Za-z_][A-Za-z0-9_]*)\b', doc_text)
      ):
        self.assertTrue(hasattr(bp, attr), f'Missing bp.{attr}')

  def test_cli_shaping_subcommands_end_to_end(self):
    with tempfile.TemporaryDirectory() as tmp_dir_str:
      run_dir = pathlib.Path(tmp_dir_str) / 'human_run'
      with contextlib.redirect_stdout(io.StringIO()):
        cli_lib.main(['human', '-o', str(run_dir), '--no-shape', '--no-viz'])
      self.assertTrue((run_dir / 'solved_bus.json').exists())

      buf_tools = io.StringIO()
      with contextlib.redirect_stdout(buf_tools):
        cli_lib.main(['tools', str(run_dir)])
      ctx = json.loads(buf_tools.getvalue())
      self.assertEqual(ctx['tools'], list(bp.list_shaping_tools()))
      self.assertTrue((run_dir / 'shaping_context.json').exists())

      buf_probe = io.StringIO()
      with contextlib.redirect_stdout(buf_probe):
        cli_lib.main(['probe', str(run_dir), '--elements', 'left_leg'])
      probe_rep = json.loads(buf_probe.getvalue())
      self.assertIn('left_leg', probe_rep)
      self.assertTrue((run_dir / 'probe_report.json').exists())

      buf_pose = io.StringIO()
      with contextlib.redirect_stdout(buf_pose):
        cli_lib.main(['pose', str(run_dir), '--no-search', '--no-viz'])
      pose_rep = json.loads(buf_pose.getvalue())
      self.assertIn('fidelity', pose_rep)
      self.assertTrue((run_dir / 'pose_report.json').exists())
      self.assertTrue((run_dir / 'shaping_state.json').exists())

      calls = json.dumps([
          {
              'name': 'simple_fold',
              'arguments': {
                  'element_name': 'left_leg',
                  'angle': 45.0,
                  'depth': 0.0,
              },
          },
          {
              'name': 'simple_fold',
              'arguments': {
                  'element_name': 'right_leg',
                  'angle': -45.0,
                  'depth': 0.0,
              },
          },
      ])
      buf_act = io.StringIO()
      with contextlib.redirect_stdout(buf_act):
        cli_lib.main([
            'actuate',
            str(run_dir),
            '--reset',
            '--no-viz',
            '--calls',
            calls,
        ])
      act_out = json.loads(buf_act.getvalue())
      self.assertLen(act_out, 2)
      self.assertTrue(all(item['ok'] for item in act_out))
      self.assertTrue((run_dir / 'shaped_cp.svg').exists())
      self.assertTrue((run_dir / 'shaping_history.jsonl').exists())

      buf_score = io.StringIO()
      with contextlib.redirect_stdout(buf_score):
        cli_lib.main(['score', str(run_dir)])
      score_out = json.loads(buf_score.getvalue())
      self.assertTrue(score_out['clean_gate']['locally_flat_foldable'])
      self.assertTrue((run_dir / 'fidelity_report.json').exists())


if __name__ == '__main__':
  absltest.main()
