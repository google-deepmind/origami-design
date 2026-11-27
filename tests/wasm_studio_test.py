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

"""Tests for the standalone COrigami Python WASM Editor (`docs/editor/`)."""

import base64
from collections.abc import Callable
import hashlib
import importlib.util
import io
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import threading
from typing import Any, cast
import urllib.request
import zipfile

from absl.testing import absltest

_REPO_ROOT = pathlib.Path(__file__).parents[1]
_DOCS_DIR = _REPO_ROOT / 'docs'
_EDITOR_DIR = _DOCS_DIR / 'editor'
_MODELS_DIR = _DOCS_DIR / 'models'
_SERVE_SCRIPT = _REPO_ROOT / 'scripts' / 'serve_studio.py'


def _resolve_node_bin() -> str | None:
  node = os.environ.get('NODE_BIN') or shutil.which('node')
  if node and not os.path.isabs(node) and not os.path.exists(node):
    node = os.path.join(
        os.environ.get('TEST_SRCDIR', ''),
        os.environ.get('TEST_WORKSPACE', ''),
        node,
    )
  if node and (os.path.exists(node) or shutil.which(node)):
    return node
  return None


def _load_serve_studio_module():
  spec = importlib.util.spec_from_file_location('serve_studio', _SERVE_SCRIPT)
  assert spec is not None and spec.loader is not None
  mod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(mod)
  return mod


def _extract_worker_python_sources() -> tuple[str, str]:
  """Extracts `PYTHON_BOOTSTRAP` and `PYTHON_BRIDGE` from `docs/editor/js/py_worker.js`."""
  worker_js = _EDITOR_DIR / 'js' / 'py_worker.js'
  node = _resolve_node_bin()
  if node:
    js_src = worker_js.read_text(encoding='utf-8')
    # Evaluate only up to `async function fetchWheelBytes` so browser globals
    # (`self.onmessage`) are never needed.
    prefix = js_src.split('async function fetchWheelBytes', 1)[0]
    probe = (
        'globalThis.self = {postMessage() {}};\n'
        + prefix
        + '\nconsole.log(JSON.stringify([PYTHON_BOOTSTRAP, PYTHON_BRIDGE]));\n'
    )
    res = subprocess.run(
        [node, '-e', probe],
        capture_output=True,
        text=True,
        check=True,
    )
    boot, bridge = json.loads(res.stdout)
    return boot, bridge

  source = worker_js.read_text(encoding='utf-8')
  m_boot = re.search(
      r'const PYTHON_BOOTSTRAP = `(?P<body>.*?)`;', source, flags=re.DOTALL
  )
  m_bridge = re.search(
      r'const PYTHON_BRIDGE = `(?P<body>.*?)`;', source, flags=re.DOTALL
  )
  assert m_boot is not None and m_bridge is not None
  return (
      m_boot.group('body').replace('\\\\', '\\'),
      m_bridge.group('body').replace('\\\\', '\\'),
  )


class WasmStudioTest(absltest.TestCase):

  def test_editor_is_separate_app_from_showcase_studio(self):
    studio_html = (_DOCS_DIR / 'index.html').read_text(encoding='utf-8')
    editor_html = (_EDITOR_DIR / 'index.html').read_text(encoding='utf-8')

    # Original Studio remains separate and untouched by WASM editor UI.
    self.assertIn('<title>COrigami Studio</title>', studio_html)
    self.assertNotIn('flap-shaper-card', studio_html)

    # Standalone Editor app lives under `docs/editor/` and links back to `../`.
    self.assertIn('<title>COrigami Editor</title>', editor_html)
    self.assertIn('id="flap-shaper-card"', editor_html)
    self.assertIn('id="flap-select"', editor_html)
    self.assertIn('id="gallery-popover"', editor_html)
    for tab in ('shape', 'pattern', 'packing', 'sticks', 'export'):
      self.assertIn(f'data-tab="{tab}"', editor_html)
    self.assertIn('href="../"', editor_html)

    urls = re.findall(r'(?:src|href)="([^":]+)"', editor_html)
    self.assertContainsSubset({'style.css', 'js/main.js', '../'}, set(urls))
    for url in urls:
      target = (_EDITOR_DIR / url).resolve()
      self.assertTrue(target.exists(), f'Missing editor link target: {url}')
    # Verify every `byId(...)` in `main.js` exists in `editor/index.html` and
    # every method invoked on `stage`, `cpEditor`, `stickEditor`, and `wasm`
    # exists on the target class (guards against runtime TypeError in UI).
    main_js = (_EDITOR_DIR / 'js' / 'main.js').read_text(encoding='utf-8')
    html_ids = set(re.findall(r'id="([^"]+)"', editor_html))
    js_ids = set(re.findall(r"byId\('([^']+)'\)", main_js))
    self.assertEmpty(js_ids - html_ids)

    for var_name, js_name in (
        ('stage', 'stage.js'),
        ('cpEditor', 'cp_editor.js'),
        ('stickEditor', 'stick_editor.js'),
        ('wasm', 'wasm_client.js'),
    ):
      cls_src = (_EDITOR_DIR / 'js' / js_name).read_text(encoding='utf-8')
      calls = set(
          re.findall(rf'\b{var_name}(?:\?)?\.([a-zA-Z0-9_]+)\s*\(', main_js)
      )
      defined = set(
          re.findall(
              r'^\s*(?:async\s+|get\s+|set\s+)?([a-zA-Z0-9_]+)\s*\(',
              cls_src,
              flags=re.M,
          )
      )
      self.assertEmpty(
          calls - defined, f'{var_name} missing methods in {js_name}'
      )

    for script in (_EDITOR_DIR / 'js').glob('*.js'):
      source = script.read_text(encoding='utf-8')
      for module in re.findall(r"from '(\.\.?/[^']+)'", source):
        resolved = (script.parent / module).resolve()
        self.assertTrue(
            resolved.is_file(), f'{script.name} -> missing import {module}'
        )

  def test_core_pipeline_never_imports_scipy_or_matplotlib(self):
    import bp_pipeline as bp  # pylint: disable=g-import-not-at-top

    tree = {
        'name': 'tiny_biped',
        'category': 'custom',
        'complexity': 1,
        'root': 'trunk',
        'children': [
            {
                'name': 'trunk',
                'start': 'hip',
                'end': 'neck',
                'length': 2,
                'width': 1,
                'azimuth_angle': 0,
                'elevation_angle': 90,
            },
            {
                'name': 'arm_l',
                'start': 'neck',
                'end': 'hand_l',
                'length': 2,
                'width': 1,
                'azimuth_angle': 90,
                'elevation_angle': 0,
            },
            {
                'name': 'arm_r',
                'start': 'neck',
                'end': 'hand_r',
                'length': 2,
                'width': 1,
                'azimuth_angle': -90,
                'elevation_angle': 0,
            },
            {
                'name': 'leg_l',
                'start': 'hip',
                'end': 'foot_l',
                'length': 2,
                'width': 1,
                'azimuth_angle': 135,
                'elevation_angle': -45,
            },
            {
                'name': 'leg_r',
                'start': 'hip',
                'end': 'foot_r',
                'length': 2,
                'width': 1,
                'azimuth_angle': -135,
                'elevation_angle': -45,
            },
        ],
    }
    design = bp.prepare(tree)
    packed = bp.pack(design)
    solved = bp.solve_best(packed)
    shaped = bp.shape(solved)
    mesh = bp.build_folded_mesh_3d(shaped, packed=packed)
    _ = bp.mesh_to_fold_dict(mesh, 'tiny_biped')
    _ = bp.to_svg(shaped)

    heavy = [
        m
        for m in sys.modules
        if m == 'scipy'
        or m.startswith('scipy.')
        or m == 'matplotlib'
        or m.startswith('matplotlib.')
    ]
    self.assertEmpty(heavy, f'Unexpected heavy imports loaded: {heavy}')

    if sys.executable and os.path.exists(sys.executable):
      env = dict(os.environ)
      src_path = str(_REPO_ROOT / 'src')
      existing_pp = env.get('PYTHONPATH', '')
      env['PYTHONPATH'] = (
          f'{src_path}{os.pathsep}{existing_pp}' if existing_pp else src_path
      )
      res = subprocess.run(
          [
              sys.executable,
              '-c',
              (
                  'import sys, bp_pipeline as bp; '
                  'assert not [m for m in sys.modules '
                  'if m.startswith(("scipy", "matplotlib"))]'
              ),
          ],
          env=env,
          capture_output=True,
          text=True,
          check=False,
      )
      self.assertEqual(res.returncode, 0, msg=res.stderr or res.stdout)

  def test_python_312_313_eager_annotations_and_classifiers(self):
    pyproject_text = (_REPO_ROOT / 'pyproject.toml').read_text(encoding='utf-8')
    self.assertIn('requires-python = ">=3.12"', pyproject_text)
    for ver in ('3.12', '3.13', '3.14'):
      self.assertIn(f'Programming Language :: Python :: {ver}', pyproject_text)

    import inspect  # pylint: disable=g-import-not-at-top
    import typing  # pylint: disable=g-import-not-at-top
    import bp_pipeline as bp  # pylint: disable=g-import-not-at-top
    from bp_pipeline import export as export_mod  # pylint: disable=g-import-not-at-top

    # Verify `BpsImportResult.stick_figure` return annotation resolves cleanly.
    sf_fget = export_mod.BpsImportResult.stick_figure.fget
    self.assertIsNotNone(sf_fget)
    hints = typing.get_type_hints(sf_fget)
    self.assertIs(hints['return'], bp.StickFigure)

    # Simulate Python <= 3.13 eager annotation evaluation (`Format.VALUE = 1`)
    # on every module in `bp_pipeline` that does not opt into PEP 563
    # (`from __future__ import annotations`).
    pkg_dir = _REPO_ROOT / 'src' / 'bp_pipeline'
    checked_modules = 0
    for py_path in sorted(pkg_dir.rglob('*.py')):
      rel = py_path.relative_to(_REPO_ROOT / 'src').with_suffix('')
      parts = rel.parts[:-1] if rel.parts[-1] == '__init__' else rel.parts
      mod_name = '.'.join(parts)
      if mod_name.endswith('.__main__') or mod_name == 'bp_pipeline.cli':
        continue
      mod = importlib.import_module(mod_name)
      checked_modules += 1
      if 'annotations' in mod.__dict__:
        continue
      targets = [mod]
      for _, member in inspect.getmembers(mod):
        if getattr(member, '__module__', None) != mod_name:
          continue
        targets.append(member)
        if inspect.isclass(member):
          for _, attr in member.__dict__.items():
            if isinstance(attr, property) and attr.fget is not None:
              targets.append(attr.fget)
            elif isinstance(attr, (staticmethod, classmethod)):
              targets.append(attr.__func__)
            elif inspect.isfunction(attr):
              targets.append(attr)
      for obj in targets:
        ann_fn = getattr(obj, '__annotate__', None)
        if callable(ann_fn):
          ann_fn(1)  # Format.VALUE: raises NameError/AttributeError if broken!
    self.assertGreater(checked_modules, 30)

  def test_bootstrap_shims_cover_bp_pipeline_without_external_deps(self):
    if not (sys.executable and os.path.exists(sys.executable)):
      self.skipTest(
          'Python executable unavailable for hermetic shim subprocess'
      )
    boot_py, _ = _extract_worker_python_sources()
    env = dict(os.environ)
    src_path = str(_REPO_ROOT / 'src')
    existing_pp = env.get('PYTHONPATH', '')
    env['PYTHONPATH'] = (
        f'{src_path}{os.pathsep}{existing_pp}' if existing_pp else src_path
    )
    probe = f"""
import importlib.abc
import sys

class _BlockExternalShimmedPkgs(importlib.abc.MetaPathFinder):
  def find_spec(self, fullname, path, target=None):
    if fullname in ('immutabledict', 'more_itertools', 'absl') or fullname.startswith(('immutabledict.', 'more_itertools.', 'absl.')):
      raise ModuleNotFoundError(f'Blocked external package in shim test: {{fullname}}')
    return None

for _k in list(sys.modules):
  if _k in ('immutabledict', 'more_itertools', 'absl') or _k.startswith(('immutabledict.', 'more_itertools.', 'absl.')):
    del sys.modules[_k]
sys.meta_path.insert(0, _BlockExternalShimmedPkgs())

exec({boot_py!r}, {{'__name__': '__boot__'}})

import bp_pipeline as bp
tree = {{
    'name': 'shim_check',
    'category': 'custom',
    'complexity': 1,
    'root': 'trunk',
    'children': [
        {{'name': 'trunk', 'start': 'a', 'end': 'b', 'length': 2, 'width': 1, 'azimuth_angle': 0, 'elevation_angle': 90}},
        {{'name': 'arm_l', 'start': 'b', 'end': 'c', 'length': 2, 'width': 1, 'azimuth_angle': 90, 'elevation_angle': 0}},
        {{'name': 'arm_r', 'start': 'b', 'end': 'd', 'length': 2, 'width': 1, 'azimuth_angle': -90, 'elevation_angle': 0}},
    ],
}}
design = bp.prepare(tree)
packed = bp.pack(design)
solved = bp.solve_best(packed)
state = bp.ShapingState.from_solved(solved)
flaps, _ = state.shaping_elements
res = bp.apply_shaping_tool(state, 'simple_fold', element_name=flaps[0].name, angle=25, depth=0.3)
assert res.success, res.message
shaped = bp.shape(solved)
rep = bp.evaluate_fidelity(shaped, tree=design.source)
assert rep.fidelity >= 0.0
bps_doc = bp.to_bp_studio(packed)
assert bp.import_bps(bps_doc).packed is not None
"""
    proc = subprocess.run(
        [sys.executable, '-c', probe],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    self.assertEqual(proc.returncode, 0, msg=proc.stderr or proc.stdout)

  def test_svg_parser_rejects_doctype_and_entity_and_numpy_helpers(self):
    import numpy as np  # pylint: disable=g-import-not-at-top
    from bp_pipeline.crease_pattern.parser import svg as svg_parser  # pylint: disable=g-import-not-at-top
    from bp_pipeline.packing import river as river_mod  # pylint: disable=g-import-not-at-top
    from bp_pipeline.packing import scoring as scoring_mod  # pylint: disable=g-import-not-at-top
    from bp_pipeline.shaping import geometric_utils  # pylint: disable=g-import-not-at-top

    valid_doc = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10">'
        '<rect x="0" y="0" width="10" height="10" stroke="black"/>'
        '<line x1="0" y1="0" x2="10" y2="10" stroke="red"/>'
        '</svg>'
    )
    self.assertTrue(svg_parser.valid_xml(valid_doc))
    self.assertTrue(svg_parser.valid_svg(valid_doc))
    self.assertGreaterEqual(len(svg_parser.parse_svg(valid_doc).creases), 5)

    doctype_only = '<!DOCTYPE svg>' + valid_doc
    standalone_entity = '<!ENTITY xxe "payload">' + valid_doc
    entity_doc = (
        '<?xml version="1.0"?>'
        '<!DOCTYPE svg [<!ENTITY xxe SYSTEM "file:///etc/passwd">]>'
        + valid_doc
    )
    for unsafe in (doctype_only, standalone_entity, entity_doc):
      self.assertFalse(svg_parser.valid_xml(unsafe))
      self.assertFalse(svg_parser.valid_svg(unsafe))
      with self.assertRaises(svg_parser.SVGParseError):
        svg_parser.parse_svg(unsafe)
      with self.assertRaises(svg_parser.SVGParseError):
        svg_parser.parse_svg(unsafe, repair=True)

    # Direct boundary & ray-cast checks on `_point_in_polygon_2d`.
    tri = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]], dtype=float)
    self.assertFalse(
        geometric_utils._point_in_polygon_2d(0.5, 0.5, tri[:2], 1e-6)
    )
    self.assertTrue(
        geometric_utils._point_in_polygon_2d(0.4, 0.4, tri, 0.0)
    )
    # Point near the closing segment `(poly[-1], poly[0])` along `x = 0`.
    self.assertTrue(
        geometric_utils._point_in_polygon_2d(-0.005, 1.0, tri, 0.01)
    )
    self.assertFalse(
        geometric_utils._point_in_polygon_2d(-0.02, 1.0, tri, 0.01)
    )
    self.assertFalse(
        geometric_utils._point_in_polygon_2d(-0.005, 1.0, tri, -0.5)
    )

    # Direct L_inf distance checks on `_chessboard_distance_transform`.
    self.assertEqual(
        scoring_mod._chessboard_distance_transform(
            np.zeros((0, 3), dtype=bool)
        ).shape,
        (0, 3),
    )
    self.assertEqual(
        scoring_mod._chessboard_distance_transform(
            np.zeros((3, 0), dtype=bool)
        ).shape,
        (3, 0),
    )
    np.testing.assert_array_equal(
        scoring_mod._chessboard_distance_transform(
            np.zeros((2, 3), dtype=bool)
        ),
        np.zeros((2, 3), dtype=np.int32),
    )
    all_true = scoring_mod._chessboard_distance_transform(
        np.ones((2, 3), dtype=bool)
    )
    np.testing.assert_array_equal(all_true, np.full((2, 3), -1, dtype=np.int32))

    # Direct 4-connected component checks on `_label_4connected`.
    empty_labels, empty_count = river_mod._label_4connected(
        np.zeros((3, 3), dtype=bool)
    )
    self.assertEqual(empty_count, 0)
    np.testing.assert_array_equal(
        empty_labels, np.zeros((3, 3), dtype=np.int32)
    )
    comp_mask = np.array(
        [[True, True, False], [False, False, False], [False, False, True]],
        dtype=bool,
    )
    comp_labels, comp_count = river_mod._label_4connected(comp_mask)
    self.assertEqual(comp_count, 2)
    self.assertEqual(int(comp_labels[0, 0]), int(comp_labels[0, 1]))
    self.assertNotEqual(int(comp_labels[0, 0]), int(comp_labels[2, 2]))
    self.assertEqual(int(comp_labels[1, 1]), 0)

    # Single False in bottom-right corner forces the backward row & col pass.
    mask_br = np.ones((4, 5), dtype=bool)
    mask_br[3, 4] = False
    dist_br = scoring_mod._chessboard_distance_transform(mask_br)
    self.assertEqual(int(dist_br[0, 0]), 4)
    self.assertEqual(int(dist_br[3, 0]), 4)
    self.assertEqual(int(dist_br[0, 4]), 3)
    self.assertEqual(int(dist_br[3, 4]), 0)

  def test_wheel_builder_and_dynamic_dev_server(self):
    serve_mod = _load_serve_studio_module()
    parsed_defaults = serve_mod._build_arg_parser().parse_args([])
    self.assertEqual(parsed_defaults.host, '127.0.0.1')
    wheel_bytes = serve_mod.build_wheel_bytes(_REPO_ROOT)
    self.assertGreater(len(wheel_bytes), 50_000)
    self.assertLess(len(wheel_bytes), 2_000_000)

    with zipfile.ZipFile(io.BytesIO(wheel_bytes), 'r') as zf:
      names = set(zf.namelist())
      self.assertIn('bp_pipeline/__init__.py', names)
      self.assertIn('bp_pipeline-0.1.0.dist-info/METADATA', names)
      self.assertIn('bp_pipeline-0.1.0.dist-info/WHEEL', names)
      self.assertIn('bp_pipeline-0.1.0.dist-info/RECORD', names)
      for name in names:
        self.assertNotIn('__pycache__', name)
        self.assertFalse(name.endswith('/BUILD') or name.endswith('.pyc'), name)

      record_lines = (
          zf.read('bp_pipeline-0.1.0.dist-info/RECORD')
          .decode('utf-8')
          .strip()
          .splitlines()
      )
      for row in record_lines:
        arcname, digest_spec, size_str = row.split(',')
        if arcname.endswith('/RECORD'):
          continue
        payload = zf.read(arcname)
        self.assertLen(payload, int(size_str))
        expected = (
            'sha256='
            + base64.urlsafe_b64encode(hashlib.sha256(payload).digest())
            .rstrip(b'=')
            .decode('ascii')
        )
        self.assertEqual(digest_spec, expected)

    with serve_mod.create_server('127.0.0.1', 0, _REPO_ROOT) as httpd:
      port = httpd.server_address[1]
      thread = threading.Thread(target=httpd.serve_forever, daemon=True)
      thread.start()
      try:
        with urllib.request.urlopen(
            f'http://127.0.0.1:{port}{serve_mod.WHEEL_URL_PATH}', timeout=5
        ) as resp:
          self.assertEqual(resp.status, 200)
          served_wheel = resp.read()
        self.assertEqual(served_wheel, wheel_bytes)

        with urllib.request.urlopen(
            f'http://127.0.0.1:{port}/index.html', timeout=5
        ) as resp:
          self.assertEqual(resp.status, 200)
          studio_html = resp.read().decode('utf-8')
        self.assertIn('COrigami Studio', studio_html)

        with urllib.request.urlopen(
            f'http://127.0.0.1:{port}/editor/index.html', timeout=5
        ) as resp:
          self.assertEqual(resp.status, 200)
          editor_html = resp.read().decode('utf-8')
        self.assertIn('COrigami Editor', editor_html)
      finally:
        httpd.shutdown()

  def test_pyodide_worker_rpc_end_to_end(self):
    boot_py, bridge_py = _extract_worker_python_sources()
    ns: dict[str, object] = {'__name__': '__py_worker_test__'}
    exec(boot_py, ns)  # pylint: disable=exec-used
    exec(bridge_py, ns)  # pylint: disable=exec-used
    dispatch = cast(Callable[[str, str], str], ns['dispatch_rpc'])
    self.assertTrue(callable(dispatch))

    def rpc(
        action: str, payload: dict[str, Any] | None = None
    ) -> dict[str, Any]:
      return json.loads(dispatch(action, json.dumps(payload or {})))

    init_info = rpc('init')
    self.assertGreaterEqual(len(init_info['examples']), 40)
    self.assertGreaterEqual(len(init_info['allToolNames']), 20)

    # 1. Stick Figure -> Pack -> Update Packing -> Solve -> Shape -> Export.
    tiny_tree = {
        'name': 'tiny_star',
        'root': 'body',
        'children': [
            {
                'name': 'body',
                'start': 'base',
                'end': 'top',
                'length': 2,
                'width': 1,
                'azimuth_angle': 0,
                'elevation_angle': 90,
            },
            {
                'name': 'wing_left',
                'start': 'top',
                'end': 'tip_left',
                'length': 2,
                'width': 1,
                'azimuth_angle': 80,
                'elevation_angle': 10,
            },
            {
                'name': 'wing_right',
                'start': 'top',
                'end': 'tip_right',
                'length': 2,
                'width': 1,
                'azimuth_angle': -80,
                'elevation_angle': 10,
            },
        ],
    }
    sf_snap = rpc('set_stick_figure', {'stickFigure': tiny_tree})
    self.assertTrue(sf_snap['treeStats']['isSymmetric'])

    packed_snap = rpc(
        'pack', {'config': {'maxRetries': 4, 'timeoutSeconds': 10.0}}
    )
    self.assertIsNotNone(packed_snap['packing'])
    self.assertGreater(packed_snap['packing']['gridSize'], 0)
    self.assertNotEmpty(packed_snap['packing']['creases'])
    self.assertNotEmpty(packed_snap['packing']['leafDistances'])
    self.assertIn('vertices_coords', packed_snap['foldDict'])

    # Test round-trip packing update (both valid and incomplete/dirty moves).
    upd_snap = rpc(
        'update_packing', {'gridSize': packed_snap['packing']['gridSize']}
    )
    self.assertTrue(upd_snap['packingValid'])
    self.assertIsNotNone(upd_snap['packing'])
    dirty_snap = rpc(
        'update_packing',
        {'gridSize': packed_snap['packing']['gridSize'] + 4},
    )
    self.assertFalse(dirty_snap['packingValid'])
    self.assertNotEmpty(dirty_snap['bpsValidationErrors'])

    solved_snap = rpc('solve', {'config': {'checkGlobalFf': True}})
    self.assertTrue(solved_snap['hasSolved'])
    self.assertTrue(solved_snap['solveStats']['globallyFlatFoldable'])

    ff_res = rpc('check_global_ff', {})
    self.assertTrue(ff_res['flatFoldable'])

    # Apply a shaping tool + undo + reset.
    flap_elems = [
        e['name'] for e in solved_snap['shapingElements'] if e['kind'] == 'flap'
    ]
    self.assertNotEmpty(flap_elems)
    tool_snap = rpc(
        'apply_tool',
        {
            'tool': 'simple_fold',
            'args': {'element_name': flap_elems[0], 'angle': 30, 'depth': 0.3},
            'symmetric': False,
        },
    )
    self.assertLen(tool_snap['shapingHistory'], 1)
    undone_snap = rpc('undo_shaping')
    self.assertEmpty(undone_snap['shapingHistory'])

    shaped_snap = rpc(
        'shape', {'config': {'method': 'shape', 'strategy': 'DEFAULT'}}
    )
    self.assertTrue(shaped_snap['hasShaped'])
    self.assertIsNotNone(shaped_snap['fidelity'])
    for fid_key in (
        'overall',
        'pose',
        'width',
        'separation',
        'foldVertexErrorPct',
    ):
      self.assertIn(fid_key, shaped_snap['fidelity'])
      self.assertIsInstance(shaped_snap['fidelity'][fid_key], (int, float))

    exported = rpc('export_files')
    self.assertIsNotNone(exported['foldJson'])
    self.assertIsNotNone(exported['svgText'])
    self.assertIsNotNone(exported['cpText'])
    self.assertIsNotNone(exported['bpsJson'])
    self.assertIsNotNone(exported['stickFigureJson'])
    zip_bytes = base64.b64decode(exported['zipBase64'])
    with zipfile.ZipFile(io.BytesIO(zip_bytes), 'r') as zf:
      self.assertContainsSubset(
          {
              'shaped_3d.fold',
              'crease_pattern.svg',
              'crease_pattern.cp',
              'stick_figure.json',
          },
          set(zf.namelist()),
      )

    # 2. Universal file drop: ORIPA .cp, SVG, 2D .fold, and XXE SVG rejection.
    cp_drop = rpc(
        'drop_files',
        {
            'files': [{
                'name': 'diagonal.cp',
                'text': (
                    '1 -200 -200 200 -200\n'
                    '1 200 -200 200 200\n'
                    '1 200 200 -200 200\n'
                    '1 -200 200 -200 -200\n'
                    '2 -200 -200 200 200\n'
                ),
            }],
        },
    )
    self.assertEqual(cp_drop['droppedKind'], 'cp')
    self.assertLen(cp_drop['foldDict']['faces_vertices'], 2)

    svg_drop = rpc(
        'drop_files',
        {
            'files': [{
                'name': 'crease_pattern.svg',
                'text': exported['svgText'],
            }],
        },
    )
    self.assertEqual(svg_drop['droppedKind'], 'svg')
    self.assertNotEmpty(svg_drop['foldDict']['faces_vertices'])

    fold_2d_drop = rpc(
        'drop_files',
        {
            'files': [{
                'name': 'flat_only.fold',
                'text': json.dumps({
                    'file_spec': 1.1,
                    'vertices_coords': [[0, 0], [1, 0], [1, 1], [0, 1]],
                    'edges_vertices': [[0, 1], [1, 2], [2, 3], [3, 0], [0, 2]],
                    'edges_assignment': ['B', 'B', 'B', 'B', 'V'],
                    'edges_foldAngle': [0, 0, 0, 0, 120],
                }),
            }],
        },
    )
    self.assertEqual(fold_2d_drop['droppedKind'], 'fold_2d')
    self.assertLen(fold_2d_drop['foldDict']['faces_vertices'], 2)

    bps_drop_pack_only = rpc(
        'drop_files',
        {
            'files': [{'name': 'tiny.bps', 'text': exported['bpsJson']}],
            'autoPipeline': False,
        },
    )
    self.assertEqual(bps_drop_pack_only['droppedKind'], 'bps')
    self.assertFalse(bps_drop_pack_only['hasSolved'])

    bps_drop_solved = rpc(
        'drop_files',
        {
            'files': [{'name': 'tiny.bps', 'text': exported['bpsJson']}],
            'autoPipeline': True,
        },
    )
    self.assertEqual(bps_drop_solved['droppedKind'], 'bps')
    self.assertTrue(bps_drop_solved['hasSolved'])

    xxe_svg = (
        '<?xml version="1.0"?>'
        '<!DOCTYPE svg [<!ENTITY xxe SYSTEM "file:///etc/passwd">]>'
        '<svg xmlns="http://www.w3.org/2000/svg"><text>&xxe;</text></svg>'
    )
    with self.assertRaises(ValueError):
      rpc('drop_files', {'files': [{'name': 'evil.svg', 'text': xxe_svg}]})

    doctype_svg = '<!DOCTYPE svg>' + exported['svgText']
    with self.assertRaises(ValueError):
      rpc(
          'drop_files',
          {'files': [{'name': 'doctype.svg', 'text': doctype_svg}]},
      )

    # Verify `.zip` member count and uncompressed byte size caps.
    too_many_buf = io.BytesIO()
    with zipfile.ZipFile(too_many_buf, 'w', zipfile.ZIP_DEFLATED) as zf:
      for i in range(65):
        zf.writestr(f'f_{i}.cp', '1 0 0 1 0\n')
    with self.assertRaisesRegex(ValueError, 'too many files'):
      rpc(
          'drop_files',
          {
              'files': [{
                  'name': 'many.zip',
                  'base64': base64.b64encode(too_many_buf.getvalue()).decode(
                      'ascii'
                  ),
              }]
          },
      )

    huge_buf = io.BytesIO()
    with zipfile.ZipFile(huge_buf, 'w', zipfile.ZIP_DEFLATED) as zf:
      zf.writestr('bomb.cp', '0' * (33 * 1024 * 1024))
    with self.assertRaisesRegex(ValueError, 'maximum uncompressed size'):
      rpc(
          'drop_files',
          {
              'files': [{
                  'name': 'bomb.zip',
                  'base64': base64.b64encode(huge_buf.getvalue()).decode(
                      'ascii'
                  ),
              }]
          },
      )

  def test_editor_js_modules_and_crease_shaper_under_node(self):
    node = _resolve_node_bin()
    if os.environ.get('NODE_BIN'):
      self.assertIsNotNone(node)
    if not node:
      self.skipTest('node runtime not available')

    eagle_fold = (_MODELS_DIR / 'golden-eagle' / 'shaped_3d.fold').resolve()
    eagle_sf = (_MODELS_DIR / 'golden-eagle' / 'stick_figure.json').resolve()
    dragon_fold = (_MODELS_DIR / 'wyrmlord-dragon' / 'shaped_3d.fold').resolve()
    dragon_sf = (
        _MODELS_DIR / 'wyrmlord-dragon' / 'stick_figure.json'
    ).resolve()
    pegasus_fold = (
        _MODELS_DIR / 'winged-pegasus' / 'shaped_3d.fold'
    ).resolve()
    pegasus_sf = (
        _MODELS_DIR / 'winged-pegasus' / 'stick_figure.json'
    ).resolve()
    scorp_fold = (
        _MODELS_DIR / 'emperor-scorpion' / 'shaped_3d.fold'
    ).resolve()
    scorp_sf = (
        _MODELS_DIR / 'emperor-scorpion' / 'stick_figure.json'
    ).resolve()

    with tempfile.TemporaryDirectory() as tmpdir:
      tmp = pathlib.Path(tmpdir)
      (tmp / 'package.json').write_text(
          '{"type": "module"}\n', encoding='utf-8'
      )
      # Preserve exact relative structure (`js/` and `editor/js/`) with writable
      # directory permissions (`TEST_SRCDIR` runfiles directories are 0555).
      (tmp / 'js').mkdir(parents=True, exist_ok=True)
      (tmp / 'editor' / 'js').mkdir(parents=True, exist_ok=True)
      for js_file in (_DOCS_DIR / 'js').glob('*.js'):
        shutil.copyfile(js_file, tmp / 'js' / js_file.name)
      for js_file in (_EDITOR_DIR / 'js').glob('*.js'):
        shutil.copyfile(js_file, tmp / 'editor' / 'js' / js_file.name)

      for js_check in sorted((tmp / 'editor' / 'js').glob('*.js')):
        chk = subprocess.run(
            [node, '--check', str(js_check)],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(
            chk.returncode,
            0,
            msg=f'{js_check.name}: {chk.stderr or chk.stdout}',
        )

      script_path = tmp / 'editor' / 'js' / 'check_editor.mjs'
      script_path.write_text(
          f"""
      import fs from 'node:fs';
      import assert from 'node:assert/strict';
      import {{Folder, parseFold, strainPercent}} from '../../js/fold.js';
      import {{buildFoldingSchedule}} from '../../js/schedule.js';
      import {{
        addCreaseSegment,
        analyzeVertices,
        buildFoldFromCpGraph,
        createBlankSquareGraph,
        extractPlanarFaces,
        transformGraph,
      }} from './cp_geometry.js';
      import {{
        computeStickSkeleton,
        createStarterStickFigure,
        findMirroredStickName,
      }} from './stick_editor.js';
      import {{
        applyFlapShaping,
        buildSubFlapCatalog,
        findMirroredRegionIndex,
        findMirroredSubFlapId,
        formatRegionName,
      }} from './flap_shaper.js';

      // 1. Planar CP graph & Kawasaki/Maekawa diagnostics (`cp_geometry.js`).
      let g = createBlankSquareGraph(8);
      assert.equal(extractPlanarFaces(g.vertices, g.edges).length, 1);
      g = addCreaseSegment(g, [0, 0], [0.5, 0.5], 'M', -180, /* mirror= */ true);
      g = addCreaseSegment(g, [0.5, 0.5], [0.5, 1], 'M', -180, false);
      g = addCreaseSegment(g, [0.5, 0.5], [0.5, 0], 'V', 180, false);
      assert.deepEqual(analyzeVertices(g), []);
      const built = buildFoldFromCpGraph(g, 'Waterbomb Half-Base');
      assert.equal(built.foldDict.faces_vertices.length, 4);
      assert.equal(built.model.faces.length, 4);

      const badMaekawa = transformGraph(g, 'none');
      const mIdx = badMaekawa.assignments.indexOf('M');
      badMaekawa.assignments[mIdx] = 'V';
      badMaekawa.foldAngles[mIdx] = 180;
      const maekawaViolations = analyzeVertices(badMaekawa);
      assert.equal(maekawaViolations.length, 1);
      assert.equal(maekawaViolations[0].kind, 'maekawa');
      assert.deepEqual(analyzeVertices(transformGraph(g, 'invertMV')), []);

      // 2. Stick skeleton forward kinematics & bilateral mirror lookup.
      const starter = createStarterStickFigure();
      assert.equal(findMirroredStickName(starter, 'arm_left'), 'arm_right');
      assert.equal(findMirroredStickName(
          {{children: [{{name: 'wing_L'}}, {{name: 'wing_R'}}]}}, 'wing_L'), 'wing_R');
      assert.equal(findMirroredStickName(starter, 'spine'), null);
      const skel = computeStickSkeleton(starter);
      const lHand = skel.joints.get('hand_left');
      const rHand = skel.joints.get('hand_right');
      assert(Math.abs(lHand[0] + rHand[0]) < 1e-6);
      assert(Math.abs(lHand[1] - rHand[1]) < 1e-6);
      assert(Math.abs(lHand[2] - rHand[2]) < 1e-6);

      // 3. Unmerged sub-flap catalog & pure crease-backed shaping (`flap_shaper.js`).
      const rawEagle = JSON.parse(
          fs.readFileSync({json.dumps(str(eagle_fold))}, 'utf8'));
      const sf = JSON.parse(
          fs.readFileSync({json.dumps(str(eagle_sf))}, 'utf8'));
      const rawDragon = JSON.parse(
          fs.readFileSync({json.dumps(str(dragon_fold))}, 'utf8'));
      const dragonSf = JSON.parse(
          fs.readFileSync({json.dumps(str(dragon_sf))}, 'utf8'));

      const eagle = parseFold(rawEagle);
      const eagleCat = buildSubFlapCatalog(eagle, sf);
      for (const expectedStick of ['neck', 'head', 'beak', 'beak_hook', 'wing_L', 'wing_R']) {{
        const entry = eagleCat.subFlaps.find((s) => s.stickName === expectedStick);
        assert(entry, `missing sub-flap ${{expectedStick}}`);
        assert(entry.faceIndices.length > 0, `empty faces on ${{expectedStick}}`);
      }}

      const dragonModel = parseFold(rawDragon);
      const dragonCat = buildSubFlapCatalog(dragonModel, dragonSf);
      const shinL = dragonCat.subFlaps.find((s) => s.stickName === 'hind_shin_L');
      const shinR = dragonCat.subFlaps.find((s) => s.stickName === 'hind_shin_R');
      const thighL = dragonCat.subFlaps.find((s) => s.stickName === 'hind_thigh_L');
      assert(shinL && shinR && thighL);
      assert.equal(findMirroredSubFlapId(dragonCat.subFlaps, shinL.id), shinR.id);

      const bentShin = applyFlapShaping(rawDragon, {{
        region: shinL.regionIdx,
        fracStart: shinL.fracStart,
        fracEnd: shinL.fracEnd,
        mirror: true,
        mode: 'bend',
        pitchDeg: 35,
        depth: 0.4,
      }});
      const bentShinModel = parseFold(bentShin.foldDict);
      assert(bentShinModel.folded !== null);
      assert(strainPercent(bentShinModel) < 4.0);

      const eagleRegs = rawEagle['file_bp:regions'];
      const wingLIdx = eagleRegs.findIndex((r) => r.sticks.includes('wing_L'));
      const wingRIdx = eagleRegs.findIndex((r) => r.sticks.includes('wing_R'));
      const tailIdx = eagleRegs.findIndex((r) => r.sticks.includes('tail'));
      assert.equal(findMirroredRegionIndex(eagleRegs, wingLIdx), wingRIdx);
      assert.equal(findMirroredRegionIndex(eagleRegs, tailIdx), null);
      assert.equal(formatRegionName(eagleRegs, wingLIdx), 'wing L');

      const countMv = (fd) =>
          fd.edges_assignment.filter((a) => a === 'M' || a === 'V').length;

      const bentWings = applyFlapShaping(rawEagle, {{
        region: wingLIdx,
        mirror: true,
        mode: 'bend',
        pitchDeg: 40,
        depth: 0.35,
      }});
      assert.deepEqual(bentWings.appliedRegions, [wingLIdx, wingRIdx]);
      assert(countMv(bentWings.foldDict) > countMv(rawEagle));
      const bentModel = parseFold(bentWings.foldDict);
      assert(bentModel.folded !== null);
      assert(strainPercent(bentModel) < 2.0);
      const bentFolder = new Folder(bentModel);
      bentFolder.setSchedule(buildFoldingSchedule(bentModel, sf));
      bentFolder.positionsAt(0, true);
      bentFolder.positionsAt(1, true);

      const dragonRegs = rawDragon['file_bp:regions'];
      const dTail = dragonRegs.findIndex((r) => r.sticks.includes('tail_whip'));
      const dWingL = dragonRegs.findIndex((r) => r.sticks.includes('wing_L'));

      const curledDragon = applyFlapShaping(rawDragon, {{
        region: dTail,
        mirror: false,
        mode: 'curve',
        pitchDeg: 75,
        depth: 0.2,
        steps: 3,
      }});
      assert(countMv(curledDragon.foldDict) > countMv(rawDragon));

      const pleatedDragon = applyFlapShaping(curledDragon.foldDict, {{
        region: dWingL,
        mirror: true,
        mode: 'pleat',
        pitchDeg: 35,
        depth: 0.3,
        steps: 2,
      }});
      assert(countMv(pleatedDragon.foldDict) > countMv(curledDragon.foldDict));

      const crimpedDragon = applyFlapShaping(pleatedDragon.foldDict, {{
        region: dWingL,
        mirror: true,
        mode: 'crimp',
        turnDeg: 30,
        depth: 0.25,
      }});
      assert(countMv(crimpedDragon.foldDict) > countMv(pleatedDragon.foldDict));

      const narrowedDragon = applyFlapShaping(crimpedDragon.foldDict, {{
        region: dWingL,
        mirror: true,
        mode: 'narrow',
        narrowPct: 55,
        tuckDeg: 165,
        depth: 0.15,
        bothSides: true,
      }});
      assert(countMv(narrowedDragon.foldDict) > countMv(crimpedDragon.foldDict));

      const shapedDragonModel = parseFold(narrowedDragon.foldDict);
      assert(shapedDragonModel.folded !== null);
      assert(strainPercent(shapedDragonModel) < 3.0);

      // 4. Exhaustive guarantee: every selectable flap, sub-flap, and river
      // adds real M/V creases (`addedCreases > 0` & `edges_bp:userShaping`)
      // and preserves a valid 3D folded frame (`parsed.folded !== null`).
      const flapModes = [
        {{mode: 'bend', pitchDeg: 35, depth: 0.30}},
        {{
          mode: 'narrow',
          narrowPct: 55,
          tuckDeg: 165,
          depth: 0.12,
          bothSides: true,
        }},
        {{
          mode: 'narrow',
          narrowPct: 55,
          tuckDeg: 165,
          depth: 0.12,
          bothSides: false,
        }},
        {{mode: 'crimp', turnDeg: 35, depth: 0.25}},
        {{mode: 'curve', pitchDeg: 60, depth: 0.20, steps: 3}},
        {{mode: 'pleat', pitchDeg: 35, depth: 0.25, steps: 2}},
      ];
      for (const [label, rawDoc, cat] of [
        ['golden-eagle', rawEagle, eagleCat],
        ['wyrmlord-dragon', rawDragon, dragonCat],
      ]) {{
        for (const sub of cat.subFlaps) {{
          for (const opts of flapModes) {{
            if (sub.kind === 'river' && opts.mode === 'narrow') continue;
            const shaped = applyFlapShaping(rawDoc, {{
              region: sub.regionIdx,
              fracStart: sub.fracStart,
              fracEnd: sub.fracEnd,
              mirror: false,
              ...opts,
            }});
            const tag = `${{label}}/${{sub.stickName}} (${{opts.mode}})`;
            assert(shaped.addedCreases > 0, `0 creases added on ${{tag}}`);
            const userFlags = shaped.foldDict['edges_bp:userShaping'] || [];
            assert(userFlags.some(Boolean), `missing userShaping on ${{tag}}`);
            const parsed = parseFold(shaped.foldDict);
            assert(parsed.folded !== null, `high strain rejected ${{tag}}`);
          }}
        }}
      }}

      // Single-triangle distal sub-flap (`winged-pegasus/hind_cannon_L`) and
      // curled C-tail sub-flap + enclosed seam exclusion (`emperor-scorpion`).
      const rawPegasus = JSON.parse(
          fs.readFileSync({json.dumps(str(pegasus_fold))}, 'utf8'));
      const pegasusSf = JSON.parse(
          fs.readFileSync({json.dumps(str(pegasus_sf))}, 'utf8'));
      const pegasusCat = buildSubFlapCatalog(parseFold(rawPegasus), pegasusSf);
      const hindCannon = pegasusCat.subFlaps.find(
          (s) => s.stickName === 'hind_cannon_L');
      assert(hindCannon);
      for (const opts of flapModes) {{
        const res = applyFlapShaping(rawPegasus, {{
          region: hindCannon.regionIdx,
          fracStart: hindCannon.fracStart,
          fracEnd: hindCannon.fracEnd,
          mirror: true,
          ...opts,
        }});
        assert(res.addedCreases > 0);
        assert(parseFold(res.foldDict).folded !== null);
      }}

      const rawScorp = JSON.parse(
          fs.readFileSync({json.dumps(str(scorp_fold))}, 'utf8'));
      const scorpSf = JSON.parse(
          fs.readFileSync({json.dumps(str(scorp_sf))}, 'utf8'));
      const scorpCat = buildSubFlapCatalog(parseFold(rawScorp), scorpSf);
      assert.equal(
          scorpCat.subFlaps.find((s) => s.stickName === 'body_front'),
          undefined);
      const tail4 = scorpCat.subFlaps.find((s) => s.stickName === 'tail_4');
      assert(tail4);
      for (const opts of flapModes) {{
        const res = applyFlapShaping(rawScorp, {{
          region: tail4.regionIdx,
          fracStart: tail4.fracStart,
          fracEnd: tail4.fracEnd,
          mirror: false,
          ...opts,
        }});
        assert(res.addedCreases > 0);
        assert(parseFold(res.foldDict).folded !== null);
      }}

      // 5. Verify 2D SVG `drawCreasePattern` renders bold `userShaping` paths
      // and zooms `viewBox` onto `focusFaces`.
      const makeFakeEl = (tag) => ({{
        tag,
        attrs: {{}},
        dataset: {{}},
        children: [],
        textContent: '',
        setAttribute(k, v) {{ this.attrs[k] = String(v); }},
        append(...kids) {{ this.children.push(...kids); }},
        replaceChildren(...kids) {{ this.children = [...kids]; }},
      }});
      globalThis.document = {{
        createElementNS(_ns, tag) {{ return makeFakeEl(tag); }},
      }};
      const {{drawCreasePattern, drawPacking}} = await import('./diagram.js');
      const fullSvg = makeFakeEl('svg');
      drawCreasePattern(fullSvg, eagle, {{subFlaps: eagleCat.subFlaps}});
      const fullSpan = Number(fullSvg.attrs.viewBox.split(' ')[2]);

      // Verify `'reserved'` regions render with `#d8d4cc` and cannot be picked
      // or shaped (`buildSubFlapCatalog` + `applyFlapShaping` + `drawPacking`).
      const withReservedRaw = structuredClone(rawEagle);
      const resIdx = withReservedRaw['file_bp:regions'].length;
      withReservedRaw['file_bp:regions'].push({{
        kind: 'reserved',
        sticks: ['_reserved_pocket'],
      }});
      withReservedRaw['faces_bp:region'][0] = resIdx;
      const withReservedModel = parseFold(withReservedRaw);
      const withReservedCat = buildSubFlapCatalog(withReservedModel, sf);
      assert.equal(withReservedCat.faceSubFlap[0], -1);
      assert.equal(
          withReservedCat.subFlaps.find((s) => s.kind === 'reserved'),
          undefined);
      for (const opts of flapModes) {{
        const resAttempt = applyFlapShaping(withReservedRaw, {{
          region: resIdx,
          mirror: true,
          ...opts,
        }});
        assert.equal(resAttempt.addedCreases, 0);
        assert.deepEqual(resAttempt.appliedRegions, []);
      }}
      const packSvg = makeFakeEl('svg');
      drawPacking(packSvg, withReservedModel, null, {{
        subFlaps: withReservedCat.subFlaps,
      }});
      const reservedPaths = packSvg.children[0].children.filter(
          (c) => c.attrs.fill === '#d8d4cc');
      assert.equal(reservedPaths.length, 1);
      assert.equal(reservedPaths[0].dataset.subflap, undefined);
      assert.equal(reservedPaths[0].dataset.region, undefined);

      const shapedWing = applyFlapShaping(rawEagle, {{
        region: wingLIdx,
        mirror: true,
        mode: 'narrow',
        narrowPct: 55,
        tuckDeg: 165,
        depth: 0.12,
        bothSides: true,
      }});
      const shapedWingModel = parseFold(shapedWing.foldDict);
      const shapedEagleCat = buildSubFlapCatalog(shapedWingModel, sf);
      const talonSub = shapedEagleCat.subFlaps.find(
          (s) => s.stickName === 'talon_L');
      const zoomSvg = makeFakeEl('svg');
      drawCreasePattern(zoomSvg, shapedWingModel, {{
        highlightFaces: new Set(talonSub.faceIndices),
        focusFaces: new Set(talonSub.faceIndices),
        userShaping: shapedWing.foldDict['edges_bp:userShaping'],
        subFlaps: shapedEagleCat.subFlaps,
      }});
      const zoomSpan = Number(zoomSvg.attrs.viewBox.split(' ')[2]);
      assert(zoomSpan < fullSpan * 0.40, 'expected zoomed viewBox on talon_L');
      const haloPaths = zoomSvg.children[0].children.filter(
          (c) => c.attrs['stroke-width'] === '5');
      assert.equal(haloPaths.length, 1, 'expected bold halo on user creases');
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


if __name__ == '__main__':
  absltest.main()
