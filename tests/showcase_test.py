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

"""Tests for curated origami showcase models and static GitHub Pages studio."""

import json
import math
from pathlib import Path

from bp_pipeline import render_3d
from bp_pipeline.crease_pattern.parser import svg as svg_parser
from bp_pipeline.stick_figures import stick_figure as sf_lib

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = REPO_ROOT / 'docs'
CATALOG_PATH = DOCS_DIR / 'models' / 'catalog.json'


def test_showcase_studio_static_files_exist():
  assert (DOCS_DIR / '.nojekyll').is_file()
  for rel in ('index.html', 'style.css', 'app.js', 'models/catalog.json'):
    path = DOCS_DIR / rel
    assert path.is_file(), f'Missing required studio asset: {path}'
    assert path.stat().st_size > 100


def test_showcase_catalog_and_model_bundles():
  catalog = json.loads(CATALOG_PATH.read_text(encoding='utf-8'))
  assert len(catalog) == 10

  seen_slugs = set()
  for entry in catalog:
    slug = entry['slug']
    assert slug not in seen_slugs
    seen_slugs.add(slug)

    # 1. Verify StickFigure JSON loads cleanly via bp_pipeline
    sf_path = DOCS_DIR / entry['files']['stickFigure']
    assert sf_path.is_file()
    sf_dict = json.loads(sf_path.read_text(encoding='utf-8'))
    tree = sf_lib.StickFigure.from_dict(sf_dict)
    assert len(tree.children) == entry['sticks']

    # 2. Verify all pipeline SVGs exist and validate cleanly
    for svg_key in ('packingSvg', 'solvingSvg', 'shapedSvg'):
      svg_path = DOCS_DIR / entry['files'][svg_key]
      assert svg_path.is_file(), f'Missing {svg_key} for {slug}'
      svg_text = svg_path.read_text(encoding='utf-8')
      assert svg_parser.valid_svg(svg_text), f'Invalid SVG {svg_key} for {slug}'

    # 3. Verify FOLD v1.2 export has both 2D CP and 3D upright frame
    fold_path = DOCS_DIR / entry['files']['fold']
    assert fold_path.is_file()
    fold_data = json.loads(fold_path.read_text(encoding='utf-8'))
    assert fold_data['file_spec'] == 1.2
    assert len(fold_data['vertices_coords']) == entry['vertices']
    assert len(fold_data['faces_vertices']) == entry['faces']
    assert len(fold_data['file_frames']) == 1
    frame_3d = fold_data['file_frames'][0]['vertices_coords']
    assert len(frame_3d) == entry['vertices']
    assert all(len(v) == 3 for v in frame_3d)

    # 4. Verify web bundle.json integrity & unit upright quaternion
    bundle_path = DOCS_DIR / entry['bundleUrl']
    assert bundle_path.is_file()
    bundle = json.loads(bundle_path.read_text(encoding='utf-8'))
    quat = bundle['uprightQuat']
    assert len(quat) == 4
    q_norm = math.sqrt(sum(c * c for c in quat))
    assert abs(q_norm - 1.0) < 1e-3

    # 5. Verify preview thumbnail exists
    preview_path = DOCS_DIR / entry['preview']
    assert preview_path.is_file()
    assert preview_path.stat().st_size > 5_000

  # 6. Verify bp_pipeline 3D hinge folder folds representative showcase SVG
  eagle_svg = (DOCS_DIR / 'models/golden-eagle/shaped_cp.svg').read_text(
      encoding='utf-8'
  )
  cp = svg_parser.parse_svg(eagle_svg)
  folded_mesh = render_3d.build_folded_mesh_3d(cp, tol=1e-4)
  assert len(folded_mesh.faces) > 50
