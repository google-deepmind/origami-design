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
import pathlib
import re

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


if __name__ == '__main__':
  absltest.main()
