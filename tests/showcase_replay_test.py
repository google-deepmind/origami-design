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

"""End-to-end replay tests for showcase models with `shaping_history.json`."""

import json
import pathlib

from absl.testing import absltest
from absl.testing import parameterized
from bp_pipeline import cli

_DOCS_DIR = pathlib.Path(__file__).parents[1] / 'docs'
_MODELS_DIR = _DOCS_DIR / 'models'
_CATALOG = json.loads(
    (_MODELS_DIR / 'catalog.json').read_text(encoding='utf-8')
)


def _reproducible_models():
  """Yields catalog entries that record `shaping_history.json`."""
  for entry in _CATALOG:
    if (_MODELS_DIR / entry['slug'] / 'shaping_history.json').is_file():
      yield entry['slug'].replace('-', '_'), entry


class ShowcaseReplayTest(parameterized.TestCase):

  @parameterized.named_parameters(_reproducible_models())
  def test_recorded_history_reproduces_shaped_3d_fold(self, entry):
    slug = entry['slug']
    model_dir = _MODELS_DIR / slug
    out_dir = pathlib.Path(self.create_tempdir().full_path)

    pipeline_rc = cli.main([
        str(model_dir / 'stick_figure.json'),
        '-o',
        str(out_dir),
        '--no-viz',
        *entry.get('pipeline_args', ()),
    ])
    self.assertIn(pipeline_rc, (0, None))

    actuate_rc = cli.main([
        'actuate',
        str(out_dir),
        '--no-viz',
        '--reset',
        '--calls',
        str(model_dir / 'shaping_history.json'),
    ])
    self.assertIn(actuate_rc, (0, None))

    replayed = json.loads(
        (out_dir / 'shaped_3d.fold').read_text(encoding='utf-8')
    )
    expected = json.loads(
        (model_dir / 'shaped_3d.fold').read_text(encoding='utf-8')
    )
    self.assertEqual(replayed, expected)


if __name__ == '__main__':
  absltest.main()
