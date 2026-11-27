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

"""Pytest hooks for absl.flags and in-tree package/fixture discovery."""

import importlib.util
import pathlib
import sys
from typing import Any

from absl import flags


def _ensure_test_paths() -> None:
  """Adds in-tree source and shared test-fixture dirs when not installed."""
  tests_dir = pathlib.Path(__file__).absolute().parent
  pkg_root = tests_dir.parent
  repo_root = pkg_root.parent.parent

  # Only add source trees when packages are not already installed in the active
  # environment, so wheel-installed C++ extensions in site-packages are never
  # shadowed by raw `src/` directories.
  if importlib.util.find_spec('bp_pipeline_flat_folder') is None:
    ff_src = pkg_root / 'src'
    if (ff_src / 'bp_pipeline_flat_folder').is_dir():
      sys.path.insert(0, str(ff_src))

  if importlib.util.find_spec('bp_pipeline') is None:
    bp_src = repo_root / 'src'
    if (bp_src / 'bp_pipeline').is_dir():
      sys.path.insert(0, str(bp_src))

  if importlib.util.find_spec('cp_test_fixtures') is None:
    shared_tests = repo_root / 'tests'
    if (shared_tests / 'cp_test_fixtures.py').is_file():
      sys.path.insert(0, str(shared_tests))


def pytest_configure(config: Any) -> None:
  del config
  if not flags.FLAGS.is_parsed():
    flags.FLAGS.mark_as_parsed()
  _ensure_test_paths()
