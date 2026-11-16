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

"""Utility functions for combined heuristics."""

import collections
from typing import Sequence

from . import geometric_fold_3d_heuristics
from . import heuristics


def get_ancestor_chain(
    flap_info_dict: dict[str, heuristics.Node],
    ancestor_name: str,
) -> Sequence[str]:
  """Returns the ancestor chain for the given element."""
  ancestor_chain = []
  for _ in range(len(flap_info_dict) + 1):
    if ancestor_name is None:
      break
    ancestor_chain.append(ancestor_name)
    ancestor_name = flap_info_dict[ancestor_name].parent
  assert ancestor_name is None, 'Ancestor name is not None.'
  return ancestor_chain


def sort_creases_info_by_narrowed_ancestor(
    creases_and_info: list[geometric_fold_3d_heuristics.ElementShapingResult],
    flap_info: dict[str, heuristics.Node],
    narrowed_element_names: set[str],
) -> dict[str, list[geometric_fold_3d_heuristics.ElementShapingResult]]:
  """Sorts creases info by narrowed ancestor."""
  element_name_to_creases_and_info_map = collections.defaultdict(list)
  for info in creases_and_info:
    element_name = info['shaping_element'].name
    ancestor_names = set(get_ancestor_chain(flap_info, element_name))
    narrowed_ancestor_name = ancestor_names.intersection(narrowed_element_names)
    assert len(narrowed_ancestor_name) == 1, (
        'Expected one narrowed ancestor, but found'
        f' {len(narrowed_ancestor_name)}.'
        f' ancestor_names: {ancestor_names}'
        f' element_name: {element_name}'
    )
    narrowed_ancestor_name = narrowed_ancestor_name.pop()
    element_name_to_creases_and_info_map[narrowed_ancestor_name].append(info)

  return element_name_to_creases_and_info_map
