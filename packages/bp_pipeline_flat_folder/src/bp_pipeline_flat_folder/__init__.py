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

"""C++20 global flat-foldability checker and exact rational SAT checker.

Installing `bp-pipeline-flat-folder` alone does not modify `bp_pipeline`
defaults; call the functions and `FlatFolderSession` exported here by name.
"""

from . import fast_flat_folder
from . import sat_certificate_checker

BackendName = fast_flat_folder.BackendName
FastFlatFolderParityError = fast_flat_folder.FastFlatFolderParityError
FastFoldResult = fast_flat_folder.FastFoldResult
FlatFoldSolution = fast_flat_folder.FlatFoldSolution
FlatFolderOutput = fast_flat_folder.FlatFolderOutput
FlatFolderSession = fast_flat_folder.FlatFolderSession
FoldVerdict = fast_flat_folder.FoldVerdict

analyze_fold_file = fast_flat_folder.analyze_fold_file
check_cp = fast_flat_folder.check_cp
compute_folded_vertices_cp = fast_flat_folder.compute_folded_vertices_cp
solve_flat_folded_state = fast_flat_folder.solve_flat_folded_state

__all__ = (
    'BackendName',
    'FastFlatFolderParityError',
    'FastFoldResult',
    'FlatFoldSolution',
    'FlatFolderOutput',
    'FlatFolderSession',
    'FoldVerdict',
    'analyze_fold_file',
    'check_cp',
    'compute_folded_vertices_cp',
    'fast_flat_folder',
    'sat_certificate_checker',
    'solve_flat_folded_state',
)
