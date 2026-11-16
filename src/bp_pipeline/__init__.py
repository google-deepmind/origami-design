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

"""Automated Box-Pleating Origami Packing and Crease Pattern Solving."""

import importlib
import typing
from typing import Any

if typing.TYPE_CHECKING:
  from . import bps_types
  from . import cli
  from . import rectangle
  from . import render_3d
  from . import render_summary
  from . import timeout
  from . import utils_tree
  from .api import *  # pylint: disable=wildcard-import

_LAZY_SUBMODULES = frozenset({
    'bps_types',
    'cli',
    'rectangle',
    'render_3d',
    'render_summary',
    'timeout',
    'utils_tree',
})
_API_EXPORTS = frozenset({
    'BpsImportResult',
    'Crease',
    'CreasePattern',
    'DEFAULT_SOLVER_CONFIG',
    'DEFAULT_SVG_COLORS',
    'DesignTree',
    'Fold',
    'FoldStrategy3D',
    'FoldedMesh3D',
    'FoldedModel',
    'PRODUCTION_SWEEP',
    'PackedCreasePattern',
    'PackerOptions',
    'PackingConfig',
    'PackingFailedError',
    'PackingTaskConfig',
    'PipelineConfig',
    'PipelineResult',
    'Preset',
    'ShapedCreasePattern',
    'ShapingFailedError',
    'ShapingState',
    'ShapingStrategy',
    'ShapingTaskConfig',
    'ShapingToolConfig',
    'SolvedCreasePattern',
    'SolverConfig',
    'SolvingFailedError',
    'SolvingTaskConfig',
    'StackLayoutConfig',
    'StageSnapshot',
    'SolveStage',
    'Stick',
    'StickFigure',
    'ToolResult',
    'apply_shaping_tool',
    'apply_shaping_tools',
    'build_folded_mesh_3d',
    'dumps',
    'evaluate_fidelity',
    'fold',
    'from_bp_studio',
    'from_dict',
    'from_fold',
    'import_bps',
    'list_examples',
    'load_example',
    'list_packers',
    'list_shaping_tools',
    'list_solvers',
    'loads',
    'mesh_to_fold_dict',
    'pack',
    'pose',
    'prepare',
    'probe_shaping',
    'register_packer',
    'register_solver',
    'render_packed_overlay_svg',
    'render_packing_grid',
    'render_stick_figure_svg',
    'save_3d_turnaround',
    'save_bp_studio',
    'save_fold',
    'save_pipeline_summary',
    'save_stick_figure_views',
    'save_svg',
    'shape',
    'solve',
    'solve_best',
    'solve_sweep',
    'stack',
    'stage_display_name',
    'to_bp_studio',
    'to_dict',
    'to_fold',
    'to_svg',
})


def __getattr__(name: str) -> Any:  # pylint: disable=invalid-name
  if name in _LAZY_SUBMODULES:
    value = importlib.import_module(f'.{name}', __name__)
    globals()[name] = value
    return value
  if name in _API_EXPORTS:
    api_mod = importlib.import_module('.api', __name__)
    value = getattr(api_mod, name)
    globals()[name] = value
    return value
  raise AttributeError(f'module {__name__!r} has no attribute {name!r}')


def __dir__() -> list[str]:  # pylint: disable=invalid-name
  return sorted(set(globals()) | _LAZY_SUBMODULES | _API_EXPORTS)
