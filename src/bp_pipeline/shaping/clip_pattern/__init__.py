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
"""Folded-grid clip-pattern shaping for box-pleated flaps and rivers."""

from .grid_shaper import add_shaping_creases
from .grid_shaper import CellTriangle
from .grid_shaper import ClipPatternShaper
from .grid_shaper import EdgeKind
from .grid_shaper import find_align_triangles
from .grid_shaper import FoldGrid
from .grid_shaper import shape_clip_pattern
from .grid_shaper import TriangleKey
from .templates import bend
from .templates import bend_when_turn_and_narrow
from .templates import build_fold_pattern
from .templates import build_template_pattern
from .templates import narrow
from .templates import normalize_angle
from .templates import PatternSegment
from .templates import pleat_to_hide
from .templates import rotate
from .templates import TemplateName
from .templates import turn_and_narrow

__all__ = [
    'CellTriangle',
    'ClipPatternShaper',
    'EdgeKind',
    'FoldGrid',
    'PatternSegment',
    'TemplateName',
    'TriangleKey',
    'add_shaping_creases',
    'bend',
    'bend_when_turn_and_narrow',
    'build_fold_pattern',
    'build_template_pattern',
    'find_align_triangles',
    'narrow',
    'normalize_angle',
    'pleat_to_hide',
    'rotate',
    'shape_clip_pattern',
    'turn_and_narrow',
]
