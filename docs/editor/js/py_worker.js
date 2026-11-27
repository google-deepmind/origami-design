// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/**
 * @fileoverview Background Web Worker running `bp_pipeline` in Pyodide (WASM).
 *
 * All computation (file parsing, stick-figure preparation, box-pleating
 * packing, flat-foldable hinge solving, 3D shaping, global flat-foldability
 * verification, and deterministic 3D mesh folding) executes 100% locally in
 * the user's browser tab with zero server uploads or telemetry.
 */

/* global importScripts, loadPyodide */

const DEFAULT_PYODIDE_CDN = 'https://cdn.jsdelivr.net/pyodide/v0.28.0/full/';
const WHEEL_CANDIDATES = [
  '../../py/bp_pipeline-0.1.0-py3-none-any.whl',
  '../py/bp_pipeline-0.1.0-py3-none-any.whl',
  './py/bp_pipeline-0.1.0-py3-none-any.whl',
];

let pyodideReadyPromise = null;
let pyDispatch = null;

/**
 * @param {number} id
 * @param {string} stage
 * @param {string=} detail
 */
function postProgress(id, stage, detail = '') {
  self.postMessage({type: 'progress', id, stage, detail});
}

const PYTHON_BOOTSTRAP = `
import base64
import collections
from collections.abc import Iterable, Iterator, Mapping
import copy
import io
import json
import logging as _std_logging
import math
import re
import sys
import time
import types
import urllib.parse
import xml.etree.ElementTree as _StdET
import zipfile

# Lightweight pure-Python shims for tiny utility packages so bp_pipeline works
# with zero PyPI network calls (100% offline/airgapped once Pyodide loads).
if 'immutabledict' not in sys.modules:
  _immut_mod = types.ModuleType('immutabledict')
  class immutabledict(Mapping):
    def __init__(self, *args, **kwargs):
      self._d = dict(*args, **kwargs)
      self._hash = None
    def __getitem__(self, k):
      return self._d[k]
    def __iter__(self):
      return iter(self._d)
    def __len__(self):
      return len(self._d)
    def __repr__(self):
      return f'immutabledict({self._d!r})'
    def __hash__(self):
      if self._hash is None:
        self._hash = hash(frozenset(self._d.items()))
      return self._hash
    def __eq__(self, other):
      if isinstance(other, Mapping):
        return dict(self._d) == dict(other)
      return NotImplemented
    def __or__(self, other):
      if isinstance(other, Mapping):
        merged = dict(self._d)
        merged.update(other)
        return immutabledict(merged)
      return NotImplemented
    def copy(self):
      return immutabledict(self._d)
  _immut_mod.immutabledict = immutabledict
  sys.modules['immutabledict'] = _immut_mod

if 'more_itertools' not in sys.modules:
  _mi_mod = types.ModuleType('more_itertools')
  def all_unique(iterable, key=None):
    seen = set()
    seen_list = []
    for item in iterable:
      val = item if key is None else key(item)
      try:
        if val in seen:
          return False
        seen.add(val)
      except TypeError:
        if val in seen_list:
          return False
        seen_list.append(val)
    return True
  _mi_mod.all_unique = all_unique
  sys.modules['more_itertools'] = _mi_mod

if 'absl' not in sys.modules:
  _absl_mod = types.ModuleType('absl')
  _absl_log = types.ModuleType('absl.logging')
  _logger = _std_logging.getLogger('bp_pipeline')
  _absl_log.debug = _logger.debug
  _absl_log.info = _logger.info
  _absl_log.warning = _logger.warning
  _absl_log.warn = _logger.warning
  _absl_log.error = _logger.error
  _absl_log.exception = _logger.exception
  _absl_log.fatal = _logger.critical
  _absl_log.vlog = lambda level, msg, *a, **kw: None
  _absl_log.level_debug = lambda: False
  _absl_log.INFO = _std_logging.INFO
  _absl_log.WARNING = _std_logging.WARNING
  _absl_log.ERROR = _std_logging.ERROR
  _absl_log.DEBUG = _std_logging.DEBUG
  _absl_mod.logging = _absl_log
  sys.modules['absl'] = _absl_mod
  sys.modules['absl.logging'] = _absl_log
`;

const PYTHON_BRIDGE = `
import bp_pipeline as bp
from bp_pipeline import bus
from bp_pipeline import export as export_lib
from bp_pipeline import folder_3d
from bp_pipeline import render_3d
from bp_pipeline.crease_pattern import crease as crease_lib
from bp_pipeline.crease_pattern import crease_pattern as cp_lib
from bp_pipeline.crease_pattern import point as point_lib
from bp_pipeline.crease_pattern.parser import cp_format
from bp_pipeline.crease_pattern.parser import svg as svg_parser
from bp_pipeline.shaping import toolbox as toolbox_lib
try:
  from bp_pipeline.simulator.flat_folder import fold_cp
except ImportError:
  from bp_pipeline.third_party.flat_folder import fold_cp
from bp_pipeline.stick_figures import stick_figure as sf_lib

import numpy as np


TOOL_CATALOG = [
    {
        'name': 'simple_fold',
        'label': 'Simple Fold (2D Zigzag)',
        'group': 'Limb Bends',
        'params': [
            {'key': 'element_name', 'type': 'element', 'label': 'Flap / River'},
            {'key': 'angle', 'type': 'number', 'label': 'Angle (°)', 'default': 45, 'min': -175, 'max': 175, 'step': 5},
            {'key': 'depth', 'type': 'number', 'label': 'Depth [0..0.9]', 'default': 0.25, 'min': 0.0, 'max': 0.9, 'step': 0.05},
            {'key': 'fold_percent', 'type': 'number', 'label': 'Fold %', 'default': 0.75, 'min': 0.1, 'max': 1.0, 'step': 0.05},
        ],
    },
    {
        'name': 'crimp_fold',
        'label': 'Crimp Fold',
        'group': 'Limb Bends',
        'params': [
            {'key': 'element_name', 'type': 'element', 'label': 'Flap / River'},
            {'key': 'angle', 'type': 'number', 'label': 'Angle (°)', 'default': 35, 'min': -160, 'max': 160, 'step': 5},
            {'key': 'depth', 'type': 'number', 'label': 'Depth [0..0.9]', 'default': 0.3, 'min': 0.0, 'max': 0.9, 'step': 0.05},
            {'key': 'fold_order', 'type': 'select', 'label': 'Order', 'default': 'MV', 'options': ['MV', 'VM']},
        ],
    },
    {
        'name': 'reverse_fold',
        'label': 'Inside / Outside Reverse Fold',
        'group': 'Limb Bends',
        'params': [
            {'key': 'element_name', 'type': 'element', 'label': 'Flap'},
            {'key': 'angle', 'type': 'number', 'label': 'Angle (°)', 'default': 60, 'min': -170, 'max': 170, 'step': 5},
            {'key': 'depth', 'type': 'number', 'label': 'Depth [0..0.9]', 'default': 0.35, 'min': 0.0, 'max': 0.9, 'step': 0.05},
        ],
    },
    {
        'name': 'bend_to_direction',
        'label': '3D Bend to Azimuth / Elevation',
        'group': '3D Posing',
        'params': [
            {'key': 'element_name', 'type': 'element', 'label': 'Flap / River'},
            {'key': 'azimuth', 'type': 'number', 'label': 'Target Azimuth (°)', 'default': 30, 'min': -180, 'max': 180, 'step': 5},
            {'key': 'elevation', 'type': 'number', 'label': 'Target Elevation (°)', 'default': 25, 'min': -88, 'max': 88, 'step': 5},
            {'key': 'depth', 'type': 'number', 'label': 'Depth [0..0.9]', 'default': 0.2, 'min': 0.0, 'max': 0.9, 'step': 0.05},
        ],
    },
    {
        'name': 'fold_spine',
        'label': 'Fold Spine in Half (3D)',
        'group': '3D Posing',
        'params': [
            {'key': 'fold_percent', 'type': 'number', 'label': 'Spine Fold %', 'default': 0.65, 'min': 0.05, 'max': 1.0, 'step': 0.05},
            {'key': 'fold_type', 'type': 'select', 'label': 'Polarity', 'default': 'mountain', 'options': ['mountain', 'valley']},
        ],
    },
    {
        'name': 'narrow_central_flap',
        'label': 'Narrow Central Flap (Taper)',
        'group': 'Sculpt & Detail',
        'params': [
            {'key': 'element_name', 'type': 'element', 'label': 'Flap'},
            {'key': 'tip_width', 'type': 'number', 'label': 'Tip Width', 'default': 0.25, 'min': 0.05, 'max': 0.85, 'step': 0.05},
            {'key': 'start_offset', 'type': 'number', 'label': 'Start Offset', 'default': 0.0, 'min': 0.0, 'max': 0.7, 'step': 0.05},
            {'key': 'fold_type', 'type': 'select', 'label': 'Polarity', 'default': 'auto', 'options': ['auto', 'mountain', 'valley']},
        ],
    },
    {
        'name': 'fan_out_fold',
        'label': 'Fan Out Wing / Tail',
        'group': 'Sculpt & Detail',
        'params': [
            {'key': 'element_name', 'type': 'element', 'label': 'Flap'},
            {'key': 'num_pleats', 'type': 'number', 'label': 'Pleats', 'default': 3, 'min': 1, 'max': 8, 'step': 1},
            {'key': 'fold_percent', 'type': 'number', 'label': 'Fold %', 'default': 0.6, 'min': 0.1, 'max': 1.0, 'step': 0.05},
        ],
    },
    {
        'name': 'squash_fold',
        'label': 'Squash Fold',
        'group': 'Sculpt & Detail',
        'params': [
            {'key': 'element_name', 'type': 'element', 'label': 'Flap'},
            {'key': 'fold_percent', 'type': 'number', 'label': 'Fold %', 'default': 0.85, 'min': 0.1, 'max': 1.0, 'step': 0.05},
        ],
    },
    {
        'name': 'curve_flap',
        'label': 'Curve Flap (Multi-Crimp Arc)',
        'group': 'Sculpt & Detail',
        'params': [
            {'key': 'element_name', 'type': 'element', 'label': 'Flap'},
            {'key': 'total_angle', 'type': 'number', 'label': 'Total Arc (°)', 'default': 75, 'min': -180, 'max': 180, 'step': 5},
            {'key': 'steps', 'type': 'number', 'label': 'Segments', 'default': 3, 'min': 2, 'max': 5, 'step': 1},
        ],
    },
    {
        'name': 'radial_fan',
        'label': 'Radial Fan Pleats',
        'group': 'Sculpt & Detail',
        'params': [
            {'key': 'element_name', 'type': 'element', 'label': 'Flap'},
            {'key': 'rays', 'type': 'number', 'label': 'Rays', 'default': 4, 'min': 2, 'max': 8, 'step': 1},
            {'key': 'fold_percent', 'type': 'number', 'label': 'Fold %', 'default': 0.5, 'min': 0.1, 'max': 1.0, 'step': 0.05},
        ],
    },
    {
        'name': 'thin_limbs_3d',
        'label': 'Auto-Thin Width-1 Limbs (3D)',
        'group': '3D Posing',
        'params': [],
    },
    {
        'name': 'shape_flap_template',
        'label': 'Shape Flap Silhouette Template',
        'group': 'Sculpt & Detail',
        'params': [
            {'key': 'element_name', 'type': 'element', 'label': 'Flap'},
            {'key': 'template', 'type': 'select', 'label': 'Template', 'default': 'taper', 'options': ['taper', 'point', 'diamond', 'split_tip']},
        ],
    },
]


def _normalize_stroke_color(raw: str) -> str:
  if not raw:
    return 'gray'
  s = raw.strip().lower().replace(' ', '')
  if s in ('red', '#f00', '#ff0000', '#dc2626', '#ef4444') or s.startswith('rgb(255,0,0'):
    return 'red'
  if s in ('blue', '#00f', '#0000ff', '#2563eb', '#3b82f6') or s.startswith('rgb(0,0,255'):
    return 'blue'
  if s in ('black', '#000', '#000000', '#111827', '#1f2937') or s.startswith('rgb(0,0,0'):
    return 'black'
  if s in ('green', '#0f0', '#00ff00', '#16a34a') or s.startswith('rgb(0,255,0'):
    return 'green'
  if s in ('yellow', '#ff0', '#ffff00', 'gray', 'grey', '#888', '#808080', '#9ca3af'):
    return 'yellow'
  if s.startswith('#') and len(s) in (4, 7):
    try:
      if len(s) == 4:
        r, g, b = int(s[1]*2, 16), int(s[2]*2, 16), int(s[3]*2, 16)
      else:
        r, g, b = int(s[1:3], 16), int(s[3:5], 16), int(s[5:7], 16)
      if r > 160 and g < 90 and b < 90:
        return 'red'
      if b > 150 and r < 90:
        return 'blue'
      if r < 65 and g < 65 and b < 65:
        return 'black'
      if g > 140 and r < 90 and b < 90:
        return 'green'
    except ValueError:
      pass
  return 'yellow'


def _parse_svg_path_segments(d_attr: str) -> list[tuple[float, float, float, float]]:
  tokens = re.findall(r'[MmLlHhVvZz]|[-+]?(?:\\d+\\.?\\d*|\\.\\d+)(?:[eE][-+]?\\d+)?', d_attr or '')
  segs = []
  idx = 0
  cmd = ''
  cx, cy = 0.0, 0.0
  sx, sy = 0.0, 0.0
  while idx < len(tokens):
    tok = tokens[idx]
    if tok.isalpha():
      cmd = tok
      idx += 1
      if cmd in ('Z', 'z'):
        if math.hypot(cx - sx, cy - sy) > 1e-9:
          segs.append((cx, cy, sx, sy))
        cx, cy = sx, sy
      continue
    if cmd in ('M', 'm'):
      x, y = float(tokens[idx]), float(tokens[idx + 1])
      idx += 2
      if cmd == 'm':
        x += cx
        y += cy
      cx, cy = x, y
      sx, sy = x, y
      cmd = 'L' if cmd == 'M' else 'l'
    elif cmd in ('L', 'l'):
      x, y = float(tokens[idx]), float(tokens[idx + 1])
      idx += 2
      if cmd == 'l':
        x += cx
        y += cy
      if math.hypot(x - cx, y - cy) > 1e-9:
        segs.append((cx, cy, x, y))
      cx, cy = x, y
    elif cmd in ('H', 'h'):
      x = float(tokens[idx])
      idx += 1
      if cmd == 'h':
        x += cx
      if abs(x - cx) > 1e-9:
        segs.append((cx, cy, x, cy))
      cx = x
    elif cmd in ('V', 'v'):
      y = float(tokens[idx])
      idx += 1
      if cmd == 'v':
        y += cy
      if abs(y - cy) > 1e-9:
        segs.append((cx, cy, cx, y))
      cy = y
    else:
      idx += 1
  return segs


def _normalize_external_svg_to_cp(svg_text: str) -> cp_lib.CreasePattern:
  """Parses any SVG crease pattern (including Origami Simulator / Inkscape SVGs)."""
  lowered_svg = svg_text.lower()
  if '<!doctype' in lowered_svg or '<!entity' in lowered_svg:
    raise ValueError(
        'XML DOCTYPE and ENTITY declarations are not permitted in SVG files.'
    )
  try:
    return svg_parser.parse_svg(svg_text, repair=True, light_mode=True)
  except Exception:
    pass

  root = _StdET.fromstring(svg_text)
  raw_segs: list[tuple[float, float, float, float, str, float]] = []

  def walk(el, inherited_stroke='', inherited_opacity=1.0):
    style_str = el.attrib.get('style', '')
    style_map = {}
    for part in style_str.split(';'):
      if ':' in part:
        k, v = part.split(':', 1)
        style_map[k.strip().lower()] = v.strip()
    stroke = el.attrib.get('stroke') or style_map.get('stroke') or inherited_stroke
    op_str = (
        el.attrib.get('stroke-opacity')
        or style_map.get('stroke-opacity')
        or el.attrib.get('opacity')
        or style_map.get('opacity')
    )
    opacity = float(op_str) if op_str is not None else inherited_opacity
    tag = el.tag.split('}')[-1].lower()
    norm_color = _normalize_stroke_color(stroke)
    if tag == 'line':
      x1 = float(el.attrib.get('x1', '0'))
      y1 = float(el.attrib.get('y1', '0'))
      x2 = float(el.attrib.get('x2', '0'))
      y2 = float(el.attrib.get('y2', '0'))
      raw_segs.append((x1, y1, x2, y2, norm_color, opacity))
    elif tag == 'rect':
      rx = float(el.attrib.get('x', '0'))
      ry = float(el.attrib.get('y', '0'))
      rw = float(el.attrib.get('width', '0'))
      rh = float(el.attrib.get('height', '0'))
      if rw > 0 and rh > 0:
        c = norm_color if stroke else 'black'
        raw_segs.extend([
            (rx, ry, rx + rw, ry, c, 1.0),
            (rx + rw, ry, rx + rw, ry + rh, c, 1.0),
            (rx + rw, ry + rh, rx, ry + rh, c, 1.0),
            (rx, ry + rh, rx, ry, c, 1.0),
        ])
    elif tag in ('polyline', 'polygon'):
      pts = [float(v) for v in re.findall(r'[-+]?(?:\\d+\\.?\\d*|\\.\\d+)', el.attrib.get('points', ''))]
      pairs = [(pts[i], pts[i + 1]) for i in range(0, len(pts) - 1, 2)]
      for i in range(len(pairs) - 1):
        raw_segs.append((pairs[i][0], pairs[i][1], pairs[i+1][0], pairs[i+1][1], norm_color, opacity))
      if tag == 'polygon' and len(pairs) >= 3:
        raw_segs.append((pairs[-1][0], pairs[-1][1], pairs[0][0], pairs[0][1], norm_color, opacity))
    elif tag == 'path':
      for x1, y1, x2, y2 in _parse_svg_path_segments(el.attrib.get('d', '')):
        raw_segs.append((x1, y1, x2, y2, norm_color, opacity))
    for child in el:
      walk(child, stroke, opacity)

  walk(root)
  if not raw_segs:
    raise ValueError('No line, path, or rect segments found in SVG.')

  xs = [x for seg in raw_segs for x in (seg[0], seg[2])]
  ys = [y for seg in raw_segs for y in (seg[1], seg[3])]
  min_x, max_x = min(xs), max(xs)
  min_y, max_y = min(ys), max(ys)
  span = max(max_x - min_x, max_y - min_y)
  if span <= 1e-9:
    raise ValueError('Degenerate zero-area SVG crease pattern.')

  color_to_fold = {
      'black': crease_lib.Fold.BORDER,
      'red': crease_lib.Fold.MOUNTAIN,
      'blue': crease_lib.Fold.VALLEY,
      'green': crease_lib.Fold.HINGE,
      'yellow': crease_lib.Fold.TRIANGLE,
  }
  creases = []
  for x1, y1, x2, y2, col, op in raw_segs:
    ux1 = min(1.0, max(0.0, (x1 - min_x) / span))
    uy1 = min(1.0, max(0.0, (y1 - min_y) / span))
    ux2 = min(1.0, max(0.0, (x2 - min_x) / span))
    uy2 = min(1.0, max(0.0, (y2 - min_y) / span))
    if math.hypot(ux2 - ux1, uy2 - uy1) <= 1e-6:
      continue
    fold_enum = color_to_fold.get(col, crease_lib.Fold.TRIANGLE)
    pct = max(0.0, min(1.0, float(op))) if fold_enum in (crease_lib.Fold.MOUNTAIN, crease_lib.Fold.VALLEY) else 1.0
    creases.append(
        crease_lib.Crease(
            point_lib.Point(x=ux1, y=uy1, require_on_paper=False),
            point_lib.Point(x=ux2, y=uy2, require_on_paper=False),
            fold=fold_enum,
            fold_percent=pct,
        )
    )
  return cp_lib.CreasePattern(creases, light_mode=True)


def _normalize_cp_file_to_cp(cp_text: str) -> cp_lib.CreasePattern:
  """Parses ORIPA .cp text with any coordinate bounding box (e.g. [-200, 200] or [0, 1])."""
  rows = []
  for raw_line in cp_text.splitlines():
    line = raw_line.strip()
    if not line or line.startswith('#') or line.startswith('//'):
      continue
    parts = line.split()
    if len(parts) < 5:
      continue
    code = int(float(parts[0]))
    x1, y1, x2, y2 = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
    rows.append((code, x1, y1, x2, y2))
  if not rows:
    raise ValueError('No valid segments found in .cp file.')
  xs = [v for r in rows for v in (r[1], r[3])]
  ys = [v for r in rows for v in (r[2], r[4])]
  min_x, max_x = min(xs), max(xs)
  min_y, max_y = min(ys), max(ys)
  span = max(max_x - min_x, max_y - min_y)
  if span <= 1e-9:
    raise ValueError('Degenerate .cp coordinate range.')
  code_map = {
      1: crease_lib.Fold.BORDER,
      2: crease_lib.Fold.MOUNTAIN,
      3: crease_lib.Fold.VALLEY,
      4: crease_lib.Fold.TRIANGLE,
  }
  creases = []
  has_border = False
  for code, x1, y1, x2, y2 in rows:
    ux1 = min(1.0, max(0.0, (x1 - min_x) / span))
    uy1 = min(1.0, max(0.0, (y1 - min_y) / span))
    ux2 = min(1.0, max(0.0, (x2 - min_x) / span))
    uy2 = min(1.0, max(0.0, (y2 - min_y) / span))
    if math.hypot(ux2 - ux1, uy2 - uy1) <= 1e-6:
      continue
    fold_enum = code_map.get(code, crease_lib.Fold.TRIANGLE)
    if fold_enum == crease_lib.Fold.BORDER:
      has_border = True
    creases.append(
        crease_lib.Crease(
            point_lib.Point(x=ux1, y=uy1, require_on_paper=False),
            point_lib.Point(x=ux2, y=uy2, require_on_paper=False),
            fold=fold_enum,
        )
    )
  if not has_border:
    for bx1, by1, bx2, by2 in ((0,0,1,0),(1,0,1,1),(1,1,0,1),(0,1,0,0)):
      creases.append(
          crease_lib.Crease(
              point_lib.Point(x=float(bx1), y=float(bx1 * 0 + by1)),
              point_lib.Point(x=float(bx2), y=float(by2)),
              fold=crease_lib.Fold.BORDER,
          )
      )
  return cp_lib.CreasePattern(creases, light_mode=True)


def _fold_arbitrary_2d_graph_to_fold_dict(
    vertices_2d: list[list[float]],
    edges: list[list[int]],
    assignments: list[str],
    fold_angles: list[float] | None = None,
    *,
    title: str = 'Custom Crease Pattern',
    grid_size: int | None = None,
    regions: list[dict] | None = None,
    base_fold_angles: list[float] | None = None,
    shaping_flags: list[bool] | None = None,
) -> dict:
  """Splits crossings, extracts bounded faces, and folds any 2D graph in 3D.

  Coordinates 'vertices_2d' are in Studio's FOLD flat frame ('[0, 1]^2' with
  '+y' up; if outside '[0, 1]^2', normalized automatically preserving aspect).
  """
  if not vertices_2d or not edges:
    raise ValueError('Crease pattern needs vertices and edges.')

  xs = [float(p[0]) for p in vertices_2d]
  ys = [float(p[1]) for p in vertices_2d]
  min_x, max_x = min(xs), max(xs)
  min_y, max_y = min(ys), max(ys)
  span = max(max_x - min_x, max_y - min_y)
  if span <= 1e-9:
    raise ValueError('Degenerate zero-span vertex coordinates.')

  need_norm = min_x < -1e-4 or min_y < -1e-4 or max_x > 1.0001 or max_y > 1.0001
  norm_pts = [
      (
          (float(p[0]) - min_x) / span if need_norm else float(p[0]),
          (float(p[1]) - min_y) / span if need_norm else float(p[1]),
      )
      for p in vertices_2d
  ]

  # Convert FOLD flat frame (x_f, y_f) -> unit paper (1 - x_f, 1 - y_f) used by
  # folder_3d & render_3d so handedness matches mesh_to_fold_dict 1:1.
  tol = 1e-5
  v_index = folder_3d._SpatialVertexIndex(tol)
  raw_records = []
  angle_by_default = {'M': -180.0, 'V': 180.0, 'B': 0.0, 'F': 0.0, 'U': 0.0, 'C': 0.0}
  for e_idx, (u_raw, v_raw) in enumerate(edges):
    u0, v0 = int(u_raw), int(v_raw)
    if u0 == v0 or u0 < 0 or v0 < 0 or u0 >= len(norm_pts) or v0 >= len(norm_pts):
      continue
    ux1, uy1 = 1.0 - norm_pts[u0][0], 1.0 - norm_pts[u0][1]
    ux2, uy2 = 1.0 - norm_pts[v0][0], 1.0 - norm_pts[v0][1]
    if math.hypot(ux2 - ux1, uy2 - uy1) <= tol:
      continue
    vi1 = v_index.add(ux1, uy1)
    vi2 = v_index.add(ux2, uy2)
    if vi1 == vi2:
      continue
    raw_a = str(assignments[e_idx]).upper() if e_idx < len(assignments) else 'F'
    assign = raw_a if raw_a in ('B', 'M', 'V', 'F', 'U') else 'F'
    if fold_angles is not None and e_idx < len(fold_angles) and fold_angles[e_idx] is not None:
      ang = float(fold_angles[e_idx])
    else:
      ang = angle_by_default.get(assign, 0.0)
    raw_records.append(folder_3d._EdgeRecord(vi1, vi2, assign, ang))

  # Ensure outer border exists if the user only drew interior creases.
  has_border = any(r.assignment == 'B' for r in raw_records)
  if not has_border:
    b0 = v_index.add(0.0, 0.0)
    b1 = v_index.add(1.0, 0.0)
    b2 = v_index.add(1.0, 1.0)
    b3 = v_index.add(0.0, 1.0)
    for p_a, p_b in ((b0, b1), (b1, b2), (b2, b3), (b3, b0)):
      raw_records.append(folder_3d._EdgeRecord(p_a, p_b, 'B', 0.0))

  split_records = folder_3d._deduplicate_edges(
      folder_3d._split_planar_edges(v_index, raw_records, tol)
  )
  coords_u = np.asarray(v_index.coords, dtype=float)
  edge_pairs = [(r.v1, r.v2) for r in split_records]
  faces_cw = folder_3d.extract_planar_faces(coords_u, edge_pairs)
  if not faces_cw:
    raise ValueError('No bounded planar faces found (make sure border closes the sheet).')

  graph = folder_3d.FoldGraph(
      vertices_2d=coords_u,
      edges=np.asarray(edge_pairs, dtype=int),
      assignments=tuple(r.assignment for r in split_records),
      fold_angles=np.asarray([r.fold_angle for r in split_records], dtype=float),
      faces=tuple(np.asarray(f, dtype=int) for f in faces_cw),
  )

  # Select anchor face near sheet center.
  verts_centered = [(float(x - 0.5), float(y - 0.5)) for x, y in coords_u]
  anchor = render_3d._select_centroid_face(verts_centered, faces_cw)
  matrices, reached = folder_3d._solve_face_transforms(
      graph, anchor_face=anchor, strict=False
  )
  verts_3d = folder_3d._average_vertex_positions(
      graph.vertices_2d, graph.faces, matrices, reached
  )
  raw_pts_3d = [
      (float(x - 0.5), float(y - 0.5), float(z))
      for x, y, z in verts_3d
  ]
  aligned_pts_3d = render_3d._align_points_to_camera_frame(
      raw_pts_3d,
      verts_centered,
      list(faces_cw[anchor]),
      None,
  )
  nv = max(1, len(aligned_pts_3d))
  cx = sum(p[0] for p in aligned_pts_3d) / nv
  cy = sum(p[1] for p in aligned_pts_3d) / nv
  cz = sum(p[2] for p in aligned_pts_3d) / nv
  v3_exact = [(p[0] - cx, p[1] - cy, p[2] - cz) for p in aligned_pts_3d]
  edge_key_angle = {
      (min(r.v1, r.v2), max(r.v1, r.v2)): r.fold_angle
      for r in split_records
  }
  mesh = render_3d.FoldedMesh3D(
      vertices_2d=verts_centered,
      vertices_3d=v3_exact,
      faces=[list(f) for f in faces_cw],
      edges=edge_pairs,
      edges_assignment=[r.assignment for r in split_records],
      edges_fold_angle=[r.fold_angle for r in split_records],
      edge_key_angle=edge_key_angle,
      grid_size=grid_size,
  )
  fold_dict = render_3d.mesh_to_fold_dict(mesh, title)
  return fold_dict


def _safe_build_fold_dict(target, title: str, packed=None) -> dict:
  """Builds a Studio 'shaped_3d.fold' dict with strict=True first, then relaxed fallback."""
  try:
    mesh = bp.build_folded_mesh_3d(target, packed=packed)
    return bp.mesh_to_fold_dict(mesh, title)
  except Exception:
    pattern = folder_3d._unwrap_fold_target(target)
    folded = folder_3d.fold_3d(pattern, strict=False, add_border=True)
    verts_centered = [
        (float(x - 0.5), float(y - 0.5)) for x, y in folded.vertices_2d
    ]
    v3_exact = [
        (float(0.5 - p[0]), float(0.5 - p[1]), float(p[2]))
        for p in folded.vertices_3d
    ]
    edges_list = [(int(u), int(v)) for u, v in folded.edges]
    angles_list = [float(a) for a in folded.fold_angles]
    mesh = render_3d.FoldedMesh3D(
        vertices_2d=verts_centered,
        vertices_3d=v3_exact,
        faces=[[int(v) for v in f] for f in folded.faces],
        edges=edges_list,
        edges_assignment=list(folded.assignments),
        edges_fold_angle=angles_list,
        edge_key_angle={
            (min(u, v), max(u, v)): ang
            for (u, v), ang in zip(edges_list, angles_list)
        },
        grid_size=packed.grid_size if packed is not None else None,
    )
    return bp.mesh_to_fold_dict(mesh, title)


def _compute_leaf_distances(sf: sf_lib.StickFigure | None) -> list[dict]:
  """Computes exact tree path distances between leaf sticks for BP Studio checks."""
  if sf is None or not sf.children:
    return []
  adj: dict[str, list[tuple[str, float]]] = {}
  deg: dict[str, int] = {}
  for s in sf.children:
    adj.setdefault(s.start, []).append((s.end, float(s.length)))
    adj.setdefault(s.end, []).append((s.start, float(s.length)))
    deg[s.start] = deg.get(s.start, 0) + 1
    deg[s.end] = deg.get(s.end, 0) + 1
  leaves: list[tuple[str, str]] = []
  for s in sf.children:
    if deg.get(s.end, 0) == 1:
      leaves.append((s.name, s.end))
    elif deg.get(s.start, 0) == 1:
      leaves.append((s.name, s.start))
  out = []
  for i in range(len(leaves)):
    name_a, tip_a = leaves[i]
    dists = {tip_a: 0.0}
    queue = [tip_a]
    while queue:
      cur = queue.pop(0)
      for nb, w in adj.get(cur, ()):
        if nb not in dists:
          dists[nb] = dists[cur] + w
          queue.append(nb)
    for j in range(i + 1, len(leaves)):
      name_b, tip_b = leaves[j]
      if tip_b in dists:
        out.append({'a': name_a, 'b': name_b, 'dist': round(dists[tip_b], 4)})
  return out


def _serialize_packed(packed: bus.PackedCreasePattern | None) -> dict | None:
  if packed is None:
    return None
  tree_ref = packed.packed_tree or (packed.design.merged if packed.design is not None else None)
  stick_lengths = {s.name: int(round(s.length)) for s in (tree_ref.children if tree_ref else ())}
  g = int(packed.grid_size)
  flaps = []
  for fid, fl in packed.flaps.items():
    r = fl.rect
    centers = [[int(pt.x), int(pt.y)] for pt in fl.circle_centers]
    rad = stick_lengths.get(fl.stick, 0)
    if rad <= 0 and centers:
      cx0, cy0 = centers[0]
      rad = max(1, cx0 - int(r.x), int(r.x + r.width) - cx0, cy0 - int(r.y), int(r.y + r.height) - cy0)
    flaps.append({
        'id': str(fid),
        'stick': fl.stick,
        'x': int(r.x),
        'y': int(r.y),
        'width': int(r.width),
        'height': int(r.height),
        'radius': int(max(1, rad)),
        'circle_centers': centers,
    })
  rivers = []
  for rid, rv in packed.rivers.items():
    rivers.append({
        'id': str(rid),
        'stick': rv.stick,
        'width': int(rv.width),
        'is_on_axis': bool(rv.is_on_axis),
        'rects': [
            {'x': int(rr.x), 'y': int(rr.y), 'width': int(rr.width), 'height': int(rr.height)}
            for rr in rv.rects
        ],
    })
  hinges = []
  for h in packed.hinges:
    hinges.append({
        'x1': float(h.segment.p1.x),
        'y1': float(h.segment.p1.y),
        'x2': float(h.segment.p2.x),
        'y2': float(h.segment.p2.y),
        'elements': list(h.elements),
    })
  fold_code_map = {
      crease_lib.Fold.BORDER: 'B',
      crease_lib.Fold.MOUNTAIN: 'M',
      crease_lib.Fold.VALLEY: 'V',
      crease_lib.Fold.HINGE: 'H',
  }
  creases = []
  for c in packed.creases.creases:
    creases.append({
        'x1': round(float(c.p1.x), 6),
        'y1': round(float(c.p1.y), 6),
        'x2': round(float(c.p2.x), 6),
        'y2': round(float(c.p2.y), 6),
        'fold': fold_code_map.get(c.fold, 'F'),
    })
  val_errors = []
  try:
    bps_doc = bp.to_bp_studio(packed)
    imp = bp.import_bps(bps_doc)
    val_errors = list(imp.validation_errors)
  except Exception:
    pass
  return {
      'gridSize': g,
      'flaps': flaps,
      'rivers': rivers,
      'hinges': hinges,
      'creases': creases,
      'leafDistances': _compute_leaf_distances(tree_ref),
      'svg': bp.to_svg(packed),
      'errors': val_errors,
  }


def _parse_stick_figure_dict(raw: dict) -> sf_lib.StickFigure:
  """Normalizes minimal stick-figure dicts so 'category'/'complexity' have defaults."""
  normalized = dict(raw)
  children = list(normalized.get('children') or [])
  normalized.setdefault('name', 'custom_model')
  normalized.setdefault('category', 'custom')
  normalized.setdefault('complexity', max(1, min(5, len(children) // 4 + 1)))
  if not normalized.get('root') and children:
    normalized['root'] = children[0].get('name', 'root')
  return sf_lib.StickFigure.from_dict(normalized)


_MAX_ZIP_MEMBERS = 64
_MAX_ZIP_UNCOMPRESSED_BYTES = 32 * 1024 * 1024


class _StudioSession:

  def __init__(self):
    self.title = 'Custom Model'
    self.stick_figure: sf_lib.StickFigure | None = None
    self.design: bus.DesignTree | None = None
    self.packed: bus.PackedCreasePattern | None = None
    self.solved: bus.SolvedCreasePattern | None = None
    self.shaped: bus.ShapedCreasePattern | None = None
    self.raw_cp: cp_lib.CreasePattern | None = None
    self.last_fold_dict: dict | None = None
    self.shaping_state: toolbox_lib.ShapingState | None = None
    self.shaping_snapshots: list[tuple[toolbox_lib.ShapingState, dict]] = []
    self.shaping_history: list[dict] = []

  def _active_elements(self) -> list[dict]:
    if self.shaping_state is not None:
      try:
        flaps, rivers = self.shaping_state.shaping_elements
        out = [{'name': f.name, 'kind': 'flap'} for f in flaps]
        out.extend({'name': r.name, 'kind': 'river'} for r in rivers)
        return out
      except Exception:
        pass
    if self.stick_figure is not None:
      return [
          {'name': s.name, 'kind': 'river' if s.is_river else 'flap'}
          for s in self.stick_figure.children
      ]
    return []

  def _fidelity_summary(self) -> dict | None:
    target = self.shaped if self.shaped is not None else self.solved
    if target is None or self.stick_figure is None:
      return None
    try:
      rep = bp.evaluate_fidelity(target, tree=self.stick_figure)
      comps = rep.components or {}
      return {
          'overall': round(float(rep.fidelity), 2),
          'score': round(float(rep.fidelity), 2),
          'pose': round(float(comps.get('pose', 0.0)), 2),
          'width': round(float(comps.get('width', 0.0)), 2),
          'separation': round(float(comps.get('separation', 0.0)), 2),
          'cleanGate': bool(rep.clean_gate.passed),
          'poseGate': bool(rep.pose_gate.passed),
          'foldVertexErrorPct': round(float(rep.fold_vertex_error_percent), 2),
      }
    except Exception:
      return None

  def snapshot(self, extra: dict | None = None) -> dict:
    payload = {
        'title': self.title,
        'stickFigure': self.stick_figure.to_dict() if self.stick_figure is not None else None,
        'packing': _serialize_packed(self.packed),
        'hasSolved': self.solved is not None,
        'hasShaped': self.shaped is not None,
        'foldDict': self.last_fold_dict,
        'shapingElements': self._active_elements(),
        'shapingHistory': list(self.shaping_history),
        'fidelity': self._fidelity_summary(),
    }
    if extra:
      payload.update(extra)
    return payload

  def init_info(self) -> dict:
    return {
        'examples': bp.list_examples(),
        'tools': TOOL_CATALOG,
        'allToolNames': list(bp.list_shaping_tools()),
    }

  def sync_showcase(self, *, title: str, fold_dict: dict | None, stick_figure_dict: dict | None) -> dict:
    self.title = title or 'Model'
    self.last_fold_dict = fold_dict
    self.packed = None
    self.solved = None
    self.shaped = None
    self.shaping_state = None
    self.shaping_snapshots.clear()
    self.shaping_history.clear()
    if stick_figure_dict and isinstance(stick_figure_dict.get('children'), list):
      self.stick_figure = _parse_stick_figure_dict(stick_figure_dict)
      try:
        self.design = bp.prepare(self.stick_figure)
      except Exception:
        self.design = None
    else:
      self.stick_figure = None
      self.design = None
    if fold_dict is not None:
      try:
        self.raw_cp = bp.from_fold(fold_dict)
      except Exception:
        self.raw_cp = None
    return self.snapshot()

  def load_example_tree(self, name: str) -> dict:
    sf = bp.load_example(name)
    self.title = sf.name or name
    self.stick_figure = sf
    self.design = bp.prepare(sf)
    self.packed = None
    self.solved = None
    self.shaped = None
    self.shaping_state = None
    self.shaping_snapshots.clear()
    self.shaping_history.clear()
    return self.snapshot({
        'treeStats': {
            'numSticks': len(sf.children),
            'isSymmetric': bool(self.design.is_symmetric),
            'minGridSize': int(self.design.min_grid_size),
        }
    })

  def set_stick_figure(self, sf_dict: dict) -> dict:
    sf = _parse_stick_figure_dict(sf_dict)
    self.stick_figure = sf
    self.title = sf.name or self.title
    self.design = bp.prepare(sf)
    return self.snapshot({
        'treeStats': {
            'numSticks': len(sf.children),
            'isSymmetric': bool(self.design.is_symmetric),
            'minGridSize': int(self.design.min_grid_size),
        }
    })

  def pack(self, sf_dict: dict | None, config: dict | None) -> dict:
    cfg = config or {}
    if sf_dict is not None:
      self.stick_figure = _parse_stick_figure_dict(sf_dict)
      self.title = self.stick_figure.name or self.title
    if self.stick_figure is None:
      raise ValueError('Load or create a stick figure before packing.')
    self.design = bp.prepare(self.stick_figure)
    method = cfg.get('packer', 'pack')
    if method == 'stack':
      self.packed = bp.stack(self.design)
    else:
      max_retries = int(cfg.get('maxRetries', 8))
      timeout_s = float(cfg.get('timeoutSeconds', 20.0))
      allow_paired = bool(cfg.get('allowPairedAxis', False))
      task_cfg = bp.PackingTaskConfig(
          allow_paired_axis_placement=allow_paired,
          packing_config=bp.PackingConfig(
              use_level_shifter=False,
              enforce_proximity_constraints=False,
          ),
          max_retries=max_retries,
          max_cps_per_grid_size=1,
          max_number_of_cps=1,
          timeout_seconds=timeout_s,
          sweep_timeout_seconds=max(2.0, timeout_s / max(1, min(max_retries, 4))),
      )
      self.packed = bp.pack(self.design, config=task_cfg)
    self.solved = None
    self.shaped = None
    self.shaping_state = None
    self.shaping_snapshots.clear()
    self.shaping_history.clear()
    # Build an unfolded sheet preview in 3D Stage so Packing tab highlights sync cleanly.
    self.last_fold_dict = _safe_build_fold_dict(
        self.packed.creases, self.title, packed=self.packed
    )
    return self.snapshot()

  def update_packing(self, edit: dict) -> dict:
    """Applies interactive flap/grid edits via BPS round-trip & validates layout."""
    if self.packed is None and self.stick_figure is None:
      raise ValueError('No active packing to update.')
    base_bps = bp.to_bp_studio(self.packed if self.packed is not None else self.stick_figure)
    sheet_size = int(edit.get('gridSize') or (self.packed.grid_size if self.packed is not None else 16))
    if edit.get('bpsDoc'):
      base_bps = edit['bpsDoc']
    else:
      design_sec = base_bps.setdefault('design', {})
      layout_sec = design_sec.setdefault('layout', {})
      layout_sec['sheet'] = {'width': sheet_size, 'height': sheet_size}
      if edit.get('flapMoves'):
        moves = edit['flapMoves']  # list of {oldX, oldY, newX, newY}
        for lf in layout_sec.get('flaps', []):
          for mv in moves:
            if int(lf.get('x', -999)) == int(mv['oldX']) and int(lf.get('y', -999)) == int(mv['oldY']):
              lf['x'] = int(mv['newX'])
              lf['y'] = int(mv['newY'])
              break

    imp = bp.import_bps(base_bps)
    if imp.packed is not None:
      self.packed = imp.packed
      if imp.design is not None:
        self.design = imp.design
        self.stick_figure = imp.design.source
      self.solved = None
      self.shaped = None
      self.last_fold_dict = _safe_build_fold_dict(
          self.packed.creases, self.title, packed=self.packed
      )
      return self.snapshot({
          'packingValid': True,
          'bpsValidationErrors': list(imp.validation_errors),
      })
    # Keep user's in-progress dragged flap layout on the client without snap-resetting!
    return {
        'title': self.title,
        'packingValid': False,
        'bpsValidationErrors': list(imp.validation_errors),
    }

  def solve(self, config: dict | None) -> dict:
    if self.packed is None:
      if self.stick_figure is not None:
        try:
          self.pack(None, {'maxRetries': 6, 'timeoutSeconds': 10.0})
        except Exception:
          self.pack(None, {'packer': 'stack'})
      else:
        raise ValueError('Pack a stick figure or import a .bps layout first.')
    cfg = config or {}
    solver_cfg = bp.SolverConfig(
        max_steps=int(cfg.get('maxSteps', 2500)),
        max_time_limit=float(cfg.get('maxTime', 25.0)),
        max_number_of_cps=1,
        check_local_flat_foldability=bool(cfg.get('checkLocalFf', True)),
        check_global_flat_foldability=bool(cfg.get('checkGlobalFf', True)),
        remove_central_crease=bool(cfg.get('removeCentralCrease', False)),
        use_partition_order=bool(cfg.get('usePartitionOrder', True)),
        disallow_trapping_edge_flaps=bool(cfg.get('disallowTrapping', True)),
        hinges_from_stick_figure=bool(cfg.get('hingesFromStickFigure', True)),
    )
    self.solved = bp.solve_best(self.packed, config=solver_cfg)
    self.shaped = None
    self.raw_cp = self.solved.creases
    self.shaping_snapshots.clear()
    self.shaping_history.clear()
    try:
      self.shaping_state = bp.ShapingState.from_solved(self.solved)
    except Exception:
      self.shaping_state = None
    self.last_fold_dict = _safe_build_fold_dict(
        self.solved, self.title, packed=self.packed
    )
    return self.snapshot({
        'solveStats': {
            'globallyFlatFoldable': bool(self.solved.is_globally_flat_foldable),
            'centralCreaseRemoved': bool(self.solved.central_crease_removed),
            'gridSize': int(self.solved.grid_size),
            'numCreases': len(self.solved.creases.creases),
        }
    })

  def shape(self, config: dict | None) -> dict:
    if self.solved is None:
      self.solve(None)
    cfg = config or {}
    method = cfg.get('method', 'pose')
    if method == 'pose':
      pose_mode = cfg.get('poseMode', 'figure')
      narrow = bool(cfg.get('narrow', True))
      budget = float(cfg.get('timeBudget', 15.0))
      self.shaped = bp.pose(
          self.solved,
          config=bp.PoseConfig(mode=pose_mode, narrow=narrow, time_budget_s=budget),
      )
    else:
      strat_name = str(cfg.get('strategy', 'DEFAULT')).upper()
      strat_enum = getattr(bp.FoldStrategy3D, strat_name, bp.FoldStrategy3D.DEFAULT)
      shape_cfg = bp.ShapingTaskConfig(fold_strategy_3d=strat_enum, max_number_of_cps=1)
      self.shaped = bp.shape(self.solved, config=shape_cfg, design=self.design)

    self.raw_cp = self.shaped.creases
    try:
      self.shaping_state = bp.ShapingState.from_solved(
          self.solved,
          pattern=self.shaped.creases,
      )
    except Exception:
      pass
    self.shaping_snapshots.clear()
    self.last_fold_dict = _safe_build_fold_dict(
        self.shaped, self.title, packed=self.packed
    )
    return self.snapshot()

  def apply_tool(self, tool_name: str, args: dict, symmetric: bool = False) -> dict:
    if self.shaping_state is None:
      if self.solved is None:
        if self.stick_figure is not None or self.packed is not None:
          self.solve(None)
        else:
          raise ValueError('Solve a base crease pattern before applying shaping tools.')
      self.shaping_state = bp.ShapingState.from_solved(
          self.solved,
          pattern=self.shaped.creases if self.shaped is not None else None,
      )
    prev_state = self.shaping_state
    prev_fold = self.last_fold_dict
    effective_tool = 'apply_symmetric' if symmetric and tool_name not in ('fold_spine', 'thin_limbs_3d', 'apply_symmetric') else tool_name
    effective_args = {'tool': tool_name, 'args': dict(args)} if effective_tool == 'apply_symmetric' else dict(args)

    next_state, res = bp.apply_shaping_tool(self.shaping_state, effective_tool, effective_args)
    if not res.ok:
      raise ValueError(res.message or f'Tool {tool_name} could not be applied cleanly.')
    self.shaping_snapshots.append((prev_state, prev_fold))
    self.shaping_state = next_state
    call_rec = {'tool': res.tool, 'arguments': dict(res.arguments)}
    self.shaping_history.append(call_rec)
    try:
      self.shaped = self.shaping_state.to_shaped(self.solved)
      self.raw_cp = self.shaped.creases
      self.last_fold_dict = _safe_build_fold_dict(
          self.shaped, self.title, packed=self.packed
      )
    except Exception:
      self.raw_cp = self.shaping_state.pattern
      self.last_fold_dict = _safe_build_fold_dict(
          self.raw_cp, self.title, packed=self.packed
      )
    if prev_fold and self.last_fold_dict:
      try:
        prev_flat = (prev_fold.get('file_frames') or [prev_fold])[0]['vertices_coords']
        prev_edges = prev_fold.get('edges_vertices') or []
        prev_assigns = prev_fold.get('edges_assignment') or []
        prev_user = prev_fold.get('edges_bp:userShaping') or [False] * len(prev_edges)
        old_mv = {}
        for idx_e, (u0, v0) in enumerate(prev_edges):
          key = (
              round(0.5 * (prev_flat[u0][0] + prev_flat[v0][0]), 5),
              round(0.5 * (prev_flat[u0][1] + prev_flat[v0][1]), 5),
          )
          old_mv[key] = (
              prev_assigns[idx_e] if idx_e < len(prev_assigns) else 'F',
              bool(prev_user[idx_e]) if idx_e < len(prev_user) else False,
          )
        cur_flat = (self.last_fold_dict.get('file_frames') or [self.last_fold_dict])[0]['vertices_coords']
        cur_edges = self.last_fold_dict.get('edges_vertices') or []
        cur_assigns = self.last_fold_dict.get('edges_assignment') or []
        user_flags = []
        for idx_e, (u0, v0) in enumerate(cur_edges):
          a = cur_assigns[idx_e] if idx_e < len(cur_assigns) else 'F'
          key = (
              round(0.5 * (cur_flat[u0][0] + cur_flat[v0][0]), 5),
              round(0.5 * (cur_flat[u0][1] + cur_flat[v0][1]), 5),
          )
          prev_info = old_mv.get(key)
          is_new_mv = a in ('M', 'V') and (prev_info is None or prev_info[0] != a or prev_info[1])
          user_flags.append(bool(is_new_mv))
        self.last_fold_dict['edges_bp:userShaping'] = user_flags
      except Exception:
        pass
    return self.snapshot({
        'toolResult': {'ok': True, 'message': res.message}
    })

  def undo_shaping(self) -> dict:
    if not self.shaping_snapshots:
      return self.snapshot()
    prev_state, prev_fold = self.shaping_snapshots.pop()
    self.shaping_state = prev_state
    if self.shaping_history:
      self.shaping_history.pop()
    self.last_fold_dict = prev_fold
    if self.solved is not None:
      try:
        self.shaped = self.shaping_state.to_shaped(self.solved)
      except Exception:
        pass
    return self.snapshot()

  def reset_shaping(self) -> dict:
    if self.solved is None:
      return self.snapshot()
    self.shaped = None
    self.shaping_snapshots.clear()
    self.shaping_history.clear()
    self.shaping_state = bp.ShapingState.from_solved(self.solved)
    self.last_fold_dict = _safe_build_fold_dict(
        self.solved, self.title, packed=self.packed
    )
    return self.snapshot()

  def fold_cp_graph(self, graph_payload: dict) -> dict:
    fold_dict = _fold_arbitrary_2d_graph_to_fold_dict(
        vertices_2d=graph_payload['vertices'],
        edges=graph_payload['edges'],
        assignments=graph_payload['assignments'],
        fold_angles=graph_payload.get('foldAngles'),
        title=graph_payload.get('title') or self.title,
        grid_size=graph_payload.get('gridSize'),
    )
    self.last_fold_dict = fold_dict
    try:
      self.raw_cp = bp.from_fold(fold_dict)
    except Exception:
      pass
    return self.snapshot()

  def check_global_ff(self, graph_payload: dict | None) -> dict:
    cp_obj = self.raw_cp
    if graph_payload is not None:
      fold_dict = _fold_arbitrary_2d_graph_to_fold_dict(
          vertices_2d=graph_payload['vertices'],
          edges=graph_payload['edges'],
          assignments=graph_payload['assignments'],
          fold_angles=graph_payload.get('foldAngles'),
          title=self.title,
      )
      cp_obj = bp.from_fold(fold_dict)
    if cp_obj is None and self.solved is not None:
      cp_obj = self.solved.creases
    if cp_obj is None:
      raise ValueError('No crease pattern available to verify.')
    cp_str = cp_format.to_cp(cp_obj, scale=1, ndigits=6, grid=cp_obj.grid_size or -1)
    out = fold_cp.analyze_fold_file(
        filename='pattern.cp',
        doc_content=cp_str,
        file_ext='.cp',
        generate_state_data=True,
    )
    ff = out.flat_folder_fold
    is_ff = bool(ff is not None and ff.folded_state_fold is not None and not ff.conflict_reason)
    pct, _, non_ff = cp_obj.vertex_flat_foldability_percentage()
    return {
        'flatFoldable': is_ff,
        'localFfPercent': round(float(pct), 1),
        'localViolationCount': len(non_ff),
        'conflictReason': getattr(ff, 'conflict_reason', None) if ff is not None else 'flat-folder failed',
    }

  def drop_files(self, files: list[dict], auto_pipeline: bool = True) -> dict:
    """Universal zero-upload local file drop handler."""
    expanded: list[tuple[str, str]] = []
    for item in files:
      fname = str(item.get('name', 'file'))
      lower = fname.lower()
      if lower.endswith('.zip') and item.get('base64'):
        raw_bytes = base64.b64decode(item['base64'])
        with zipfile.ZipFile(io.BytesIO(raw_bytes), 'r') as zf:
          members = [
              zi
              for zi in zf.infolist()
              if not zi.is_dir() and '__MACOSX' not in zi.filename
          ]
          if len(members) > _MAX_ZIP_MEMBERS:
            raise ValueError(
                f'ZIP archive contains too many files ({len(members)} > {_MAX_ZIP_MEMBERS}).'
            )
          declared_total = 0
          for zi in members:
            declared_total += int(zi.file_size)
            if declared_total > _MAX_ZIP_UNCOMPRESSED_BYTES:
              raise ValueError(
                  'ZIP archive exceeds maximum uncompressed size of 32 MiB.'
              )
          decoded_total = 0
          for zi in members:
            base = zi.filename.rsplit('/', 1)[-1]
            with zf.open(zi, 'r') as member_fp:
              chunk = member_fp.read(
                  _MAX_ZIP_UNCOMPRESSED_BYTES + 1 - decoded_total
              )
            decoded_total += len(chunk)
            if decoded_total > _MAX_ZIP_UNCOMPRESSED_BYTES:
              raise ValueError(
                  'ZIP archive exceeds maximum uncompressed size of 32 MiB.'
              )
            try:
              expanded.append((base, chunk.decode('utf-8')))
            except UnicodeDecodeError:
              pass
      elif item.get('text') is not None:
        expanded.append((fname, str(item['text'])))

    if not expanded:
      raise ValueError('No readable origami files found.')

    sf_json = None
    history_json = None
    bps_json = None
    fold_json = None
    svg_map: dict[str, str] = {}
    cp_text = None
    inferred_title = None

    for fname, text in expanded:
      lower = fname.lower()
      stem = fname.rsplit('.', 1)[0]
      if lower.endswith('.bps') or lower.endswith('.bps.json'):
        bps_json = json.loads(text)
        inferred_title = inferred_title or stem
      elif lower.endswith('.cp') or lower.endswith('.ori'):
        cp_text = text
        inferred_title = inferred_title or stem
      elif lower.endswith('.svg'):
        svg_map[lower] = text
        inferred_title = inferred_title or stem
      elif lower.endswith('.fold'):
        fold_json = json.loads(text)
        inferred_title = fold_json.get('file_title') or inferred_title or stem
      elif lower.endswith('.json'):
        parsed = json.loads(text)
        if isinstance(parsed, dict) and isinstance(parsed.get('children'), list):
          sf_json = parsed
          inferred_title = parsed.get('name') or inferred_title or stem
        elif isinstance(parsed, list) and parsed and isinstance(parsed[0], dict) and 'tool' in parsed[0]:
          history_json = parsed
        elif isinstance(parsed, dict) and ('design' in parsed or 'tree' in parsed or 'layout' in parsed):
          bps_json = parsed
          inferred_title = inferred_title or stem
        elif isinstance(parsed, dict) and ('vertices_coords' in parsed or 'file_frames' in parsed):
          fold_json = parsed
          inferred_title = parsed.get('file_title') or inferred_title or stem

    if inferred_title:
      self.title = inferred_title

    # Case 1: Box Pleating Studio (.bps / .bps.json)
    if bps_json is not None:
      self.packed = None
      self.solved = None
      self.shaped = None
      self.shaping_state = None
      self.shaping_snapshots.clear()
      self.shaping_history.clear()
      self.raw_cp = None
      imp = bp.import_bps(bps_json)
      if imp.design is not None:
        self.design = imp.design
        self.stick_figure = imp.design.source
      if imp.packed is not None:
        self.packed = imp.packed
        if auto_pipeline:
          try:
            snap = self.solve({'checkGlobalFf': True})
            snap['droppedKind'] = 'bps'
            return snap
          except Exception:
            self.last_fold_dict = _safe_build_fold_dict(
                self.packed.creases, self.title, packed=self.packed
            )
            return self.snapshot({
                'droppedKind': 'bps',
                'bpsValidationErrors': list(imp.validation_errors),
            })
        self.last_fold_dict = _safe_build_fold_dict(
            self.packed.creases, self.title, packed=self.packed
        )
      elif self.stick_figure is not None and auto_pipeline:
        self.pack(None, None)
        snap = self.solve({'checkGlobalFf': True})
        snap['droppedKind'] = 'bps'
        return snap
      return self.snapshot({
          'droppedKind': 'bps',
          'bpsValidationErrors': list(imp.validation_errors),
      })

    # Case 2: Explicit .fold (either 2D creasePattern or 3D foldedForm)
    if fold_json is not None:
      if sf_json is not None:
        self.stick_figure = _parse_stick_figure_dict(sf_json)
        self.design = bp.prepare(self.stick_figure)
      # Check if already has both 3D keyframe and faces_vertices + 2D frame
      has_faces = bool(fold_json.get('faces_vertices'))
      has_3d = bool(
          fold_json.get('vertices_coords')
          and len(fold_json['vertices_coords'][0]) == 3
          and fold_json.get('file_frames')
      )
      if has_faces and has_3d:
        self.last_fold_dict = fold_json
        try:
          self.raw_cp = bp.from_fold(fold_json)
        except Exception:
          pass
        return self.snapshot({'droppedKind': 'fold_3d'})

      # 2D or face-less .fold file -> fold with our deterministic 3D folder!
      frames = [fold_json] + list(fold_json.get('file_frames') or [])
      flat_frame = next(
          (f for f in frames if f.get('vertices_coords') and len(f['vertices_coords'][0]) == 2),
          fold_json,
      )
      v2d = [[float(p[0]), float(p[1])] for p in flat_frame['vertices_coords']]
      edges = flat_frame.get('edges_vertices') or fold_json.get('edges_vertices')
      assigns = flat_frame.get('edges_assignment') or fold_json.get('edges_assignment') or ['F'] * len(edges)
      angles = flat_frame.get('edges_foldAngle') or fold_json.get('edges_foldAngle')
      self.last_fold_dict = _fold_arbitrary_2d_graph_to_fold_dict(
          vertices_2d=v2d,
          edges=edges,
          assignments=assigns,
          fold_angles=angles,
          title=self.title,
      )
      try:
        self.raw_cp = bp.from_fold(self.last_fold_dict)
      except Exception:
        pass
      return self.snapshot({'droppedKind': 'fold_2d'})

    # Case 3: .svg (including COrigami zip bundle with solving_cp.svg + shaping_cp.svg)
    if svg_map:
      if sf_json is not None:
        self.stick_figure = _parse_stick_figure_dict(sf_json)
      chosen_key = (
          next((k for k in svg_map if 'shap' in k), None)
          or next((k for k in svg_map if 'solv' in k), None)
          or next(iter(svg_map))
      )
      cp_obj = _normalize_external_svg_to_cp(svg_map[chosen_key])
      self.raw_cp = cp_obj
      self.last_fold_dict = _safe_build_fold_dict(cp_obj, self.title)
      return self.snapshot({'droppedKind': 'svg'})

    # Case 4: ORIPA .cp file
    if cp_text is not None:
      cp_obj = _normalize_cp_file_to_cp(cp_text)
      self.raw_cp = cp_obj
      self.last_fold_dict = _safe_build_fold_dict(cp_obj, self.title)
      return self.snapshot({'droppedKind': 'cp'})

    # Case 5: Standalone stick_figure.json (+ optional shaping_history.json)
    if sf_json is not None:
      self.set_stick_figure(sf_json)
      if auto_pipeline:
        self.pack(None, None)
        self.solve({'checkGlobalFf': True})
        if history_json:
          for call in history_json:
            tname = call.get('tool')
            targs = call.get('arguments') or {}
            if tname:
              try:
                self.apply_tool(tname, targs, symmetric=False)
              except Exception:
                pass
        else:
          try:
            self.shape({'method': 'pose', 'poseMode': 'figure', 'narrow': True})
          except Exception:
            pass
      return self.snapshot({'droppedKind': 'stick_figure'})

    raise ValueError('Unsupported file format.')

  def export_files(self) -> dict:
    if self.shaped is not None:
      active_target = self.shaped
    elif self.solved is not None:
      active_target = self.solved
    elif self.packed is not None:
      active_target = self.packed
    elif self.raw_cp is not None:
      active_target = self.raw_cp
    elif self.last_fold_dict:
      active_target = bp.from_fold(self.last_fold_dict)
    else:
      active_target = None
    svg_str = bp.to_svg(active_target) if active_target is not None else None
    cp_str = None
    if active_target is not None:
      cp_only = (
          active_target
          if isinstance(active_target, cp_lib.CreasePattern)
          else active_target.creases
      )
      cp_str = cp_format.to_cp(cp_only, scale=1, ndigits=6, grid=cp_only.grid_size or -1)
    bps_str = None
    bps_source = self.packed if self.packed is not None else self.stick_figure
    if bps_source is not None:
      try:
        bps_str = json.dumps(bp.to_bp_studio(bps_source), indent=2)
      except Exception:
        bps_str = None
    fold_str = json.dumps(self.last_fold_dict, indent=2) if self.last_fold_dict else None
    sf_str = json.dumps(self.stick_figure.to_dict(), indent=2) if self.stick_figure else None
    hist_str = json.dumps(self.shaping_history, indent=2) if self.shaping_history else None

    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, 'w', zipfile.ZIP_DEFLATED) as zf:
      if fold_str:
        zf.writestr('shaped_3d.fold', fold_str)
      if svg_str:
        zf.writestr('crease_pattern.svg', svg_str)
      if cp_str:
        zf.writestr('crease_pattern.cp', cp_str)
      if bps_str:
        zf.writestr('project.bps.json', bps_str)
      if sf_str:
        zf.writestr('stick_figure.json', sf_str)
      if hist_str:
        zf.writestr('shaping_history.json', hist_str)
      if self.packed is not None:
        zf.writestr('packing_cp.svg', bp.to_svg(self.packed))
      if self.solved is not None:
        zf.writestr('solving_cp.svg', bp.to_svg(self.solved))
      if self.shaped is not None:
        zf.writestr('shaping_cp.svg', bp.to_svg(self.shaped))

    return {
        'foldJson': fold_str,
        'svgText': svg_str,
        'cpText': cp_str,
        'bpsJson': bps_str,
        'stickFigureJson': sf_str,
        'shapingHistoryJson': hist_str,
        'zipBase64': base64.b64encode(zip_buf.getvalue()).decode('ascii'),
    }


_SESSION = _StudioSession()


def dispatch_rpc(action: str, payload_json: str) -> str:
  params = json.loads(payload_json) if payload_json else {}
  if action == 'init':
    res = _SESSION.init_info()
  elif action == 'sync_showcase':
    res = _SESSION.sync_showcase(
        title=params.get('title', 'Model'),
        fold_dict=params.get('foldDict'),
        stick_figure_dict=params.get('stickFigure'),
    )
  elif action == 'load_example_tree':
    res = _SESSION.load_example_tree(params['name'])
  elif action == 'set_stick_figure':
    res = _SESSION.set_stick_figure(params['stickFigure'])
  elif action == 'pack':
    res = _SESSION.pack(params.get('stickFigure'), params.get('config'))
  elif action == 'update_packing':
    res = _SESSION.update_packing(params)
  elif action == 'solve':
    res = _SESSION.solve(params.get('config'))
  elif action == 'shape':
    res = _SESSION.shape(params.get('config'))
  elif action == 'apply_tool':
    res = _SESSION.apply_tool(
        params['tool'],
        params.get('args') or {},
        bool(params.get('symmetric', False)),
    )
  elif action == 'undo_shaping':
    res = _SESSION.undo_shaping()
  elif action == 'reset_shaping':
    res = _SESSION.reset_shaping()
  elif action == 'fold_cp_graph':
    res = _SESSION.fold_cp_graph(params)
  elif action == 'check_global_ff':
    res = _SESSION.check_global_ff(params.get('graph'))
  elif action == 'drop_files':
    res = _SESSION.drop_files(
        params.get('files') or [],
        bool(params.get('autoPipeline', True)),
    )
  elif action == 'export_files':
    res = _SESSION.export_files()
  else:
    raise ValueError(f'Unknown worker action: {action!r}')
  return json.dumps(res)
`;

/** @return {!Promise<!Uint8Array>} */
async function fetchWheelBytes() {
  for (const rel of WHEEL_CANDIDATES) {
    const url = new URL(rel, self.location.href).href;
    try {
      const resp = await fetch(url);
      if (resp.ok) return new Uint8Array(await resp.arrayBuffer());
    } catch (_) {
      // Try next candidate.
    }
  }
  throw new Error(
      'bp_pipeline wheel not found under docs/py/. Run python3 scripts/serve_studio.py ' +
      'for local preview or build the wheel in CI.');
}

/**
 * @param {number} reqId
 * @return {!Promise<!Function>}
 */
async function ensurePyodide(reqId) {
  if (pyDispatch) return pyDispatch;
  if (pyodideReadyPromise) return pyodideReadyPromise;

  pyodideReadyPromise = (async () => {
    postProgress(reqId, 'Loading Python WebAssembly runtime…');
    importScripts(`${DEFAULT_PYODIDE_CDN}pyodide.js`);
    const pyodide = await loadPyodide({indexURL: DEFAULT_PYODIDE_CDN});

    postProgress(reqId, 'Loading NumPy, NetworkX, lxml & Pydantic…');
    await pyodide.loadPackage(['numpy', 'networkx', 'lxml', 'pydantic']);

    postProgress(reqId, 'Installing COrigami bp_pipeline engine…');
    await pyodide.runPythonAsync(PYTHON_BOOTSTRAP);
    const wheelBytes = await fetchWheelBytes();
    if (typeof pyodide.unpackArchive === 'function') {
      pyodide.unpackArchive(wheelBytes, 'wheel');
    } else {
      pyodide.FS.writeFile('/tmp/bp_pipeline.whl', wheelBytes);
      await pyodide.runPythonAsync(`
import sys, zipfile
zipfile.ZipFile('/tmp/bp_pipeline.whl').extractall('/tmp/bp_site')
if '/tmp/bp_site' not in sys.path:
  sys.path.insert(0, '/tmp/bp_site')
`);
    }

    await pyodide.runPythonAsync(PYTHON_BRIDGE);
    pyDispatch = pyodide.globals.get('dispatch_rpc');
    postProgress(reqId, 'Python WASM engine ready');
    return pyDispatch;
  })();

  try {
    return await pyodideReadyPromise;
  } catch (err) {
    pyodideReadyPromise = null;
    throw err;
  }
}

/** @param {!MessageEvent<*>} event */
self.onmessage = async (event) => {
  const {id, action, payload} = event.data || {};
  try {
    const dispatch = await ensurePyodide(id);
    postProgress(id, `Running ${action.replaceAll('_', ' ')}…`);
    const outJson = dispatch(action, JSON.stringify(payload ?? {}));
    const result = JSON.parse(outJson);
    self.postMessage({type: 'result', id, result});
  } catch (error) {
    const msg = error?.message || String(error);
    // Strip verbose Python frames down to the actionable ValueError/RuntimeError message when present.
    const cleanMatch = msg.match(/(?:ValueError|RuntimeError|PackingFailedError|SolvingFailedError|ShapingFailedError|SVGParseError):\s*([^\n]+)/);
    self.postMessage({
      type: 'error',
      id,
      error: cleanMatch ? cleanMatch[1] : msg,
      detail: msg,
    });
  }
};
