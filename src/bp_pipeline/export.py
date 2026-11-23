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

"""Export utilities for BoxPleatingTreeData (.bps.json), SVG (.svg), and FOLD (.fold)."""

import collections
from collections.abc import Mapping, Sequence
import copy
import dataclasses
import io
import json
import math
import os
import pathlib
from typing import Any, Literal, TypeAlias, cast

import numpy as np

from . import bps_types
from . import bus
from . import folder_3d as fold_3d
from . import rectangle as rectangle_lib
from .crease_pattern import crease as crease_lib
from .crease_pattern import crease_pattern as cp_lib
from .crease_pattern import point as point_lib
from .crease_pattern.parser import svg as svg_parser
from .packing import compute_ridges
from .packing import config as packing_config_lib
from .packing import packing_output as packing_output_lib
from .packing import river as river_lib
from .packing import types as packing_types
from .packing import utils_packing_flaps
from .packing import utils_packing_svg
from .packing import utils_tracker
from .solving import hinges
from .solving import solvability
from .stick_figures import converters as stick_converters
from .stick_figures import stick as stick_lib
from .stick_figures import stick_figure as stick_figure_lib

FoldAssignment: TypeAlias = Literal['B', 'M', 'V', 'F', 'U', 'C']
FileTarget: TypeAlias = str | os.PathLike[str] | io.TextIOBase
BpStudioExportable: TypeAlias = (
    stick_figure_lib.StickFigure
    | bus.DesignTree
    | bus.PackedCreasePattern
    | bus.SolvedCreasePattern
    | bus.ShapedCreasePattern
    | bus.PipelineResult
)
CreasePatternExportable: TypeAlias = (
    cp_lib.CreasePattern
    | bus.PackedCreasePattern
    | bus.SolvedCreasePattern
    | bus.ShapedCreasePattern
    | bus.PipelineResult
)


DEFAULT_SVG_COLORS: Mapping[crease_lib.Fold, str] = {
    crease_lib.Fold.BORDER: 'black',
    crease_lib.Fold.MOUNTAIN: 'red',
    crease_lib.Fold.VALLEY: 'blue',
    crease_lib.Fold.HINGE: 'green',
    crease_lib.Fold.TRIANGLE: 'yellow',
    crease_lib.Fold.MOUNTAIN_ZEBRA: 'orange',
    crease_lib.Fold.VALLEY_ZEBRA: 'turquoise',
}

_COLOR_KEY_ALIASES: Mapping[str, tuple[crease_lib.Fold, ...]] = {
    'border': (crease_lib.Fold.BORDER,),
    'boundary': (crease_lib.Fold.BORDER,),
    'mountain': (crease_lib.Fold.MOUNTAIN,),
    'valley': (crease_lib.Fold.VALLEY,),
    'hinge': (crease_lib.Fold.HINGE,),
    'triangle': (crease_lib.Fold.TRIANGLE,),
    'axial': (crease_lib.Fold.TRIANGLE,),
    'ridge': (
        crease_lib.Fold.MOUNTAIN_ZEBRA,
        crease_lib.Fold.VALLEY_ZEBRA,
    ),
    'mountain_zebra': (crease_lib.Fold.MOUNTAIN_ZEBRA,),
    'valley_zebra': (crease_lib.Fold.VALLEY_ZEBRA,),
    'unassigned': (crease_lib.Fold.HINGE,),
}

_FOLD_TO_ASSIGNMENT: Mapping[crease_lib.Fold, FoldAssignment] = {
    crease_lib.Fold.BORDER: 'B',
    crease_lib.Fold.MOUNTAIN: 'M',
    crease_lib.Fold.MOUNTAIN_ZEBRA: 'M',
    crease_lib.Fold.VALLEY: 'V',
    crease_lib.Fold.VALLEY_ZEBRA: 'V',
    crease_lib.Fold.TRIANGLE: 'F',
    crease_lib.Fold.HINGE: 'U',
}

_FOLD_TO_ANGLE_SIGN: Mapping[crease_lib.Fold, float] = {
    crease_lib.Fold.BORDER: 0.0,
    crease_lib.Fold.MOUNTAIN: -180.0,
    crease_lib.Fold.MOUNTAIN_ZEBRA: -180.0,
    crease_lib.Fold.VALLEY: 180.0,
    crease_lib.Fold.VALLEY_ZEBRA: 180.0,
    crease_lib.Fold.TRIANGLE: 0.0,
    crease_lib.Fold.HINGE: 0.0,
}


def _write_text(
    target: FileTarget,
    content: str,
) -> None:
  """Writes text content to a filesystem path or a file-like object."""
  if isinstance(target, (str, os.PathLike)):
    path = pathlib.Path(target)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding='utf-8')
    return
  if hasattr(target, 'write'):
    target.write(content)
    return
  raise TypeError(
      'Expected file path or file-like object with write(), got'
      f' {type(target)!r}'
  )


def _extract_bps_stages(
    obj: BpStudioExportable,
    *,
    use_merged_tree: bool = False,
) -> tuple[
    stick_figure_lib.StickFigure,
    bus.DesignTree | None,
    bus.PackedCreasePattern | None,
    int,
]:
  """Extracts `(StickFigure, DesignTree, PackedCreasePattern, scale_factor)`."""
  if isinstance(obj, bus.PipelineResult):
    sf = obj.design.merged if use_merged_tree else obj.design.source
    return sf, obj.design, obj.packed, max(1, int(obj.solved.scale_factor))
  if isinstance(obj, bus.ShapedCreasePattern):
    return _extract_bps_stages(obj.solved, use_merged_tree=use_merged_tree)
  if isinstance(obj, bus.SolvedCreasePattern):
    sf, design, packed, _ = _extract_bps_stages(
        obj.packed, use_merged_tree=use_merged_tree
    )
    return sf, design, packed, max(1, int(obj.scale_factor))
  if isinstance(obj, bus.PackedCreasePattern):
    design = obj.design
    if design is None and obj.packed_tree is not None:
      design = bus.DesignTree.from_stick_figure(obj.packed_tree)
    sf = (
        (design.merged if use_merged_tree else design.source)
        if design is not None
        else obj.packed_tree
    )
    if sf is None:
      raise ValueError(
          'PackedCreasePattern has no DesignTree or packed_tree; cannot export'
          ' tree structure to BoxPleatingTreeData.'
      )
    return sf, design, obj, 1
  if isinstance(obj, bus.DesignTree):
    sf = obj.merged if use_merged_tree else obj.source
    return sf, obj, None, 1
  if isinstance(obj, stick_figure_lib.StickFigure):
    design = bus.DesignTree.from_stick_figure(obj)
    sf = design.merged if use_merged_tree else design.source
    return sf, design, None, 1
  raise TypeError(
      'to_bp_studio() expected DesignTree, StickFigure, PackedCreasePattern,'
      f' SolvedCreasePattern, or PipelineResult; got {type(obj)!r}'
  )


def _assign_node_ids_and_coords(
    sf: stick_figure_lib.StickFigure,
    *,
    tree_sheet_size: int = 20,
) -> tuple[
    dict[str, int],
    list[bps_types.NodeDict],
    list[bps_types.EdgeDict],
    dict[str, tuple[int, int]],
]:
  """Computes 2D node coordinates and edges that round-trip via BPS format."""
  sticks_by_name = {s.name: s for s in sf.children}
  ordered_sticks: list[stick_lib.Stick] = []
  if sf.root in sticks_by_name:
    ordered_sticks.append(sticks_by_name[sf.root])
  for s in sf.children:
    if s.name != sf.root:
      ordered_sticks.append(s)

  raw_nodes: list[str] = []
  seen_nodes: set[str] = set()
  for s in ordered_sticks:
    for endpoint in (s.start, s.end):
      if endpoint not in seen_nodes:
        seen_nodes.add(endpoint)
        raw_nodes.append(endpoint)

  all_int_ids = all(n.isdigit() for n in raw_nodes)
  if all_int_ids and len({int(n) for n in raw_nodes}) == len(raw_nodes):
    node_id_map = {n: int(n) for n in raw_nodes}
  else:
    node_id_map = {n: idx for idx, n in enumerate(raw_nodes)}

  adj: dict[str, list[tuple[str, stick_lib.Stick, bool]]] = {
      n: [] for n in raw_nodes
  }
  for s in ordered_sticks:
    adj[s.start].append((s.end, s, True))
    adj[s.end].append((s.start, s, False))

  coords: dict[str, tuple[float, float]] = {}
  visited_sticks: set[str] = set()
  directed_edges: list[tuple[str, str, stick_lib.Stick]] = []

  if ordered_sticks:
    root_stick = ordered_sticks[0]
    coords[root_stick.start] = (0.0, 0.0)
    queue: collections.deque[str] = collections.deque([root_stick.start])
    while queue:
      curr = queue.popleft()
      cx, cy = coords[curr]
      for nbr, stick, is_forward in adj.get(curr, ()):
        if stick.name in visited_sticks:
          continue
        visited_sticks.add(stick.name)
        angle_deg = (
            float(stick.azimuth_angle)
            if is_forward
            else (float(stick.azimuth_angle) + 180.0) % 360.0
        )
        theta = math.radians(angle_deg + 90.0)
        length = float(max(stick.length, 1))
        nx_coord = cx + length * math.cos(theta)
        ny_coord = cy + length * math.sin(theta)
        if nbr not in coords:
          coords[nbr] = (nx_coord, ny_coord)
          queue.append(nbr)
        if is_forward:
          directed_edges.append((curr, nbr, stick))
        else:
          directed_edges.append((nbr, curr, stick))

  for n in raw_nodes:
    if n not in coords:
      coords[n] = (0.0, 0.0)

  xs = [pt[0] for pt in coords.values()]
  ys = [pt[1] for pt in coords.values()]
  min_x, max_x = min(xs, default=0.0), max(xs, default=0.0)
  min_y, max_y = min(ys, default=0.0), max(ys, default=0.0)
  span = max(max_x - min_x, max_y - min_y)
  center_x = 0.5 * (min_x + max_x)
  center_y = 0.5 * (min_y + max_y)
  half_sheet = 0.5 * float(tree_sheet_size)
  usable_radius = max(1.0, half_sheet - 2.0)
  scale = (2.0 * usable_radius / span) if span > 2.0 * usable_radius else 1.0

  nodes_out: list[bps_types.NodeDict] = []
  for n in sorted(raw_nodes, key=lambda key: node_id_map[key]):
    rx, ry = coords[n]
    tx = round(half_sheet + (rx - center_x) * scale, 6)
    ty = round(half_sheet + (ry - center_y) * scale, 6)
    if math.isclose(tx, round(tx), abs_tol=1e-6):
      tx = float(int(round(tx)))
    if math.isclose(ty, round(ty), abs_tol=1e-6):
      ty = float(int(round(ty)))
    node_label = '' if n.isdigit() else n
    nodes_out.append({
        'id': node_id_map[n],
        'name': node_label,
        'x': tx,
        'y': ty,
    })

  edge_by_stick = {s.name: (u, v, s) for u, v, s in directed_edges}
  edges_out: list[bps_types.EdgeDict] = []
  stick_edge_ids: dict[str, tuple[int, int]] = {}
  for s in ordered_sticks:
    u, v, _ = edge_by_stick.get(s.name, (s.start, s.end, s))
    n1 = node_id_map[u]
    n2 = node_id_map[v]
    raw_len = float(s.length)
    edge_len: int | float = (
        int(round(raw_len))
        if math.isclose(raw_len, round(raw_len), abs_tol=1e-6)
        else raw_len
    )
    edges_out.append({
        'n1': n1,
        'n2': n2,
        'length': edge_len,
    })
    stick_edge_ids[s.name] = (n1, n2)

  return node_id_map, nodes_out, edges_out, stick_edge_ids


def _flap_to_bps_dict(
    flap_node_id: int,
    flap: bus.Flap,
    *,
    scale_factor: int = 1,
    sheet_size: int | None = None,
) -> bps_types.FlapDict:
  """Converts a `bus.Flap` into a `BoxPleatingTreeData` `FlapDict`."""
  if flap.circle_centers:
    xs = [c.x * scale_factor for c in flap.circle_centers]
    ys = [c.y * scale_factor for c in flap.circle_centers]
    if sheet_size is not None and sheet_size > 0:
      xs = [max(0, min(sheet_size, x)) for x in xs]
      ys = [max(0, min(sheet_size, y)) for y in ys]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    return {
        'id': int(flap_node_id),
        'x': int(x_min),
        'y': int(y_min),
        'width': int(x_max - x_min),
        'height': int(y_max - y_min),
    }
  rect = flap.rect.scaled(scale_factor)
  if sheet_size is not None and sheet_size > 0:
    rect = rect.clipped(sheet_size)
  cx = rect.x + rect.width // 2
  cy = rect.y + rect.height // 2
  return {
      'id': int(flap_node_id),
      'x': int(cx),
      'y': int(cy),
      'width': 0,
      'height': 0,
  }


def to_bp_studio(
    obj: BpStudioExportable,
    *,
    title: str | None = None,
    description: str | None = None,
    sheet_size: int | None = None,
    tree_sheet_size: int = 20,
    use_merged_tree: bool = False,
) -> bps_types.BoxPleatingTreeData:
  """Exports a design or packed/solved result to `BoxPleatingTreeData` schema.

  Args:
    obj: A `StickFigure`, `DesignTree`, `PackedCreasePattern`,
      `SolvedCreasePattern`, or `PipelineResult`.
    title: Optional design title override. Defaults to the stick figure name.
    description: Optional description string for the design.
    sheet_size: Optional layout grid sheet width/height override.
    tree_sheet_size: Width and height of the 2D tree visualization sheet.
    use_merged_tree: If True, exports `design.merged` instead of
      `design.source`.

  Returns:
    A `BoxPleatingTreeData` dictionary (BP Studio-style) compatible with
    `bps_to_stick_figure` and `from_bp_studio`.
  """
  sf, design, packed, scale_factor = _extract_bps_stages(
      obj, use_merged_tree=use_merged_tree
  )
  _, nodes_out, edges_out, stick_edge_ids = _assign_node_ids_and_coords(
      sf, tree_sheet_size=tree_sheet_size
  )

  if sheet_size is not None and sheet_size > 0:
    layout_size = int(sheet_size)
  elif packed is not None and packed.grid_size > 0:
    layout_size = int(packed.grid_size * scale_factor)
  elif design is not None and design.min_grid_size > 0:
    layout_size = int(design.min_grid_size)
  else:
    layout_size = max(16, int(sf.get_packing_gridsize()))

  degree: dict[int, int] = {}
  for edge in edges_out:
    degree[edge['n1']] = degree.get(edge['n1'], 0) + 1
    degree[edge['n2']] = degree.get(edge['n2'], 0) + 1

  leaf_node_to_stick: dict[int, stick_lib.Stick] = {}
  for s in sf.children:
    if s.name not in stick_edge_ids:
      continue
    n1, n2 = stick_edge_ids[s.name]
    if degree.get(n2, 0) == 1:
      leaf_node_to_stick[n2] = s
    elif degree.get(n1, 0) == 1 and n1 not in leaf_node_to_stick:
      leaf_node_to_stick[n1] = s

  flaps_out: list[bps_types.FlapDict] = []
  emitted_flap_ids: set[int] = set()
  used_bus_flaps: set[str] = set()

  if packed is not None:
    for leaf_id in sorted(leaf_node_to_stick):
      stick = leaf_node_to_stick[leaf_id]
      matched_flap = packed.flap_for_stick(stick.name, design=design)
      if matched_flap is not None:
        flaps_out.append(
            _flap_to_bps_dict(
                leaf_id,
                matched_flap,
                scale_factor=scale_factor,
                sheet_size=layout_size,
            )
        )
        emitted_flap_ids.add(leaf_id)
        used_bus_flaps.add(matched_flap.id)

    unused_flaps = [fid for fid in packed.flaps if fid not in used_bus_flaps]
    if unused_flaps:
      raise ValueError(
          'Cannot export stacked packing to Box Pleating Studio (.bps):'
          f' internal flap(s) {unused_flaps} do not correspond to tree leaf'
          ' nodes.'
      )
  else:
    flaps_out = []

  stretches_out: list[dict[str, Any]] = []
  if packed is not None and packed.rivers:
    for river_id, river in packed.rivers.items():
      rects_payload = [
          {
              'x': int(r.x * scale_factor),
              'y': int(r.y * scale_factor),
              'width': int(r.width * scale_factor),
              'height': int(r.height * scale_factor),
          }
          for r in river.rects
      ]
      bps_stick_name = river.stick
      if river.stick in stick_edge_ids:
        n1, n2 = stick_edge_ids[river.stick]
        bps_stick_name = f'{n1}_{n2}'
      stretches_out.append({
          'id': river_id,
          'stick': bps_stick_name,
          'width': int(river.width * scale_factor),
          'kind': river.kind,
          'rects': rects_payload,
      })

  resolved_title = title if title is not None else (sf.name or 'Origami Design')
  resolved_desc = (
      description
      if description is not None
      else f'Exported by bp_pipeline ({sf.category or "origami"})'
  )
  mode = 'layout' if packed is not None else 'tree'

  bps_dict: bps_types.BoxPleatingTreeData = {
      'version': '0.7',
      'design': {
          'title': resolved_title,
          'description': resolved_desc,
          'mode': mode,
          'layout': {
              'sheet': {
                  'type': 'rect',
                  'width': int(layout_size),
                  'height': int(layout_size),
              },
              'scale_factor': scale_factor,
              'flaps': flaps_out,
              'stretches': stretches_out,
          },
          'tree': {
              'sheet': {
                  'type': 'rect',
                  'width': int(tree_sheet_size),
                  'height': int(tree_sheet_size),
              },
              'nodes': nodes_out,
              'edges': edges_out,
          },
      },
  }
  return bps_dict


def save_bp_studio(
    obj: BpStudioExportable,
    path_or_file: FileTarget,
    *,
    indent: int = 2,
    **kwargs: Any,
) -> None:
  """Writes a `BoxPleatingTreeData` `.bps.json` file to `path_or_file`."""
  data = to_bp_studio(obj, **kwargs)
  _write_text(path_or_file, json.dumps(data, indent=indent) + '\n')


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class BpsImportResult:
  """Result of importing a Box Pleating Studio (.bps) document.

  Attributes:
    design: Canonical `DesignTree` derived from the stick figure tree.
    packed: Validated, solver-ready `PackedCreasePattern` if a valid layout was
      present; None if tree-only or if layout validation failed.
    validation_errors: Tuple of error strings describing geometric or metric
      conflicts if layout validation failed; empty if valid or tree-only.
    first_passing_config: First passing solver configuration from is_bankable if
      bankable; None otherwise.
  """

  design: bus.DesignTree
  packed: bus.PackedCreasePattern | None = None
  validation_errors: tuple[str, ...] = ()
  first_passing_config: hinges.SolverConfig | None = None

  @property
  def stick_figure(self) -> stick_lib.StickFigure:
    """Returns the source stick figure of the design."""
    return self.design.source

  @property
  def is_valid(self) -> bool:
    """Returns True if there are no validation errors."""
    return not self.validation_errors

  @property
  def has_layout(self) -> bool:
    """Returns True if a layout is present (valid or invalid)."""
    return self.packed is not None or bool(self.validation_errors)


@dataclasses.dataclass(frozen=True)
class _ImportedRiver:
  """A minimal rectilinear river imported from external layout data."""

  rectangles: list[rectangle_lib.Rectangle]
  width: int
  grid_size: int

  def asdict(self, name: str = '') -> dict[str, Any]:
    return {
        'name': name,
        'rectangles': [r.asdict() for r in self.rectangles],
        'river_type': 'reconstructed',
        'width': self.width,
        'grid_size': self.grid_size,
    }


def validate_packing(
    design: bus.DesignTree,
    grid_size: int,
    flaps: Sequence[bps_types.FlapDict | Mapping[str, Any]],
    stretches: Sequence[Mapping[str, Any]] = (),
) -> tuple[
    bus.PackedCreasePattern | None, tuple[str, ...], hinges.SolverConfig | None
]:
  """Validates flap placements against geometric and metric box-pleating rules."""
  errors: list[str] = []
  grid_size = int(grid_size)
  if grid_size <= 0:
    errors.append(f'Grid size must be positive, got {grid_size}.')
    return None, tuple(errors), None

  if design.min_grid_size > 0 and grid_size < design.min_grid_size:
    errors.append(
        f'Grid size {grid_size} is smaller than minimum required grid size'
        f' {design.min_grid_size}.'
    )

  node_degree: dict[str, int] = collections.defaultdict(int)
  incident_sticks: dict[str, list[stick_lib.Stick]] = collections.defaultdict(
      list
  )
  for s in design.merged.children:
    node_degree[s.start] += 1
    node_degree[s.end] += 1
    incident_sticks[s.start].append(s)
    incident_sticks[s.end].append(s)

  leaf_sticks_by_id: dict[int | str, stick_lib.Stick] = {}
  for node_id_str, deg in node_degree.items():
    if deg == 1:
      key: int | str = (
          int(node_id_str) if node_id_str.isdigit() else node_id_str
      )
      leaf_sticks_by_id[key] = incident_sticks[node_id_str][0]

  flaps_by_id: dict[int | str, Mapping[str, Any]] = {}
  for f in flaps:
    raw_id = f.get('id')
    f_id: int | str = int(raw_id) if str(raw_id).isdigit() else str(raw_id)
    if f_id in flaps_by_id:
      errors.append(f'Duplicate flap for node ID {f_id}.')
    flaps_by_id[f_id] = f
    if f_id not in leaf_sticks_by_id:
      alt_id = (
          str(f_id)
          if isinstance(f_id, int)
          else (int(f_id) if f_id.isdigit() else None)
      )
      if alt_id is not None and alt_id in leaf_sticks_by_id:
        leaf_sticks_by_id[f_id] = leaf_sticks_by_id[alt_id]
      else:
        errors.append(
            f'Flap ID {f_id} does not correspond to any leaf node in the stick'
            ' figure.'
        )

  for leaf_id, stick in list(leaf_sticks_by_id.items()):
    if leaf_id not in flaps_by_id:
      errors.append(
          f"Leaf node {leaf_id} (stick '{stick.name}') has no flap placement."
      )

  if errors:
    return None, tuple(errors), None

  for leaf_id, flap in flaps_by_id.items():
    stick = leaf_sticks_by_id[leaf_id]
    fx = int(flap.get('x', 0))
    fy = int(flap.get('y', 0))
    fw = int(flap.get('width', 0))
    fh = int(flap.get('height', 0))
    if fx < 0 or fy < 0 or fw < 0 or fh < 0:
      errors.append(
          f"Flap '{stick.name}' (node {leaf_id}) has negative coordinates or"
          f' dimensions: ({fx}, {fy}, {fw}, {fh}).'
      )
    if fx + fw > grid_size or fy + fh > grid_size:
      errors.append(
          f"Flap '{stick.name}' (node {leaf_id}) center bounds [{fx}..{fx+fw},"
          f' {fy}..{fy+fh}] exceed sheet boundaries [0..{grid_size},'
          f' 0..{grid_size}].'
      )

  if errors:
    return None, tuple(errors), None

  tree_adj: dict[str, list[tuple[str, int]]] = collections.defaultdict(list)
  for s in design.merged.children:
    w = max(1, round(s.length))
    tree_adj[s.start].append((s.end, w))
    tree_adj[s.end].append((s.start, w))

  all_pairs_dist: dict[str, dict[str, int]] = {}
  for start_node in tree_adj:
    distances: dict[str, int] = {start_node: 0}
    queue: collections.deque[str] = collections.deque([start_node])
    while queue:
      curr = queue.popleft()
      d_curr = distances[curr]
      for nbr, weight in tree_adj[curr]:
        if nbr not in distances:
          distances[nbr] = d_curr + weight
          queue.append(nbr)
    all_pairs_dist[start_node] = distances

  leaf_list = sorted(leaf_sticks_by_id.keys(), key=str)
  for i in range(len(leaf_list)):
    for j in range(i + 1, len(leaf_list)):
      u_id = leaf_list[i]
      v_id = leaf_list[j]
      u_stick = leaf_sticks_by_id[u_id]
      v_stick = leaf_sticks_by_id[v_id]
      u_str = str(u_id)
      v_str = str(v_id)
      if v_str not in all_pairs_dist.get(u_str, {}):
        errors.append(
            f'Tree metric check failed: nodes {u_id} and {v_id} are'
            ' disconnected in stick figure.'
        )
        continue
      d_tree = all_pairs_dist[u_str][v_str]
      flap_u = flaps_by_id[u_id]
      flap_v = flaps_by_id[v_id]
      xu, yu = int(flap_u.get('x', 0)), int(flap_u.get('y', 0))
      wu, hu = int(flap_u.get('width', 0)), int(flap_u.get('height', 0))
      xv, yv = int(flap_v.get('x', 0)), int(flap_v.get('y', 0))
      wv, hv = int(flap_v.get('width', 0)), int(flap_v.get('height', 0))

      dx = max(0, max(xu, xv) - min(xu + wu, xv + wv))
      dy = max(0, max(yu, yv) - min(yu + hu, yv + hv))
      d_inf = max(dx, dy)
      if d_inf < d_tree:
        errors.append(
            f"Flaps '{u_stick.name}' (node {u_id}) and '{v_stick.name}' (node"
            f' {v_id}) violate tree metric: grid distance {d_inf} < tree path'
            f' distance {d_tree}.'
        )

  tracker = utils_tracker.CreasePatternTracker(grid_size)
  for leaf_id in leaf_list:
    stick = leaf_sticks_by_id[leaf_id]
    r = max(1, round(stick.length))
    flap = flaps_by_id[leaf_id]
    fx, fy = int(flap.get('x', 0)), int(flap.get('y', 0))
    fw = int(flap.get('width', 0))
    fh = int(flap.get('height', 0))
    poly_offset_x = fx - r
    poly_offset_y = fy - r
    poly_w = fw + 2 * r
    poly_h = fh + 2 * r

    x_start = max(0, poly_offset_x)
    y_start = max(0, poly_offset_y)
    x_end = min(grid_size, fx + fw + r)
    y_end = min(grid_size, fy + fh + r)
    rect_w = max(0, x_end - x_start)
    rect_h = max(0, y_end - y_start)
    rect = rectangle_lib.Rectangle(
        name=stick.name,
        x_start=float(x_start),
        y_start=float(y_start),
        x_len=float(rect_w),
        y_len=float(rect_h),
    )
    if not utils_packing_flaps.is_unoccupied(tracker.occupancy_grid, rect):
      overlappers = sorted({
          s.source_id
          for s in tracker.grid[y_start:y_end, x_start:x_end].ravel()
          if s.source_id
      })
      errors.append(
          f"Flap '{stick.name}' (node {leaf_id}) overlaps with"
          f" {', '.join(repr(o) for o in overlappers)} at [{x_start}..{x_end},"
          f' {y_start}..{y_end}].'
      )
    else:
      center_pt = point_lib.Point(
          x=(poly_offset_x + poly_w / 2.0) / grid_size,
          y=(poly_offset_y + poly_h / 2.0) / grid_size,
          require_on_paper=False,
      )
      placement = packing_types.PolygonPlacement(
          offset=(poly_offset_x, poly_offset_y),
          size=(poly_w, poly_h),
          center=center_pt,
      )
      tracker.add_polygon(stick.name, placement)

  leaf_stick_names = {s.name for s in leaf_sticks_by_id.values()}
  internal_sticks = {
      s.name: s
      for s in design.merged.children
      if s.name not in leaf_stick_names
  }

  for stretch in stretches:
    river_id = str(stretch.get('id', ''))
    width = max(1, int(stretch.get('width', 1)))
    stick_name = stretch.get('stick')
    stick_obj = None
    if stick_name:
      if stick_name in internal_sticks:
        stick_obj = internal_sticks[stick_name]
      else:
        parts = str(stick_name).split('_')
        if len(parts) == 2:
          rev_name = f'{parts[1]}_{parts[0]}'
          if rev_name in internal_sticks:
            stick_obj = internal_sticks[rev_name]
        if stick_obj is None:
          for m_name, s_chain in design.merged_from.items():
            if (
                stick_name in s_chain
                or (len(parts) == 2 and f'{parts[1]}_{parts[0]}' in s_chain)
            ) and m_name in internal_sticks:
              stick_obj = internal_sticks[m_name]
              break

      if stick_obj is None:
        if stick_name in leaf_stick_names:
          errors.append(
              f"Stretch river '{river_id}' refers to stick '{stick_name}',"
              ' which is a leaf flap, not an internal branch.'
          )
        else:
          errors.append(
              f"Stretch river '{river_id}' refers to unknown stick"
              f" '{stick_name}' not present in stick figure."
          )
      else:
        expected_len = max(1, round(stick_obj.length))
        if width != expected_len:
          errors.append(
              f"Stretch river '{river_id}' has width {width}, but internal"
              f" stick '{stick_name}' has length {expected_len}."
          )

    tracker_river_key = stick_obj.name if stick_obj is not None else river_id
    rect_dicts = stretch.get('rects', [])
    rects: list[rectangle_lib.Rectangle] = []
    for r_data in rect_dicts:
      rx = int(r_data.get('x', 0))
      ry = int(r_data.get('y', 0))
      rw = int(r_data.get('width', 0))
      rh = int(r_data.get('height', 0))
      if rx < 0 or ry < 0 or rx + rw > grid_size or ry + rh > grid_size:
        errors.append(
            f"River '{river_id}' rect [{rx}..{rx+rw}, {ry}..{ry+rh}] exceeds"
            f' sheet bounds [0..{grid_size}, 0..{grid_size}].'
        )
      r_rect = rectangle_lib.Rectangle(
          name=tracker_river_key,
          x_start=float(rx),
          y_start=float(ry),
          x_len=float(rw),
          y_len=float(rh),
      )
      rects.append(r_rect)
      if not utils_packing_flaps.is_unoccupied(tracker.occupancy_grid, r_rect):
        overlappers = sorted({
            s.source_id
            for s in tracker.grid[ry : ry + rh, rx : rx + rw].ravel()
            if s.source_id
        })
        errors.append(
            f"River '{river_id}' overlaps with"
            f" {', '.join(repr(o) for o in overlappers)} at [{rx}..{rx+rw},"
            f' {ry}..{ry+rh}].'
        )
      else:
        utils_packing_flaps.set_grid_occupation(
            grid=tracker.occupancy_grid,
            rect=r_rect,
            occupied=True,
            metadata_grid=tracker.grid,
            state=utils_packing_flaps.GridCellState(
                type=utils_packing_flaps.GridCellType.RIVER,
                source_id=tracker_river_key,
            ),
        )
    if rects and not errors:
      factory = river_lib.get_reconstructed_river_factory()
      if factory is not None:
        tracker.packed_rivers[tracker_river_key] = factory(
            rectangles=rects,
            width=width,
            grid_size=grid_size,
        )
      else:
        tracker.packed_rivers[tracker_river_key] = _ImportedRiver(
            rectangles=rects,
            width=width,
            grid_size=grid_size,
        )

  if errors:
    return None, tuple(errors), None

  unused_mask = ~tracker.occupancy_grid
  num_unused = int(unused_mask.sum())
  if num_unused > 0:
    ys, xs = np.where(unused_mask)
    sample_locs = [f'({xs[k]},{ys[k]})' for k in range(min(5, len(xs)))]
    suffix = '...' if len(xs) > 5 else ''
    errors.append(
        f'Layout leaves {num_unused} uncovered cells (zero unused area'
        f' required); e.g. at {", ".join(sample_locs)}{suffix}.'
    )
    return None, tuple(errors), None

  try:
    y_ridge_at_border = True
    flap_shapes = compute_ridges.compute_flap_shapes(
        tracker, y_ridge_at_border=y_ridge_at_border
    )
    packed_sf = copy.deepcopy(design.merged)
    for s in packed_sf.children:
      if s.name in flap_shapes:
        s.flap_tip_shape = flap_shapes[s.name]
      else:
        s.flap_tip_shape = stick_lib.FlapTipShape.NOT_APPLICABLE
    unmerged_sf = copy.deepcopy(design.source)
    for s in unmerged_sf.children:
      if s.name in flap_shapes:
        s.flap_tip_shape = flap_shapes[s.name]
      else:
        s.flap_tip_shape = stick_lib.FlapTipShape.NOT_APPLICABLE
    creases, flap_rects = utils_packing_svg.build_packing_crease_pattern(
        grid_size, tracker, y_ridge_at_border=y_ridge_at_border
    )
    hinges_info = utils_packing_svg.extract_hinges_info(tracker, grid_size)
    p_output = packing_output_lib.PackingOutput(
        success=True,
        packing_cp=creases,
        flap_rectangles=flap_rects,
        packing_rivers=dict(tracker.packed_rivers),
        stick_figure=packed_sf,
        unmerged_stick_figure=unmerged_sf,
        details={
            'grid_size': grid_size,
            'is_symmetric': design.is_symmetric,
            'perfect_tiling_found': True,
            'unused_area': 0,
        },
        hinges=hinges_info,
        tracker=tracker,
        config=packing_config_lib.PackingConfig(),
    )

    bank_res = solvability.is_bankable(p_output, design=design)
    if not bank_res.ok:
      errors.extend(bank_res.reasons)
      return None, tuple(errors), None

    packed_cp = bus.PackedCreasePattern.from_packing_output(
        p_output, design=design, grid_size=grid_size
    )
    return packed_cp, (), bank_res.first_passing_config
  except (
      ValueError,
      TypeError,
      KeyError,
      packing_output_lib.PackingValidationError,
  ) as err:
    errors.append(f'Crease pattern synthesis or validation failed: {err}')
    return None, tuple(errors), None


def import_bps(
    bps_data_or_json: bps_types.BoxPleatingTreeData | Mapping[str, Any] | str,
) -> BpsImportResult:
  """Imports a Box Pleating Studio (.bps) document with geometric validation."""
  if isinstance(bps_data_or_json, str):
    parsed = json.loads(bps_data_or_json)
  else:
    parsed = dict(bps_data_or_json)

  sf = stick_converters.bps_to_stick_figure(
      cast(bps_types.BoxPleatingTreeData, parsed)
  )
  design = bus.DesignTree.from_stick_figure(sf)

  design_dict = parsed.get('design', {})
  mode = str(design_dict.get('mode', 'tree')).lower().strip()
  layout = design_dict.get('layout') or {}
  flaps = layout.get('flaps') or []
  stretches = layout.get('stretches') or []

  if mode == 'tree' or not flaps:
    return BpsImportResult(
        design=design,
        packed=None,
        validation_errors=(),
        first_passing_config=None,
    )

  sheet = layout.get('sheet') or {}
  grid_size = int(
      sheet.get('width') or sheet.get('height') or design.min_grid_size
  )

  scale_factor = int(layout.get('scale_factor', 1))
  scale_errors: list[str] = []
  if scale_factor > 1:
    if grid_size % scale_factor != 0:
      scale_errors.append(
          f'Grid size {grid_size} is not divisible by scale_factor'
          f' {scale_factor}.'
      )
    for f in flaps:
      f_id = f.get('id')
      for k in ('x', 'y', 'width', 'height'):
        val = int(f.get(k, 0))
        if val % scale_factor != 0:
          scale_errors.append(
              f"Flap {f_id} coordinate '{k}'={val} is not divisible by"
              f' scale_factor {scale_factor}.'
          )
    for st in stretches:
      st_id = st.get('id')
      st_w = int(st.get('width', 1))
      if st_w % scale_factor != 0:
        scale_errors.append(
            f"Stretch '{st_id}' width {st_w} is not divisible by scale_factor"
            f' {scale_factor}.'
        )
      for r in st.get('rects', []):
        for k in ('x', 'y', 'width', 'height'):
          val = int(r.get(k, 0))
          if val % scale_factor != 0:
            scale_errors.append(
                f"Stretch '{st_id}' rect coordinate '{k}'={val} is not"
                f' divisible by scale_factor {scale_factor}.'
            )
    if scale_errors:
      return BpsImportResult(
          design=design,
          packed=None,
          validation_errors=tuple(scale_errors),
          first_passing_config=None,
      )

    grid_size = max(1, grid_size // scale_factor)
    scaled_flaps = []
    for f in flaps:
      scaled_flaps.append({
          'id': f.get('id'),
          'x': int(f.get('x', 0)) // scale_factor,
          'y': int(f.get('y', 0)) // scale_factor,
          'width': int(f.get('width', 0)) // scale_factor,
          'height': int(f.get('height', 0)) // scale_factor,
      })
    flaps = scaled_flaps
    scaled_stretches = []
    for st in stretches:
      s_rects = [
          {
              'x': int(r.get('x', 0)) // scale_factor,
              'y': int(r.get('y', 0)) // scale_factor,
              'width': int(r.get('width', 0)) // scale_factor,
              'height': int(r.get('height', 0)) // scale_factor,
          }
          for r in st.get('rects', [])
      ]
      scaled_stretches.append({
          'id': st.get('id'),
          'stick': st.get('stick'),
          'width': int(st.get('width', 1)) // scale_factor,
          'kind': st.get('kind', 'straight'),
          'rects': s_rects,
      })
    stretches = scaled_stretches

  packed_cp, errors, first_passing_config = validate_packing(
      design,
      grid_size=grid_size,
      flaps=flaps,
      stretches=stretches,
  )
  return BpsImportResult(
      design=design,
      packed=packed_cp,
      validation_errors=errors,
      first_passing_config=first_passing_config,
  )


def from_bp_studio(
    bps_data_or_json: bps_types.BoxPleatingTreeData | Mapping[str, Any] | str,
) -> stick_figure_lib.StickFigure:
  """Parses a `BoxPleatingTreeData` dict or JSON string into a `StickFigure`."""
  return import_bps(bps_data_or_json).design.source


def _extract_cp_and_grid(
    obj: CreasePatternExportable,
    *,
    grid_size: int | None = None,
) -> tuple[cp_lib.CreasePattern, int, str | None]:
  """Extracts `(CreasePattern, resolved_grid_size, optional_title)`."""
  if isinstance(obj, bus.PipelineResult):
    shaped_obj = obj.shaped
    if isinstance(shaped_obj, bus.ShapedCreasePattern):
      g = grid_size if grid_size is not None else shaped_obj.grid_size
      return shaped_obj.creases, g, obj.design.source.name
    g = grid_size if grid_size is not None else obj.solved.grid_size
    return obj.solved.creases, g, obj.design.source.name
  if isinstance(obj, bus.ShapedCreasePattern):
    g = grid_size if grid_size is not None else obj.grid_size
    title = (
        obj.solved.packed.design.source.name
        if obj.solved.packed.design is not None
        else None
    )
    return obj.creases, g, title
  if isinstance(obj, bus.SolvedCreasePattern):
    g = grid_size if grid_size is not None else obj.grid_size
    title = (
        obj.packed.design.source.name if obj.packed.design is not None else None
    )
    return obj.creases, g, title
  if isinstance(obj, bus.PackedCreasePattern):
    g = grid_size if grid_size is not None else obj.grid_size
    title = obj.design.source.name if obj.design is not None else None
    return obj.creases, g, title
  if isinstance(obj, cp_lib.CreasePattern):
    g = grid_size if grid_size is not None else obj.grid_size
    return obj, g, None
  raise TypeError(
      'Expected CreasePattern, PackedCreasePattern, SolvedCreasePattern,'
      f' or PipelineResult; got {type(obj)!r}'
  )


def _resolve_color_map(
    colors: Mapping[crease_lib.Fold | str, str] | None,
) -> dict[crease_lib.Fold, str]:
  """Builds a complete `Fold -> CSS color` mapping from user overrides."""
  cmap = dict(DEFAULT_SVG_COLORS)
  if not colors:
    return cmap
  for key, color_val in colors.items():
    if isinstance(key, crease_lib.Fold):
      cmap[key] = color_val
      continue
    norm_key = str(key).strip().lower()
    if norm_key in _COLOR_KEY_ALIASES:
      for fold_enum in _COLOR_KEY_ALIASES[norm_key]:
        cmap[fold_enum] = color_val
      continue
    matched = False
    for fold_enum in crease_lib.Fold:
      if fold_enum.name.lower() == norm_key or fold_enum.value == norm_key:
        cmap[fold_enum] = color_val
        matched = True
        break
    if not matched:
      raise ValueError(f'Unknown crease fold key in colors mapping: {key!r}')
  return cmap


def to_svg(
    obj: CreasePatternExportable,
    *,
    grid_size: int | None = None,
    include_grid: bool = False,
    merge_creases: bool = True,
    pretty: bool = True,
    replace_border: bool = True,
    dashed_valley: bool = False,
    stroke_width: float | int | str | None = None,
    colors: Mapping[crease_lib.Fold | str, str] | None = None,
    scale: float = 1000.0,
    ndigits: int = 8,
) -> str:
  """Renders a crease pattern or pipeline stage object to a clean SVG string.

  Args:
    obj: A `CreasePattern`, `PackedCreasePattern`, `SolvedCreasePattern`, or
      `PipelineResult`.
    grid_size: Optional grid size override. When positive, coordinates are
      emitted in `[0, grid_size]` grid units.
    include_grid: If True, renders background grid lines inside a `<g>` group.
    merge_creases: If True, merges collinear contiguous creases before
      rendering.
    pretty: If True, formats the XML output with newlines and indentation.
    replace_border: If True, emits the outer boundary as a `<rect>` element.
    dashed_valley: If True, adds `stroke-dasharray` to valley creases.
    stroke_width: Optional stroke width applied to crease elements.
    colors: Optional custom color overrides keyed by `Fold` or fold name
      (`'border'`, `'mountain'`, `'valley'`, `'hinge'`, `'ridge'`, `'axial'`).
    scale: ViewBox scale when no positive `grid_size` is active.
    ndigits: Number of decimal digits for coordinate formatting.

  Returns:
    A valid vector SVG string representing the crease pattern.
  """
  cp, resolved_grid, _ = _extract_cp_and_grid(obj, grid_size=grid_size)
  if merge_creases:
    merged = crease_lib.merge_creases(cp.creases)
    render_cp = cp_lib.CreasePattern(merged, light_mode=True)
  else:
    render_cp = cp

  render_grid = resolved_grid if (resolved_grid and resolved_grid > 0) else -1
  resolved_colors = _resolve_color_map(colors) if colors is not None else None
  if stroke_width is not None:
    effective_stroke_width = stroke_width
  elif render_grid > 0:
    effective_stroke_width = round(max(1, render_grid) / 450.0, 4)
  else:
    effective_stroke_width = None
  return svg_parser.to_svg(
      render_cp,
      scale=scale,
      ndigits=ndigits,
      grid=render_grid,
      pretty=pretty,
      replace_border=replace_border,
      colors=resolved_colors,
      stroke_width=effective_stroke_width,
      dashed_valley=dashed_valley,
      grid_lines=include_grid,
  )


def save_svg(
    obj: CreasePatternExportable,
    path_or_file: FileTarget,
    **kwargs: Any,
) -> None:
  """Writes an SVG representation of `obj` to `path_or_file`."""
  svg_str = to_svg(obj, **kwargs)
  _write_text(path_or_file, svg_str)


def compute_faces_vertices(
    vertices_coords: Sequence[Sequence[float]],
    edges_vertices: Sequence[Sequence[int]],
) -> list[list[int]]:
  """Computes bounded planar `faces_vertices` from vertices and edges."""
  return fold_3d.extract_planar_faces(vertices_coords, edges_vertices)


def to_fold(
    obj: CreasePatternExportable,
    *,
    title: str | None = None,
    merge_creases: bool = False,
    hinge_assignment: FoldAssignment = 'U',
    include_faces: bool = False,
    ndigits: int | None = 8,
) -> dict[str, Any]:
  """Exports a crease pattern or pipeline stage object to FOLD v1.2 format.

  Args:
    obj: A `CreasePattern`, `PackedCreasePattern`, `SolvedCreasePattern`, or
      `PipelineResult`.
    title: Optional `file_title` metadata field.
    merge_creases: If True, merges collinear creases before extracting vertices.
    hinge_assignment: FOLD edge assignment for `Fold.HINGE` creases (`'U'` by
      default, or `'F'`).
    include_faces: If True, computes planar `faces_vertices`.
    ndigits: Optional decimal rounding precision for `vertices_coords`.

  Returns:
    A FOLD v1.2 dictionary containing `file_spec`, `file_creator`,
    `frame_classes`, `vertices_coords`, `edges_vertices`, `edges_assignment`,
    and `edges_foldAngle`.
  """
  cp, resolved_grid, inferred_title = _extract_cp_and_grid(obj)
  if merge_creases:
    merged = crease_lib.merge_creases(cp.creases)
    cp = cp_lib.CreasePattern(
        merged,
        light_mode=False,
        grid_size=resolved_grid if resolved_grid else -1,
    )

  def _fmt_coord(val: float) -> float:
    if ndigits is None:
      return float(val)
    rounded = round(float(val), ndigits)
    return 0.0 if rounded == 0.0 else rounded

  key_digits = ndigits if ndigits is not None else 6
  vertices_coords: list[list[float]] = []
  vertex_index_by_key: dict[tuple[float, float], int] = {}

  def _find_or_add_vertex(pt: point_lib.Point) -> int:
    cx = _fmt_coord(pt.x)
    cy = _fmt_coord(pt.y)
    key = (round(cx, key_digits), round(cy, key_digits))
    existing = vertex_index_by_key.get(key)
    if existing is not None:
      return existing
    idx = len(vertices_coords)
    vertices_coords.append([cx, cy])
    vertex_index_by_key[key] = idx
    return idx

  for v in cp.vertices:
    _find_or_add_vertex(v.point)

  edges_vertices: list[list[int]] = []
  edges_assignment: list[str] = []
  edges_fold_angle: list[float] = []

  for c in cp.creases:
    u = _find_or_add_vertex(c.p1)
    v = _find_or_add_vertex(c.p2)
    edges_vertices.append([u, v])
    if c.fold == crease_lib.Fold.HINGE:
      assignment = hinge_assignment
    else:
      assignment = _FOLD_TO_ASSIGNMENT.get(c.fold, 'U')
    edges_assignment.append(assignment)
    sign = _FOLD_TO_ANGLE_SIGN.get(c.fold, 0.0)
    angle = round(sign * float(c.fold_percent), 6)
    edges_fold_angle.append(0.0 if angle == 0.0 else angle)

  fold_dict: dict[str, Any] = {
      'file_spec': 1.2,
      'file_creator': 'bp_pipeline',
      'frame_classes': ['creasePattern'],
  }
  resolved_title = title if title is not None else inferred_title
  if resolved_title:
    fold_dict['file_title'] = resolved_title

  fold_dict['vertices_coords'] = vertices_coords
  fold_dict['edges_vertices'] = edges_vertices
  fold_dict['edges_assignment'] = edges_assignment
  fold_dict['edges_foldAngle'] = edges_fold_angle

  if include_faces:
    fold_dict['faces_vertices'] = compute_faces_vertices(
        vertices_coords, edges_vertices
    )

  return fold_dict


def save_fold(
    obj: CreasePatternExportable,
    path_or_file: FileTarget,
    *,
    indent: int = 2,
    **kwargs: Any,
) -> None:
  """Writes a FOLD v1.2 `.fold` JSON file to `path_or_file`."""
  fold_dict = to_fold(obj, **kwargs)
  _write_text(path_or_file, json.dumps(fold_dict, indent=indent) + '\n')


def from_fold(
    fold_data_or_json: Mapping[str, Any] | str,
    *,
    grid_size: int = -1,
    light_mode: bool = False,
) -> cp_lib.CreasePattern:
  """Converts a FOLD creasePattern dict or JSON string into a `CreasePattern`."""
  if isinstance(fold_data_or_json, str):
    data = json.loads(fold_data_or_json)
  else:
    data = dict(fold_data_or_json)

  for required_key in ('vertices_coords', 'edges_vertices', 'edges_assignment'):
    if required_key not in data:
      raise ValueError(f'Missing required FOLD key {required_key!r}.')

  vertices_coords = data['vertices_coords']
  edges_vertices = data['edges_vertices']
  edges_assignment = data['edges_assignment']
  edges_fold_angle = data.get('edges_foldAngle')

  assignment_to_fold: Mapping[str, crease_lib.Fold] = {
      'B': crease_lib.Fold.BORDER,
      'M': crease_lib.Fold.MOUNTAIN,
      'V': crease_lib.Fold.VALLEY,
      'F': crease_lib.Fold.TRIANGLE,
      'U': crease_lib.Fold.HINGE,
      'C': crease_lib.Fold.BORDER,
  }

  creases: list[crease_lib.Crease] = []
  for idx, (u, v) in enumerate(edges_vertices):
    c1 = vertices_coords[u]
    c2 = vertices_coords[v]
    p1 = point_lib.Point(x=float(c1[0]), y=float(c1[1]))
    p2 = point_lib.Point(x=float(c2[0]), y=float(c2[1]))
    assign = str(edges_assignment[idx]).upper()
    fold_type = assignment_to_fold.get(assign, crease_lib.Fold.HINGE)
    fold_percent = 1.0
    if (
        edges_fold_angle is not None
        and idx < len(edges_fold_angle)
        and fold_type in (crease_lib.Fold.MOUNTAIN, crease_lib.Fold.VALLEY)
    ):
      raw_angle = abs(float(edges_fold_angle[idx]))
      if raw_angle > 0.0:
        fold_percent = min(1.0, max(0.001, raw_angle / 180.0))
    creases.append(
        crease_lib.Crease(
            p1,
            p2,
            fold=fold_type,
            fold_percent=fold_percent,
        )
    )

  return cp_lib.CreasePattern(
      creases,
      light_mode=light_mode,
      grid_size=grid_size,
  )


__all__ = [
    'BpStudioExportable',
    'BpsImportResult',
    'CreasePatternExportable',
    'DEFAULT_SVG_COLORS',
    'FoldAssignment',
    'compute_faces_vertices',
    'from_bp_studio',
    'from_fold',
    'import_bps',
    'save_bp_studio',
    'save_fold',
    'save_svg',
    'to_bp_studio',
    'to_fold',
    'to_svg',
    'validate_packing',
]
