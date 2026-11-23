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

"""Immutable shaping state, configuration, and shared verification helpers."""

from collections.abc import Callable, Mapping, Sequence
import copy
import dataclasses
import functools
import json
import math
import re
from typing import Any

import numpy as np

from .. import bus
from .. import folder_3d
from ..crease_pattern import crease as crease_utils
from ..crease_pattern import crease_pattern as crease_pattern_lib
from ..crease_pattern import point as point_utils
from ..crease_pattern.parser import svg as svg_parser
from ..packing import compute_ridges
from ..packing import config as packing_config_lib
from ..packing import packing_output as packing_output_lib
from ..packing import types as packing_types
from ..solving import hinges
from ..stick_figures import angle_utils
from ..stick_figures import stick as stick_lib
from ..stick_figures import stick_figure as stick_figure_lib
from . import bus_adapter
from . import clip_pattern
from . import fidelity
from . import folded_region_fold
from . import geometric_fold_3d_heuristics
from . import geometric_simple_fold
from . import heuristics
from . import heuristics_3d
from . import heuristics_utils
from . import overlay_3d
from . import paper_layout as paper_layout_lib
from . import pose_alignment
from . import shaping_elements
from . import shaping_utils

ShapingFlap = shaping_elements.ShapingFlap
ShapingRiver = shaping_elements.ShapingRiver
ShapingElement = ShapingFlap | ShapingRiver

# Maximum deterministic-folder vertex strain (%) accepted for shaped patterns.
MAX_RENDERABLE_VERTEX_ERROR = 1.0

# Minimum non-degenerate segment length in normalized paper coordinates.
MIN_SEGMENT_LENGTH = 5e-4

_LEFT_RIGHT_WORDS = (
    ('_left', '_right'),
    ('_right', '_left'),
    ('left_', 'right_'),
    ('right_', 'left_'),
    ('-left', '-right'),
    ('-right', '-left'),
)
_LR_DIGIT_RE = re.compile(r'^(.*[_-])([LlRr])(\d+(?:[_-].*)?)$')


@dataclasses.dataclass(frozen=True, kw_only=True)
class ToolConfig:
  """Options governing iterative shaping actuators.

  Attributes:
    check_flat_foldability: Reject edits that degrade vertex foldability.
    use_heuristics_for_fold_order: Choose zig-zag mountain/valley order from the
      stick figure and base hinges when `fold_order` is omitted.
    use_heuristics_for_depth: Adjust zig-zag depth to avoid base hinges.
    use_heuristics_for_phase: Infer zig-zag phase from tree orientations.
    convert_small_angles: Split tiny zig-zag turns into two larger turns.
    clip_slope_thresh: Slope threshold above which `rotation_3d` clips lines.
    max_slope_thresh: Slope threshold above which `rotation_3d` skips lines.
    fold_strategy_3d: Strategy used by `rotation_3d`.
    check_all_crease_intersections: Re-check all pairwise crease intersections
      when compiling shaped patterns (slower, stricter).
    cp_tolerance: Point/intersection tolerance for shaped patterns.
    use_tree_root: Anchor 3-D rotation target angles to `tree.root` rather than
      the longest outer chain, preserving body-centric poses.
    fan_out_time_budget_s: Wall-clock budget (seconds) for one `fan_out_fold`.
    fan_out_solver_timeout_s: Cooperative timeout (seconds) per flat-foldable
      solve step inside `fan_out_fold`.
  """

  check_flat_foldability: bool = True
  use_heuristics_for_fold_order: bool = True
  use_heuristics_for_depth: bool = True
  use_heuristics_for_phase: bool = False
  convert_small_angles: bool = True
  clip_slope_thresh: float = 20.0
  max_slope_thresh: float = 20.0
  fold_strategy_3d: heuristics_3d.FoldStrategy3D = (
      heuristics_3d.FoldStrategy3D.DEFAULT
  )
  check_all_crease_intersections: bool = False
  cp_tolerance: float = crease_utils.DEFAULT_TOLERANCE
  use_tree_root: bool = True
  fan_out_time_budget_s: float = 20.0
  fan_out_solver_timeout_s: float = 2.0

  @property
  def apply_simple_fold_heuristics(self) -> bool:
    """Whether any simple-fold heuristic is enabled."""
    return (
        self.use_heuristics_for_fold_order
        or self.use_heuristics_for_depth
        or self.use_heuristics_for_phase
        or self.convert_small_angles
    )


@dataclasses.dataclass(frozen=True, kw_only=True)
class ToolResult:
  """Recorded outcome of a single `apply_tool` invocation.

  Attributes:
    tool: Canonical tool name.
    arguments: JSON-serializable arguments (resolved with chosen sub-plans when
      a search tool succeeds, so replaying `history` is exact).
    ok: Whether the tool succeeded and updated the state.
    message: Human-readable summary or error message.
    num_creases: Crease count of `state.pattern` after the call.
  """

  tool: str
  arguments: dict[str, Any]
  ok: bool
  message: str
  num_creases: int

  def asdict(self) -> dict[str, Any]:
    """Converts the result into a plain dictionary."""
    return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True, kw_only=True)
class ToolOutcome:
  """Internal return bundle carrying an updated state plus structured metadata.

  Attributes:
    state: Updated `ShapingState`.
    note: Concise diagnostic note appended to `ToolResult.message`.
    resolved_arguments: Optional argument overrides stored in `state.history` to
      make deterministic replay possible without re-searching.
  """

  state: 'ShapingState'
  note: str = ''
  resolved_arguments: dict[str, Any] | None = None


def _with_tolerance(
    pattern: crease_pattern_lib.CreasePattern, tolerance: float
) -> crease_pattern_lib.CreasePattern:
  """Returns `pattern` or a shallow copy with `tolerance` applied."""
  if math.isclose(pattern.tolerance, tolerance, rel_tol=0.0, abs_tol=1e-15):
    return pattern
  cloned = copy.copy(pattern)
  cloned.tolerance = tolerance
  return cloned


# Cached properties valid as long as base_pattern / packing / solving hold.
_PACKING_CACHED_PROPERTIES = (
    'hinge_segments',
    'merged_shaping_elements',
    'base_graph_data',
    'base_model',
)
# Cached properties that additionally depend on `tree` and `config`.
_TREE_CACHED_PROPERTIES = (
    'paper_layout',
    'shaping_heuristic_data',
    'unmerged_shaping_elements',
    'root_frame_spec',
)
# Cached properties that additionally depend on `narrowed_elements`.
_NARROWING_CACHED_PROPERTIES = (
    'shaping_elements',
    'flap_hierarchy',
    'clip_pattern_adapter',
)


@dataclasses.dataclass(frozen=True)
class ShapingState:
  """Immutable context and current crease pattern for iterative 3-D shaping.

  Every actuator takes a `ShapingState` and returns an updated copy via
  `replace` / `with_pattern`, leaving the input untouched. Expensive derived
  structures (shaping elements, heuristic metadata, flap hierarchy, folded
  graph, paper layout) are computed lazily and shared across updates when their
  underlying inputs have not changed.

  Attributes:
    tree: Active unmerged stick figure (with assigned flap tip shapes).
    packing: Packing output associated with the base pattern.
    solving: Solver result containing the unshaped flat-foldable base.
    pattern: Current crease pattern.
    base_pattern: Original flat-foldable base crease pattern.
    config: Actuator options.
    narrowed_elements: Elements narrowed by grid-doubling zig-zag folds.
    history: Sequence of `ToolResult.asdict()` records applied so far.
    flat_backbone: Latest 2-D flat-foldable backbone when `pattern` carries a
      partial-fold 3-D overlay, or `None`.
    base_tree: Initial unmerged stick figure preserved for `reset()`.
  """

  tree: stick_figure_lib.StickFigure
  packing: packing_output_lib.PackingOutput
  solving: hinges.SolverResult
  pattern: crease_pattern_lib.CreasePattern
  base_pattern: crease_pattern_lib.CreasePattern
  config: ToolConfig = ToolConfig()
  narrowed_elements: tuple[str, ...] = ()
  history: tuple[dict[str, Any], ...] = ()
  flat_backbone: crease_pattern_lib.CreasePattern | None = None
  base_tree: stick_figure_lib.StickFigure | None = None

  @classmethod
  def create(
      cls,
      tree: stick_figure_lib.StickFigure,
      packing: packing_output_lib.PackingOutput,
      solving: hinges.SolverResult,
      *,
      pattern: crease_pattern_lib.CreasePattern | None = None,
      flat_backbone: crease_pattern_lib.CreasePattern | None = None,
      narrowed_elements: Sequence[str] = (),
      history: Sequence[Mapping[str, Any]] = (),
      config: ToolConfig = ToolConfig(),
  ) -> 'ShapingState':
    """Constructs a `ShapingState` without mutating caller inputs."""
    tree_copy = copy.deepcopy(tree)
    if any(
        s.is_flap and s.flap_tip_shape in (None, stick_lib.FlapTipShape.UNSET)
        for s in tree_copy.children
    ):
      compute_ridges.assign_flap_shapes_from_packing_output(tree_copy, packing)
    base_pattern = _with_tolerance(solving.crease_pattern, config.cp_tolerance)
    if pattern is None:
      active_pattern = base_pattern
      resolved_backbone = None
    else:
      active_pattern = _with_tolerance(pattern, config.cp_tolerance)
      if flat_backbone is not None:
        resolved_backbone = _with_tolerance(flat_backbone, config.cp_tolerance)
      elif overlay_3d.has_3d_overlay(active_pattern):
        resolved_backbone = (
            overlay_3d.as_flat_backbone(active_pattern)
            if active_pattern.local_flat_foldability()
            else base_pattern
        )
      else:
        resolved_backbone = None
    return cls(
        tree=tree_copy,
        packing=packing,
        solving=solving,
        pattern=active_pattern,
        base_pattern=base_pattern,
        config=config,
        narrowed_elements=tuple(dict.fromkeys(narrowed_elements)),
        history=tuple(dict(item) for item in history),
        flat_backbone=resolved_backbone,
        base_tree=tree_copy,
    )

  @classmethod
  def from_solved(
      cls,
      solved: bus.SolvedCreasePattern,
      *,
      pattern: crease_pattern_lib.CreasePattern | None = None,
      config: ToolConfig = ToolConfig(),
  ) -> 'ShapingState':
    """Builds a `ShapingState` from a `SolvedCreasePattern` bus artifact."""
    packing_out, solver_res = bus_adapter.legacy_inputs(solved)
    tree = (
        packing_out.unmerged_stick_figure
        if packing_out.unmerged_stick_figure is not None
        else packing_out.stick_figure
    )
    if tree is None:
      raise ValueError('SolvedCreasePattern has no associated StickFigure.')
    return cls.create(
        tree, packing_out, solver_res, pattern=pattern, config=config
    )

  def to_shaped(
      self,
      solved: bus.SolvedCreasePattern,
      *,
      strategy: bus.ShapingStrategy | str = bus.ShapingStrategy.TOOLBOX,
      time_taken: float = 0.0,
      diagnostics: Mapping[str, Any] | None = None,
  ) -> bus.ShapedCreasePattern:
    """Wraps `self.pattern` into an immutable `ShapedCreasePattern` record."""
    return bus.ShapedCreasePattern.from_crease_pattern(
        self.pattern,
        solved=solved,
        strategy=strategy,
        time_taken=time_taken,
        diagnostics=diagnostics,
    )

  def to_metadata_dict(self) -> dict[str, Any]:
    """Exports compact JSON-serializable session metadata (no inline SVGs)."""
    data: dict[str, Any] = {
        'version': 1,
        'grid_size': self.pattern.grid_size,
        'tolerance': self.pattern.tolerance,
        'use_tree_root': self.config.use_tree_root,
        'narrowed_elements': list(self.narrowed_elements),
        'has_flat_backbone': self.flat_backbone is not None,
        'history': [dict(entry) for entry in self.history],
    }
    if self.base_tree is not None and self.tree is not self.base_tree:
      data['tree'] = self.tree.to_dict()
    return data

  @classmethod
  def from_metadata_dict(
      cls,
      solved: bus.SolvedCreasePattern,
      metadata: Mapping[str, Any],
      *,
      pattern_svg: str,
      backbone_svg: str | None = None,
      config: ToolConfig | None = None,
  ) -> 'ShapingState':
    """Restores a `ShapingState` from `to_metadata_dict()` and SVG files."""
    tol = float(metadata.get('tolerance', crease_utils.DEFAULT_TOLERANCE))
    use_root = bool(metadata.get('use_tree_root', True))
    resolved_cfg = config or ToolConfig(
        cp_tolerance=tol, use_tree_root=use_root
    )
    pattern = deserialize_pattern_svg(
        pattern_svg, grid_size=-1, tolerance=resolved_cfg.cp_tolerance
    )
    backbone = None
    if backbone_svg is not None:
      backbone = deserialize_pattern_svg(
          backbone_svg,
          grid_size=-1,
          tolerance=resolved_cfg.cp_tolerance,
      )
    base = cls.from_solved(solved, pattern=pattern, config=resolved_cfg)
    tree = base.tree
    if 'tree' in metadata and isinstance(metadata['tree'], Mapping):
      tree = stick_figure_lib.StickFigure.from_dict(dict(metadata['tree']))
      if any(
          s.is_flap and s.flap_tip_shape in (None, stick_lib.FlapTipShape.UNSET)
          for s in tree.children
      ):
        compute_ridges.assign_flap_shapes_from_packing_output(
            tree, base.packing
        )
    return base.replace(
        tree=tree,
        flat_backbone=backbone if backbone is not None else base.flat_backbone,
        narrowed_elements=tuple(metadata.get('narrowed_elements', ())),
        history=tuple(dict(item) for item in metadata.get('history', ())),
    )

  # -- Derived properties ------------------------------------------------------

  @property
  def packing_cp(self) -> crease_pattern_lib.CreasePattern:
    """Packing crease pattern."""
    if not isinstance(
        self.packing.packing_cp, crease_pattern_lib.CreasePattern
    ):
      raise ValueError('Packing output has no crease pattern.')
    return self.packing.packing_cp

  @property
  def packing_grid_size(self) -> int:
    """Integer grid size of the packing."""
    return self.packing_cp.grid_size

  @property
  def solving_grid_size(self) -> int:
    """Integer grid size of the solved base pattern."""
    return self.base_pattern.grid_size

  @property
  def central_crease_removed(self) -> bool:
    """Whether the central axis crease was removed during solving."""
    return bool(self.solving.final_state.solver_config.remove_central_crease)

  @property
  def flap_rectangles(self) -> list[shaping_utils.Rectangle]:
    """Packing flap rectangles for the merged stick figure."""
    if self.packing.flap_rectangles is None:
      raise ValueError('Packing output has no flap rectangles.')
    return self.packing.flap_rectangles

  @property
  def packing_config(self) -> packing_config_lib.PackingConfig:
    """Packing configuration (default instance when absent)."""
    return self.packing.config or packing_config_lib.PackingConfig()

  def updated_hinge_info(self) -> list[Any]:
    """Returns packing hinges updated with the solver's M/V assignments."""
    hinge_info_list = copy.deepcopy(self.packing.hinges or [])
    return heuristics.update_hinge_info_from_solver_result(
        hinge_info_list, self.solving
    )

  @functools.cached_property
  def hinge_segments(self) -> list[crease_utils.Crease]:
    """Non-empty hinge crease segments from `updated_hinge_info`."""
    return [
        hinge.coordinate
        for hinge in self.updated_hinge_info()
        if hinge.coordinate is not None
    ]

  @functools.cached_property
  def paper_layout(self) -> paper_layout_lib.PaperLayout:
    """Paper-space layout of joints and contours for fidelity measurement."""
    return paper_layout_lib.build_paper_layout(self.tree, self.packing)

  @functools.cached_property
  def merged_shaping_elements(
      self,
  ) -> tuple[list[ShapingFlap], list[ShapingRiver]]:
    """Shaping flaps and rivers corresponding to the merged packing tree."""
    return shaping_elements.convert_packing_elements_to_shaping_elements(
        packing_rectangles=self.flap_rectangles,
        packing_rivers=self.packing.packing_rivers,
        packing_grid_size=self.packing_grid_size,
        solving_grid_size=self.solving_grid_size,
        base_cp=self.base_pattern,
    )

  @functools.cached_property
  def merge_converter(self) -> dict[str, str] | None:
    """Maps merged stick names of an edited `tree` back to `base_tree`."""
    if self.base_tree is None or self.tree is self.base_tree:
      return None
    try:
      conv = stick_figure_lib.compute_merge_converter_for_modified_tree(
          self.base_tree, self.tree
      )
    except (KeyError, ValueError):
      return None
    return conv if any(k != v for k, v in conv.items()) else None

  @functools.cached_property
  def shaping_heuristic_data(self) -> heuristics_utils.ShapingData:
    """Flap orientation nodes, unmerged elements, and hinge mappings."""
    flaps, rivers = self.merged_shaping_elements
    return heuristics_utils.get_heuristics_shaping_data(
        stick_figure=self.tree,
        shaping_flaps=flaps,
        shaping_rivers=rivers,
        hinge_info_list=self.updated_hinge_info(),
        central_crease_removed=self.central_crease_removed,
        shape_with_unmerged_tree=True,
        merge_converter=self.merge_converter,
        use_tree_root=self.config.use_tree_root,
    )

  @functools.cached_property
  def unmerged_shaping_elements(
      self,
  ) -> tuple[list[ShapingFlap], list[ShapingRiver]]:
    """Shaping flaps and rivers for every stick of the unmerged tree."""
    data = self.shaping_heuristic_data
    return data['shaping_flaps'], data['shaping_rivers']

  @functools.cached_property
  def shaping_elements(self) -> tuple[list[ShapingFlap], list[ShapingRiver]]:
    """Unmerged shaping elements with doubled grids on narrowed subtrees."""
    flaps, rivers = self.unmerged_shaping_elements
    if not self.narrowed_elements:
      return flaps, rivers
    to_update: set[str] = set()
    flap_info = self.shaping_heuristic_data['flap_info']
    for name in self.narrowed_elements:
      stack = [name]
      while stack:
        curr = stack.pop()
        if curr in to_update:
          continue
        to_update.add(curr)
        if curr in flap_info:
          stack.extend(flap_info[curr].children)
    new_flaps = [
        dataclasses.replace(f, grid_size=f.grid_size * 2)
        if f.name in to_update
        else f
        for f in flaps
    ]
    new_rivers = [
        dataclasses.replace(r, grid_size=r.grid_size * 2, width=r.width * 2)
        if r.name in to_update
        else r
        for r in rivers
    ]
    return new_flaps, new_rivers

  @functools.cached_property
  def flap_hierarchy(self) -> shaping_utils.FlapHierarchy:
    """Spatial containment hierarchy of the current unmerged flaps."""
    flaps, _ = self.shaping_elements
    return shaping_utils.FlapHierarchy([f.to_legacy_rectangle() for f in flaps])

  @functools.cached_property
  def clip_pattern_adapter(
      self,
  ) -> clip_pattern.ClipPatternShaper:
    """Layer-aware clip-pattern shaper built from the base pattern."""
    flaps, rivers = self.shaping_elements
    root_flaps = {r.name for r in self.flap_hierarchy.parent_map.values()}
    return clip_pattern.ClipPatternShaper.from_packing_layout(
        base_cp=self.base_pattern,
        flap_rectangles=[
            f.to_legacy_rectangle() for f in flaps if f.name in root_flaps
        ],
        river_rectangles=[r.covering_legacy_rectangles for r in rivers],
    )

  @functools.cached_property
  def base_graph_data(self) -> geometric_simple_fold.GraphData:
    """2-D flat-folded face graph of the unshaped base crease pattern."""
    return geometric_simple_fold.prepare_graph_data(self.base_pattern)

  @functools.cached_property
  def base_model(self) -> folder_3d.FoldedModel:
    """3-D folded geometry of the unshaped base crease pattern."""
    return folder_3d.fold_3d(self.base_pattern)

  @functools.cached_property
  def root_frame_spec(self) -> dict[str, Any] | None:
    """Cached anchor points and frame transform for `measured_directions`."""
    try:
      base_model = self.base_model
      layout = self.paper_layout
      root = heuristics.get_first_stick_for_shaping(
          self.tree, use_tree_root=self.config.use_tree_root
      )
      flaps, rivers = self.unmerged_shaping_elements
      elem = shaping_elements.get_element_by_name(root.name, flaps, rivers)
      grid = self.solving_grid_size
      samples = np.arange(0.25, grid, 0.5) / grid
      points: list[tuple[float, float]] = []
      elem_scale = 1.0
      for scale in (1.0, float(grid)):
        found: list[tuple[float, float]] = []
        for x in samples:
          for y in samples:
            pt = point_utils.Point(
                x=float(x * scale),
                y=float(y * scale),
                require_on_paper=False,
            )
            if elem.is_internal_point(pt, strict=True):
              found.append((float(x), float(y)))
        if len(found) > len(points):
          points, elem_scale = found, scale
      if not points:
        return None
      outside_flaps = [
          p
          for p in points
          if not any(
              x0 < p[0] < x1 and y0 < p[1] < y1
              for x0, x1, y0, y1 in layout.flap_rectangles
          )
      ]
      root_points = np.asarray(outside_flaps or points, dtype=float)
      up_reference = None
      ref = geometric_fold_3d_heuristics.root_reference_point(elem)
      if ref is not None:
        face = base_model.locate_faces(
            np.asarray(ref, dtype=float)[None] / elem_scale
        )[0]
        if face >= 0:
          up_reference = base_model.face_matrices[face][:3, 2]
      alignment = angle_utils.get_rotation_matrix_to_align(
          np.array(root.get_coords(None), dtype=float),
          np.array([0.0, 1.0, 0.0]),
      )
      root_axis_points = None
      try:
        axis_elem = elem
        if isinstance(elem, ShapingRiver):
          banks = self.shaping_heuristic_data['outer_bank_indices']
          axis_elem = elem.update_outer_bank_index(banks[elem.name])
        start, end = shaping_elements.get_unfolded_root_vector(
            axis_elem, self.tree
        )
        root_axis_points = (
            np.array([start.x, start.y], dtype=float) / elem_scale,
            np.array([end.x, end.y], dtype=float) / elem_scale,
        )
      except (AssertionError, ValueError, KeyError, IndexError):
        root_axis_points = None
      return {
          'root_name': root.name,
          'root_points': root_points,
          'up_reference': up_reference,
          'root_alignment': alignment,
          'root_axis_points': root_axis_points,
      }
    except (
        RuntimeError,
        ValueError,
        AssertionError,
        KeyError,
        IndexError,
        np.linalg.LinAlgError,
    ):
      return None

  @property
  def active_flat_pattern(self) -> crease_pattern_lib.CreasePattern:
    """Active 2-D flat-foldable backbone pattern."""
    return (
        self.flat_backbone if self.flat_backbone is not None else self.pattern
    )

  @functools.cached_property
  def graph_data(self) -> geometric_simple_fold.GraphData:
    """Cached 2-D flat-folded face graph of `active_flat_pattern`."""
    active = self.active_flat_pattern
    if active is self.base_pattern:
      return self.base_graph_data
    return geometric_simple_fold.prepare_graph_data(active)

  def element(self, name: str) -> ShapingElement:
    """Looks up the shaping flap or river named `name`."""
    flaps, rivers = self.shaping_elements
    return shaping_elements.get_element_by_name(name, flaps, rivers)

  def element_names(self) -> list[str]:
    """Returns all shaping flap names followed by all shaping river names."""
    flaps, rivers = self.shaping_elements
    return [f.name for f in flaps] + [r.name for r in rivers]

  def element_polygon(self, name: str) -> list[shaping_utils.Rectangle]:
    """Returns legacy bounding rectangles covering `name`."""
    elem = self.element(name)
    if isinstance(elem, ShapingFlap):
      return [elem.to_legacy_rectangle()]
    return list(elem.covering_legacy_rectangles)

  def chain_polygon(self, name: str) -> list[shaping_utils.Rectangle] | None:
    """Returns the merged chain flap rectangle containing `name`, if any."""
    _, merged_names = self.tree.merge_linear_chains()
    flaps, _ = self.merged_shaping_elements
    for merged_name, sub_names in merged_names.items():
      if name == merged_name or name in sub_names:
        for flap in flaps:
          if flap.name == merged_name:
            return [flap.to_legacy_rectangle()]
    return None

  def reference_point(
      self, name: str, graph_data: geometric_simple_fold.GraphData | None = None
  ) -> np.ndarray:
    """Returns the 2-D centroid of the nearest mapped ancestor of `name`."""
    active_gd = self.graph_data if graph_data is None else graph_data
    topo = paper_layout_lib.topology(self.tree)
    curr = name
    while True:
      parent = topo.parent.get(curr)
      if parent is None:
        break
      pts = folded_region_fold.element_folded_points(
          active_gd, self.element_polygon(parent)
      )
      if pts.size:
        return pts.mean(axis=0)
      curr = parent
    folded_v = active_gd.get('folded_vertices', [])
    valid = [v[:2] for v in folded_v if v is not None]
    if valid:
      return np.asarray(valid, dtype=float).mean(axis=0)
    return np.array([0.5, 0.5], dtype=float)

  # -- Functional updates ------------------------------------------------------

  def replace(self, **changes: Any) -> 'ShapingState':
    """Returns a modified copy, preserving still-valid cached properties."""
    new_state = dataclasses.replace(self, **changes)
    keep: set[str] = set()
    if (
        new_state.base_pattern is self.base_pattern
        and new_state.packing is self.packing
        and new_state.solving is self.solving
    ):
      keep.update(_PACKING_CACHED_PROPERTIES)
      if new_state.tree is self.tree and new_state.config == self.config:
        keep.update(_TREE_CACHED_PROPERTIES)
        if new_state.narrowed_elements == self.narrowed_elements:
          keep.update(_NARROWING_CACHED_PROPERTIES)
    if new_state.active_flat_pattern is self.active_flat_pattern:
      keep.add('graph_data')
    for prop in keep:
      if prop in self.__dict__:
        new_state.__dict__[prop] = self.__dict__[prop]
    return new_state

  def with_pattern(
      self,
      pattern: crease_pattern_lib.CreasePattern,
      *,
      allow_3d_vertices: bool = False,
  ) -> 'ShapingState':
    """Returns a copy with `pattern`, checking foldability and 3-D overlays."""
    prepared = _with_tolerance(pattern, self.config.cp_tolerance)
    if self.config.check_flat_foldability:
      ok_ff, err_ff = overlay_3d.preserves_foldability(
          self.pattern, prepared, allow_new_3d_vertices=allow_3d_vertices
      )
      if not ok_ff:
        raise ValueError(err_ff)
    if not overlay_3d.has_3d_overlay(prepared):
      next_backbone = None
    elif prepared.local_flat_foldability():
      next_backbone = overlay_3d.as_flat_backbone(prepared)
    elif self.flat_backbone is not None:
      next_backbone = self.flat_backbone
    elif self.pattern.local_flat_foldability():
      next_backbone = overlay_3d.as_flat_backbone(self.pattern)
    else:
      next_backbone = None
    return self.replace(pattern=prepared, flat_backbone=next_backbone)

  def with_narrowed(self, *names: str) -> 'ShapingState':
    """Returns a copy recording `names` as grid-doubled narrowed elements."""
    merged = tuple(dict.fromkeys((*self.narrowed_elements, *names)))
    if merged == self.narrowed_elements:
      return self
    return self.replace(narrowed_elements=merged)

  def reset(self) -> 'ShapingState':
    """Returns a clean state reset to `base_pattern` and `base_tree`."""
    return self.replace(
        tree=self.base_tree if self.base_tree is not None else self.tree,
        pattern=self.base_pattern,
        narrowed_elements=(),
        history=(),
        flat_backbone=None,
    )


# ------------------------------------------------------------------------------
# Crease, serialization, and fold-verification helpers.
# ------------------------------------------------------------------------------


def crease_key(
    crease: crease_utils.Crease, ndigits: int = 6
) -> tuple[float, float, float, float]:
  """Returns canonical rounded endpoints `(x1, y1, x2, y2)` of `crease`."""
  p1 = (round(crease.p1_init.x, ndigits), round(crease.p1_init.y, ndigits))
  p2 = (round(crease.p2_init.x, ndigits), round(crease.p2_init.y, ndigits))
  return (*p1, *p2) if p1 <= p2 else (*p2, *p1)


def crease_multiset(
    pattern: crease_pattern_lib.CreasePattern,
) -> dict[tuple[Any, ...], int]:
  """Returns a multiset of `(endpoints, fold, fold_percent)` for M/V creases."""
  counts: dict[tuple[Any, ...], int] = {}
  for c in pattern.creases:
    if not c.mountain_or_valley():
      continue
    key = (crease_key(c), c.fold, round(c.fold_percent, 5))
    counts[key] = counts.get(key, 0) + 1
  return counts


def new_crease_count(
    before: crease_pattern_lib.CreasePattern,
    after: crease_pattern_lib.CreasePattern,
) -> int:
  """Counts creases in `after` whose geometry is absent from `before`."""
  old = {crease_key(c) for c in before.creases}
  return sum(1 for c in after.creases if crease_key(c) not in old)


def with_fold_percent(
    pattern: crease_pattern_lib.CreasePattern,
    keys: set[tuple[float, float, float, float]],
    fold_percent: float,
) -> crease_pattern_lib.CreasePattern:
  """Returns `pattern` with M/V creases in `keys` set to `fold_percent`."""
  updated: list[crease_utils.Crease] = []
  for c in pattern.creases:
    if c.mountain_or_valley() and crease_key(c) in keys:
      updated.append(
          crease_utils.Crease(
              c.p1_init,
              c.p2_init,
              fold=c.fold,
              fold_percent=fold_percent,
              elevation=c.elevation,
          )
      )
    else:
      updated.append(c)
  return crease_pattern_lib.CreasePattern(
      creases=updated,
      grid_size=-1,
      light_mode=True,
      tolerance=pattern.tolerance,
  )


def preserve_base_borders(
    before: crease_pattern_lib.CreasePattern,
    after: crease_pattern_lib.CreasePattern,
) -> crease_pattern_lib.CreasePattern:
  """Restores missing unit-square border segments split at border vertices."""
  non_borders = [c for c in after.creases if c.fold != crease_utils.Fold.BORDER]
  old_borders = [
      c for c in before.creases if c.fold == crease_utils.Fold.BORDER
  ]
  if not old_borders:
    return after
  tol = after.tolerance
  pts_by_side: list[list[float]] = [
      [0.0, 1.0],
      [0.0, 1.0],
      [0.0, 1.0],
      [0.0, 1.0],
  ]
  for c in non_borders:
    for p in (c.p1_init, c.p2_init):
      if math.isclose(p.y, 0.0, abs_tol=tol):
        pts_by_side[0].append(min(1.0, max(0.0, p.x)))
      if math.isclose(p.y, 1.0, abs_tol=tol):
        pts_by_side[1].append(min(1.0, max(0.0, p.x)))
      if math.isclose(p.x, 0.0, abs_tol=tol):
        pts_by_side[2].append(min(1.0, max(0.0, p.y)))
      if math.isclose(p.x, 1.0, abs_tol=tol):
        pts_by_side[3].append(min(1.0, max(0.0, p.y)))

  def _dedup(coords: list[float]) -> list[float]:
    ordered = sorted(coords)
    out = [ordered[0]]
    for val in ordered[1:]:
      if not math.isclose(val, out[-1], abs_tol=tol):
        out.append(val)
    return out

  border_creases: list[crease_utils.Crease] = []
  for side_idx, raw in enumerate(pts_by_side):
    vals = _dedup(raw)
    for a, b in zip(vals, vals[1:]):
      if side_idx == 0:
        p1 = point_utils.Point(x=a, y=0.0)
        p2 = point_utils.Point(x=b, y=0.0)
      elif side_idx == 1:
        p1 = point_utils.Point(x=a, y=1.0)
        p2 = point_utils.Point(x=b, y=1.0)
      elif side_idx == 2:
        p1 = point_utils.Point(x=0.0, y=a)
        p2 = point_utils.Point(x=0.0, y=b)
      else:
        p1 = point_utils.Point(x=1.0, y=a)
        p2 = point_utils.Point(x=1.0, y=b)
      border_creases.append(
          crease_utils.Crease(p1, p2, fold=crease_utils.Fold.BORDER)
      )
  rebuilt = crease_pattern_lib.CreasePattern(
      creases=non_borders + border_creases,
      grid_size=-1,
      light_mode=after.light_mode,
      tolerance=after.tolerance,
  )
  return rebuilt


def serialize_pattern_svg(pattern: crease_pattern_lib.CreasePattern) -> str:
  """Serializes a crease pattern to a high-precision SVG string."""
  return svg_parser.to_svg(pattern, grid=-1, ndigits=8)


def deserialize_pattern_svg(
    svg_text: str,
    *,
    grid_size: int | None = None,
    tolerance: float = crease_utils.DEFAULT_TOLERANCE,
) -> crease_pattern_lib.CreasePattern:
  """Parses a `CreasePattern` from SVG without snapping off-grid tips."""
  effective_grid = grid_size if grid_size is not None and grid_size > 0 else -1
  parsed = svg_parser.parse_svg(
      svg_text, grid_size=effective_grid, light_mode=True
  )
  parsed.tolerance = tolerance
  return parsed


def check_flat_foldability(pattern: crease_pattern_lib.CreasePattern) -> None:
  """Raises `ValueError` unless every internal vertex is flat-foldable."""
  percentage, _, _ = pattern.vertex_flat_foldability_percentage()
  if not math.isclose(percentage, 1.0):
    raise ValueError(
        'Crease pattern is not flat-foldable after shaping:'
        f' {percentage:.1%} of vertices satisfy Kawasaki/Maekawa.'
    )


def all_internal_vertices_3d_foldable(
    pattern: crease_pattern_lib.CreasePattern,
) -> bool:
  """Returns True when every internal vertex satisfies 3-D Belcastro-Hull."""
  return all(not v.internal or v.is_foldable for v in pattern.vertices)


def verify_3d_fold(
    pattern: crease_pattern_lib.CreasePattern,
    *,
    max_vertex_error: float = MAX_RENDERABLE_VERTEX_ERROR,
) -> tuple[bool, str, folder_3d.FoldedModel | None]:
  """Folds `pattern` with the deterministic 3-D folder and checks strain."""
  for c in pattern.creases:
    if c.mountain_or_valley() and (
        math.hypot(c.p1_init.x - c.p2_init.x, c.p1_init.y - c.p2_init.y)
        < MIN_SEGMENT_LENGTH
    ):
      return False, f'degenerate crease (< {MIN_SEGMENT_LENGTH})', None
  try:
    model = folder_3d.fold_3d(pattern)
  except (RuntimeError, ValueError, AssertionError, KeyError) as error:
    return False, f'folder: {str(error)[:120]}', None
  if model.vertex_error > max_vertex_error:
    return (
        False,
        f'vertex error {model.vertex_error:.2f}% > {max_vertex_error:.2f}%',
        None,
    )
  return True, f'folds (strain {model.vertex_error:.3f}%)', model


def accept_pattern(
    state: ShapingState,
    shaped: crease_pattern_lib.CreasePattern,
    *,
    allow_3d_vertices: bool = False,
    verify_fold: bool = True,
    max_vertex_error: float = MAX_RENDERABLE_VERTEX_ERROR,
) -> tuple[ShapingState | None, str, folder_3d.FoldedModel | None]:
  """Checks foldability and 3-D folding, returning `(state, reason, model)`."""
  try:
    candidate = state.with_pattern(shaped, allow_3d_vertices=allow_3d_vertices)
  except ValueError as error:
    return None, str(error), None
  model = None
  if verify_fold:
    ok, reason, model = verify_3d_fold(
        candidate.pattern, max_vertex_error=max_vertex_error
    )
    if not ok:
      return None, reason, None
  return candidate, 'ok', model


def run_with_3d_overlay(
    state: ShapingState,
    fn: Callable[[ShapingState], ToolOutcome],
    *,
    verify_fold: bool = True,
) -> ToolOutcome:
  """Runs a 2-D backbone operation on `flat_backbone` and replays 3-D relief."""
  if state.flat_backbone is None:
    out = fn(state)
    st = out.state
    if st.flat_backbone is None and overlay_3d.has_3d_overlay(st.pattern):
      if st.pattern.local_flat_foldability():
        st = st.replace(flat_backbone=overlay_3d.as_flat_backbone(st.pattern))
      elif state.pattern.local_flat_foldability():
        st = st.replace(
            flat_backbone=overlay_3d.as_flat_backbone(state.pattern)
        )
      return dataclasses.replace(out, state=st)
    return out

  before = crease_multiset(state.pattern)
  flat_in = state.replace(pattern=state.flat_backbone, flat_backbone=None)
  replay_discarded = False
  flat_ok = False
  try:
    flat_out = fn(flat_in)
    flat_ok = True
    combined = overlay_3d.replay_3d_overlay(
        state.flat_backbone, state.pattern, flat_out.state.pattern
    )
    if crease_multiset(combined) == before and crease_multiset(
        flat_out.state.pattern
    ) != crease_multiset(flat_in.pattern):
      replay_discarded = True
      raise ValueError('3-D overlay replay discarded fold-angle-only change.')
    allow_3d = not flat_out.state.pattern.local_flat_foldability()
    accepted, reason, _ = accept_pattern(
        state,
        combined,
        allow_3d_vertices=allow_3d,
        verify_fold=verify_fold,
    )
    if accepted is None:
      raise ValueError(reason)
    if not overlay_3d.has_3d_overlay(accepted.pattern):
      next_bb = None
    elif flat_out.state.flat_backbone is not None:
      next_bb = flat_out.state.flat_backbone
    elif flat_out.state.pattern.local_flat_foldability():
      next_bb = overlay_3d.as_flat_backbone(flat_out.state.pattern)
    else:
      next_bb = state.flat_backbone
    final_state = flat_out.state.replace(
        pattern=accepted.pattern, flat_backbone=next_bb
    )
    return dataclasses.replace(flat_out, state=final_state)
  except (
      ValueError,
      AssertionError,
      KeyError,
      IndexError,
      ZeroDivisionError,
  ) as primary_error:
    if not flat_ok and len(state.flat_backbone.creases) == len(
        state.pattern.creases
    ):
      raise
    try:
      direct_out = fn(state.replace(flat_backbone=None))
    except (
        ValueError,
        AssertionError,
        KeyError,
        IndexError,
        ZeroDivisionError,
    ) as direct_error:
      raise primary_error from direct_error
    if replay_discarded and crease_multiset(direct_out.state.pattern) == before:
      raise ValueError(
          'Operation left the crease pattern unchanged because its target'
          ' creases are already controlled by the active 3-D overlay.'
      ) from primary_error
    accepted, reason, _ = accept_pattern(
        state,
        direct_out.state.pattern,
        allow_3d_vertices=True,
        verify_fold=verify_fold or overlay_3d.has_3d_overlay(state.pattern),
    )
    if accepted is None:
      raise ValueError(reason) from primary_error
    next_bb = direct_out.state.flat_backbone
    if next_bb is None and overlay_3d.has_3d_overlay(accepted.pattern):
      next_bb = state.flat_backbone
    return dataclasses.replace(
        direct_out,
        state=direct_out.state.replace(
            pattern=accepted.pattern, flat_backbone=next_bb
        ),
    )


def preserve_3d_overlay(
    fn: Callable[..., ShapingState],
) -> Callable[..., ShapingState]:
  """Decorator adapting `run_with_3d_overlay` to state-returning tools."""

  @functools.wraps(fn)
  def wrapped(state: ShapingState, *args: Any, **kwargs: Any) -> ShapingState:
    verify = bool(
        kwargs.get('verify_fold', state.config.check_flat_foldability)
    )
    return run_with_3d_overlay(
        state,
        lambda st: ToolOutcome(state=fn(st, *args, **kwargs)),
        verify_fold=verify,
    ).state

  return wrapped


# ------------------------------------------------------------------------------
# Bilateral symmetry, direction measurement, and state inspection.
# ------------------------------------------------------------------------------


def resolve_symmetric_element(
    state: ShapingState, element_name: str
) -> str | None:
  """Resolves the bilateral mirror partner of `element_name`, or `None`."""
  known = set(state.element_names()) | {s.name for s in state.tree.children}
  if element_name in state.tree:
    explicit = state.tree[element_name].symmetric_stick_name
    if explicit and explicit != element_name and explicit in known:
      return explicit

  suffix = packing_types.COPY_SUFFIX
  if element_name.endswith(suffix):
    base = element_name[: -len(suffix)]
    if base in known:
      return base
  elif f'{element_name}{suffix}' in known:
    return f'{element_name}{suffix}'

  inferred = fidelity.infer_symmetric_stick(element_name, known)
  if inferred and inferred != element_name:
    return inferred

  for src, dst in _LEFT_RIGHT_WORDS:
    if element_name.endswith(src):
      cand = element_name[: -len(src)] + dst
      if cand in known:
        return cand
    if element_name.startswith(src):
      cand = dst + element_name[len(src) :]
      if cand in known:
        return cand

  match = _LR_DIGIT_RE.match(element_name)
  if match is not None:
    prefix, side, tail = match.groups()
    flip = {'L': 'R', 'R': 'L', 'l': 'r', 'r': 'l'}[side]
    cand = f'{prefix}{flip}{tail}'
    if cand in known:
      return cand

  # Chain-slot lookup when merged chains carry symmetric partner links.
  merged_tree, merged_map = state.tree.merge_linear_chains()
  for m_name, chain in merged_map.items():
    if element_name in chain and m_name in merged_tree:
      partner_m = merged_tree[m_name].symmetric_stick_name
      if partner_m and partner_m != m_name and partner_m in merged_map:
        partner_chain = merged_map[partner_m]
        idx = chain.index(element_name)
        if idx < len(partner_chain):
          cand = partner_chain[idx]
          if cand != element_name and cand in known:
            return cand
  return None


def is_left_element(state: ShapingState, element_name: str) -> bool:
  """Returns True when `element_name`'s paper footprint lies left of x = 0.5."""
  elem = state.element(element_name)
  if isinstance(elem, ShapingFlap):
    return (elem.x_start + elem.x_len * 0.5) < 0.5 - 1e-4
  rects = list(elem.covering_legacy_rectangles)
  if not rects:
    return False
  mid_x = sum(r.x_start + r.x_len * 0.5 for r in rects) / len(rects)
  return mid_x < 0.5 - 1e-4


def measured_directions(
    state: ShapingState,
    pattern: crease_pattern_lib.CreasePattern | None = None,
    *,
    model: folder_3d.FoldedModel | None = None,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
  """Measures realized figure-frame directions in `rotation_3d`'s root frame.

  Falls back to `fidelity.stick_directions` when the root frame cannot be
  constructed.

  Args:
    state: Current shaping state.
    pattern: Optional crease pattern override (defaults to `state.pattern`).
    model: Optional pre-folded `FoldedModel` for `pattern`.

  Returns:
    A tuple `(report, directions)` where `directions[name]` maps to
    `{'target': (az, el), 'realized': (az, el) | None, 'error': float | None}`.
  """
  target_cp = state.pattern if pattern is None else pattern
  folded = folder_3d.fold_3d(target_cp) if model is None else model
  layout = state.paper_layout
  report_obj, fitted = fidelity.stick_directions(
      state.tree, state.packing, folded, layout=layout
  )
  report = report_obj.to_dict()
  spec = state.root_frame_spec
  if spec is None:
    return report, fitted
  try:
    measured = pose_alignment.root_frame_directions(
        state.tree,
        layout,
        folded,
        state.base_model,
        **spec,
    )
  except (ValueError, KeyError, IndexError, np.linalg.LinAlgError):
    measured = None
  if measured is None:
    return report, fitted
  directions = measured['directions']
  updated = dict(report)
  updated['direction_error_deg_fit'] = report.get('direction_error_deg', {})
  updated['direction_error_deg'] = {
      name: info['error']
      for name, info in directions.items()
      if info['error'] is not None
  }
  updated['direction_frame'] = measured['frame']
  if 'side_crossings' in measured:
    updated['side_crossings_fit'] = report.get('side_crossings', [])
    updated['side_crossings'] = measured['side_crossings']
    updated['pose_gate'] = fidelity.pose_gate(
        updated['components'], updated
    ).to_dict()
  return updated, directions


def flap_orientation_info(state: ShapingState) -> dict[str, dict[str, Any]]:
  """Returns hierarchy, direction, and parent metadata per unmerged element."""
  info = state.shaping_heuristic_data['flap_info']
  out: dict[str, dict[str, Any]] = {}
  for name, node in info.items():
    out[name] = {
        'parent': node.parent,
        'children': list(node.children),
        'direction': str(node.absolute_direction),
        'narrowed': name in state.narrowed_elements,
    }
  return out


def element_spans_axis(element: ShapingElement) -> bool:
  """Returns True if `element` crosses or touches the `x = 0.5` axis."""
  if isinstance(element, ShapingFlap):
    return bool(element.contains_central_crease)
  xs: list[float] = []
  for seg in (*element.bank_1, *element.bank_2):
    xs.append(seg.p1_init.x)
    xs.append(seg.p2_init.x)
  if xs and min(xs) < 0.5 - 1e-4 and max(xs) > 0.5 + 1e-4:
    return True
  return bool(element.borders_central_crease)


def element_table(state: ShapingState) -> list[dict[str, Any]]:
  """Summarizes each shaping element's geometry, symmetry, and target angles."""
  flaps, rivers = state.shaping_elements
  topo = paper_layout_lib.topology(state.tree)
  rows: list[dict[str, Any]] = []
  for flap in flaps:
    stick = state.tree[flap.name] if flap.name in state.tree else None
    rows.append({
        'name': flap.name,
        'kind': 'flap',
        'parent': topo.parent.get(flap.name),
        'symmetric': resolve_symmetric_element(state, flap.name),
        'length': float(flap.grid_unit_radius),
        'width': float(stick.width) if stick is not None else 1.0,
        'az': float(stick.azimuth_angle) if stick is not None else 0.0,
        'el': float(stick.elevation_angle) if stick is not None else 0.0,
        'spans_axis': element_spans_axis(flap),
        'narrowed': flap.name in state.narrowed_elements,
        'grid_size': int(flap.grid_size),
    })
  for river in rivers:
    stick = state.tree[river.name] if river.name in state.tree else None
    rows.append({
        'name': river.name,
        'kind': 'river',
        'parent': topo.parent.get(river.name),
        'symmetric': resolve_symmetric_element(state, river.name),
        'length': float(river.width),
        'width': float(river.width),
        'az': float(stick.azimuth_angle) if stick is not None else 0.0,
        'el': float(stick.elevation_angle) if stick is not None else 0.0,
        'spans_axis': element_spans_axis(river),
        'narrowed': river.name in state.narrowed_elements,
        'grid_size': int(river.grid_size),
    })
  return rows


def describe(state: ShapingState) -> dict[str, Any]:
  """Returns a JSON-serializable summary of `state` and its elements."""
  pct, _, _ = state.pattern.vertex_flat_foldability_percentage()
  return {
      'root': state.tree.root,
      'grid_size': state.solving_grid_size,
      'central_crease_removed': state.central_crease_removed,
      'num_creases': len(state.pattern.creases),
      'base_creases': len(state.base_pattern.creases),
      'flat_foldability': round(pct, 4),
      'has_3d_overlay': overlay_3d.has_3d_overlay(state.pattern),
      'narrowed_elements': list(state.narrowed_elements),
      'history_length': len(state.history),
      'elements': element_table(state),
  }


def history_json(state: ShapingState, *, indent: int = 2) -> str:
  """Formats `state.history` as a JSON string."""
  return json.dumps(list(state.history), indent=indent)
