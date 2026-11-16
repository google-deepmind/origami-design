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

"""Stateful actuator toolbox for iterative 3D origami shaping.

Provides composable shaping actuators over `bp_pipeline` crease patterns so
designers and agents can shape a flat-foldable base crease pattern step by step:

  state = ShapingState.create(tree, packing_output, solver_output)
  state = simple_fold(state, element_name='tail_1', angle=45, depth=0.0)
  state = crimp_fold(state, element_name='leg_L1', angle=60, depth=0.5)
  ...

Every tool returns a new state whose `pattern` is the shaped crease pattern
(the input state is left untouched) and raises ValueError with an explanation
when the operation is impossible or breaks flat-foldability. `apply_tool`
dispatches by tool name from JSON-style calls and never raises; it records the
outcome in `state.history` instead.

The tool semantics (parameter ranges, defaults, preconditions) are those of
the RL declarations; see the shaping skill reference for the full catalogue.
"""

from collections.abc import Callable, Iterator, Mapping, Sequence
import contextlib
import contextvars
import copy
import dataclasses
import functools
import json
import logging as std_logging
import math
import os
import re
import time
import traceback
from typing import Any, TypeVar, cast

from absl import logging
import numpy as np

from ..crease_pattern import crease as crease_utils
from ..crease_pattern import crease_pattern as crease_pattern_lib
from ..crease_pattern import point as point_utils
from .. import bus
from .. import utils_tree
from . import bus_adapter
from ..packing import compute_ridges
from ..packing import config as packing_config_lib
from ..packing import packing_output as packing_output_lib
from ..solving import hinges
from . import level_shifter_utils as utils_ls
from ..crease_pattern.parser import svg as svg_parser
from . import folder_3d as folder
from ..stick_figures import angle_utils
from ..stick_figures import stick as stick_lib
from ..stick_figures import stick_figure as stick_figure_lib
from ..stick_figures import traversal
from . import fidelity
from . import toolbox_extras as shaping_extras
from . import toolbox_fan_squash as shaping_fan_squash
from . import toolbox_fold3d as shaping_fold3d
from .clip_pattern import clip_pattern_shaping_adapter
from . import combined_heuristics
from . import crimp_fold_shaping
from . import geometric_fold_3d_heuristics
from . import geometric_simple_fold as geometric_simple_fold_lib
from . import heuristics
from . import heuristics_3d
from . import heuristics_utils
from . import reverse_fold_shaping
from . import shaping_elements
from . import shaping_packing_utils
from . import shaping_utils
from . import simple_fold_shaping
from . import squash_sink_shaping
from . import types as shaping_types
from . import zigzag_shaping
from . import zigzag_shaping_utils

ShapingFlap = shaping_elements.ShapingFlap
ShapingRiver = shaping_elements.ShapingRiver
ShapingElement = ShapingFlap | ShapingRiver

_FOLD_TYPES = {
    'mountain': crease_utils.Fold.MOUNTAIN,
    'valley': crease_utils.Fold.VALLEY,
}
_CLIP_PATTERN_MARGIN = 0.03
# Tools in the catalogue; keys accept an optional `shaping:`/`tree:` prefix.
TOOL_NAMES = (
    'simple_fold',
    'simple_fold_heuristic',
    'crimp_fold',
    'reverse_fold',
    'clip_pattern_fold',
    'rotation_3d',
    'shape_angle_width',
    'thin_limbs_3d',
    'fan_out_fold',
    'fan_out_fold_heuristic',
    'geometric_simple_fold',
    'squash_fold',
    'tree_edit',
    'level_shifter',
    'reverse_fold_3d',
    'crimp_fold_3d',
    'open_flap',
    'inscribed_flap',
    'pleat_to_hide',
    'simple_fold_narrow',
    'narrow_central_flap',
    'curve_flap',
    'apply_symmetric',
    'fold_spine_3d',
    'spread_fork',
    'radial_fan',
    'open_river_3d',
    'bend_to_direction',
)


@dataclasses.dataclass(frozen=True, kw_only=True)
class ToolConfig:
  """Options shared by the shaping tools (RL `ShapingConfig` defaults).

  Attributes:
    check_flat_foldability: Reject a tool result that is not locally
      flat-foldable at every vertex.
    use_heuristics_for_fold_order: `simple_fold` picks mountain/valley order
      from the tree.
    use_heuristics_for_depth: `simple_fold` adjusts the fold depth from the
      tree.
    use_heuristics_for_phase: `simple_fold` picks the zigzag phase from the
      tree.
    convert_small_angles: `simple_fold` converts small angles into two larger
      ones (disabled automatically when narrowing).
    clip_slope_thresh: `rotation_3d` clips simple folds steeper than this.
    max_slope_thresh: `rotation_3d` drops simple folds steeper than this.
    fold_strategy_3d: `rotation_3d` folding strategy.
    check_all_crease_intersections: Re-check every crease intersection when a
      tool recompiles the pattern (slow); otherwise only new creases.
    cp_tolerance: Vertex merging / intersection tolerance for the pattern.
    use_designer_root_3d: `rotation_3d`/`shape_angle_width` measure every target
      angle from the stick figure's own root stick (the body the designer chose)
      instead of the library's `get_first_stick_for_shaping` choice. The library
      aligns its root to az=0, el=0 and never shapes it; when it picks a tail
      segment (scorpion: `metasoma_1`, az=180 el=30) or the rear body (moose:
      `pelvis`, az=180) the whole pose comes out pitched or turned around and
      left/right swapped (angle investigation, direction score 0.38 -> 0.93
      scorpion, 0.61 -> 1.00 moose).
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
  use_designer_root_3d: bool = True

  @property
  def apply_simple_fold_heuristics(self) -> bool:
    """Whether any simple-fold heuristic (order, depth, small angles) is on."""
    return (
        self.use_heuristics_for_fold_order
        or self.use_heuristics_for_depth
        or self.convert_small_angles
        or self.use_heuristics_for_phase
    )


@dataclasses.dataclass(frozen=True, kw_only=True)
class ToolResult:
  """Outcome of `apply_tool`.

  Attributes:
    tool: Tool name.
    arguments: Arguments the tool was called with.
    ok: Whether the tool succeeded (the state was updated).
    message: Success message or the error explanation.
    num_creases: Number of creases in the pattern after the call.
  """

  tool: str
  arguments: dict[str, Any]
  ok: bool
  message: str
  num_creases: int

  def asdict(self) -> dict[str, Any]:
    return dataclasses.asdict(self)


def check_flat_foldability(pattern: crease_pattern_lib.CreasePattern) -> None:
  """Raises ValueError unless every vertex is locally flat-foldable."""
  percentage, _, _ = pattern.vertex_flat_foldability_percentage()
  if not math.isclose(percentage, 1.0):
    raise ValueError(
        'Crease pattern is not flat-foldable after shaping:'
        f' {percentage:.1%} of vertices satisfy Kawasaki/Maekawa.'
    )


def _pre_process_angle(angle_deg: float) -> float:
  """Wraps an angle to (-180, 180], converts to radians, clips tiny angles."""
  wrapped = angle_utils.wrap_angle_centered(angle_deg, 360.0)
  radians = math.radians(wrapped)
  if math.isclose(radians, 0, abs_tol=crease_utils.DEFAULT_TOLERANCE):
    return 0.0
  return zigzag_shaping_utils.clip_angle_by_min_angle(radians)


@dataclasses.dataclass(frozen=True)
class ShapingState:
  """Everything the shaping tools need, plus the current crease pattern.

  Instances are immutable; tools return updated copies via
  `dataclasses.replace`. Derived data (shaping elements, heuristic data, flap
  hierarchy, clip-pattern adapter, folded-graph data) is computed lazily from
  the base pattern and cached on the instance.

  Attributes:
    tree: The unmerged stick figure (flap tip shapes assigned).
    packing: The packing output used by the solver.
    solving: The solver result (base crease pattern + state).
    pattern: The current (shaped) crease pattern.
    base_pattern: The unshaped flat-foldable base crease pattern.
    config: Tool options.
    narrowed_elements: Elements whose grid size was doubled by narrowing; their
      descendants are implied.
    history: Tool calls applied so far, as `ToolResult` dicts.
    flat_backbone: Latest 2-D flat-foldable backbone pattern when 3-D overlays
      are active, or None.
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

  @classmethod
  def create(
      cls,
      tree: stick_figure_lib.StickFigure,
      packing: packing_output_lib.PackingOutput,
      solving: hinges.SolverResult,
      *,
      pattern: crease_pattern_lib.CreasePattern | None = None,
      config: ToolConfig = ToolConfig(),
  ) -> 'ShapingState':
    """Builds a state from pipeline outputs.

    Args:
      tree: The unmerged stick figure.
      packing: The packing output (`PackingResult.packing_outputs[i]`).
      solving: The matching solver output.
      pattern: Current pattern to continue from; defaults to the base pattern.
      config: Tool options.

    Returns:
      A fresh shaping state.
    """
    tree = copy.deepcopy(tree)
    if any(
        s.is_flap and s.flap_tip_shape in (None, stick_lib.FlapTipShape.UNSET)
        for s in tree.children
    ):
      compute_ridges.assign_flap_shapes_from_packing_output(tree, packing)
    base_pattern = solving.crease_pattern
    base_pattern.tolerance = config.cp_tolerance
    flat_backbone = None
    if pattern is None:
      pattern = base_pattern
    else:
      pattern.tolerance = config.cp_tolerance
      if fidelity.has_3d_overlay(pattern):
        flat_backbone = (
            fidelity.as_flat_backbone(pattern)
            if pattern.local_flat_foldability()
            else base_pattern
        )
    return cls(
        tree=tree,
        packing=packing,
        solving=solving,
        pattern=pattern,
        base_pattern=base_pattern,
        config=config,
        flat_backbone=flat_backbone,
    )


  @classmethod
  def from_solved(
      cls,
      solved: bus.SolvedCreasePattern,
      *,
      pattern: crease_pattern_lib.CreasePattern | None = None,
      config: ToolConfig = ToolConfig(),
  ) -> 'ShapingState':
    """Constructs a `ShapingState` directly from an immutable `SolvedCreasePattern`."""
    packing_out, solver_res = bus_adapter.legacy_inputs(solved)
    tree = (
        packing_out.unmerged_stick_figure
        if packing_out.unmerged_stick_figure is not None
        else packing_out.stick_figure
    )
    if tree is None:
      raise ValueError('SolvedCreasePattern has no associated StickFigure.')
    return cls.create(
        tree,
        packing_out,
        solver_res,
        pattern=pattern,
        config=config,
    )

  def to_shaped(
      self,
      solved: bus.SolvedCreasePattern,
      *,
      strategy: str = 'toolbox',
      time_taken: float = 0.0,
  ) -> bus.ShapedCreasePattern:
    """Wraps the current shaped crease pattern into a `ShapedCreasePattern` bus record."""
    return bus.ShapedCreasePattern.from_crease_pattern(
        self.pattern,
        solved=solved,
        strategy=strategy,
        time_taken=time_taken,
    )

  # -- Derived data --------------------------------------------------------

  @property
  def packing_cp(self) -> crease_pattern_lib.CreasePattern:
    """The packing crease pattern."""
    if not isinstance(
        self.packing.packing_cp, crease_pattern_lib.CreasePattern
    ):
      raise ValueError('Packing output has no crease pattern.')
    return self.packing.packing_cp

  @property
  def packing_grid_size(self) -> int:
    """Grid size of the packing."""
    return self.packing_cp.grid_size

  @property
  def solving_grid_size(self) -> int:
    """Grid size of the solved base pattern."""
    return self.base_pattern.grid_size

  @property
  def central_crease_removed(self) -> bool:
    """Whether the solver removed the central crease of the base."""
    return self.solving.final_state.solver_config.remove_central_crease

  @property
  def flap_rectangles(self) -> list[shaping_utils.Rectangle]:
    """Flap rectangles of the packing (merged tree)."""
    if self.packing.flap_rectangles is None:
      raise ValueError('Packing output has no flap rectangles.')
    return self.packing.flap_rectangles

  @property
  def packing_config(self) -> packing_config_lib.PackingConfig:
    """Packing configuration (defaults when the packing was reloaded)."""
    if self.packing.config is None:
      return packing_config_lib.PackingConfig()
    return self.packing.config

  def updated_hinge_info(self) -> list[Any]:
    """Packing hinges updated with the solver's assignments."""
    hinge_info_list = copy.deepcopy(self.packing.hinges or [])
    return heuristics.update_hinge_info_from_solver_result(
        hinge_info_list, self.solving
    )

  @functools.cached_property
  def hinge_segments(self) -> list[crease_utils.Crease]:
    """Non-empty hinge coordinates from `updated_hinge_info` (cached)."""
    return [
        hinge.coordinate
        for hinge in self.updated_hinge_info()
        if hinge.coordinate is not None
    ]

  @functools.cached_property
  def paper_layout(self) -> fidelity.Layout:
    """Paper layout of joints and stick contours on the base packing."""
    return fidelity.paper_layout(self.tree, self.packing)

  @functools.cached_property
  def merged_shaping_elements(
      self,
  ) -> tuple[list[ShapingFlap], list[ShapingRiver]]:
    """Shaping flaps/rivers of the merged tree (packing rectangles)."""
    return shaping_elements.convert_packing_elements_to_shaping_elements(
        packing_rectangles=self.flap_rectangles,
        packing_rivers=self.packing.packing_rivers,
        packing_grid_size=self.packing_grid_size,
        solving_grid_size=self.solving_grid_size,
        base_cp=self.base_pattern,
    )

  @functools.cached_property
  def shaping_heuristic_data(self) -> heuristics_utils.ShapingData:
    """Flap info, unmerged elements and hinge data used by heuristics."""
    flaps, rivers = self.merged_shaping_elements
    return heuristics_utils.get_heuristics_shaping_data(
        stick_figure=self.tree,
        shaping_flaps=flaps,
        shaping_rivers=rivers,
        hinge_info_list=self.updated_hinge_info(),
        central_crease_removed=self.central_crease_removed,
        shape_with_unmerged_tree=True,
    )

  @functools.cached_property
  def unmerged_shaping_elements(
      self,
  ) -> tuple[list[ShapingFlap], list[ShapingRiver]]:
    """Shaping flaps/rivers for every stick of the unmerged tree."""
    data = self.shaping_heuristic_data
    return data['shaping_flaps'], data['shaping_rivers']

  @functools.cached_property
  def shaping_elements(
      self,
  ) -> tuple[list[ShapingFlap], list[ShapingRiver]]:
    """Unmerged elements, with grid sizes doubled for narrowed elements."""
    flaps, rivers = self.unmerged_shaping_elements
    if not self.narrowed_elements:
      return flaps, rivers
    to_update = set()
    flap_info = self.shaping_heuristic_data['flap_info']
    for element_name in self.narrowed_elements:
      queue = [element_name]
      while queue:
        current = queue.pop(0)
        if current in to_update:
          continue
        to_update.add(current)
        if current in flap_info:
          queue.extend(flap_info[current].children)
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
    """Hierarchy of the (possibly narrowed) unmerged flaps."""
    flaps, _ = self.shaping_elements
    return shaping_utils.FlapHierarchy([f.to_legacy_rectangle() for f in flaps])

  @functools.cached_property
  def clip_pattern_adapter(
      self,
  ) -> clip_pattern_shaping_adapter.ClipPatternShapingAdapter:
    """Drawing-machine adapter positioned on every layer of each flap."""
    flaps, rivers = self.shaping_elements
    root_flaps = {r.name for r in self.flap_hierarchy.parent_map.values()}
    packing_svg = shaping_packing_utils.generate_svg_string(
        flap_rectangles=[
            f.to_legacy_rectangle() for f in flaps if f.name in root_flaps
        ],
        rivers_rectangles=[r.covering_legacy_rectangles for r in rivers],
        scale_factor=self.solving_grid_size,
    )
    return clip_pattern_shaping_adapter.ClipPatternShapingAdapter.from_base_cp_and_packing_svg(
        base_cp=self.base_pattern, packing_svg=packing_svg
    )

  @functools.cached_property
  def base_graph_data(self) -> geometric_simple_fold_lib.GraphData:
    """Folded-form graph of the base pattern."""
    return geometric_simple_fold_lib.prepare_graph_data(self.base_pattern)

  @property
  def active_flat_pattern(self) -> crease_pattern_lib.CreasePattern:
    """2-D Kawasaki-clean backbone pattern (`flat_backbone` when set)."""
    return (
        self.flat_backbone if self.flat_backbone is not None else self.pattern
    )

  @property
  def graph_data(self) -> geometric_simple_fold_lib.GraphData:
    """Folded-form graph of the 2-D backbone pattern (cached for the base)."""
    active = self.active_flat_pattern
    if active is self.base_pattern:
      return self.base_graph_data
    return geometric_simple_fold_lib.prepare_graph_data(active)

  def element(self, name: str) -> ShapingElement:
    """Returns the shaping flap or river called `name`."""
    flaps, rivers = self.shaping_elements
    return shaping_elements.get_element_by_name(name, flaps, rivers)

  def element_names(self) -> list[str]:
    flaps, rivers = self.shaping_elements
    return [e.name for e in flaps] + [e.name for e in rivers]

  # -- Updates -------------------------------------------------------------

  def replace(self, **changes: Any) -> 'ShapingState':
    """`dataclasses.replace` that keeps still-valid cached derived data."""
    new_state = dataclasses.replace(self, **changes)
    keep = set(_BASE_CACHED_PROPERTIES)
    if new_state.narrowed_elements == self.narrowed_elements:
      keep |= set(_NARROWING_CACHED_PROPERTIES)
    for name in keep:
      if name in self.__dict__:
        new_state.__dict__[name] = self.__dict__[name]
    return new_state

  def with_pattern(
      self, pattern: crease_pattern_lib.CreasePattern
  ) -> 'ShapingState':
    """Returns a copy with a new current pattern (checked if configured)."""
    if self.config.check_flat_foldability:
      ok_ff, err_ff = fidelity.preserves_foldability(self.pattern, pattern)
      if not ok_ff:
        raise ValueError(err_ff)
    if not fidelity.has_3d_overlay(pattern):
      next_backbone = None
    elif pattern.local_flat_foldability():
      next_backbone = fidelity.as_flat_backbone(pattern)
    else:
      next_backbone = (
          self.flat_backbone
          if self.flat_backbone is not None
          else fidelity.as_flat_backbone(self.pattern)
      )
    return self.replace(pattern=pattern, flat_backbone=next_backbone)

  def reset(self) -> 'ShapingState':
    """Returns a copy back at the base pattern with no narrowing/history."""
    return self.replace(
        pattern=self.base_pattern,
        narrowed_elements=(),
        history=(),
        flat_backbone=None,
    )


# Cached properties that depend only on the base pattern / packing / tree.
_BASE_CACHED_PROPERTIES = (
    'hinge_segments',
    'paper_layout',
    'merged_shaping_elements',
    'shaping_heuristic_data',
    'unmerged_shaping_elements',
    'base_graph_data',
)
# Cached properties that additionally depend on `narrowed_elements`.
_NARROWING_CACHED_PROPERTIES = (
    'shaping_elements',
    'flap_hierarchy',
    'clip_pattern_adapter',
)

_ToolFn = TypeVar('_ToolFn', bound=Callable[..., 'ShapingState'])


def _crease_multiset(
    pattern: crease_pattern_lib.CreasePattern,
) -> Mapping[tuple[Any, ...], int]:
  """Multiset of (rounded endpoints, fold, fold_percent) over M/V creases.

  Two patterns with equal multisets fold identically, so a tool whose output
  has the same multiset as its input made no visible change.

  Args:
    pattern: A crease pattern.

  Returns:
    Mapping from crease signature to its multiplicity.
  """
  counts: dict[tuple[Any, ...], int] = {}
  for c in pattern.creases:
    if not c.mountain_or_valley():
      continue
    p1 = (round(c.p1_init.x, 6), round(c.p1_init.y, 6))
    p2 = (round(c.p2_init.x, 6), round(c.p2_init.y, 6))
    if p2 < p1:
      p1, p2 = p2, p1
    key = (p1, p2, c.fold, round(c.fold_percent, 5))
    counts[key] = counts.get(key, 0) + 1
  return counts


def _preserve_3d_overlay(fn: _ToolFn) -> _ToolFn:
  """Runs a 2-D backbone tool on `state.flat_backbone` and replays 3-D relief.

  The replay restores the 3-D overlay's partial fold angles on the creases it
  owns. A tool whose only effect is a fold-percent change on such creases
  (e.g. `squash_fold`'s pleat flare after `rotation_3d` posed the flap) would
  therefore report success while the pattern stays byte-identical. That case
  is detected and the tool is retried directly on the 3-D pattern; if that
  leaves the pattern unchanged too, the call fails instead of reporting a
  no-op as progress.
  """

  @functools.wraps(fn)
  def wrapped(*args: Any, **kwargs: Any) -> ShapingState:
    state: Any = args[0] if args else kwargs.get('state')
    if state is None or getattr(state, 'flat_backbone', None) is None:
      out = fn(*args, **kwargs)
      if getattr(
          out, 'flat_backbone', None
      ) is None and fidelity.has_3d_overlay(out.pattern):
        if out.pattern.local_flat_foldability():
          out = out.replace(
              flat_backbone=fidelity.as_flat_backbone(out.pattern)
          )
        elif state is not None and state.pattern.local_flat_foldability():
          out = out.replace(
              flat_backbone=fidelity.as_flat_backbone(state.pattern)
          )
      return out

    def _call_with_state(st: ShapingState) -> ShapingState:
      if args:
        return fn(st, *args[1:], **kwargs)
      call_kw = dict(kwargs)
      call_kw['state'] = st
      return fn(*args, **call_kw)

    notes = _NOTES.get()
    notes_len = len(notes) if notes is not None else 0
    resolved = _RESOLVED_ARGUMENTS.get()
    saved_resolved = dict(resolved) if resolved is not None else None
    primary_error: Exception | None = None
    flat_called_ok = False
    replay_discarded_change = False
    before = _crease_multiset(state.pattern)
    try:
      flat_in = state.replace(pattern=state.flat_backbone, flat_backbone=None)
      flat_out = _call_with_state(flat_in)
      flat_called_ok = True
      combined = fidelity.replay_3d_overlay(
          state.flat_backbone, state.pattern, flat_out.pattern
      )
      if _crease_multiset(combined) == before and _crease_multiset(
          flat_out.pattern
      ) != _crease_multiset(flat_in.pattern):
        replay_discarded_change = True
        raise ValueError(
            f'{fn.__name__} changed only fold angles of creases that belong'
            ' to the 3-D overlay; replaying the overlay discarded the change'
            ' (pattern unchanged). Retrying on the 3-D pattern.'
        )
      if state.config.check_flat_foldability:
        ok_ff, err_ff = fidelity.preserves_foldability(
            state.pattern,
            combined,
            allow_new_3d_vertices=not flat_out.pattern.local_flat_foldability(),
        )
        if not ok_ff:
          raise ValueError(err_ff)
      if kwargs.get('verify_fold', True):
        model = fidelity.FoldedModel.from_pattern(combined)
        if model.vertex_error > _MAX_RENDERABLE_VERTEX_ERROR:
          raise ValueError(
              f'3-D overlay fold vertex error {model.vertex_error:.2f}% >'
              f' {_MAX_RENDERABLE_VERTEX_ERROR}%'
          )
      if not fidelity.has_3d_overlay(combined):
        next_bb = None
      elif flat_out.flat_backbone is not None:
        next_bb = flat_out.flat_backbone
      elif flat_out.pattern.local_flat_foldability():
        next_bb = fidelity.as_flat_backbone(flat_out.pattern)
      else:
        next_bb = state.flat_backbone
      return flat_out.replace(pattern=combined, flat_backbone=next_bb)
    except (
        ValueError,
        AssertionError,
        KeyError,
        IndexError,
        ZeroDivisionError,
    ) as err:
      primary_error = err
      if notes is not None:
        del notes[notes_len:]
      if resolved is not None and saved_resolved is not None:
        resolved.clear()
        resolved.update(saved_resolved)
      if not flat_called_ok and len(state.flat_backbone.creases) == len(
          state.pattern.creases
      ):
        raise

    try:
      direct_out = _call_with_state(state.replace(flat_backbone=None))
    except Exception as exc:  # pylint: disable=broad-exception-caught
      assert primary_error is not None
      raise primary_error from exc
    if (
        replay_discarded_change
        and _crease_multiset(direct_out.pattern) == before
    ):
      raise ValueError(
          f'{fn.__name__} left the pattern unchanged: its fold-angle change'
          ' sits on creases that the 3-D overlay already owns (the flap is'
          ' posed). Apply it before the 3-D tools or use another ridge /'
          ' element.'
      ) from primary_error
    if state.config.check_flat_foldability:
      ok_ff, err_ff = fidelity.preserves_foldability(
          state.pattern,
          direct_out.pattern,
          allow_new_3d_vertices=True,
      )
      if not ok_ff:
        raise ValueError(err_ff)
    if kwargs.get('verify_fold', True):
      direct_model = fidelity.FoldedModel.from_pattern(direct_out.pattern)
      if direct_model.vertex_error > _MAX_RENDERABLE_VERTEX_ERROR:
        raise ValueError(
            '3-D overlay fold vertex error'
            f' {direct_model.vertex_error:.2f}% >'
            f' {_MAX_RENDERABLE_VERTEX_ERROR}%'
        )
    if (
        fidelity.has_3d_overlay(direct_out.pattern)
        and direct_out.flat_backbone is None
    ):
      direct_out = direct_out.replace(flat_backbone=state.flat_backbone)
    return direct_out

  return cast(_ToolFn, wrapped)


def serialize_pattern_svg(pattern: crease_pattern_lib.CreasePattern) -> str:
  """Serializes a crease pattern to SVG string for state persistence."""
  return svg_parser.to_svg(pattern, grid=-1, ndigits=8)


def deserialize_pattern_svg(
    svg_str: str, *, tolerance: float = crease_utils.DEFAULT_TOLERANCE
) -> crease_pattern_lib.CreasePattern:
  """Parses a crease pattern from an SVG string."""
  pattern = svg_parser.parse_svg(svg_str, grid_size=-1)
  pattern.tolerance = tolerance
  return pattern


# ------------------------------------------------------------------------------
# Context for the agent.
# ------------------------------------------------------------------------------


def element_table(state: ShapingState) -> str:
  """Describes every stick as the RL shaping prompt does.

  Format per line: `name: start->end r=<length> θ=<azimuth> ϕ=<elevation>
  d=<width> t=<type>` with type R (river), T (triangular tip flap), F
  (rectangular tip flap) or S (sub-flap without a tip), in breadth-first order
  from the root, preceded by `central_fold_removed=<bool>`.

  Args:
    state: The shaping state.

  Returns:
    The table as text.
  """
  lines = [f'central_fold_removed={state.central_crease_removed}']
  for stick in traversal.breadth_first_traversal(state.tree):
    if stick.is_river:
      type_ = 'R'
    elif stick.flap_tip_shape == stick_lib.FlapTipShape.TRIANGULAR:
      type_ = 'T'
    elif stick.flap_tip_shape == stick_lib.FlapTipShape.RECTANGULAR:
      type_ = 'F'
    else:
      type_ = 'S'
    lines.append(
        f'{stick.name}: {stick.start}->{stick.end} r={stick.length}'
        f' θ={stick.azimuth_angle:.0f} ϕ={stick.elevation_angle:.0f}'
        f' d={stick.width:.1f} t={type_}'
    )
  return '\n'.join(lines)


def flap_orientation_info(state: ShapingState) -> dict[str, Any]:
  """Reports how each flap points when the base is folded (UP or DOWN).

  Args:
    state: The shaping state.

  Returns:
    `{'root_stick': name, 'orientations': {flap: 'UP'|'DOWN'}}` keyed by
    merged flap names (as in the packing). The root stick should not be shaped.
  """
  merged_tree, _ = state.tree.merge_linear_chains()
  flaps, rivers = state.merged_shaping_elements
  follow_root = (
      heuristics.compute_root_stick_direction_follow_stick_figure_root(
          merged_tree, flaps, rivers
      )
  )
  flap_info = heuristics.get_flap_info_dict(
      merged_tree, state.packing, state.solving, follow_root
  )
  orientations = {}
  for name, node in flap_info.items():
    direction = node.absolute_direction
    if direction == heuristics.AbsFlapDirection.DOWN:
      orientations[name] = 'DOWN'
    else:
      orientations[name] = 'UP'
  root = heuristics.get_first_stick_for_shaping(merged_tree)
  return {'root_stick': root.name, 'orientations': orientations}


def describe(state: ShapingState) -> dict[str, Any]:
  """Summary of the state for reports: elements, grid sizes, crease count."""
  flaps, rivers = state.shaping_elements
  return {
      'name': state.tree.name,
      'packing_grid_size': state.packing_grid_size,
      'solving_grid_size': state.solving_grid_size,
      'central_crease_removed': state.central_crease_removed,
      'num_creases': len(state.pattern.creases),
      'flaps': [f.name for f in flaps],
      'rivers': [r.name for r in rivers],
      'narrowed_elements': list(state.narrowed_elements),
      'num_tool_calls': len(state.history),
  }


# ------------------------------------------------------------------------------
# Simple fold family (zig-zag shaping).
# ------------------------------------------------------------------------------


def _zigzag_params(
    state: ShapingState,
    *,
    element_name: str,
    angle: float,
    depth: float,
    fold_percent: float = 1.0,
) -> tuple[ShapingElement, zigzag_shaping.ZigzagShapingParamsForElement]:
  """Builds zig-zag parameters for one element from tool arguments."""
  element = state.element(element_name)
  angle_radians = _pre_process_angle(angle)
  args: dict[str, Any] = {
      'target_flap_angle': abs(angle_radians),
      'even_phase': angle_radians >= 0,
      'fold_order': zigzag_shaping_utils.MOUNTAIN_TO_VALLEY,
      'fold_percent': fold_percent,
  }
  if isinstance(element, ShapingFlap):
    args['depth'] = min(
        int(element.grid_unit_radius * depth), element.grid_unit_radius - 1
    )
    params = shaping_types.ZigzagParams(**args)
  else:
    args['depth'] = min(int(element.width * depth), element.width - 1)
    args['bank_index'] = element.longer_bank_index
    params = shaping_types.RiverZigzagParams(**args)
  return element, zigzag_shaping.ZigzagShapingParamsForElement(
      element_name=element_name, zigzag_params=params
  )


def _apply_simple_fold_heuristics(
    state: ShapingState,
    element: ShapingElement,
    params: zigzag_shaping.ZigzagShapingParamsForElement,
    *,
    narrow: bool,
    convert_small_angles: bool | None = None,
) -> list[zigzag_shaping.ZigzagShapingParamsForElement]:
  """Adjusts fold order / depth / phase of `params` using the tree.

  Args:
    state: Current state.
    element: The element being folded.
    params: Zig-zag parameters from the tool arguments.
    narrow: Whether the fold also narrows the element.
    convert_small_angles: Overrides `state.config.convert_small_angles` (never
      applied when narrowing).

  Returns:
    The adjusted zig-zag parameters (two requests when a small angle is
    converted into two large ones).
  """
  if convert_small_angles is None:
    convert_small_angles = state.config.convert_small_angles
  flaps, rivers = state.shaping_elements
  flap_info = state.shaping_heuristic_data['flap_info']
  element_orientations = {
      name: node.absolute_direction for name, node in flap_info.items()
  }
  target_flap_angle = params.zigzag_params.target_flap_angle or 0.0
  signed = dataclasses.replace(
      params,
      zigzag_params=dataclasses.replace(
          params.zigzag_params,
          target_flap_angle=target_flap_angle
          * (1 if params.zigzag_params.even_phase else -1),
      ),
  )
  adjusted = simple_fold_shaping.apply_heuristics_to_shaping_params(
      grid_size=element.grid_size,
      flaps=flaps,
      rivers=rivers,
      element_orientations=element_orientations,
      hinge_segments=state.hinge_segments,
      element_shaping_params=[signed],
      use_heuristics_for_fold_order=state.config.use_heuristics_for_fold_order,
      use_heuristics_for_depth=state.config.use_heuristics_for_depth,
      use_heuristics_for_phase=state.config.use_heuristics_for_phase,
      # Converting small angles does not play nicely with narrowing.
      convert_small_angles=convert_small_angles and not narrow,
  )
  if not adjusted:
    raise ValueError('Shaping heuristics produced no parameters.')
  return list(adjusted)


@_preserve_3d_overlay
def simple_fold(
    state: ShapingState,
    *,
    element_name: str,
    angle: float,
    depth: float = 0.0,
    narrow: bool = False,
    fold_percent: float = 1.0,
    fold_order: str | None = None,
) -> ShapingState:
  """Folds a flap or river to an angle with a zig-zag (simple fold).

  Args:
    state: Current state.
    element_name: Flap or river name from the stick figure.
    angle: Target angle in degrees, in [-180, -8] or [8, 180]; the sign picks
      the zig-zag phase (side). Angles in (-8, 8) are clipped to +/-8.
    depth: Fraction in [0, 1) of the element length at which the fold starts (0
      = base of the flap).
    narrow: Also halve the width of the folded flap (not for rivers).
    fold_percent: Fraction of the fold to apply, 1.0 = fully folded (180).
      Smaller values leave the fold partly open for a 3-D effect.
    fold_order: `MV` (mountain first) or `VM` (valley first) to force the
      zig-zag's crease order instead of the tree heuristic; with `fold_percent`
      < 1 it picks the side an open fold bends to.

  Returns:
    The new state.

  Raises:
    ValueError: On a zero angle, unknown element, or when no creases result.
  """
  if math.isclose(angle, 0):
    raise ValueError('angle must be non-zero.')
  original_config = state.config
  orders = {
      'MV': zigzag_shaping_utils.MOUNTAIN_TO_VALLEY,
      'VM': zigzag_shaping_utils.VALLEY_TO_MOUNTAIN,
  }
  if fold_order is not None:
    if fold_order not in orders:
      raise ValueError(f'fold_order must be MV or VM, got {fold_order!r}.')
    state = state.replace(
        config=dataclasses.replace(
            state.config, use_heuristics_for_fold_order=False
        )
    )
  element, params = _zigzag_params(
      state,
      element_name=element_name,
      angle=angle,
      depth=depth,
      fold_percent=fold_percent,
  )
  if fold_order is not None:
    params = dataclasses.replace(
        params,
        zigzag_params=dataclasses.replace(
            params.zigzag_params, fold_order=orders[fold_order]
        ),
    )
  if isinstance(element, ShapingRiver) and narrow:
    raise ValueError('Rivers cannot be narrowed; use narrow=False.')
  flaps, rivers = state.shaping_elements
  # The small-angle heuristic turns a fold below 90 degrees into two
  # near-180-degree zig-zags of tiny depth (~0.06 grid units). On fine grids
  # (children of narrowed elements have a doubled grid) that depth falls below
  # the point tolerance and the zig-zag walk leaves the element
  # (`zigzag_shaping.py: assert edge_index is not None`), e.g. knees on
  # narrowed legs. If the converted fold fails, retry once with the plain
  # zig-zag at the requested angle; calls that succeeded before are unchanged.
  convert_modes = [state.config.convert_small_angles and not narrow]
  if (
      convert_modes[0]
      and state.config.apply_simple_fold_heuristics
      and (params.zigzag_params.target_flap_angle or 0.0) < math.pi / 2
  ):
    convert_modes.append(False)
  errors: list[Exception] = []
  for convert_small_angles in convert_modes:
    if state.config.apply_simple_fold_heuristics:
      element_params = _apply_simple_fold_heuristics(
          state,
          element,
          params,
          narrow=narrow,
          convert_small_angles=convert_small_angles,
      )
    else:
      element_params = [params]
    try:
      shaped = simple_fold_shaping.shape_cp_with_simple_folds(
          cp=state.pattern,
          element_shaping_params=element_params,
          grid_size=element.grid_size,
          flaps=flaps,
          rivers=rivers,
          central_crease_removed=state.central_crease_removed,
          narrow=narrow,
          light_mode=not state.config.check_all_crease_intersections,
      )
      if len(shaped.creases) == len(state.pattern.creases):
        raise ValueError(
            f'No shaping creases found for simple fold of {element_name}.'
        )
      new_state = state.with_pattern(shaped)
    except (ValueError, AssertionError) as error:
      errors.append(error)
      continue
    if narrow:
      new_state = new_state.replace(
          narrowed_elements=new_state.narrowed_elements + (element_name,),
      )
    return new_state.replace(config=original_config)
  if len(errors) == 1:
    raise errors[0]
  raise ValueError(
      f'simple fold of {element_name} failed with the small-angle conversion'
      f' ({_describe_error(errors[0])}) and as a plain zig-zag'
      f' ({_describe_error(errors[1])}).'
  ) from errors[1]


@_preserve_3d_overlay
def simple_fold_heuristic(
    state: ShapingState, *, element_name_list: Sequence[str] | None = None
) -> ShapingState:
  """Shapes elements with angles computed from the stick figure.

  Must be used before any other shaping tool (it reasons about the base).

  Args:
    state: Current state.
    element_name_list: Elements to shape; all elements when empty/None.

  Returns:
    The new state.
  """
  flaps, rivers = state.merged_shaping_elements
  shaped = heuristics_utils.shape_cp_with_heuristics_pipeline_wrapper(
      cp=state.pattern,
      stick_figure=state.tree,
      packing_output=state.packing,
      solving_output=state.solving,
      solving_grid_size=state.solving_grid_size,
      shaping_flaps=flaps,
      shaping_rivers=rivers,
      central_crease_removed=state.central_crease_removed,
      element_name_list=list(element_name_list) if element_name_list else None,
      shape_rivers=True,
      shape_with_unmerged_tree=True,
  )
  return state.with_pattern(shaped)


@_preserve_3d_overlay
def crimp_fold(
    state: ShapingState,
    *,
    element_name: str,
    angle: float,
    depth: float,
) -> ShapingState:
  """Adds a crimp fold: changes the angle and shortens the flap (knees, tails).

  Args:
    state: Current state.
    element_name: Flap name.
    angle: Angle in degrees in (-180, -8] or [8, 180); 180 is not allowed.
    depth: Fraction in (0, 1) of the flap length where the crimp starts.

  Returns:
    The new state.

  Raises:
    ValueError: On invalid arguments or if neither fold order is flat-foldable
      and the `crimp_fold_3d` fallback fails too.
  """
  if math.isclose(angle, 0):
    raise ValueError('angle must be non-zero.')
  if math.isclose(
      abs(angle_utils.wrap_angle_centered(angle, 360.0)), 180.0, abs_tol=1e-5
  ):
    raise ValueError('angle=180 is not allowed for crimp folds.')
  if not 0.0 < depth < 1.0:
    raise ValueError(f'depth={depth} must be in (0, 1).')
  element, params = _zigzag_params(
      state, element_name=element_name, angle=angle, depth=depth
  )
  zigzag = params.zigzag_params
  crimp_params = crimp_fold_shaping.CrimpFoldShapingParams(
      element_name=element_name,
      crimp_fold_params=[
          crimp_fold_shaping.CrimpFoldParams(
              depth=int(zigzag.depth),
              fold_order=[crease_utils.Fold.MOUNTAIN, crease_utils.Fold.VALLEY],
              fold_percent=1.0,
              zigzag_flap_angle=zigzag.target_flap_angle or 0.0,
              zigzag_even_phase=zigzag.even_phase,
              zigzag_fold_order=[
                  crease_utils.Fold.VALLEY,
                  crease_utils.Fold.MOUNTAIN,
              ],
              bank_index=getattr(zigzag, 'bank_index', None),
          )
      ],
  )
  flaps, rivers = state.shaping_elements
  inscribed, zigzags = crimp_fold_shaping.get_crimp_fold_creases(
      cp=state.pattern,
      grid_size=element.grid_size,
      flaps=flaps,
      element_shaping_params=[crimp_params],
      rivers=rivers,
      central_crease_removed=state.central_crease_removed,
  )
  new_creases = list(inscribed) + list(zigzags)
  if not new_creases:
    raise ValueError(
        f'No shaping creases found for crimp fold of {element_name}.'
    )
  pattern = state.pattern
  tips, _ = crease_pattern_lib.get_tips_and_grid_creases(
      pattern.creases + new_creases, grid_size=-1
  )
  split_inscribed = crease_pattern_lib.split_creases(
      inscribed, pattern.creases + list(zigzags), tips
  )
  split_zigzags = crease_pattern_lib.split_creases(
      zigzags, pattern.creases + list(inscribed) + list(zigzags), tips
  )
  split_base = crease_pattern_lib.split_creases(
      pattern.creases, split_inscribed + split_zigzags, tips
  )
  fold_orders = [
      (
          [crease_utils.Fold.MOUNTAIN, crease_utils.Fold.VALLEY],
          [crease_utils.Fold.VALLEY, crease_utils.Fold.MOUNTAIN],
      ),
      (
          [crease_utils.Fold.VALLEY, crease_utils.Fold.MOUNTAIN],
          [crease_utils.Fold.MOUNTAIN, crease_utils.Fold.VALLEY],
      ),
  ]
  last_error: Exception | None = None
  for fold_order, zigzag_fold_order in fold_orders:
    try:
      new_inscribed, new_zigzags = crimp_fold_shaping.assign_crimp_fold_orders(
          split_inscribed, split_zigzags, fold_order, zigzag_fold_order
      )
      modified = crimp_fold_shaping.fix_flat_foldability(
          list(split_base),
          inscribed_rectangle_creases=new_inscribed,
          zigzag_creases=new_zigzags,
      )
      candidate = crease_pattern_lib.CreasePattern(
          creases=modified + new_inscribed + new_zigzags, light_mode=True
      )
      return state.with_pattern(candidate)
    except (ValueError, AssertionError) as error:
      last_error = error
  # Phase 10 (Matrix 10): the zig-zag crimp is not flat-foldable on ~30% of
  # the gallery flaps (edge/corner flaps, flaps next to rivers). The local
  # 3-D crimp cuts only the flap's own layers in the folded form and composes
  # with the rest of the pattern; use it as a fallback with the same intent.
  try:
    fallback = shaping_fold3d.crimp_fold_3d(
        state, element_name=element_name, angle=angle, depth=depth
    )
  except (ValueError, AssertionError) as fallback_error:
    raise ValueError(
        f'Crimp fold of {element_name} failed for both fold orders; last'
        f' error: {last_error}. crimp_fold_3d fallback: {fallback_error}'
    ) from fallback_error
  _note(
      f'Zig-zag crimp of {element_name} not flat-foldable ({last_error});'
      f' applied crimp_fold_3d instead. {shaping_fold3d.last_report_message()}'
  )
  return fallback


@_preserve_3d_overlay
def reverse_fold(
    state: ShapingState,
    *,
    element_name: str,
    angle: float,
    orientation: str = 'horizontal',
    fold_type: str = 'valley',
) -> ShapingState:
  """Adds a reverse fold (heads, tails, beaks, horns, limbs).

  Tries the requested orientation first and the other one as a fallback, and
  both fold assignments, keeping the first flat-foldable result; when none is
  flat-foldable it applies the local `reverse_fold_3d` (depth 0.5) instead.

  Args:
    state: Current state.
    element_name: Flap name.
    angle: Angle in degrees in [-180, 180].
    orientation: `horizontal` or `vertical`.
    fold_type: Base fold type to try first, `valley` or `mountain`.

  Returns:
    The new state.

  Raises:
    ValueError: If no orientation/fold assignment is flat-foldable and the
      `reverse_fold_3d` fallback fails too.
  """
  if orientation not in ('horizontal', 'vertical'):
    raise ValueError("orientation must be 'horizontal' or 'vertical'.")
  element, params = _zigzag_params(
      state, element_name=element_name, angle=angle, depth=0.0
  )
  zigzag = params.zigzag_params
  reverse_params = reverse_fold_shaping.make_reverse_fold_params(
      flap_angle=zigzag.target_flap_angle,
      fold_order=(crease_utils.Fold.VALLEY, crease_utils.Fold.MOUNTAIN),
      even_phase=zigzag.even_phase,
      bank_index=getattr(zigzag, 'bank_index', None),
      depth=int(zigzag.depth or 0),
      fold_percent=zigzag.fold_percent,
  )
  element_params = reverse_fold_shaping.ReverseFoldShapingParams(
      flap_name=element_name,
      reverse_fold_params=[reverse_params],
      orientation=orientation,
  )
  flaps, rivers = state.shaping_elements
  # Use the element's own grid: children of narrowed elements live on a doubled
  # grid (see `ShapingState.shaping_elements`); zig-zags on the solving grid
  # miss the narrowed layers and are not flat-foldable.
  base_creases, instructions = reverse_fold_shaping.get_reverse_fold_creases(
      cp=state.pattern,
      grid_size=element.grid_size,
      rectangles=flaps,
      element_shaping_params=[element_params],
      rivers=rivers,
  )
  zigzag_creases = [instruction.crease for instruction in instructions]
  tips, _ = crease_pattern_lib.get_tips_and_grid_creases(
      base_creases + zigzag_creases, grid_size=-1
  )
  split_base = crease_pattern_lib.split_creases(
      base_creases, zigzag_creases, tips
  )
  split_zigzags = [
      crease_pattern_lib.split_creases(
          [instruction.crease], base_creases + zigzag_creases, tips
      )
      for instruction in instructions
  ]
  base_fold = _FOLD_TYPES.get(fold_type.lower())
  if base_fold is None:
    raise ValueError("fold_type must be 'valley' or 'mountain'.")
  if element_params.fold_type:
    base_fold = element_params.fold_type
  other = 'vertical' if orientation == 'horizontal' else 'horizontal'
  last_error: Exception | None = None
  for current_orientation in (orientation, other):
    for fold, flip_base in (
        (base_fold, False),
        (base_fold, True),
        (base_fold.flip(), True),
        (base_fold.flip(), False),
    ):
      try:
        assigned = reverse_fold_shaping.assign_reverse_fold_orders(
            instructions, fold, current_orientation, flip_base
        )
        final_zigzags = []
        for assigned_zigzag, geometries in zip(
            assigned, split_zigzags, strict=True
        ):
          for geometry in geometries:
            final_zigzags.append(
                dataclasses.replace(geometry, fold=assigned_zigzag.fold)
            )
        candidate = crease_pattern_lib.CreasePattern(
            creases=list(split_base) + final_zigzags, light_mode=True
        )
        return state.with_pattern(candidate)
      except (ValueError, AssertionError) as error:
        last_error = error
  # Phase 10 (Matrix 10): the zig-zag reverse fold is flat-foldable on only
  # ~20% of the gallery flaps (M9: 40%). The local 3-D reverse fold cuts the
  # flap's own layers in the folded form; use it as a fallback with the same
  # intent (turn the distal part by `angle`).
  try:
    fallback = shaping_fold3d.reverse_fold_3d(
        state, element_name=element_name, angle=angle, depth=0.5
    )
  except (ValueError, AssertionError) as fallback_error:
    raise ValueError(
        f'Reverse fold of {element_name} failed for all orientations and fold'
        f' assignments; last error: {last_error}. reverse_fold_3d fallback:'
        f' {fallback_error}'
    ) from fallback_error
  _note(
      f'Zig-zag reverse fold of {element_name} not flat-foldable'
      f' ({last_error}); applied reverse_fold_3d instead.'
      f' {shaping_fold3d.last_report_message()}'
  )
  return fallback


# ------------------------------------------------------------------------------
# Clip pattern (drawing machine) shaping.
# ------------------------------------------------------------------------------


@_preserve_3d_overlay
def clip_pattern_fold(
    state: ShapingState,
    *,
    element_name: str,
    step: float = 0.0,
    bend_angle: float = 0.0,
    rotate_angle: float = 0.0,
    enable_narrow: bool = False,
    narrow_symmetric: bool | None = None,
    narrow_n_times: int = 2,
    template_name: str = 'fold',
    length: float | None = None,
    pleat_distance: float = 0.5,
    verify_fold: bool = True,
) -> ShapingState:
  """Draws a shaping template on every layer of an element."""
  result_state = shaping_extras.clip_pattern_fold(
      state,
      element_name=element_name,
      step=step,
      bend_angle=bend_angle,
      rotate_angle=rotate_angle,
      enable_narrow=enable_narrow,
      narrow_symmetric=narrow_symmetric,
      narrow_n_times=narrow_n_times,
      template_name=template_name,
      length=length,
      pleat_distance=pleat_distance,
      verify_fold=verify_fold,
  )
  msg = shaping_extras.last_report_message()
  if msg:
    _note(msg)
  return result_state


@_preserve_3d_overlay
def pleat_to_hide(
    state: ShapingState,
    *,
    element_name: str,
    step: float = 0.0,
    length: float | None = None,
    pleat_distance: float = 0.5,
    verify_fold: bool = True,
) -> ShapingState:
  """Accordion pleats an element to shorten it without repacking."""
  result_state = shaping_extras.pleat_to_hide(
      state,
      element_name=element_name,
      step=step,
      length=length,
      pleat_distance=pleat_distance,
      verify_fold=verify_fold,
  )
  msg = shaping_extras.last_report_message()
  if msg:
    _note(msg)
  return result_state


@_preserve_3d_overlay
def simple_fold_narrow(
    state: ShapingState,
    *,
    element_name: str,
    angle: float = 45.0,
    depth: float = 0.0,
    fold_order: str | None = None,
    fold_percent: float = 1.0,
    fallback_clip_narrow: bool = True,
    verify_fold: bool = True,
) -> ShapingState:
  """Narrows a flap or river with the hardened narrowing pipeline."""
  result_state = shaping_extras.simple_fold_narrow(
      state,
      element_name=element_name,
      angle=angle,
      depth=depth,
      fold_order=fold_order,
      fold_percent=fold_percent,
      fallback_clip_narrow=fallback_clip_narrow,
      verify_fold=verify_fold,
  )
  msg = shaping_extras.last_report_message()
  if msg:
    _note(msg)
  return result_state


@_preserve_3d_overlay
def narrow_central_flap(
    state: ShapingState,
    *,
    element_name: str,
    tip_width: float = 0.25,
    start_offset: float = 0.0,
    fold_type: str = 'auto',
    verify_fold: bool = True,
    require_symmetric: bool = True,
) -> ShapingState:
  """Narrows both halves of a flap spanning x = 0.5 with a symmetric taper."""
  result_state = shaping_extras.narrow_central_flap(
      state,
      element_name=element_name,
      tip_width=tip_width,
      start_offset=start_offset,
      fold_type=fold_type,
      verify_fold=verify_fold,
      require_symmetric=require_symmetric,
  )
  msg = shaping_extras.last_report_message()
  if msg:
    _note(msg)
  return result_state


@_preserve_3d_overlay
def curve_flap(
    state: ShapingState,
    *,
    element_name: str,
    n_segments: int = 3,
    total_bend_deg: float = 90.0,
    direction: str = 'up',
    method: str = 'auto',
    fold_percent: float = 1.0,
    verify_fold: bool = True,
) -> ShapingState:
  """Curves a flap or merged chain into a smooth arc."""
  result_state = shaping_extras.curve_flap(
      state,
      element_name=element_name,
      n_segments=n_segments,
      total_bend_deg=total_bend_deg,
      direction=direction,
      method=method,
      fold_percent=fold_percent,
      verify_fold=verify_fold,
  )
  msg = shaping_extras.last_report_message()
  if msg:
    _note(msg)
  return result_state


def inscribed_flap(
    state: ShapingState,
    *,
    element_name: str,
    depths: Sequence[int] | int = (1,),
    fold_order: str = 'MV',
    fold_percent: float = 0.5,
    verify_fold: bool = True,
) -> ShapingState:
  """Shapes a wide flap or river with concentric inscribed rectangles."""
  result_state = shaping_extras.inscribed_flap(
      state,
      element_name=element_name,
      depths=depths,
      fold_order=fold_order,
      fold_percent=fold_percent,
      verify_fold=verify_fold,
  )
  msg = shaping_extras.last_report_message()
  if msg:
    _note(msg)
  return result_state


# ------------------------------------------------------------------------------
# Tool notes: details a successful tool call reports next to "succeeded".
# ------------------------------------------------------------------------------

_NOTES: contextvars.ContextVar[list[str] | None] = contextvars.ContextVar(
    'shaping_tool_notes', default=None
)
# Arguments a searching tool resolved (e.g. the plan `thin_limbs_3d` chose);
# `apply_tool` records them in the history instead of the call's arguments so
# a replay applies the same folds without searching again.
_RESOLVED_ARGUMENTS: contextvars.ContextVar[dict[str, Any] | None] = (
    contextvars.ContextVar('shaping_resolved_arguments', default=None)
)


def _resolve_arguments(arguments: dict[str, Any]) -> None:
  """Records the arguments that replay the tool call being applied."""
  holder = _RESOLVED_ARGUMENTS.get()
  if holder is not None:
    holder.clear()
    holder.update(arguments)


def _note(message: str) -> None:
  """Adds a note to the result message of the tool call being applied."""
  notes = _NOTES.get()
  if notes is not None:
    notes.append(message)
  logging.info('Tool note: %s', message)


class _MessageCollector(std_logging.Handler):
  """Collects the messages of log records (library warnings)."""

  def __init__(self):
    super().__init__(level=std_logging.WARNING)
    self.messages: list[str] = []

  def emit(self, record: std_logging.LogRecord) -> None:
    try:
      self.messages.append(record.getMessage())
    except (TypeError, ValueError):  # Malformed log arguments.
      pass


@contextlib.contextmanager
def _library_warnings() -> Iterator[list[str]]:
  """Collects warnings the shaping library logs while the block runs."""
  collector = _MessageCollector()
  root = std_logging.getLogger()
  root.addHandler(collector)
  try:
    yield collector.messages
  finally:
    root.removeHandler(collector)


@contextlib.contextmanager
def _designer_root_3d(state: 'ShapingState') -> Iterator[None]:
  """Makes the 3-D heuristics measure angles from the stick figure's root.

  `geometric_fold_3d_heuristics` and `heuristics_3d.get_flap_post_dict` call
  `heuristics.get_first_stick_for_shaping`, rotate the stick figure so that
  stick lies at az=0/el=0 and build every element's target posture from it.
  The library's choice can be a tail segment or the rear half of the body;
  then the requested az/el are realized in a pitched or reversed frame. While
  the block runs, that function returns the designer's root stick (mapped to
  its merged chain for merged trees). No-op when
  `state.config.use_designer_root_3d` is False.

  Args:
    state: The shaping state (its tree's root is used).

  Yields:
    Nothing.
  """
  if not state.config.use_designer_root_3d:
    yield
    return
  root_name = state.tree.root
  _, merged_names = state.tree.merge_linear_chains()
  merged_of = {}
  for merged, subs in merged_names.items():
    for sub in subs:
      merged_of[sub] = merged
  original = heuristics.get_first_stick_for_shaping

  def designer_root(tree: stick_figure_lib.StickFigure) -> stick_lib.Stick:
    if root_name in tree:
      return tree[root_name]
    merged_name = merged_of.get(root_name)
    if merged_name is not None and merged_name in tree:
      return tree[merged_name]
    return original(tree)

  heuristics.get_first_stick_for_shaping = designer_root
  try:
    yield
  finally:
    heuristics.get_first_stick_for_shaping = original


# Library warnings of the 3-D heuristics (`geometric_fold_3d_heuristics`) that
# explain why an element received no creases, mapped to advice for the agent.
_3D_SKIP_REASONS = (
    (
        r'Flat foldability violation detected after shaping element: (\S+?)\.',
        (
            'its fold breaks flat-foldability with the current pattern (try'
            ' other el/az, a different depth, rotate it in a separate call, or'
            ' re-solve with --solve_remove_central_crease if only one of a'
            ' mirrored pair fails)'
        ),
    ),
    (
        r'Slope of simple fold direction is too large for element: (\S+?)\.'
        r' Slope: \S+ Skipping',
        'its rotation axis is too steep (nudge az/el by ~10 degrees)',
    ),
    (
        r'Simple fold direction is vertical for element: (\S+?)\.',
        'its rotation axis is vertical (nudge az/el by ~10 degrees)',
    ),
    (
        r'Rotation angle is close to 0 for element: (\S+?)\.',
        'it already points that way (no rotation needed)',
    ),
    (
        r'Cut line start and end points are too close for element: (\S+?)\.',
        'its cut line degenerated (try a different depth)',
    ),
    (
        r'Failed to shape element: (\S+?) with',
        'the 3-D geometry failed (see the workbench log)',
    ),
    (
        r'No simple fold info found for element: (\S+?)\.',
        'no fold could be computed for it from the stick figure',
    ),
)


def _skip_reasons(messages: Sequence[str]) -> dict[str, str]:
  """Maps element names to why the 3-D heuristics skipped them."""
  reasons: dict[str, str] = {}
  for message in messages:
    for pattern, reason in _3D_SKIP_REASONS:
      match = re.search(pattern, message)
      if match:
        reasons.setdefault(match.group(1), reason)
        break
  return reasons


# ------------------------------------------------------------------------------
# Tree-driven 3-D heuristics.
# ------------------------------------------------------------------------------


# `rotation_3d` retries elements left out for flat-foldability with their cut
# this deep into the flap (fraction of the flap from its base).
_LEFT_OUT_RETRY_DEPTH = 0.4


@_preserve_3d_overlay
def rotation_3d(
    state: ShapingState,
    *,
    rotations: Sequence[Mapping[str, Any]],
    require_all: bool = False,
    allow_narrowed: bool = False,
) -> ShapingState:
  """Rotates flaps in 3-D using the geometric folding heuristics.

  Elements whose fold would break flat-foldability (or needs no fold) are
  left out by the heuristics; the result message names them and why. The
  call fails if no element received creases.

  Args:
    state: Current state.
    rotations: Items `{element_name, el, az, depth=0.0}`: elevation in [-90,
      90], azimuth in [-180, 180] degrees, depth fraction in [0, 1) where the
      rotation starts. The angles are the absolute direction of the element in
      the stick figure's frame (x forward at az=0, y left, z up), like the stick
      figure's own angles, measured from the stick figure's root stick (see
      `ToolConfig.use_designer_root_3d`; the root is never rotated), not turns
      relative to the parent. A good baseline is the stick figure's own angles
      (`recipes.figure_pose` applies them with measured feedback), nudged where
      the model should differ from the stick figure.
    require_all: Fail unless every listed element received creases.
    allow_narrowed: Rotate elements already narrowed by `simple_fold_narrow`
      instead of refusing them. Only for limbs narrowed on the base with a small
      turn (|angle| <= 20) right before this call, as
      `recipes.figure_pose(narrow=True)` does: measured to fold (vertex error
      0%) and keep direction on moose/scorpion/crab (narrow investigation).
      Always measure the result; wide narrowing turns (45) or narrowing after an
      earlier rotation do not survive.

  Returns:
    The new state.
  """
  if not rotations:
    raise ValueError('rotations must not be empty.')
  unknown = sorted(
      r['element_name']
      for r in rotations
      if r['element_name'] not in state.tree
  )
  if unknown:
    raise ValueError(
        f'rotation_3d: unknown elements {unknown}; known:'
        f' {state.element_names()}'
    )
  _, merged_names = state.tree.merge_linear_chains()
  narrowed_with_chains = set(state.narrowed_elements)
  for sub_names in merged_names.values():
    if narrowed_with_chains.intersection(sub_names):
      narrowed_with_chains.update(sub_names)
  narrowed = sorted(
      {r['element_name'] for r in rotations} & narrowed_with_chains
  )
  if narrowed and not allow_narrowed:
    raise ValueError(
        f'rotation_3d cannot rotate narrowed elements {narrowed}: the 3-D'
        ' geometric templates assume the original flap width (results do not'
        ' fold). Rotate before narrowing, or leave narrowed elements out of'
        ' `rotations`; for knees on narrowed legs use simple_fold/crimp_fold'
        ' on the distal segment. (Limbs narrowed on the base with a small'
        ' turn can be rotated with allow_narrowed=True.)'
    )
  flaps, rivers = state.merged_shaping_elements

  def shape(
      requested: Sequence[Mapping[str, Any]], priority: Sequence[str]
  ) -> tuple[Any, set[str], list[str], list[str]]:
    updates = []
    for rotation in requested:
      update: dict[str, Any] = {
          'element_name': rotation['element_name'],
          'depth': rotation.get('depth', 0.0),
      }
      if 'az' in rotation:
        update['azimuth'] = rotation['az']
      if 'el' in rotation:
        update['elevation'] = rotation['el']
      updates.append(update)
    tree, element_names, merge_converter = (
        stick_figure_lib.split_by_frac_depth_and_update_tree(
            state.tree, updates
        )
    )
    with _library_warnings() as warnings, _designer_root_3d(state):
      get_3d = (
          geometric_fold_3d_heuristics.get_shaping_creases_with_3d_geometric_folding_heuristics
      )
      result = get_3d(
          base_cp=state.pattern,
          stick_figure=tree,
          shaping_flaps=flaps,
          shaping_rivers=rivers,
          hinge_info_list=state.updated_hinge_info(),
          central_crease_removed=state.central_crease_removed,
          shape_with_unmerged_tree=True,
          merge_converter=merge_converter,
          hparams=geometric_fold_3d_heuristics.GeometricFolding3DHParams(
              shape_subset=element_names,
              check_ff_violations=True,
              strategy=state.config.fold_strategy_3d,
              default_inscribe_hparams=geometric_fold_3d_heuristics.InscribeHParams(
                  avoid_hinges=True,
                  clip_slope_thresh=state.config.clip_slope_thresh,
                  max_slope_thresh=state.config.max_slope_thresh,
              ),
              priority=tuple(priority),
              direction_graph_data=state.base_graph_data,
          ),
          graph_data=state.graph_data,
          sacrificial_flaps=state.packing.sacrificial_flaps or [],
      )
    shaped = {
        name
        for name, infos in result['detailed_shaping_info'].items()
        if any(info['shaping_creases'] for info in infos)
    }
    return result, shaped, list(warnings), list(element_names)

  def conflicting_of(attempt) -> list[str]:
    _, shaped, messages, names = attempt
    return [
        name
        for name, reason in _skip_reasons(messages).items()
        if name in names and name not in shaped and 'flat-foldability' in reason
    ]

  best = shape(rotations, ())
  # Elements are shaped one at a time and one whose creases conflict with an
  # earlier element's is left out, so which one of a conflicting pair (often
  # a mirrored pair: triceratops femur_L or femur_R depending on the order of
  # `rotations`) survives was an accident of order. Retry with the left-out
  # elements first, then with their cut moved into the flap (depth
  # `_LEFT_OUT_RETRY_DEPTH`, away from the hinge creases it collided with:
  # moose front_arm_l/r rotate to 6 degrees that way), and keep whichever
  # attempt shapes the most elements.
  conflicting = conflicting_of(best)
  if conflicting:
    retry = shape(rotations, conflicting)
    if len(retry[1]) > len(best[1]):
      _note(
          f'Shaped {sorted(conflicting)} first: {len(retry[1])} elements'
          f' rotated instead of {len(best[1])}.'
      )
      best = retry
  conflicting = conflicting_of(best)
  deeper = [
      dict(r, depth=_LEFT_OUT_RETRY_DEPTH)
      if r['element_name'] in conflicting and not r.get('depth')
      else r
      for r in rotations
  ]
  if conflicting and deeper != list(rotations):
    retry = shape(deeper, conflicting)
    if len(retry[1]) > len(best[1]):
      _note(
          f'Cut {sorted(conflicting)} at depth {_LEFT_OUT_RETRY_DEPTH}:'
          f' {len(retry[1])} elements rotated instead of {len(best[1])}.'
      )
      best = retry
  result, shaped_names, warnings, element_names = best
  reasons = _skip_reasons(warnings)
  skipped = [
      f'{name} ({reasons.get(name, "no creases")})'
      for name in element_names
      if name not in shaped_names
  ]
  if not result['shaping_creases']:
    raise ValueError(
        'rotation_3d added no creases: ' + '; '.join(skipped or element_names)
    )
  if skipped and require_all:
    raise ValueError(
        'rotation_3d could not rotate every element (require_all=True): '
        + '; '.join(skipped)
    )
  if skipped:
    _note(f'Rotated {sorted(shaped_names)}; left out: {"; ".join(skipped)}.')
  shaped = state.pattern.update_with_shaping_creases(
      result['shaping_creases'],
      split_shaping_by_shaping=state.config.check_all_crease_intersections,
  )
  return state.with_pattern(shaped)


@_preserve_3d_overlay
def shape_angle_width(
    state: ShapingState,
    *,
    updates: Sequence[Mapping[str, Any]],
    only_listed: bool = False,
) -> ShapingState:
  """Rotates flaps in 3-D and optionally narrows them (combined heuristics).

  Flaps not listed keep their stick figure angles/widths and, unless
  `only_listed`, are shaped to them too (the whole tree in one call). Use once
  per round.

  Args:
    state: Current state.
    updates: Items `{element_name, el?, az?, narrow=False, depth=0.0}`.
    only_listed: Shape only the listed elements. On large trees shaping every
      stick at once can give a flat-foldable pattern the deterministic folder
      cannot fold (strip fallback: 32-stick scorpion, body river el=-35).

  Returns:
    The new state.
  """
  if not updates:
    raise ValueError('updates must not be empty.')
  unknown = sorted(
      u['element_name'] for u in updates if u['element_name'] not in state.tree
  )
  if unknown:
    raise ValueError(
        f'shape_angle_width: unknown elements {unknown}; known:'
        f' {state.element_names()}'
    )
  requested = {u['element_name'] for u in updates}
  tree_updates = []
  for item in updates:
    stick = state.tree[item['element_name']]
    azimuth = item.get('az')
    elevation = item.get('el')
    tree_updates.append({
        'element_name': item['element_name'],
        'depth': item.get('depth', 0.0),
        'azimuth': stick.azimuth_angle if azimuth is None else azimuth,
        'elevation': stick.elevation_angle if elevation is None else elevation,
        'width': 0.5 if item.get('narrow', False) else max(1.0, stick.width),
    })
  # `combined_heuristics._compute_narrowed_flaps` treats every stick with
  # `width < 1.0` as a narrowing target (and asserts its first simple fold has
  # a 180-degree rotation). Clamp unlisted thin sticks to width 1.0 so only
  # sticks explicitly requested with `narrow=True` are narrowed.
  for stick in state.tree.children:
    if stick.name not in requested and stick.width < 1.0:
      tree_updates.append({
          'element_name': stick.name,
          'depth': 0.0,
          'azimuth': stick.azimuth_angle,
          'elevation': stick.elevation_angle,
          'width': 1.0,
      })
  tree, element_names, merge_converter = (
      stick_figure_lib.split_by_frac_depth_and_update_tree(
          state.tree, tree_updates
      )
  )
  # The first names belong to `updates` (same order); the rest are the clamped
  # unlisted sticks.
  shape_subset = element_names[: len(updates)] if only_listed else None
  flaps, rivers = state.merged_shaping_elements
  with _designer_root_3d(state):
    shaped = combined_heuristics.shape_cp_with_combined_heuristics(
        base_cp=state.pattern,
        stick_figure=tree,
        packing_output=state.packing,
        solving_output=state.solving,
        solving_grid_size=state.solving_grid_size,
        central_crease_removed=state.central_crease_removed,
        merge_converter=merge_converter,
        hparams=combined_heuristics.CombinedHeuristicsHParams(
            geometric_fold_hparams=geometric_fold_3d_heuristics.GeometricFolding3DHParams(
                default_inscribe_hparams=geometric_fold_3d_heuristics.InscribeHParams(
                    avoid_hinges=True
                ),
                check_ff_violations=True,
                strategy=heuristics_3d.FoldStrategy3D.SHALLOW_FOLDS,
                shape_subset=shape_subset,
            ),
        ),
        shaping_flaps=flaps,
        shaping_rivers=rivers,
        graph_data=state.base_graph_data,
    )
  return state.with_pattern(shaped)


# ------------------------------------------------------------------------------
# Thin limbs posed in 3-D (narrow + out-of-plane bend + knee, searched).
# ------------------------------------------------------------------------------

# Candidate narrowing turns (degrees) for the proximal segment.
_THIN_NARROW_ANGLES = (
    45.0,
    -45.0,
    90.0,
    -90.0,
    135.0,
    -135.0,
)
# Zig-zag (angle, fold order) pairs tried for the partial (3-D) bends; with
# fold_percent < 1 the fold stays open, so the narrowed segment leaves the
# paper plane. A 180-degree zig-zag bends about the segment's transverse axis
# (the fold order picks the side); +-90 also turn it (heuristic order).
_THIN_BEND_ANGLES = ((180.0, 'MV'), (180.0, 'VM'), (90.0, None), (-90.0, None))
_THIN_FOLD_PERCENTS = (0.35, 0.65)
# Flat second turns (fold_percent 1) tried in the bend stage, for limbs whose
# in-plane direction is beyond the narrowing turn's reach.
_THIN_FLAT_TURNS = (60.0, -60.0, 120.0, -120.0)
_THIN_BEND_DEPTH = 0.5
_THIN_KNEE_DEPTH = 0.25
# A stage keeps a candidate only if it lowers the limb's error by this much.
_THIN_MIN_GAIN_DEG = 2.0

_FOLD_EVAL_CACHE: dict[str, Any] = {}


def _fold_for_evaluation(pattern: crease_pattern_lib.CreasePattern) -> Any:
  """Folds like `lib.fold_and_render`; returns a `fidelity.FoldedModel`."""
  svg_str = svg_parser.to_svg(pattern, ndigits=8)
  cached = _FOLD_EVAL_CACHE.get(svg_str)
  if cached is not None:
    return cached
  start = time.perf_counter()
  folded = folder.get_fold(
      svg_str,
      vert_tol=1e-3,
      simulation_tolerance=10.0,
  )
  result = fidelity.FoldedModel.from_fold3d(folded, time.perf_counter() - start)
  _FOLD_EVAL_CACHE[svg_str] = result
  return result


def _descendant_sticks(
    tree: stick_figure_lib.StickFigure, name: str
) -> list[str]:
  """`name` and every stick beyond it (away from the root)."""
  topo = fidelity.topology(tree)
  children: dict[str, list[str]] = {}
  for stick_name, parent in topo.parent.items():
    if parent is not None:
      children.setdefault(parent, []).append(stick_name)
  result, queue = [], [name]
  while queue:
    current = queue.pop(0)
    result.append(current)
    queue.extend(children.get(current, []))
  return result


@dataclasses.dataclass(frozen=True)
class LimbFrame:
  """Alignment frame for limb directions, fixed while limbs change.

  Attributes:
    anchors: Folded positions (reference model) of the joints outside the limbs;
      they do not move relative to each other when only limbs are shaped, so
      every candidate is registered onto them rigidly.
    fit: The `fidelity` robust similarity of the reference model (all joints)
      onto the stick figure.
    joints: All folded joint geometries of the reference model.
  """

  anchors: dict[str, np.ndarray]
  fit: Any
  joints: dict[str, Any] | None = None


def _limb_joints(
    state: ShapingState, limbs: Sequence[tuple[str, str | None]]
) -> set[str]:
  """Joints that can move when the limbs are shaped.

  The distal joints of the limbs' sticks, and the limbs' root joints (their
  paper points include the limb flaps' hinges, so their folded mean moves).

  Args:
    state: Current state.
    limbs: `(proximal, distal)` stick names.
  """
  topo = fidelity.topology(state.tree)
  joints = set()
  for proximal, _ in limbs:
    joints.add(topo.proximal[proximal])
    for name in _descendant_sticks(state.tree, proximal):
      joints.add(topo.distal[name])
  return joints


def _limb_geometry(
    state: ShapingState,
    target_tree: stick_figure_lib.StickFigure | None = None,
    cached_context: tuple[Any, Any, float] | None = None,
) -> dict[str, Any]:
  """Folds the pattern and locates only the stick-figure joints."""
  try:
    model = _fold_for_evaluation(state.pattern)
  except Exception as error:  # pylint: disable=broad-exception-caught
    raise ValueError(f'the folder failed: {_describe_error(error)}') from error
  if cached_context is not None:
    layout_joints, target, to_tree_units = cached_context
  else:
    layout = state.paper_layout
    tree = target_tree or state.tree
    target = fidelity.stick_figure_joint_positions(tree)
    to_tree_units = layout.grid_size / layout.grid_units_per_tree_unit
    layout_joints = layout.joints
  folded, _ = model.locate_groups(layout_joints)
  return {
      'joints': {
          name: {
              'folded': (
                  (folded[name] * to_tree_units).tolist()
                  if name in folded
                  else None
              ),
              'target': (
                  target[name].tolist()
                  if isinstance(target[name], np.ndarray)
                  else target[name]
              ),
          }
          for name in target
      }
  }


def _robust_fit(joints: Mapping[str, Mapping[str, Any]]) -> Any:
  """`fidelity`'s robust similarity of the located joints onto the figure."""
  located = sorted(n for n, j in joints.items() if j['folded'] is not None)
  if len(located) < 3:
    raise ValueError(f'only {len(located)} joints located on the paper.')
  source = np.array([joints[n]['folded'] for n in located])
  goal = np.array([joints[n]['target'] for n in located])
  size = math.sqrt(float(((goal - goal.mean(axis=0)) ** 2).sum(axis=1).mean()))
  fit, _ = fidelity.fit_similarity_robust(source, goal, size)
  return fit


def limb_frame(
    state: ShapingState,
    limbs: Sequence[tuple[str, str | None]],
    target_tree: stick_figure_lib.StickFigure | None = None,
) -> LimbFrame:
  """The alignment frame of `state` for `limb_direction_errors`.

  Args:
    state: Reference state, ideally with every part posed roughly right (e.g.
      the limbs rotated with `rotation_3d`), so the whole-model fit is good.
    limbs: `(proximal, distal)` stick names.
    target_tree: Target stick figure (defaults to `state.tree`).

  Returns:
    The frame.
  """
  joints = _limb_geometry(state, target_tree)['joints']
  moved = _limb_joints(state, limbs)
  anchors = {
      n: np.asarray(j['folded'])
      for n, j in joints.items()
      if j['folded'] is not None and n not in moved
  }
  return LimbFrame(anchors=anchors, fit=_robust_fit(joints), joints=joints)


def limb_direction_errors(
    state: ShapingState,
    limbs: Sequence[tuple[str, str | None]],
    frame: LimbFrame | None = None,
    target_tree: stick_figure_lib.StickFigure | None = None,
    cached_context: tuple[Any, Any, float] | None = None,
) -> dict[str, float]:
  """Folds the pattern and measures limb directions against the stick figure.

  With a `frame`, the folded model is first registered rigidly onto the
  frame's anchors (joints outside the limbs) and then mapped with the frame's
  fit, so each limb's error does not depend on how the other limbs are posed.
  Without one (or when fewer than 3 non-collinear anchors are located), the
  `fidelity` robust fit of all joints is used, as in `--stage=score`.

  Args:
    state: Current state.
    limbs: `(proximal, distal)` stick names (`distal` may be None).
    frame: Optional fixed frame from `limb_frame`.
    target_tree: Stick figure with the target directions (defaults to
      `state.tree`; `thin_limbs_3d` passes one with per-limb az/el overrides).
    cached_context: Optional precomputed (layout_joints, target_positions,
      to_tree_units) tuple to avoid repeated paper_layout / tree traversals.

  Returns:
    `{stick: angle in degrees}` for every proximal and distal stick; 90 for a
    segment that collapsed onto itself, 180 for one that is not located.

  Raises:
    ValueError: If the folder fails or too few joints are located.
  """
  joints = _limb_geometry(state, target_tree, cached_context=cached_context)[
      'joints'
  ]
  topo = fidelity.topology(state.tree)
  transform = None
  if frame is not None:
    common = sorted(
        n for n in frame.anchors if joints.get(n, {}).get('folded') is not None
    )
    if len(common) >= 3:
      source = np.array([joints[n]['folded'] for n in common])
      singular = np.linalg.svd(source - source.mean(axis=0), compute_uv=False)
      if singular[1] > 1e-3 * max(singular[0], 1e-12):
        registration = fidelity.fit_similarity(
            source,
            np.array([frame.anchors[n] for n in common]),
            allow_reflection=False,
        )
        transform = (
            frame.fit.scale
            * frame.fit.rotation
            @ (registration.scale * registration.rotation)
        )
  if transform is None:
    fit = _robust_fit(joints)
    transform = fit.scale * fit.rotation
  targets = np.array([j['target'] for j in joints.values()])
  # `fidelity`'s collapse threshold: 2 % of the figure's RMS radius.
  min_length = 0.02 * math.sqrt(
      float(((targets - targets.mean(axis=0)) ** 2).sum(axis=1).mean())
  )
  errors = {}
  for proximal, distal in limbs:
    for name in (proximal, distal):
      if name is None:
        continue
      near, far = topo.proximal[name], topo.distal[name]
      if joints[near]['folded'] is None or joints[far]['folded'] is None:
        errors[name] = 180.0
        continue
      folded = transform @ (
          np.asarray(joints[far]['folded']) - np.asarray(joints[near]['folded'])
      )
      target = np.asarray(joints[far]['target']) - np.asarray(
          joints[near]['target']
      )
      if np.linalg.norm(folded) < min_length:
        errors[name] = 90.0
        continue
      cosine = float(
          folded @ target / (np.linalg.norm(folded) * np.linalg.norm(target))
      )
      errors[name] = round(
          math.degrees(math.acos(min(1.0, max(-1.0, cosine)))), 1
      )
  return errors


def _limb_distal(state: ShapingState, proximal: str) -> str | None:
  """The next stick of `proximal`'s merged chain (its knee segment), if any."""
  _, merged_names = state.tree.merge_linear_chains()
  for sub_names in merged_names.values():
    if proximal in sub_names:
      index = list(sub_names).index(proximal)
      if index + 1 < len(sub_names):
        return sub_names[index + 1]
  return None


def _thin_step_arguments(
    stage: str, element_name: str, value: Any
) -> dict[str, Any] | None:
  """`simple_fold` arguments for one plan step (None: step skipped)."""
  if value is None:
    return None
  if stage == 'narrow_angle':
    return {'element_name': element_name, 'angle': float(value), 'narrow': True}
  arguments = {
      'element_name': element_name,
      'angle': float(value['angle']),
      'depth': float(value['depth']),
      'fold_percent': float(value['fold_percent']),
  }
  if value.get('fold_order'):
    arguments['fold_order'] = value['fold_order']
  return arguments


def _prepare_batch_simple_fold(
    state: ShapingState, arguments: Mapping[str, Any]
) -> tuple[str, int, bool, list[zigzag_shaping.ZigzagShapingParamsForElement]]:
  """Prepares zig-zag params for one element in a batched simple-fold step."""
  element_name = arguments['element_name']
  angle = float(arguments['angle'])
  depth = float(arguments.get('depth', 0.0))
  narrow = bool(arguments.get('narrow', False))
  fold_percent = float(arguments.get('fold_percent', 1.0))
  fold_order = arguments.get('fold_order')
  orders = {
      'MV': zigzag_shaping_utils.MOUNTAIN_TO_VALLEY,
      'VM': zigzag_shaping_utils.VALLEY_TO_MOUNTAIN,
  }
  local_state = state
  if fold_order is not None:
    if fold_order not in orders:
      raise ValueError(f'fold_order must be MV or VM, got {fold_order!r}.')
    local_state = state.replace(
        config=dataclasses.replace(
            state.config, use_heuristics_for_fold_order=False
        )
    )
  element, params = _zigzag_params(
      local_state,
      element_name=element_name,
      angle=angle,
      depth=depth,
      fold_percent=fold_percent,
  )
  if fold_order is not None:
    params = dataclasses.replace(
        params,
        zigzag_params=dataclasses.replace(
            params.zigzag_params, fold_order=orders[fold_order]
        ),
    )
  if isinstance(element, ShapingRiver) and narrow:
    raise ValueError('Rivers cannot be narrowed; use narrow=False.')
  if local_state.config.apply_simple_fold_heuristics:
    element_params = _apply_simple_fold_heuristics(
        local_state,
        element,
        params,
        narrow=narrow,
        convert_small_angles=(
            local_state.config.convert_small_angles and not narrow
        ),
    )
  else:
    element_params = [params]
  flaps, rivers = state.shaping_elements
  preview = simple_fold_shaping.get_simple_fold_shaping_creases(
      grid_size=element.grid_size,
      flaps=flaps,
      rivers=rivers,
      element_shaping_params=element_params,
      central_crease_removed=state.central_crease_removed,
  )
  if not preview:
    raise ValueError(
        f'No shaping creases found for simple fold of {element_name}.'
    )
  return element_name, element.grid_size, narrow, element_params


def _apply_thin_steps(
    state: ShapingState,
    steps: Sequence[tuple[int, dict[str, Any]]],
) -> tuple[ShapingState, set[int]]:
  """Applies `simple_fold` steps; returns the state and the limbs that took."""
  if len(steps) > 1:
    prepared = []
    grid_and_narrow = set()
    for limb_index, arguments in steps:
      try:
        name, grid_size, narrow, ep = _prepare_batch_simple_fold(
            state, arguments
        )
        prepared.append((limb_index, name, ep))
        grid_and_narrow.add((grid_size, narrow))
      except (ValueError, AssertionError, KeyError, IndexError):
        continue
    if len(prepared) > 1 and len(grid_and_narrow) == 1:
      common_grid_size, common_narrow = next(iter(grid_and_narrow))
      flaps, rivers = state.shaping_elements
      combined_params = [p for _, _, ep in prepared for p in ep]
      try:
        shaped = simple_fold_shaping.shape_cp_with_simple_folds(
            cp=state.pattern,
            element_shaping_params=combined_params,
            grid_size=common_grid_size,
            flaps=flaps,
            rivers=rivers,
            central_crease_removed=state.central_crease_removed,
            narrow=common_narrow,
            light_mode=not state.config.check_all_crease_intersections,
        )
        new_state = state.with_pattern(shaped)
        if common_narrow:
          new_state = new_state.replace(
              narrowed_elements=new_state.narrowed_elements
              + tuple(name for _, name, _ in prepared),
          )
        return new_state, {idx for idx, _, _ in prepared}
      except (ValueError, AssertionError, KeyError, IndexError):
        pass
  applied = set()
  for limb_index, arguments in steps:
    try:
      state = simple_fold(state, **arguments)
    except (ValueError, AssertionError, KeyError, IndexError) as error:
      logging.info(
          'thin_limbs_3d candidate %s failed: %s',
          arguments,
          _describe_error(error),
      )
      continue
    applied.add(limb_index)
  return state, applied


def _thin_limbs_frame(
    state: ShapingState,
    pairs: Sequence[tuple[str, str | None]],
    target_tree: stick_figure_lib.StickFigure,
) -> LimbFrame | None:
  """Alignment frame for the limb search."""
  try:
    return limb_frame(state, pairs, target_tree)
  except (ValueError, AssertionError):
    pass

  rotations = [
      {
          'element_name': name,
          'el': target_tree[name].elevation_angle,
          'az': target_tree[name].azimuth_angle,
      }
      for pair in pairs
      for name in pair
      if name is not None
  ]
  token = _NOTES.set([])  # The preview's notes are not this tool's.
  try:
    return limb_frame(
        rotation_3d(state, rotations=rotations), pairs, target_tree
    )
  except (ValueError, AssertionError) as error:
    logging.info('thin_limbs_3d: no frame from reference: %s', error)
  finally:
    _NOTES.reset(token)
  return None


@_preserve_3d_overlay
def thin_limbs_3d(
    state: ShapingState,
    *,
    limbs: Sequence[Mapping[str, Any]],
    evaluate: Any = None,
) -> ShapingState:
  """Narrows limbs and poses them in 3-D (thin legs that leave the plane).

  Each limb is a proximal segment (e.g. `leg_L1_femur`) with an optional
  distal segment (`leg_L1_tarsus`; defaults to the next stick of the merged
  chain). Three `simple_fold` steps build it, each searched per limb unless
  given:

  1. `narrow_angle`: `simple_fold(proximal, angle, narrow=True)`: halves the
     limb's width and turns it in the paper plane.
  2. `bend`: `simple_fold(proximal, angle, depth, fold_percent < 1)` on the
     narrowed segment: an open fold that lifts it out of the paper plane.
  3. `knee`: the same on the distal segment (skipped for one-segment limbs).

  Candidates are scored by folding the pattern with the deterministic folder
  and measuring each segment's direction against the stick figure
  (`limb_direction_errors`, in a frame fitted on a preview with the limbs
  rotated by `rotation_3d` and registered on the joints outside the limbs);
  all limbs are tried at once per candidate, so a call costs about 30 folds.
  Pose the body, tail and other chains first (e.g. `rotation_3d`), and do not
  rotate or narrow the limbs themselves before. The chosen plan is reported in
  the result message and recorded in the history in place of the call's
  arguments, so a replay applies the same folds without searching.

  Args:
    state: Current state.
    limbs: Items `{proximal, distal?, narrow_angle?, bend?, knee?, az?, el?,
      distal_az?, distal_el?}`; `az`/`el` (degrees, stick-figure convention)
      replace the stick figure's target direction of the proximal/distal segment
      (e.g. to fan legs out more than the stick figure); `bend` and `knee` are
      `{angle, depth, fold_percent, fold_order?}` (`simple_fold` arguments) or
      None (no fold). Given steps are applied as is; missing ones are searched.
    evaluate: Optional `(state, [(proximal, distal), ...]) -> {stick: degrees}`
      replacing `limb_direction_errors` in the frame of `_thin_limbs_frame`
      (tests).

  Returns:
    The new state.

  Raises:
    ValueError: On unknown elements, or if no limb could be narrowed.
  """
  if not limbs:
    raise ValueError('limbs must not be empty.')
  names = set(state.element_names())
  plans: list[dict[str, Any]] = []
  for item in limbs:
    proximal = item['proximal']
    distal = item.get('distal', _limb_distal(state, proximal))
    unknown = [
        n for n in (proximal, distal) if n is not None and n not in names
    ]
    if unknown:
      raise ValueError(
          f'thin_limbs_3d: unknown elements {unknown}; known:'
          f' {state.element_names()}'
      )
    if isinstance(state.element(proximal), ShapingRiver):
      raise ValueError(f'{proximal} is a river; only flaps can be narrowed.')
    plan = {'proximal': proximal, 'distal': distal}
    plan.update({
        k: item[k]
        for k in (
            'narrow_angle',
            'bend',
            'knee',
            'az',
            'el',
            'distal_az',
            'distal_el',
        )
        if k in item
    })
    if distal is None:
      plan['knee'] = None
    plans.append(plan)
  pairs = [(plan['proximal'], plan['distal']) for plan in plans]
  azimuths, elevations = {}, {}
  for plan in plans:
    for prefix, name in (('', plan['proximal']), ('distal_', plan['distal'])):
      if name is not None and f'{prefix}az' in plan:
        azimuths[name] = float(plan[f'{prefix}az'])
      if name is not None and f'{prefix}el' in plan:
        elevations[name] = float(plan[f'{prefix}el'])
  target_tree = (
      state.tree.update_tree(azimuths, elevations, {})
      if azimuths or elevations
      else state.tree
  )
  layout = state.paper_layout
  target_positions = fidelity.stick_figure_joint_positions(target_tree)
  to_tree_units = layout.grid_size / layout.grid_units_per_tree_unit
  cached_context = (layout.joints, target_positions, to_tree_units)
  frame: LimbFrame | None = None

  is_custom_eval = evaluate is not None
  if evaluate is None:
    frame = _thin_limbs_frame(state, pairs, target_tree)
    evaluate = functools.partial(
        limb_direction_errors,
        frame=frame,
        target_tree=target_tree,
        cached_context=cached_context,
    )

  # Identify symmetric limb pairs in plans
  symmetric_pairs: list[tuple[int, int]] = []
  paired_indices: set[int] = set()
  for i, plan_i in enumerate(plans):
    if i in paired_indices:
      continue
    sym_name = resolve_symmetric_element(state, plan_i['proximal'])
    if sym_name:
      for j, plan_j in enumerate(plans):
        if (
            j != i
            and j not in paired_indices
            and plan_j['proximal'] == sym_name
        ):
          symmetric_pairs.append((i, j))
          paired_indices.add(i)
          paired_indices.add(j)
          break

  def element_for(stage: str, index: int) -> str:
    return pairs[index][1 if stage == 'knee' else 0]

  def limb_error(errors: Mapping[str, float], index: int, stage: str) -> float:
    proximal, distal = pairs[index]
    if stage == 'knee':
      return 0.5 * (errors.get(proximal, 180.0) + errors.get(distal, 180.0))
    return errors.get(proximal, 180.0)

  def apply_stage(
      current: ShapingState, stage: str, values: Mapping[int, Any]
  ) -> tuple[ShapingState, set[int]]:
    steps = []
    for i, value in values.items():
      arguments = _thin_step_arguments(stage, element_for(stage, i), value)
      if arguments is not None:
        steps.append((i, arguments))
    return _apply_thin_steps(current, steps)

  eval_cache: dict[tuple[Any, ...], Mapping[str, float]] = {}

  def _pattern_eval_key(
      pattern: crease_pattern_lib.CreasePattern,
  ) -> tuple[Any, ...]:
    return tuple(
        (
            round(c.p1_init.x, 7),
            round(c.p1_init.y, 7),
            round(c.p2_init.x, 7),
            round(c.p2_init.y, 7),
            c.fold.value,
            round(c.fold_percent, 4),
        )
        for c in pattern.creases
    )

  def _cached_evaluate(candidate_state: ShapingState) -> Mapping[str, float]:
    key = _pattern_eval_key(candidate_state.pattern)
    if key not in eval_cache:
      eval_cache[key] = evaluate(candidate_state, pairs)
    return eval_cache[key]

  def search(
      current: ShapingState,
      stage: str,
      candidates: Mapping[int, Sequence[Any]],
      best: dict[int, tuple[float, Any]],
      target_todo: Sequence[int] = (),
  ) -> None:
    """Updates `best[i] = (error, value)` with each limb's candidates."""
    logging.info(
        'THIN STAMP stage %s search called with %d cands at %.3f',
        stage,
        len(candidates),
        time.perf_counter(),
    )
    rounds = max((len(v) for v in candidates.values()), default=0)
    for k in range(rounds):
      tried = {i: v[k] for i, v in candidates.items() if k < len(v)}
      candidate_state, applied = apply_stage(current, stage, tried)
      if not applied:
        continue
      if candidate_state.config.check_flat_foldability:
        try:
          check_flat_foldability(candidate_state.pattern)
        except Exception:  # pylint: disable=broad-exception-caught
          continue
      try:
        errors = _cached_evaluate(candidate_state)
      except ValueError as error:
        logging.info('thin_limbs_3d: candidate %s not scored: %s', tried, error)
        continue
      for i in applied:
        error = limb_error(errors, i, stage)
        if error < best[i][0]:
          best[i] = (error, tried[i])
      if target_todo and all(best[i][0] < 20.0 for i in target_todo):
        break

  def bend_directions(depth: float) -> list[dict[str, Any]]:
    candidates = []
    for angle, order in _THIN_BEND_ANGLES:
      candidate: dict[str, Any] = {
          'angle': angle,
          'depth': depth,
          'fold_percent': 0.5,
      }
      if order:
        candidate['fold_order'] = order
      candidates.append(candidate)
    return candidates

  pred_angle_fn = None
  _pred_cand_error = None
  if not is_custom_eval:
    if frame is not None and frame.joints is not None:
      init_geom = frame.joints
    else:
      init_geom = _limb_geometry(
          state, target_tree, cached_context=cached_context
      )['joints']
    topo = fidelity.topology(state.tree)
    if frame is not None:
      common = sorted(
          n
          for n in frame.anchors
          if init_geom.get(n, {}).get('folded') is not None
      )
    else:
      common = []
    if frame is not None and len(common) >= 3:
      source = np.array([init_geom[n]['folded'] for n in common])
      reg = fidelity.fit_similarity(
          source,
          np.array([frame.anchors[n] for n in common]),
          allow_reflection=False,
      )
      transform = (
          frame.fit.scale * frame.fit.rotation @ (reg.scale * reg.rotation)
      )
    else:
      fit = _robust_fit(init_geom)
      transform = fit.scale * fit.rotation

    n_vec = transform @ np.array([0.0, 0.0, 1.0])
    n_norm = np.linalg.norm(n_vec)
    n_hat = n_vec / n_norm if n_norm > 1e-6 else np.array([0.0, 0.0, 1.0])

    def _rodrigues(v: np.ndarray, axis: np.ndarray, deg: float) -> np.ndarray:
      r = math.radians(deg)
      return (
          v * math.cos(r)
          + np.cross(axis, v) * math.sin(r)
          + axis * np.dot(axis, v) * (1.0 - math.cos(r))
      )

    def _pred_angle_error(idx: int, ang_deg: float, elem_idx: int = 0) -> float:
      elem = pairs[idx][elem_idx]
      if elem is None:
        return 180.0
      near_j, far_j = topo.proximal[elem], topo.distal[elem]
      if (
          init_geom[near_j]['folded'] is None
          or init_geom[far_j]['folded'] is None
      ):
        return 180.0
      v_0 = transform @ (
          np.asarray(init_geom[far_j]['folded'])
          - np.asarray(init_geom[near_j]['folded'])
      )
      t_vec = np.asarray(init_geom[far_j]['target']) - np.asarray(
          init_geom[near_j]['target']
      )
      norm_v = np.linalg.norm(v_0)
      norm_t = np.linalg.norm(t_vec)
      if norm_v < 1e-6 or norm_t < 1e-6:
        return 90.0
      u_0 = v_0 / norm_v
      u_t = t_vec / norm_t
      u_rot = _rodrigues(u_0, n_hat, ang_deg)
      u_rot /= max(np.linalg.norm(u_rot), 1e-6)
      cos_val = float(np.dot(u_rot, u_t))
      return round(math.degrees(math.acos(min(1.0, max(-1.0, cos_val)))), 1)

    def _pred_cand_error(stg: str, idx: int, cand: Any) -> float:
      if stg == 'narrow_angle':
        return _pred_angle_error(idx, float(cand), 0)
      elem_idx = 1 if stg == 'knee' else 0
      elem = pairs[idx][elem_idx]
      if elem is None:
        return 180.0
      near_j, far_j = topo.proximal[elem], topo.distal[elem]
      if (
          init_geom[near_j]['folded'] is None
          or init_geom[far_j]['folded'] is None
      ):
        return 180.0
      v_0 = transform @ (
          np.asarray(init_geom[far_j]['folded'])
          - np.asarray(init_geom[near_j]['folded'])
      )
      t_vec = np.asarray(init_geom[far_j]['target']) - np.asarray(
          init_geom[near_j]['target']
      )
      norm_v = np.linalg.norm(v_0)
      norm_t = np.linalg.norm(t_vec)
      if norm_v < 1e-6 or norm_t < 1e-6:
        return 90.0
      u_0 = v_0 / norm_v
      u_t = t_vec / norm_t
      narrow_ang = float(plans[idx].get('narrow_angle') or 0.0)
      u_cur = _rodrigues(u_0, n_hat, narrow_ang)
      u_cur /= max(np.linalg.norm(u_cur), 1e-6)

      if cand.get('fold_percent', 0.5) == 1.0:
        # Flat turn
        u_next = _rodrigues(u_cur, n_hat, float(cand['angle']))
        cos_val = float(np.dot(u_next, u_t))
        return round(math.degrees(math.acos(min(1.0, max(-1.0, cos_val)))), 1)
      else:
        # 3-D transverse bend: only beneficial if target has out-of-plane elevation
        z_target = abs(float(np.dot(u_t, n_hat)))
        if z_target < 0.2:
          return 90.0
        t_hat = np.cross(n_hat, u_cur)
        norm_t_hat = np.linalg.norm(t_hat)
        if norm_t_hat > 1e-6:
          t_hat /= norm_t_hat
        ang = float(cand['angle']) * float(cand.get('fold_percent', 0.5))
        u_b1 = _rodrigues(u_cur, t_hat, ang)
        u_b2 = _rodrigues(u_cur, -t_hat, ang)
        c1 = float(np.dot(u_b1, u_t))
        c2 = float(np.dot(u_b2, u_t))
        e1 = math.degrees(math.acos(min(1.0, max(-1.0, c1))))
        e2 = math.degrees(math.acos(min(1.0, max(-1.0, c2))))
        return round(min(e1, e2), 1)

    pred_angle_fn = _pred_angle_error

  current = state
  flat_turns = [
      {'angle': angle, 'depth': _THIN_BEND_DEPTH, 'fold_percent': 1.0}
      for angle in _THIN_FLAT_TURNS
  ]
  stages = (
      ('narrow_angle', (45.0, 90.0, -45.0, -90.0), 22.5),
      ('bend', bend_directions(_THIN_BEND_DEPTH), 0.0),
      ('knee', bend_directions(_THIN_KNEE_DEPTH), 0.0),
  )
  for stage, coarse, step in stages:
    logging.info(
        'THIN STAMP entering stage %s at %.3f', stage, time.perf_counter()
    )
    todo = [i for i, plan in enumerate(plans) if stage not in plan]
    cur_errs = {}
    if stage in ('bend', 'knee') and todo:
      try:
        cur_errs = _cached_evaluate(current)
      except Exception:  # pylint: disable=broad-exception-caught
        cur_errs = {}
    if stage == 'bend' and todo and not is_custom_eval:
      # Skip bend search if limb error after narrow is already < 15.0 degrees
      if cur_errs:
        skipped = [
            i for i in todo if limb_error(cur_errs, i, 'narrow_angle') < 15.0
        ]
        for i in skipped:
          plans[i]['bend'] = None
        todo = [
            i for i in todo if limb_error(cur_errs, i, 'narrow_angle') >= 15.0
        ]
    if stage == 'knee' and todo and not is_custom_eval:
      # Skip knee search if limb error after bend is already < 15.0 degrees
      if cur_errs:
        skipped = [i for i in todo if limb_error(cur_errs, i, 'bend') < 15.0]
        for i in skipped:
          plans[i]['knee'] = None
        todo = [i for i in todo if limb_error(cur_errs, i, 'bend') >= 15.0]
    best: dict[int, tuple[float, Any]] = {i: (math.inf, None) for i in todo}
    if todo and stage != 'narrow_angle':
      # Keeping the segment as it is must be beaten by a real gain.
      if cur_errs:
        for i in todo:
          best[i] = (limb_error(cur_errs, i, stage) - _THIN_MIN_GAIN_DEG, None)
    if todo:
      if stage == 'narrow_angle' and pred_angle_fn is not None:
        # Evaluate 2-D angle candidates on the CP without 3-D folding per
        # candidate using Rodrigues rotation around paper normal.
        for i in todo:
          cand_pool = [45.0, 90.0, -45.0, -90.0]
          cand_pool.sort(key=lambda a: pred_angle_fn(i, a))
          best_coarse = cand_pool[0]
          if pred_angle_fn(i, best_coarse) >= 20.0:
            finer = [best_coarse - step, best_coarse + step]
            finer.sort(key=lambda a: pred_angle_fn(i, a))
            all_ranked = [finer[0], best_coarse, finer[1]] + cand_pool[1:]
          else:
            all_ranked = cand_pool

          best_ang = None
          best_err = math.inf
          for cand_ang in all_ranked:
            try:
              _, applied_i = apply_stage(current, 'narrow_angle', {i: cand_ang})
              if i in applied_i:
                best_ang = cand_ang
                best_err = pred_angle_fn(i, cand_ang)
                break
            except Exception:  # pylint: disable=broad-exception-caught
              continue

          if best_ang is None:
            best_ang = all_ranked[0]
            best_err = pred_angle_fn(i, best_ang)
          best[i] = (best_err, best_ang)

        for i, j in symmetric_pairs:
          if i in todo and j in todo:
            ang_i = best[i][1]
            ang_j = best[j][1]
            if ang_i is not None and ang_j is not None:
              j_valid = False
              try:
                _, app_j = apply_stage(current, 'narrow_angle', {j: -ang_i})
                j_valid = j in app_j
              except Exception:  # pylint: disable=broad-exception-caught
                pass
              i_valid = False
              try:
                _, app_i = apply_stage(current, 'narrow_angle', {i: -ang_j})
                i_valid = i in app_i
              except Exception:  # pylint: disable=broad-exception-caught
                pass
              if j_valid and (
                  not i_valid
                  or best[i][0] + pred_angle_fn(j, -ang_i)
                  <= pred_angle_fn(i, -ang_j) + best[j][0]
              ):
                best[j] = (pred_angle_fn(j, -ang_i), -ang_i)
              elif i_valid:
                best[i] = (pred_angle_fn(i, -ang_j), -ang_j)
      else:
        coarse_candidates: dict[int, list[Any]] = {}
        if stage == 'bend':
          needs_flat = any(
              limb_error(cur_errs, i, 'narrow_angle') >= 35.0 for i in todo
          )
          b_cands = bend_directions(_THIN_BEND_DEPTH) + (
              flat_turns if needs_flat else []
          )
          for i, j in symmetric_pairs:
            if i in todo and j in todo:
              paired_b = []
              for c in b_cands:
                c_opp = dict(c)
                if 'angle' in c:
                  c_opp['angle'] = -c['angle']
                if c.get('fold_order') == 'MV':
                  c_opp['fold_order'] = 'VM'
                elif c.get('fold_order') == 'VM':
                  c_opp['fold_order'] = 'MV'
                paired_b.append(c_opp)
              coarse_candidates[i] = b_cands
              coarse_candidates[j] = paired_b
          for i in todo:
            if i not in coarse_candidates:
              coarse_candidates[i] = b_cands
        else:
          for i in todo:
            coarse_candidates[i] = list(coarse)

        # Prune candidates whose 2-D prediction cannot beat cur_err - _THIN_MIN_GAIN_DEG.
        if not is_custom_eval and _pred_cand_error is not None and cur_errs:
          for i in todo:
            cur_e = limb_error(cur_errs, i, stage)
            threshold = cur_e - _THIN_MIN_GAIN_DEG
            cands = coarse_candidates.get(i, [])
            filtered = [
                c for c in cands if _pred_cand_error(stage, i, c) < threshold
            ]
            if filtered:
              filtered.sort(key=lambda c: _pred_cand_error(stage, i, c))
              coarse_candidates[i] = [filtered[0]]
            else:
              coarse_candidates[i] = []

        if any(coarse_candidates.get(i) for i in todo):
          search(current, stage, coarse_candidates, best, target_todo=todo)
        finer = {}
        unresolved = (
            [i for i in todo if best[i][0] >= 20.0 and best[i][1] is not None]
            if not is_custom_eval
            else list(todo)
        )
        if unresolved:
          if stage == 'narrow_angle':
            finer = {
                i: [best[i][1] - step, best[i][1] + step]
                for i in unresolved
                if best[i][1] is not None
            }
          else:
            for i in unresolved:
              if i not in finer:
                val = best[i][1]
                if val is None:
                  continue
                if isinstance(val, dict) and val.get('fold_percent', 1.0) < 1.0:
                  finer[i] = [
                      dict(val, fold_percent=percent)
                      for percent in _THIN_FOLD_PERCENTS
                  ]
          if finer:
            search(current, stage, finer, best, target_todo=unresolved)
    for i in todo:
      plans[i][stage] = best[i][1]
    current, applied = apply_stage(
        current, stage, {i: plan[stage] for i, plan in enumerate(plans)}
    )
    if stage == 'narrow_angle':
      if not applied:
        raise ValueError(
            'thin_limbs_3d could not narrow any limb (check the proximal'
            ' segments with --stage=probe; narrowing needs flaps on the paper'
            ' border).'
        )
      # Fold once per accepted plan at the end of narrow_angle to seed cur_errs
      if todo and evaluate is None:
        try:
          cur_errs = _cached_evaluate(current)
          for i in todo:
            best[i] = (limb_error(cur_errs, i, stage), plans[i][stage])
        except Exception:
          cur_errs = {}
      for i, plan in enumerate(plans):
        if i not in applied:
          plan.update(narrow_angle=None, bend=None, knee=None)
  try:
    errors = _cached_evaluate(current)
  except ValueError as error:
    raise ValueError(f'thin_limbs_3d result does not fold: {error}') from error
  summary = []
  for i, plan in enumerate(plans):
    shown = '/'.join(f'{errors[n]:.0f}' for n in pairs[i] if n in errors)
    summary.append(
        f'{plan["proximal"]}: narrow {plan["narrow_angle"]}, bend'
        f' {plan["bend"]}, knee {plan["knee"]} (error {shown} deg)'
    )
  _note('Plan: ' + '; '.join(summary) + '.')
  _resolve_arguments({'limbs': plans})
  return current


@_preserve_3d_overlay
def tree_edit(
    state: ShapingState,
    *,
    tree: Mapping[str, Any],
    stick_name_mapping: Mapping[str, str],
    narrow: bool = False,
) -> ShapingState:
  """Shapes by editing the stick figure (subdivide sticks, rotate sub-sticks).

  The edited tree must stay packing-equivalent to the original: new joints may
  only subdivide existing sticks (sub-stick lengths sum to the original) and
  sub-sticks may be rotated freely.

  Args:
    state: Current state.
    tree: Edited stick figure as a JSON-compatible dict (all sticks).
    stick_name_mapping: Edited stick name -> original (merged) stick name.
    narrow: Whether to narrow flaps while shaping.

  Returns:
    The new state.
  """
  edited_tree = stick_figure_lib.StickFigure.from_dict(dict(tree))
  if not utils_tree.check_trees_are_packing_equivalent(
      state.tree,
      edited_tree,
      state.packing_config,
      check_traversal_order=False,
  ):
    raise ValueError('Edited tree is not packing-equivalent to the base tree.')
  merge_converter = utils_tree.get_merge_converter_from_substick_map(
      edited_tree, dict(stick_name_mapping)
  )
  flaps, rivers = state.shaping_elements
  shaped = heuristics_utils.shape_cp_with_heuristics_pipeline_wrapper(
      cp=state.pattern,
      packing_output=state.packing,
      solving_output=state.solving,
      solving_grid_size=state.solving_grid_size,
      stick_figure=edited_tree,
      merge_converter=merge_converter,
      shaping_flaps=flaps,
      shaping_rivers=rivers,
      central_crease_removed=state.central_crease_removed,
      narrow=narrow,
  )
  return state.with_pattern(shaped)


# ------------------------------------------------------------------------------
# Fan out, geometric folds, squash, level shifters.
# ------------------------------------------------------------------------------


def _flap_rectangle(
    state: ShapingState, element_name: str
) -> shaping_utils.Rectangle:
  """Returns the legacy rectangle of a flap, rejecting rivers."""
  element = state.element(element_name)
  if not isinstance(element, ShapingFlap):
    raise ValueError(
        f'{element_name} is a {type(element).__name__}; this tool only'
        ' applies to flaps.'
    )
  return shaping_utils.Rectangle(
      name=element.name,
      x_start=element.x_start,
      y_start=element.y_start,
      x_len=element.x_len,
      y_len=element.y_len,
  )


@_preserve_3d_overlay
def fan_out_fold(
    state: ShapingState,
    *,
    element_name: str,
    orientation: str,
    move_delta: float,
    fold_type: str | None = None,
    open_percent: float = 1.0,
) -> ShapingState:
  """Fans out a flap by moving border crease points (wings, fins, tails, pincers)."""
  res = shaping_fan_squash.fan_out_fold_impl(
      state,
      element_name=element_name,
      orientation=orientation,
      move_delta=move_delta,
      fold_type=fold_type,
      open_percent=open_percent,
  )
  msg = shaping_fan_squash.last_report_message()
  if msg:
    _note(msg)
  return res


@_preserve_3d_overlay
def fan_out_fold_heuristic(
    state: ShapingState,
    *,
    element_name: str,
    move_delta: float = 1.0,
    open_percent: float = 1.0,
) -> ShapingState:
  """Fans out a flap, picking fold type and orientation automatically."""
  res = shaping_fan_squash.fan_out_fold_heuristic_impl(
      state,
      element_name=element_name,
      move_delta=move_delta,
      open_percent=open_percent,
  )
  msg = shaping_fan_squash.last_report_message()
  if msg:
    _note(msg)
  return res


@_preserve_3d_overlay
def geometric_simple_fold(
    state: ShapingState,
    *,
    folds: Sequence[Mapping[str, Any]],
    sequential: bool = False,
) -> ShapingState:
  """Folds along cut lines given in folded-form coordinates.

  Args:
    state: Current state.
    folds: Items `{element_names, start_x, start_y, end_x, end_y, fold_type}`
      with coordinates in [0, 1] of the folded form; `element_names` restricts
      the fold to those elements (empty = every intersecting element).
    sequential: Apply folds one after another (rebuilding the folded graph in
      between) instead of simultaneously on the current pattern.

  Returns:
    The new state.
  """
  if not folds:
    raise ValueError('folds must not be empty.')
  flaps, rivers = state.shaping_elements
  element_map: dict[str, ShapingElement] = {f.name: f for f in flaps}
  element_map.update({r.name: r for r in rivers})
  configs = []
  for fold in folds:
    names = fold.get('element_names') or []
    limiting_polygon = None
    if names:
      if names[0] not in element_map:
        raise ValueError(
            f'Element {names[0]!r} not found; known: {sorted(element_map)}'
        )
      limiting_polygon = element_map[names[0]]
    fold_type = _FOLD_TYPES.get(str(fold['fold_type']).lower())
    if fold_type is None:
      raise ValueError("fold_type must be 'mountain' or 'valley'.")
    configs.append({
        'cut_line_start': (fold['start_x'], fold['start_y']),
        'cut_line_end': (fold['end_x'], fold['end_y']),
        'fold_assignment': fold_type,
        'limiting_polygon': limiting_polygon,
    })
  pattern = state.pattern
  if sequential:
    first, *rest = configs
    pattern = geometric_simple_fold_lib.shape_cp_geometric(
        pattern, **first, graph_data=state.graph_data
    )
    for config in rest:
      pattern = geometric_simple_fold_lib.shape_cp_geometric(
          pattern,
          **config,
          graph_data=geometric_simple_fold_lib.prepare_graph_data(pattern),
      )
  else:
    graph_data = state.graph_data
    new_creases = []
    for config in configs:
      new_creases.extend(
          geometric_simple_fold_lib.get_geometric_fold_creases(
              crease_pattern=pattern, graph_data=graph_data, **config
          )
      )
    pattern = crease_pattern_lib.CreasePattern(
        creases=list(pattern.creases) + new_creases, grid_size=-1
    )
  return state.with_pattern(pattern)


@_preserve_3d_overlay
def squash_fold(
    state: ShapingState,
    *,
    element_name: str,
    ridge_index: int | None = None,
    starting_position_index: int | None = None,
    ending_position_index: int | None = None,
    open_percent: float = 0.45,
) -> ShapingState:
  """Squashes a flap open so it lies flat (squash/sink along a ridge).

  Args:
    state: Current state.
    element_name: Flap name.
    ridge_index: Which ridge of the flap to squash (deterministic order,
      typically 0..3); the first ridge when None.
    starting_position_index: Start vertex index along the ridge (vertices are
      intersections with outgoing creases, counted from the corner side).
    ending_position_index: End vertex index (exclusive); defaults to start+2.
    open_percent: fold_percent of the squash's new creases. 1.0 folds the squash
      flat (the outline barely changes); about 0.3-0.5 leaves it open so the
      flap spreads into a broad 3-D plate (scorpion chela: folded breadth x1.5
      at 0.3).

  Returns:
    The new state.
  """
  res = shaping_fan_squash.squash_fold_impl(
      state,
      element_name=element_name,
      ridge_index=ridge_index,
      starting_position_index=starting_position_index,
      ending_position_index=ending_position_index,
      open_percent=open_percent,
  )
  msg = shaping_fan_squash.last_report_message()
  if msg:
    _note(msg)
  return res


def _open_new_creases(
    before: crease_pattern_lib.CreasePattern,
    after: crease_pattern_lib.CreasePattern,
    fold_percent: float,
) -> crease_pattern_lib.CreasePattern:
  """`after` with the creases not in `before` set to `fold_percent`."""

  def key(crease: crease_utils.Crease) -> tuple[float, ...]:
    ends = sorted([
        (round(crease.p1_init.x, 9), round(crease.p1_init.y, 9)),
        (round(crease.p2_init.x, 9), round(crease.p2_init.y, 9)),
    ])
    return (*ends[0], *ends[1])

  old = {key(crease) for crease in before.creases}
  creases = [
      crease
      if key(crease) in old
      else crease_utils.Crease(
          crease.p1_init,
          crease.p2_init,
          fold=crease.fold,
          fold_percent=fold_percent,
          elevation=crease.elevation,
      )
      for crease in after.creases
  ]
  opened = crease_pattern_lib.CreasePattern(creases=creases)
  opened.tolerance = after.tolerance
  return opened


def _squash_ridges(element: ShapingFlap) -> list[crease_utils.Crease]:
  """Center-to-corner ridges of a flap that a squash can open, sorted.

  The squash library orients a ridge from its corner to its circle center, so
  the diagonals of a square flap (one circle center) are split at the center
  and the central ridge of a wide flap is left out.

  Args:
    element: The flap.

  Returns:
    The ridges in a deterministic order (the `ridge_index` order).
  """
  centers = element.circle_centers
  if len(centers) == 1:
    ridges = [
        crease_utils.Crease(centers[0], corner, fold=crease_utils.Fold.MOUNTAIN)
        for corner in element.corners
    ]
  else:
    central = crease_utils.Crease(
        centers[0], centers[1], fold=crease_utils.Fold.MOUNTAIN
    )
    ridges = [r for r in element.get_ridges() if not r.is_collinear(central)]
  ridges.sort(key=lambda r: (r.p1.x, r.p1.y, r.p2.x, r.p2.y))
  return ridges


def _ridge_segment_counts(
    pattern: crease_pattern_lib.CreasePattern, element: ShapingFlap
) -> list[int]:
  """Number of crease segments on each squash ridge of `element`."""
  rectangles = [element.to_legacy_rectangle()]
  counts = []
  for ridge in _squash_ridges(element):
    try:
      start_side = squash_sink_shaping.get_start_side_of_ridge(
          ridge, ridge, rectangles
      )
    except ValueError:
      counts.append(-1)
      continue
    counts.append(
        len(
            squash_sink_shaping.get_oriented_ordered_crease_segments_on_ridge(
                pattern, ridge, start_side
            )
        )
    )
  return counts


def _elevated_flaps_by_name(
    state: ShapingState, *, group_mirrored: bool
) -> dict[str, utils_ls.LevelShiftable]:
  """Flaps that can take a level shifter (width > 1), keyed by merged name."""
  symmetric_pairs = {
      s.name: s.symmetric_stick_name
      for s in state.tree.children
      if s.symmetric_stick_name is not None
  }
  elevated = utils_ls.get_elevated_flaps(
      state.flap_rectangles,
      state.packing.sacrificial_flaps or [],
      state.packing_grid_size,
      group_mirrored=group_mirrored,
      symmetric_pairs=symmetric_pairs,
  )
  return {flap.main_flap.name: flap for flap in elevated}


def _valid_level_shifter_positions(
    flap: utils_ls.LevelShiftable, pattern: crease_pattern_lib.CreasePattern
) -> list[utils_ls.LSPosition]:
  return sorted(flap.valid_positions(flap.axial_paths(pattern.creases).keys()))


_NO_LEVEL_SHIFTER_BASE = (
    'This base has no level shifters: the packing has no sacrificial flaps'
    ' (packed with PackingConfig.use_level_shifter=False, the default). Level'
    ' shifters are planned at packing time: re-pack with'
    ' PackingTaskConfig(use_level_shifter=True) (flaps need width >= 2 and'
    ' length > width; rivers width >= 2) and re-solve; only such bases list'
    ' level_shifter_positions.'
)


def level_shifter_unavailable_reason(state: ShapingState) -> str | None:
  """Why `level_shifter` cannot apply to this base, or None if it can."""
  if not state.packing.sacrificial_flaps:
    return _NO_LEVEL_SHIFTER_BASE
  return None


def level_shifter_positions(state: ShapingState) -> dict[str, list[Any]]:
  """Lists valid level-shifter positions per (unmerged) element name.

  Empty when the base was packed without level shifters (see
  `level_shifter_unavailable_reason`).
  """
  if not state.packing.sacrificial_flaps:
    logging.info('level_shifter_positions: %s', _NO_LEVEL_SHIFTER_BASE)
    return {}
  _, merged_names = state.tree.merge_linear_chains()
  positions: dict[str, list[Any]] = {}
  for merged_name, flap in _elevated_flaps_by_name(
      state, group_mirrored=False
  ).items():
    try:
      valid = _valid_level_shifter_positions(flap, state.pattern)
    except (ValueError, AssertionError, KeyError) as error:
      logging.warning('Level shifter positions for %s: %s', merged_name, error)
      continue
    for name in merged_names.get(merged_name, [merged_name]):
      positions[name] = [str(position) for position in valid]
  return positions


@_preserve_3d_overlay
def level_shifter(
    state: ShapingState, *, element_name: str, position_percent: float
) -> ShapingState:
  """Widens a flap with a level shifter (experimental; flap width > 1 only).

  Args:
    state: Current state.
    element_name: Flap name (stick width must be > 1).
    position_percent: Position in [0, 1] mapped linearly onto the sorted valid
      positions (0 = only the part near other flaps widens, 1 = whole flap).

  Returns:
    The new state.
  """
  reason = level_shifter_unavailable_reason(state)
  if reason:
    raise ValueError(reason)
  by_name = _elevated_flaps_by_name(state, group_mirrored=True)
  if not by_name:
    raise ValueError('No level-shiftable flaps in this packing.')
  _, merged_names = state.tree.merge_linear_chains()
  unmerged_to_merged = {}
  for merged, unmerged_list in merged_names.items():
    for unmerged in unmerged_list:
      unmerged_to_merged[unmerged] = merged
  merged_name = unmerged_to_merged.get(element_name, element_name)
  flap = by_name.get(merged_name)
  if flap is None:
    raise ValueError(
        f'{element_name} cannot take a level shifter; candidates:'
        f' {sorted(by_name)}'
    )
  valid = _valid_level_shifter_positions(flap, state.pattern)
  if not valid:
    raise ValueError(f'No valid level shifter positions for {element_name}.')
  position = utils_ls.get_ls_position_from_percent(
      valid, float(position_percent)
  )
  creases = flap.apply_level_shifter(state.pattern.creases, position)
  pattern = crease_pattern_lib.CreasePattern(
      creases=creases,
      grid_size=state.solving_grid_size,
      light_mode=not state.config.check_all_crease_intersections,
  )
  return state.with_pattern(pattern)


def _flap_plate_breadth(
    model: fidelity.FoldedModel, element: ShapingFlap, grid_size: int
) -> float | None:
  """Secondary principal extent (grid units) of a flap's folded paper region."""
  rect = element.to_legacy_rectangle()
  x0 = max(0.0, min(rect.x_start, rect.x_start + rect.x_len))
  x1 = min(1.0, max(rect.x_start, rect.x_start + rect.x_len))
  y0 = max(0.0, min(rect.y_start, rect.y_start + rect.y_len))
  y1 = min(1.0, max(rect.y_start, rect.y_start + rect.y_len))
  if x1 - x0 <= 2e-4 or y1 - y0 <= 2e-4:
    return None
  xs = np.linspace(x0 + 1e-4, x1 - 1e-4, 17)
  ys = np.linspace(y0 + 1e-4, y1 - 1e-4, 17)
  points = np.array([(x, y) for x in xs for y in ys], dtype=float)
  located = model.locate(points)
  located = located[~np.isnan(located).any(axis=1)]
  if len(located) < 3:
    return None
  centered = located - located.mean(axis=0)
  singular = np.linalg.svd(centered, compute_uv=False) / math.sqrt(len(located))
  return float(singular[1] * grid_size)


def _flap_crease_predicates(
    element: ShapingFlap, *, ridge_only: bool
) -> list[tuple[str, Any]]:
  """Candidate crease-selection predicates for `open_flap`."""
  rect = element.to_legacy_rectangle()
  rx0 = min(rect.x_start, rect.x_start + rect.x_len)
  rx1 = max(rect.x_start, rect.x_start + rect.x_len)
  ry0 = min(rect.y_start, rect.y_start + rect.y_len)
  ry1 = max(rect.y_start, rect.y_start + rect.y_len)
  rbox = (max(0.0, rx0), min(1.0, rx1), max(0.0, ry0), min(1.0, ry1))

  cboxes: list[tuple[float, float, float, float]] = []
  if element.circle_centers:
    cc0 = element.circle_centers[0]
    rad = min(
        abs(cc0.x - rx0),
        abs(rx1 - cc0.x),
        abs(cc0.y - ry0),
        abs(ry1 - cc0.y),
    )
    for cc in element.circle_centers:
      bx0, bx1 = max(0.0, cc.x - rad), min(1.0, cc.x + rad)
      by0, by1 = max(0.0, cc.y - rad), min(1.0, cc.y + rad)
      if bx1 - bx0 > 1e-4 and by1 - by0 > 1e-4:
        cboxes.append((bx0, bx1, by0, by1))

  all_ridges = element.get_ridges()

  def on_paper_border(pt: Any) -> bool:
    return pt.x < 1e-4 or pt.x > 1.0 - 1e-4 or pt.y < 1e-4 or pt.y > 1.0 - 1e-4

  def in_box(pt: Any, box: tuple[float, float, float, float]) -> bool:
    x0, x1, y0, y1 = box
    return x0 - 1e-4 <= pt.x <= x1 + 1e-4 and y0 - 1e-4 <= pt.y <= y1 + 1e-4

  def strict_box(pt: Any, box: tuple[float, float, float, float]) -> bool:
    if not in_box(pt, box):
      return False
    x0, x1, y0, y1 = box
    for xv in (x0, x1):
      if 1e-4 < xv < 1.0 - 1e-4 and abs(pt.x - xv) < 1e-4:
        return False
    for yv in (y0, y1):
      if 1e-4 < yv < 1.0 - 1e-4 and abs(pt.y - yv) < 1e-4:
        return False
    return True

  def in_cboxes(pt: Any) -> bool:
    return any(in_box(pt, b) for b in cboxes)

  def strict_cboxes(pt: Any) -> bool:
    return any(strict_box(pt, b) for b in cboxes)

  def is_ridge(c: crease_utils.Crease) -> bool:
    return any(c.is_collinear(r) for r in all_ridges)

  def touches_cc(c: crease_utils.Crease) -> bool:
    return any(
        c.p1_init.isclose(cc) or c.p2_init.isclose(cc)
        for cc in element.circle_centers
    )

  if ridge_only:
    return [
        (
            'ridges_in',
            lambda c: in_box(c.p1_init, rbox)
            and in_box(c.p2_init, rbox)
            and is_ridge(c),
        ),
        (
            'cbox_ridges',
            lambda c: (strict_cboxes(c.p1_init) or strict_cboxes(c.p2_init))
            and is_ridge(c),
        ),
        (
            'cc_diag',
            lambda c: touches_cc(c)
            and not c.is_horizontal
            and not c.is_vertical,
        ),
    ]

  preds: list[tuple[str, Any]] = [
      (
          'cbox_strict_vert',
          lambda c: (strict_cboxes(c.p1_init) or strict_cboxes(c.p2_init))
          and c.is_vertical,
      ),
      (
          'cbox_strict_horiz',
          lambda c: (strict_cboxes(c.p1_init) or strict_cboxes(c.p2_init))
          and c.is_horizontal,
      ),
      (
          'cbox_strict_axis',
          lambda c: (strict_cboxes(c.p1_init) or strict_cboxes(c.p2_init))
          and (c.is_horizontal or c.is_vertical),
      ),
      (
          'rect_strict_vert',
          lambda c: (strict_box(c.p1_init, rbox) or strict_box(c.p2_init, rbox))
          and c.is_vertical,
      ),
      (
          'rect_strict_horiz',
          lambda c: (strict_box(c.p1_init, rbox) or strict_box(c.p2_init, rbox))
          and c.is_horizontal,
      ),
      (
          'rect_border_vert',
          lambda c: (
              (strict_box(c.p1_init, rbox) and on_paper_border(c.p2_init))
              or (strict_box(c.p2_init, rbox) and on_paper_border(c.p1_init))
          )
          and c.is_vertical,
      ),
      (
          'rect_border_horiz',
          lambda c: (
              (strict_box(c.p1_init, rbox) and on_paper_border(c.p2_init))
              or (strict_box(c.p2_init, rbox) and on_paper_border(c.p1_init))
          )
          and c.is_horizontal,
      ),
      (
          'rect_both_vert',
          lambda c: in_box(c.p1_init, rbox)
          and in_box(c.p2_init, rbox)
          and c.is_vertical,
      ),
      (
          'rect_both_horiz',
          lambda c: in_box(c.p1_init, rbox)
          and in_box(c.p2_init, rbox)
          and c.is_horizontal,
      ),
      (
          'cbox_both_all',
          lambda c: in_cboxes(c.p1_init) and in_cboxes(c.p2_init),
      ),
      (
          'rect_strict_axis',
          lambda c: (strict_box(c.p1_init, rbox) or strict_box(c.p2_init, rbox))
          and (c.is_horizontal or c.is_vertical),
      ),
      (
          'rect_both_all',
          lambda c: in_box(c.p1_init, rbox) and in_box(c.p2_init, rbox),
      ),
  ]
  if len(cboxes) > 1:
    for idx, box in enumerate(cboxes):
      preds.extend([
          (
              f'cb{idx}_strict_vert',
              lambda c, b=box: (
                  strict_box(c.p1_init, b) or strict_box(c.p2_init, b)
              )
              and c.is_vertical,
          ),
          (
              f'cb{idx}_strict_horiz',
              lambda c, b=box: (
                  strict_box(c.p1_init, b) or strict_box(c.p2_init, b)
              )
              and c.is_horizontal,
          ),
          (
              f'cb{idx}_border_vert',
              lambda c, b=box: (
                  (strict_box(c.p1_init, b) and on_paper_border(c.p2_init))
                  or (strict_box(c.p2_init, b) and on_paper_border(c.p1_init))
              )
              and c.is_vertical,
          ),
          (
              f'cb{idx}_border_horiz',
              lambda c, b=box: (
                  (strict_box(c.p1_init, b) and on_paper_border(c.p2_init))
                  or (strict_box(c.p2_init, b) and on_paper_border(c.p1_init))
              )
              and c.is_horizontal,
          ),
      ])
  preds.extend([
      (
          'ridges_in',
          lambda c: in_box(c.p1_init, rbox)
          and in_box(c.p2_init, rbox)
          and is_ridge(c),
      ),
      (
          'cc_diag',
          lambda c: touches_cc(c) and not c.is_horizontal and not c.is_vertical,
      ),
  ])
  return preds


# Vertex error (percent) above which the fold stage refuses a pattern
# (`lib.fold_and_render`); tools must not keep candidates above it.
_MAX_RENDERABLE_VERTEX_ERROR = 1.0


def open_flap(
    state: ShapingState,
    *,
    element_name: str,
    fold_percent: float = 0.3,
    ridge_only: bool = False,
) -> ShapingState:
  """Opens a flap's creases to make it spread into a broad 3-D plate.

  Tries clean geometric crease subsets on the flap (axial pleats within the
  flap's circle squares or rectangle, border-connected pleats, and diagonal
  ridges, leaving paper-border edges untouched), verifies each candidate with
  the deterministic folder, and keeps the folding pattern that spreads the flap
  widest.

  Args:
    state: Current state.
    element_name: Flap name.
    fold_percent: Target fold_percent for opened creases in (0, 1] (0.3-0.5
      works well for broad plates; falls back to 0.5 when a wider opening does
      not fold cleanly).
    ridge_only: When True, only open center-to-corner / tip diagonal ridges
      (tent-like shape).

  Returns:
    The new state with the flap partially opened.
  """
  if not 0.0 < fold_percent <= 1.0:
    raise ValueError(f'fold_percent must be in (0, 1], got {fold_percent}.')
  element = state.element(element_name)
  if not isinstance(element, ShapingFlap):
    raise ValueError(f'{element_name} is not a flap; open_flap needs a flap.')

  grid_size = state.solving_grid_size or 1
  base_breadth = 0.0
  try:
    base_model = _fold_for_evaluation(state.pattern)
    measured = _flap_plate_breadth(base_model, element, grid_size)
    if measured is not None:
      base_breadth = measured
  except Exception:  # pylint: disable=broad-exception-caught
    pass

  preds = _flap_crease_predicates(element, ridge_only=ridge_only)
  unique_subsets: list[tuple[str, tuple[int, ...]]] = []
  seen_indices: set[tuple[int, ...]] = set()
  for mode_name, pred in preds:
    indices = tuple(
        idx
        for idx, c in enumerate(state.pattern.creases)
        if c.fold != crease_utils.Fold.BORDER
        and not math.isclose(c.fold_percent, fold_percent, abs_tol=1e-6)
        and pred(c)
    )
    if indices and indices not in seen_indices:
      seen_indices.add(indices)
      unique_subsets.append((mode_name, indices))

  if not unique_subsets:
    raise ValueError(
        f'No creases to open on {element_name} at fold_percent={fold_percent}.'
    )

  percents_to_try = [float(fold_percent)]
  if not ridge_only and abs(fold_percent - 0.5) > 0.05 and fold_percent < 1.0:
    percents_to_try.append(0.5)

  best_choice: (
      tuple[float, crease_pattern_lib.CreasePattern, str, int, float] | None
  ) = None
  last_fold_error: Exception | None = None

  for target_fp in percents_to_try:
    for mode_name, indices in unique_subsets:
      index_set = set(indices)
      new_creases = [
          crease_utils.Crease(
              c.p1_init,
              c.p2_init,
              fold=c.fold,
              fold_percent=target_fp,
              elevation=c.elevation,
          )
          if idx in index_set
          else c
          for idx, c in enumerate(state.pattern.creases)
      ]
      candidate_cp = crease_pattern_lib.CreasePattern(
          creases=new_creases,
          light_mode=True,
          tolerance=state.pattern.tolerance,
      )
      try:
        model = _fold_for_evaluation(candidate_cp)
      except Exception as error:  # pylint: disable=broad-exception-caught
        last_fold_error = error
        continue
      # _fold_for_evaluation retries with simulation_tolerance=10, but the
      # fold stage (lib.fold_and_render) refuses vertex errors above 1%:
      # such an opening scores here and then cannot be rendered (moose
      # antler_shovel_r+l ridge 0.7: 1.48% -> fold stage error).
      if model.vertex_error > _MAX_RENDERABLE_VERTEX_ERROR:
        last_fold_error = ValueError(
            f'mode {mode_name} folds with {model.vertex_error:.2f}% vertex'
            f' error (> {_MAX_RENDERABLE_VERTEX_ERROR}%, the fold stage limit)'
        )
        continue
      breadth = _flap_plate_breadth(model, element, grid_size)
      score = breadth if breadth is not None else 0.0
      if base_breadth > 1e-6 and score <= base_breadth * 1.01:
        last_fold_error = ValueError(
            f'mode {mode_name} did not widen {element_name}'
            f' (breadth {score:.3f} vs {base_breadth:.3f})'
        )
        continue
      if best_choice is None or score > best_choice[0]:
        best_choice = (score, candidate_cp, mode_name, len(indices), target_fp)
      if base_breadth > 0 and score / base_breadth >= 1.65:
        break
    if (
        best_choice is not None
        and base_breadth > 0
        and best_choice[0] / base_breadth >= 1.35
    ):
      break

  if best_choice is None:
    raise ValueError(
        f'Opened {element_name} (fold_percent={fold_percent}) does not fold:'
        f' {last_fold_error}'
    ) from last_fold_error

  score, chosen_cp, chosen_mode, opened_count, chosen_fp = best_choice
  ratio_str = (
      f', breadth x{score / base_breadth:.2f}'
      if base_breadth > 1e-6 and score > 0
      else ''
  )
  _note(
      f'Opened {opened_count} creases ({chosen_mode}) on {element_name} to'
      f' fold_percent={chosen_fp}{ratio_str}.'
  )
  return state.with_pattern(chosen_cp)


def reverse_fold_3d(
    state: ShapingState,
    *,
    element_name: str,
    angle: float,
    depth: float = 0.5,
    fold_percent: float = 1.0,
    verify_fold: bool = True,
) -> ShapingState:
  """Local reverse fold on one flap, cut in the folded state."""
  new_state = shaping_fold3d.reverse_fold_3d(
      state,
      element_name=element_name,
      angle=angle,
      depth=depth,
      fold_percent=fold_percent,
      verify_fold=verify_fold,
  )
  _note(shaping_fold3d.last_report_message())
  return new_state


def crimp_fold_3d(
    state: ShapingState,
    *,
    element_name: str,
    angle: float,
    depth: float = 0.4,
    gap: float | None = None,
    fold_percent: float = 1.0,
    verify_fold: bool = True,
) -> ShapingState:
  """Local crimp (two opposite reverse folds) on one flap."""
  new_state = shaping_fold3d.crimp_fold_3d(
      state,
      element_name=element_name,
      angle=angle,
      depth=depth,
      gap=gap,
      fold_percent=fold_percent,
      verify_fold=verify_fold,
  )
  _note(shaping_fold3d.last_report_message())
  return new_state


def resolve_symmetric_element(
    state: ShapingState, element_name: str
) -> str | None:
  """Finds the bilateral mirror partner of an element in the crease pattern.

  Looks up `symmetric_stick_name` on the stick figure if present in the tree,
  or checks standard bilateral naming conventions:
  `_L <-> _R`, `_l <-> _r`, `_left <-> _right`, `left <-> right`,
  `L1 <-> R1`, `l1 <-> r1`, etc.

  Args:
    state: Current ShapingState.
    element_name: Name of the element to resolve.

  Returns:
    Name of the mirror partner element, or None if no partner is found.
  """
  if (
      element_name.startswith(('major_', 'minor_'))
      or 'major_' in element_name
      or 'minor_' in element_name
  ):
    return None

  all_names = set(state.element_names())
  if element_name in state.tree:
    stick = state.tree[element_name]
    sym = getattr(stick, 'symmetric_stick_name', None)
    if (
        sym
        and sym in all_names
        and sym != element_name
        and not (
            sym.startswith(('major_', 'minor_'))
            or 'major_' in sym
            or 'minor_' in sym
        )
    ):
      return sym

  candidates = []
  # _L / _R suffix with optional number
  if re.search(r'([_])L([0-9]*)$', element_name):
    candidates.append(re.sub(r'([_])L([0-9]*)$', r'\g<1>R\2', element_name))
  elif re.search(r'([_])R([0-9]*)$', element_name):
    candidates.append(re.sub(r'([_])R([0-9]*)$', r'\g<1>L\2', element_name))

  # _l / _r suffix with optional number
  if re.search(r'([_])l([0-9]*)$', element_name):
    candidates.append(re.sub(r'([_])l([0-9]*)$', r'\g<1>r\2', element_name))
  elif re.search(r'([_])r([0-9]*)$', element_name):
    candidates.append(re.sub(r'([_])r([0-9]*)$', r'\g<1>l\2', element_name))

  # Infix _L_ / _R_ or _l_ / _r_
  if '_L_' in element_name:
    candidates.append(element_name.replace('_L_', '_R_', 1))
  elif '_R_' in element_name:
    candidates.append(element_name.replace('_R_', '_L_', 1))
  if '_l_' in element_name:
    candidates.append(element_name.replace('_l_', '_r_', 1))
  elif '_r_' in element_name:
    candidates.append(element_name.replace('_r_', '_l_', 1))

  # Infix _L<digit>_ / _R<digit>_ (e.g. leg_L1_1 <-> leg_R1_1,
  # walk_L4_merus <-> walk_R4_merus)
  if re.search(r'_L([0-9]+)_', element_name):
    candidates.append(re.sub(r'_L([0-9]+)_', r'_R\1_', element_name))
  elif re.search(r'_R([0-9]+)_', element_name):
    candidates.append(re.sub(r'_R([0-9]+)_', r'_L\1_', element_name))
  if re.search(r'_l([0-9]+)_', element_name):
    candidates.append(re.sub(r'_l([0-9]+)_', r'_r\1_', element_name))
  elif re.search(r'_r([0-9]+)_', element_name):
    candidates.append(re.sub(r'_r([0-9]+)_', r'_l\1_', element_name))

  # Prefix L<digit> / R<digit> (e.g. L1_femur <-> R1_femur)
  if re.match(r'^L([0-9]+)', element_name):
    candidates.append(re.sub(r'^L([0-9]+)', r'R\1', element_name))
  elif re.match(r'^R([0-9]+)', element_name):
    candidates.append(re.sub(r'^R([0-9]+)', r'L\1', element_name))

  # left / right words
  if 'left' in element_name:
    candidates.append(element_name.replace('left', 'right', 1))
  elif 'right' in element_name:
    candidates.append(element_name.replace('right', 'left', 1))
  if 'Left' in element_name:
    candidates.append(element_name.replace('Left', 'Right', 1))
  elif 'Right' in element_name:
    candidates.append(element_name.replace('Right', 'Left', 1))

  for cand in candidates:
    if (
        cand in all_names
        and cand != element_name
        and not (
            cand.startswith(('major_', 'minor_'))
            or 'major_' in cand
            or 'minor_' in cand
        )
    ):
      return cand

  for cand in candidates:
    if (
        cand in state.tree
        and cand != element_name
        and not (
            cand.startswith(('major_', 'minor_'))
            or 'major_' in cand
            or 'minor_' in cand
        )
    ):
      return cand

  return None


# `apply_symmetric` scores up to this many accepted (left, right) candidate
# pairs by 3-D asymmetry, stopping early at `_SYMMETRIC_ASYMMETRY_OK`;
# asymmetries within `_SYMMETRIC_ASYMMETRY_TIE` count as ties (the earlier,
# primary-parameter candidate wins).
_SYMMETRIC_MAX_SCORED = 4
_SYMMETRIC_ASYMMETRY_OK = 0.05
_SYMMETRIC_ASYMMETRY_TIE = 0.03
# Above this asymmetry the next depth/angle of the ladder is scored too.
_SYMMETRIC_ASYMMETRY_BAD = 0.15


def _bilateral_asymmetry(
    state: ShapingState,
    candidate: ShapingState,
    model: Any,
    left_element: str,
    right_element: str,
) -> float | None:
  """3-D asymmetry of a left/right pair in the stick figure's frame.

  Maps the folded joints into the figure frame (fidelity alignment: x
  forward, y left, z up) and measures, relative to the sticks' length, the
  vertical elevation difference of the two sides (mean of tip and midpoint
  `|z_L - z_R|`) plus half the mirror error of the tips (`|p_L - M p_R|`, M
  flips y). A dihedral pair (wings at 90 degrees) whose sides fold at
  different heights in side/front views scores high even when each side
  folds cleanly.

  Args:
    state: State before the pair was applied (tree and packing).
    candidate: State after both sides.
    model: The candidate's folded model.
    left_element: Left element (stick or merged flap name).
    right_element: Right element.

  Returns:
    The asymmetry (0 = perfectly mirrored), or None if it cannot be measured.
  """
  try:
    report = fidelity.score_model(
        candidate.tree,
        candidate.packing,
        model,
        layout=state.paper_layout,
        include_geometry=True,
    )
  except Exception:  # pylint: disable=broad-exception-caught
    return None
  geometry = report.get('geometry', {})
  sticks = geometry.get('sticks', {})
  joints = geometry.get('joints', {})
  alignment = report.get('alignment', {})
  if not sticks or not joints or 'rotation' not in alignment:
    return None
  rotation = np.asarray(alignment['rotation'], dtype=float)
  scale = float(alignment['scale'])
  translation = np.asarray(alignment['translation'], dtype=float)
  _, merged_names = state.tree.merge_linear_chains()

  def _chain(name: str) -> list[str]:
    if name in sticks:
      return [name]
    return [n for n in merged_names.get(name, []) if n in sticks]

  def _frame(joint: str) -> np.ndarray | None:
    folded = joints.get(joint, {}).get('folded')
    if folded is None:
      return None
    return scale * rotation @ np.asarray(folded, dtype=float) + translation

  chain_l, chain_r = _chain(left_element), _chain(right_element)
  if not chain_l or not chain_r:
    return None
  tip_l = _frame(sticks[chain_l[-1]]['distal'])
  tip_r = _frame(sticks[chain_r[-1]]['distal'])
  base_l = _frame(sticks[chain_l[0]]['proximal'])
  base_r = _frame(sticks[chain_r[0]]['proximal'])
  if tip_l is None or tip_r is None or base_l is None or base_r is None:
    return None
  length = max(
      1e-6,
      0.5
      * (
          sum(sticks[n]['length'] for n in chain_l)
          + sum(sticks[n]['length'] for n in chain_r)
      ),
  )
  mid_l, mid_r = 0.5 * (tip_l + base_l), 0.5 * (tip_r + base_r)
  dz = 0.5 * (abs(tip_l[2] - tip_r[2]) + abs(mid_l[2] - mid_r[2]))
  mirror = np.array([1.0, -1.0, 1.0])
  mirror_err = float(np.linalg.norm(tip_l - mirror * tip_r))
  return float((dz + 0.5 * mirror_err) / length)


@_preserve_3d_overlay
def apply_symmetric(
    state: ShapingState,
    tool: str | Any = '',
    left_element: str = '',
    right_element: str = '',
    **params: Any,
) -> ShapingState:
  """Applies a tool symmetrically to a left/right pair of elements in 3-D.

  Locked candidate pairs are tested on (left_element, right_element) together,
  evaluating directional parameters (such as angle, fold_order) in both mirrored
  and matched signs. Each candidate pair is folded with the 3-D deterministic
  folder, and its 3-D bilateral symmetry error is measured against the body's
  sagittal midplane. The candidate minimizing mirror error is chosen. If no
  candidate succeeds on both sides, rolls back cleanly.

  Args:
    state: Current ShapingState.
    tool: Tool name string (or callable).
    left_element: Left element name.
    right_element: Right element name (auto-resolved from left_element if
      empty).
    **params: Common parameters for the tool (e.g. angle, depth, fold_percent).

  Returns:
    Updated ShapingState.

  Raises:
    ValueError: If partner cannot be resolved or no candidate pair succeeds.
  """
  start_time = time.perf_counter()
  if not tool and 'tool' in params:
    tool = params.pop('tool')
  if not left_element:
    if 'left_element' in params:
      left_element = params.pop('left_element')
    elif 'element_name' in params:
      left_element = params.pop('element_name')
    elif 'element' in params:
      left_element = params.pop('element')
  if not right_element and 'right_element' in params:
    right_element = params.pop('right_element')
  allow_fallback = params.pop('fallback', True)

  if not left_element:
    raise ValueError('apply_symmetric: left_element is required.')

  if not right_element:
    resolved = resolve_symmetric_element(state, left_element)
    if not resolved:
      raise ValueError(
          'apply_symmetric: could not resolve symmetric partner for'
          f' {left_element!r}'
      )
    right_element = resolved

  names = set(state.element_names())
  if left_element not in names:
    raise ValueError(f'apply_symmetric: unknown left_element {left_element!r}')
  if right_element not in names:
    raise ValueError(
        f'apply_symmetric: unknown right_element {right_element!r}'
    )

  tool_name = (
      normalize_tool_name(tool)
      if isinstance(tool, str)
      else getattr(tool, '__name__', str(tool))
  )
  tool_fn: Any = _TOOLS.get(tool_name) if isinstance(tool, str) else tool
  if tool_fn is None or not callable(tool_fn):
    raise ValueError(f'apply_symmetric: unknown tool {tool!r}')

  left_override = params.pop('left_arguments', None)
  right_override = params.pop('right_arguments', None)

  # Determine directional parameters
  angle_source = left_override if isinstance(left_override, Mapping) else params
  angle_key = None
  for k in ('angle', 'turn_angle', 'bend_angle'):
    if k in angle_source:
      angle_key = k
      break

  if isinstance(left_override, Mapping) and isinstance(right_override, Mapping):
    candidate_pairs: list[tuple[dict[str, Any], dict[str, Any]]] = [
        (dict(left_override), dict(right_override))
    ]
  else:
    # Build depth ladder: primary depth first!
    depth_val = params.get('depth')
    if depth_val is not None:
      if tool_name == 'simple_fold' and float(depth_val) == 0.0:
        raw_depths = [0.0]
      else:
        raw_depths = [float(depth_val), 0.35, 0.5, 0.25, 0.6]
    elif tool_name in (
        'reverse_fold_3d',
        'crimp_fold_3d',
        'crimp_fold',
    ):
      raw_depths = [0.5, 0.35]
    else:
      raw_depths = [None]
    seen_d = set()
    depth_ladder = []
    for val_d in raw_depths:
      if val_d is None:
        depth_ladder.append(None)
        break
      if val_d >= 0.0 and val_d not in seen_d:
        seen_d.add(val_d)
        depth_ladder.append(val_d)

    # Build angle ladder: primary angle first!
    if angle_key is not None and abs(float(params[angle_key])) > 1e-4:
      raw_ang = float(params[angle_key])
      raw_angles = [raw_ang, round(raw_ang * 0.75, 2), round(raw_ang * 0.5, 2)]
      seen_a = set()
      angle_ladder = []
      for val_a in raw_angles:
        if val_a in seen_a:
          continue
        # The requested angle is always tried; only the derived fallback
        # angles (0.75x, 0.5x) are dropped when they get too small to matter.
        # Filtering the primary angle too left the ladder empty for
        # |angle| < 15 (e.g. a 14 degree narrowing turn) and raised IndexError.
        if angle_ladder and abs(val_a) < 15.0:
          continue
        seen_a.add(val_a)
        angle_ladder.append(val_a)
    elif tool_name == 'rotation_3d':
      az_val = params.get('az')
      if az_val is not None:
        angle_ladder = [float(az_val)]
        angle_key = 'az'
      else:
        angle_ladder = [None]
    else:
      angle_ladder = [None]

    # Narrow ladder for simple_fold: primary narrow first!
    if tool_name == 'simple_fold' and 'narrow' in params:
      narrow_ladder = [params['narrow'], not params['narrow']]
    else:
      narrow_ladder = [None]

    order_val = params.get('fold_order')
    opp_order = (
        'VM'
        if order_val == 'MV'
        else ('MV' if order_val == 'VM' else order_val)
    )

    # Priority configs:
    # Tier 1: (primary_narrow, primary_depth, primary_ang)
    # Tier 2: (alt_narrow, primary_depth, primary_ang)
    # Tier 3: (primary/alt_narrow, alt_depths, primary_ang)
    # Tier 4: (primary/alt_narrow, all_depths, alt_angles)
    param_configs: list[tuple[Any, Any, Any]] = []
    p_narrow, p_depth, p_ang = (
        narrow_ladder[0],
        depth_ladder[0],
        angle_ladder[0],
    )
    param_configs.append((p_narrow, p_depth, p_ang))

    for n in narrow_ladder[1:]:
      param_configs.append((n, p_depth, p_ang))

    for d in depth_ladder[1:]:
      param_configs.append((p_narrow, d, p_ang))
      for n in narrow_ladder[1:]:
        param_configs.append((n, d, p_ang))

    for a in angle_ladder[1:]:
      param_configs.append((p_narrow, p_depth, a))
      for n in narrow_ladder[1:]:
        param_configs.append((n, p_depth, a))
      for d in depth_ladder[1:]:
        param_configs.append((p_narrow, d, a))

    candidate_pairs = []
    for cur_narrow, cur_depth, cur_ang in param_configs:
      base_p = dict(params)
      if cur_narrow is not None:
        base_p['narrow'] = cur_narrow
      if cur_depth is not None:
        base_p['depth'] = cur_depth
      if cur_ang is not None and angle_key is not None:
        # 1. Opposite angles (standard bilateral reflection)
        c_l = dict(base_p, **{angle_key: cur_ang})
        c_r = dict(base_p, **{angle_key: -cur_ang})
        if order_val:
          candidate_pairs.append((
              dict(c_l, fold_order=order_val),
              dict(c_r, fold_order=opp_order),
          ))
          candidate_pairs.append((
              dict(c_l, fold_order=order_val),
              dict(c_r, fold_order=order_val),
          ))
        else:
          candidate_pairs.append((c_l, c_r))

        # 2. Inverted opposite angles
        c_l = dict(base_p, **{angle_key: -cur_ang})
        c_r = dict(base_p, **{angle_key: cur_ang})
        if order_val:
          candidate_pairs.append((
              dict(c_l, fold_order=opp_order),
              dict(c_r, fold_order=order_val),
          ))
        else:
          candidate_pairs.append((c_l, c_r))

      elif 'direction' in params:
        dir_val = params['direction']
        opp_dir = (
            'negative'
            if dir_val == 'positive'
            else ('positive' if dir_val == 'negative' else dir_val)
        )
        candidate_pairs.append((
            dict(base_p, direction=dir_val),
            dict(base_p, direction=opp_dir),
        ))
        candidate_pairs.append((
            dict(base_p, direction=dir_val),
            dict(base_p, direction=dir_val),
        ))
      elif tool_name in ('fan_out_fold', 'fan_out_fold_heuristic'):
        delta_val = float(params.get('move_delta', 1.0))
        open_pct = float(params.get('open_percent', 1.0))
        locked = dict(base_p, open_percent=open_pct)
        candidate_pairs.append((
            dict(locked, move_delta=delta_val),
            dict(locked, move_delta=delta_val),
        ))
        candidate_pairs.append((
            dict(locked, move_delta=delta_val),
            dict(locked, move_delta=-delta_val),
        ))
        candidate_pairs.append((
            dict(locked, move_delta=-delta_val),
            dict(locked, move_delta=delta_val),
        ))
      elif tool_name == 'squash_fold':
        open_pct = float(params.get('open_percent', 1.0))
        locked = dict(base_p, open_percent=open_pct)
        candidate_pairs.append((dict(locked), dict(locked)))
      elif 'orientation' in params or 'fold_type' in params:
        ori = params.get('orientation', 'horizontal')
        opp_ori = 'vertical' if ori == 'horizontal' else 'horizontal'
        ft = params.get('fold_type', 'valley')
        opp_ft = 'mountain' if ft == 'valley' else 'valley'
        candidate_pairs.append((
            dict(base_p, orientation=ori, fold_type=ft),
            dict(base_p, orientation=ori, fold_type=opp_ft),
        ))
        candidate_pairs.append((
            dict(base_p, orientation=ori, fold_type=ft),
            dict(base_p, orientation=opp_ori, fold_type=ft),
        ))
        candidate_pairs.append((
            dict(base_p, orientation=ori, fold_type=ft),
            dict(base_p, orientation=opp_ori, fold_type=opp_ft),
        ))
        candidate_pairs.append((
            dict(base_p, orientation=ori, fold_type=ft),
            dict(base_p, orientation=ori, fold_type=ft),
        ))
      else:
        candidate_pairs.append((dict(base_p), dict(base_p)))

  best_state: ShapingState | None = None
  best_mirror_err = math.inf
  best_cands: tuple[dict[str, Any], dict[str, Any]] | None = None
  last_err = 'no candidate tried'

  topo = fidelity.topology(state.tree)
  distal_l = topo.distal.get(left_element, left_element)
  distal_r = topo.distal.get(right_element, right_element)
  prox_l = topo.proximal.get(left_element)
  prox_r = topo.proximal.get(right_element)
  mirror = np.array([1.0, -1.0, 1.0])
  layout = state.paper_layout
  target_joints = fidelity.stick_figure_joint_positions(state.tree)
  to_tree_units = layout.grid_size / layout.grid_units_per_tree_unit

  cache_l: dict[str, tuple[ShapingState | None, str]] = {}
  narrow_failed = False
  unrecoverable_r_count = 0
  unrecoverable_failures = (
      'The cut does not separate',
      'The cut leaves faces too small',
      'is not a flap',
  )

  # Pre-compute base symmetry once if tree is symmetric
  base_sym = None
  min_sym = None
  if getattr(state.tree, 'is_symmetric_tree', False):
    try:
      base_rep = fidelity.score_pattern(
          state.tree, state.packing, state.pattern
      )
      base_sym = float(base_rep.get('components', {}).get('symmetry', 0.0))
      min_sym = params.get('min_symmetry', max(0.0, base_sym - 0.10))
    except Exception:  # pylint: disable=broad-exception-caught
      pass

  # Accepted candidate pairs: (asymmetry, order, state, left args, right args).
  # Only sign / fold-order variants of the first accepted configuration
  # (narrow, depth, |angle|) are scored, so the requested sizes are kept.
  scored: list[tuple[float | None, int, Any, Any, Any]] = []

  def _config(cand: Mapping[str, Any]) -> tuple[Any, Any, Any]:
    angle = cand.get(angle_key) if angle_key else None
    return (
        cand.get('narrow'),
        cand.get('depth'),
        abs(float(angle)) if angle is not None else None,
    )

  for cand_l, cand_r in candidate_pairs:
    if time.perf_counter() - start_time > 15.0 and (
        scored or len(cache_l) >= 2
    ):
      break
    if (
        scored
        and _config(cand_l) != _config(scored[0][3])
        and min((a for a, *_ in scored if a is not None), default=math.inf)
        <= _SYMMETRIC_ASYMMETRY_BAD
    ):
      # Other sizes only when the requested one leaves the sides lopsided.
      break
    if narrow_failed and (cand_l.get('narrow') or cand_r.get('narrow')):
      continue
    if unrecoverable_r_count >= 2:
      break

    call_l = dict(cand_l, element_name=left_element)
    call_r = dict(cand_r, element_name=right_element)
    if tool_name == 'rotation_3d':
      try:
        st_lr = rotation_3d(state, rotations=[call_l, call_r])
      except Exception as e:  # pylint: disable=broad-exception-caught
        last_err = f'rotation_3d failed: {e}'
        continue
    else:
      # JSON key: arguments may hold lists (e.g. inscribed_flap depths).
      key_l = json.dumps(call_l, sort_keys=True, default=str)
      if key_l in cache_l:
        st_l, err_l = cache_l[key_l]
        if st_l is None:
          last_err = f'left {left_element} failed: {err_l}'
          continue
      else:
        try:
          st_l = tool_fn(state, **call_l)
          cache_l[key_l] = (st_l, '')
        except Exception as e:  # pylint: disable=broad-exception-caught
          err_msg = str(e)
          cache_l[key_l] = (None, err_msg)
          if (
              cand_l.get('narrow')
              or 'Narrowing failed' in err_msg
              or 'No creases found with point' in err_msg
          ):
            narrow_failed = True
          last_err = f'left {left_element} failed: {err_msg}'
          continue
      try:
        st_lr = tool_fn(st_l, **call_r)
      except Exception as e:  # pylint: disable=broad-exception-caught
        err_msg = str(e)
        if any(uf in err_msg for uf in unrecoverable_failures):
          unrecoverable_r_count += 1
        if (
            cand_r.get('narrow')
            or 'Narrowing failed' in err_msg
            or 'No creases found with point' in err_msg
        ):
          narrow_failed = True
        last_err = f'right {right_element} failed: {err_msg}'
        continue

    if state.config.check_flat_foldability:
      ok_ff, err_ff = fidelity.preserves_foldability(
          state.pattern, st_lr.pattern
      )
      if not ok_ff:
        last_err = f'flat foldability failed: {err_ff}'
        continue

    # Verify 3-D fold once per accepted candidate pair
    try:
      lr_model = _fold_for_evaluation(st_lr.pattern)
    except Exception as e:  # pylint: disable=broad-exception-caught
      last_err = f'3-D fold failed: {e}'
      continue
    if lr_model.vertex_error > _MAX_RENDERABLE_VERTEX_ERROR:
      # Folds only with the relaxed tolerance; the fold stage refuses it.
      last_err = f'3-D fold vertex error {lr_model.vertex_error:.2f}% > 1%'
      continue

    if base_sym is not None and min_sym is not None:
      try:
        fold_rep = fidelity.score_pattern(
            state.tree, st_lr.packing, st_lr.pattern
        )
        fold_sym = float(fold_rep.get('components', {}).get('symmetry', 0.0))
        if fold_sym < min_sym:
          last_err = (
              f'symmetry degraded to {fold_sym:.3f} (base {base_sym:.3f})'
          )
          continue
      except Exception:  # pylint: disable=broad-exception-caught
        pass

    best_state = st_lr
    best_cands = (cand_l, cand_r)
    asym = _bilateral_asymmetry(
        state, st_lr, lr_model, left_element, right_element
    )
    scored.append((asym, len(scored), st_lr, cand_l, cand_r))
    if (
        asym is None
        or asym <= _SYMMETRIC_ASYMMETRY_OK
        or len(scored) >= _SYMMETRIC_MAX_SCORED
    ):
      break

  if scored:
    # Lowest 3-D asymmetry (elevation difference of the two sides + mirror
    # error); an earlier (primary-parameter) candidate wins near-ties.
    def _rank(item: tuple[Any, ...]) -> tuple[float, int]:
      if item[0] is None:
        return (math.inf, item[1])
      return (float(round(item[0] / _SYMMETRIC_ASYMMETRY_TIE)), item[1])

    chosen = min(scored, key=_rank)
    best_state, best_cands = chosen[2], (chosen[3], chosen[4])
    best_mirror_err = chosen[0] if chosen[0] is not None else 0.05

  if best_state is None or best_cands is None:
    if allow_fallback and tool_name == 'reverse_fold_3d':
      try:
        fb_params = dict(params)
        fb_params.pop('depth', None)
        return apply_symmetric(
            state,
            tool='simple_fold',
            left_element=left_element,
            right_element=right_element,
            fallback=False,
            **fb_params,
        )
      except Exception:  # pylint: disable=broad-exception-caught
        pass
    raise ValueError(
        f'apply_symmetric({tool_name}, {left_element}, {right_element}) failed:'
        f' {last_err} (best mirror_err={best_mirror_err:.3f})'
    )

  delta_c = len(best_state.pattern.creases) - len(state.pattern.creases)
  sym_score = max(0.0, 1.0 - best_mirror_err)
  dur = time.perf_counter() - start_time
  ang_l = best_cands[0].get(angle_key, '') if angle_key else ''
  ang_r = best_cands[1].get(angle_key, '') if angle_key else ''
  ang_str = f'angles=({ang_l}°, {ang_r}°), ' if angle_key else ''
  depth_str = (
      f'depth={best_cands[0].get("depth")}, '
      if 'depth' in best_cands[0]
      else ''
  )
  _resolve_arguments({
      'tool': tool_name,
      'left_element': left_element,
      'right_element': right_element,
      'left_arguments': best_cands[0],
      'right_arguments': best_cands[1],
  })
  note = (
      f'apply_symmetric({tool_name}, {left_element}, {right_element}):'
      f' {ang_str}{depth_str}+{delta_c} creases, 3D'
      f' mirror_err={best_mirror_err:.3f} (elevation + mirror asymmetry,'
      f' best of {len(scored)} scored), sym={sym_score:.2f}, {dur:.2f}s'
  )
  _note(note)
  return best_state


def fold_spine_3d(state: ShapingState, **params: Any) -> ShapingState:
  """Sagittal midplane fold giving volumetric 3-D body depth."""
  if 'spine_fold_percent' in params:
    params['fold_percent'] = params.pop('spine_fold_percent')
  res = shaping_extras.fold_spine_3d(state, **params)
  msg = shaping_extras.last_report_message()
  if msg:
    _note(msg)
  return res


@_preserve_3d_overlay
def spread_fork(state: ShapingState, **params: Any) -> ShapingState:
  """Spreads parallel co-planar limbs/pincers into a visible V-shaped open gap."""
  call_params = dict(params)
  if 'element_a' not in call_params and 'element_name' in call_params:
    call_params['element_a'] = call_params.pop('element_name')
  if 'element_b' not in call_params and 'partner_element' in call_params:
    call_params['element_b'] = call_params.pop('partner_element')
  if 'element_a' in call_params and 'element_b' not in call_params:
    partner = resolve_symmetric_element(state, str(call_params['element_a']))
    if partner:
      call_params['element_b'] = partner
  res = shaping_extras.spread_fork(state, **call_params)
  msg = shaping_extras.last_report_message()
  if msg:
    _note(msg)
  return res


@_preserve_3d_overlay
def radial_fan(
    state: ShapingState,
    *,
    elements: Sequence[str],
    arc_deg: float | None = None,
    center_az: float | None = None,
    base_el: float | None = None,
    dome_el: float = 0.0,
    fan_wide: bool = True,
    open_percent: float = 1.0,
    verify_fold: bool = True,
) -> ShapingState:
  """Poses sibling rays along a smooth circular arc and fans broad rays.

  Needs at least two sibling ray elements (e.g. feather or frill sub-rays that
  share a joint); a single flap such as one wing is rejected.
  """
  res = shaping_extras.radial_fan(
      state,
      elements=elements,
      arc_deg=arc_deg,
      center_az=center_az,
      base_el=base_el,
      dome_el=dome_el,
      fan_wide=fan_wide,
      open_percent=open_percent,
      verify_fold=verify_fold,
  )
  msg = shaping_extras.last_report_message()
  if msg:
    _note(msg)
  return res


def open_river_3d(
    state: ShapingState,
    *,
    element_name: str,
    fold_percent: float = 0.65,
    max_vertex_error: float = 0.35,
    verify_fold: bool = True,
) -> ShapingState:
  """Partially opens interior accordion pleats of a wide river in 3-D."""
  res = shaping_extras.open_river_3d(
      state,
      element_name=element_name,
      fold_percent=fold_percent,
      max_vertex_error=max_vertex_error,
      verify_fold=verify_fold,
  )
  msg = shaping_extras.last_report_message()
  if msg:
    _note(msg)
  return res


# ------------------------------------------------------------------------------
# Direction-targeted bends: pick the bend tool's sign and size by measurement.
# ------------------------------------------------------------------------------

# Bend tools `bend_to_direction` searches, with their depth argument range.
_BEND_TOOLS = ('crimp_fold_3d', 'reverse_fold_3d', 'simple_fold')


def _unit_vector(azimuth: float, elevation: float) -> np.ndarray:
  a, e = math.radians(azimuth), math.radians(elevation)
  return np.array(
      [math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)]
  )


def _angle_between(first: np.ndarray, second: np.ndarray) -> float:
  cosine = float(np.dot(first, second)) / max(
      1e-12, float(np.linalg.norm(first) * np.linalg.norm(second))
  )
  return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def measured_directions(
    state: ShapingState,
    pattern: crease_pattern_lib.CreasePattern | None = None,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
  """Folds `pattern` and measures every stick in `rotation_3d`'s own frame.

  See `fidelity.root_frame_directions`: the root element's faces give the
  rigid transform back to the unshaped base, where the root stick's
  start->end is forward and the paper's top side at
  `geometric_fold_3d_heuristics.root_reference_point` is up (the posture
  normal the 3-D heuristics pin their frame to). Unlike
  `fidelity.stick_directions` there is no figure fit, so a flap's error is
  its own (no offset from the corner-anchored rotation axis, no frame tilt
  from other flaps, no mirror). Falls back to `fidelity.stick_directions`
  when the root frame cannot be built.

  Args:
    state: The shaping state.
    pattern: Pattern to measure (default `state.pattern`).

  Returns:
    (fidelity report with `direction_error_deg` replaced by the measured
    errors and the fitted ones under `direction_error_deg_fit`, per-stick
    `{target, realized, error}` in the figure frame).
  """
  pattern = state.pattern if pattern is None else pattern
  model = fidelity.FoldedModel.from_pattern(pattern)
  layout = state.paper_layout
  report, fitted = fidelity.stick_directions(
      state.tree, state.packing, model, layout=layout
  )
  try:
    base_model = fidelity.FoldedModel.from_pattern(state.base_pattern)
    with _designer_root_3d(state):
      root = heuristics.get_first_stick_for_shaping(state.tree)
    flaps, rivers = state.unmerged_shaping_elements
    element = shaping_elements.get_element_by_name(root.name, flaps, rivers)
    grid = state.solving_grid_size
    unit_samples = np.arange(0.25, grid, 0.5) / grid
    # Shaping elements are in unit-paper or grid coordinates depending on
    # how they were built: keep the scale that finds the element.
    points = []
    element_scale = 1.0
    for scale in (1.0, float(grid)):
      found = []
      for x in unit_samples:
        for y in unit_samples:
          pt = point_utils.Point(
              x=float(x * scale), y=float(y * scale), require_on_paper=False
          )
          if element.is_internal_point(pt, strict=True):
            found.append((x, y))
      if len(found) > len(points):
        points, element_scale = found, scale
    if not points:
      raise ValueError(f'no paper points inside root element {root.name}')
    points = np.array(
        [
            p
            for p in points
            if not any(
                x0 < p[0] < x1 and y0 < p[1] < y1
                for x0, x1, y0, y1 in layout.flap_rectangles
            )
        ]
        or points
    )
    # "Up": the paper's top side at the root reference point, the normal the
    # 3-D heuristics pin their posture frame to (independent of which face
    # the flat folded state happens to number 0).
    up_reference = None
    reference = geometric_fold_3d_heuristics.root_reference_point(element)
    if reference is not None:
      reference_face = base_model.locate_faces(
          np.asarray(reference, dtype=float)[None] / element_scale
      )[0]
      if reference_face >= 0:
        up_reference = base_model.matrices[reference_face][:3, 2]
    alignment = angle_utils.get_rotation_matrix_to_align(
        np.array(root.get_coords(None), dtype=float),
        np.array([0.0, 1.0, 0.0]),
    )
    try:
      axis_element = element
      if isinstance(element, shaping_elements.ShapingRiver):
        banks = state.shaping_heuristic_data['outer_bank_indices']
        axis_element = element.update_outer_bank_index(banks[element.name])
      start, end = shaping_elements.get_unfolded_root_vector(
          axis_element, state.tree
      )
      root_axis_points = (
          np.array([start.x, start.y]) / element_scale,
          np.array([end.x, end.y]) / element_scale,
      )
    except (AssertionError, ValueError, KeyError, IndexError) as error:
      logging.info('measured_directions: no root vector: %s', error)
      root_axis_points = None
    measured = fidelity.root_frame_directions(
        state.tree,
        layout,
        model,
        base_model,
        root_name=root.name,
        root_points=points,
        up_reference=up_reference,
        root_alignment=alignment,
        root_axis_points=root_axis_points,
    )
  except (ValueError, KeyError, IndexError, np.linalg.LinAlgError) as error:
    logging.warning('measured_directions: root frame unavailable: %s', error)
    measured = None
  if measured is None:
    return report, fitted
  directions = measured['directions']
  report = dict(report)
  report['direction_error_deg_fit'] = report.get('direction_error_deg', {})
  report['direction_error_deg'] = {
      name: info['error']
      for name, info in directions.items()
      if info['error'] is not None
  }
  report['direction_frame'] = measured['frame']
  if 'side_crossings' in measured:
    # The root-frame side check (no fit) replaces the alignment-frame one.
    report['side_crossings_fit'] = report.get('side_crossings', [])
    report['side_crossings'] = measured['side_crossings']
    report['pose_gate'] = fidelity.pose_gate(report['components'], report)
  return report, directions


# Not wrapped in `_preserve_3d_overlay`: that would run the search on the flat
# backbone, so every candidate was folded and measured without the posed 3-D
# relief (Pegasus fore_lower: chosen as "95 -> 80 deg" on the backbone, 64 ->
# 160 deg on the real fold). The candidate tools handle the overlay
# themselves (`simple_fold` is wrapped, the 3-D folds cut the folded state).
def bend_to_direction(
    state: ShapingState,
    *,
    element_name: str,
    az: float | None = None,
    el: float | None = None,
    measure: str | None = None,
    tools: Sequence[str] = _BEND_TOOLS,
    angles: Sequence[float] = (30.0, 45.0, 60.0, 90.0, 120.0),
    depths: Sequence[float] = (0.2, 0.4),
    tolerance: float = 15.0,
    max_candidates: int = 36,
    chosen: Mapping[str, Any] | None = None,
) -> ShapingState:
  """Bends a flap so a stick points at an absolute direction (measured).

  The bend tools (`crimp_fold_3d`, `reverse_fold_3d`, `simple_fold`) take an
  in-plane turn whose plane and sign depend on how the flap lies in the
  folded base (angle investigation: a requested 45 degrees moved limbs by
  14-21 degrees median, and mirrored limbs need opposite signs for the 3-D
  folds but the same sign for `simple_fold`). This tool tries them with both
  signs, several sizes and depths, folds every candidate, measures where the
  `measure` stick now points in the stick figure's frame, and keeps the
  candidate closest to (az, el), with a penalty when it moves other sticks
  away from their stick-figure directions. Works on narrowed limbs (unlike
  `rotation_3d`). The search has two phases: first every tool, sign and
  depth at the size closest to the needed turn (finds which combination
  turns the stick the right way), then the other sizes for the best one; it
  stops below `tolerance` degrees (at most about 16 folds by default).

  Args:
    state: Current state.
    element_name: Flap to bend (the cut goes through this flap).
    az: Target azimuth of the `measure` stick in the figure frame (x forward, y
      left, z up); default: its stick-figure azimuth.
    el: Target elevation; default: its stick-figure elevation.
    measure: Stick whose direction is targeted (default `element_name`).
    tools: Bend tools to try.
    angles: Turn magnitudes (degrees) to try, each with both signs.
    depths: Cut depths (fraction from the flap's base) to try.
    tolerance: Stop as soon as a candidate is within this many degrees.
    max_candidates: Upper bound on tool calls.
    chosen: `{tool, angle, depth}` from a previous call's history: apply it
      directly without searching (replay).

  Returns:
    The new state.

  Raises:
    ValueError: If no candidate folds or none points closer than before.
  """
  if element_name not in state.tree:
    raise ValueError(
        f'bend_to_direction: unknown element {element_name}; known:'
        f' {state.element_names()}'
    )
  if chosen is not None:
    if chosen.get('tool') not in _BEND_TOOLS:
      raise ValueError(f'chosen tool must be among {_BEND_TOOLS}.')
    replay: dict[str, Any] = {
        'element_name': element_name,
        'angle': float(chosen['angle']),
        'depth': float(chosen['depth']),
    }
    if chosen['tool'] != 'simple_fold':
      replay['verify_fold'] = False  # As in the search: same variant.
    return _TOOLS[chosen['tool']](state, **replay)
  measure = measure or element_name
  if measure not in state.tree:
    raise ValueError(f'bend_to_direction: unknown measure stick {measure}.')
  unknown_tools = [t for t in tools if t not in _BEND_TOOLS]
  if unknown_tools:
    raise ValueError(f'tools must be among {_BEND_TOOLS}; got {unknown_tools}.')
  stick = state.tree[measure]
  target = _unit_vector(
      stick.azimuth_angle if az is None else az,
      stick.elevation_angle if el is None else el,
  )

  def evaluate(candidate: ShapingState) -> tuple[float, float] | None:
    try:
      model = fidelity.FoldedModel.from_pattern(candidate.pattern)
      report, directions = measured_directions(candidate)
    except Exception:  # pylint: disable=broad-exception-caught
      return None
    if model.vertex_error > _MAX_RENDERABLE_VERTEX_ERROR:
      return None  # The fold stage refuses it (see open_flap).
    realized = directions.get(measure, {}).get('realized')
    if realized is None:
      return None
    error = _angle_between(_unit_vector(*realized), target)
    others = [
        e
        for name, e in report['direction_error_deg'].items()
        if name != measure and e is not None
    ]
    return error, (sum(others) / len(others) if others else 0.0)

  baseline = evaluate(state)
  if baseline is None:
    raise ValueError('bend_to_direction: the current pattern does not fold.')
  base_error, base_others = baseline
  if base_error <= tolerance:
    _note(f'{measure} already within {base_error:.0f} deg of the target.')
    return state
  best: tuple[float, float, ShapingState, str, float, float] | None = None
  tried = 0
  seen: set[tuple[str, float, float]] = set()

  def run(tool: str, depth: float, angle: float) -> bool:
    """Tries one candidate; returns True when it is good enough to stop."""
    nonlocal best, tried
    if (tool, depth, angle) in seen or tried >= max_candidates:
      return False
    seen.add((tool, depth, angle))
    tried += 1
    arguments: dict[str, Any] = {
        'element_name': element_name,
        'angle': angle,
        'depth': depth,
    }
    if tool != 'simple_fold':
      # Every candidate is folded and measured below; the tools' own fold
      # verification (and its depth fallbacks) would fold each one again.
      arguments['verify_fold'] = False
    # Candidates' own notes would bury the result in the message.
    notes_token = _NOTES.set([])
    try:
      candidate = _TOOLS[tool](state, **arguments)
    except (ValueError, AssertionError, KeyError, IndexError) as error:
      logging.info('bend_to_direction %s %s: %s', tool, angle, error)
      return False
    finally:
      _NOTES.reset(notes_token)
    scored = evaluate(candidate)
    if scored is None:
      return False
    error, others = scored
    cost = error + 0.5 * max(0.0, others - base_others)
    if best is None or cost < best[0]:
      best = (cost, error, candidate, tool, angle, depth)
    return error <= tolerance and others <= base_others + 5.0

  # Phase 1: which tool, sign and depth turn the stick the right way (one
  # size near the needed correction: 2 x len(tools) x len(depths) folds).
  first = min(angles, key=lambda a: abs(a - base_error))
  done = False
  for depth in depths:
    for tool in tools:
      for sign in (1.0, -1.0):
        if run(tool, depth, sign * first):
          done = True
          break
      if done:
        break
    if done:
      break
  # Phase 2: the other sizes for the best tool/sign/depth only.
  if not done and best is not None:
    tool, sign, depth = best[3], math.copysign(1.0, best[4]), best[5]
    for angle in sorted(angles, key=lambda a: abs(a - base_error)):
      if run(tool, depth, sign * angle):
        break
  if best is None or best[1] >= base_error:
    raise ValueError(
        f'bend_to_direction: no candidate brought {measure} closer than'
        f' {base_error:.0f} deg ({tried} tried); try other depths or bend the'
        ' parent element.'
    )
  assert best is not None
  _, error, candidate, tool, angle, depth = best  # pylint: disable=unpacking-non-sequence
  _note(
      f'{measure}: {base_error:.0f} -> {error:.0f} deg off target with'
      f' {tool}(angle={angle:g}, depth={depth:g}); {tried} candidates.'
  )
  _resolve_arguments({
      'element_name': element_name,
      'az': az,
      'el': el,
      'measure': measure,
      'chosen': {'tool': tool, 'angle': angle, 'depth': depth},
  })
  return candidate


# ------------------------------------------------------------------------------
# Dispatch.
# ------------------------------------------------------------------------------

_TOOLS = {
    'simple_fold': simple_fold,
    'simple_fold_heuristic': simple_fold_heuristic,
    'crimp_fold': crimp_fold,
    'reverse_fold': reverse_fold,
    'clip_pattern_fold': clip_pattern_fold,
    'rotation_3d': rotation_3d,
    'shape_angle_width': shape_angle_width,
    'thin_limbs_3d': thin_limbs_3d,
    'fan_out_fold': fan_out_fold,
    'fan_out_fold_heuristic': fan_out_fold_heuristic,
    'geometric_simple_fold': geometric_simple_fold,
    'squash_fold': squash_fold,
    'tree_edit': tree_edit,
    'level_shifter': level_shifter,
    'reverse_fold_3d': reverse_fold_3d,
    'crimp_fold_3d': crimp_fold_3d,
    'open_flap': open_flap,
    'inscribed_flap': inscribed_flap,
    'pleat_to_hide': pleat_to_hide,
    'simple_fold_narrow': simple_fold_narrow,
    'narrow_central_flap': narrow_central_flap,
    'curve_flap': curve_flap,
    'apply_symmetric': apply_symmetric,
    'fold_spine_3d': fold_spine_3d,
    'spread_fork': spread_fork,
    'radial_fan': radial_fan,
    'open_river_3d': open_river_3d,
    'bend_to_direction': bend_to_direction,
}


def normalize_tool_name(name: str) -> str:
  """Strips the RL `shaping:`/`tree:` prefixes from a tool name."""
  return name.split(':', 1)[-1]


def _describe_error(error: BaseException) -> str:
  """Describes a tool failure, locating message-less exceptions.

  Library assertions often carry no message; the agent then only sees
  "failed: ". Those are reported with the exception type and the innermost
  frame (file, line and source) of the shaping library instead.

  Args:
    error: The caught exception.

  Returns:
    A one-line description.
  """
  message = str(error).strip()
  if message and not isinstance(error, (AssertionError, KeyError, IndexError)):
    return message
  frames = traceback.extract_tb(error.__traceback__)
  location = ''
  for frame in reversed(frames):
    if 'bp_pipeline' in frame.filename:
      source = (frame.line or '').strip()
      location = f' at {os.path.basename(frame.filename)}:{frame.lineno}' + (
          f' (`{source}`)' if source else ''
      )
      break
  detail = f': {message}' if message else ''
  return f'{type(error).__name__}{detail}{location}'


def apply_tool(
    state: ShapingState, name: str, arguments: Mapping[str, Any] | None = None
) -> tuple[ShapingState, ToolResult]:
  """Applies one tool by name, recording the outcome in the state history.

  Unlike the tool functions, this never raises for tool failures: the returned
  state is unchanged (apart from history) and the result explains the error.

  Args:
    state: Current state.
    name: Tool name, optionally prefixed with `shaping:`/`tree:`.
    arguments: Keyword arguments for the tool.

  Returns:
    The new state and the tool result.
  """
  arguments = dict(arguments or {})
  tool_name = normalize_tool_name(name)
  tool = _TOOLS.get(tool_name)
  if tool is None:
    result = ToolResult(
        tool=tool_name,
        arguments=arguments,
        ok=False,
        message=f'Unknown tool {name!r}; available: {", ".join(TOOL_NAMES)}',
        num_creases=len(state.pattern.creases),
    )
    return _with_history(state, result), result
  notes: list[str] = []
  notes_token = _NOTES.set(notes)
  resolved: dict[str, Any] = {}
  resolved_token = _RESOLVED_ARGUMENTS.set(resolved)
  try:
    new_state = tool(state, **arguments)
  except (ValueError, AssertionError, KeyError, IndexError, TypeError) as error:
    logging.exception('Tool %s failed with %s', tool_name, arguments)
    result = ToolResult(
        tool=tool_name,
        arguments=arguments,
        ok=False,
        message=f'{tool_name} failed: {_describe_error(error)}',
        num_creases=len(state.pattern.creases),
    )
    return _with_history(state, result), result
  finally:
    _NOTES.reset(notes_token)
    _RESOLVED_ARGUMENTS.reset(resolved_token)
  result = ToolResult(
      tool=tool_name,
      arguments=resolved or arguments,
      ok=True,
      message=' '.join([f'{tool_name} succeeded.'] + notes),
      num_creases=len(new_state.pattern.creases),
  )
  return _with_history(new_state, result), result


def apply_tools(
    state: ShapingState, calls: Sequence[Mapping[str, Any]]
) -> tuple[ShapingState, list[ToolResult]]:
  """Applies a list of `{"name": ..., "arguments": {...}}` calls in order."""
  results = []
  for call in calls:
    state, result = apply_tool(state, call['name'], call.get('arguments'))
    results.append(result)
  return state, results


def _with_history(state: ShapingState, result: ToolResult) -> ShapingState:
  return state.replace(history=state.history + (result.asdict(),))


def history_json(state: ShapingState) -> str:
  """The tool-call history as JSON lines."""
  return '\n'.join(json.dumps(entry, default=str) for entry in state.history)


# ------------------------------------------------------------------------------
# Probing: which basic folds work on which elements of the current pattern.
# ------------------------------------------------------------------------------

_PROBE_CALLS = (
    ('simple_fold', {'angle': 45.0, 'depth': 0.0}),
    ('simple_fold', {'angle': -45.0, 'depth': 0.0}),
    ('simple_fold', {'angle': 90.0, 'depth': 0.0}),
    ('simple_fold', {'angle': 90.0, 'depth': 0.0, 'narrow': True}),
    # Straight narrowing (no bend) with the drawing-machine template: works on
    # flaps and rivers, 1/2 or 1/3 of the width, from the base (`step`=0) or
    # only the distal part (`step` >= 1).
    (
        'clip_pattern_fold',
        {'template_name': 'fold', 'enable_narrow': True, 'narrow_n_times': 2},
    ),
    (
        'clip_pattern_fold',
        {'template_name': 'fold', 'enable_narrow': True, 'narrow_n_times': 3},
    ),
    ('crimp_fold', {'angle': 90.0, 'depth': 0.25}),
    ('reverse_fold', {'angle': 45.0, 'orientation': 'horizontal'}),
    ('squash_fold', {'open_percent': 0.5}),
    ('reverse_fold_3d', {'angle': 45.0, 'depth': 0.5}),
    ('crimp_fold_3d', {'angle': 60.0, 'depth': 0.4}),
    ('inscribed_flap', {'depths': [1], 'fold_percent': 0.5}),
    ('open_flap', {'fold_percent': 0.3}),
)


def probe(
    state: ShapingState, element_names: Sequence[str] | None = None
) -> dict[str, dict[str, str]]:
  """Tries a set of basic folds on each element from the current pattern.

  Each probe starts from `state` (probes are not chained) and the state is not
  modified. Useful before planning a shaping sequence, and for reporting which
  tools misbehave on a given base.

  Args:
    state: Current state.
    element_names: Elements to probe; all elements when None.

  Returns:
    `{element: {"tool(args)": "ok" | "failed: reason"}}`.
  """
  names = list(element_names) if element_names else state.element_names()
  report: dict[str, dict[str, str]] = {}
  for name in names:
    outcomes = {}
    for tool_name, arguments in _PROBE_CALLS:
      call = dict(arguments, element_name=name)
      _, result = apply_tool(state, tool_name, call)
      label = ', '.join(f'{k}={v}' for k, v in arguments.items())
      key = f'{tool_name}({label})'
      outcomes[key] = 'ok' if result.ok else f'failed: {result.message[:120]}'
    report[name] = outcomes
  return report


def list_shaping_tools() -> tuple[str, ...]:
  """Returns the tuple of available shaping actuator names."""
  return TOOL_NAMES
