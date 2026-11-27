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

"""End-to-end packing and solving pipeline for Box Pleating."""

from __future__ import annotations

import collections
from collections.abc import Callable, Iterator, Sequence
import contextvars
import copy
import dataclasses
import datetime
import functools
import itertools
import json
import sys
import time
import traceback
from typing import Any, Self

from absl import logging

from . import bus
from . import codec
from . import timeout
from .crease_pattern import crease
from .crease_pattern import crease_pattern
from .crease_pattern import visualize
from .crease_pattern.parser import svg as svg_parser
from .packing import config as packing_config_lib
from .packing import grid_size as grid_size_lib
from .packing import packing_output as packing_output_lib
from .packing import shaping_flap_templates
from .packing import utils_packing_flaps
from .packing import utils_solver
from .shaping import bus_adapter
from .shaping import geometric_fold_3d_heuristics
from .shaping import heuristics_3d
from .solving import central_fold_utils
from .solving import hinges as hinge_utils
from .solving import level_shifters as level_shifters_lib
from .solving import solvability
from .stick_figures import shaping_templates
from .stick_figures import stick_figure
from .stick_figures import visualize as stick_figure_visualize

_FoldStrategy3D = heuristics_3d.FoldStrategy3D

_SVG_N_DIGITS = 8
_FIGURE_SIZE = 8
# Reserve ~45% of outer timeout budget for the Pass 2 recovery sweep.
_PASS_1_BUDGET_FRACTION = 0.55

cp_to_svg = functools.partial(
    svg_parser.to_svg, pretty=True, replace_border=True, ndigits=_SVG_N_DIGITS
)


def _cp_to_svg_merged(cp: crease_pattern.CreasePattern, **kwargs) -> str:
  """Returns an SVG string for a crease pattern with merged creases."""
  merged = crease.merge_creases(cp.creases)
  merged_cp = crease_pattern.CreasePattern(merged, light_mode=True)
  return cp_to_svg(merged_cp, **kwargs)


def _infer_grid_from_creases(creases: Sequence[Any]) -> int:
  """Infers integer grid size from crease coordinates."""
  if not creases:
    return 0
  try:
    max_coord = max(max(c.p1.x, c.p1.y, c.p2.x, c.p2.y) for c in creases)
    if max_coord > 1.5:
      return int(round(max_coord))
    vals = set()
    for c in creases:
      for v in (c.p1.x, c.p1.y, c.p2.x, c.p2.y):
        if v > 1e-6:
          vals.add(v)
    if not vals:
      return 0
    for g in range(1, 129):
      if all(abs(v * g - round(v * g)) < 1e-4 for v in vals):
        return g
  except (TypeError, ValueError, AttributeError):
    pass
  return 0


def _safe_grid_size(cp: Any) -> int:
  """Returns integer grid_size from a CreasePattern or PackingOutput safely."""
  if cp is None:
    return 0
  if hasattr(cp, 'details') and cp.details is not None:
    d = cp.details
    g = getattr(d, 'grid_size', None)
    if g is None and isinstance(d, dict):
      g = d.get('grid_size')
    if g is not None and g > 0:
      return int(g)
  if hasattr(cp, 'packing_cp'):
    cp = cp.packing_cp
    if cp is None:
      return 0
  g = getattr(cp, 'grid_size', None)
  if g is not None and g > 0:
    return int(g)
  if hasattr(cp, 'creases') and cp.creases:
    return _infer_grid_from_creases(cp.creases)
  return 0


def get_stick_figure_visualization(tree: stick_figure.StickFigure) -> bytes:
  """Returns a 2D visualization for a stick figure."""
  from matplotlib import pyplot as plt  # pylint: disable=g-import-not-at-top

  figure = stick_figure_visualize.visualize_stick_from_all_views(
      tree=tree, figsize=_FIGURE_SIZE
  )
  image_bytes = visualize.figure_to_image(figure=figure, file_format='png')
  plt.close(figure)
  return image_bytes


def _check_nested_field(obj: Any, dotted_key: str) -> bool:
  """Checks whether a potentially dotted field name exists on a dataclass."""
  parts = dotted_key.split('.')
  curr = obj
  for part in parts:
    if not dataclasses.is_dataclass(curr) or not hasattr(curr, part):
      return False
    curr = getattr(curr, part)
  return True


def _replace_single_field(obj: Any, dotted_key: str, value: Any) -> Any:
  """Replaces a potentially nested dataclass field by dotted key."""
  if '.' not in dotted_key:
    return dataclasses.replace(obj, **{dotted_key: value})
  head, tail = dotted_key.split('.', 1)
  sub_obj = getattr(obj, head)
  return dataclasses.replace(
      obj, **{head: _replace_single_field(sub_obj, tail, value)}
  )


def _replace_nested_fields(obj: Any, **params: Any) -> Any:
  """Replaces multiple potentially nested dataclass fields."""
  result = obj
  for k, v in params.items():
    result = _replace_single_field(result, k, v)
  return result


@dataclasses.dataclass(slots=True, kw_only=True)
class TaskConfig:
  """Common config shared by all tasks."""

  timeout_seconds: float | None = 3600
  use_sweep: bool = False
  sweep_hparams: dict[str, Sequence[Any]] = dataclasses.field(
      default_factory=dict
  )
  sweep_timeout_seconds: float | None = None

  def config_iter(self) -> Iterator[Self]:
    if not self.use_sweep or not self.sweep_hparams:
      param_iter = [{}]
    else:
      keys = list(self.sweep_hparams.keys())
      values_product = itertools.product(*(self.sweep_hparams[k] for k in keys))
      param_iter = [dict(zip(keys, vals)) for vals in values_product]
    for params in param_iter:
      yield _replace_nested_fields(self, **params, sweep_hparams={})

  def __post_init__(self):
    if self.timeout_seconds is not None and self.timeout_seconds <= 0:
      raise ValueError('timeout_seconds must be positive.')

    missing_fields = [
        k for k in self.sweep_hparams.keys() if not _check_nested_field(self, k)
    ]
    if missing_fields:
      raise ValueError(
          f'sweep_hparams keys must be valid config fields: {missing_fields}'
      )


@dataclasses.dataclass(slots=True, kw_only=True)
class PackingTaskConfig(TaskConfig):
  """Config for CREASE_PATTERN_PACKING task."""

  extra_grid_size: int = 0
  allow_paired_axis_placement: bool = False
  max_cps_per_grid_size: int = 2
  max_number_of_cps: int = 1
  packing_config: packing_config_lib.PackingConfig = dataclasses.field(
      default_factory=packing_config_lib.PackingConfig
  )
  grid_mode: stick_figure.GridMode = 'v4'  # Backtracking start grid mode.
  max_retries: int = 30
  packer: str = 'default'
  packer_config: Any | None = None

  use_sweep: bool = True
  sweep_hparams: dict[str, Sequence[Any]] = dataclasses.field(
      default_factory=dict
  )
  timeout_seconds: float | None = None
  sweep_timeout_seconds: float | None = 119

  @property
  def extra_grid_sizes(self) -> Sequence[int]:
    """Returns candidate extra_grid_size values for packing sweep."""
    if 'extra_grid_size' in self.sweep_hparams:
      return self.sweep_hparams['extra_grid_size']
    return range(self.extra_grid_size, self.extra_grid_size + self.max_retries)

  def config_iter(self) -> Iterator[Self]:
    if not self.use_sweep:
      yield _replace_nested_fields(self, sweep_hparams={})
      return
    effective_hparams = dict(self.sweep_hparams)
    effective_hparams.setdefault('extra_grid_size', self.extra_grid_sizes)
    keys = list(effective_hparams.keys())
    values_product = itertools.product(*(effective_hparams[k] for k in keys))
    for vals in values_product:
      params = dict(zip(keys, vals))
      yield _replace_nested_fields(self, **params, sweep_hparams={})

  @classmethod
  def default(cls, **overrides: Any) -> Self:
    """Returns a default PackingTaskConfig."""
    return cls(**overrides)

  @classmethod
  def reference(cls, **overrides: Any) -> Self:
    """Returns a PackingTaskConfig matching baseline reference behavior."""
    defaults: dict[str, Any] = {
        'grid_mode': 'v2',
        'packing_config': packing_config_lib.PackingConfig(
            river_backtracking=False,
            legacy_iteration_order=True,
            reject_degenerate_hugging_rivers=False,
            star_hub_rivers=False,
            symmetry_line_obstacles=False,
            prune_partial_border_overhang=False,
            stretch_intermediate_pocket_flaps=False,
            memoize_pocket_tiling=False,
            reject_provably_unsolvable_packings=False,
            recovery=False,
        ),
    }
    defaults.update(overrides)
    return cls(**defaults)

  @classmethod
  def with_recovery(cls, **overrides: Any) -> Self:
    """Returns a PackingTaskConfig with river backtracking recovery enabled."""
    defaults: dict[str, Any] = {
        'packing_config': packing_config_lib.PackingConfig(
            river_backtracking=True,
            legacy_iteration_order=False,
            reject_degenerate_hugging_rivers=True,
            star_hub_rivers=True,
            symmetry_line_obstacles=False,
            prune_partial_border_overhang=False,
            stretch_intermediate_pocket_flaps=False,
            memoize_pocket_tiling=True,
            reject_provably_unsolvable_packings=False,
            recovery=True,
        ),
    }
    defaults.update(overrides)
    return cls(**defaults)


@dataclasses.dataclass(slots=True, kw_only=True)
class SolvingTaskConfig(TaskConfig):
  """Config for SOLVE_FLAT_FOLDABLE_CREASE_PATTERN task."""

  max_steps: int = 8000
  max_number_of_cps: int = 3
  check_local_flat_foldability: bool = True
  check_global_flat_foldability: bool = True
  use_python_flat_folder: bool = True
  tie_break_by_hinge_symmetry: bool = True
  remove_central_crease: bool = False
  use_partition_order: bool = True
  disallow_trapping_edge_flaps: bool = True
  hinges_from_stick_figure: bool = False
  scale_factor: int = 1
  allow_misaligned_symmetric_hinges: bool = False

  use_sweep: bool = True
  sweep_hparams: dict[str, Sequence[Any]] = dataclasses.field(
      default_factory=lambda: {
          'hinges_from_stick_figure': [True, False],
          'remove_central_crease': [True, False],
      }
  )
  sweep_timeout_seconds: float | None = 300
  timeout_seconds: float | None = None
  up_down_method: str = 'flaps_down'
  stop_on_first_solve: bool = False
  enable_recovery: bool = False
  enable_fallback_configs: bool = False
  enable_lazy_repack: bool = False
  enable_timeout_retries: bool = False
  max_branching_factor: int | None = None
  solver: str = 'default'
  num_workers: int = 1

  @classmethod
  def default(cls, **overrides: Any) -> Self:
    """Returns a default SolvingTaskConfig matching baseline reference behavior."""
    return cls(**overrides)

  @classmethod
  def reference(cls, **overrides: Any) -> Self:
    """Returns a SolvingTaskConfig matching baseline reference behavior."""
    defaults: dict[str, Any] = {
        'stop_on_first_solve': False,
        'enable_recovery': False,
        'enable_fallback_configs': False,
        'enable_lazy_repack': False,
        'enable_timeout_retries': False,
    }
    defaults.update(overrides)
    return cls(**defaults)

  @classmethod
  def with_recovery(cls, **overrides: Any) -> Self:
    """Returns a SolvingTaskConfig with multi-stage solver recovery enabled."""
    defaults: dict[str, Any] = {
        'stop_on_first_solve': True,
        'enable_recovery': True,
        'enable_fallback_configs': True,
        'enable_lazy_repack': True,
        'enable_timeout_retries': True,
    }
    defaults.update(overrides)
    return cls(**defaults)

  def config_iter(self) -> Iterator[Self]:
    if not self.use_sweep:
      yield _replace_nested_fields(self, sweep_hparams={})
      return
    if not self.enable_recovery:
      # Baseline reference sweep: 6 configs:
      # (hinges_from_stick_figure in [True, False]) x
      # (remove_central_crease in [True, False])
      # where hinges_from_sf=True yields ['azimuth', 'flaps_down'] and False
      # yields ['flaps_down'].
      for hinges_from_sf in [True, False]:
        for remove_cc in [True, False]:
          methods = (
              ['azimuth', 'flaps_down'] if hinges_from_sf else ['flaps_down']
          )
          for method in methods:
            yield _replace_nested_fields(
                self,
                sweep_hparams={},
                remove_central_crease=remove_cc,
                hinges_from_stick_figure=hinges_from_sf,
                up_down_method=method,
                disallow_trapping_edge_flaps=self.disallow_trapping_edge_flaps,
                allow_misaligned_symmetric_hinges=self.allow_misaligned_symmetric_hinges,
            )
      return
    base_solver_cfg = self.to_solver_config()
    for s_cfg in solvability.solve_cascade(base_solver_cfg, stage='primary'):
      yield _replace_nested_fields(
          self,
          sweep_hparams={},
          remove_central_crease=s_cfg.remove_central_crease,
          hinges_from_stick_figure=s_cfg.hinges_from_stick_figure,
          up_down_method=s_cfg.up_down_method,
          disallow_trapping_edge_flaps=s_cfg.disallow_trapping_edge_flaps,
          allow_misaligned_symmetric_hinges=s_cfg.allow_misaligned_symmetric_hinges,
      )

  def fallback_config_iter(self) -> Iterator[Self]:
    """Yields fallback solver configurations when primary sweep finds no CP."""
    if (
        not self.use_sweep
        or not self.enable_recovery
        or not self.enable_fallback_configs
    ):
      return
    base_solver_cfg = self.to_solver_config()
    for idx, s_cfg in enumerate(
        solvability.solve_cascade(base_solver_cfg, stage='fallback')
    ):
      scale = 1 if idx < 4 else 2
      yield _replace_nested_fields(
          self,
          use_sweep=False,
          sweep_hparams={},
          remove_central_crease=s_cfg.remove_central_crease,
          hinges_from_stick_figure=s_cfg.hinges_from_stick_figure,
          up_down_method=s_cfg.up_down_method,
          disallow_trapping_edge_flaps=s_cfg.disallow_trapping_edge_flaps,
          allow_misaligned_symmetric_hinges=s_cfg.allow_misaligned_symmetric_hinges,
          scale_factor=scale,
          max_steps=s_cfg.max_steps,
      )

  def to_solver_config(
      self,
      *,
      solution_callback: Callable[[Any], Any] = lambda x: x,
  ) -> hinge_utils.SolverConfig:
    """Converts SolvingTaskConfig to a SolverConfig instance."""
    max_time_limit = float(
        self.sweep_timeout_seconds
        if self.sweep_timeout_seconds is not None
        else 300.0
    )
    effective_max_cps = (
        1 if self.stop_on_first_solve else self.max_number_of_cps
    )
    effective_max_attempts_per_cp = (
        100 * max(1, self.max_number_of_cps)
        if self.stop_on_first_solve
        else 100
    )
    return hinge_utils.SolverConfig(
        max_steps=self.max_steps if self.max_steps is not None else 8000,
        max_time_limit=max_time_limit,
        max_number_of_cps=effective_max_cps,
        max_attempts_per_cp=effective_max_attempts_per_cp,
        check_local_flat_foldability=self.check_local_flat_foldability,
        check_global_flat_foldability=self.check_global_flat_foldability,
        use_python_flat_folder=self.use_python_flat_folder,
        tie_break_by_hinge_symmetry=self.tie_break_by_hinge_symmetry,
        remove_central_crease=self.remove_central_crease,
        use_partition_order=self.use_partition_order,
        disallow_trapping_edge_flaps=self.disallow_trapping_edge_flaps,
        hinges_from_stick_figure=self.hinges_from_stick_figure,
        sort_solutions=True,
        solution_callback=solution_callback,
        allow_misaligned_symmetric_hinges=self.allow_misaligned_symmetric_hinges,
        up_down_method=self.up_down_method,
        allow_constraint_relaxation=self.enable_recovery,
        recovery=self.enable_recovery,
        max_branching_factor=self.max_branching_factor,
        solver=self.solver,
        num_workers=self.num_workers,
    )

  def __post_init__(self):
    TaskConfig.__post_init__(self)
    if self.scale_factor <= 0:
      raise ValueError('scale_factor must be a positive number.')


@dataclasses.dataclass(slots=True, kw_only=True)
class ShapingTaskConfig(TaskConfig):
  """Config for 3D geometric SHAPING task."""

  apply_simple_fold_shaping: bool = True
  fold_strategy_3d: Any = _FoldStrategy3D.DEFAULT
  shape_with_unmerged_tree: bool = True
  filter_by_local_flat_foldability: bool = True
  clip_slope_thresh: float = 3.0
  max_slope_thresh: float = 10.0
  max_number_of_cps: int = 8
  cp_tolerance: float = 0.001
  enable_strategy_fallback: bool = False

  @classmethod
  def default(cls, **overrides: Any) -> Self:
    """Returns a default ShapingTaskConfig."""
    return cls(**overrides)

  @classmethod
  def with_strategy_sweep(cls, **overrides: Any) -> Self:
    """Returns a ShapingTaskConfig sweeping 3D geometric fold strategies."""
    defaults: dict[str, Any] = {
        'use_sweep': True,
        'sweep_hparams': {
            'fold_strategy_3d': [
                _FoldStrategy3D.DEFAULT,
                _FoldStrategy3D.ROTATE_BEND,
                _FoldStrategy3D.SHALLOW_FOLDS,
            ],
        },
    }
    defaults.update(overrides)
    return cls(**defaults)


@dataclasses.dataclass(slots=True, kw_only=True)
class PipelineConfig:
  """Unified configuration for the full box pleating pipeline."""

  packing: PackingTaskConfig = dataclasses.field(
      default_factory=PackingTaskConfig
  )
  solving: SolvingTaskConfig = dataclasses.field(
      default_factory=SolvingTaskConfig
  )
  shaping: ShapingTaskConfig = dataclasses.field(
      default_factory=ShapingTaskConfig
  )

  @classmethod
  def default(cls, **overrides: Any) -> Self:
    """Returns the default pipeline config with balanced settings."""
    return cls(**overrides)

  @classmethod
  def reference(cls, **overrides: Any) -> Self:
    """Returns pipeline configuration matching baseline reference behavior."""
    defaults = {
        'packing': PackingTaskConfig.reference(),
        'solving': SolvingTaskConfig.reference(),
        'shaping': ShapingTaskConfig.default(),
    }
    defaults.update(overrides)
    return cls(**defaults)

  @classmethod
  def with_recovery(cls, **overrides: Any) -> Self:
    """Returns pipeline configuration with all multi-stage recovery enabled."""
    defaults = {
        'packing': PackingTaskConfig.with_recovery(),
        'solving': SolvingTaskConfig.with_recovery(),
        'shaping': ShapingTaskConfig.with_strategy_sweep(),
    }
    defaults.update(overrides)
    return cls(**defaults)


@dataclasses.dataclass(kw_only=True)
class TaskSpecificInput:
  """Parent dataclass for task-specific inputs and results."""

  name: str
  non_serializable_fields: Sequence[str] = ()
  serializable_properties: Sequence[str] = ()

  def asdict(self) -> dict[str, Any]:
    """Returns a dict representation, omitting non-serializable objects."""
    result = {}
    for field in dataclasses.fields(self):
      field_name = field.name
      if field_name in self.non_serializable_fields:
        continue
      if field_name in ('non_serializable_fields', 'serializable_properties'):
        continue
      value = getattr(self, field_name)
      if isinstance(value, list) and value and hasattr(value[0], 'asdict'):
        result[field_name] = [item.asdict() for item in value]
      elif hasattr(value, 'asdict'):
        result[field_name] = value.asdict()
      else:
        result[field_name] = value

    for prop in self.serializable_properties:
      result[prop] = getattr(self, prop)
    return result


@dataclasses.dataclass(kw_only=True)
class StickFigureSample(TaskSpecificInput):
  """A stick figure object."""

  stick_figure: stick_figure.StickFigure
  stick_figure_json: str
  name: str = ''

  non_serializable_fields: Sequence[str] = ('stick_figure',)
  serializable_properties: Sequence[str] = (
      'stick_figure_2d_png',
      'stick_figure_3d_png',
  )

  def __post_init__(self):
    if not self.name:
      self.name = self.stick_figure.name

  @functools.cached_property
  def stick_figure_2d_png(self) -> bytes:
    """Generates a 2D visualization for a stick figure."""
    return get_stick_figure_visualization(self.stick_figure)

  @functools.cached_property
  def stick_figure_3d_png(self) -> bytes:
    """Generates a 3D visualization for a stick figure."""
    return b''


@dataclasses.dataclass(slots=True, kw_only=True)
class StickFigureInput(TaskSpecificInput):
  """Result for STICK_FIGURE_GENERATION task."""

  category: str
  complexity: int
  is_symmetric_tree: bool
  unmerged_stick_figure: StickFigureSample
  merged_stick_figure: StickFigureSample
  merged_stick_name_dict: dict[str, list[str]]
  has_river: bool
  max_grid_size: int | None = 0

  non_serializable_fields: Sequence[str] = ('merged_stick_name_dict',)


@dataclasses.dataclass(kw_only=True)
class PackingResult(TaskSpecificInput):
  """Result for PACK_CREASE_PATTERN task."""

  packing_cps: list[crease_pattern.CreasePattern]
  packing_outputs: list[packing_output_lib.PackingOutput]
  attempted_grid_sizes: list[int]
  time_spent_per_grid_sizes: list[float]
  sequence_status: str
  duration_seconds: float
  max_grid_size_reached: bool = False
  errors_per_retry: list[str] = dataclasses.field(default_factory=list)
  stick_figure_input: StickFigureInput | None = None
  packing_config: PackingTaskConfig | None = None

  non_serializable_fields: Sequence[str] = (
      'packing_cps',
      'stick_figure_input',
      'packing_config',
  )
  serializable_properties: Sequence[str] = (
      'packing_cp_png',
      'packing_cp_svg',
      'packing_cp_highlighted_png',
  )

  def with_promoted_packing(
      self,
      winning_out: packing_output_lib.PackingOutput,
  ) -> Self:
    """Returns a copy with `winning_out` promoted to index 0."""
    if self.packing_outputs and self.packing_outputs[0] is winning_out:
      return self
    promoted_outputs = [
        winning_out,
        *[o for o in self.packing_outputs if o is not winning_out],
    ]
    promoted_cps = [
        o.packing_cp for o in promoted_outputs if o.packing_cp is not None
    ]
    return dataclasses.replace(
        self,
        packing_outputs=promoted_outputs,
        packing_cps=promoted_cps,
    )

  @property
  def packing_cp_png(self) -> bytes:
    """Returns the packing crease-patterns as a PNG image."""
    return visualize.render_cps_to_png_bytes(self.packing_cps)

  @property
  def packing_cp_svg(self) -> list[str]:
    """Returns the packing crease-patterns as SVG strings."""
    svgs = [
        _cp_to_svg_merged(cp, grid=_safe_grid_size(cp))
        for cp in self.packing_cps
    ]
    if (
        svgs
        and self.packing_outputs
        and any('xm_meta_worker' in k for k in sys.modules)
    ):
      try:
        unmerged_sf = self.packing_outputs[0].unmerged_stick_figure
        design = (
            bus.DesignTree.from_stick_figure(unmerged_sf)
            if unmerged_sf is not None
            else None
        )
        packed_bus = bus.PackedCreasePattern.from_packing_output(
            self.packing_outputs[0], design=design
        )
        packed_json = json.dumps(codec.to_dict(packed_bus), sort_keys=True)
        svgs[0] = f'{svgs[0]}\n<!-- bp_packed_json: {packed_json} -->'
      except Exception:  # pylint: disable=broad-exception-caught
        pass
    return svgs

  @property
  def packing_cp_highlighted_png(self) -> bytes:
    """Returns the packing crease-patterns as a highlighted PNG image."""
    return visualize.render_highlighted_cps_to_png_bytes(
        self.packing_cps, self.packing_outputs
    )


@dataclasses.dataclass(kw_only=True)
class SolveResult(TaskSpecificInput):
  """Result for SOLVE_FLAT_FOLDABLE_CREASE_PATTERN task."""

  sequence_status: str
  flat_foldable_cps_count: int
  flat_foldable_cps: list[crease_pattern.CreasePattern]
  scaling_factors_used: list[int]
  solver_outputs: list[hinge_utils.SolverResult]
  packing_outputs: list[packing_output_lib.PackingOutput]
  time_taken: list[float]
  duration_seconds: float
  avg_time_per_cp_seconds: float
  solver_errors: list[str] = dataclasses.field(default_factory=list)
  solver_states: list[hinge_utils.State]

  non_serializable_fields: Sequence[str] = (
      'flat_foldable_cps',
      'solver_outputs',
      'packing_outputs',
  )
  serializable_properties: Sequence[str] = (
      'flat_foldable_cps_png',
      'flat_foldable_cps_svg',
  )

  @functools.cached_property
  def flat_foldable_cps_png(self) -> bytes:
    """PNG image of all flat-foldable crease-patterns found for the task."""
    return visualize.render_cps_to_png_bytes(self.flat_foldable_cps)

  @functools.cached_property
  def flat_foldable_cps_svg(self) -> list[str]:
    """List of flat-foldable crease-patterns found for the task as SVGs."""
    svgs = [
        _cp_to_svg_merged(cp, grid=_safe_grid_size(cp))
        for cp in self.flat_foldable_cps
    ]
    if (
        svgs
        and self.solver_outputs
        and self.packing_outputs
        and any('xm_meta_worker' in k for k in sys.modules)
    ):
      try:
        unmerged_sf = self.packing_outputs[0].unmerged_stick_figure
        design = (
            bus.DesignTree.from_stick_figure(unmerged_sf)
            if unmerged_sf is not None
            else None
        )
        packed_bus = bus.PackedCreasePattern.from_packing_output(
            self.packing_outputs[0], design=design
        )
        scale_factor = (
            self.scaling_factors_used[0] if self.scaling_factors_used else 1
        )
        solved_bus = bus.SolvedCreasePattern.from_solver_result(
            self.solver_outputs[0],
            packed=packed_bus,
            scale_factor=scale_factor,
        )
        solved_json = json.dumps(codec.to_dict(solved_bus), sort_keys=True)
        svgs[0] = f'{svgs[0]}\n<!-- bp_solved_json: {solved_json} -->'
      except Exception:  # pylint: disable=broad-exception-caught
        pass
    return svgs


@dataclasses.dataclass(slots=True, kw_only=True)
class ShapingInstance:
  """A single instance for shaping."""

  tree: stick_figure.StickFigure
  packing: packing_output_lib.PackingOutput
  solving: hinge_utils.SolverResult


@dataclasses.dataclass(slots=True, kw_only=True)
class ShapingTaskInput(TaskSpecificInput):
  """Input for SHAPING task."""

  shaping_instances: list[ShapingInstance]
  non_serializable_fields: Sequence[str] = ('shaping_instances',)


@dataclasses.dataclass(kw_only=True)
class ShapingTaskResult(TaskSpecificInput):
  """Result for SHAPING task."""

  sequence_status: str
  shaped_cps: list[crease_pattern.CreasePattern] = dataclasses.field(
      default_factory=list
  )
  solving_indices: list[int]
  shaping_indices: list[int]
  time_taken: list[float]
  duration_seconds: float
  avg_time_per_cp_seconds: float
  shaping_errors: list[str] = dataclasses.field(default_factory=list)
  tree_similarity_scores: list[float] = dataclasses.field(default_factory=list)
  strategies_used: list[str] = dataclasses.field(default_factory=list)

  non_serializable_fields: Sequence[str] = ('shaped_cps', 'strategies_used')
  serializable_properties: Sequence[str] = ('shaped_cps_png', 'shaped_cps_svg')

  @functools.cached_property
  def shaped_cps_png(self) -> bytes:
    """Returns the shaped crease-patterns as a PNG image."""
    return visualize.render_cps_to_png_bytes(self.shaped_cps)

  @functools.cached_property
  def shaped_cps_svg(self) -> list[str]:
    """Returns the shaped crease-patterns as SVG strings."""
    return [_cp_to_svg_merged(cp, grid=cp.grid_size) for cp in self.shaped_cps]


ShapingResult = ShapingTaskResult


def identify_stick_figures_with_rivers(
    task_input: stick_figure.StickFigure,
) -> bool:
  """Identifies whether a stick figure contains rivers."""
  return any(s.is_river for s in task_input.children)


def process_stick_figure(
    tree: stick_figure.StickFigure | bus.DesignTree,
) -> StickFigureInput:
  """Processes a single stick figure into a `StickFigureInput`."""
  design_tree = (
      tree
      if isinstance(tree, bus.DesignTree)
      else bus.DesignTree.from_stick_figure(tree)
  )
  merged_stick_name_dict = {
      merged_name: list(source_names)
      for merged_name, source_names in design_tree.merged_from.items()
      if len(source_names) > 1 or merged_name != source_names[0]
  }
  return StickFigureInput(
      name=design_tree.source.name,
      category=design_tree.merged.category,
      complexity=design_tree.merged.complexity,
      is_symmetric_tree=design_tree.is_symmetric,
      max_grid_size=design_tree.max_grid_size,
      unmerged_stick_figure=StickFigureSample(
          stick_figure=design_tree.source,
          stick_figure_json=design_tree.source.to_json(),
      ),
      merged_stick_name_dict=merged_stick_name_dict,
      merged_stick_figure=StickFigureSample(
          stick_figure=design_tree.merged,
          stick_figure_json=design_tree.merged.to_json(),
      ),
      has_river=identify_stick_figures_with_rivers(design_tree.merged),
  )


def _attempt_packing(
    *,
    stick_figure_input: StickFigureInput,
    grid_size: int,
    is_symmetric: bool,
    allow_paired_axis_placement: bool,
    max_solutions: int,
    config: packing_config_lib.PackingConfig,
    solution_callback: Callable[
        [packing_output_lib.PackingOutput],
        packing_output_lib.PackingOutput | None,
    ] = lambda x: x,
    packer: str = 'default',
    packer_config: Any | None = None,
) -> list[packing_output_lib.PackingOutput]:
  """Attempts to pack the stick figure with the given grid size."""
  timeout.maybe_timeout_cooperatively()

  if packer not in ('default', 'builtin'):
    from . import plugins  # pylint: disable=g-import-not-at-top

    grid_fn = plugins.get_grid_packer(packer)
    if grid_fn is None:
      raise ValueError(f'Unknown or unregistered grid packer {packer!r}')
    return grid_fn(
        stick_figure_input=stick_figure_input,
        grid_size=grid_size,
        is_symmetric=is_symmetric,
        allow_paired_axis_placement=allow_paired_axis_placement,
        max_solutions=max_solutions,
        config=config,
        solution_callback=solution_callback,
        packer_config=packer_config,
    )

  try:
    logging.info(
        'Trying to pack stick figure %s with grid size %d, is_symmetric: %s'
        ', allow_paired_axis_placement: %s, config: %s',
        stick_figure_input.name,
        grid_size,
        is_symmetric,
        allow_paired_axis_placement,
        config,
    )
    merge_converter = {}
    for (
        merged,
        unmerged_list,
    ) in stick_figure_input.merged_stick_name_dict.items():
      for unmerged in unmerged_list:
        merge_converter[unmerged] = merged
    packing_output_list = utils_solver.find_tiling_and_generate_crease_pattern(
        tree=stick_figure_input.merged_stick_figure.stick_figure,
        grid_size=grid_size,
        is_symmetric=is_symmetric,
        allow_paired_axis_placement=allow_paired_axis_placement,
        config=config,
        max_solutions=max_solutions,
        unmerged_tree=stick_figure_input.unmerged_stick_figure.stick_figure,
        merge_converter=merge_converter,
        solution_callback=solution_callback,
    )
    return packing_output_list
  except (AssertionError, ValueError) as e:
    logging.exception(
        'Failed to pack stick figure %s with grid size %d, exception: %s',
        stick_figure_input.name,
        grid_size,
        e,
    )
    return [
        packing_output_lib.PackingOutput(
            success=False,
            packing_cp=None,
            flap_rectangles=None,
            packing_rivers={},
            stick_figure=None,
            error_message=str(e),
            details=None,
            hinges=None,
            tracker=None,
            config=config,
        )
    ]


def _run_packing_sweep_pass(
    *,
    stick_figure_input: StickFigureInput,
    config: PackingTaskConfig,
    base_grid_size: int,
    is_symmetric: bool,
    max_grid_size: int | None,
    start_time: float,
    is_recovery_pass: bool = False,
    packing_outputs: list[packing_output_lib.PackingOutput],
    errors_per_retry: list[str],
    attempted_grid_sizes: list[int],
    time_spent_per_grid_sizes: list[float],
    solution_callback: (
        Callable[
            [packing_output_lib.PackingOutput],
            packing_output_lib.PackingOutput | None,
        ]
        | None
    ) = None,
) -> bool:
  """Runs a single sweep pass across grid sizes, appending successful outputs.

  Args:
    stick_figure_input: Stick figure input being packed.
    config: Packing task configuration.
    base_grid_size: Inferred base grid size.
    is_symmetric: Whether the stick figure tree is symmetric.
    max_grid_size: Optional upper bound on grid size.
    start_time: Wall-clock start time of packing stage.
    is_recovery_pass: Whether this pass runs with recovery flags enabled.
    packing_outputs: Output list to which successful packings are appended.
    errors_per_retry: List tracking error messages per retry.
    attempted_grid_sizes: List tracking attempted grid sizes.
    time_spent_per_grid_sizes: List tracking duration per grid size.
    solution_callback: Optional callback receiving each candidate PackingOutput.

  Returns:
    True if max_grid_size was reached, False otherwise.
  """
  max_grid_size_reached = False
  intermediate_solutions: list[packing_output_lib.PackingOutput] = []

  for sweep_config in config.config_iter():
    if config.timeout_seconds is not None and (
        time.perf_counter() - start_time > config.timeout_seconds
    ):
      errors_per_retry.append(
          f'Packing stage timeout {config.timeout_seconds}s reached.'
      )
      break
    if timeout.check_deadline().expired:
      errors_per_retry.append('Outer cooperative timeout reached.')
      break
    extra_grid_size = sweep_config.extra_grid_size
    if is_symmetric:
      extra_grid_size *= 2
    grid_size = base_grid_size + extra_grid_size

    if (
        max_grid_size is not None
        and max_grid_size > 0
        and grid_size > max_grid_size
    ):
      max_grid_size_reached = True
      errors_per_retry.append(f'Max grid size {max_grid_size} reached.')
      break

    attempted_grid_sizes.append(grid_size)
    start_time_per_grid_size = time.perf_counter()
    retry_error_message = None

    solutions_needed = config.max_number_of_cps - len(packing_outputs)
    if solutions_needed <= 0:
      break

    intermediate_solutions.clear()

    def _on_solution(result):
      if solution_callback is not None and result.success:
        if result.unmerged_stick_figure is None:
          result.unmerged_stick_figure = (
              stick_figure_input.unmerged_stick_figure.stick_figure
          )
        result = solution_callback(result)
        if result is None:
          return None
      intermediate_solutions.append(result)
      return result

    active_sweep_timeout = config.sweep_timeout_seconds
    if is_recovery_pass:
      active_packing_config = dataclasses.replace(
          sweep_config.packing_config,
          star_hub_rivers=True,
          symmetry_line_obstacles=True,
          prune_partial_border_overhang=True,
          stretch_intermediate_pocket_flaps=True,
      )
    else:
      active_packing_config = sweep_config.packing_config

    try:
      with timeout.cooperative_timeout(
          datetime.timedelta(seconds=active_sweep_timeout)  # pyrefly: ignore[bad-argument-type]
      ):
        packing_output_list = _attempt_packing(
            stick_figure_input=stick_figure_input,
            grid_size=grid_size,
            is_symmetric=is_symmetric,
            allow_paired_axis_placement=sweep_config.allow_paired_axis_placement,
            max_solutions=min(solutions_needed, config.max_cps_per_grid_size),
            config=active_packing_config,
            solution_callback=_on_solution,
            packer=sweep_config.packer,
            packer_config=sweep_config.packer_config,
        )
    except timeout.StageTimeoutError:
      message = (
          f'Packing timed out for stick figure {stick_figure_input.name} '
          f'with grid size {grid_size}'
      )
      logging.warning(message)
      retry_error_message = f'Grid size {grid_size}: Packing timed out.'
      errors_per_retry.append(retry_error_message)
      if intermediate_solutions:
        packing_outputs.extend(
            [output for output in intermediate_solutions if output.success]
        )
      break
    except timeout.CooperativeTimeoutError as err:
      message = (
          'Packing (cooperative) timeout for stick figure'
          f' {stick_figure_input.name} with grid size {grid_size}.\n'
          f'Original exception: {err}\n'
          f'Original traceback: {traceback.format_exc()}'
      )
      logging.warning(message)
      retry_error_message = (
          f'Grid size {grid_size}: Packing cooperative timeout: '
          f'{err.expired_timeout.prettyprint()}'
      )
      errors_per_retry.append(retry_error_message)
      if intermediate_solutions:
        packing_outputs.extend(
            [output for output in intermediate_solutions if output.success]
        )
      if len(packing_outputs) >= config.max_number_of_cps:
        del packing_outputs[config.max_number_of_cps :]
        break
      if timeout.check_deadline().expired:
        break
      continue

    packing_outputs.extend([
        packing_output
        for packing_output in packing_output_list
        if packing_output.success
    ])

    time_spent_per_grid_sizes.append(
        time.perf_counter() - start_time_per_grid_size
    )

    if len(packing_outputs) >= config.max_number_of_cps:
      del packing_outputs[config.max_number_of_cps :]
      break

    if retry_error_message:
      errors_per_retry.append(retry_error_message)
    elif packing_output_list:
      errors = '; '.join(
          [p.error_message for p in packing_output_list if p.error_message]
      )
      errors_per_retry.append(f'Grid size {grid_size}: {errors}')
    else:
      errors_per_retry.append(
          f'Grid size {grid_size}: failed with no specific error.'
      )

  return max_grid_size_reached


def get_base_grid_size(
    stick_figure_obj: stick_figure.StickFigure,
    config: PackingTaskConfig,
    *,
    is_symmetric: bool | None = None,
) -> int:
  """Returns the even-aligned starting grid size for a packing sweep."""
  effective_is_sym = (
      stick_figure_obj.is_symmetric_tree
      if is_symmetric is None
      else is_symmetric
  )
  v4_is_sym = effective_is_sym and not config.allow_paired_axis_placement
  if config.grid_mode == 'v4':
    base_grid_size = grid_size_lib.calculate_v4_grid_size(
        stick_figure_obj,
        use_level_shifter=config.packing_config.use_level_shifter,
        is_symmetric=v4_is_sym,
        flap_dim_fn=utils_packing_flaps.get_flap_dim,
    )
  else:
    base_grid_size = stick_figure_obj.get_packing_gridsize(
        use_level_shifter=config.packing_config.use_level_shifter,
        mode=config.grid_mode,
        is_symmetric=v4_is_sym,
    )
  if effective_is_sym and base_grid_size % 2 == 1:
    base_grid_size += 1
  return base_grid_size


def pack_crease_pattern(
    stick_figure_input: (
        StickFigureInput | stick_figure.StickFigure | bus.DesignTree
    ),
    config: PackingTaskConfig | None = None,
    *,
    solution_callback: (
        Callable[
            [packing_output_lib.PackingOutput],
            packing_output_lib.PackingOutput | None,
        ]
        | None
    ) = None,
) -> PackingResult:
  """Packs the given stick figure into a crease pattern."""
  if isinstance(stick_figure_input, (stick_figure.StickFigure, bus.DesignTree)):
    stick_figure_input = process_stick_figure(stick_figure_input)
  if config is None:
    config = PackingTaskConfig()

  start_time = time.perf_counter()

  is_symmetric = stick_figure_input.is_symmetric_tree
  stick_figure_obj = stick_figure_input.merged_stick_figure.stick_figure
  base_grid_size = get_base_grid_size(
      stick_figure_obj, config, is_symmetric=is_symmetric
  )
  from . import plugins  # pylint: disable=g-import-not-at-top

  if (
      plugins.resolve_packer(config.packer) == 'builtin'
      and config.extra_grid_size == 0
      and 'extra_grid_size' not in config.sweep_hparams
      and shaping_templates.has_shaping_flaps(stick_figure_obj)
  ):
    base_grid_size = max(
        base_grid_size,
        shaping_flap_templates.estimate_template_min_grid_size(
            stick_figure_obj, is_symmetric
        ),
    )
  max_grid_size = stick_figure_input.max_grid_size
  packing_outputs: list[packing_output_lib.PackingOutput] = []
  errors_per_retry: list[str] = []

  attempted_grid_sizes: list[int] = []
  time_spent_per_grid_sizes: list[float] = []

  # Pass 1: Primary packing sweep
  pass_1_time = None
  active_timeout = timeout.get_timeout()
  if config.packing_config.recovery:
    if active_timeout is not None:
      pass_1_time = (
          active_timeout.remaining.total_seconds() * _PASS_1_BUDGET_FRACTION
      )
    elif config.timeout_seconds is not None:
      pass_1_time = config.timeout_seconds * _PASS_1_BUDGET_FRACTION

  with timeout.cooperative_timeout(pass_1_time):
    max_grid_size_reached = _run_packing_sweep_pass(
        stick_figure_input=stick_figure_input,
        config=config,
        base_grid_size=base_grid_size,
        is_symmetric=is_symmetric,
        max_grid_size=max_grid_size,
        start_time=start_time,
        is_recovery_pass=False,
        packing_outputs=packing_outputs,
        errors_per_retry=errors_per_retry,
        attempted_grid_sizes=attempted_grid_sizes,
        time_spent_per_grid_sizes=time_spent_per_grid_sizes,
        solution_callback=solution_callback,
    )

  # Pass 2: Recovery sweep (star-hub rivers & overhang pruning) when primary
  # pass found no valid packing, recovery is enabled, and time budget allows.
  all_recovery_flags_enabled = (
      config.packing_config.star_hub_rivers
      and config.packing_config.symmetry_line_obstacles
      and config.packing_config.prune_partial_border_overhang
      and config.packing_config.stretch_intermediate_pocket_flaps
  )
  if (
      not packing_outputs
      and config.packing_config.recovery
      and not all_recovery_flags_enabled
      and not timeout.check_deadline().expired
      and (
          config.timeout_seconds is None
          or time.perf_counter() - start_time < config.timeout_seconds
      )
  ):
    max_grid_size_reached = _run_packing_sweep_pass(
        stick_figure_input=stick_figure_input,
        config=config,
        base_grid_size=base_grid_size,
        is_symmetric=is_symmetric,
        max_grid_size=max_grid_size,
        start_time=start_time,
        is_recovery_pass=True,
        packing_outputs=packing_outputs,
        errors_per_retry=errors_per_retry,
        attempted_grid_sizes=attempted_grid_sizes,
        time_spent_per_grid_sizes=time_spent_per_grid_sizes,
        solution_callback=solution_callback,
    )

  if not packing_outputs:
    error_log = '\n'.join(errors_per_retry)
    raise ValueError(
        'CREASE_PATTERN_PACKING: No packing CP found for stick figure'
        f' {stick_figure_input.name} after'
        f' {len(attempted_grid_sizes)} retries.\n'
        f'Errors per grid size:\n{error_log}'
    )

  unmerged_sf = stick_figure_input.unmerged_stick_figure.stick_figure
  for output in packing_outputs:
    if output.unmerged_stick_figure is None:
      output.unmerged_stick_figure = unmerged_sf
  packing_cps = [
      output.packing_cp
      for output in packing_outputs
      if output.packing_cp is not None
  ]

  result = PackingResult(
      name=stick_figure_input.name,
      packing_cps=packing_cps,
      packing_outputs=packing_outputs,
      attempted_grid_sizes=attempted_grid_sizes,
      time_spent_per_grid_sizes=time_spent_per_grid_sizes,
      sequence_status='SUCCESS in CREASE_PATTERN_PACKING',
      duration_seconds=time.perf_counter() - start_time,
      max_grid_size_reached=max_grid_size_reached,
      errors_per_retry=errors_per_retry,
      stick_figure_input=stick_figure_input,
      packing_config=config,
  )
  logging.info(
      'Finished task-sequence CREASE_PATTERN_PACKING in %s seconds. Packed CP'
      ' found for stick figure %s.',
      result.duration_seconds,
      stick_figure_input.name,
  )
  return result


def _cp_signature(cp: crease_pattern.CreasePattern | None) -> int | None:
  """Returns a canonical structural hash for deduplicating packed CPs."""
  if cp is None:
    return None
  return cp.hash_value


def _packing_output_key(
    output: packing_output_lib.PackingOutput,
) -> tuple[int, int | None]:
  """Returns a value-based key `(grid_size, cp_hash)` for a PackingOutput."""
  if output.packing_cp is None:
    return (0, None)
  return (_safe_grid_size(output), output.packing_cp.hash_value)


def _remaining_solve_budget_s(
    start_time: float,
    solve_timeout_seconds: float | None,
) -> float | None:
  """Returns remaining seconds across stage timeout and cooperative deadline."""
  candidates: list[float] = []
  if solve_timeout_seconds is not None:
    candidates.append(
        max(0.0, solve_timeout_seconds - (time.perf_counter() - start_time))
    )
  deadline_status = timeout.check_deadline()
  if deadline_status.expired:
    return 0.0
  if deadline_status.remaining is not None:
    candidates.append(max(0.0, deadline_status.remaining.total_seconds()))
  return min(candidates) if candidates else None


def iter_additional_packing_candidates(
    packing_result: PackingResult,
    *,
    start_time: float,
    solve_timeout_seconds: float | None,
    max_extra_candidates: int = 7,
    max_candidates_per_grid_size: int = 2,
) -> Iterator[packing_output_lib.PackingOutput]:
  """Lazily yields distinct candidate packings across up to 4 grid sizes."""
  stick_figure_input = packing_result.stick_figure_input
  pack_cfg = packing_result.packing_config
  if (
      stick_figure_input is None
      or pack_cfg is None
      or not packing_result.packing_outputs
  ):
    return

  first_out = packing_result.packing_outputs[0]
  first_grid = _safe_grid_size(first_out)
  if first_grid <= 0:
    return

  is_symmetric = stick_figure_input.is_symmetric_tree
  step = 2 if is_symmetric else 1
  max_grid_size = stick_figure_input.max_grid_size
  base_pack_cfg = first_out.config or pack_cfg.packing_config
  unmerged_sf = stick_figure_input.unmerged_stick_figure.stick_figure

  seen_sigs: set[int] = set()
  grid_counts: dict[int, int] = {}
  for out in packing_result.packing_outputs:
    if out.packing_cp is not None:
      sig = _cp_signature(out.packing_cp)
      if sig is not None:
        seen_sigs.add(sig)
      g = _safe_grid_size(out)
      if g > 0:
        grid_counts[g] = grid_counts.get(g, 0) + 1

  yielded_count = 0
  candidate_grids = [
      first_grid,
      first_grid + step,
      first_grid + 2 * step,
      first_grid + 3 * step,
  ]

  first_grid_pack_s = (
      packing_result.time_spent_per_grid_sizes[-1]
      if packing_result.time_spent_per_grid_sizes
      else 0.0
  )

  def _make_solution_callback(bucket, cfg):
    def callback(res):
      if (
          cfg.reject_provably_unsolvable_packings
          and res.packing_cp is not None
          and solvability.is_provably_unsolvable(res.packing_cp)
      ):
        return None
      bucket.append(res)
      return res

    return callback

  for cand_grid in candidate_grids:
    if yielded_count >= max_extra_candidates:
      return
    if (
        max_grid_size is not None
        and max_grid_size > 0
        and cand_grid > max_grid_size
    ):
      continue
    if grid_counts.get(cand_grid, 0) >= max_candidates_per_grid_size:
      continue

    base_cap_s = 65.0 if cand_grid >= 46 else 35.0
    exhaust_cap_s = 45.0 if cand_grid >= 46 else 30.0
    if cand_grid == first_grid:
      base_cap_s = max(base_cap_s, min(90.0, first_grid_pack_s * 1.3))
      exhaust_cap_s = max(exhaust_cap_s, min(75.0, first_grid_pack_s * 1.1))
    grid_configs: list[tuple[int, packing_config_lib.PackingConfig, float]] = [
        (max_candidates_per_grid_size, base_pack_cfg, base_cap_s),
        (
            max_candidates_per_grid_size + 1,
            dataclasses.replace(base_pack_cfg, exhaust_solutions=True),
            exhaust_cap_s,
        ),
    ]

    found_solution_on_grid = False
    for cfg_idx, (max_sols, cand_pack_cfg, cap_s) in enumerate(grid_configs):
      if yielded_count >= max_extra_candidates:
        return
      if grid_counts.get(cand_grid, 0) >= max_candidates_per_grid_size:
        break
      if cfg_idx > 0 and not found_solution_on_grid:
        break

      rem_s = _remaining_solve_budget_s(start_time, solve_timeout_seconds)
      if rem_s is not None and rem_s < 15.0:
        return
      attempt_timeout_s = cap_s
      if rem_s is not None:
        attempt_timeout_s = min(cap_s, max(15.0, rem_s * 0.25))
      if pack_cfg.sweep_timeout_seconds is not None:
        attempt_timeout_s = min(
            attempt_timeout_s, float(pack_cfg.sweep_timeout_seconds)
        )

      intermediate_solutions: list[packing_output_lib.PackingOutput] = []
      solution_callback = _make_solution_callback(
          intermediate_solutions, cand_pack_cfg
      )

      try:
        with timeout.cooperative_timeout(
            datetime.timedelta(seconds=attempt_timeout_s)
        ):
          cand_list = _attempt_packing(
              stick_figure_input=stick_figure_input,
              grid_size=cand_grid,
              is_symmetric=is_symmetric,
              allow_paired_axis_placement=pack_cfg.allow_paired_axis_placement,
              max_solutions=max_sols,
              config=cand_pack_cfg,
              solution_callback=solution_callback,
              packer=pack_cfg.packer,
              packer_config=pack_cfg.packer_config,
          )
      except (timeout.StageTimeoutError, timeout.CooperativeTimeoutError):
        cand_list = list(intermediate_solutions)
        if timeout.check_deadline().expired:
          return

      for out in cand_list:
        if not out.success or out.packing_cp is None:
          continue
        found_solution_on_grid = True
        sig = _cp_signature(out.packing_cp)
        if sig is None or sig in seen_sigs:
          continue
        seen_sigs.add(sig)
        g = _safe_grid_size(out)
        if g > 0:
          grid_counts[g] = grid_counts.get(g, 0) + 1
        if out.unmerged_stick_figure is None:
          out.unmerged_stick_figure = unmerged_sf
        yielded_count += 1
        yield out
        if (
            yielded_count >= max_extra_candidates
            or grid_counts.get(cand_grid, 0) >= max_candidates_per_grid_size
        ):
          break


def _solver_config_key(cfg: SolvingTaskConfig) -> tuple[Any, ...]:
  """Returns a hashable key for solver configuration settings that define state."""
  return (
      cfg.scale_factor,
      cfg.hinges_from_stick_figure,
      cfg.remove_central_crease,
      cfg.up_down_method,
      cfg.use_partition_order,
      cfg.disallow_trapping_edge_flaps,
      cfg.allow_misaligned_symmetric_hinges,
      cfg.tie_break_by_hinge_symmetry,
      cfg.check_local_flat_foldability,
      cfg.check_global_flat_foldability,
      cfg.use_python_flat_folder,
      cfg.solver,
  )


def _solver_instance_key(
    packing_output: packing_output_lib.PackingOutput,
    cfg: SolvingTaskConfig,
) -> tuple[tuple[int, int | None], tuple[Any, ...]]:
  """Returns a hashable key `(packing_key, config_key)` for a solver instance."""
  return (_packing_output_key(packing_output), _solver_config_key(cfg))


_CURRENT_RECOVERY_CTX: contextvars.ContextVar[Any] = contextvars.ContextVar(
    '_CURRENT_RECOVERY_CTX', default=None
)


def _solve_single_config(
    config: SolvingTaskConfig,
    packing_output: packing_output_lib.PackingOutput,
    recovery_ctx: _SolverRecoveryContext | None = None,
) -> list[hinge_utils.SolverResult] | str:
  """Solves for flat-foldable crease patterns for a single config."""
  if recovery_ctx is None:
    recovery_ctx = _CURRENT_RECOVERY_CTX.get()
  cp = packing_output.packing_cp
  if cp is None:
    raise ValueError(f'CP is None in packing output: {packing_output}')

  raw_grid_size = _safe_grid_size(cp)
  scale = config.scale_factor if config.scale_factor is not None else 1
  grid_size = int(raw_grid_size * scale)
  intermediate_solutions: list[hinge_utils.SolverResult] = []

  def append_to_intermediate_solutions(solve_result):
    if solve_result not in intermediate_solutions:
      intermediate_solutions.append(solve_result)
    return solve_result

  solver_config = config.to_solver_config(
      solution_callback=append_to_intermediate_solutions
  )
  solver_key = _solver_instance_key(packing_output, config)
  solver: hinge_utils.Solver | None = None
  if recovery_ctx is not None and config.enable_recovery:
    solver = recovery_ctx.active_solvers.get(solver_key)

  try:
    if packing_output.sacrificial_flaps and packing_output.flap_rectangles:
      ls_items = level_shifters_lib.solve_with_pocket_level_shifters(
          packing_output,
          solver_config,
          scale_factor=scale,
          resolved_solver='builtin',
      )
      if not ls_items:
        raise hinge_utils.NoSolutionError('No level-shifter solutions found.')
      return [sol for sol, _, _, _ in ls_items]

    if solver is not None:
      solver.config = solver_config.bind(
          grid_size=(
              solver.config.grid_size
              if solver.config.grid_size is not None
              else grid_size
          ),
          packing_information=solver.config.packing_information,
          preassignments=solver.config.preassignments,
      )
      if solver.solutions:
        intermediate_solutions.extend(solver.solutions)
    else:
      solver = solver_config.make_from_cp(
          cp, grid_size=grid_size, packing_information=packing_output
      )
      if recovery_ctx is not None and config.enable_recovery:
        recovery_ctx.active_solvers[solver_key] = solver

    if solver_config.remove_central_crease:
      can_remove_central = (
          hinge_utils.central_fold_utils.should_try_to_remove_central_crease(
              solver.init_state.unvisited_hinges
              + solver.init_state.ridges
              + solver.init_state.pleats
              + solver.init_state.unassigned_pleats,
              solver.init_state.unvisited_hinges,
          )
      )
      if not can_remove_central:
        if recovery_ctx is not None and config.enable_recovery:
          recovery_ctx.exhausted_config_keys.add(solver_key)
        raise hinge_utils.NoSolutionError('Cannot remove central crease.')

    _ = solver.solve()
    if (
        solver.is_exhausted
        and recovery_ctx is not None
        and config.enable_recovery
    ):
      recovery_ctx.exhausted_config_keys.add(solver_key)
    if not intermediate_solutions:
      raise hinge_utils.NoSolutionError('No solutions found.')
    return solver_config.process_solutions(intermediate_solutions)
  except (timeout.SweepLevelTimeoutError, timeout.CooperativeTimeoutError):
    if (
        solver is not None
        and solver.is_exhausted
        and recovery_ctx is not None
        and config.enable_recovery
    ):
      recovery_ctx.exhausted_config_keys.add(solver_key)
    if intermediate_solutions:
      return solver_config.process_solutions(intermediate_solutions)
    raise
  except (
      ValueError,
      hinge_utils.NoSolutionError,
      IndexError,
  ) as e:
    if recovery_ctx is not None and config.enable_recovery:
      if 'Cannot remove central crease' in str(e):
        pack_key = _packing_output_key(packing_output)
        recovery_ctx.dead_central_crease_keys.add(
            (pack_key, config.scale_factor)
        )
      if solver is not None and solver.is_exhausted:
        recovery_ctx.exhausted_config_keys.add(solver_key)
    return traceback.format_exc()


_SweepPair = tuple[packing_output_lib.PackingOutput, SolvingTaskConfig, bool]


def _cross_product_pairs(
    outputs: Sequence[packing_output_lib.PackingOutput],
    configs: Sequence[SolvingTaskConfig],
    *,
    is_fallback: bool,
) -> list[_SweepPair]:
  """Builds `(packing_output, solving_config, is_fallback)` pairs."""
  pairs: list[_SweepPair] = []
  for out in outputs:
    for cfg in configs:
      pairs.append((out, cfg, is_fallback))
  return pairs


@dataclasses.dataclass(slots=True)
class _SolverRecoveryContext:
  """Mutable state shared across solver recovery stages."""

  initial_outputs: list[packing_output_lib.PackingOutput]
  all_tried_outputs: list[packing_output_lib.PackingOutput]
  primary_guided: list[SolvingTaskConfig]
  primary_unguided: list[SolvingTaskConfig]
  fallback_guided_scale1: list[SolvingTaskConfig]
  fallback_guided_scale2: list[SolvingTaskConfig]
  timed_out_guided_pairs: list[_SweepPair] = dataclasses.field(
      default_factory=list
  )
  timed_out_unguided_pairs: list[_SweepPair] = dataclasses.field(
      default_factory=list
  )
  active_solvers: dict[
      tuple[tuple[int, int | None], tuple[Any, ...]], hinge_utils.Solver
  ] = dataclasses.field(default_factory=dict)
  dead_central_crease_keys: set[tuple[tuple[int, int | None], int]] = (
      dataclasses.field(default_factory=set)
  )
  exhausted_config_keys: set[tuple[tuple[int, int | None], tuple[Any, ...]]] = (
      dataclasses.field(default_factory=set)
  )

  @property
  def initial_keys(self) -> frozenset[tuple[int, int | None]]:
    """Returns keys for the initial packing outputs."""
    return frozenset(_packing_output_key(o) for o in self.initial_outputs)

  @property
  def first_grid_keys(self) -> frozenset[tuple[int, int | None]]:
    """Returns keys for all tried packings on the initial grid size."""
    first_grid = 0
    if self.initial_outputs:
      first_grid = _safe_grid_size(self.initial_outputs[0])
    same_grid = {
        _packing_output_key(o)
        for o in self.all_tried_outputs
        if _safe_grid_size(o) == first_grid and first_grid > 0
    }
    return frozenset(same_grid | self.initial_keys)


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class _RecoveryStageSpec:
  """Declarative specification for a post-repack solver recovery stage."""

  name: str
  build_pairs: Callable[[_SolverRecoveryContext], Sequence[_SweepPair]]
  max_per_config_timeout_s: float
  min_remaining_budget_s: float | None = None
  record_timeouts: bool = False
  requires_fallback_configs: bool = False
  requires_timeout_retries: bool = False


def _build_scale2_guided_pairs(
    ctx: _SolverRecoveryContext,
) -> list[_SweepPair]:
  """Builds scale-2 guided fallback pairs on one packing per distinct grid."""
  seen_grids: set[int] = set()
  representative_outputs: list[packing_output_lib.PackingOutput] = []
  for out in ctx.all_tried_outputs:
    g = _safe_grid_size(out)
    if g not in seen_grids:
      seen_grids.add(g)
      representative_outputs.append(out)
  pairs: list[_SweepPair] = []
  for out in representative_outputs:
    base_grid = _safe_grid_size(out)
    for cfg in ctx.fallback_guided_scale2:
      if cfg.scale_factor > 1 and base_grid > 44:
        continue
      pairs.append((out, cfg, True))
  return pairs


def _build_greedy_fallback_pairs(
    ctx: _SolverRecoveryContext,
) -> list[_SweepPair]:
  """Builds greedy fallback pairs with max_branching_factor=1 on first grid."""
  greedy_configs = [
      _replace_nested_fields(cfg, max_branching_factor=1)
      for cfg in ctx.primary_guided
  ]
  fg_outputs = [
      o
      for o in ctx.all_tried_outputs
      if _packing_output_key(o) in ctx.first_grid_keys
  ]
  return _cross_product_pairs(fg_outputs, greedy_configs, is_fallback=True)


def _build_primary_unguided_pairs(
    ctx: _SolverRecoveryContext,
) -> list[_SweepPair]:
  """Builds unguided sweep pairs across all tried packings."""
  return _cross_product_pairs(
      ctx.all_tried_outputs, ctx.primary_unguided, is_fallback=True
  )


def _build_unpartitioned_guided_pairs(
    ctx: _SolverRecoveryContext,
) -> list[_SweepPair]:
  """Builds unpartitioned guided fallback pairs on the initial packing."""
  unpartitioned_configs = [
      _replace_nested_fields(cfg, use_partition_order=False)
      for cfg in ctx.primary_guided
  ]
  return _cross_product_pairs(
      ctx.initial_outputs[:1], unpartitioned_configs, is_fallback=True
  )


def _build_initial_guided_timeout_retries(
    ctx: _SolverRecoveryContext,
) -> list[_SweepPair]:
  """Builds extended-timeout retry for the first timed-out initial guided pair."""
  init_keys = ctx.initial_keys
  return [
      pair
      for pair in ctx.timed_out_guided_pairs
      if _packing_output_key(pair[0]) in init_keys and not pair[2]
  ][:1]


def _build_first_grid_unguided_timeout_retries(
    ctx: _SolverRecoveryContext,
) -> list[_SweepPair]:
  """Builds extended-timeout retries for timed-out unguided configs on grid 0."""
  fg_keys = ctx.first_grid_keys
  return [
      pair
      for pair in ctx.timed_out_unguided_pairs
      if _packing_output_key(pair[0]) in fg_keys
  ]


def _build_remaining_timeout_retries(
    ctx: _SolverRecoveryContext,
) -> list[_SweepPair]:
  """Builds extended-timeout retries for remaining timed-out configs."""
  init_keys = ctx.initial_keys
  fg_keys = ctx.first_grid_keys
  retries: list[_SweepPair] = [
      pair
      for pair in ctx.timed_out_unguided_pairs
      if _packing_output_key(pair[0]) not in fg_keys
  ]
  seen_relaxed_grids: set[int] = set()
  for pair in ctx.timed_out_guided_pairs:
    if _packing_output_key(pair[0]) in init_keys or pair[2]:
      continue
    g = _safe_grid_size(pair[0])
    if g not in seen_relaxed_grids:
      seen_relaxed_grids.add(g)
      retries.append(pair)
  retries.extend(
      pair
      for pair in ctx.timed_out_guided_pairs
      if _packing_output_key(pair[0]) in init_keys and pair[2]
  )
  return retries


_POST_REPACK_RECOVERY_STAGES: tuple[_RecoveryStageSpec, ...] = (
    # Double the grid resolution (scale_factor=2) on one representative packing
    # per grid size when unit-grid pleats cannot resolve tight ridge junctions.
    _RecoveryStageSpec(
        name='scale2_guided_fallback',
        build_pairs=_build_scale2_guided_pairs,
        max_per_config_timeout_s=20.0,
        requires_fallback_configs=True,
    ),
    # Fast greedy fallback when extremely low on time.
    _RecoveryStageSpec(
        name='greedy_fallback',
        build_pairs=_build_greedy_fallback_pairs,
        max_per_config_timeout_s=5.0,
        requires_fallback_configs=True,
    ),
    # Fall back to unguided hinge search (`hinges_from_stick_figure=False`) when
    # stick-figure orientation heuristics over-constrain hinge flatness.
    _RecoveryStageSpec(
        name='primary_unguided',
        build_pairs=_build_primary_unguided_pairs,
        max_per_config_timeout_s=25.0,
        record_timeouts=True,
    ),
    # Disable partition-by-partition solving order on the initial packing in
    # case cross-partition hinge dependencies blocked partitioned search.
    _RecoveryStageSpec(
        name='unpartitioned_guided',
        build_pairs=_build_unpartitioned_guided_pairs,
        max_per_config_timeout_s=45.0,
        min_remaining_budget_s=15.0,
        requires_fallback_configs=True,
    ),
    # Re-run the first timed-out guided config on the initial packing with a
    # generous budget (260s) now that fast alternatives have been exhausted.
    _RecoveryStageSpec(
        name='retry_initial_guided_timeouts',
        build_pairs=_build_initial_guided_timeout_retries,
        max_per_config_timeout_s=260.0,
        min_remaining_budget_s=20.0,
        requires_timeout_retries=True,
    ),
    # Re-run timed-out unguided configs on the minimum grid size with a 220s
    # budget before spending remaining time on larger relaxed grids.
    _RecoveryStageSpec(
        name='retry_first_grid_unguided_timeouts',
        build_pairs=_build_first_grid_unguided_timeout_retries,
        max_per_config_timeout_s=220.0,
        min_remaining_budget_s=20.0,
        requires_timeout_retries=True,
    ),
    # Spend any remaining budget (up to 160s) on timed-out configs from larger
    # relaxed grids or fallback sweeps.
    _RecoveryStageSpec(
        name='retry_remaining_timeouts',
        build_pairs=_build_remaining_timeout_retries,
        max_per_config_timeout_s=160.0,
        min_remaining_budget_s=20.0,
        requires_timeout_retries=True,
    ),
)


def _scaled_per_config_timeout_s(
    base_s: float,
    cands: Sequence[packing_output_lib.PackingOutput],
) -> float:
  """Scales per-config timeout quadratically for grids above 30 (capped at 60s).

  Per-step vertex/hinge intersection work grows roughly with O(grid_size^2), so
  grids above 30 need proportionally more wall-clock time to reach the same
  search depth before Stage 1/2 probe caps cut them off.

  Args:
    base_s: Base timeout in seconds at grid size <= 30.
    cands: Candidate packings whose maximum grid size scales `base_s`.

  Returns:
    Scaled per-config timeout in seconds, capped at 60.0s.
  """
  max_g = max(
      (_safe_grid_size(o) for o in cands),
      default=30,
  )
  if max_g <= 0:
    max_g = 30
  scale = max(1.0, (max_g / 30.0) ** 2)
  return min(60.0, base_s * scale)


def _run_solver_sweep_pairs(
    sweep_pairs: Sequence[_SweepPair],
    *,
    config: SolvingTaskConfig,
    recovery_ctx: _SolverRecoveryContext,
    start_time: float,
    solver_outputs: list[hinge_utils.SolverResult],
    solver_errors: list[str],
    scaling_factors_used: list[int],
    packing_outputs_used: list[packing_output_lib.PackingOutput],
    max_per_config_timeout_s: float | None = None,
    record_timeouts: bool = True,
) -> None:
  """Executes a sequence of (packing_output, solving_config, is_fallback) pairs."""
  token = _CURRENT_RECOVERY_CTX.set(recovery_ctx)
  try:
    for idx, (packing_output, solving_config, is_fallback) in enumerate(
        sweep_pairs
    ):
      if solver_outputs and (config.stop_on_first_solve or is_fallback):
        break

      if config.timeout_seconds is not None and (
          time.perf_counter() - start_time >= config.timeout_seconds
      ):
        solver_errors.append(
            f'Solving stage timeout {config.timeout_seconds}s reached.'
        )
        break

      deadline_status = timeout.check_deadline()
      if deadline_status.expired:
        solver_errors.append(
            'Outer cooperative timeout expired before solver config.'
        )
        break

      pack_key = _packing_output_key(packing_output)
      if config.enable_recovery:
        if (
            solving_config.remove_central_crease
            and (pack_key, solving_config.scale_factor)
            in recovery_ctx.dead_central_crease_keys
        ):
          logging.info(
              'Skipping config with remove_central_crease=True on dead'
              ' packing %s',
              pack_key,
          )
          continue
        solver_key = _solver_instance_key(packing_output, solving_config)
        if solver_key in recovery_ctx.exhausted_config_keys:
          logging.info(
              'Skipping solver with already-exhausted search frontier: %s',
              solver_key,
          )
          continue

      orig_solving_config = solving_config
      effective_sweep_timeout = solving_config.sweep_timeout_seconds
      if max_per_config_timeout_s is not None and (
          effective_sweep_timeout is None
          or max_per_config_timeout_s < effective_sweep_timeout
      ):
        effective_sweep_timeout = max_per_config_timeout_s

      if (
          config.enable_recovery
          and max_per_config_timeout_s is not None
          and max_per_config_timeout_s > 60.0
      ):
        solving_config = _replace_nested_fields(
            solving_config,
            max_steps=max(solving_config.max_steps or 0, 1_000_000),
        )

      if config.enable_recovery and deadline_status.remaining is not None:
        rem_s = deadline_status.remaining.total_seconds()
        if solver_outputs and rem_s < 5.0:
          break
        if solving_config.hinges_from_stick_figure:
          remaining_guided = sum(
              1
              for _, cfg, fb in sweep_pairs[idx:]
              if cfg.hinges_from_stick_figure and fb == is_fallback
          )
          budget_s = max(5.0, rem_s / max(1, remaining_guided))
          if solver_outputs:
            budget_s = min(budget_s, 15.0)
        else:
          budget_s = max(1.0, rem_s - 0.5 if solver_outputs else rem_s)
        if (
            effective_sweep_timeout is None
            or budget_s < effective_sweep_timeout
        ):
          effective_sweep_timeout = budget_s

      if (
          effective_sweep_timeout is not None
          and effective_sweep_timeout != solving_config.sweep_timeout_seconds
      ):
        solving_config = _replace_nested_fields(
            solving_config, sweep_timeout_seconds=effective_sweep_timeout
        )

      logging.info(
          'Solving for flat-foldable crease pattern with %s', solving_config
      )
      config_timed_out = False
      try:
        if effective_sweep_timeout is not None:
          with timeout.cooperative_timeout(
              datetime.timedelta(seconds=effective_sweep_timeout)
          ):
            solutions_or_error_trace = _solve_single_config(
                config=solving_config,
                packing_output=packing_output,
            )
        else:
          solutions_or_error_trace = _solve_single_config(
              config=solving_config,
              packing_output=packing_output,
          )
      except (
          timeout.SweepLevelTimeoutError,
          timeout.CooperativeTimeoutError,
      ):
        config_timed_out = True
        solutions_or_error_trace = traceback.format_exc()

      if isinstance(solutions_or_error_trace, str):
        solver_key = _solver_instance_key(packing_output, orig_solving_config)
        cached_solver = recovery_ctx.active_solvers.get(solver_key)
        timed_out_or_incomplete = config_timed_out or (
            cached_solver is not None and not cached_solver.is_exhausted
        )
        if (
            record_timeouts
            and timed_out_or_incomplete
            and orig_solving_config.scale_factor == 1
        ):
          if orig_solving_config.hinges_from_stick_figure:
            recovery_ctx.timed_out_guided_pairs.append(
                (packing_output, orig_solving_config, is_fallback)
            )
          else:
            recovery_ctx.timed_out_unguided_pairs.append(
                (packing_output, orig_solving_config, is_fallback)
            )
        error_trace = (
            'Solving Error with: \n remove_central_crease = '
            + str(solving_config.remove_central_crease)
            + '\nhinges_from_stick_figure = '
            + str(solving_config.hinges_from_stick_figure)
            + '\nscale_factor = '
            + str(solving_config.scale_factor)
            + '\nError: \n'
            + solutions_or_error_trace
        )
        logging.warning(error_trace)
        solver_errors.append(error_trace)
        if timeout.check_deadline().expired:
          break
      else:
        solutions: list[hinge_utils.SolverResult] = solutions_or_error_trace
        solver_outputs.extend(solutions)
        scaling_factors_used.extend(
            [int(solving_config.scale_factor) for _ in solutions]
        )
        packing_outputs_used.extend([packing_output for _ in solutions])
        if config.stop_on_first_solve or timeout.check_deadline().expired:
          break
  finally:
    _CURRENT_RECOVERY_CTX.reset(token)


def _run_scale1_guided_on_candidates(
    cands: Sequence[packing_output_lib.PackingOutput],
    *,
    config: SolvingTaskConfig,
    recovery_ctx: _SolverRecoveryContext,
    primary_guided: Sequence[SolvingTaskConfig],
    fallback_guided_scale1: Sequence[SolvingTaskConfig],
    start_time: float,
    solver_outputs: list[hinge_utils.SolverResult],
    solver_errors: list[str],
    scaling_factors_used: list[int],
    packing_outputs_used: list[packing_output_lib.PackingOutput],
    primary_base_timeout_s: float | None,
    fallback_base_timeout_s: float | None,
) -> None:
  """Runs scale-1 guided configs on candidate packings, falling back if needed."""
  primary_pairs = _cross_product_pairs(cands, primary_guided, is_fallback=False)
  _run_solver_sweep_pairs(
      primary_pairs,
      config=config,
      recovery_ctx=recovery_ctx,
      start_time=start_time,
      solver_outputs=solver_outputs,
      solver_errors=solver_errors,
      scaling_factors_used=scaling_factors_used,
      packing_outputs_used=packing_outputs_used,
      max_per_config_timeout_s=(
          _scaled_per_config_timeout_s(primary_base_timeout_s, cands)
          if primary_base_timeout_s is not None
          else None
      ),
  )
  if not solver_outputs and config.enable_fallback_configs:
    fallback_pairs = _cross_product_pairs(
        cands, fallback_guided_scale1, is_fallback=True
    )
    _run_solver_sweep_pairs(
        fallback_pairs,
        config=config,
        recovery_ctx=recovery_ctx,
        start_time=start_time,
        solver_outputs=solver_outputs,
        solver_errors=solver_errors,
        scaling_factors_used=scaling_factors_used,
        packing_outputs_used=packing_outputs_used,
        max_per_config_timeout_s=(
            _scaled_per_config_timeout_s(fallback_base_timeout_s, cands)
            if fallback_base_timeout_s is not None
            else None
        ),
    )


def solve_crease_pattern(
    packing_outputs: (
        PackingResult
        | Sequence[packing_output_lib.PackingOutput]
        | packing_output_lib.PackingOutput
    ),
    config: SolvingTaskConfig | None = None,
    *,
    name: str = 'crease_pattern',
) -> SolveResult:
  """Solves for flat-foldable crease patterns from packed crease patterns."""
  if isinstance(packing_outputs, PackingResult):
    name = packing_outputs.name
    outputs_seq = list(packing_outputs.packing_outputs)
  elif isinstance(packing_outputs, packing_output_lib.PackingOutput):
    outputs_seq = [packing_outputs]
  else:
    outputs_seq = list(packing_outputs)

  if config is None:
    config = SolvingTaskConfig()

  if config.enable_recovery and len(outputs_seq) > 1:

    def _sort_key(
        o: packing_output_lib.PackingOutput,
    ) -> tuple[bool, int, int]:
      if o.packing_cp is None:
        return (True, 0, 0)
      return solvability.score_packing_solvability(o.packing_cp)

    outputs_seq.sort(key=_sort_key)

  start_time = time.perf_counter()
  packing_outputs_used: list[packing_output_lib.PackingOutput] = []
  solver_outputs: list[hinge_utils.SolverResult] = []
  scaling_factors_used: list[int] = []
  solver_errors: list[str] = []

  primary_configs = list(config.config_iter())
  primary_guided = [c for c in primary_configs if c.hinges_from_stick_figure]
  primary_unguided = [
      c for c in primary_configs if not c.hinges_from_stick_figure
  ]
  fallback_guided = list(config.fallback_config_iter())
  fallback_guided_scale1 = [c for c in fallback_guided if c.scale_factor == 1]
  fallback_guided_scale2 = [c for c in fallback_guided if c.scale_factor > 1]

  can_retry_packing = (
      config.enable_recovery
      and config.enable_lazy_repack
      and isinstance(packing_outputs, PackingResult)
      and packing_outputs.stick_figure_input is not None
      and packing_outputs.packing_config is not None
  )

  recovery_ctx = _SolverRecoveryContext(
      initial_outputs=outputs_seq,
      all_tried_outputs=list(outputs_seq),
      primary_guided=primary_guided,
      primary_unguided=primary_unguided,
      fallback_guided_scale1=fallback_guided_scale1,
      fallback_guided_scale2=fallback_guided_scale2,
  )

  if not config.enable_recovery:
    primary_pairs = _cross_product_pairs(
        outputs_seq, primary_configs, is_fallback=False
    )
    _run_solver_sweep_pairs(
        primary_pairs,
        config=config,
        recovery_ctx=recovery_ctx,
        start_time=start_time,
        solver_outputs=solver_outputs,
        solver_errors=solver_errors,
        scaling_factors_used=scaling_factors_used,
        packing_outputs_used=packing_outputs_used,
        max_per_config_timeout_s=None,
        record_timeouts=False,
    )
  else:
    # Stage 1: Scale-1 guided sweep on the initial packed crease pattern(s).
    _run_scale1_guided_on_candidates(
        outputs_seq,
        config=config,
        recovery_ctx=recovery_ctx,
        primary_guided=primary_guided,
        fallback_guided_scale1=fallback_guided_scale1,
        start_time=start_time,
        solver_outputs=solver_outputs,
        solver_errors=solver_errors,
        scaling_factors_used=scaling_factors_used,
        packing_outputs_used=packing_outputs_used,
        primary_base_timeout_s=20.0 if can_retry_packing else None,
        fallback_base_timeout_s=10.0 if can_retry_packing else None,
    )

    # Stage 2: Lazily pack alternative candidates across up to 4 grid sizes.
    if (
        not solver_outputs
        and can_retry_packing
        and isinstance(packing_outputs, PackingResult)
    ):
      for cand_output in iter_additional_packing_candidates(
          packing_outputs,
          start_time=start_time,
          solve_timeout_seconds=config.timeout_seconds,
          max_extra_candidates=7,
          max_candidates_per_grid_size=2,
      ):
        recovery_ctx.all_tried_outputs.append(cand_output)
        _run_scale1_guided_on_candidates(
            [cand_output],
            config=config,
            recovery_ctx=recovery_ctx,
            primary_guided=primary_guided,
            fallback_guided_scale1=fallback_guided_scale1,
            start_time=start_time,
            solver_outputs=solver_outputs,
            solver_errors=solver_errors,
            scaling_factors_used=scaling_factors_used,
            packing_outputs_used=packing_outputs_used,
            primary_base_timeout_s=15.0,
            fallback_base_timeout_s=10.0,
        )
        if solver_outputs:
          break

    # Stage 3+: Declarative post-repack recovery schedule.
    for stage in _POST_REPACK_RECOVERY_STAGES:
      if solver_outputs:
        break
      if stage.requires_fallback_configs and not config.enable_fallback_configs:
        continue
      if stage.requires_timeout_retries and not config.enable_timeout_retries:
        continue
      if stage.min_remaining_budget_s is not None:
        rem_budget_s = _remaining_solve_budget_s(
            start_time, config.timeout_seconds
        )
        if (
            rem_budget_s is not None
            and rem_budget_s < stage.min_remaining_budget_s
        ):
          continue
      stage_pairs = stage.build_pairs(recovery_ctx)
      if stage_pairs:
        _run_solver_sweep_pairs(
            stage_pairs,
            config=config,
            recovery_ctx=recovery_ctx,
            start_time=start_time,
            solver_outputs=solver_outputs,
            solver_errors=solver_errors,
            scaling_factors_used=scaling_factors_used,
            packing_outputs_used=packing_outputs_used,
            max_per_config_timeout_s=stage.max_per_config_timeout_s,
            record_timeouts=stage.record_timeouts,
        )

  duration_seconds = time.perf_counter() - start_time
  if not solver_outputs:
    raise ValueError(
        '\n\nSOLVE_FLAT_FOLDABLE_CREASE_PATTERN: ' + '\n'.join(solver_errors)
    )

  cps = [s.crease_pattern for s in solver_outputs]
  avg_time_per_cp = duration_seconds / len(solver_outputs)

  return SolveResult(
      name=name,
      sequence_status='SUCCESS in SOLVE_FLAT_FOLDABLE_CREASE_PATTERN',
      flat_foldable_cps_count=len(solver_outputs),
      flat_foldable_cps=cps,
      scaling_factors_used=scaling_factors_used,
      solver_outputs=solver_outputs,
      packing_outputs=packing_outputs_used,
      time_taken=[s.time_taken for s in solver_outputs],
      duration_seconds=duration_seconds,
      avg_time_per_cp_seconds=avg_time_per_cp,
      solver_errors=solver_errors,
      solver_states=[s.final_state for s in solver_outputs],
  )


def pack_and_solve_crease_pattern(
    stick_figure_input: (
        StickFigureInput | stick_figure.StickFigure | bus.DesignTree
    ),
    packing_config: PackingTaskConfig | None = None,
    solving_config: SolvingTaskConfig | None = None,
) -> tuple[PackingResult, SolveResult]:
  """Packs and solves flat-foldable crease patterns for a stick figure."""
  packing_result = pack_crease_pattern(stick_figure_input, packing_config)
  solve_result = solve_crease_pattern(packing_result, solving_config)
  if solve_result.packing_outputs:
    packing_result = packing_result.with_promoted_packing(
        solve_result.packing_outputs[0]
    )
  return packing_result, solve_result


def should_include_shaped_cp(
    *,
    shaped_cp: crease_pattern.CreasePattern,
    filter_by_local_flat_foldability: bool,
) -> bool:
  """Checks if a shaped crease pattern should be included in the results."""
  return (
      not filter_by_local_flat_foldability or shaped_cp.local_flat_foldability()
  )


def get_hash_from_config(val: int, flag1: bool, flag2: bool, flag3: str) -> int:
  """Returns a hash of a solving configuration tuple."""
  return hash((val, flag1, flag2, flag3))


def _prepare_packing_for_shaping(
    packing_output_used: packing_output_lib.PackingOutput,
    solver_output: hinge_utils.SolverResult,
) -> packing_output_lib.PackingOutput:
  """Prepares a PackingOutput for shaping, shortening flaps if central crease removed."""
  if solver_output.final_state.solver_config.remove_central_crease:
    return central_fold_utils.repack_cp_with_shortened_flaps(
        packing_output_used,
        solver_output.crease_pattern.grid_size,
    )
  return packing_output_used


def _prepare_shaping_instances(
    task_data: (
        ShapingTaskInput
        | SolveResult
        | bus.SolvedCreasePattern
        | Sequence[bus.SolvedCreasePattern]
    ),
    *,
    stick_figure_input: (
        StickFigureInput | stick_figure.StickFigure | bus.DesignTree | None
    ) = None,
    name: str = 'crease_pattern',
) -> tuple[str, list[ShapingInstance]]:
  """Normalizes shaping inputs into `(task_name, list[ShapingInstance])`."""
  explicit_unmerged: stick_figure.StickFigure | None = None
  explicit_merged: stick_figure.StickFigure | None = None
  if isinstance(stick_figure_input, StickFigureInput):
    explicit_unmerged = stick_figure_input.unmerged_stick_figure.stick_figure
    explicit_merged = stick_figure_input.merged_stick_figure.stick_figure
    name = stick_figure_input.name or name
  elif isinstance(stick_figure_input, bus.DesignTree):
    explicit_unmerged = stick_figure_input.source
    explicit_merged = stick_figure_input.merged
    name = stick_figure_input.source.name or name
  elif isinstance(stick_figure_input, stick_figure.StickFigure):
    explicit_unmerged = stick_figure_input
    name = stick_figure_input.name or name

  if isinstance(task_data, ShapingTaskInput):
    instances: list[ShapingInstance] = []
    for inst in task_data.shaping_instances:
      bus_adapter.ensure_unmerged_flap_shapes(
          inst.tree, inst.packing.stick_figure
      )
      instances.append(
          ShapingInstance(
              tree=inst.tree,
              packing=inst.packing,
              solving=inst.solving,
          )
      )
    return task_data.name or name, instances

  if isinstance(task_data, SolveResult):
    task_name = task_data.name or name
    instances: list[ShapingInstance] = []
    for solver_output, packing_output_used in zip(
        task_data.solver_outputs,
        task_data.packing_outputs,
    ):
      tree = (
          explicit_unmerged
          or packing_output_used.unmerged_stick_figure
          or packing_output_used.stick_figure
      )
      if tree is None:
        raise ValueError(
            'Shaping requires a stick figure on PackingOutput or '
            'stick_figure_input.'
        )
      bus_adapter.ensure_unmerged_flap_shapes(
          tree, explicit_merged or packing_output_used.stick_figure
      )
      prepared_packing = _prepare_packing_for_shaping(
          packing_output_used, solver_output
      )
      instances.append(
          ShapingInstance(
              tree=tree,
              packing=prepared_packing,
              solving=solver_output,
          )
      )
    return task_name, instances

  solved_seq: Sequence[bus.SolvedCreasePattern]
  if isinstance(task_data, bus.SolvedCreasePattern):
    solved_seq = (task_data,)
  else:
    solved_seq = tuple(task_data)

  instances = []
  task_name = name
  for solved_cp in solved_seq:
    prepared_packing, solver_output = bus_adapter.legacy_inputs(solved_cp)
    if explicit_unmerged is not None:
      tree = copy.deepcopy(explicit_unmerged)
      merged_tree = (
          explicit_merged
          or prepared_packing.stick_figure
          or (
              solved_cp.packed.design.merged
              if solved_cp.packed.design is not None
              else None
          )
      )
      bus_adapter.ensure_unmerged_flap_shapes(tree, merged_tree)
    else:
      tree = (
          prepared_packing.unmerged_stick_figure
          or prepared_packing.stick_figure
      )
    if tree is None:
      raise ValueError(
          'Shaping requires a DesignTree on SolvedCreasePattern.packed or '
          'stick_figure_input.'
      )
    if tree.name and task_name == 'crease_pattern':
      task_name = tree.name
    instances.append(
        ShapingInstance(
            tree=tree,
            packing=prepared_packing,
            solving=solver_output,
        )
    )
  return task_name, instances


def _shape_instance_once(
    *,
    cp: crease_pattern.CreasePattern,
    stick_figure_tree: stick_figure.StickFigure,
    packing: packing_output_lib.PackingOutput,
    solution: hinge_utils.SolverResult,
    solving_grid_size: int,
    was_central_crease_removed: bool,
    config: ShapingTaskConfig,
) -> tuple[crease_pattern.CreasePattern, str]:
  """Runs a single 3D geometric shaping configuration on one instance."""
  # `cp` is already a grid-snapped, split CreasePattern; copy cheaply rather
  # than rerunning full non-light CreasePattern construction per candidate.
  base_cp = copy.copy(cp)
  base_cp.tolerance = config.cp_tolerance
  shaped_cp = base_cp
  strategy_name = 'none'

  if config.apply_simple_fold_shaping:
    strategy_name = config.fold_strategy_3d.name.lower()
    shaped_cp = geometric_fold_3d_heuristics.shape_cp_with_3d_geometric_folding_heuristics_pipeline_wrapper(
        base_cp=base_cp,
        stick_figure=stick_figure_tree,
        packing_output=packing,
        solving_output=solution,
        solving_grid_size=solving_grid_size,
        central_crease_removed=was_central_crease_removed,
        shape_with_unmerged_tree=config.shape_with_unmerged_tree,
        hparams=geometric_fold_3d_heuristics.GeometricFolding3DHParams(
            default_inscribe_hparams=geometric_fold_3d_heuristics.InscribeHParams(
                avoid_hinges=True,
                clip_slope_thresh=config.clip_slope_thresh,
                max_slope_thresh=config.max_slope_thresh,
            ),
            strategy=config.fold_strategy_3d,
            check_ff_violations=True,
        ),
    )

  return shaped_cp, strategy_name


def shape_crease_pattern(
    task_data: (
        ShapingTaskInput
        | SolveResult
        | bus.SolvedCreasePattern
        | Sequence[bus.SolvedCreasePattern]
    ),
    config: ShapingTaskConfig | None = None,
    *,
    stick_figure_input: (
        StickFigureInput | stick_figure.StickFigure | bus.DesignTree | None
    ) = None,
    name: str = 'crease_pattern',
) -> ShapingTaskResult:
  """Shapes solved crease patterns using 3D geometric fold heuristics."""
  if config is None:
    config = ShapingTaskConfig()

  task_name, shaping_instances = _prepare_shaping_instances(
      task_data, stick_figure_input=stick_figure_input, name=name
  )
  start_time = time.perf_counter()

  shaped_cps_before_filtering: list[crease_pattern.CreasePattern] = []
  solving_configs_before_filtering: list[int] = []
  solving_indices_before_filtering: list[int] = []
  tree_scores_before_filtering: list[float] = []
  strategies_before_filtering: list[str] = []
  time_taken: list[float] = []
  shaping_errors: list[str] = []

  swept_configs = list(config.config_iter())

  for i_instance, instance in enumerate(shaping_instances):
    if config.timeout_seconds is not None and (
        time.perf_counter() - start_time >= config.timeout_seconds
    ):
      shaping_errors.append(
          f'Shaping stage timeout {config.timeout_seconds}s reached.'
      )
      break
    if timeout.check_deadline().expired:
      shaping_errors.append('Outer cooperative timeout reached during shaping.')
      break

    stick_figure_tree = instance.tree
    solution = instance.solving
    cp = solution.crease_pattern
    packing = instance.packing
    packing_cp = packing.packing_cp
    if not isinstance(packing_cp, crease_pattern.CreasePattern):
      raise ValueError('Shaping instance packing_cp must be a CreasePattern.')
    if not isinstance(packing.flap_rectangles, list):
      shaping_errors.append(
          'ValueError: Shaping instance flap_rectangles must be a list.'
      )
      continue
    was_central_crease_removed = (
        solution.final_state.solver_config.remove_central_crease
    )
    solving_grid_size = cp.grid_size

    for active_cfg in swept_configs:
      instance_start = time.perf_counter()
      candidate_configs = [active_cfg]
      if (
          active_cfg.enable_strategy_fallback
          and not active_cfg.use_sweep
          and active_cfg.apply_simple_fold_shaping
      ):
        for alt_strategy in (
            heuristics_3d.FoldStrategy3D.DEFAULT,
            heuristics_3d.FoldStrategy3D.ROTATE_BEND,
            heuristics_3d.FoldStrategy3D.SHALLOW_FOLDS,
        ):
          if alt_strategy != active_cfg.fold_strategy_3d:
            candidate_configs.append(
                dataclasses.replace(active_cfg, fold_strategy_3d=alt_strategy)
            )

      chosen_cp: crease_pattern.CreasePattern | None = None
      chosen_strategy = active_cfg.fold_strategy_3d.name.lower()
      for cand_idx, cand_cfg in enumerate(candidate_configs):
        try:
          shaped_cp, strategy_name = _shape_instance_once(
              cp=cp,
              stick_figure_tree=stick_figure_tree,
              packing=packing,
              solution=solution,
              solving_grid_size=solving_grid_size,
              was_central_crease_removed=was_central_crease_removed,
              config=cand_cfg,
          )
          if chosen_cp is None:
            chosen_cp = shaped_cp
            chosen_strategy = strategy_name
          if should_include_shaped_cp(
              shaped_cp=shaped_cp,
              filter_by_local_flat_foldability=(
                  cand_cfg.filter_by_local_flat_foldability
              ),
          ):
            chosen_cp = shaped_cp
            chosen_strategy = strategy_name
            break
        except (ValueError, AssertionError, IndexError):
          if cand_idx == 0:
            error_trace = traceback.format_exc()
            shaping_errors.append(error_trace)
            logging.info(
                'Failed to shape crease pattern for %s: %s',
                task_name,
                error_trace,
            )

      if chosen_cp is not None:
        shaped_cps_before_filtering.append(chosen_cp)
        solving_configs_before_filtering.append(
            get_hash_from_config(
                solving_grid_size,
                was_central_crease_removed,
                solution.final_state.solver_config.hinges_from_stick_figure,
                solution.final_state.solver_config.up_down_method,
            )
        )
        solving_indices_before_filtering.append(i_instance)
        tree_scores_before_filtering.append(0.0)
        strategies_before_filtering.append(chosen_strategy)
        time_taken.append(time.perf_counter() - instance_start)

  shaped_cps: list[crease_pattern.CreasePattern] = []
  solving_indices: list[int] = []
  solving_configs_dict: collections.Counter[int] = collections.Counter()
  tree_similarity_scores: list[float] = []
  strategies_used: list[str] = []
  for shaped_cp, solving_config_hash, i_instance, tree_score, strat in zip(
      shaped_cps_before_filtering,
      solving_configs_before_filtering,
      solving_indices_before_filtering,
      tree_scores_before_filtering,
      strategies_before_filtering,
  ):
    if should_include_shaped_cp(
        shaped_cp=shaped_cp,
        filter_by_local_flat_foldability=config.filter_by_local_flat_foldability,
    ):
      if solving_configs_dict[solving_config_hash] < config.max_number_of_cps:
        shaped_cps.append(shaped_cp)
        solving_indices.append(i_instance)
        tree_similarity_scores.append(tree_score)
        strategies_used.append(strat)
        solving_configs_dict[solving_config_hash] += 1

  duration_seconds = time.perf_counter() - start_time
  if not shaped_cps_before_filtering:
    avg_time_per_cp = 0.0
  else:
    avg_time_per_cp = duration_seconds / len(shaped_cps_before_filtering)

  result = ShapingTaskResult(
      name=task_name,
      sequence_status='SUCCESS in SHAPING',
      shaped_cps=shaped_cps,
      solving_indices=solving_indices,
      shaping_indices=solving_indices_before_filtering,
      time_taken=time_taken,
      duration_seconds=duration_seconds,
      avg_time_per_cp_seconds=avg_time_per_cp,
      shaping_errors=shaping_errors,
      tree_similarity_scores=tree_similarity_scores,
      strategies_used=strategies_used,
  )
  logging.info(
      'Finished task-sequence SHAPING in %.4f seconds. Shaped %d '
      'crease-patterns. Average time per CP: %.4f seconds.',
      result.duration_seconds,
      len(shaped_cps),
      result.avg_time_per_cp_seconds,
  )
  return result
