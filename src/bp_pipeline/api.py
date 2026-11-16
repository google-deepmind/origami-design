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

"""Public user-facing API for Box-Pleating Origami packing and solving."""

from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
import copy
import dataclasses
import json
import pathlib
import time
from typing import Any, Literal

from . import bus
from . import codec
from . import export as export_lib
from . import pipeline
from . import plugins
from . import render_3d
from . import render_summary
from . import viz
from .crease_pattern import crease as crease_lib
from .crease_pattern import crease_pattern as cp_lib
from .packing import config as packing_config_lib
from .packing import stacking as stacking_lib
from .shaping import fidelity as fidelity_lib
from .shaping import heuristics_3d
from .solving import hinges as hinge_utils
from .stick_figures import stick as stick_lib
from .stick_figures import stick_figure as stick_figure_lib
from .stick_figures.examples import example_trees

Preset = Literal['demo', 'fast', 'production']

DesignTree = bus.DesignTree
PackedCreasePattern = bus.PackedCreasePattern
SolvedCreasePattern = bus.SolvedCreasePattern
ShapedCreasePattern = bus.ShapedCreasePattern
SolveStage = bus.SolveStage
StageSnapshot = bus.StageSnapshot
Stick = stick_lib.Stick
StickFigure = stick_figure_lib.StickFigure
Crease = crease_lib.Crease
Fold = crease_lib.Fold
CreasePattern = cp_lib.CreasePattern
PackingConfig = packing_config_lib.PackingConfig
StackLayoutConfig = stacking_lib.StackLayoutConfig
SolverConfig = hinge_utils.SolverConfig
FoldStrategy3D = heuristics_3d.FoldStrategy3D
PackingTaskConfig = pipeline.PackingTaskConfig
SolvingTaskConfig = pipeline.SolvingTaskConfig
ShapingTaskConfig = pipeline.ShapingTaskConfig
register_packer = plugins.register_packer
register_solver = plugins.register_solver
list_packers = plugins.list_packers
list_solvers = plugins.list_solvers
PipelineConfig = pipeline.PipelineConfig
ShapingStrategy = bus.ShapingStrategy

from_dict = codec.from_dict
to_dict = codec.to_dict
dumps = codec.dumps
loads = codec.loads
render_packing_grid = viz.render_packing_grid
DEFAULT_SVG_COLORS = export_lib.DEFAULT_SVG_COLORS
to_bp_studio = export_lib.to_bp_studio
save_bp_studio = export_lib.save_bp_studio
from_bp_studio = export_lib.from_bp_studio
import_bps = export_lib.import_bps
BpsImportResult = export_lib.BpsImportResult
to_svg = export_lib.to_svg
save_svg = export_lib.save_svg
to_fold = export_lib.to_fold
save_fold = export_lib.save_fold
from_fold = export_lib.from_fold
render_stick_figure_svg = viz.render_stick_figure_svg
render_packed_overlay_svg = viz.render_packed_overlay_svg
stage_display_name = viz.stage_display_name
FoldedMesh3D = render_3d.FoldedMesh3D
FoldedModel = fidelity_lib.FoldedModel
build_folded_mesh_3d = render_3d.build_folded_mesh_3d
evaluate_fidelity = fidelity_lib.evaluate_fidelity
mesh_to_fold_dict = render_3d.mesh_to_fold_dict
save_3d_turnaround = render_3d.save_3d_turnaround
save_stick_figure_views = render_summary.save_stick_figure_views
save_pipeline_summary = render_summary.save_pipeline_summary

DEFAULT_SOLVER_CONFIG: hinge_utils.SolverConfig = (
    pipeline.SolvingTaskConfig().to_solver_config()
)
PRODUCTION_SWEEP: tuple[hinge_utils.SolverConfig, ...] = tuple(
    c.to_solver_config() for c in pipeline.SolvingTaskConfig().config_iter()
)


class PackingFailedError(RuntimeError):
  """Raised when packing cannot find a valid crease pattern layout."""

  def __init__(
      self,
      message: str,
      *,
      errors: Sequence[str] = (),
  ):
    super().__init__(message)
    self.errors: tuple[str, ...] = tuple(errors)


class SolvingFailedError(RuntimeError):
  """Raised when hinge solving cannot find a flat-foldable crease pattern."""

  def __init__(
      self,
      message: str,
      *,
      errors: Sequence[str] = (),
  ):
    super().__init__(message)
    self.errors: tuple[str, ...] = tuple(errors)


class ShapingFailedError(RuntimeError):
  """Raised when shaping cannot produce a valid shaped crease pattern."""

  def __init__(
      self,
      message: str,
      *,
      errors: Sequence[str] = (),
  ):
    super().__init__(message)
    self.errors: tuple[str, ...] = tuple(errors)


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class PipelineResult(bus.PipelineResult):
  """Typed end-to-end result of `fold(...)`."""

  def to_bp_studio(self, **kwargs: Any) -> Any:
    """Exports this pipeline result to BoxPleatingTreeData."""
    return export_lib.to_bp_studio(self, **kwargs)

  def save_bp_studio(
      self, path_or_file: export_lib.FileTarget, **kwargs: Any
  ) -> None:
    """Writes this pipeline result to a `.bps.json` file."""
    export_lib.save_bp_studio(self, path_or_file, **kwargs)

  def save_svg(
      self, path_or_file: export_lib.FileTarget, **kwargs: Any
  ) -> None:
    """Writes the solved crease pattern to an SVG file."""
    export_lib.save_svg(self, path_or_file, **kwargs)

  def to_fold(self, **kwargs: Any) -> dict[str, Any]:
    """Exports the solved crease pattern to FOLD v1.2 format."""
    return export_lib.to_fold(self, **kwargs)

  def save_fold(
      self, path_or_file: export_lib.FileTarget, **kwargs: Any
  ) -> None:
    """Writes the solved crease pattern to a `.fold` JSON file."""
    export_lib.save_fold(self, path_or_file, **kwargs)

  def save_3d_turnaround(
      self, out_path: str | pathlib.Path, **kwargs: Any
  ) -> render_3d.FoldedMesh3D:
    """Folds the shaped (or solved) pattern and saves a 3D turnaround PNG."""
    target = self.shaped if self.shaped is not None else self.solved
    return render_3d.save_3d_turnaround(target, out_path, **kwargs)

  def save_pipeline_summary(
      self, out_path: str | pathlib.Path, **kwargs: Any
  ) -> pathlib.Path:
    """Saves a 5-panel end-to-end summary PNG for this pipeline result."""
    return render_summary.save_pipeline_summary(
        out_path,
        packed=self.packed,
        solved=self.solved,
        shaped=self.shaped,
        **kwargs,
    )


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class _PresetSpec:
  packing_config: pipeline.PackingTaskConfig

  sweep_configs: tuple[hinge_utils.SolverConfig, ...]
  stop_on_first_solve: bool


_FAST_PACKING_SUB_CONFIG = packing_config_lib.PackingConfig(
    use_level_shifter=False,
    enforce_proximity_constraints=False,
)

_PRESETS: Mapping[Preset, _PresetSpec] = {
    'demo': _PresetSpec(
        packing_config=pipeline.PackingTaskConfig(
            packing_config=_FAST_PACKING_SUB_CONFIG,
            max_retries=5,
            max_cps_per_grid_size=1,
            max_number_of_cps=1,
            sweep_timeout_seconds=15.0,
        ),
        sweep_configs=(
            hinge_utils.SolverConfig(
                max_steps=1500,
                max_time_limit=20.0,
                max_number_of_cps=1,
                disallow_trapping_edge_flaps=True,
                hinges_from_stick_figure=True,
                remove_central_crease=True,
                up_down_method='azimuth',
            ),
            hinge_utils.SolverConfig(
                max_steps=1500,
                max_time_limit=20.0,
                max_number_of_cps=1,
                disallow_trapping_edge_flaps=True,
                hinges_from_stick_figure=False,
                remove_central_crease=False,
                up_down_method='flaps_down',
            ),
        ),
        stop_on_first_solve=True,
    ),
    'fast': _PresetSpec(
        packing_config=pipeline.PackingTaskConfig(
            packing_config=_FAST_PACKING_SUB_CONFIG,
            max_retries=10,
            max_cps_per_grid_size=1,
            max_number_of_cps=1,
            sweep_timeout_seconds=30.0,
        ),
        sweep_configs=tuple(
            dataclasses.replace(
                cfg, max_steps=3000, max_time_limit=60.0, max_number_of_cps=1
            )
            for cfg in PRODUCTION_SWEEP
        ),
        stop_on_first_solve=True,
    ),
    'production': _PresetSpec(
        packing_config=pipeline.PackingTaskConfig(
            max_cps_per_grid_size=1,
            max_number_of_cps=1,
        ),
        sweep_configs=PRODUCTION_SWEEP,
        stop_on_first_solve=False,
    ),
}


def list_examples() -> list[str]:
  """Returns the names of all built-in example stick figures."""
  return [example.name for example in example_trees.list_examples()]


def load_example(name: str) -> stick_figure_lib.StickFigure:
  """Loads a deep copy of a built-in example stick figure by name or constant."""
  matches = [e for e in example_trees.list_examples() if e.name == name]
  if len(matches) == 1:
    return copy.deepcopy(matches[0])
  candidate = getattr(example_trees, name.upper(), None)
  if isinstance(candidate, stick_figure_lib.StickFigure):
    return copy.deepcopy(candidate)
  raise ValueError(
      f'Unknown example stick figure {name!r}. Available examples:'
      f' {list_examples()}'
  )


def prepare(
    source: (
        stick_figure_lib.StickFigure | bus.DesignTree | str | Mapping[str, Any]
    ),
) -> bus.DesignTree:
  """Normalizes a StickFigure, DesignTree, JSON string, dict, or example name."""
  if isinstance(source, bus.DesignTree):
    return source
  if isinstance(source, stick_figure_lib.StickFigure):
    return bus.DesignTree.from_stick_figure(copy.deepcopy(source))
  if isinstance(source, Mapping):
    if 'schema_version' in source and 'type' in source:
      decoded = codec.from_dict(source)
      if isinstance(decoded, bus.DesignTree):
        return decoded
      if isinstance(decoded, stick_figure_lib.StickFigure):
        return bus.DesignTree.from_stick_figure(decoded)
      raise ValueError(
          f'Expected StickFigure or DesignTree dict, got {type(decoded)!r}'
      )
    sf = codec.stick_figure_from_dict(source)
    return bus.DesignTree.from_stick_figure(sf)
  if isinstance(source, str):
    stripped = source.strip()
    if stripped.startswith('{'):
      raw = json.loads(stripped)
      if isinstance(raw, Mapping):
        return prepare(raw)
      raise ValueError('JSON input to prepare() must decode to an object.')
    return bus.DesignTree.from_stick_figure(load_example(source))
  raise TypeError(
      'prepare() expected StickFigure, DesignTree, dict, JSON string, or'
      f' example name; got {type(source)!r}'
  )


def _resolve_packing_task_config(
    config: (
        pipeline.PackingTaskConfig | packing_config_lib.PackingConfig | None
    ),
) -> pipeline.PackingTaskConfig:
  """Normalizes a packing configuration into a PackingTaskConfig."""
  if config is None:
    return pipeline.PackingTaskConfig(
        max_cps_per_grid_size=1,
        max_number_of_cps=1,
    )
  if isinstance(config, packing_config_lib.PackingConfig):
    return pipeline.PackingTaskConfig(
        packing_config=config,
        max_cps_per_grid_size=1,
        max_number_of_cps=1,
    )
  if isinstance(config, pipeline.PackingTaskConfig):
    return config
  raise TypeError(
      'Expected PackingTaskConfig, PackingConfig, or None; got'
      f' {type(config)!r}'
  )


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class PackerOptions:
  """Generic execution options for packing engines in `bp.pack(...)`.

  Attributes:
    timeout_seconds: Overall packing time budget in seconds.
    per_grid_timeout_seconds: Optional per-grid-size time budget in seconds.
    max_solutions: Maximum candidate packings to collect per grid size.
    grid_size: Explicit target grid size override (or None to search upward from
      the tree's minimum packing grid size).
    num_workers: Optional worker-thread count for parallel packers.
  """

  timeout_seconds: float | None = None
  per_grid_timeout_seconds: float | None = None
  max_solutions: int | None = None
  grid_size: int | None = None
  num_workers: int | None = None


def _resolve_packer_options(
    packer_options: PackerOptions | None,
    *,
    task_cfg: pipeline.PackingTaskConfig,
    design: bus.DesignTree,
) -> PackerOptions:
  """Resolves generic packer execution options against `task_cfg` defaults."""
  opts = packer_options if packer_options is not None else PackerOptions()
  timeout_seconds = opts.timeout_seconds
  if timeout_seconds is None:
    timeout_seconds = (
        task_cfg.timeout_seconds
        if task_cfg.timeout_seconds is not None
        else task_cfg.sweep_timeout_seconds
    )
  if timeout_seconds is None:
    timeout_seconds = 60.0

  max_solutions = (
      opts.max_solutions
      if opts.max_solutions is not None
      else (task_cfg.max_cps_per_grid_size or 1)
  )

  target_grid = opts.grid_size
  # When a caller pins `PackingTaskConfig` to a single non-zero grid offset
  # (`max_retries == 1` and `extra_grid_size != 0`), forward that exact grid
  # size to registered plugins rather than sweeping upward from minimum grid.
  # Symmetric trees increment in steps of 2 around the central symmetry axis.
  if (
      target_grid is None
      and task_cfg.extra_grid_size != 0
      and task_cfg.max_retries == 1
  ):
    base_g = design.source.get_packing_gridsize()
    extra = task_cfg.extra_grid_size
    target_grid = base_g + (
        extra * 2 if design.source.is_symmetric_tree else extra
    )

  return PackerOptions(
      timeout_seconds=timeout_seconds,
      per_grid_timeout_seconds=opts.per_grid_timeout_seconds,
      max_solutions=max_solutions,
      grid_size=target_grid,
      num_workers=opts.num_workers,
  )


def _apply_packer_options_to_default_config(
    task_cfg: pipeline.PackingTaskConfig,
    opts: PackerOptions,
    *,
    design: bus.DesignTree,
) -> pipeline.PackingTaskConfig:
  """Applies explicit `PackerOptions` fields to a default `PackingTaskConfig`."""
  updates: dict[str, Any] = {}
  if opts.timeout_seconds is not None:
    updates['timeout_seconds'] = opts.timeout_seconds
  if opts.per_grid_timeout_seconds is not None:
    updates['sweep_timeout_seconds'] = opts.per_grid_timeout_seconds
  elif (
      opts.timeout_seconds is not None
      and task_cfg.sweep_timeout_seconds is not None
      and opts.timeout_seconds < task_cfg.sweep_timeout_seconds
  ):
    updates['sweep_timeout_seconds'] = opts.timeout_seconds
  if opts.max_solutions is not None:
    updates['max_cps_per_grid_size'] = opts.max_solutions
    updates['max_number_of_cps'] = max(
        task_cfg.max_number_of_cps, opts.max_solutions
    )
  if opts.grid_size is not None:
    base_g = design.source.get_packing_gridsize()
    delta = opts.grid_size - base_g
    step = 2 if design.source.is_symmetric_tree else 1
    if delta < 0 or delta % step != 0:
      raise PackingFailedError(
          f'Invalid grid_size={opts.grid_size} for base grid {base_g}'
          f' (step={step}).'
      )
    updates['extra_grid_size'] = delta // step
    updates['max_retries'] = 1
  return dataclasses.replace(task_cfg, **updates) if updates else task_cfg


def pack(
    design: bus.DesignTree,
    config: (
        pipeline.PackingTaskConfig | packing_config_lib.PackingConfig | None
    ) = None,
    *,
    packer: str | None = None,
    packer_options: PackerOptions | None = None,
    packer_config: Any = None,
    solution_callback: (
        Callable[[bus.PackedCreasePattern], bool | None] | None
    ) = None,
) -> bus.PackedCreasePattern:
  """Packs a DesignTree into an immutable PackedCreasePattern.

  Args:
    design: DesignTree to pack (call bp.prepare(...) first).
    config: Optional packing task configuration or PackingConfig preset.
    packer: Optional packing engine name ('default', or a custom packer
      registered via register_packer). Overrides `config.packer` when provided.
    packer_options: Optional engine-agnostic `PackerOptions` controlling
      timeout, solution limits, target grid size, and worker count.
    packer_config: Optional opaque plugin-specific configuration passed through
      untouched to the registered packer plugin. Overrides
      `config.packer_config` when provided.
    solution_callback: Optional callback that observes or rejects each candidate
      packing (returning False drops it); cannot transform the solution.

  Returns:
    The winning PackedCreasePattern.

  Raises:
    TypeError: If design is not a DesignTree or packer_options is not a
      PackerOptions instance.
    PackingFailedError: If packing fails or no valid packing is found.
  """
  if not isinstance(design, bus.DesignTree):
    raise TypeError(
        'pack() expected DesignTree (call bp.prepare(...) first), got'
        f' {type(design)!r}'
    )
  if packer_options is not None and not isinstance(
      packer_options, PackerOptions
  ):
    raise TypeError(
        'pack() expected packer_options to be PackerOptions or None, got'
        f' {type(packer_options)!r}'
    )
  task_cfg = _resolve_packing_task_config(config)
  effective_packer = packer if packer is not None else task_cfg.packer
  effective_packer_config = (
      packer_config if packer_config is not None else task_cfg.packer_config
  )

  task_cfg = dataclasses.replace(
      task_cfg,
      packer=effective_packer,
      packer_config=effective_packer_config,
  )
  raw_callback = None
  if solution_callback is not None:

    def _raw_callback(output: Any) -> Any:
      packed = bus.PackedCreasePattern.from_packing_output(
          output, design=design
      )
      res = solution_callback(packed)
      if res is not None and not res:
        return None
      return output

    raw_callback = _raw_callback

  if effective_packer != 'default':
    opts = _resolve_packer_options(
        packer_options, task_cfg=task_cfg, design=design
    )
    packer_fn = plugins.get_packer(effective_packer)
    if packer_fn is None:
      raise ValueError(
          f'Unknown or unregistered packer {effective_packer!r}. '
          f'Available: {plugins.list_packers()}'
      )
    try:
      output = packer_fn(
          design,
          config=task_cfg.packing_config,
          grid_size=opts.grid_size,
          max_solutions=opts.max_solutions,
          timeout_seconds=opts.timeout_seconds,
          per_grid_timeout_seconds=opts.per_grid_timeout_seconds,
          num_workers=opts.num_workers,
          packer_config=effective_packer_config,
          solution_callback=raw_callback,
      )
    except Exception as err:
      raise PackingFailedError(str(err), errors=(str(err),)) from err

    if output is None or not getattr(output, 'success', True):
      err_msg = (
          getattr(output, 'error_message', None)
          or f'{effective_packer} packing failed'
      )
      raise PackingFailedError(err_msg, errors=(err_msg,))
    return bus.PackedCreasePattern.from_packing_output(
        output,
        design=design,
    )

  if packer_options is not None:
    task_cfg = _apply_packer_options_to_default_config(
        task_cfg, packer_options, design=design
    )

  try:
    result = pipeline.pack_crease_pattern(
        design, config=task_cfg, solution_callback=raw_callback
    )
  except ValueError as err:
    raise PackingFailedError(str(err), errors=(str(err),)) from err
  return bus.PackedCreasePattern.from_packing_output(
      result.packing_outputs[0],
      design=design,
  )


def stack(
    design: bus.DesignTree,
    *,
    force_asymmetric: bool = False,
    stacking_config: stacking_lib.StackLayoutConfig | None = None,
) -> bus.PackedCreasePattern:
  """Packs a DesignTree using the standalone 1D strip-stacking layout utility."""
  if not isinstance(design, bus.DesignTree):
    raise TypeError(
        'stack() expected DesignTree (call bp.prepare(...) first), got'
        f' {type(design)!r}'
    )
  effective_config = (
      stacking_config
      if stacking_config is not None
      else stacking_lib.StackLayoutConfig()
  )
  try:
    packing_output = stacking_lib.pack_stick_figure_with_stacking(
        copy.deepcopy(design.source),
        force_asymmetric=force_asymmetric,
        stacking_config=effective_config,
    )
  except Exception as err:
    raise PackingFailedError(str(err), errors=(str(err),)) from err
  return bus.PackedCreasePattern.from_packing_output(
      packing_output,
      design=design,
  )


def _require_packed_cp(packed: bus.PackedCreasePattern) -> None:
  if not isinstance(packed, bus.PackedCreasePattern):
    raise TypeError(
        'Expected PackedCreasePattern (use'
        ' PackedCreasePattern.from_packing_output(...) or'
        f' PackedCreasePattern.from_crease_pattern(...)), got {type(packed)!r}'
    )


def solve(
    packed: bus.PackedCreasePattern,
    config: hinge_utils.SolverConfig | None = None,
    *,
    scale_factor: int = 1,
    record_stages: bool = False,
    solver: str | None = None,
) -> Iterator[bus.SolvedCreasePattern]:
  """Streams SolvedCreasePattern candidates as the hinge solver discovers them."""
  _require_packed_cp(packed)
  if scale_factor <= 0:
    raise ValueError(f'scale_factor must be positive, got {scale_factor}')
  effective_config = config if config is not None else DEFAULT_SOLVER_CONFIG
  if solver is not None:
    effective_config = dataclasses.replace(effective_config, solver=solver)
  scaled_grid_size = int(packed.grid_size * scale_factor)
  try:
    solver = effective_config.make_from_cp(
        packed.creases,
        grid_size=scaled_grid_size,
        packing_information=packed,
        record_stages=record_stages,
    )
  except hinge_utils.NoSolutionError:
    return
  except (ValueError, IndexError) as err:
    raise SolvingFailedError(str(err), errors=(str(err),)) from err

  try:
    for sol in solver.iter_solutions():
      stages: tuple[bus.StageSnapshot, ...] = ()
      if record_stages and solver.initial_stage_snapshots:
        final_cps = (
            solver.borders
            + sol.final_state.ridges
            + sol.final_state.pleats
            + sol.final_state.hinges
        )
        stage7_cp = cp_lib.CreasePattern(final_cps, grid_size=scaled_grid_size)
        stages = tuple(
            bus.StageSnapshot(
                name=name, creases=stage_cp, grid_size=scaled_grid_size
            )
            for name, stage_cp in (
                *solver.initial_stage_snapshots,
                ('(7) Final Solving CP', stage7_cp),
            )
        )
      yield bus.SolvedCreasePattern.from_solver_result(
          sol,
          packed=packed,
          config=solver.config,
          scale_factor=scale_factor,
          stages=stages,
      )
  except hinge_utils.NoSolutionError:
    return
  except (ValueError, IndexError) as err:
    raise SolvingFailedError(str(err), errors=(str(err),)) from err


def _sort_solved_candidates(
    candidates: list[bus.SolvedCreasePattern],
    *,
    sort_solutions: bool,
) -> list[bus.SolvedCreasePattern]:
  if not sort_solutions:
    return candidates
  return sorted(
      candidates,
      key=lambda c: c.config.solution_score(c),
      reverse=True,
  )


def solve_best(
    packed: bus.PackedCreasePattern,
    config: hinge_utils.SolverConfig | None = None,
    *,
    scale_factor: int = 1,
    record_stages: bool = False,
    solver: str | None = None,
) -> bus.SolvedCreasePattern:
  """Solves for flat-foldable crease patterns and returns the highest-scoring one."""
  _require_packed_cp(packed)
  effective_config = config if config is not None else DEFAULT_SOLVER_CONFIG
  if solver is not None:
    effective_config = dataclasses.replace(effective_config, solver=solver)
  candidates = list(
      solve(
          packed,
          effective_config,
          scale_factor=scale_factor,
          record_stages=record_stages,
      )
  )
  if not candidates:
    raise SolvingFailedError(
        'No flat-foldable crease pattern found for the given configuration.'
    )
  sorted_candidates = _sort_solved_candidates(
      candidates, sort_solutions=effective_config.sort_solutions
  )
  return sorted_candidates[0]


# Doubling a packing grid larger than 44 yields an 88+ crease grid whose
# branching factor exceeds interactive fallback budgets.
_MAX_FALLBACK_UPSCALE_GRID_SIZE = 44
_MAX_FALLBACK_UPSCALE_STEPS = 2500


def _iter_fallback_configs(
    base_configs: Sequence[hinge_utils.SolverConfig],
    *,
    base_scale_factor: int,
    grid_size: int,
) -> Iterator[tuple[hinge_utils.SolverConfig, int]]:
  """Yields (SolverConfig, scale_factor) fallback pairs from guided configs."""
  guided = [c for c in base_configs if c.hinges_from_stick_figure]
  if not guided:
    guided = list(base_configs[:1])
  for cfg in guided:
    yield (
        dataclasses.replace(
            cfg,
            disallow_trapping_edge_flaps=False,
            allow_misaligned_symmetric_hinges=True,
        ),
        base_scale_factor,
    )
  if base_scale_factor == 1 and grid_size <= _MAX_FALLBACK_UPSCALE_GRID_SIZE:
    for cfg in guided:
      yield (
          dataclasses.replace(
              cfg,
              disallow_trapping_edge_flaps=False,
              allow_misaligned_symmetric_hinges=True,
              max_steps=min(cfg.max_steps, _MAX_FALLBACK_UPSCALE_STEPS),
          ),
          2,
      )


def _run_solver_configs(
    packed: bus.PackedCreasePattern,
    config_scale_pairs: Iterable[tuple[hinge_utils.SolverConfig, int]],
    *,
    stop_after_first: bool,
    errors: list[str],
) -> list[bus.SolvedCreasePattern]:
  """Runs `(config, scale_factor)` pairs on `packed`, gathering solutions."""
  collected: list[bus.SolvedCreasePattern] = []
  for cfg, sf in config_scale_pairs:
    try:
      found = list(solve(packed, cfg, scale_factor=sf))
    except SolvingFailedError as err:
      errors.extend(err.errors or (str(err),))
      continue
    if found:
      collected.extend(
          _sort_solved_candidates(found, sort_solutions=cfg.sort_solutions)
      )
      if stop_after_first:
        break
  return collected


def solve_sweep(
    packed: bus.PackedCreasePattern,
    configs: Sequence[hinge_utils.SolverConfig] = PRODUCTION_SWEEP,
    *,
    scale_factor: int = 1,
    solver: str | None = None,
) -> list[bus.SolvedCreasePattern]:
  """Runs a sweep of solver configurations and returns all discovered solutions."""
  _require_packed_cp(packed)
  if solver is not None:
    configs = [dataclasses.replace(c, solver=solver) for c in configs]
  errors: list[str] = []
  guided_configs = [c for c in configs if c.hinges_from_stick_figure]
  unguided_configs = [c for c in configs if not c.hinges_from_stick_figure]

  all_solutions = _run_solver_configs(
      packed,
      ((cfg, scale_factor) for cfg in guided_configs),
      stop_after_first=False,
      errors=errors,
  )
  if not all_solutions and guided_configs:
    all_solutions = _run_solver_configs(
        packed,
        _iter_fallback_configs(
            guided_configs,
            base_scale_factor=scale_factor,
            grid_size=packed.grid_size,
        ),
        stop_after_first=True,
        errors=errors,
    )
  if not all_solutions:
    all_solutions = _run_solver_configs(
        packed,
        ((cfg, scale_factor) for cfg in unguided_configs),
        stop_after_first=True,
        errors=errors,
    )

  if not all_solutions:
    raise SolvingFailedError(
        f'No flat-foldable crease pattern found across {len(configs)} sweep'
        ' configurations.',
        errors=errors,
    )
  return all_solutions


def fold(
    stick_figure_or_json: (
        stick_figure_lib.StickFigure | bus.DesignTree | str | Mapping[str, Any]
    ),
    *,
    preset: Preset = 'demo',
    packer: str | None = None,
    solver: str | None = None,
    shape: bool = False,  # pylint: disable=redefined-outer-name
    shaping_config: pipeline.ShapingTaskConfig | None = None,
) -> PipelineResult:
  """One-call end-to-end box-pleating packing, solving, and optional shaping."""
  if preset not in _PRESETS:
    raise ValueError(
        f'Unknown preset {preset!r}. Expected one of {tuple(_PRESETS)}.'
    )
  spec = _PRESETS[preset]
  design = prepare(stick_figure_or_json)
  task_cfg = _resolve_packing_task_config(spec.packing_config)
  if packer is not None:
    task_cfg = dataclasses.replace(task_cfg, packer=packer)
  try:
    if packer is not None:
      packed = pack(design, config=task_cfg, packer=packer)
      packing_result = pipeline.PackingResult(
          name=design.source.name,
          sequence_status='SUCCESS in CREASE_PATTERN_PACKING',
          packed_cps_count=1,
          packed_cps=[packed.creases],
          packing_outputs=[packed.to_packing_output()],
          duration_seconds=0.0,
          stick_figure_input=design.source,
          packing_config=task_cfg.packing_config,
      )
    else:
      packing_result = pipeline.pack_crease_pattern(design, config=task_cfg)
      packed = bus.PackedCreasePattern.from_packing_output(
          packing_result.packing_outputs[0],
          design=design,
      )
  except ValueError as err:
    raise PackingFailedError(str(err), errors=(str(err),)) from err

  effective_sweep_configs = spec.sweep_configs
  if solver is not None:
    effective_sweep_configs = tuple(
        dataclasses.replace(c, solver=solver) for c in effective_sweep_configs
    )

  if spec.stop_on_first_solve:
    start_time = time.perf_counter()
    errors: list[str] = []
    guided_configs = [
        c for c in effective_sweep_configs if c.hinges_from_stick_figure
    ]
    unguided_configs = [
        c for c in effective_sweep_configs if not c.hinges_from_stick_figure
    ]

    def _try_scale1_guided(
        cand_packed: bus.PackedCreasePattern,
    ) -> list[bus.SolvedCreasePattern]:
      found = _run_solver_configs(
          cand_packed,
          ((cfg, 1) for cfg in guided_configs),
          stop_after_first=True,
          errors=errors,
      )
      if found or not guided_configs:
        return found
      return _run_solver_configs(
          cand_packed,
          (
              (fb_cfg, fb_scale)
              for fb_cfg, fb_scale in _iter_fallback_configs(
                  guided_configs,
                  base_scale_factor=1,
                  grid_size=cand_packed.grid_size,
              )
              if fb_scale == 1
          ),
          stop_after_first=True,
          errors=errors,
      )

    candidates = _try_scale1_guided(packed)
    tried_packings = [packed]
    if not candidates:
      for cand_out in pipeline.iter_additional_packing_candidates(
          packing_result,
          start_time=start_time,
          solve_timeout_seconds=None,
      ):
        cand_packed = bus.PackedCreasePattern.from_packing_output(
            cand_out, design=design
        )
        tried_packings.append(cand_packed)
        candidates = _try_scale1_guided(cand_packed)
        if candidates:
          packed = cand_packed
          break

    if not candidates and guided_configs:
      for cand_packed in tried_packings:
        candidates = _run_solver_configs(
            cand_packed,
            (
                (fb_cfg, fb_scale)
                for fb_cfg, fb_scale in _iter_fallback_configs(
                    guided_configs,
                    base_scale_factor=1,
                    grid_size=cand_packed.grid_size,
                )
                if fb_scale != 1
            ),
            stop_after_first=True,
            errors=errors,
        )
        if candidates:
          packed = cand_packed
          break

    if not candidates:
      for cand_packed in tried_packings:
        candidates = _run_solver_configs(
            cand_packed,
            ((cfg, 1) for cfg in unguided_configs),
            stop_after_first=True,
            errors=errors,
        )
        if candidates:
          packed = cand_packed
          break

    if not candidates:
      raise SolvingFailedError(
          f'No flat-foldable crease pattern found for {design.source.name!r}'
          f' under preset={preset!r}.',
          errors=errors,
      )
  else:
    try:
      candidates = solve_sweep(packed, effective_sweep_configs)
    except SolvingFailedError as first_err:
      errors = list(first_err.errors or (str(first_err),))
      candidates = []
      start_time = time.perf_counter()
      for cand_out in pipeline.iter_additional_packing_candidates(
          packing_result,
          start_time=start_time,
          solve_timeout_seconds=None,
      ):
        cand_packed = bus.PackedCreasePattern.from_packing_output(
            cand_out, design=design
        )
        try:
          candidates = solve_sweep(cand_packed, effective_sweep_configs)
          packed = cand_packed
          break
        except SolvingFailedError as err:
          errors.extend(err.errors or (str(err),))
      if not candidates:
        raise SolvingFailedError(
            f'No flat-foldable crease pattern found for {design.source.name!r}'
            f' under preset={preset!r}.',
            errors=errors,
        ) from first_err

  if shape:
    effective_shaping_cfg = (
        shaping_config
        if shaping_config is not None
        else pipeline.ShapingTaskConfig()
    )
    shaped_cands = _shape_candidates(
        candidates,
        config=effective_shaping_cfg,
        design=design,
    )
    return PipelineResult(
        design=design,
        packed=packed,
        solved=shaped_cands[0].solved,
        candidates=tuple(candidates),
        shaped=shaped_cands[0],
        shaped_candidates=tuple(shaped_cands),
    )

  return PipelineResult(
      design=design,
      packed=packed,
      solved=candidates[0],
      candidates=tuple(candidates),
  )


def _require_solved_cp(solved: bus.SolvedCreasePattern) -> None:
  if not isinstance(solved, bus.SolvedCreasePattern):
    raise TypeError(
        'Expected SolvedCreasePattern (use bp.solve_best(...) or '
        f'bp.solve(...)), got {type(solved)!r}'
    )


def shape(
    solved: bus.SolvedCreasePattern,
    config: pipeline.ShapingTaskConfig | None = None,
    *,
    design: stick_figure_lib.StickFigure | bus.DesignTree | None = None,
) -> bus.ShapedCreasePattern:
  """Shapes a SolvedCreasePattern into an immutable ShapedCreasePattern."""
  _require_solved_cp(solved)
  effective_config = (
      config
      if config is not None
      else pipeline.ShapingTaskConfig(max_number_of_cps=1)
  )
  shaped_cands = _shape_candidates(
      (solved,),
      config=effective_config,
      design=design,
  )
  return shaped_cands[0]


def _shape_candidates(
    solved: Sequence[bus.SolvedCreasePattern],
    config: pipeline.ShapingTaskConfig,
    *,
    design: stick_figure_lib.StickFigure | bus.DesignTree | None = None,
) -> list[bus.ShapedCreasePattern]:
  """Shapes one or more SolvedCreasePatterns under `config`."""
  solved_seq = tuple(solved)
  for item in solved_seq:
    _require_solved_cp(item)
  if not solved_seq:
    raise ValueError('shape() requires at least one SolvedCreasePattern.')
  try:
    result = pipeline.shape_crease_pattern(
        solved_seq,
        config=config,
        stick_figure_input=design,
    )
  except (ValueError, IndexError) as err:
    raise ShapingFailedError(str(err), errors=(str(err),)) from err
  if not result.shaped_cps:
    raise ShapingFailedError(
        'No locally flat-foldable shaped crease pattern produced.',
        errors=result.shaping_errors,
    )
  shaped_list: list[bus.ShapedCreasePattern] = []
  for idx, shaped_cp in enumerate(result.shaped_cps):
    sol_idx = (
        result.solving_indices[idx] if idx < len(result.solving_indices) else 0
    )
    parent_solved = solved_seq[min(sol_idx, len(solved_seq) - 1)]
    strat = (
        result.strategies_used[idx]
        if idx < len(result.strategies_used)
        else config.fold_strategy_3d.name.lower()
    )
    t_s = result.time_taken[idx] if idx < len(result.time_taken) else 0.0
    shaped_list.append(
        bus.ShapedCreasePattern.from_crease_pattern(
            shaped_cp,
            solved=parent_solved,
            strategy=strat,
            time_taken=t_s,
        )
    )
  return shaped_list


__all__ = (
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
    'ShapingStrategy',
    'ShapingTaskConfig',
    'SolvedCreasePattern',
    'SolverConfig',
    'SolvingFailedError',
    'SolvingTaskConfig',
    'StackLayoutConfig',
    'StageSnapshot',
    'SolveStage',
    'Stick',
    'StickFigure',
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
    'list_solvers',
    'loads',
    'mesh_to_fold_dict',
    'pack',
    'prepare',
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
)
