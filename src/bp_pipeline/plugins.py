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

"""Pluggable engine registry for custom packers and hinge solvers."""

from collections.abc import Callable
from importlib import metadata as importlib_metadata
import threading
from typing import Any

# Signature for high-level bp.pack() / benchmark plugins:
# (design, *, config, grid_size, max_solutions, timeout_seconds,
#  per_grid_timeout_seconds, num_workers, packer_config,
#  solution_callback) -> PackingOutput
PackerPluginFn = Callable[..., Any]

# Optional signature for single-grid pipeline._attempt_packing() plugins:
# (stick_figure_input, *, grid_size, is_symmetric, allow_paired_axis_placement,
#  max_solutions, config, solution_callback, packer_config)
#  -> list[PackingOutput]
GridPackerPluginFn = Callable[..., list[Any]]

# Signature for hinges.Solver.iter_solutions() plugins:
# (init_state, borders, config, *, source_cp) -> tuple[list[SolverResult], Any]
SolverPluginFn = Callable[..., tuple[list[Any], Any]]

_PLUGIN_ENTRY_POINT_GROUP = 'bp_pipeline.plugins'
_DEFAULT_ENGINE = 'default'
_BUILTIN_ENGINE = 'builtin'
_RESERVED_ENGINE_NAMES = frozenset({_DEFAULT_ENGINE, _BUILTIN_ENGINE})

_PACKERS: dict[str, PackerPluginFn] = {}
_GRID_PACKERS: dict[str, GridPackerPluginFn] = {}
_SOLVERS: dict[str, SolverPluginFn] = {}
_SOLVER_REQUIRES_RECOVERY: set[str] = set()
_DEFAULT_PACKER: str | None = None
_DEFAULT_SOLVER: str | None = None
_DISCOVERY_LOCK = threading.RLock()
_PLUGINS_LOADED = False
_DISCOVERY_IN_PROGRESS = False


def _ensure_plugins_loaded() -> None:
  """Loads and invokes installed entry-point plugins once."""
  global _PLUGINS_LOADED, _DISCOVERY_IN_PROGRESS
  if _PLUGINS_LOADED:
    return
  with _DISCOVERY_LOCK:
    if _PLUGINS_LOADED or _DISCOVERY_IN_PROGRESS:
      return
    _DISCOVERY_IN_PROGRESS = True
    try:
      for ep in importlib_metadata.entry_points(
          group=_PLUGIN_ENTRY_POINT_GROUP
      ):
        ep.load()()
    finally:
      _DISCOVERY_IN_PROGRESS = False
    _PLUGINS_LOADED = True


def register_packer(
    name: str,
    fn: PackerPluginFn,
    *,
    grid_fn: GridPackerPluginFn | None = None,
) -> None:
  """Registers an external packing engine."""
  if name in _RESERVED_ENGINE_NAMES:
    raise ValueError(f'Cannot register packer under reserved name {name!r}.')
  _PACKERS[name] = fn
  if grid_fn is not None:
    _GRID_PACKERS[name] = grid_fn


def set_default_packer(name: str) -> None:
  """Sets the default packing engine resolved for 'default' or None."""
  global _DEFAULT_PACKER
  if name == _DEFAULT_ENGINE:
    raise ValueError(f'Default packer cannot be {_DEFAULT_ENGINE!r}.')
  if name != _BUILTIN_ENGINE and name not in _PACKERS:
    raise ValueError(
        f'Cannot set default packer to unregistered packer {name!r}.'
    )
  if _DEFAULT_PACKER is not None and _DEFAULT_PACKER != name:
    raise ValueError(
        f'Default packer is already set to {_DEFAULT_PACKER!r}; '
        f'cannot change to {name!r}.'
    )
  _DEFAULT_PACKER = name


def resolve_packer(name: str | None) -> str:
  """Resolves a requested packer name to a registered name or 'builtin'."""
  if name == _BUILTIN_ENGINE:
    return _BUILTIN_ENGINE
  _ensure_plugins_loaded()
  if name is None or name == _DEFAULT_ENGINE:
    return _DEFAULT_PACKER if _DEFAULT_PACKER is not None else _BUILTIN_ENGINE
  return name


def get_packer(name: str) -> PackerPluginFn | None:
  """Returns the registered packing engine function, or None."""
  return _PACKERS.get(name)


def get_grid_packer(name: str) -> GridPackerPluginFn | None:
  """Returns the registered single-grid packer function, or None."""
  return _GRID_PACKERS.get(name)


def list_packers() -> tuple[str, ...]:
  """Returns the names of all available packing engines."""
  _ensure_plugins_loaded()
  return (_DEFAULT_ENGINE, _BUILTIN_ENGINE) + tuple(_PACKERS.keys())


def register_solver(
    name: str,
    fn: SolverPluginFn,
    *,
    requires_recovery: bool = False,
) -> None:
  """Registers an external hinge/crease pattern solver engine."""
  if name in _RESERVED_ENGINE_NAMES:
    raise ValueError(f'Cannot register solver under reserved name {name!r}.')
  _SOLVERS[name] = fn
  if requires_recovery:
    _SOLVER_REQUIRES_RECOVERY.add(name)
  else:
    _SOLVER_REQUIRES_RECOVERY.discard(name)


def set_default_solver(name: str) -> None:
  """Sets the default solver engine resolved for 'default' or None."""
  global _DEFAULT_SOLVER
  if name == _DEFAULT_ENGINE:
    raise ValueError(f'Default solver cannot be {_DEFAULT_ENGINE!r}.')
  if name != _BUILTIN_ENGINE and name not in _SOLVERS:
    raise ValueError(
        f'Cannot set default solver to unregistered solver {name!r}.'
    )
  if _DEFAULT_SOLVER is not None and _DEFAULT_SOLVER != name:
    raise ValueError(
        f'Default solver is already set to {_DEFAULT_SOLVER!r}; '
        f'cannot change to {name!r}.'
    )
  _DEFAULT_SOLVER = name


def resolve_solver(name: str | None) -> str:
  """Resolves a requested solver name to a registered name or 'builtin'."""
  if name == _BUILTIN_ENGINE:
    return _BUILTIN_ENGINE
  _ensure_plugins_loaded()
  if name is None or name == _DEFAULT_ENGINE:
    return _DEFAULT_SOLVER if _DEFAULT_SOLVER is not None else _BUILTIN_ENGINE
  return name


def get_solver(name: str) -> SolverPluginFn | None:
  """Returns the registered solver engine function, or None."""
  return _SOLVERS.get(name)


def solver_requires_recovery(name: str) -> bool:
  """Returns whether the registered solver requires recovery preprocessing."""
  return name in _SOLVER_REQUIRES_RECOVERY


def list_solvers() -> tuple[str, ...]:
  """Returns the names of all available hinge/crease pattern solvers."""
  _ensure_plugins_loaded()
  return (_DEFAULT_ENGINE, _BUILTIN_ENGINE) + tuple(_SOLVERS.keys())
