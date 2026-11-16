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

"""Command-line interface (`bp-pipeline`) for end-to-end origami design."""

import argparse
from collections.abc import Iterator, Mapping, Sequence
import contextlib
import json
import logging
import pathlib
import re
import signal
import sys
import threading
import time
from typing import Any

from . import api
from . import bus
from . import render_3d
from . import render_summary
from . import timeout as bp_timeout
from . import viz
from .crease_pattern.parser import cp_format
from .crease_pattern.parser import svg as svg_parser
from .shaping import fidelity as fidelity_lib
from .shaping import recipes as recipes_lib
from .shaping import toolbox as toolbox_lib
from .third_party.flat_folder import fold_cp
from .stick_figures import graph_checks
from .stick_figures import stick_figure as sf_lib

_DISCARDED_ELEMENT_RE = re.compile(
    r'Flat foldability violation detected after shaping element: (\S+)\.'
    r' Discarding'
)
_CLIPPED_ELEMENT_RE = re.compile(r'for element: (\S+)\..*Clipping\.')
_SKIPPED_ELEMENT_RE = re.compile(
    r'(?:for element:|shape element:) (\S+?)(?:\.| with ).*Skipping\.'
)

PipelineSource = (
    Mapping[str, Any]
    | sf_lib.StickFigure
    | bus.DesignTree
    | str
    | pathlib.Path
)


class _ShapingLogCaptureHandler(logging.Handler):
  """Captures shaping warning records emitted during `bp.shape(...)`."""

  def __init__(self) -> None:
    super().__init__(level=logging.WARNING)
    self.warnings: list[str] = []
    self.discarded_elements: list[str] = []
    self.clipped_elements: list[str] = []
    self.skipped_elements: list[str] = []

  def emit(self, record: logging.LogRecord) -> None:
    """Records warning messages and classifies affected element names."""
    msg = record.getMessage()
    self.warnings.append(msg)
    for pattern, bucket in (
        (_DISCARDED_ELEMENT_RE, self.discarded_elements),
        (_CLIPPED_ELEMENT_RE, self.clipped_elements),
        (_SKIPPED_ELEMENT_RE, self.skipped_elements),
    ):
      matched = pattern.search(msg)
      if matched is not None:
        elem = matched.group(1)
        if elem not in bucket:
          bucket.append(elem)


class _IntermediatePackingNoiseFilter(logging.Filter):
  """Suppresses routine per-grid tiling misses during packing sweeps."""

  def filter(self, record: logging.LogRecord) -> bool:
    """Returns False for routine intermediate packing search failure logs."""
    msg = record.getMessage()
    if (
        'Initial packing of squares FAILED' in msg
        or 'Tiling solver failed to find a perfect tiling solution' in msg
    ):
      return False
    return True


@contextlib.contextmanager
def _bounded_timeout(timeout_s: float | None) -> Iterator[None]:
  """Bounds a block with cooperative timeout plus POSIX main-thread SIGALRM."""
  with bp_timeout.cooperative_timeout(timeout_s) as active_timeout:
    if active_timeout is None:
      yield
      return
    effective_s = active_timeout.remaining.total_seconds()
    if effective_s <= 0.0:
      raise bp_timeout.CooperativeTimeoutError(
          'Post-shaping global flat-fold check timed out before start.',
          expired_timeout=active_timeout,
      )
    can_use_alarm = (
        hasattr(signal, 'SIGALRM')
        and hasattr(signal, 'ITIMER_REAL')
        and threading.current_thread() is threading.main_thread()
    )
    if not can_use_alarm:
      yield
      return

    prev_handler = signal.getsignal(signal.SIGALRM)
    if prev_handler not in (
        signal.SIG_DFL,
        signal.SIG_IGN,
    ) and not callable(prev_handler):
      yield
      return

    def _raise_on_alarm(signum: int, frame: Any) -> None:
      del signum, frame
      raise bp_timeout.CooperativeTimeoutError(
          'Post-shaping global flat-fold check timed out after'
          f' {effective_s:.1f}s.',
          expired_timeout=active_timeout,
      )

    prev_delay, prev_interval = signal.getitimer(signal.ITIMER_REAL)
    alarm_s = min(effective_s, prev_delay) if prev_delay > 0.0 else effective_s
    start_t = time.monotonic()
    signal.signal(signal.SIGALRM, _raise_on_alarm)
    signal.setitimer(signal.ITIMER_REAL, max(alarm_s, 0.001))
    try:
      yield
    finally:
      elapsed = time.monotonic() - start_t
      next_delay = max(prev_delay - elapsed, 0.001) if prev_delay > 0.0 else 0.0
      signal.setitimer(signal.ITIMER_REAL, next_delay, prev_interval)
      signal.signal(signal.SIGALRM, prev_handler)


def _load_json_dict(path: pathlib.Path) -> dict[str, Any]:
  """Loads a JSON object from `path`."""
  if not path.is_file():
    raise FileNotFoundError(f'Input StickFigure JSON file not found: {path}')
  raw = json.loads(path.read_text(encoding='utf-8'))
  if not isinstance(raw, dict):
    raise ValueError(f'Expected JSON object in {path}, got {type(raw)!r}.')
  return raw


def _resolve_input_dict(source: PipelineSource) -> dict[str, Any]:
  """Normalizes a dict, StickFigure, DesignTree, path, or example name."""
  if isinstance(source, Mapping):
    return dict(source)
  if isinstance(source, bus.DesignTree):
    return source.source.to_dict()
  if isinstance(source, sf_lib.StickFigure):
    return source.to_dict()
  if isinstance(source, pathlib.Path):
    return _load_json_dict(source)
  if isinstance(source, str):
    stripped = source.strip()
    if stripped.startswith('{'):
      raw = json.loads(stripped)
      if not isinstance(raw, dict):
        raise ValueError('JSON input string must decode to an object.')
      return raw
    candidate_path = pathlib.Path(source)
    if candidate_path.is_file() or source.lower().endswith('.json'):
      return _load_json_dict(candidate_path)
    return api.load_example(source).to_dict()
  raise TypeError(
      'Expected StickFigure dict, StickFigure, DesignTree, JSON path, or'
      f' built-in example name; got {type(source)!r}'
  )


def _write_report_json(out_dir: pathlib.Path, report: dict[str, Any]) -> None:
  """Writes `report.json` inside `out_dir` and records its path in `report`."""
  default_report_path = out_dir / 'report.json'
  report['artifacts']['report_json'] = str(default_report_path)
  default_report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')


def _write_contact_sheet(
    out_dir: pathlib.Path,
    item: bus.PackedCreasePattern | bus.SolvedCreasePattern,
    title: str,
    report: dict[str, Any],
) -> None:
  """Writes `contact_sheet.html` for `item` and records its path in `report`."""
  html_path = out_dir / 'contact_sheet.html'
  html_doc = viz.render_packing_grid([item], title=title)
  html_path.write_text(html_doc, encoding='utf-8')
  report['artifacts']['contact_sheet_html'] = str(html_path)


def run_pipeline(
    source: PipelineSource,
    *,
    output_dir: str | pathlib.Path | None = None,
    check_symmetry: bool = False,
    max_retries: int = 15,
    pack_timeout: float | None = 120.0,
    max_steps: int = 4000,
    validate_only: bool = False,
    skip_solve: bool = False,
    run_shape: bool = True,
    render_visuals: bool = True,
    use_sweep_fallback: bool = True,
    shaped_global_ff_timeout: float | None = 15.0,
) -> dict[str, Any]:
  """Validates, packs, solves, shapes, and renders a StickFigure design.

  When `output_dir` is set and `render_visuals=True` (the default), renders
  visual inspection outputs alongside the exported `.bps.json`, `.fold`, and
  `.svg` artifacts:
    - `stick_figure_views.png`: 4-view skeleton projections.
    - `pipeline_stages.png`: 5-panel end-to-end summary sheet.
    - `shaped_3d_views.png`: 4-angle shaded 3D origami turnaround.
    - `contact_sheet.html`: Self-contained interactive HTML contact sheet.

  Args:
    source: StickFigure dict, `StickFigure`, `DesignTree`, JSON file path, or
      built-in example name (e.g. `'cat'`, `'human'`, `'scorpion'`).
    output_dir: Optional directory for visual renders, `.fold`/`.bps.json`/
      `.svg` exports, and `report.json`.
    check_symmetry: If True, enforce bilateral YZ-plane symmetry during
      prevalidation.
    max_retries: Maximum grid-size increments searched during packing.
    pack_timeout: Total multi-grid packing timeout in seconds (`None` disables).
    max_steps: Maximum solver backtracking steps.
    validate_only: If True, stop after graph validation (rendering
      `stick_figure_views.png` when `output_dir` and `render_visuals` are set).
    skip_solve: If True, stop after packing (`packing_cp.svg` + `.bps.json`).
    run_shape: If True, run 3D geometric shaping (`bp.shape`) after solving.
    render_visuals: If True (default), render PNG turnarounds/summaries and
      `contact_sheet.html` whenever `output_dir` is provided.
    use_sweep_fallback: If True, fall back to `bp.solve_sweep` when default
      solving fails or is not globally flat-foldable.
    shaped_global_ff_timeout: Timeout in seconds for post-shaping global
      flat-foldability verification (`None` disables).

  Returns:
    A structured JSON-serializable report dictionary.
  """
  data = _resolve_input_dict(source)
  out_dir = pathlib.Path(output_dir) if output_dir is not None else None
  if out_dir is not None:
    out_dir.mkdir(parents=True, exist_ok=True)

  report: dict[str, Any] = {
      'name': data.get('name', 'unnamed'),
      'valid': False,
      'validation_errors': [],
      'packing': None,
      'solving': None,
      'shaping': None,
      'artifacts': {},
  }

  errors = graph_checks.prevalidate(
      data,
      check_symmetry=check_symmetry,
      use_shaping_flaps=False,
  )
  report['validation_errors'] = errors
  if errors:
    if out_dir is not None:
      _write_report_json(out_dir, report)
    return report

  design = api.prepare(data)
  tree = design.source
  if not graph_checks.is_valid_stick_figure(tree):
    report['validation_errors'] = ['Stick figure failed graph validity check.']
    if out_dir is not None:
      _write_report_json(out_dir, report)
    return report

  report['valid'] = True
  report['is_symmetric_tree'] = design.is_symmetric
  report['num_sticks'] = len(tree.children)
  report['num_joints'] = len(tree.joints)

  if out_dir is not None and render_visuals:
    views_path = render_summary.save_stick_figure_views(
        tree, out_dir / 'stick_figure_views.png'
    )
    report['artifacts']['stick_figure_views_png'] = str(views_path)

  if validate_only:
    if out_dir is not None:
      _write_report_json(out_dir, report)
    return report

  t0 = time.monotonic()
  try:
    task_config = api.PackingTaskConfig(
        packing_config=api.PackingConfig(use_level_shifter=False),
        max_retries=max_retries,
        max_cps_per_grid_size=1,
        max_number_of_cps=1,
        sweep_timeout_seconds=pack_timeout,
    )
    packed = api.pack(design, task_config)
    pack_time = round(time.monotonic() - t0, 3)
    gs_sq = (
        float(packed.grid_size * packed.grid_size)
        if packed.grid_size > 0
        else 1.0
    )
    flap_area = sum(f.rect.w * f.rect.h for f in packed.flaps.values()) / gs_sq
    river_area = 0.0
    for rv in packed.rivers.values():
      river_area += sum(r.w * r.h for r in rv.rects)
    river_area /= gs_sq
    report['packing'] = {
        'success': True,
        'grid_size': packed.grid_size,
        'min_grid_size': design.min_grid_size,
        'max_grid_size': design.max_grid_size,
        'num_flaps': len(packed.flaps),
        'num_rivers': len(packed.rivers),
        'flap_area_ratio': round(flap_area, 3),
        'river_area_ratio': round(river_area, 3),
        'time_s': pack_time,
    }
    if out_dir is not None:
      packing_svg_path = out_dir / 'packing_cp.svg'
      packing_svg_path.write_text(packed.to_svg(), encoding='utf-8')
      report['artifacts']['packing_cp_svg'] = str(packing_svg_path)
      bps_path = out_dir / 'layout.bps.json'
      api.save_bp_studio(packed, bps_path, title=tree.name)
      report['artifacts']['bp_studio_json'] = str(bps_path)
  except Exception as exc:  # pylint: disable=broad-exception-caught
    report['packing'] = {
        'success': False,
        'error': str(exc),
        'time_s': round(time.monotonic() - t0, 3),
    }
    if out_dir is not None:
      _write_report_json(out_dir, report)
    return report

  if skip_solve:
    if out_dir is not None:
      if render_visuals:
        _write_contact_sheet(out_dir, packed, tree.name, report)
        summary_path = render_summary.save_pipeline_summary(
            out_dir / 'pipeline_stages.png',
            name=tree.name,
            packed=packed,
            color_top=data.get('color_top'),
            color_bottom=data.get('color_bottom'),
        )
        report['artifacts']['pipeline_stages_png'] = str(summary_path)
      _write_report_json(out_dir, report)
    return report

  t_solve = time.monotonic()
  solved: bus.SolvedCreasePattern | None = None
  solved_ff: fold_cp.FlatFolderFOLD | None = None
  solve_method_used = 'default_flaps_down'
  try:
    solver_config = api.SolverConfig(
        max_number_of_cps=1,
        max_steps=max_steps,
        max_time_limit=60.0,
        hinges_from_stick_figure=True,
        remove_central_crease=False,
        up_down_method='flaps_down',
    )
    try:
      solved = api.solve_best(packed, solver_config)
    except Exception:  # pylint: disable=broad-exception-caught
      if not use_sweep_fallback:
        raise
      sweep_res = api.solve_sweep(packed)
      if not sweep_res:
        raise RuntimeError('solve_sweep found no flat-foldable solutions.')  # pylint: disable=raise-missing-from
      solved = sweep_res[0]
      solve_method_used = 'solve_sweep_fallback'

    if (
        not solved.is_globally_flat_foldable
        and use_sweep_fallback
        and solve_method_used == 'default_flaps_down'
    ):
      sweep_res = api.solve_sweep(packed)
      for cand in sweep_res:
        if cand.is_globally_flat_foldable:
          solved = cand
          solve_method_used = 'solve_sweep_globally_ff'
          break

    solve_time = round(time.monotonic() - t_solve, 3)
    is_local_ff = solved.creases.local_flat_foldability()
    cp_str = cp_format.to_cp(
        solved.creases, scale=1, ndigits=3, grid=solved.grid_size
    )
    ff_out = fold_cp.analyze_fold_file(
        filename='pattern.cp',
        doc_content=cp_str,
        file_ext='.cp',
        generate_state_data=True,
    )
    solved_ff = ff_out.flat_folder_fold

    flat_fold_method_error = None
    try:
      _ = solved.flat_fold()
    except Exception as ff_exc:  # pylint: disable=broad-exception-caught
      flat_fold_method_error = f'{type(ff_exc).__name__}: {ff_exc}'

    report['solving'] = {
        'success': True,
        'method': solve_method_used,
        'flat_foldable_cps_count': 1,
        'grid_size': solved.creases.grid_size,
        'is_locally_flat_foldable': is_local_ff,
        'is_globally_flat_foldable': solved.is_globally_flat_foldable,
        'flat_folder_states': solved_ff.states if solved_ff else None,
        'flat_folder_conflict': (
            solved_ff.conflict_reason if solved_ff else None
        ),
        'flat_fold_method_error': flat_fold_method_error,
        'time_s': solve_time,
    }
    if out_dir is not None:
      solving_svg_path = out_dir / 'solving_cp.svg'
      solving_svg_path.write_text(solved.to_svg(), encoding='utf-8')
      report['artifacts']['solving_cp_svg'] = str(solving_svg_path)
      fold_path = out_dir / 'solved.fold'
      api.save_fold(solved, fold_path, title=tree.name)
      report['artifacts']['solved_fold'] = str(fold_path)
      solved_bus_path = out_dir / 'solved_bus.json'
      solved_bus_path.write_text(api.dumps(solved), encoding='utf-8')
      report['artifacts']['solved_bus_json'] = str(solved_bus_path)
      if solved_ff and solved_ff.folded_state_svg:
        folded_svg_path = out_dir / 'solved_folded_state.svg'
        folded_svg_path.write_text(solved_ff.folded_state_svg, encoding='utf-8')
        report['artifacts']['solved_folded_state_svg'] = str(folded_svg_path)
  except Exception as exc:  # pylint: disable=broad-exception-caught
    report['solving'] = {
        'success': False,
        'error': str(exc),
        'time_s': round(time.monotonic() - t_solve, 3),
    }

  shaped: bus.ShapedCreasePattern | None = None
  shaped_ff: fold_cp.FlatFolderFOLD | None = None
  shaped_mesh_3d: render_3d.FoldedMesh3D | None = None
  if run_shape and solved is not None:
    t_shape = time.monotonic()
    try:
      log_capture = _ShapingLogCaptureHandler()
      root_logger = logging.getLogger()
      root_logger.addHandler(log_capture)
      try:
        shaped = api.shape(solved)
      finally:
        root_logger.removeHandler(log_capture)
      shape_time = round(time.monotonic() - t_shape, 3)
      shaped_cp_str = cp_format.to_cp(
          shaped.creases, scale=1, ndigits=6, grid=shaped.grid_size
      )

      global_ff_timed_out = False
      shaped_is_global_ff: bool | None = None
      shaped_ff_conflict: str | None = None
      shaped_flat_fold_error: str | None = None
      try:
        with _bounded_timeout(shaped_global_ff_timeout):
          shaped_ff_out = fold_cp.analyze_fold_file(
              filename='pattern.cp',
              doc_content=shaped_cp_str,
              file_ext='.cp',
              generate_state_data=True,
          )
          shaped_ff = shaped_ff_out.flat_folder_fold
          shaped_is_global_ff = bool(shaped_ff and shaped_ff.flat_foldable)
          shaped_ff_conflict = shaped_ff.conflict_reason if shaped_ff else None
          try:
            _ = shaped.flat_fold(strict=False)
          except (TimeoutError, bp_timeout.CooperativeTimeoutError):
            raise
          except Exception as sff_exc:  # pylint: disable=broad-exception-caught
            shaped_flat_fold_error = f'{type(sff_exc).__name__}: {sff_exc}'
      except (TimeoutError, bp_timeout.CooperativeTimeoutError) as ff_to:
        global_ff_timed_out = True
        shaped_is_global_ff = None
        shaped_ff_conflict = f'Timed out: {ff_to}'

      render_3d_error = None
      try:
        shaped_mesh_3d = render_3d.build_folded_mesh_3d(shaped)
      except Exception as r3d_exc:  # pylint: disable=broad-exception-caught
        render_3d_error = f'{type(r3d_exc).__name__}: {r3d_exc}'

      report['shaping'] = {
          'success': True,
          'num_creases_before': len(solved.creases.creases),
          'num_creases_after': len(shaped.creases.creases),
          'is_locally_flat_foldable': shaped.creases.local_flat_foldability(),
          'is_globally_flat_foldable': shaped_is_global_ff,
          'global_ff_timed_out': global_ff_timed_out,
          'flat_folder_conflict': shaped_ff_conflict,
          'shaped_flat_fold_method_error': shaped_flat_fold_error,
          'discarded_elements': log_capture.discarded_elements,
          'clipped_elements': log_capture.clipped_elements,
          'skipped_elements': log_capture.skipped_elements,
          'shaping_warnings': log_capture.warnings,
          'num_3d_facets': len(shaped_mesh_3d.faces) if shaped_mesh_3d else 0,
          'render_3d_error': render_3d_error,
          'time_s': shape_time,
      }
      if out_dir is not None:
        shaped_svg_path = out_dir / 'shaped_cp.svg'
        api.save_svg(shaped, shaped_svg_path)
        report['artifacts']['shaped_cp_svg'] = str(shaped_svg_path)
        shaped_fold_path = out_dir / 'shaped.fold'
        api.save_fold(shaped, shaped_fold_path, title=f'{tree.name}_shaped')
        report['artifacts']['shaped_fold'] = str(shaped_fold_path)
        if shaped_mesh_3d is not None:
          shaped_3d_fold_path = out_dir / 'shaped_3d.fold'
          shaped_3d_fold_path.write_text(
              json.dumps(shaped_mesh_3d.to_fold_dict(tree.name), indent=2),
              encoding='utf-8',
          )
          report['artifacts']['shaped_3d_fold'] = str(shaped_3d_fold_path)
          if render_visuals:
            shaped_3d_png_path = out_dir / 'shaped_3d_views.png'
            render_3d.render_shaped_3d_turnaround_png(
                shaped_3d_png_path,
                name=tree.name,
                mesh=shaped_mesh_3d,
                color_top=data.get('color_top'),
            )
            report['artifacts']['shaped_3d_views_png'] = str(shaped_3d_png_path)
    except Exception as exc:  # pylint: disable=broad-exception-caught
      report['shaping'] = {
          'success': False,
          'error': str(exc),
          'time_s': round(time.monotonic() - t_shape, 3),
      }

  if out_dir is not None:
    if render_visuals:
      _write_contact_sheet(
          out_dir, solved if solved is not None else packed, tree.name, report
      )
      summary_png_path = render_summary.save_pipeline_summary(
          out_dir / 'pipeline_stages.png',
          name=tree.name,
          packed=packed,
          solved=solved,
          shaped=shaped,
          solved_ff=solved_ff,
          shaped_ff=shaped_ff,
          shaped_mesh_3d=shaped_mesh_3d,
          color_top=data.get('color_top'),
          color_bottom=data.get('color_bottom'),
      )
      report['artifacts']['pipeline_stages_png'] = str(summary_png_path)
    _write_report_json(out_dir, report)

  return report


validate_and_solve = run_pipeline


def _is_non_design_json_filename(filename: str) -> bool:
  """Returns True if `filename` is a report, metadata, or layout artifact."""
  lower = filename.lower()
  return (
      lower == 'batch_summary.json'
      or lower == 'report.json'
      or lower.endswith('_report.json')
      or lower.endswith('.report.json')
      or lower in ('shaping_state.json', 'shaping_context.json', 'pose_plan.json')
      or lower.endswith('_bus.json')
      or lower.endswith('.bps.json')
      or lower.endswith('.metadata.json')
  )


def _build_argument_parser() -> argparse.ArgumentParser:
  """Builds the CLI argument parser for `bp-pipeline`."""
  parser = argparse.ArgumentParser(
      prog='bp-pipeline',
      description=(
          'Validate, pack, solve, 3D-shape, and render StickFigure origami'
          ' designs.'
      ),
  )
  parser.add_argument(
      'target',
      nargs='?',
      default=None,
      help=(
          'Input StickFigure JSON file path or built-in example name '
          "(e.g. 'cat', 'human', 'scorpion')."
      ),
  )
  parser.add_argument(
      '-i',
      '--input',
      dest='input_target',
      default=None,
      help='Path to a StickFigure JSON file or built-in example name.',
  )
  parser.add_argument(
      '-b',
      '--batch-dir',
      '--batch_dir',
      dest='batch_dir',
      default=None,
      help='Directory of StickFigure JSON files to process in batch.',
  )
  parser.add_argument(
      '-o',
      '--output-dir',
      '--output_dir',
      dest='output_dir',
      default=None,
      help=(
          'Directory to write visual renders (PNG/HTML) and .bps.json/.fold/'
          '.svg exports.'
      ),
  )
  parser.add_argument(
      '--list-examples',
      '--list_examples',
      dest='list_examples',
      action='store_true',
      help='Print all built-in example stick figure names and exit.',
  )
  parser.add_argument(
      '--validate-only',
      '--validate_only',
      dest='validate_only',
      action='store_true',
      help='Only validate the StickFigure (and render 4-view skeleton if -o).',
  )
  parser.add_argument(
      '--check-symmetry',
      '--check_symmetry',
      dest='check_symmetry',
      action='store_true',
      help='Require bilateral symmetry across the YZ plane.',
  )
  parser.add_argument(
      '--max-retries',
      '--max_retries',
      dest='max_retries',
      type=int,
      default=15,
      help=(
          'Maximum grid-size retries for search packing (symmetric trees step'
          ' by +2 grid units per retry).'
      ),
  )
  parser.add_argument(
      '--pack-timeout',
      '--pack_timeout',
      dest='pack_timeout',
      type=float,
      default=120.0,
      help=(
          'Total timeout in seconds for the multi-grid packing sweep'
          ' (<= 0 disables timeout).'
      ),
  )
  parser.add_argument(
      '--max-steps',
      '--max_steps',
      dest='max_steps',
      type=int,
      default=4000,
      help='Maximum solver search steps.',
  )
  parser.add_argument(
      '--skip-solve',
      '--skip_solve',
      dest='skip_solve',
      action='store_true',
      help='Only validate and pack (skip hinge/pleat solving and 3D shaping).',
  )
  parser.add_argument(
      '--no-shape',
      '--no_shape',
      dest='no_shape',
      action='store_true',
      help='Skip 3D geometric shaping after flat-foldable base solving.',
  )
  parser.add_argument(
      '--no-viz',
      '--no_viz',
      dest='no_viz',
      action='store_true',
      help=(
          'Disable PNG/HTML visual renders when --output-dir is set (still'
          ' exports .bps.json, .fold, .svg, and report.json).'
      ),
  )
  parser.add_argument(
      '--report-json',
      '--report_json',
      dest='report_json',
      default=None,
      help='Optional file path to write the clean JSON report.',
  )
  parser.add_argument(
      '--shaped-global-ff-timeout',
      '--shaped_global_ff_timeout',
      dest='shaped_global_ff_timeout',
      type=float,
      default=15.0,
      help=(
          'Timeout in seconds for post-shaping global flat-foldability check'
          ' (<= 0 disables timeout).'
      ),
  )
  return parser



_SHAPING_SUBCOMMANDS = frozenset({'tools', 'probe', 'pose', 'actuate', 'score'})


def _load_cli_shaping_state(
    target: str,
    *,
    out_dir: pathlib.Path | None = None,
    reset: bool = False,
) -> tuple[bus.SolvedCreasePattern, toolbox_lib.ShapingState, pathlib.Path | None]:
  """Loads `(SolvedCreasePattern, ShapingState, effective_out_dir)` for CLI subcommands."""
  cand = pathlib.Path(target)
  effective_out = out_dir
  solved: bus.SolvedCreasePattern | None = None
  initial_pattern = None

  if cand.is_dir():
    if effective_out is None:
      effective_out = cand
    bus_file = cand / 'solved_bus.json'
    bps_file = cand / 'layout.bps.json'
    solving_svg = cand / 'solving_cp.svg'
    if bus_file.is_file():
      loaded = api.loads(bus_file.read_text(encoding='utf-8'))
      if isinstance(loaded, bus.SolvedCreasePattern):
        solved = loaded
    elif bps_file.is_file():
      imported = api.import_bps(bps_file)
      if imported.packed is None:
        raise ValueError(f'No packed layout found in {bps_file}.')
      if solving_svg.is_file():
        solving_grid = imported.packed.grid_size
        rep_file = cand / 'report.json'
        if rep_file.is_file():
          rep_data = json.loads(rep_file.read_text(encoding='utf-8'))
          solving_grid = int(
              rep_data.get('solving', {}).get('grid_size') or solving_grid
          )
        cp = svg_parser.parse_svg(
            solving_svg.read_text(encoding='utf-8'),
            grid_size=solving_grid,
        )
        solved = bus.SolvedCreasePattern.from_crease_pattern(
            cp,
            packed=imported.packed,
            central_crease_removed=(solving_grid < imported.packed.grid_size),
        )
      else:
        solved = api.solve_best(imported.packed)
    else:
      raise FileNotFoundError(
          f'Directory {cand} has neither solved_bus.json nor layout.bps.json.'
      )

    state = toolbox_lib.ShapingState.from_solved(solved)
    state_file = cand / 'shaping_state.json'
    shaped_svg = cand / 'shaped_cp.svg'
    if not reset and state_file.is_file():
      payload = json.loads(state_file.read_text(encoding='utf-8'))
      pat = toolbox_lib.deserialize_pattern_svg(payload['pattern_svg'])
      backbone = None
      if payload.get('flat_backbone_svg'):
        backbone = toolbox_lib.deserialize_pattern_svg(
            payload['flat_backbone_svg']
        )
      state = state.replace(
          pattern=pat,
          narrowed_elements=tuple(payload.get('narrowed_elements', ())),
          history=tuple(payload.get('history', ())),
          flat_backbone=backbone,
      )
    elif not reset and shaped_svg.is_file():
      pat = toolbox_lib.deserialize_pattern_svg(
          shaped_svg.read_text(encoding='utf-8')
      )
      state = toolbox_lib.ShapingState.from_solved(solved, pattern=pat)
    return solved, state, effective_out

  if cand.is_file() and target.lower().endswith('.bps.json'):
    imported = api.import_bps(cand)
    if imported.packed is None:
      raise ValueError(f'No packed layout found in {cand}.')
    solved = api.solve_best(imported.packed)
  elif cand.is_file() and target.lower().endswith('.json'):
    raw_text = cand.read_text(encoding='utf-8')
    raw_obj = json.loads(raw_text)
    obj_type = raw_obj.get('type') if isinstance(raw_obj, dict) else None
    if obj_type in ('SolvedCreasePattern', 'ShapedCreasePattern', 'PipelineResult'):
      loaded = api.loads(raw_text)
      if isinstance(loaded, bus.PipelineResult):
        solved = loaded.solved
        if not reset and loaded.shaped is not None:
          initial_pattern = loaded.shaped.creases
      elif isinstance(loaded, bus.ShapedCreasePattern):
        solved = loaded.solved
        if not reset:
          initial_pattern = loaded.creases
      elif isinstance(loaded, bus.SolvedCreasePattern):
        solved = loaded
    else:
      solved = api.fold(cand, preset='demo', shape=False).solved
  else:
    solved = api.fold(target, preset='demo', shape=False).solved

  assert solved is not None
  state = toolbox_lib.ShapingState.from_solved(solved, pattern=initial_pattern)
  if effective_out is not None:
    effective_out.mkdir(parents=True, exist_ok=True)
    (effective_out / 'solved_bus.json').write_text(
        api.dumps(solved), encoding='utf-8'
    )
  return solved, state, effective_out


def _save_cli_shaping_outputs(
    out_dir: pathlib.Path,
    solved: bus.SolvedCreasePattern,
    state: toolbox_lib.ShapingState,
    *,
    render_visuals: bool = True,
) -> bus.ShapedCreasePattern:
  """Persists shaped SVG/FOLD/3D renders and `shaping_state.json` into `out_dir`."""
  out_dir.mkdir(parents=True, exist_ok=True)
  bus_path = out_dir / 'solved_bus.json'
  if not bus_path.is_file():
    bus_path.write_text(api.dumps(solved), encoding='utf-8')

  shaped = state.to_shaped(solved)
  api.save_svg(shaped, out_dir / 'shaped_cp.svg')
  api.save_fold(shaped, out_dir / 'shaped.fold', title=f'{state.tree.name}_shaped')

  state_payload: dict[str, Any] = {
      'pattern_svg': toolbox_lib.serialize_pattern_svg(state.pattern),
      'narrowed_elements': list(state.narrowed_elements),
      'history': list(state.history),
  }
  if state.flat_backbone is not None:
    state_payload['flat_backbone_svg'] = toolbox_lib.serialize_pattern_svg(
        state.flat_backbone
    )
  (out_dir / 'shaping_state.json').write_text(
      json.dumps(state_payload, indent=2, default=str), encoding='utf-8'
  )
  (out_dir / 'shaping_history.jsonl').write_text(
      toolbox_lib.history_json(state) + '\n', encoding='utf-8'
  )

  try:
    mesh_3d = render_3d.build_folded_mesh_3d(shaped)
    (out_dir / 'shaped_3d.fold').write_text(
        json.dumps(mesh_3d.to_fold_dict(state.tree.name), indent=2),
        encoding='utf-8',
    )
    if render_visuals:
      render_3d.render_shaped_3d_turnaround_png(
          out_dir / 'shaped_3d_views.png',
          name=state.tree.name,
          mesh=mesh_3d,
          color_top=state.tree.color_top,
      )
  except Exception as err:  # pylint: disable=broad-exception-caught
    logging.warning('3D turnaround render skipped: %s', err)
  return shaped


def _run_shaping_subcommand(argv: Sequence[str]) -> None:
  """Dispatches `bp-pipeline tools|probe|pose|actuate|score` subcommands."""
  subcmd = argv[0]
  parser = argparse.ArgumentParser(prog=f'bp-pipeline {subcmd}')
  parser.add_argument(
      'target',
      help=(
          'Output directory, .bps.json, StickFigure .json, or built-in example'
          ' name.'
      ),
  )
  parser.add_argument(
      '-o',
      '--output-dir',
      '--output_dir',
      dest='output_dir',
      default=None,
      help='Directory to write shaping artifacts and reports.',
  )
  if subcmd == 'probe':
    parser.add_argument(
        '--elements',
        default='',
        help='Comma-separated element names to probe (default: all elements).',
    )
  elif subcmd == 'pose':
    parser.add_argument(
        '--no-search',
        dest='no_search',
        action='store_true',
        help='Disable coordinate search fallback (use greedy fallback).',
    )
    parser.add_argument(
        '--no-figure',
        dest='no_figure',
        action='store_true',
        help='Skip closed-loop figure_pose at stick-figure angles.',
    )
    parser.add_argument(
        '--narrow',
        action='store_true',
        help='Narrow thin limbs on the base before 3D rotation.',
    )
    parser.add_argument(
        '--time-budget',
        dest='time_budget',
        type=float,
        default=120.0,
        help='Wall-time budget in seconds for search mode.',
    )
    parser.add_argument(
        '--no-viz',
        dest='no_viz',
        action='store_true',
        help='Skip 3D PNG turnaround rendering.',
    )
  elif subcmd == 'actuate':
    parser.add_argument(
        '--calls',
        required=True,
        help='JSON string or path to JSON file of [{"name", "arguments"}] calls.',
    )
    parser.add_argument(
        '--reset',
        action='store_true',
        help='Start from the unshaped solved base pattern.',
    )
    parser.add_argument(
        '--no-viz',
        dest='no_viz',
        action='store_true',
        help='Skip 3D PNG turnaround rendering.',
    )

  args = parser.parse_args(list(argv[1:]))
  out_arg = pathlib.Path(args.output_dir) if args.output_dir else None
  reset_flag = bool(getattr(args, 'reset', False)) or subcmd == 'pose'
  solved, state, out_dir = _load_cli_shaping_state(
      args.target, out_dir=out_arg, reset=reset_flag
  )

  if subcmd == 'tools':
    ctx: dict[str, Any] = toolbox_lib.describe(state)
    ctx['element_table'] = toolbox_lib.element_table(state)
    try:
      ctx['flap_orientation'] = toolbox_lib.flap_orientation_info(state)
    except (ValueError, KeyError, AssertionError) as err:
      ctx['flap_orientation'] = {'error': str(err)}
    try:
      ctx['level_shifter_positions'] = toolbox_lib.level_shifter_positions(state)
    except (ValueError, KeyError, AssertionError) as err:
      ctx['level_shifter_positions'] = {'error': str(err)}
    ctx['tools'] = list(toolbox_lib.TOOL_NAMES)
    if out_dir is not None:
      out_dir.mkdir(parents=True, exist_ok=True)
      (out_dir / 'shaping_context.json').write_text(
          json.dumps(ctx, indent=2, default=str), encoding='utf-8'
      )
    print(json.dumps(ctx, indent=2, default=str))
    return

  if subcmd == 'probe':
    elem_list = [e.strip() for e in args.elements.split(',') if e.strip()]
    probe_rep = toolbox_lib.probe(state, elem_list or None)
    if out_dir is not None:
      out_dir.mkdir(parents=True, exist_ok=True)
      (out_dir / 'probe_report.json').write_text(
          json.dumps(probe_rep, indent=2), encoding='utf-8'
      )
    print(json.dumps(probe_rep, indent=2))
    return

  if subcmd == 'pose':
    shaped, pose_rep = recipes_lib.pose_solved(
        solved,
        search=not args.no_search,
        figure=not args.no_figure,
        narrow=args.narrow,
        time_budget_s=args.time_budget,
    )
    posed_state = toolbox_lib.ShapingState.from_solved(
        solved, pattern=shaped.creases
    )
    if out_dir is not None:
      _save_cli_shaping_outputs(
          out_dir, solved, posed_state, render_visuals=not args.no_viz
      )
      (out_dir / 'pose_report.json').write_text(
          json.dumps(pose_rep, indent=2, default=str), encoding='utf-8'
      )
    print(json.dumps(pose_rep, indent=2, default=str))
    return

  if subcmd == 'actuate':
    raw_calls = args.calls.strip()
    if raw_calls.startswith('['):
      calls_list = json.loads(raw_calls)
    else:
      calls_list = json.loads(
          pathlib.Path(raw_calls).read_text(encoding='utf-8')
      )
    if not isinstance(calls_list, list):
      raise ValueError('--calls must decode to a JSON list of tool calls.')
    next_state, results = toolbox_lib.apply_tools(state, calls_list)
    if out_dir is not None:
      _save_cli_shaping_outputs(
          out_dir, solved, next_state, render_visuals=not args.no_viz
      )
    out_list = [r.asdict() for r in results]
    print(json.dumps(out_list, indent=2, default=str))
    return

  if subcmd == 'score':
    shaped_target = state.to_shaped(solved)
    fid_rep = fidelity_lib.evaluate_fidelity(shaped_target)
    if out_dir is not None:
      out_dir.mkdir(parents=True, exist_ok=True)
      (out_dir / 'fidelity_report.json').write_text(
          json.dumps(fid_rep, indent=2, default=str), encoding='utf-8'
      )
    print(json.dumps(fid_rep, indent=2, default=str))
    return


def main(argv: Sequence[str] | None = None) -> None:
  """Entry point for the `bp-pipeline` CLI."""
  effective_argv = list(sys.argv[1:] if argv is None else argv)
  if effective_argv and effective_argv[0] in _SHAPING_SUBCOMMANDS:
    _run_shaping_subcommand(effective_argv)
    return
  parser = _build_argument_parser()
  args = parser.parse_args(effective_argv)

  if args.list_examples:
    for example_name in api.list_examples():
      print(example_name)
    return

  if args.target and args.input_target:
    parser.error('Pass either a positional target or --input, not both.')
  target = args.input_target or args.target

  pack_timeout = args.pack_timeout if args.pack_timeout > 0.0 else None
  ff_timeout = (
      args.shaped_global_ff_timeout
      if args.shaped_global_ff_timeout > 0.0
      else None
  )
  render_visuals = not args.no_viz
  run_shape = not args.no_shape

  noise_filter = _IntermediatePackingNoiseFilter()
  root_logger = logging.getLogger()
  filtered_targets: list[logging.Filterer] = [root_logger]
  filtered_targets.extend(root_logger.handlers)
  for log_target in filtered_targets:
    log_target.addFilter(noise_filter)

  try:
    if args.batch_dir:
      batch_path = pathlib.Path(args.batch_dir)
      base_out = (
          pathlib.Path(args.output_dir)
          if args.output_dir
          else batch_path / 'outputs'
      )
      base_out.mkdir(parents=True, exist_ok=True)
      summary_file = base_out / 'batch_summary.json'
      batch_reports: dict[str, Any] = {}
      for json_file in sorted(batch_path.glob('*.json')):
        if _is_non_design_json_filename(json_file.name):
          continue
        data = json.loads(json_file.read_text(encoding='utf-8'))
        if (
            not isinstance(data, dict)
            or 'children' not in data
            or 'root' not in data
        ):
          print(
              f'=== Skipping non-StickFigure JSON {json_file.name} ===',
              flush=True,
          )
          continue
        print(f'=== Processing {json_file.name} ===', flush=True)
        design_out = base_out / json_file.stem
        rep = run_pipeline(
            data,
            output_dir=design_out,
            check_symmetry=args.check_symmetry,
            max_retries=args.max_retries,
            pack_timeout=pack_timeout,
            max_steps=args.max_steps,
            validate_only=args.validate_only,
            skip_solve=args.skip_solve,
            run_shape=run_shape,
            render_visuals=render_visuals,
            shaped_global_ff_timeout=ff_timeout,
        )
        batch_reports[json_file.name] = rep
        summary_file.write_text(
            json.dumps(batch_reports, indent=2), encoding='utf-8'
        )
      batch_json = json.dumps(batch_reports, indent=2)
      summary_file.write_text(batch_json, encoding='utf-8')
      if args.report_json:
        report_path = pathlib.Path(args.report_json)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(batch_json, encoding='utf-8')
      print(batch_json)
      return

    if not target:
      parser.error(
          'Provide a target (JSON path or example name), --input, --batch-dir,'
          ' or --list-examples.'
      )

    out_dir = pathlib.Path(args.output_dir) if args.output_dir else None
    report = run_pipeline(
        target,
        output_dir=out_dir,
        check_symmetry=args.check_symmetry,
        max_retries=args.max_retries,
        pack_timeout=pack_timeout,
        max_steps=args.max_steps,
        validate_only=args.validate_only,
        skip_solve=args.skip_solve,
        run_shape=run_shape,
        render_visuals=render_visuals,
        shaped_global_ff_timeout=ff_timeout,
    )
  finally:
    for log_target in filtered_targets:
      log_target.removeFilter(noise_filter)

  report_json_str = json.dumps(report, indent=2)
  if args.report_json:
    report_path = pathlib.Path(args.report_json)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_json_str, encoding='utf-8')
  print(report_json_str)
  if (
      not report['valid']
      or (report['packing'] and not report['packing'].get('success'))
      or (report['solving'] and not report['solving'].get('success'))
  ):
    sys.exit(1)


if __name__ == '__main__':
  main()
