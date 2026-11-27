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
#
# Ported from Flat-Folder (https://github.com/origamimagiro/flat-folder),
# Copyright (c) 2022 Jason S. Ku (origamimagiro), licensed under the MIT
# License; see LICENSE in this directory.
# ==============================================================================

"""Fast C++20 global flat-foldability checker and incremental session API.

Provides:
1. A drop-in replacement for `bp_pipeline.third_party.flat_folder.fold_cp`
   (`analyze_fold_file`, `get_absolute_face_orders_from_edges`,
   `visualize_folded_state`, `FlatFolderFOLD`, `FlatFolderOutput`,
   `GUI_COLORS`).
2. `FlatFolderSession`: stateful 3-tier incremental/memoized flat-folding engine
   (Tier-1 canonical `.cp` result memo, Tier-2 M/V-invariant planar arrangement
   & folded cell overlap support cache, Tier-3 canonical unassigned component
   CSP solution cache) with automatic `.cp` delta segment seeding, optional
   reference `fold_cp` backend toggle (`backend="python_ref"`), and configurable
   runtime shadow auditing (`shadow_audit_every_n`).
3. `solve_flat_folded_state`: flat-folded state solver in right-handed grid
   coordinates with exact coherent `face_group_fn` precedent tie-breaking.
4. `FlatFolderSession.extract_unfolded_faces_cp` and
   `compute_folded_vertices_cp`: public unfolded-face and folded-vertex
   geometry helpers without exposing the private C++ extension module.
"""

import collections
from collections.abc import Callable, Sequence
import dataclasses
import enum
import math
import threading
import time
from typing import Any, Literal

# pylint: disable=g-bad-import-order,g-import-not-at-top
from bp_pipeline.third_party.flat_folder import fold_cp
from bp_pipeline.third_party.flat_folder import io_utils
from . import _fast_flat_folder
from . import sat_certificate_checker

# pylint: enable=g-bad-import-order,g-import-not-at-top

CertificateCheckResult = sat_certificate_checker.CertificateCheckResult
CertificateRejectionKind = sat_certificate_checker.CertificateRejectionKind
verify_cpp_sat_certificate = sat_certificate_checker.verify_cpp_sat_certificate
verify_reference_sat_certificate = (
    sat_certificate_checker.verify_reference_sat_certificate
)

BackendName = Literal["cpp", "python_ref"]
Point2DFloat = tuple[float, float]
SegmentCoords = tuple[float, float, float, float]

# Re-export exact `fold_cp` data structures for 1-import drop-in compatibility.
FlatFolderFOLD = fold_cp.FlatFolderFOLD
FlatFolderOutput = fold_cp.FlatFolderOutput
GUI_COLORS = fold_cp.GUI_COLORS
visualize_folded_state = fold_cp.visualize_folded_state

_ASSIGNMENT_STR_TO_INT: dict[str, int] = {
    "B": 1,
    "M": 2,
    "V": 3,
    "F": 4,
    "U": 5,
}

# Default LRU cap on distinct `stream_id` `.cp` history entries retained for
# Wave-1 auto-diff seeding (`FlatFolderSession._prev_cp_by_stream`).
#
# Derivation: coupled to `FastFlatFolderEngine::kDefaultMaxSupportEntries`
# (`256`). Retaining prior `.cp` text for more than 256 concurrent streams
# cannot improve Tier-2 warm-seeding hits because older streams' folded support
# entries have already been LRU-evicted from `support_cache`. Bounds retained
# text to `<= 256` documents (~7.5 MiB at `p99` ~30 KiB/doc, < 3% of
# `kDefaultMaxSupportBytes = 256 MiB`).
_DEFAULT_MAX_STREAM_HISTORY = 256


class FastFlatFolderParityError(RuntimeError):
  """Raised by runtime shadow auditing if C++ and reference `fold_cp` diverge."""


_ENGINE_CLAIM_LOCK = threading.Lock()
_engine_claim_rejections = 0


def engine_claim_rejections() -> int:
  """Returns process-wide count of SAT certificates rejected as engine claims."""
  with _ENGINE_CLAIM_LOCK:
    return _engine_claim_rejections


def _record_global_engine_claim_rejection() -> None:
  """Increments the process-wide counter of engine-claim rejections."""
  global _engine_claim_rejections
  with _ENGINE_CLAIM_LOCK:
    _engine_claim_rejections += 1


def _classify_cert_failure(
    chk: CertificateCheckResult, engine_label: str
) -> tuple[bool | None, int, str, str, bool]:
  """Maps a failed exact certificate check into a `FastFoldResult` outcome.

  Args:
    chk: Failed `CertificateCheckResult` returned by the exact checker.
    engine_label: Human-readable solver name included in engine-claim messages.

  Returns:
    Tuple `(flat_foldable, states, exit_stage, conflict_reason,
    is_engine_claim)`.
  """
  kind = sat_certificate_checker.classify_rejection_reason(chk.reason)
  if kind == CertificateRejectionKind.PATTERN:
    return (
        False,
        0,
        "certificate_rejected",
        f"SAT certificate rejected: {chk.reason}",
        False,
    )
  if kind == CertificateRejectionKind.CANNOT_CHECK:
    return (
        None,
        0,
        "could_not_decide",
        f"SAT certificate uncheckable: {chk.reason}",
        False,
    )
  return (
      None,
      0,
      "could_not_decide",
      f"{engine_label} engine claim rejected: {chk.reason}",
      True,
  )


class FoldVerdict(enum.StrEnum):
  """Three-valued decision returned by `check_cp` and `FlatFolderSession`."""

  FOLDABLE = "foldable"
  NOT_FOLDABLE = "not_foldable"
  COULD_NOT_DECIDE = "could_not_decide"


@dataclasses.dataclass(frozen=True, kw_only=True)
class FastFoldResult:
  """Structured outcome and cache/timing telemetry for `check_cp`.

  `flat_foldable` is the single stored three-valued answer, and the `verdict`
  property derives the corresponding `FoldVerdict`:
    - `FOLDABLE` (`flat_foldable is True`)
    - `NOT_FOLDABLE` (`flat_foldable is False`, with a reason)
    - `COULD_NOT_DECIDE` (`flat_foldable is None`, with a diagnostic reason;
      used uniformly for sweep event-limit / inconclusive engine exits,
      `taco_only=True` Stage-1 passes, patterns the exact checker cannot check
      such as `non_octilinear_cp` or `slit_or_dangling_edge`, and exact-checker
      rejections of the engine's claimed layer order or fields).
  """

  flat_foldable: bool | None
  passed_taco_stage: bool
  states: int
  exit_stage: str
  conflict_reason: str | None = None
  conflict_faces: tuple[int, ...] = ()
  local_violations: tuple[str, ...] = ()
  seeded_hit: bool = False
  memo_cache_hit: bool = False
  support_cache_hit: bool = False
  component_cache_hits: int = 0
  component_cache_misses: int = 0
  used_exact_octilinear: bool = False
  used_exact_folded_octilinear: bool = False
  num_faces: int = 0
  num_cells: int = 0
  num_face_pairs: int = 0
  num_unassigned_components: int = 0
  wall_seconds: float = 0.0
  shadow_audited: bool = False
  certificate_verified: bool = False
  engine_claim_rejected: bool = False
  vertex_coords: tuple[Point2DFloat, ...] = ()
  folded_vertices: tuple[Point2DFloat, ...] = ()

  @property
  def verdict(self) -> FoldVerdict:
    if self.flat_foldable is None:
      return FoldVerdict.COULD_NOT_DECIDE
    return (
        FoldVerdict.FOLDABLE if self.flat_foldable else FoldVerdict.NOT_FOLDABLE
    )


@dataclasses.dataclass(frozen=True, kw_only=True)
class FlatFoldSolution:
  """Solved flat-folded state in right-handed grid coordinates."""

  face_paper_polys_rh: tuple[tuple[Point2DFloat, ...], ...]
  face_folded_polys_rh: tuple[tuple[Point2DFloat, ...], ...]
  face_top_up: tuple[bool, ...]
  face_orders: tuple[int, ...]
  cell_polygons_folded_rh: tuple[tuple[Point2DFloat, ...], ...]
  cell_faces_ordered: tuple[tuple[int, ...], ...]


@dataclasses.dataclass(frozen=True, kw_only=True)
class ShadowAuditStats:
  """Runtime shadow-audit and certificate-claim telemetry for `FlatFolderSession`.

  Attributes:
    total_calls: Total CP checks observed by the session auditor.
    accepted_calls: Checks returning a positive `flat_foldable=True` verdict.
    shadow_audits_run: Checks cross-validated against Python `fold_cp`.
    mismatches: Unexplained C++ vs. Python parity divergences.
    reference_incorrect: Divergences where `sat_certificate_checker` verified
      C++ or refuted Python.
    engine_claim_rejections: Positive claims rejected by the exact checker due
      to claimed order or field mismatches rather than a pattern proof.
  """

  total_calls: int = 0
  accepted_calls: int = 0
  shadow_audits_run: int = 0
  mismatches: int = 0
  reference_incorrect: int = 0
  engine_claim_rejections: int = 0


def diff_cp_changed_segments(
    prev_cp_doc: str, cp_doc: str
) -> tuple[SegmentCoords, ...]:
  """Extracts segments whose crease code changed between two `.cp` strings."""

  def _parse_map(
      doc: str,
  ) -> dict[SegmentCoords, tuple[int, SegmentCoords]]:
    out: dict[SegmentCoords, tuple[int, SegmentCoords]] = {}
    for line in doc.splitlines():
      parts = line.split()
      if len(parts) < 5 or parts[0].startswith("#"):
        continue
      try:
        raw_code = int(parts[0])
        x1 = float(parts[1])
        y1 = float(parts[2])
        x2 = float(parts[3])
        y2 = float(parts[4])
      except ValueError:
        continue
      if not (
          math.isfinite(x1)
          and math.isfinite(y1)
          and math.isfinite(x2)
          and math.isfinite(y2)
      ):
        continue
      code = raw_code if 1 <= raw_code <= 4 else 5
      if code == 4:
        continue
      cx1 = 0.0 if x1 == 0.0 else x1
      cy1 = 0.0 if y1 == 0.0 else y1
      cx2 = 0.0 if x2 == 0.0 else x2
      cy2 = 0.0 if y2 == 0.0 else y2
      if (cx1, cy1) == (cx2, cy2):
        continue
      if (cy2, cx2) < (cy1, cx1):
        key: SegmentCoords = (cx2, cy2, cx1, cy1)
      else:
        key = (cx1, cy1, cx2, cy2)
      out[key] = (code, key)
    return out

  prev_map = _parse_map(prev_cp_doc)
  cur_map = _parse_map(cp_doc)
  changed: list[SegmentCoords] = []
  for key in prev_map.keys() | cur_map.keys():
    prev_entry = prev_map.get(key)
    cur_entry = cur_map.get(key)
    prev_code = 4 if prev_entry is None else prev_entry[0]
    cur_code = 4 if cur_entry is None else cur_entry[0]
    if prev_code != cur_code:
      if cur_entry is not None:
        changed.append(cur_entry[1])
      elif prev_entry is not None:
        changed.append(prev_entry[1])
  return tuple(changed)


def _to_seed_segments(
    changed_segments: Sequence[SegmentCoords] | None,
) -> list[_fast_flat_folder.SeedSegment]:
  if not changed_segments:
    return []
  return [
      _fast_flat_folder.SeedSegment(float(x1), float(y1), float(x2), float(y2))
      for x1, y1, x2, y2 in changed_segments
  ]


def _make_parse_error_output(filename: str | None) -> FlatFolderOutput:
  """Returns a clean `FlatFolderOutput` for document parse failures."""
  return FlatFolderOutput(
      flat_folder_fold=FlatFolderFOLD(
          file=filename,
          flat_foldable=False,
          states=0,
          folded_state_fold=None,
          folded_state_svg=None,
          folded_state_svg_back=None,
          conflict_reason="IO Parser failed",
          conflict_faces=[],
          local_violations=[],
          computation_log=[],
      )
  )


def _build_flat_folder_output_from_cpp(
    filename: str | None,
    cpp_res: _fast_flat_folder.SolveResult,
    generate_state_data: bool,
) -> FlatFolderOutput:
  """Converts a C++ `SolveResult` into `fold_cp.FlatFolderOutput`."""
  flat_foldable = bool(cpp_res.flat_foldable)
  states = int(cpp_res.states)
  conflict_reason = cpp_res.conflict_reason or None
  conflict_faces = list(cpp_res.conflict_faces)
  local_violations = list(cpp_res.local_violations)

  folded_state_fold: dict[str, Any] | None = None
  svg_front: str | None = None
  svg_back: str | None = None

  if flat_foldable and generate_state_data:
    folded_vertices = [[x, y] for x, y in cpp_res.folded_vertices]
    edge_vertices = [[u, v] for u, v in cpp_res.edge_vertices]
    face_vertices = [list(f) for f in cpp_res.face_vertices]
    face_orders = list(cpp_res.face_orders)
    face_orientations = list(cpp_res.face_orientations)

    folded_state_fold = {
        "vertices_coords": folded_vertices,
        "edges_vertices": edge_vertices,
        "faces_vertices": face_vertices,
        "faceOrders": [face_orders],
        "faceOrientations": face_orientations,
    }

    cell_faces_ordered = cpp_res.cell_faces_ordered
    segment_cells = [list(sc) for sc in cpp_res.segment_cells]
    segment_edges = [list(se) for se in cpp_res.segment_edges]
    edge_faces = [list(ef) for ef in cpp_res.edge_faces]
    cell_vertices = [list(cv) for cv in cpp_res.cell_vertices]
    points = [[x, y] for x, y in cpp_res.cell_points]
    segment_points = [[u, v] for u, v in cpp_res.segment_points]

    cell_top_front = [(s[-1] if s else None) for s in cell_faces_ordered]
    svg_front = fold_cp._create_svg_data(  # pylint: disable=protected-access
        cell_top_front,
        segment_cells,
        segment_edges,
        edge_faces,
        face_orientations,
        cell_vertices,
        points,
        segment_points,
        invert_colors=False,
    )

    cell_top_back = [(s[0] if s else None) for s in cell_faces_ordered]
    svg_back = fold_cp._create_svg_data(  # pylint: disable=protected-access
        cell_top_back,
        segment_cells,
        segment_edges,
        edge_faces,
        face_orientations,
        cell_vertices,
        points,
        segment_points,
        invert_colors=True,
        mirror_display=True,
    )

  return FlatFolderOutput(
      flat_folder_fold=FlatFolderFOLD(
          file=filename,
          flat_foldable=flat_foldable,
          states=states,
          folded_state_fold=folded_state_fold,
          folded_state_svg=svg_front,
          folded_state_svg_back=svg_back,
          conflict_reason=conflict_reason,
          conflict_faces=conflict_faces,
          local_violations=local_violations,
          computation_log=[],
      )
  )


def get_absolute_face_orders_from_edges(
    edges: Sequence[Sequence[int]], num_faces: int
) -> list[int]:
  """Drop-in C++ replacement for `fold_cp.get_absolute_face_orders_from_edges`."""
  pairs = [(int(e[0]), int(e[1])) for e in edges]
  return list(_fast_flat_folder.compute_absolute_face_orders(pairs, num_faces))


class FlatFolderSession:
  """Stateful incremental/memoized flat-foldability session.

  Preserves M/V-invariant planar arrangements, folded cell overlaps, compiled
  tortilla parity Union-Find + taco-taco constraints (`SupportCache`), and
  canonical unassigned component CSP solutions (`ComponentCache`) across calls,
  while optionally auto-seeding Wave-1 propagation from `.cp` deltas.

  Thread-safe: C++ Tier-1/2/3 LRU caches synchronize internally via
  `absl::Mutex`, and per-session stream-delta / shadow-audit state is guarded by
  `threading.Lock` without holding the lock across solver execution.
  """

  def __init__(
      self,
      *,
      backend: BackendName = "cpp",
      use_memo_cache: bool = True,
      use_support_cache: bool = True,
      use_component_cache: bool = True,
      auto_seed_deltas: bool = True,
      shadow_audit_every_n: int = 0,
      max_memo_entries: int = 512,
      max_support_entries: int = 256,
      max_component_entries: int = 2048,
      max_stream_history: int = _DEFAULT_MAX_STREAM_HISTORY,
  ) -> None:
    if backend not in ("cpp", "python_ref"):
      raise ValueError(f"Unsupported backend {backend!r}.")
    if shadow_audit_every_n < 0:
      raise ValueError("shadow_audit_every_n must be >= 0.")
    if max_stream_history < 0:
      raise ValueError("max_stream_history must be >= 0.")
    self._backend: BackendName = backend
    self._use_memo_cache = use_memo_cache
    self._use_support_cache = use_support_cache
    self._use_component_cache = use_component_cache
    self._auto_seed_deltas = auto_seed_deltas
    self._shadow_audit_every_n = shadow_audit_every_n
    self._max_stream_history = max_stream_history
    self._engine = _fast_flat_folder.FastFlatFolderEngine(
        max_memo_entries, max_support_entries, max_component_entries
    )
    self._lock = threading.Lock()
    self._prev_cp_by_stream: collections.OrderedDict[str, str] = (
        collections.OrderedDict()
    )
    self._total_calls = 0
    self._accepted_calls = 0
    self._shadow_audits_run = 0
    self._shadow_mismatches = 0
    self._reference_incorrect = 0
    self._engine_claim_rejections = 0

  @property
  def backend(self) -> BackendName:
    return self._backend

  def extract_unfolded_faces_cp(
      self,
      cp_doc: str,
      *,
      flip_to_top_face_up: bool = True,
  ) -> tuple[tuple[Point2DFloat, ...], tuple[tuple[int, ...], ...]]:
    """Extracts unfolded planar face polygons `(vertex_coords, face_vertices)`.

    Args:
      cp_doc: ORIPA `.cp` crease-pattern text.
      flip_to_top_face_up: Whether to apply top-face-up orientation convention.

    Returns:
      `(vertex_coords, face_vertices)` tuples for the unfolded planar mesh.

    Raises:
      ValueError: If `cp_doc` cannot be parsed into a valid unfolded mesh.
    """
    v_coords, f_verts = self._engine.extract_unfolded_faces_cp(
        cp_doc, flip_to_top_face_up
    )
    return (
        tuple((float(x), float(y)) for x, y in v_coords),
        tuple(tuple(int(vi) for vi in face) for face in f_verts),
    )

  def compute_folded_vertices_cp(
      self,
      cp_doc: str,
      *,
      kawasaki_epsilon: float = 5e-2,
  ) -> tuple[tuple[Point2DFloat, ...], tuple[Point2DFloat, ...]] | None:
    """Returns `(vertex_coords, folded_vertices)` for a flat-foldable `.cp` doc.

    Args:
      cp_doc: ORIPA `.cp` crease-pattern text.
      kawasaki_epsilon: Tolerance in radians for Kawasaki's check.

    Returns:
      `(vertex_coords, folded_vertices)` tuples when `cp_doc` is flat-foldable,
      or `None` when parsing fails or the crease pattern is not flat-foldable.
    """
    opts = _fast_flat_folder.SolveOptions()
    opts.flip_to_top_face_up = True
    opts.compute_layer_order = True
    opts.solution_limit = 1
    opts.kawasaki_epsilon = kawasaki_epsilon
    opts.use_memo_cache = self._use_memo_cache
    opts.use_support_cache = self._use_support_cache
    opts.use_component_cache = self._use_component_cache
    cpp_res = self._engine.analyze_cp(cp_doc, opts)
    if not cpp_res.flat_foldable or not cpp_res.vertex_coords:
      return None
    return tuple(cpp_res.vertex_coords), tuple(cpp_res.folded_vertices)

  @property
  def engine_claim_rejections(self) -> int:
    """Count of engine-claim certificate rejections observed in this session."""
    with self._lock:
      return self._engine_claim_rejections

  def clear_caches(self) -> None:
    """Clears all C++ memo/support/component caches and stream delta history."""
    self._engine.clear_caches()
    with self._lock:
      self._prev_cp_by_stream.clear()

  def cache_stats(self) -> _fast_flat_folder.EngineCacheStats:
    """Returns C++ Tier-1 / Tier-2 / Tier-3 cache occupancy and hit counters."""
    return self._engine.get_cache_stats()

  def shadow_audit_stats(self) -> ShadowAuditStats:
    """Returns runtime shadow-audit and engine-claim counters."""
    with self._lock:
      return ShadowAuditStats(
          total_calls=self._total_calls,
          accepted_calls=self._accepted_calls,
          shadow_audits_run=self._shadow_audits_run,
          mismatches=self._shadow_mismatches,
          reference_incorrect=self._reference_incorrect,
          engine_claim_rejections=self._engine_claim_rejections,
      )

  def _record_stream_cp_locked(self, stream_id: str, cp_doc: str) -> str | None:
    """Records `cp_doc` for `stream_id` under LRU cap and returns prior doc."""
    if self._max_stream_history <= 0:
      return None
    prev_doc = self._prev_cp_by_stream.get(stream_id)
    self._prev_cp_by_stream[stream_id] = cp_doc
    self._prev_cp_by_stream.move_to_end(stream_id)
    while len(self._prev_cp_by_stream) > self._max_stream_history:
      self._prev_cp_by_stream.popitem(last=False)
    return prev_doc

  def _resolve_changed_segments(
      self,
      cp_doc: str,
      changed_segments: Sequence[SegmentCoords] | None,
      stream_id: str | None,
  ) -> Sequence[SegmentCoords] | None:
    """Resolves explicit or auto-diffed changed segments for hint seeding."""
    if changed_segments is not None:
      if stream_id is not None:
        with self._lock:
          self._record_stream_cp_locked(stream_id, cp_doc)
      return changed_segments
    if not self._auto_seed_deltas or stream_id is None:
      return None
    with self._lock:
      prev_doc = self._record_stream_cp_locked(stream_id, cp_doc)
    if prev_doc is None or prev_doc == cp_doc:
      return None
    deltas = diff_cp_changed_segments(prev_doc, cp_doc)
    # Only seed when a bounded subset of segments changed (e.g. local branch or
    # single-flap shaping edit); if the entire tree changed, skip diff seeding.
    if not deltas or len(deltas) > 32:
      return None
    return deltas

  def _maybe_shadow_audit_cp(
      self,
      cp_doc: str,
      flat_foldable: bool | None,
      states: int,
      kawasaki_epsilon: float,
  ) -> bool:
    """Periodically cross-checks a CP result against Python fold_cp."""
    with self._lock:
      self._total_calls += 1
      total_calls = self._total_calls
      if flat_foldable:
        self._accepted_calls += 1
      accepted_calls = self._accepted_calls
      if self._shadow_audit_every_n <= 0 or flat_foldable is None:
        return False
      should_audit = False
      if flat_foldable:
        should_audit = (accepted_calls % self._shadow_audit_every_n) == 1 or (
            self._shadow_audit_every_n == 1
        )
      elif (
          self._shadow_audit_every_n == 1
          or (total_calls % (self._shadow_audit_every_n * 8)) == 0
      ):
        should_audit = True
      if not should_audit:
        return False
      self._shadow_audits_run += 1

    ref = fold_cp.analyze_fold_file(
        "<shadow_audit>", cp_doc, ".cp", False, kawasaki_epsilon
    ).flat_folder_fold
    ref_ff = bool(ref.flat_foldable)
    ref_states = int(ref.states or 0)
    if bool(flat_foldable) != ref_ff or (ref_ff and states != ref_states):
      # Reference `fold_cp` (`graph_creation._SweepLineRunner`) has a float
      # sweep topology defect on dense 45-degree folded collinear overlaps that
      # drops/shifts segments and causes false UNSATs (`Constraint variable
      # missing`, `Precision Error: unstable graph`, or spurious constraint
      # conflicts). When C++ finds SAT and the independent exact SAT-certificate
      # checker verifies the layer order, classify as `reference_incorrect`.
      if bool(flat_foldable) and not ref_ff:
        cert_opts = _fast_flat_folder.SolveOptions()
        cert_opts.flip_to_top_face_up = True
        cert_opts.taco_only = False
        cert_opts.compute_layer_order = True
        cert_opts.solution_limit = 1
        cert_opts.kawasaki_epsilon = kawasaki_epsilon
        cert_res_cpp = self._engine.analyze_cp(cp_doc, cert_opts)
        if verify_cpp_sat_certificate(cp_doc, cert_res_cpp).ok:
          with self._lock:
            self._reference_incorrect += 1
          return True
      elif not bool(flat_foldable) and ref_ff:
        # Reference `fold_cp` skips second-arrival holonomy/parity checks around
        # cutout holes and interior `B`-coded lines and can falsely return SAT.
        # Accept C++'s rejection only when the exact checker refutes Python's
        # SAT for a geometric or certificate violation.
        ref_full = fold_cp.analyze_fold_file(
            "<shadow_audit>", cp_doc, ".cp", True, kawasaki_epsilon
        ).flat_folder_fold
        if ref_full.flat_foldable and ref_full.folded_state_fold is not None:
          v_coords, _, _, _, _, f_verts, _ = io_utils.parse_document(
              cp_doc, "cp", True
          )
          ref_chk = verify_reference_sat_certificate(
              cp_doc,
              unfolded_vertices=v_coords or (),
              face_vertices=f_verts or (),
              folded_state_fold=ref_full.folded_state_fold,
              flip_to_top_face_up=True,
          )
          if (
              not ref_chk.ok
              and ref_chk.rejection_kind == CertificateRejectionKind.PATTERN
          ):
            with self._lock:
              self._reference_incorrect += 1
            return True
      with self._lock:
        self._shadow_mismatches += 1
      raise FastFlatFolderParityError(
          "Shadow audit mismatch vs reference fold_cp: "
          f"cpp=(flat_foldable={flat_foldable}, states={states}) vs "
          f"ref=(flat_foldable={ref_ff}, states={ref_states}, "
          f"reason={ref.conflict_reason!r})"
      )
    return True

  def check_cp(
      self,
      cp_doc: str,
      *,
      taco_only: bool = False,
      changed_segments: Sequence[SegmentCoords] | None = None,
      stream_id: str | None = "default",
      kawasaki_epsilon: float = 5e-2,
      ignore_flat_creases: bool = False,
      solution_limit: int = 1,
      verify_certificate: bool = True,
  ) -> FastFoldResult:
    """Checks global flat-foldability of a `.cp` string.

    Returns one of three explicit verdicts in `result.verdict` (mirrored by
    `result.flat_foldable`):
      - `FoldVerdict.FOLDABLE` (`True`): foldable (confirmed by the independent
        exact-rational SAT certificate checker by default).
      - `FoldVerdict.NOT_FOLDABLE` (`False`): proved not foldable (or malformed
        input / exact-checker refutation), with `conflict_reason`.
      - `FoldVerdict.COULD_NOT_DECIDE` (`None`): inconclusive -- covers BOTH
        engine limitations/failures (derived Bentley-Ottmann sweep event limit
        exceeded, unstable float arrangement, disconnected face in folded BFS,
        `taco_only=True` partial screen) AND a folding the engine found that the
        exact checker cannot check (e.g. off the 45-degree lines,
        `non_octilinear_cp`, or `excessive_common_denominator`).

    Confirmation is skipped when `taco_only=True` or `ignore_flat_creases=True`;
    check `result.certificate_verified` to see whether confirmation ran and
    passed. Internal search loops that screen many candidates before final
    certified acceptance may pass `verify_certificate=False` explicitly.

    Args:
      cp_doc: Raw `.cp` crease-pattern document text.
      taco_only: If True, stops after Stage-1 tortilla/taco-taco propagation
        (skips exact certificate confirmation).
      changed_segments: Optional modified crease segments for Wave-1 seeding.
      stream_id: Stream key for auto-diff delta seeding, or None to disable.
      kawasaki_epsilon: Tolerance in radians for Kawasaki's local angle check.
      ignore_flat_creases: If True, omits type-4 (`F`) auxiliary flat segments
        (skips exact certificate confirmation).
      solution_limit: Maximum states enumerated per unassigned component (`0` =
        unlimited).
      verify_certificate: If True (default), confirms positive verdicts via
        exact rational certificate checking (unless `taco_only` or
        `ignore_flat_creases` is True).

    Returns:
      Structured `FastFoldResult` with 3-valued `verdict`, `flat_foldable`,
      diagnostics, and telemetry.
    """
    if self._backend == "python_ref":
      t0 = time.perf_counter()
      ref = fold_cp.analyze_fold_file(
          "<python_ref>",
          cp_doc,
          ".cp",
          bool(verify_certificate),
          kawasaki_epsilon,
      ).flat_folder_fold
      elapsed = time.perf_counter() - t0
      ff: bool | None = bool(ref.flat_foldable)
      cert_ok = False
      engine_claim_rejected = False
      reason = ref.conflict_reason
      exit_stage = "python_ref"
      states = int(ref.states or 0)
      if (
          ff is not None
          and ff
          and verify_certificate
          and not taco_only
          and not ignore_flat_creases
      ):
        v_coords, _, _, _, _, f_verts, _ = io_utils.parse_document(
            cp_doc, "cp", True
        )
        chk = verify_reference_sat_certificate(
            cp_doc,
            unfolded_vertices=v_coords or (),
            face_vertices=f_verts or (),
            folded_state_fold=ref.folded_state_fold or {},
            flip_to_top_face_up=True,
        )
        if chk.ok:
          cert_ok = True
        else:
          ff, states, exit_stage, reason, engine_claim_rejected = (
              _classify_cert_failure(chk, "Python fold_cp")
          )
          if engine_claim_rejected:
            with self._lock:
              self._engine_claim_rejections += 1
            _record_global_engine_claim_rejection()
      return FastFoldResult(
          flat_foldable=ff,
          passed_taco_stage=bool(ff),
          states=states,
          exit_stage=exit_stage,
          conflict_reason=reason,
          conflict_faces=tuple(ref.conflict_faces),
          local_violations=tuple(ref.local_violations),
          wall_seconds=elapsed,
          certificate_verified=cert_ok,
          engine_claim_rejected=engine_claim_rejected,
      )

    resolved_deltas = self._resolve_changed_segments(
        cp_doc, changed_segments, stream_id
    )
    need_cert = verify_certificate and not taco_only and not ignore_flat_creases
    opts = _fast_flat_folder.SolveOptions()
    opts.flip_to_top_face_up = True
    opts.taco_only = taco_only
    opts.compute_layer_order = bool(need_cert)
    opts.ignore_flat_creases = ignore_flat_creases
    opts.solution_limit = solution_limit
    opts.kawasaki_epsilon = kawasaki_epsilon
    opts.use_memo_cache = self._use_memo_cache
    opts.use_support_cache = self._use_support_cache
    opts.use_component_cache = self._use_component_cache
    opts.changed_segments = _to_seed_segments(resolved_deltas)

    cpp_res = self._engine.analyze_cp(cp_doc, opts)
    ff_val: bool | None = (
        None if cpp_res.flat_foldable is None else bool(cpp_res.flat_foldable)
    )
    states_val = int(cpp_res.states)
    exit_stage = cpp_res.exit_stage
    conflict_reason = cpp_res.conflict_reason or None
    cert_verified = False
    engine_claim_rejected = False
    if ff_val is not None and ff_val and need_cert:
      chk = verify_cpp_sat_certificate(
          cp_doc, cpp_res, flip_to_top_face_up=True
      )
      if chk.ok:
        cert_verified = True
      else:
        (
            ff_val,
            states_val,
            exit_stage,
            conflict_reason,
            engine_claim_rejected,
        ) = _classify_cert_failure(chk, "C++ flat-folder")
        if engine_claim_rejected:
          with self._lock:
            self._engine_claim_rejections += 1
          _record_global_engine_claim_rejection()

    audited = False
    if not taco_only and not ignore_flat_creases and solution_limit == 1:
      audited = self._maybe_shadow_audit_cp(
          cp_doc, ff_val, states_val, kawasaki_epsilon
      )

    vertex_coords_out: tuple[Point2DFloat, ...] = ()
    folded_vertices_out: tuple[Point2DFloat, ...] = ()
    if ff_val:
      raw_vertex_coords = cpp_res.vertex_coords
      if raw_vertex_coords:
        raw_folded_vertices = cpp_res.folded_vertices
        if len(raw_vertex_coords) == len(raw_folded_vertices):
          vertex_coords_out = tuple(raw_vertex_coords)
          folded_vertices_out = tuple(raw_folded_vertices)
    return FastFoldResult(
        flat_foldable=ff_val,
        passed_taco_stage=bool(cpp_res.passed_taco_stage),
        states=states_val,
        exit_stage=exit_stage,
        conflict_reason=conflict_reason,
        conflict_faces=tuple(cpp_res.conflict_faces),
        local_violations=tuple(cpp_res.local_violations),
        seeded_hit=bool(cpp_res.seeded_hit),
        memo_cache_hit=bool(cpp_res.memo_cache_hit),
        support_cache_hit=bool(cpp_res.support_cache_hit),
        component_cache_hits=int(cpp_res.component_cache_hits),
        component_cache_misses=int(cpp_res.component_cache_misses),
        used_exact_octilinear=bool(cpp_res.used_exact_octilinear),
        used_exact_folded_octilinear=bool(cpp_res.used_exact_folded_octilinear),
        num_faces=int(cpp_res.num_faces),
        num_cells=int(cpp_res.num_cells),
        num_face_pairs=int(cpp_res.num_face_pairs),
        num_unassigned_components=int(cpp_res.num_unassigned_components),
        wall_seconds=float(cpp_res.wall_seconds),
        shadow_audited=audited,
        certificate_verified=cert_verified,
        engine_claim_rejected=engine_claim_rejected,
        vertex_coords=vertex_coords_out,
        folded_vertices=folded_vertices_out,
    )

  def analyze_fold_file(
      self,
      filename: str | None,
      doc_content: str,
      file_ext: str,
      generate_state_data: bool,
      kawasaki_epsilon: float = 5e-2,
      *,
      flip_to_top_face_up: bool = True,
      changed_segments: Sequence[SegmentCoords] | None = None,
      face_group_ranks: Sequence[int] | None = None,
      stream_id: str | None = None,
  ) -> FlatFolderOutput:
    """Session analyze_fold_file without exact check; see check_cp."""
    if (
        self._backend == "python_ref"
        and flip_to_top_face_up
        and not face_group_ranks
    ):
      return fold_cp.analyze_fold_file(
          filename, doc_content, file_ext, generate_state_data, kawasaki_epsilon
      )

    ext = file_ext[1:] if file_ext.startswith(".") else file_ext
    opts = _fast_flat_folder.SolveOptions()
    opts.flip_to_top_face_up = flip_to_top_face_up
    opts.taco_only = False
    opts.compute_layer_order = bool(generate_state_data)
    opts.solution_limit = 1
    opts.kawasaki_epsilon = kawasaki_epsilon
    opts.use_memo_cache = self._use_memo_cache
    opts.use_support_cache = self._use_support_cache
    opts.use_component_cache = self._use_component_cache
    if face_group_ranks is not None:
      opts.face_group_ranks = [int(r) for r in face_group_ranks]

    if ext == "cp":
      resolved_deltas = self._resolve_changed_segments(
          doc_content, changed_segments, stream_id
      )
      opts.changed_segments = _to_seed_segments(resolved_deltas)
      cpp_res = self._engine.analyze_cp(doc_content, opts)
      if flip_to_top_face_up and face_group_ranks is None:
        self._maybe_shadow_audit_cp(
            doc_content,
            None
            if cpp_res.flat_foldable is None
            else bool(cpp_res.flat_foldable),
            int(cpp_res.states),
            kawasaki_epsilon,
        )
      return _build_flat_folder_output_from_cpp(
          filename, cpp_res, generate_state_data
      )

    if ext in ("svg", "opx"):
      raw_lines = (
          io_utils.svg_to_lines(doc_content)
          if ext == "svg"
          else io_utils.opx_to_lines(doc_content)
      )
      segments = [
          _fast_flat_folder.InputSegment(
              float(p1[0]),
              float(p1[1]),
              float(p2[0]),
              float(p2[1]),
              _ASSIGNMENT_STR_TO_INT.get(str(asgn), 5),
          )
          for p1, p2, asgn in raw_lines
      ]
      cpp_res = self._engine.analyze_lines(segments, opts)
      return _build_flat_folder_output_from_cpp(
          filename, cpp_res, generate_state_data
      )

    if ext == "fold":
      try:
        v_coords, e_verts, e_assigns, _, f_verts = io_utils.fold_to_graph(
            doc_content
        )
        if not v_coords or not e_verts or not e_assigns:
          return _make_parse_error_output(filename)
        v_pairs = [(float(v[0]), float(v[1])) for v in v_coords]
        e_pairs = [(int(e[0]), int(e[1])) for e in e_verts]
        e_codes = [_ASSIGNMENT_STR_TO_INT.get(str(a), 5) for a in e_assigns]
        opt_faces = (
            [[int(vi) for vi in f] for f in f_verts]
            if f_verts is not None
            else []
        )
      except (ValueError, TypeError, IndexError, KeyError, AttributeError):
        return _make_parse_error_output(filename)
      cpp_res = self._engine.analyze_fold_graph(
          v_pairs, e_pairs, e_codes, opt_faces, opts
      )
      return _build_flat_folder_output_from_cpp(
          filename, cpp_res, generate_state_data
      )

    return _make_parse_error_output(filename)

  def solve_flat_folded_state(
      self,
      cp: Any,
      grid_size: int,
      kawasaki_epsilon: float = 5e-2,
      face_group_fn: Callable[[Sequence[Point2DFloat]], int] | None = None,
      *,
      cp_str: str | None = None,
      face_group_ranks: Sequence[int] | None = None,
      stream_id: str | None = "shaping",
  ) -> FlatFoldSolution:
    """Engine-only flat-folded state without exact check; see check_cp."""
    resolved_grid = max(1, int(grid_size))
    inv_g = 1.0 / float(resolved_grid)
    if cp_str is not None:
      doc = cp_str
    elif hasattr(cp, "to_cp"):
      doc = cp.to_cp()
    else:
      from bp_pipeline.crease_pattern.parser import cp_format as _cp_format  # pylint: disable=g-import-not-at-top

      doc = _cp_format.to_cp(cp, scale=1, ndigits=8, grid=resolved_grid)

    def _poly_signed_area_2d(poly: Sequence[Point2DFloat]) -> float:
      if len(poly) < 3:
        return 0.0
      area2 = 0.0
      x1, y1 = poly[-1]
      for x2, y2 in poly:
        area2 += x1 * y2 - x2 * y1
        x1, y1 = x2, y2
      return 0.5 * area2

    def _ensure_ccw(
        poly: Sequence[Point2DFloat],
    ) -> tuple[Point2DFloat, ...]:
      if _poly_signed_area_2d(poly) < 0.0:
        return tuple(reversed(poly))
      return tuple(poly)

    def _ensure_ccw_pair(
        paper_poly: Sequence[Point2DFloat],
        folded_poly: Sequence[Point2DFloat],
    ) -> tuple[tuple[Point2DFloat, ...], tuple[Point2DFloat, ...]]:
      if _poly_signed_area_2d(paper_poly) < 0.0:
        return tuple(reversed(paper_poly)), tuple(reversed(folded_poly))
      return tuple(paper_poly), tuple(folded_poly)

    resolved_ranks: list[int] = []
    if face_group_ranks is not None:
      resolved_ranks = [int(r) for r in face_group_ranks]
    elif face_group_fn is not None:
      v_coords, f_verts = self._engine.extract_unfolded_faces_cp(doc, True)
      for face in f_verts:
        poly_rh = _ensure_ccw(
            [(v_coords[vi][0] * inv_g, -v_coords[vi][1] * inv_g) for vi in face]
        )
        resolved_ranks.append(int(face_group_fn(poly_rh)))

    resolved_deltas = self._resolve_changed_segments(doc, None, stream_id)
    opts = _fast_flat_folder.SolveOptions()
    opts.flip_to_top_face_up = True
    opts.taco_only = False
    opts.compute_layer_order = True
    opts.solution_limit = 1
    opts.kawasaki_epsilon = kawasaki_epsilon
    opts.use_memo_cache = self._use_memo_cache
    opts.use_support_cache = self._use_support_cache
    opts.use_component_cache = self._use_component_cache
    opts.face_group_ranks = resolved_ranks
    opts.changed_segments = _to_seed_segments(resolved_deltas)

    cpp_res = self._engine.analyze_cp(doc, opts)
    if not cpp_res.ok_parse:
      detail = f": {cpp_res.conflict_reason}" if cpp_res.conflict_reason else ""
      raise ValueError(
          f"Flat-folder IO parser failed on crease pattern{detail}."
      )
    local_violations = cpp_res.local_violations
    if local_violations:
      summary = "; ".join(local_violations[:5])
      raise ValueError(
          f"Crease pattern has {len(local_violations)} local "
          f"flat-foldability violation(s): {summary}"
      )
    if cpp_res.flat_foldable is None:
      raise ValueError(
          "Flat-folder could not decide crease pattern: "
          f"{cpp_res.conflict_reason}"
      )
    if not cpp_res.flat_foldable:
      raise ValueError(
          f"Crease pattern is not flat-foldable: {cpp_res.conflict_reason}"
      )

    if self._shadow_audit_every_n > 0:
      self._maybe_shadow_audit_cp(
          doc, True, int(cpp_res.states), kawasaki_epsilon
      )

    v_coords = cpp_res.vertex_coords
    folded_v = cpp_res.folded_vertices
    cell_pts = cpp_res.cell_points

    paper_verts_rh = [(vx * inv_g, -vy * inv_g) for vx, vy in v_coords]
    folded_verts_rh = [(fx * inv_g, -fy * inv_g) for fx, fy in folded_v]
    arr_pts_rh = [(px * inv_g, -py * inv_g) for px, py in cell_pts]

    face_paper_polys: list[tuple[Point2DFloat, ...]] = []
    face_folded_polys: list[tuple[Point2DFloat, ...]] = []
    for face in cpp_res.face_vertices:
      raw_p = [paper_verts_rh[vi] for vi in face]
      raw_q = [folded_verts_rh[vi] for vi in face]
      ccw_p, matched_q = _ensure_ccw_pair(raw_p, raw_q)
      face_paper_polys.append(ccw_p)
      face_folded_polys.append(matched_q)

    cell_vertices = cpp_res.cell_vertices
    cell_faces_ordered = cpp_res.cell_faces_ordered
    cell_polys_rh: list[tuple[Point2DFloat, ...]] = []
    kept_cell_stacks: list[tuple[int, ...]] = []
    for cell, raw_stack in zip(cell_vertices, cell_faces_ordered, strict=True):
      stack = tuple(int(fi) for fi in raw_stack)
      if not stack:
        continue
      c_poly = _ensure_ccw([arr_pts_rh[pi] for pi in cell])
      if abs(_poly_signed_area_2d(c_poly)) <= 1e-12:
        continue
      cell_polys_rh.append(c_poly)
      kept_cell_stacks.append(stack)

    face_top_up = tuple(not bool(o) for o in cpp_res.face_orientations)
    face_orders = tuple(int(o) for o in cpp_res.face_orders)

    return FlatFoldSolution(
        face_paper_polys_rh=tuple(face_paper_polys),
        face_folded_polys_rh=tuple(face_folded_polys),
        face_top_up=face_top_up,
        face_orders=face_orders,
        cell_polygons_folded_rh=tuple(cell_polys_rh),
        cell_faces_ordered=tuple(kept_cell_stacks),
    )


def _make_ephemeral_session(
    *,
    backend: BackendName = "cpp",
    shadow_audit_every_n: int = 0,
) -> FlatFolderSession:
  """Builds a fresh stateless session for top-level calls with `session=None`."""
  return FlatFolderSession(
      backend=backend,
      use_memo_cache=False,
      use_support_cache=False,
      use_component_cache=True,
      auto_seed_deltas=False,
      shadow_audit_every_n=shadow_audit_every_n,
      max_memo_entries=0,
      max_support_entries=0,
      max_stream_history=0,
  )


def analyze_fold_file(
    filename: str | None,
    doc_content: str,
    file_ext: str,
    generate_state_data: bool,
    kawasaki_epsilon: float = 5e-2,
    *,
    session: FlatFolderSession | None = None,
    backend: BackendName = "cpp",
    shadow_audit_every_n: int = 0,
    flip_to_top_face_up: bool = True,
    changed_segments: Sequence[SegmentCoords] | None = None,
    face_group_ranks: Sequence[int] | None = None,
) -> FlatFolderOutput:
  """Drop-in analyze_fold_file without exact check; see check_cp."""
  if (
      backend == "python_ref"
      and session is None
      and flip_to_top_face_up
      and not face_group_ranks
  ):
    return fold_cp.analyze_fold_file(
        filename, doc_content, file_ext, generate_state_data, kawasaki_epsilon
    )
  active_session = session or _make_ephemeral_session(
      backend=backend,
      shadow_audit_every_n=shadow_audit_every_n,
  )
  return active_session.analyze_fold_file(
      filename,
      doc_content,
      file_ext,
      generate_state_data,
      kawasaki_epsilon,
      flip_to_top_face_up=flip_to_top_face_up,
      changed_segments=changed_segments,
      face_group_ranks=face_group_ranks,
  )


def check_cp(
    cp_doc: str,
    *,
    session: FlatFolderSession | None = None,
    backend: BackendName = "cpp",
    taco_only: bool = False,
    changed_segments: Sequence[SegmentCoords] | None = None,
    solution_limit: int = 1,
    kawasaki_epsilon: float = 5e-2,
    shadow_audit_every_n: int = 0,
    verify_certificate: bool = True,
) -> FastFoldResult:
  """Checks flat-foldability of a `.cp` string.

  Returns a three-valued `result.verdict` (`FoldVerdict.FOLDABLE`,
  `FoldVerdict.NOT_FOLDABLE`, or `FoldVerdict.COULD_NOT_DECIDE`, mirrored by
  `result.flat_foldable` as `True`, `False`, or `None`). Defaults to
  `verify_certificate=True` so positive verdicts are confirmed by the
  independent exact-rational SAT certificate checker; confirmation is skipped
  when `taco_only=True` (reported by `result.certificate_verified`). Under
  `verify_certificate=True`, `FoldVerdict.NOT_FOLDABLE` is returned only when
  the checker emits an exact proof about the pattern itself (`PATTERN`); both
  engine-claim certificate rejections (wrong layer order, mismatched fields, or
  unrecognised reasons, which name the engine and increment
  `engine_claim_rejections()`) and checker/engine scope limits (such as
  `non_octilinear_cp`, `slit_or_dangling_edge`, or sweep-event limits) return
  `FoldVerdict.COULD_NOT_DECIDE` (`flat_foldable=None`,
  `exit_stage="could_not_decide"`). Pass `verify_certificate=False` explicitly
  for uncertified screening.

  Args:
    cp_doc: Raw `.cp` crease-pattern document text.
    session: Optional `FlatFolderSession`; creates an ephemeral session if None.
    backend: Solver backend (`"cpp"` or `"python_ref"`).
    taco_only: If True, stops after Stage-1 tortilla/taco-taco propagation
      (skips exact certificate confirmation).
    changed_segments: Optional modified crease segments for Wave-1 seeding.
    solution_limit: Maximum states enumerated per unassigned component (`0` =
      unlimited).
    kawasaki_epsilon: Tolerance in radians for Kawasaki's local angle check.
    shadow_audit_every_n: Live Python `fold_cp` audit cadence (`0` disables).
    verify_certificate: If True (default), confirms positive verdicts via exact
      rational certificate checking (unless `taco_only` is True).

  Returns:
    Structured `FastFoldResult` with 3-valued `verdict`, `flat_foldable`,
    diagnostics, and telemetry.
  """
  active_session = session or _make_ephemeral_session(
      backend=backend,
      shadow_audit_every_n=shadow_audit_every_n,
  )
  return active_session.check_cp(
      cp_doc,
      taco_only=taco_only,
      changed_segments=changed_segments,
      stream_id=None if session is None else "default",
      solution_limit=solution_limit,
      kawasaki_epsilon=kawasaki_epsilon,
      verify_certificate=verify_certificate,
  )


def solve_flat_folded_state(
    cp: Any,
    grid_size: int,
    kawasaki_epsilon: float = 5e-2,
    face_group_fn: Callable[[Sequence[Point2DFloat]], int] | None = None,
    *,
    session: FlatFolderSession | None = None,
    backend: BackendName = "cpp",
    cp_str: str | None = None,
    face_group_ranks: Sequence[int] | None = None,
    shadow_audit_every_n: int = 0,
) -> FlatFoldSolution:
  """Engine-only solve_flat_folded_state without exact check; see check_cp."""
  if backend == "python_ref":
    raise ValueError(
        "backend='python_ref' is only supported on analyze_fold_file/check_cp;"
        " use backend='cpp' for solve_flat_folded_state."
    )
  active_session = session or _make_ephemeral_session(
      shadow_audit_every_n=shadow_audit_every_n,
  )
  return active_session.solve_flat_folded_state(
      cp,
      grid_size,
      kawasaki_epsilon=kawasaki_epsilon,
      face_group_fn=face_group_fn,
      cp_str=cp_str,
      face_group_ranks=face_group_ranks,
      stream_id=None if session is None else "shaping",
  )


def compute_folded_vertices_cp(
    cp_doc: str,
    *,
    session: FlatFolderSession | None = None,
    kawasaki_epsilon: float = 5e-2,
) -> tuple[tuple[Point2DFloat, ...], tuple[Point2DFloat, ...]] | None:
  """Returns `(vertex_coords, folded_vertices)` for a flat-foldable `.cp` doc.

  Uses `session` when supplied, or a fresh zero-cache C++ engine otherwise.

  Args:
    cp_doc: ORIPA `.cp` crease-pattern text.
    session: Optional `FlatFolderSession` whose caches should be reused.
    kawasaki_epsilon: Tolerance in radians for Kawasaki's check.

  Returns:
    `(vertex_coords, folded_vertices)` tuples if `cp_doc` is flat-foldable and
    has vertices, or `None` if parsing fails or the crease pattern is not
    flat-foldable.
  """
  if session is not None:
    return session.compute_folded_vertices_cp(
        cp_doc, kawasaki_epsilon=kawasaki_epsilon
    )
  opts = _fast_flat_folder.SolveOptions()
  opts.flip_to_top_face_up = True
  opts.compute_layer_order = True
  opts.solution_limit = 1
  opts.kawasaki_epsilon = kawasaki_epsilon
  opts.use_memo_cache = False
  opts.use_support_cache = False
  opts.use_component_cache = False
  cpp_res = _fast_flat_folder.FastFlatFolderEngine(0, 0, 0, 0).analyze_cp(
      cp_doc, opts
  )
  if not cpp_res.flat_foldable or not cpp_res.vertex_coords:
    return None
  return tuple(cpp_res.vertex_coords), tuple(cpp_res.folded_vertices)
