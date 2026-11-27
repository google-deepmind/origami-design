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
// ==============================================================================
//
// Ported from Flat-Folder (https://github.com/origamimagiro/flat-folder),
// Copyright (c) 2022 Jason S. Ku (origamimagiro), licensed under the MIT
// License; see LICENSE in this directory.
// ==============================================================================

// Public C++20 API for the global flat-foldability checker and 3-tier
// incremental caching engine.
//
// Provenance: The flat-foldability algorithm (planar subdivision, local
// Maekawa/Kawasaki checks, folded cell overlap extraction, tortilla/taco-taco
// constraint compilation, and component DPLL layer ordering) re-implements
// Jason S. Ku's MIT-licensed Flat-Folder algorithm (ported from the reference
// Python `flat_folder` package), with exact octilinear integer arithmetic,
// incremental caching, and stricter ingress validation.

#ifndef THIRD_PARTY_DEEPMIND_ORIGAMI_DESIGN_PACKAGES_BP_PIPELINE_FLAT_FOLDER_SRC_BP_PIPELINE_FLAT_FOLDER_FAST_FLAT_FOLDER_CORE_H_
#define THIRD_PARTY_DEEPMIND_ORIGAMI_DESIGN_PACKAGES_BP_PIPELINE_FLAT_FOLDER_SRC_BP_PIPELINE_FLAT_FOLDER_FAST_FLAT_FOLDER_CORE_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "fast_flat_folder_types.h"

namespace bp_pipeline_flat_folder {

// Returns the canonical lowercase string name of `stage`, or `""` for
// `ExitStage::kUninitialized`.
[[nodiscard]] absl::string_view ExitStageToString(ExitStage stage);

// Returns the canonical lowercase string name of `verdict` (`"foldable"`,
// `"not_foldable"`, or `"could_not_decide"`).
[[nodiscard]] absl::string_view FoldVerdictToString(FoldVerdict verdict);

// Per-call solver configuration for `FastFlatFolderEngine::Analyze*`.
struct SolveOptions {
  // If true, swaps M <-> V on input edges so face 0 sits top-face-up without
  // reflecting Y; if false, reflects Y across `[0, 1]` and preserves input
  // M/V assignments.
  bool flip_to_top_face_up = true;
  // If true, stops right after Stage 1 (local violations, tortilla parity UF,
  // and taco-taco unit propagation) without running Stage-2 bitset transitivity
  // or component DPLL. Passing Stage 1 yields `stage = ExitStage::kStage1Pass`
  // and `flat_foldable = std::nullopt`.
  bool taco_only = false;
  // If true and the pattern is flat-foldable (`ExitStage::kSat`), populates
  // unfolded/folded coordinates, cell polygons, `ordered_face_pairs`,
  // `face_orders`, and `cell_faces_ordered`.
  bool compute_layer_order = false;
  // If true, strips type-4 (`'F'`) auxiliary flat lines before building the
  // arrangement (sound for flat-foldability screening when collinear chains do
  // not rely on `'F'` crossings).
  bool ignore_flat_creases = false;
  // Maximum solutions enumerated per unassigned variable component (`1` stops
  // after finding the first valid layering; `0` means unlimited enumeration).
  // Negative values are rejected with `ExitStage::kParseError`.
  int solution_limit = 1;
  // Tolerance in radians for Kawasaki's alternating-angle check at interior
  // vertices (default `5e-2`). Must be finite and non-negative; invalid values
  // are rejected with `ExitStage::kParseError`.
  double kawasaki_epsilon = 5e-2;
  // Enables Tier-1 full-result memoization (`AnalyzeCp` / `AnalyzeLines`).
  bool use_memo_cache = false;
  // Enables Tier-2 M/V-invariant planar support caching.
  bool use_support_cache = true;
  // Enables Tier-3 canonical unassigned component CSP caching.
  bool use_component_cache = true;
  // Optional per-face integer group ranks for coherent tie-breaking on
  // unconstrained components (smaller rank prefers placement on top). When
  // non-empty, its length must equal the unfolded `num_faces`; otherwise
  // `ExitStage::kParseError` is returned.
  std::vector<int> face_group_ranks;
  // Optional recently modified crease segments used to seed Wave-1 Stage-1
  // propagation before full Wave-2 closure.
  std::vector<SeedSegment> changed_segments;
};

// Structured result returned by `FastFlatFolderEngine::Analyze*`.
//
// `flat_foldable` is the single stored three-valued answer, and `verdict()`
// derives `FoldVerdict` directly from it:
//   - `FoldVerdict::kFoldable`       <-> `flat_foldable == true`
//       (`stage == ExitStage::kSat`)
//   - `FoldVerdict::kNotFoldable`    <-> `flat_foldable == false`
//       (parse/validation error, local violation, exact geometric proof, or
//       Stage-1 / Stage-2 solver UNSAT)
//   - `FoldVerdict::kCouldNotDecide` <-> `flat_foldable == std::nullopt`
//       (`stage == ExitStage::kCouldNotDecide`, `ExitStage::kStage1Pass` when
//       `options.taco_only` is true, or default-constructed `kUninitialized`)
//
// Relationship among `ok_parse`, `passed_taco_stage`, and `stage`:
//   - `ok_parse` is `false` iff `stage == ExitStage::kParseError`.
//   - `passed_taco_stage` is `true` iff Stage 1 (local check, folded geometry,
//     tortilla parity UF, and taco-taco unit propagation) completed without
//     conflict: specifically `kSat`, `kStage1Pass`, `kStage2Transitivity`,
//     `kStage2TacoTaco`, and `kStage2ComponentUnsat`. (`kStage2Tortilla` clears
//     `passed_taco_stage` back to `false`.)
//   - `states` is the product of enumerated component solution counts (each
//     bounded by `options.solution_limit` when positive), saturating at
//     `INT64_MAX`. It is `0` whenever `stage != ExitStage::kSat`.
//
// Folded-geometry reason taxonomy (unknown failures default to inconclusive):
//   - `ExitStage::kFoldedGeometryError` (`kNotFoldable`, explicit proofs only):
//       * "Folded geometry conflict: vertex <v> closure mismatch on face <f>"
//       * "Folded geometry conflict: face <f> orientation parity mismatch ..."
//   - `ExitStage::kCouldNotDecide` (`kCouldNotDecide`, all other failures):
//       * "Sweep event limit exceeded (...)" (derived bound 2n + n(n-1)/2)
//       * "Precision Error: unstable graph"
//       * "Precision Error: empty folded cell graph"
//       * "Disconnected face encountered in folded state."
//       * "Folded vertex for face cannot be None."
//       * "Constraint variable missing" (incomplete folded overlap pairs)
//       * Any folded-stage arrangement failure (e.g. non-finite folded coords)
struct SolveResult {
  // False iff `stage == ExitStage::kParseError`.
  bool ok_parse = true;
  // Three-valued verdict: `true` (foldable), `false` (not foldable / parse
  // error), or `std::nullopt` (inconclusive / `taco_only` Stage-1 pass).
  std::optional<bool> flat_foldable;
  // True iff Stage-1 tortilla and taco-taco propagation succeeded without
  // conflict (cleared to false on `kStage2Tortilla`).
  bool passed_taco_stage = false;
  // Product of valid solution counts across unassigned components, saturated at
  // `INT64_MAX` (`0` unless `stage == ExitStage::kSat`).
  int64_t states = 0;

  // Derives the three-valued `FoldVerdict` from `flat_foldable`.
  [[nodiscard]] FoldVerdict verdict() const {
    if (!flat_foldable.has_value()) return FoldVerdict::kCouldNotDecide;
    return *flat_foldable ? FoldVerdict::kFoldable : FoldVerdict::kNotFoldable;
  }

  // Terminal pipeline stage.
  ExitStage stage = ExitStage::kUninitialized;
  // Human-readable explanation when `stage` is not `kSat` / `kStage1Pass`.
  std::string conflict_reason;
  // Sorted face indices participating in a Stage-1 or Stage-2 conflict trace.
  std::vector<int> conflict_faces;
  // Per-vertex Maekawa / Kawasaki violation messages (`kLocalViolation`).
  std::vector<std::string> local_violations;

  // Per-call cache and execution telemetry.
  bool seeded_hit = false;
  bool memo_cache_hit = false;
  bool support_cache_hit = false;
  int component_cache_hits = 0;
  int component_cache_misses = 0;
  bool used_exact_octilinear = false;
  bool used_exact_folded_octilinear = false;

  // Graph and CSP size counters (cells/pairs remain `0` on early exits before
  // folded arrangement construction).
  int num_faces = 0;
  int num_cells = 0;
  int num_face_pairs = 0;
  int num_unassigned_components = 0;

  // Per-stage elapsed wall times in seconds (zeroed on `memo_cache_hit` except
  // `wall_seconds`).
  double parse_arrange_seconds = 0.0;
  double stage1_seconds = 0.0;
  double stage2_seconds = 0.0;
  double order_seconds = 0.0;
  double wall_seconds = 0.0;

  // Layer-order and geometry outputs populated when `compute_layer_order` is
  // true and `flat_foldable == true`.
  std::vector<std::pair<double, double>> vertex_coords;
  std::vector<std::pair<double, double>> folded_vertices;
  std::vector<std::pair<int, int>> edge_vertices;
  std::vector<std::vector<int>> edge_faces;
  std::vector<std::vector<int>> face_vertices;
  // `face_orientations[f] == true` iff face `f` is flipped relative to face `0`
  // (`face_orientations[0] == false`).
  std::vector<bool> face_orientations;
  // Directed `(top_face, bottom_face)` pairs across all overlapping
  // `face_pairs` (`0 .. num_face_pairs - 1`).
  std::vector<std::pair<int, int>> ordered_face_pairs;
  // Global bottom-to-top face layer ranks (`0 .. max_rank`) computed from
  // `ordered_face_pairs`.
  std::vector<int> face_orders;

  // Folded cell arrangement geometry and per-cell bottom-to-top face stacks.
  std::vector<std::pair<double, double>> cell_points;
  std::vector<std::vector<int>> cell_vertices;
  std::vector<std::pair<int, int>> segment_points;
  std::vector<std::vector<int>> segment_cells;
  std::vector<std::vector<int>> segment_edges;
  std::vector<std::vector<int>> cell_faces_ordered;
};

// Unfolded face polygons extracted from a `.cp` document.
struct UnfoldedFaces {
  // Unfolded vertex coordinates `(x, y)`.
  std::vector<std::pair<double, double>> vertex_coords;
  // Counter-clockwise vertex index lists for each non-hole face.
  std::vector<std::vector<int>> face_vertices;
};

// Occupancy and cumulative hit/miss counters across the engine's 3 LRU caches.
struct EngineCacheStats {
  size_t memo_entries = 0;
  size_t support_entries = 0;
  size_t support_bytes = 0;
  size_t component_entries = 0;
  uint64_t memo_hits = 0;
  uint64_t memo_misses = 0;
  uint64_t support_hits = 0;
  uint64_t support_misses = 0;
  uint64_t component_hits = 0;
  uint64_t component_misses = 0;
};

// Parses a `.cp` document into `InputSegment` records, optionally dropping
// auxiliary flat (`'F'`) segments per `flat_policy`.
//
// Strips `#` comments and blank lines. Fails closed with `InvalidArgumentError`
// (citing the 1-based line number) if any non-comment line does not have
// exactly 5 valid tokens (`type x1 y1 x2 y2`) with crease code in `0..5` (`0`
// and `5` both map to `kUnassigned`), finite coordinates satisfying
// `|coord| <= kMaxCoordinateMagnitude` (`1e100`), and non-zero segment length,
// or if no segments remain after filtering.
absl::StatusOr<std::vector<InputSegment>> ParseCpDocument(
    absl::string_view cp_doc,
    FlatCreasePolicy flat_policy = FlatCreasePolicy::kKeepFlatCreases);

// Computes global bottom-to-top face layer ranks (`0 .. max_rank`) from
// directed `(top_face, bottom_face)` edges via Kahn's longest-path DAG
// relaxation. Edges with out-of-range face indices are ignored; faces in a
// directed cycle remain at rank `0`. Returns an empty vector when
// `num_faces <= 0`, and `InvalidArgumentError` when `num_faces > kMaxFaces`.
[[nodiscard]] absl::StatusOr<std::vector<int>> ComputeAbsoluteFaceOrders(
    absl::Span<const std::pair<int, int>> top_bottom_edges, int num_faces);

// Stateful C++20 flat-foldability engine with 3-tier incremental caching:
//   Tier 1: Canonical active-crease result memo (`memo_cache_`)
//   Tier 2: M/V-invariant planar arrangement, folded cell overlap, and
//           tortilla/taco-taco support cache (`support_cache_`, bounded by both
//           entry count and total byte footprint)
//   Tier 3: Canonical unassigned component CSP solution cache (`comp_cache_`)
//
// Thread-safety: `FastFlatFolderEngine` is thread-safe. All public methods may
// be invoked concurrently on a shared instance without external locking. All
// internal mutable state resides in `absl::Mutex`-guarded LRU caches storing
// immutable `std::shared_ptr<const T>` entries, and all solver workspace state
// is allocated per call.
class FastFlatFolderEngine {
 public:
  static constexpr size_t kDefaultMaxMemoEntries = 512;
  static constexpr size_t kDefaultMaxSupportEntries = 256;
  static constexpr size_t kDefaultMaxComponentEntries = 2048;
  // Across the 1,389 SAT CPs in the benchmark corpora, `num_faces` (`F`) has
  // `p50 = 28`, `p90 = 208`, `p99 = 384`, `max = 552` (with `p90` full
  // `SupportEntry` size ~1.0 MiB). A 256 MiB cap holds a full 256-entry working
  // set at `p90` without evicting active search trees, while bounding memory on
  // large `F ~ 500` runs.
  static constexpr size_t kDefaultMaxSupportBytes = 256ULL * 1024ULL * 1024ULL;

  // Constructs an engine with bounded LRU capacities for Tiers 1, 2, and 3.
  // Passing `0` for an entry capacity disables that cache tier; passing `0` for
  // `max_support_bytes` disables the support-cache byte cap.
  explicit FastFlatFolderEngine(
      size_t max_memo_entries = kDefaultMaxMemoEntries,
      size_t max_support_entries = kDefaultMaxSupportEntries,
      size_t max_component_entries = kDefaultMaxComponentEntries,
      size_t max_support_bytes = kDefaultMaxSupportBytes);
  ~FastFlatFolderEngine();

  FastFlatFolderEngine(const FastFlatFolderEngine&) = delete;
  FastFlatFolderEngine& operator=(const FastFlatFolderEngine&) = delete;

  // Parses `cp_doc` and solves global flat-foldability under `options`.
  // All errors (malformed `.cp` lines, invalid options, conflicting collinear
  // codes, local violations, folded-geometry conflicts, and engine limits) are
  // reported in-band via `SolveResult::stage`, `flat_foldable`, and
  // `conflict_reason`.
  SolveResult AnalyzeCp(absl::string_view cp_doc, const SolveOptions& options);

  // Solves global flat-foldability directly from pre-parsed `InputSegment`s
  // under `options`, using Tier-1/2/3 caches when enabled. Validation,
  // geometry, and solver outcomes are reported in-band in `SolveResult`.
  SolveResult AnalyzeLines(absl::Span<const InputSegment> lines,
                           const SolveOptions& options);

  // Validates and solves a pre-parsed `.fold` graph where vertices, edges,
  // crease assignments, and optionally `optional_face_vertices` are supplied
  // directly. Uses the Tier-3 component cache when enabled (bypasses Tier-1/2
  // segment caches). Validation, geometry, and solver outcomes are reported
  // in-band in `SolveResult`.
  SolveResult AnalyzeFoldGraph(
      absl::Span<const std::pair<double, double>> vertex_coords,
      absl::Span<const std::pair<int, int>> edge_vertices,
      absl::Span<const CreaseAssignment> edge_assignments,
      absl::Span<const std::vector<int>> optional_face_vertices,
      const SolveOptions& options);

  // Parses unfolded planar face polygons (`vertex_coords` and `face_vertices`)
  // from `cp_doc` without running the global solver (e.g. to compute per-face
  // `SolveOptions::face_group_ranks` before calling `AnalyzeCp`). Reuses a
  // warm Tier-2 support entry on cache hit without inserting on cache miss.
  // Returns `InvalidArgumentError` on parse errors, conflicting collinear
  // codes, empty meshes, or `num_faces > kMaxFaces`; `ResourceExhaustedError`
  // if the float sweep exceeds its event budget; or `AbortedError` on float
  // sweep instability.
  absl::StatusOr<UnfoldedFaces> ExtractUnfoldedFacesCp(
      absl::string_view cp_doc, FaceOrientationPolicy orientation_policy =
                                    FaceOrientationPolicy::kFlipToTopFaceUp);

  // Empties all Tier-1, Tier-2, and Tier-3 cache entries while retaining
  // cumulative hit/miss counters.
  void ClearCaches();

  // Returns a snapshot of current cache occupancy, support-cache byte usage,
  // and cumulative hit/miss counters.
  [[nodiscard]] EngineCacheStats GetCacheStats() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace bp_pipeline_flat_folder

#endif  // THIRD_PARTY_DEEPMIND_ORIGAMI_DESIGN_PACKAGES_BP_PIPELINE_FLAT_FOLDER_SRC_BP_PIPELINE_FLAT_FOLDER_FAST_FLAT_FOLDER_CORE_H_
