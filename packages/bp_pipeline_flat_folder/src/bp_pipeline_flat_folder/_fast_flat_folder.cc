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

// pybind11 bindings exposing the C++20 flat-foldability engine, options,
// structured results, cache telemetry, and `.cp` / layer-order utilities as the
// `_fast_flat_folder` Python extension module.

#include <cstddef>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "fast_flat_folder_core.h"
#include "fast_flat_folder_types.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"  // IWYU pragma: keep

namespace py = ::pybind11;

PYBIND11_MODULE(_fast_flat_folder, m) {
  using ::bp_pipeline_flat_folder::ComputeAbsoluteFaceOrders;
  using ::bp_pipeline_flat_folder::CreaseAssignment;
  using ::bp_pipeline_flat_folder::EngineCacheStats;
  using ::bp_pipeline_flat_folder::ExitStage;
  using ::bp_pipeline_flat_folder::ExitStageToString;
  using ::bp_pipeline_flat_folder::FaceOrientationPolicy;
  using ::bp_pipeline_flat_folder::FastFlatFolderEngine;
  using ::bp_pipeline_flat_folder::FlatCreasePolicy;
  using ::bp_pipeline_flat_folder::FoldVerdict;
  using ::bp_pipeline_flat_folder::InputSegment;
  using ::bp_pipeline_flat_folder::ParseCpDocument;
  using ::bp_pipeline_flat_folder::SeedSegment;
  using ::bp_pipeline_flat_folder::SolveOptions;
  using ::bp_pipeline_flat_folder::SolveResult;
  using ::bp_pipeline_flat_folder::TryParseCreaseAssignment;
  using ::bp_pipeline_flat_folder::UnfoldedFaces;

  m.doc() =
      "C++20 global flat-foldability solver and 3-tier incremental caching "
      "engine.";
  m.attr("__version__") = "0.1.0";

  py::enum_<CreaseAssignment>(
      m, "CreaseAssignment",
      "Canonical `.cp` crease assignment codes (1='B', 2='M', 3='V', 4='F', "
      "5='U').")
      .value("BOUNDARY", CreaseAssignment::kBoundary,
             "Paper boundary or hole perimeter ('B', code 1).")
      .value("MOUNTAIN", CreaseAssignment::kMountain,
             "Mountain crease ('M', code 2).")
      .value("VALLEY", CreaseAssignment::kValley,
             "Valley crease ('V', code 3).")
      .value("FLAT", CreaseAssignment::kFlat,
             "Auxiliary unfolded flat line ('F', code 4).")
      .value("UNASSIGNED", CreaseAssignment::kUnassigned,
             "Unassigned active fold ('U', code 5; code 0 also maps here).");

  py::enum_<FoldVerdict>(
      m, "FoldVerdict",
      "Three-valued global flat-foldability decision (`FOLDABLE`, "
      "`NOT_FOLDABLE`, or `COULD_NOT_DECIDE`).")
      .value("COULD_NOT_DECIDE", FoldVerdict::kCouldNotDecide,
             "Inconclusive (`flat_foldable is None`).")
      .value("FOLDABLE", FoldVerdict::kFoldable,
             "Valid flat-folded layering found (`flat_foldable is True`).")
      .value("NOT_FOLDABLE", FoldVerdict::kNotFoldable,
             "Proved not flat-foldable or invalid input (`flat_foldable is "
             "False`).");

  py::enum_<ExitStage>(
      m, "ExitStage",
      "Pipeline stage at which `FastFlatFolderEngine` terminated.")
      .value("UNINITIALIZED", ExitStage::kUninitialized,
             "Default unpopulated result.")
      .value("SAT", ExitStage::kSat, "Valid flat-folded layering found.")
      .value("STAGE1_PASS", ExitStage::kStage1Pass,
             "Passed Stage 1 under `taco_only=True`.")
      .value("PARSE_ERROR", ExitStage::kParseError,
             "Input/option validation or document parse error.")
      .value("LOCAL_VIOLATION", ExitStage::kLocalViolation,
             "Maekawa or Kawasaki local violation at an interior vertex.")
      .value("FOLDED_GEOMETRY_ERROR", ExitStage::kFoldedGeometryError,
             "Exact folded vertex closure or orientation-parity conflict.")
      .value("STAGE1_TORTILLA_ODD_CYCLE", ExitStage::kStage1TortillaOddCycle,
             "Stage-1 odd parity cycle encountered on crease initialization.")
      .value("STAGE1_TORTILLA", ExitStage::kStage1Tortilla,
             "Stage-1 tortilla parity conflict (unseeded wave).")
      .value("STAGE1_TORTILLA_SEEDED", ExitStage::kStage1TortillaSeeded,
             "Stage-1 tortilla parity conflict (Wave-1 seeded propagation).")
      .value("STAGE1_TACO_TACO", ExitStage::kStage1TacoTaco,
             "Stage-1 taco-taco conflict (unseeded wave).")
      .value("STAGE1_TACO_TACO_SEEDED", ExitStage::kStage1TacoTacoSeeded,
             "Stage-1 taco-taco conflict (Wave-1 seeded propagation).")
      .value("STAGE2_TORTILLA", ExitStage::kStage2Tortilla,
             "Stage-2 odd parity cycle during transitivity closure.")
      .value("STAGE2_TRANSITIVITY", ExitStage::kStage2Transitivity,
             "Stage-2 transitivity cycle during closure.")
      .value("STAGE2_TACO_TACO", ExitStage::kStage2TacoTaco,
             "Stage-2 taco-taco conflict during closure.")
      .value("STAGE2_COMPONENT_UNSAT", ExitStage::kStage2ComponentUnsat,
             "Stage-2 DPLL component search found zero solutions.")
      .value("COULD_NOT_DECIDE", ExitStage::kCouldNotDecide,
             "Engine or numerical limitation prevented a definite verdict.");

  py::class_<InputSegment>(
      m, "InputSegment",
      "Input line segment `(x1, y1) -> (x2, y2)` with a `.cp` crease code in "
      "`0..5` (`0` and `5` map to `UNASSIGNED`).")
      .def(py::init([](double x1, double y1, double x2, double y2,
                       int assignment) {
             std::optional<CreaseAssignment> parsed =
                 TryParseCreaseAssignment(assignment);
             if (!parsed.has_value()) {
               throw py::value_error(
                   absl::StrCat("Invalid crease assignment code ", assignment,
                                " (expected 0..5)"));
             }
             return InputSegment{
                 .x1 = x1, .y1 = y1, .x2 = x2, .y2 = y2, .assignment = *parsed};
           }),
           py::arg("x1") = 0.0, py::arg("y1") = 0.0, py::arg("x2") = 0.0,
           py::arg("y2") = 0.0,
           py::arg("assignment") =
               static_cast<int>(CreaseAssignment::kUnassigned),
           "Constructs an `InputSegment`; raises `ValueError` if `assignment` "
           "is outside `0..5`.")
      .def_readwrite("x1", &InputSegment::x1, "Start vertex X coordinate.")
      .def_readwrite("y1", &InputSegment::y1, "Start vertex Y coordinate.")
      .def_readwrite("x2", &InputSegment::x2, "End vertex X coordinate.")
      .def_readwrite("y2", &InputSegment::y2, "End vertex Y coordinate.")
      .def_property(
          "assignment",
          [](const InputSegment& self) {
            return static_cast<int>(self.assignment);
          },
          [](InputSegment& self, int code) {
            std::optional<CreaseAssignment> parsed =
                TryParseCreaseAssignment(code);
            if (!parsed.has_value()) {
              throw py::value_error(absl::StrCat(
                  "Invalid crease assignment code ", code, " (expected 0..5)"));
            }
            self.assignment = *parsed;
          },
          "Canonical crease code (`1..5`; setting `0` maps to `5`, other "
          "out-of-range values raise `ValueError`).");

  py::class_<SeedSegment>(
      m, "SeedSegment",
      "Unassigned coordinate segment `(x1, y1) -> (x2, y2)` used to prioritize "
      "Wave-1 incremental propagation around recently modified creases.")
      .def(py::init([](double x1, double y1, double x2, double y2) {
             return SeedSegment{.x1 = x1, .y1 = y1, .x2 = x2, .y2 = y2};
           }),
           py::arg("x1") = 0.0, py::arg("y1") = 0.0, py::arg("x2") = 0.0,
           py::arg("y2") = 0.0, "Constructs a `SeedSegment`.")
      .def_readwrite("x1", &SeedSegment::x1, "Start vertex X coordinate.")
      .def_readwrite("y1", &SeedSegment::y1, "Start vertex Y coordinate.")
      .def_readwrite("x2", &SeedSegment::x2, "End vertex X coordinate.")
      .def_readwrite("y2", &SeedSegment::y2, "End vertex Y coordinate.");

  py::class_<SolveOptions>(
      m, "SolveOptions",
      "Per-call configuration for `FastFlatFolderEngine.analyze_*`.")
      .def(py::init<>(), "Constructs default `SolveOptions`.")
      .def_readwrite(
          "flip_to_top_face_up", &SolveOptions::flip_to_top_face_up,
          "If True, swaps M <-> V so face 0 sits top-face-up without "
          "reflecting Y.")
      .def_readwrite(
          "taco_only", &SolveOptions::taco_only,
          "If True, stops after Stage-1 tortilla/taco-taco propagation and "
          "returns `STAGE1_PASS` (`flat_foldable=None`) on pass.")
      .def_readwrite(
          "compute_layer_order", &SolveOptions::compute_layer_order,
          "If True and SAT, populates unfolded/folded geometry, `face_orders`, "
          "`ordered_face_pairs`, and `cell_faces_ordered`.")
      .def_readwrite(
          "ignore_flat_creases", &SolveOptions::ignore_flat_creases,
          "If True, strips type-4 ('F') auxiliary lines before arrangement "
          "construction.")
      .def_readwrite(
          "solution_limit", &SolveOptions::solution_limit,
          "Maximum solutions enumerated per unassigned component (1 = first "
          "valid state; 0 = unlimited; negative values return `PARSE_ERROR`).")
      .def_readwrite(
          "kawasaki_epsilon", &SolveOptions::kawasaki_epsilon,
          "Tolerance in radians for Kawasaki's alternating-angle check at "
          "interior vertices (default 5e-2; must be finite and >= 0).")
      .def_readwrite("use_memo_cache", &SolveOptions::use_memo_cache,
                     "Enables Tier-1 full-result memoization.")
      .def_readwrite("use_support_cache", &SolveOptions::use_support_cache,
                     "Enables Tier-2 M/V-invariant planar support caching.")
      .def_readwrite("use_component_cache", &SolveOptions::use_component_cache,
                     "Enables Tier-3 canonical unassigned component caching.")
      .def_readwrite(
          "face_group_ranks", &SolveOptions::face_group_ranks,
          "Optional per-face integer ranks for coherent tie-breaking (empty or "
          "length `num_faces`; wrong length returns `PARSE_ERROR`).")
      .def_readwrite(
          "changed_segments", &SolveOptions::changed_segments,
          "Optional `SeedSegment` list prioritizing Wave-1 Stage-1 checks.");

  py::class_<SolveResult>(
      m, "SolveResult",
      "Structured verdict, diagnostics, telemetry, and optional layer-order "
      "geometry returned by `FastFlatFolderEngine.analyze_*`.")
      .def_readonly("ok_parse", &SolveResult::ok_parse,
                    "False iff `stage == ExitStage.PARSE_ERROR`.")
      .def_property_readonly(
          "verdict", &SolveResult::verdict,
          "Three-valued `FoldVerdict` derived from `flat_foldable`.")
      .def_readonly(
          "flat_foldable", &SolveResult::flat_foldable,
          "True (foldable), False (not foldable or parse error), or None "
          "(inconclusive / `taco_only` Stage-1 pass).")
      .def_readonly(
          "passed_taco_stage", &SolveResult::passed_taco_stage,
          "True iff Stage-1 tortilla and taco-taco propagation passed without "
          "conflict (cleared on `STAGE2_TORTILLA`).")
      .def_readonly(
          "states", &SolveResult::states,
          "Product of enumerated component solution counts, saturated at "
          "`INT64_MAX` (0 unless `stage == ExitStage.SAT`).")
      .def_readonly("stage", &SolveResult::stage,
                    "Terminal `ExitStage` enum value.")
      .def_property_readonly(
          "exit_stage",
          [](const SolveResult& self) {
            return std::string(ExitStageToString(self.stage));
          },
          "Canonical lowercase string name of `stage`.")
      .def_readonly(
          "conflict_reason", &SolveResult::conflict_reason,
          "Diagnostic message explaining failure or inconclusive exit.")
      .def_readonly(
          "conflict_faces", &SolveResult::conflict_faces,
          "Sorted face indices participating in a Stage-1/2 conflict.")
      .def_readonly("local_violations", &SolveResult::local_violations,
                    "Per-vertex Maekawa/Kawasaki violation descriptions.")
      .def_readonly("seeded_hit", &SolveResult::seeded_hit,
                    "True iff conflict was caught during Wave-1 seeded search.")
      .def_readonly("memo_cache_hit", &SolveResult::memo_cache_hit,
                    "True iff answered from Tier-1 memo cache.")
      .def_readonly("support_cache_hit", &SolveResult::support_cache_hit,
                    "True iff planar support was reused from Tier-2 cache.")
      .def_readonly("component_cache_hits", &SolveResult::component_cache_hits,
                    "Number of Tier-3 component cache hits during this call.")
      .def_readonly("component_cache_misses",
                    &SolveResult::component_cache_misses,
                    "Number of Tier-3 component cache misses during this call.")
      .def_readonly(
          "used_exact_octilinear", &SolveResult::used_exact_octilinear,
          "True iff the unfolded arrangement used exact integer arithmetic.")
      .def_readonly(
          "used_exact_folded_octilinear",
          &SolveResult::used_exact_folded_octilinear,
          "True iff the folded cell arrangement used exact integer arithmetic.")
      .def_readonly("num_faces", &SolveResult::num_faces,
                    "Number of non-hole unfolded faces.")
      .def_readonly("num_cells", &SolveResult::num_cells,
                    "Number of bounded folded arrangement cells.")
      .def_readonly("num_face_pairs", &SolveResult::num_face_pairs,
                    "Number of overlapping folded face pairs.")
      .def_readonly("num_unassigned_components",
                    &SolveResult::num_unassigned_components,
                    "Number of Stage-2 unassigned variable components.")
      .def_readonly("parse_arrange_seconds",
                    &SolveResult::parse_arrange_seconds,
                    "Wall seconds spent parsing and building planar support.")
      .def_readonly("stage1_seconds", &SolveResult::stage1_seconds,
                    "Wall seconds spent in Stage-1 local and taco checks.")
      .def_readonly("stage2_seconds", &SolveResult::stage2_seconds,
                    "Wall seconds spent in Stage-2 closure and component DPLL.")
      .def_readonly("order_seconds", &SolveResult::order_seconds,
                    "Wall seconds spent materializing layer-order outputs.")
      .def_readonly("wall_seconds", &SolveResult::wall_seconds,
                    "Total elapsed wall seconds for the call.")
      .def_readonly("vertex_coords", &SolveResult::vertex_coords,
                    "Unfolded vertex `(x, y)` coordinates (when "
                    "`compute_layer_order` and SAT).")
      .def_readonly("folded_vertices", &SolveResult::folded_vertices,
                    "Folded vertex `(x, y)` coordinates (when "
                    "`compute_layer_order` and SAT).")
      .def_readonly("edge_vertices", &SolveResult::edge_vertices,
                    "Unfolded mesh edge `(u, v)` vertex index pairs.")
      .def_readonly("edge_faces", &SolveResult::edge_faces,
                    "Incident face indices per unfolded mesh edge.")
      .def_readonly("face_vertices", &SolveResult::face_vertices,
                    "Counter-clockwise vertex index lists per unfolded face.")
      .def_readonly(
          "face_orientations", &SolveResult::face_orientations,
          "Per-face flip flags (`True` iff flipped relative to face 0).")
      .def_readonly(
          "ordered_face_pairs", &SolveResult::ordered_face_pairs,
          "Directed `(top_face, bottom_face)` ordering pairs over overlapping "
          "faces.")
      .def_readonly("face_orders", &SolveResult::face_orders,
                    "Global bottom-to-top face layer ranks (`0..max_rank`).")
      .def_readonly("cell_points", &SolveResult::cell_points,
                    "Folded arrangement cell vertex `(x, y)` coordinates.")
      .def_readonly("cell_vertices", &SolveResult::cell_vertices,
                    "Vertex index loops into `cell_points` for each cell.")
      .def_readonly("segment_points", &SolveResult::segment_points,
                    "Folded arrangement segment `(u, v)` vertex index pairs.")
      .def_readonly("segment_cells", &SolveResult::segment_cells,
                    "Adjacent cell indices per folded arrangement segment.")
      .def_readonly("segment_edges", &SolveResult::segment_edges,
                    "Covering unfolded edge indices per folded segment.")
      .def_readonly("cell_faces_ordered", &SolveResult::cell_faces_ordered,
                    "Bottom-to-top sorted face stacks covering each cell.");

  py::class_<EngineCacheStats>(
      m, "EngineCacheStats",
      "Occupancy, byte footprint, and cumulative hit/miss counters across "
      "`FastFlatFolderEngine`'s three LRU cache tiers.")
      .def_readonly("memo_entries", &EngineCacheStats::memo_entries,
                    "Current entries in Tier-1 memo cache.")
      .def_readonly("support_entries", &EngineCacheStats::support_entries,
                    "Current entries in Tier-2 support cache.")
      .def_readonly("support_bytes", &EngineCacheStats::support_bytes,
                    "Estimated byte footprint of Tier-2 support cache.")
      .def_readonly("component_entries", &EngineCacheStats::component_entries,
                    "Current entries in Tier-3 component cache.")
      .def_readonly("memo_hits", &EngineCacheStats::memo_hits,
                    "Cumulative Tier-1 memo cache hits.")
      .def_readonly("memo_misses", &EngineCacheStats::memo_misses,
                    "Cumulative Tier-1 memo cache misses.")
      .def_readonly("support_hits", &EngineCacheStats::support_hits,
                    "Cumulative Tier-2 support cache hits.")
      .def_readonly("support_misses", &EngineCacheStats::support_misses,
                    "Cumulative Tier-2 support cache misses.")
      .def_readonly("component_hits", &EngineCacheStats::component_hits,
                    "Cumulative Tier-3 component cache hits.")
      .def_readonly("component_misses", &EngineCacheStats::component_misses,
                    "Cumulative Tier-3 component cache misses.");

  py::class_<FastFlatFolderEngine>(
      m, "FastFlatFolderEngine",
      "Thread-safe stateful C++20 flat-foldability solver with 3-tier "
      "incremental LRU caching.")
      .def(py::init<size_t, size_t, size_t, size_t>(),
           py::arg("max_memo_entries") =
               FastFlatFolderEngine::kDefaultMaxMemoEntries,
           py::arg("max_support_entries") =
               FastFlatFolderEngine::kDefaultMaxSupportEntries,
           py::arg("max_component_entries") =
               FastFlatFolderEngine::kDefaultMaxComponentEntries,
           py::arg("max_support_bytes") =
               FastFlatFolderEngine::kDefaultMaxSupportBytes,
           "Constructs an engine with bounded LRU capacities (0 disables a "
           "cache tier or byte cap).")
      .def(
          "analyze_cp",
          // `options` is taken by value while the GIL is held so another Python
          // thread cannot mutate its `face_group_ranks` or `changed_segments`
          // vectors through `def_readwrite` after the GIL is released below.
          [](FastFlatFolderEngine& self, const std::string& cp_doc,
             SolveOptions options) {
            py::gil_scoped_release release;
            return self.AnalyzeCp(cp_doc, options);
          },
          py::arg("cp_doc"), py::arg("options") = SolveOptions{},
          "Parses a `.cp` string and solves global flat-foldability, returning "
          "a `SolveResult` (all errors reported in-band).")
      .def(
          "analyze_lines",
          // `options` is taken by value while the GIL is held so another Python
          // thread cannot mutate its vectors while the GIL is released.
          [](FastFlatFolderEngine& self, const std::vector<InputSegment>& lines,
             SolveOptions options) {
            py::gil_scoped_release release;
            return self.AnalyzeLines(lines, options);
          },
          py::arg("lines"), py::arg("options") = SolveOptions{},
          "Solves global flat-foldability from pre-parsed `InputSegment`s, "
          "returning a `SolveResult`.")
      .def(
          "analyze_fold_graph",
          // `options` is taken by value while the GIL is held so another Python
          // thread cannot mutate its vectors while the GIL is released.
          [](FastFlatFolderEngine& self,
             const std::vector<std::pair<double, double>>& vertex_coords,
             const std::vector<std::pair<int, int>>& edge_vertices,
             const std::vector<int>& edge_assignments,
             const std::vector<std::vector<int>>& optional_face_vertices,
             SolveOptions options) {
            std::vector<CreaseAssignment> typed_assignments(
                edge_assignments.size());
            for (size_t i = 0; i < edge_assignments.size(); ++i) {
              std::optional<CreaseAssignment> parsed =
                  TryParseCreaseAssignment(edge_assignments[i]);
              if (!parsed.has_value()) {
                SolveResult err;
                err.ok_parse = false;
                err.flat_foldable = false;
                err.stage = ExitStage::kParseError;
                err.conflict_reason = absl::StrCat(
                    "Invalid crease assignment code ", edge_assignments[i],
                    " at edge ", i, " (expected 0..5)");
                return err;
              }
              typed_assignments[i] = *parsed;
            }
            py::gil_scoped_release release;
            return self.AnalyzeFoldGraph(vertex_coords, edge_vertices,
                                         typed_assignments,
                                         optional_face_vertices, options);
          },
          py::arg("vertex_coords"), py::arg("edge_vertices"),
          py::arg("edge_assignments"),
          py::arg("optional_face_vertices") = std::vector<std::vector<int>>{},
          py::arg("options") = SolveOptions{},
          "Validates and solves a pre-parsed `.fold` graph (`vertex_coords`, "
          "`edge_vertices`, integer `edge_assignments` in `0..5`, and optional "
          "`optional_face_vertices`), returning a `SolveResult`.")
      .def(
          "extract_unfolded_faces_cp",
          [](FastFlatFolderEngine& self, const std::string& cp_doc,
             bool flip_to_top_face_up) {
            const FaceOrientationPolicy policy =
                flip_to_top_face_up ? FaceOrientationPolicy::kFlipToTopFaceUp
                                    : FaceOrientationPolicy::kPreserveInput;
            absl::StatusOr<UnfoldedFaces> faces;
            {
              py::gil_scoped_release release;
              faces = self.ExtractUnfoldedFacesCp(cp_doc, policy);
            }
            if (!faces.ok()) {
              throw py::value_error(std::string(faces.status().message()));
            }
            return std::make_pair(std::move(faces->vertex_coords),
                                  std::move(faces->face_vertices));
          },
          py::arg("cp_doc"), py::arg("flip_to_top_face_up") = true,
          "Returns `(vertex_coords, face_vertices)` for the unfolded mesh of "
          "`cp_doc` without running global solve; raises `ValueError` on parse "
          "or arrangement errors.")
      .def("clear_caches", &FastFlatFolderEngine::ClearCaches,
           py::call_guard<py::gil_scoped_release>(),
           "Clears all Tier-1, Tier-2, and Tier-3 cache entries while "
           "retaining cumulative hit/miss counters.")
      .def("get_cache_stats", &FastFlatFolderEngine::GetCacheStats,
           py::call_guard<py::gil_scoped_release>(),
           "Returns an `EngineCacheStats` snapshot across all three caches.");

  m.def(
      "parse_cp_document",
      [](const std::string& cp_doc, bool ignore_flat_creases) {
        const FlatCreasePolicy policy =
            ignore_flat_creases ? FlatCreasePolicy::kIgnoreFlatCreases
                                : FlatCreasePolicy::kKeepFlatCreases;
        absl::StatusOr<std::vector<InputSegment>> segments;
        {
          py::gil_scoped_release release;
          segments = ParseCpDocument(cp_doc, policy);
        }
        if (!segments.ok()) {
          throw py::value_error(std::string(segments.status().message()));
        }
        return *std::move(segments);
      },
      py::arg("cp_doc"), py::arg("ignore_flat_creases") = false,
      "Parses a `.cp` document string into a list of `InputSegment` objects; "
      "raises `ValueError` on malformed lines, invalid codes, huge/non-finite "
      "coordinates, zero-length segments, or empty documents.");

  m.def(
      "compute_absolute_face_orders",
      [](const std::vector<std::pair<int, int>>& top_bottom_edges,
         int num_faces) {
        absl::StatusOr<std::vector<int>> orders;
        {
          py::gil_scoped_release release;
          orders = ComputeAbsoluteFaceOrders(top_bottom_edges, num_faces);
        }
        if (!orders.ok()) {
          throw py::value_error(std::string(orders.status().message()));
        }
        return *std::move(orders);
      },
      py::arg("top_bottom_edges"), py::arg("num_faces"),
      "Computes global bottom-to-top face layer ranks (`0..max_rank`) from "
      "directed `(top_face, bottom_face)` edges; raises `ValueError` if "
      "`num_faces > kMaxFaces`.");
}
