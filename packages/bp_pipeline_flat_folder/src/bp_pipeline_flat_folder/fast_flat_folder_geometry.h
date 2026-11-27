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

// Internal planar arrangement, unfolded mesh, local vertex foldability, and
// rigid-body folded cell arrangement builders.
//
// Provenance: The planar sweep-line arrangement, face extraction, hole
// filtering, local Maekawa/Kawasaki checks, rigid-body folding, and folded cell
// overlap extraction re-implement Jason S. Ku's MIT-licensed Flat-Folder
// geometry pipeline (ported from the reference Python `flat_folder` package),
// accelerated with exact integer arithmetic for octilinear crease patterns.

#ifndef THIRD_PARTY_DEEPMIND_ORIGAMI_DESIGN_PACKAGES_BP_PIPELINE_FLAT_FOLDER_SRC_BP_PIPELINE_FLAT_FOLDER_FAST_FLAT_FOLDER_GEOMETRY_H_
#define THIRD_PARTY_DEEPMIND_ORIGAMI_DESIGN_PACKAGES_BP_PIPELINE_FLAT_FOLDER_SRC_BP_PIPELINE_FLAT_FOLDER_FAST_FLAT_FOLDER_GEOMETRY_H_

#include <algorithm>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "fast_flat_folder_types.h"

namespace bp_pipeline_flat_folder::internal {

// Encodes an undirected pair `{u, v}` into a canonical 64-bit key with
// `min(u, v)` in the upper 32 bits and `max(u, v)` in the lower 32 bits.
[[nodiscard]] inline uint64_t EncodeOrderPair(int u, int v) {
  if (v < u) std::swap(u, v);
  return (static_cast<uint64_t>(static_cast<uint32_t>(u)) << 32) |
         static_cast<uint32_t>(v);
}

// Encodes a directed pair `(u, v)` into a 64-bit key with `u` in the upper
// 32 bits and `v` in the lower 32 bits.
[[nodiscard]] inline uint64_t EncodeDirectedPair(int u, int v) {
  return (static_cast<uint64_t>(static_cast<uint32_t>(u)) << 32) |
         static_cast<uint32_t>(v);
}

// Sorts `items` in ascending order and removes consecutive duplicates in-place.
template <typename T>
inline void SortAndUnique(std::vector<T>* items) {
  std::sort(items->begin(), items->end());
  items->erase(std::unique(items->begin(), items->end()), items->end());
}

// Planar subdivision produced from a collection of 2D line segments.
struct ArrangementGraph {
  // Unique intersection/endpoint vertex coordinates `(x, y)`.
  std::vector<std::pair<double, double>> vertices;
  // True iff `exact_vertices` holds exact scaled integer coordinates `(X, Y)`
  // with `real = coord / exact_scale` (`exact_scale = 4 * S`, so all unfolded
  // vertices have even integer coordinates).
  bool used_exact_octilinear = false;
  int64_t exact_scale = 0;
  std::vector<std::pair<int64_t, int64_t>> exact_vertices;

  // Elementary undirected edges `(u, v)` (`u < v`) between arrangement
  // vertices.
  std::vector<std::pair<int, int>> edge_vertices;
  // Sorted indices of input segments covering each elementary edge.
  std::vector<std::vector<int>> edge_lines;
};

// Parsed unfolded crease-pattern mesh ready for local foldability checks and
// rigid-body folding.
struct UnfoldedMesh {
  bool used_exact_octilinear = false;
  int64_t exact_scale = 0;

  // Unfolded vertex coordinates `(x, y)` and optional exact scaled integers.
  std::vector<std::pair<double, double>> vertex_coords;
  std::vector<std::pair<int64_t, int64_t>> exact_vertex_coords;
  // Counter-clockwise cyclic neighbor vertex lists around each vertex.
  std::vector<std::vector<int>> vertex_adjacencies;
  // Undirected mesh edges `(u, v)`.
  std::vector<std::pair<int, int>> edge_vertices;
  // Indices of input segments overlapping each mesh edge (populated by
  // `BuildUnfoldedMeshFromLines` so `SupportCache` hits can re-extract
  // assignments without running `BuildArrangementFromSegments` a second time).
  std::vector<std::vector<int>> edge_lines;
  // Normalized edge assignments after `FaceOrientationPolicy` swap, hole
  // filtering, and single-face boundary promotion.
  std::vector<CreaseAssignment> edge_assignments;
  // Incident face indices (`1` for boundary, `2` for interior) per edge.
  std::vector<std::vector<int>> edge_faces;
  // Counter-clockwise vertex index loops for each non-hole unfolded face.
  std::vector<std::vector<int>> face_vertices;
};

// Rigid-body folded coordinates, folded cell arrangement, and face-overlap
// tables derived from an `UnfoldedMesh`.
struct FoldedArrangementData {
  // True iff the folded cell arrangement was built on the exact integer path.
  bool used_exact_folded_octilinear = false;

  // Folded 2D vertex coordinates `(x, y)` indexed by unfolded vertex ID.
  std::vector<std::pair<double, double>> folded_vertices;
  // `face_orientations[f] == true` iff face `f` is flipped relative to face `0`
  // (`face_orientations[0] == false`).
  std::vector<bool> face_orientations;

  // Folded arrangement cell vertices `(x, y)` and per-cell vertex index loops.
  std::vector<std::pair<double, double>> cell_points;
  std::vector<std::vector<int>> cell_vertices;
  // Elementary folded segments `(u, v)` referencing `cell_points`, together
  // with their covering unfolded edge indices and adjacent cell indices.
  std::vector<std::pair<int, int>> segment_points;
  std::vector<std::vector<int>> segment_edges;
  std::vector<std::vector<int>> segment_cells;

  // Sorted face IDs covering each folded cell, sorted cell IDs covered by each
  // face, and sorted overlapping face pairs `(f1, f2)` (`f1 < f2`).
  std::vector<std::vector<int>> cell_faces;
  std::vector<std::vector<int>> face_cells;
  std::vector<std::pair<int, int>> face_pairs;
};

// Builds a planar arrangement from `lines` (`(x1, y1)-(x2, y2)`), using exact
// `int64_t` octilinear arithmetic whenever all endpoints quantize onto a scaled
// lattice (or `forced_exact_scale` when positive) and all segments are
// octilinear, or the adaptive-epsilon `double` sweep otherwise. Returns an
// empty `ArrangementGraph` if `lines` is empty, contains invalid coordinates,
// exceeds the sweep event limit, or fails to stabilize numerically.
ArrangementGraph BuildArrangementFromSegments(
    absl::Span<const InputSegment> lines, int64_t forced_exact_scale = 0);

// Resolves per-edge crease assignments from overlapping input segments.
//
// Auxiliary flat lines (`CreaseAssignment::kFlat`) under a crease or boundary
// do not conflict: the non-flat code stands. Two distinct non-flat codes
// (`B`, `M`, `V`, `U`) overlapping on the same elementary edge return
// `InvalidArgumentError` citing 1-based `.cp` line numbers when
// `segment_line_numbers` is provided (or 0-based segment indices otherwise).
absl::StatusOr<std::vector<CreaseAssignment>> ResolveEdgeAssignmentsFromLines(
    absl::Span<const std::vector<int>> edge_lines,
    absl::Span<const InputSegment> lines,
    absl::Span<const int> segment_line_numbers = {});

// Builds an `UnfoldedMesh` from raw input segments, applying
// `orientation_policy`, hole filtering, and single-face boundary promotion.
// Returns `InvalidArgumentError` on empty/malformed segments, conflicting
// collinear crease codes, empty face sets, or `num_faces > kMaxFaces`;
// `ResourceExhaustedError` if the float sweep exceeds its event budget; or
// `AbortedError` if the float sweep cannot stabilize numerically.
absl::StatusOr<UnfoldedMesh> BuildUnfoldedMeshFromLines(
    absl::Span<const InputSegment> lines,
    FaceOrientationPolicy orientation_policy,
    absl::Span<const int> segment_line_numbers = {});

// Builds an `UnfoldedMesh` from pre-parsed `.fold` graph arrays. Validates
// coordinate bounds, crease codes, edge bounds and uniqueness, face bounds and
// manifold half-edge incidence, and `kMaxFaces`, returning
// `InvalidArgumentError` on any violation.
absl::StatusOr<UnfoldedMesh> BuildUnfoldedMeshFromFoldGraph(
    absl::Span<const std::pair<double, double>> vertex_coords,
    absl::Span<const std::pair<int, int>> edge_vertices,
    absl::Span<const CreaseAssignment> edge_assignments,
    absl::Span<const std::vector<int>> optional_face_vertices,
    FaceOrientationPolicy orientation_policy);

// Checks Maekawa's (`|M - V| == 2` and even active degree) and Kawasaki's
// (alternating sector-angle sum within `kawasaki_epsilon` of `pi`) local
// flat-foldability theorems at every interior vertex. Returns a list of
// human-readable violation strings (empty if all interior vertices pass).
std::vector<std::string> CheckLocalVertexFoldability(
    const UnfoldedMesh& mesh,
    absl::Span<const CreaseAssignment> edge_assignments,
    double kawasaki_epsilon);

// Explicit classification for `BuildFoldedArrangement`: only explicit exact
// geometric proofs produce `kNotFoldableProof`; any other folded-stage failure
// (including `InvalidArgumentError` from float-sweep overflow/non-finite
// folded coordinates) defaults to `kCouldNotDecide`.
enum class FoldedArrangementOutcome {
  kOk = 0,
  kNotFoldableProof = 1,
  kCouldNotDecide = 2,
};

// Return value of `BuildFoldedArrangement`, pairing the outcome classification
// with either a diagnostic `error_message` or populated `data`.
struct FoldedArrangementResult {
  FoldedArrangementOutcome outcome = FoldedArrangementOutcome::kCouldNotDecide;
  std::string error_message{};
  FoldedArrangementData data{};
};

// Computes rigid-body folded vertex coordinates, builds the folded planar cell
// arrangement, maps cells <-> faces, and extracts overlapping `face_pairs`.
// Because `edge_assignments` only distinguishes active creases `{M, V, U}` from
// non-creases `{B, F}`, the output is invariant to `M <-> V` flips. Returns
// `kNotFoldableProof` only on exact second-arrival closure or parity conflicts,
// and `kCouldNotDecide` on disconnected meshes, sweep-event-limit hits, or
// numerical/empty-cell-graph failures.
FoldedArrangementResult BuildFoldedArrangement(const UnfoldedMesh& mesh);

}  // namespace bp_pipeline_flat_folder::internal

#endif  // THIRD_PARTY_DEEPMIND_ORIGAMI_DESIGN_PACKAGES_BP_PIPELINE_FLAT_FOLDER_SRC_BP_PIPELINE_FLAT_FOLDER_FAST_FLAT_FOLDER_GEOMETRY_H_
