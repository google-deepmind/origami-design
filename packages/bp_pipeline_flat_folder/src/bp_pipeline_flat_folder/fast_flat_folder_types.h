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

// Core constants, enumerations, and input segment types for the C++20
// flat-foldability engine.
//
// Provenance: The flat-foldability algorithm and default conventions (`.cp`
// crease codes `1..5`, top-face-up orientation normalization, Kawasaki angle
// tolerance, and layer-order ranking) re-implement Jason S. Ku's MIT-licensed
// Flat-Folder algorithm (ported from the reference Python `flat_folder`
// package).

#ifndef THIRD_PARTY_DEEPMIND_ORIGAMI_DESIGN_PACKAGES_BP_PIPELINE_FLAT_FOLDER_SRC_BP_PIPELINE_FLAT_FOLDER_FAST_FLAT_FOLDER_TYPES_H_
#define THIRD_PARTY_DEEPMIND_ORIGAMI_DESIGN_PACKAGES_BP_PIPELINE_FLAT_FOLDER_SRC_BP_PIPELINE_FLAT_FOLDER_FAST_FLAT_FOLDER_TYPES_H_

#include <cmath>
#include <optional>

namespace bp_pipeline_flat_folder {

// Maximum accepted magnitude `|coord|` at ingress (`.cp`, `AnalyzeLines`,
// `AnalyzeFoldGraph`, `changed_segments`).
//
// Derivation (IEEE-754 binary64, `DBL_MAX ~= 1.7977e308`):
//   1. Unfolded segment squared lengths `dx^2 + dy^2 <= 8 * M^2` and shoelace
//      polygon areas `sum (x1 + x2)(y2 - y1) <= 4 * k * M^2` stay finite when
//      `M <= 1e100` (`8e200 << DBL_MAX`).
//   2. Rigid-body folding across up to `kMaxFaces` (`<= 8192`) faces amplifies
//      coordinates by at most `M_fold <= (1 + 2*sqrt(2)*F) * M < 5e104`, so
//      folded squared lengths (`<= 2e210`) and float sweep-line unit-vector
//      numerators (`<= 1.2e210`) divided by `|denom| >= kFloatEpsilon^2 =
//      1e-32` (`<= 1.2e242 << DBL_MAX`) cannot overflow to `inf` or produce
//      `NaN` in sort comparators.
//   3. Even un-normalized cubic determinants `8 * M^3 <= DBL_MAX` yield the
//      cube-root ceiling `(DBL_MAX / 8)^(1/3) ~= 6.08e102`, which rounds
//      cleanly to `1e100` (matching the sibling exact rational certificate
//      checker bound).
inline constexpr double kMaxCoordinateMagnitude = 1e100;

// Maximum accepted unfolded face count `F` across `.cp`, `AnalyzeLines`,
// `AnalyzeFoldGraph`, and `ComputeAbsoluteFaceOrders`.
//
// Derivation from `FastFlatFolderEngine::kDefaultMaxSupportBytes` (`256 MiB =
// 2^28` bytes) and the dense `F x F` `pair_idx_flat` lookup table
// (`sizeof(int) * F^2 = 4 * F^2` bytes):
//   1. Requiring `4 * F^2 <= 2^28` yields `F <= floor(sqrt(2^28 / 4)) = 2^13 =
//      8192`.
//   2. At `F = 8192`, `pair_idx_flat` is at most `256 MiB` and Stage-2
//      `below_bits + above_bits` (`2 * F * ceil(F / 64) * 8` bytes) is at most
//      `16 MiB` (`<= 272 MiB` total quadratic workspace), and
//      `ComputeAbsoluteFaceOrders` base tables stay under `288 KiB`.
//   3. Guarantees `F^2 <= 2^26 < INT_MAX`, `num_pairs <= F(F - 1)/2 < 2^25 <
//      INT_MAX`, and `F < kMaxTriplePackedFaces` (`2^20`), preventing signed
//      32-bit overflow and 20-bit triple-key overflow.
//   4. Gives ~15x headroom over the largest measured benchmark pattern
//      (`F = 552`) and >22x headroom over downstream `.fold` graph callers
//      (`F <= 360`).
inline constexpr int kMaxFaces = 8192;

// Returns true iff `v` is finite and satisfies `|v| <=
// kMaxCoordinateMagnitude`.
[[nodiscard]] inline bool IsValidCoordinate(double v) {
  return std::isfinite(v) && std::abs(v) <= kMaxCoordinateMagnitude;
}

// Canonical `.cp` crease assignment codes (`1 = 'B'`, `2 = 'M'`, `3 = 'V'`,
// `4 = 'F'`, `5 = 'U'`).
enum class CreaseAssignment : int {
  kBoundary = 1,    // Paper boundary or hole perimeter ('B').
  kMountain = 2,    // Mountain fold ('M').
  kValley = 3,      // Valley fold ('V').
  kFlat = 4,        // Unfolded auxiliary / flat line ('F').
  kUnassigned = 5,  // Active fold with unspecified M/V direction ('U').
};

// Parses an integer crease code in `0..5` (`0` and `5` both map to
// `CreaseAssignment::kUnassigned`). Returns `std::nullopt` for any negative
// code or code `> 5`.
[[nodiscard]] inline constexpr std::optional<CreaseAssignment>
TryParseCreaseAssignment(int raw_code) {
  if (raw_code == 0) return CreaseAssignment::kUnassigned;
  if (raw_code >= static_cast<int>(CreaseAssignment::kBoundary) &&
      raw_code <= static_cast<int>(CreaseAssignment::kUnassigned)) {
    return static_cast<CreaseAssignment>(raw_code);
  }
  return std::nullopt;
}

// Returns true iff `code` is one of the five canonical `CreaseAssignment`
// values (`1..5`).
[[nodiscard]] inline constexpr bool IsValidCreaseAssignment(
    CreaseAssignment code) {
  const int raw = static_cast<int>(code);
  return raw >= static_cast<int>(CreaseAssignment::kBoundary) &&
         raw <= static_cast<int>(CreaseAssignment::kUnassigned);
}

// Swaps `kMountain` and `kValley`, leaving all other crease codes unchanged.
[[nodiscard]] inline constexpr CreaseAssignment FlipMountainValley(
    CreaseAssignment code) {
  if (code == CreaseAssignment::kMountain) return CreaseAssignment::kValley;
  if (code == CreaseAssignment::kValley) return CreaseAssignment::kMountain;
  return code;
}

// Returns true iff `code` represents a 180-degree active fold (`kMountain`,
// `kValley`, or `kUnassigned`).
[[nodiscard]] inline constexpr bool IsActiveCrease(CreaseAssignment code) {
  return code == CreaseAssignment::kMountain ||
         code == CreaseAssignment::kValley ||
         code == CreaseAssignment::kUnassigned;
}

// Orientation normalization applied when constructing an unfolded mesh from
// input segments or a `.fold` graph.
enum class FaceOrientationPolicy {
  // Reflects Y across `[0, 1]` and keeps input M/V assignments unchanged.
  kPreserveInput = 0,
  // Keeps input Y coordinates and swaps M <-> V so face 0 sits top-face-up.
  kFlipToTopFaceUp = 1,
};

// Filter policy for auxiliary flat (`'F'`, code `4`) segments when parsing a
// `.cp` document.
enum class FlatCreasePolicy {
  // Retain type-4 (`'F'`) segments in the parsed segment list.
  kKeepFlatCreases = 0,
  // Drop type-4 (`'F'`) segments during parsing.
  kIgnoreFlatCreases = 1,
};

// Three-valued decision returned by the flat-folder engine.
// Bijection with `SolveResult::flat_foldable`:
//   - `kFoldable`       <=> `flat_foldable == true`
//   - `kNotFoldable`    <=> `flat_foldable == false`
//   - `kCouldNotDecide` <=> `flat_foldable == std::nullopt`
enum class FoldVerdict {
  kCouldNotDecide = 0,
  kFoldable = 1,
  kNotFoldable = 2,
};

// Pipeline stage at which `FastFlatFolderEngine` terminated.
// Default-constructed `SolveResult` starts at `kUninitialized`.
//
// `kFoldedGeometryError` holds ONLY exact geometric proofs of non-foldability
// (second-arrival folded-vertex closure or orientation-parity conflict on the
// exact octilinear path). Engine limitations and numerical/sweep failures
// (derived sweep event limit exceeded, unstable float arrangement, disconnected
// face in single-component BFS, internal BFS guard) report `kCouldNotDecide`.
enum class ExitStage {
  kUninitialized = 0,           // Result has not been populated.
  kSat = 1,                     // Valid flat-folded layering found.
  kStage1Pass = 2,              // Passed Stage 1 under `taco_only = true`.
  kParseError = 3,              // Input/option validation or parse failure.
  kLocalViolation = 4,          // Maekawa or Kawasaki violation at a vertex.
  kFoldedGeometryError = 5,     // Exact folded closure or parity conflict.
  kStage1TortillaOddCycle = 6,  // Stage-1 odd parity cycle on crease init.
  kStage1Tortilla = 7,          // Stage-1 tortilla conflict (unseeded wave).
  kStage1TortillaSeeded = 8,    // Stage-1 tortilla conflict (seeded wave).
  kStage1TacoTaco = 9,          // Stage-1 taco-taco conflict (unseeded wave).
  kStage1TacoTacoSeeded = 10,   // Stage-1 taco-taco conflict (seeded wave).
  kStage2Tortilla = 11,         // Stage-2 odd parity cycle during closure.
  kStage2Transitivity = 12,     // Stage-2 transitivity cycle during closure.
  kStage2TacoTaco = 13,         // Stage-2 taco-taco conflict during closure.
  kStage2ComponentUnsat = 14,   // Stage-2 DPLL component had zero solutions.
  kCouldNotDecide = 15,         // Engine or numerical limitation; inconclusive.
};

// Input line segment `(x1, y1) -> (x2, y2)` with a canonical `.cp` crease
// assignment.
struct InputSegment {
  double x1 = 0.0;
  double y1 = 0.0;
  double x2 = 0.0;
  double y2 = 0.0;
  CreaseAssignment assignment = CreaseAssignment::kUnassigned;
};

// Unassigned coordinate segment `(x1, y1) -> (x2, y2)` used to prioritize
// Wave-1 incremental propagation around recently modified creases.
struct SeedSegment {
  double x1 = 0.0;
  double y1 = 0.0;
  double x2 = 0.0;
  double y2 = 0.0;
};

}  // namespace bp_pipeline_flat_folder

#endif  // THIRD_PARTY_DEEPMIND_ORIGAMI_DESIGN_PACKAGES_BP_PIPELINE_FLAT_FOLDER_SRC_BP_PIPELINE_FLAT_FOLDER_FAST_FLAT_FOLDER_TYPES_H_
