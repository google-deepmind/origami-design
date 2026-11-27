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

// Global flat-foldability solver and 3-tier incremental session cache.
//
// Pipeline overview:
//   - Tier 2 (`SupportEntry`): For a fixed crease graph up to `M <-> V` flips,
//     builds the unfolded mesh and rigid-body folded cell arrangement, compiles
//     1-to-1 taco-tortilla and tortilla-tortilla constraints into a weighted
//     parity Union-Find over overlapping face pairs, and compiles 6-variable
//     taco-taco constraints onto dense Union-Find roots.
//   - Stage 1 (`RunStage1CreasesAndTaco`): Checks local Maekawa/Kawasaki vertex
//     foldability, assigns initial `M`/`V` crease parities into Union-Find
//     roots, and runs 729-entry Taco-Taco unit propagation (Wave-1 seeded by
//     `changed_segments`, followed by Wave-2 full closure).
//   - Stage 2 (`RunStage2InitialClosure` + `SolveAllComponents`): Propagates
//     64-bit word-parallel transitivity (`z < u < v => z < v`) and taco-taco to
//     fixpoint, partitions remaining unassigned face pairs into connected
//     components matching `solver.get_assignment_components`, and solves each
//     component via reversible-trail DPLL with Tier-3 canonical component
//     caching.

#include "fast_flat_folder_core.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <list>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/absl_check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/types/span.h"
#include "fast_flat_folder_geometry.h"
#include "fast_flat_folder_types.h"

namespace bp_pipeline_flat_folder {
namespace {

using ::bp_pipeline_flat_folder::internal::EncodeDirectedPair;
using ::bp_pipeline_flat_folder::internal::SortAndUnique;

// Maximum `num_pairs * num_words` (`524,288` 64-bit words = `4 MiB`) for
// precomputing dense per-pair common-cell face bitmasks (`pair_common_faces`).
// Larger instances compute common-cell face bitmasks on demand into a scratch
// buffer.
constexpr size_t kMaxPairCommonMaskWords = 524288;

// Maximum unassigned Union-Find roots in a component eligible for Tier-3
// canonical component memoization (`ComponentKey`).
constexpr int kMaxCachedComponentRoots = 64;

// Maximum face count (`2^20 = 1,048,576`) supported by 20-bit triple packing in
// `BuildComponentKey`.
constexpr int kMaxTriplePackedFaces = 1 << 20;

// Geometric tolerances for matching `SeedSegment`s against unfolded mesh edges.
constexpr double kSeedPointOnSegTol = 1e-7;
constexpr double kSeedDegenerateLenSq = 1e-18;
constexpr double kSeedMinEdgeLen = 1e-12;
constexpr double kSeedCollinearOverlapTol = 1e-6;

constexpr std::array<int, 6> kPow3 = {1, 3, 9, 27, 81, 243};

int64_t NowNanos() { return absl::GetCurrentTimeNanos(); }

double ElapsedSeconds(int64_t t0_ns, int64_t t1_ns) {
  return static_cast<double>(t1_ns - t0_ns) * 1e-9;
}

uint64_t CoordBits(double v) {
  // Canonicalize signed zero (-0.0 -> +0.0) and NaN payloads so equal
  // floating-point coordinates map to identical cache keys.
  if (v == 0.0) return uint64_t{0};
  if (std::isnan(v)) return uint64_t{0x7ff8000000000000ULL};
  return std::bit_cast<uint64_t>(v);
}

std::tuple<uint64_t, uint64_t, uint64_t, uint64_t> CanonicalSegmentBits(
    double x1, double y1, double x2, double y2) {
  uint64_t bx1 = CoordBits(x1);
  uint64_t by1 = CoordBits(y1);
  uint64_t bx2 = CoordBits(x2);
  uint64_t by2 = CoordBits(y2);
  bool swap_pts = false;
  if (!std::isnan(x1) && !std::isnan(y1) && !std::isnan(x2) &&
      !std::isnan(y2)) {
    swap_pts = (y2 < y1) || (y2 == y1 && x2 < x1);
  } else {
    swap_pts = std::tie(by2, bx2) < std::tie(by1, bx1);
  }
  if (swap_pts) {
    std::swap(bx1, bx2);
    std::swap(by1, by2);
  }
  return {bx1, by1, bx2, by2};
}

// ============================================================================
// Domain Enums & Unified `SolverConflict` Representation
// ============================================================================

enum class TortillaConstraintKind : uint8_t {
  kTacoTortilla = 1,
  kTortillaTortilla = 2,
};

enum class ConflictKind : uint8_t {
  kParseError = 0,
  kLocalViolation = 1,
  kFoldedGeometryError = 2,
  kTacoTortilla = 3,
  kTortillaTortilla = 4,
  kTacoTaco = 5,
  kTransitivity = 6,
  kComponentUnsat = 7,
  kCouldNotDecide = 8,
};

struct SolverConflict {
  ExitStage stage = ExitStage::kParseError;
  ConflictKind kind = ConflictKind::kParseError;
  int component_index = 0;
  std::string custom_reason{};
  std::vector<int> conflict_faces{};
  bool seeded_hit = false;
};

std::string FormatConflictReason(const SolverConflict& conflict) {
  if (!conflict.custom_reason.empty()) return conflict.custom_reason;
  switch (conflict.kind) {
    case ConflictKind::kParseError:
      return "IO Parser failed";
    case ConflictKind::kLocalViolation:
      return "Local violations found";
    case ConflictKind::kFoldedGeometryError:
      return "Folded geometry conflict";
    case ConflictKind::kTacoTortilla:
      return "taco-tortilla conflict";
    case ConflictKind::kTortillaTortilla:
      return "tortilla-tortilla conflict";
    case ConflictKind::kTacoTaco:
      return "taco-taco conflict";
    case ConflictKind::kTransitivity:
      return "transitivity conflict";
    case ConflictKind::kComponentUnsat:
      ABSL_DCHECK_GT(conflict.component_index, 0);
      return absl::StrCat("Solver failed on component ",
                          conflict.component_index);
    case ConflictKind::kCouldNotDecide:
      return "Precision Error: unstable graph";
  }
  return "IO Parser failed";
}

void PopulateConflictIntoSolveResult(const SolverConflict& conflict,
                                     SolveResult* result) {
  result->ok_parse = (conflict.kind != ConflictKind::kParseError);
  if (conflict.kind == ConflictKind::kCouldNotDecide) {
    result->flat_foldable = std::nullopt;
  } else {
    result->flat_foldable = false;
  }
  // Matches HEAD behavior: `passed_taco_stage` remains `true` only on Stage-2
  // transitivity, taco-taco, and component-DPLL UNSAT exits; tortilla odd-cycle
  // conflicts (including `kStage2Tortilla`) clear `passed_taco_stage = false`.
  if (conflict.stage != ExitStage::kStage2Transitivity &&
      conflict.stage != ExitStage::kStage2TacoTaco &&
      conflict.stage != ExitStage::kStage2ComponentUnsat) {
    result->passed_taco_stage = false;
  }
  result->states = 0;
  result->seeded_hit = conflict.seeded_hit;
  result->stage = conflict.stage;
  result->conflict_reason = FormatConflictReason(conflict);
  result->conflict_faces = conflict.conflict_faces;
}

// ============================================================================
// 729-Entry Taco-Taco Implication Table
// (`Constraints.build_implication_tables`)
// ============================================================================

enum class ImplicationStatus : uint8_t {
  kConflict = 0,
  kAlive = 1,
  kDead = 2,
  kImply = 3,
};

struct TacoTacoTableEntry {
  ImplicationStatus status = ImplicationStatus::kConflict;
  uint8_t num_implied = 0;
  std::array<std::pair<uint8_t, uint8_t>, 6> implied{};
};

constexpr std::array<TacoTacoTableEntry, 729> BuildTacoTacoTable() {
  std::array<TacoTacoTableEntry, 729> table{};
  constexpr std::array<absl::string_view, 16> kValidStrings = {
      "111112", "111121", "111222", "112111", "121112", "121222",
      "122111", "122212", "211121", "211222", "212111", "212221",
      "221222", "222111", "222212", "222221",
  };

  // 0 = conflict, 1 = alive/imply, 2 = dead.
  std::array<uint8_t, 729> state_by_idx{};
  for (absl::string_view s : kValidStrings) {
    int idx = 0;
    for (int j = 0; j < 6; ++j) {
      idx += (s[j] - '0') * kPow3[j];
    }
    state_by_idx[idx] = 2;
    table[idx].status = ImplicationStatus::kDead;
  }

  // Remaining 0-zero indices keep the default `ImplicationStatus::kConflict`.
  for (int zeros = 1; zeros <= 6; ++zeros) {
    for (int idx = 0; idx < 729; ++idx) {
      std::array<int, 6> digits{};
      int tmp = idx;
      int zero_count = 0;
      for (int j = 0; j < 6; ++j) {
        digits[j] = tmp % 3;
        if (digits[j] == 0) ++zero_count;
        tmp /= 3;
      }
      if (zero_count != zeros) continue;

      bool conflict = true;
      bool dead = true;
      TacoTacoTableEntry entry;
      for (int j = 0; j < 6; ++j) {
        if (digits[j] != 0) continue;
        int possible = 0;
        for (int c = 1; c <= 2; ++c) {
          int child_idx = idx + c * kPow3[j];
          uint8_t child_st = state_by_idx[child_idx];
          if (child_st != 2) dead = false;
          if (child_st != 0) possible |= c;
        }
        if (possible != 0) {
          conflict = false;
          if (possible < 3) {
            entry.implied[entry.num_implied++] = {
                static_cast<uint8_t>(j), static_cast<uint8_t>(possible)};
          }
        }
      }
      if (conflict) {
        entry.status = ImplicationStatus::kConflict;
        state_by_idx[idx] = 0;
      } else if (entry.num_implied > 0) {
        entry.status = ImplicationStatus::kImply;
        state_by_idx[idx] = 1;
      } else if (dead) {
        entry.status = ImplicationStatus::kDead;
        state_by_idx[idx] = 2;
      } else {
        entry.status = ImplicationStatus::kAlive;
        state_by_idx[idx] = 1;
      }
      table[idx] = entry;
    }
  }
  return table;
}

const std::array<TacoTacoTableEntry, 729>& GetTacoTacoTable() {
  static constexpr std::array<TacoTacoTableEntry, 729> kTable =
      BuildTacoTacoTable();
  return kTable;
}

// ============================================================================
// Thread-Safe Bounded Generic LRU Cache (Entry-Count + Hard Byte Cap)
// ============================================================================
// Kept self-contained using standard `std::list` + `absl::flat_hash_map` so
// `fast_flat_folder` remains portable to open-source Abseil without internal
// `util/cache` dependencies.

struct CacheCounters {
  size_t size = 0;
  size_t total_bytes = 0;
  uint64_t hits = 0;
  uint64_t misses = 0;
};

constexpr size_t SaturatingAddSize(size_t a, size_t b) {
  constexpr size_t kMax = std::numeric_limits<size_t>::max();
  return (a > kMax - b) ? kMax : (a + b);
}

constexpr size_t SaturatingMulSize(size_t a, size_t b) {
  constexpr size_t kMax = std::numeric_limits<size_t>::max();
  if (a == 0 || b == 0) return 0;
  return (a > kMax / b) ? kMax : (a * b);
}

template <typename Key, typename Value>
class LruCache {
 public:
  explicit LruCache(size_t capacity, size_t max_bytes = 0)
      : capacity_(capacity), max_bytes_(max_bytes) {}

  std::shared_ptr<const Value> Lookup(const Key& key) {
    if (capacity_ == 0) return nullptr;
    absl::MutexLock lock(mu_);
    auto it = map_.find(key);
    if (it == map_.end()) {
      ++misses_;
      return nullptr;
    }
    ++hits_;
    items_.splice(items_.begin(), items_, it->second);
    return it->second->value;
  }

  void Insert(Key key, std::shared_ptr<const Value> value,
              size_t entry_bytes = 0) {
    if (capacity_ == 0) return;
    absl::MutexLock lock(mu_);
    auto it = map_.find(key);
    if (max_bytes_ > 0 && entry_bytes > max_bytes_) {
      if (it != map_.end()) {
        total_bytes_ -= it->second->bytes;
        items_.erase(it->second);
        map_.erase(it);
      }
      return;
    }
    if (it != map_.end()) {
      total_bytes_ -= it->second->bytes;
      it->second->value = std::move(value);
      it->second->bytes = entry_bytes;
      items_.splice(items_.begin(), items_, it->second);
      while (items_.size() > 1 && max_bytes_ > 0 &&
             total_bytes_ > max_bytes_ - entry_bytes) {
        total_bytes_ -= items_.back().bytes;
        map_.erase(items_.back().key);
        items_.pop_back();
      }
      total_bytes_ = SaturatingAddSize(total_bytes_, entry_bytes);
      EvictOverflowLocked();
      return;
    }
    while (!items_.empty() &&
           (items_.size() >= capacity_ ||
            (max_bytes_ > 0 && total_bytes_ > max_bytes_ - entry_bytes))) {
      total_bytes_ -= items_.back().bytes;
      map_.erase(items_.back().key);
      items_.pop_back();
    }
    total_bytes_ = SaturatingAddSize(total_bytes_, entry_bytes);
    items_.push_front(CacheNode{.key = std::move(key),
                                .value = std::move(value),
                                .bytes = entry_bytes});
    map_[items_.front().key] = items_.begin();
  }

  void Clear() {
    absl::MutexLock lock(mu_);
    map_.clear();
    items_.clear();
    total_bytes_ = 0;
  }

  CacheCounters SnapshotCounters() const {
    absl::MutexLock lock(mu_);
    return CacheCounters{
        .size = items_.size(),
        .total_bytes = total_bytes_,
        .hits = hits_,
        .misses = misses_,
    };
  }

 private:
  struct CacheNode {
    Key key;
    std::shared_ptr<const Value> value;
    size_t bytes = 0;
  };
  using ListType = std::list<CacheNode>;

  void EvictOverflowLocked() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    while (!items_.empty() && (items_.size() > capacity_ ||
                               (max_bytes_ > 0 && total_bytes_ > max_bytes_))) {
      total_bytes_ -= items_.back().bytes;
      map_.erase(items_.back().key);
      items_.pop_back();
    }
  }

  const size_t capacity_ = 0;
  const size_t max_bytes_ = 0;
  mutable absl::Mutex mu_;
  ListType items_ ABSL_GUARDED_BY(mu_);
  absl::flat_hash_map<Key, typename ListType::iterator> map_
      ABSL_GUARDED_BY(mu_);
  size_t total_bytes_ ABSL_GUARDED_BY(mu_) = 0;
  uint64_t hits_ ABSL_GUARDED_BY(mu_) = 0;
  uint64_t misses_ ABSL_GUARDED_BY(mu_) = 0;
};

// ============================================================================
// Cache Keys (`MemoKey`, `SupportKey`, `ComponentKey`)
// ============================================================================

struct SegmentBitsKey {
  uint64_t x1 = 0;
  uint64_t y1 = 0;
  uint64_t x2 = 0;
  uint64_t y2 = 0;
  CreaseAssignment code = CreaseAssignment::kUnassigned;

  bool operator==(const SegmentBitsKey& o) const = default;
  template <typename H>
  friend H AbslHashValue(H h, const SegmentBitsKey& k) {
    return H::combine(std::move(h), k.x1, k.y1, k.x2, k.y2, k.code);
  }
};

struct SupportKey {
  bool flip_to_top_face_up = true;
  std::vector<SegmentBitsKey> segments;

  bool operator==(const SupportKey& o) const = default;
  template <typename H>
  friend H AbslHashValue(H h, const SupportKey& k) {
    return H::combine(std::move(h), k.flip_to_top_face_up, k.segments);
  }
};

struct SeedSegmentBitsKey {
  uint64_t x1 = 0;
  uint64_t y1 = 0;
  uint64_t x2 = 0;
  uint64_t y2 = 0;

  bool operator==(const SeedSegmentBitsKey& o) const = default;
  bool operator<(const SeedSegmentBitsKey& o) const {
    return std::tie(x1, y1, x2, y2) < std::tie(o.x1, o.y1, o.x2, o.y2);
  }
  template <typename H>
  friend H AbslHashValue(H h, const SeedSegmentBitsKey& k) {
    return H::combine(std::move(h), k.x1, k.y1, k.x2, k.y2);
  }
};

struct MemoKey {
  bool flip_to_top_face_up = true;
  bool taco_only = false;
  bool compute_layer_order = false;
  bool ignore_flat_creases = false;
  int solution_limit = 1;
  uint64_t kawasaki_eps_bits = 0;
  std::vector<int> face_group_ranks;
  std::vector<SeedSegmentBitsKey> changed_segments;
  std::vector<SegmentBitsKey> segments;

  bool operator==(const MemoKey& o) const = default;
  template <typename H>
  friend H AbslHashValue(H h, const MemoKey& k) {
    return H::combine(std::move(h), k.flip_to_top_face_up, k.taco_only,
                      k.compute_layer_order, k.ignore_flat_creases,
                      k.solution_limit, k.kawasaki_eps_bits, k.face_group_ranks,
                      k.changed_segments, k.segments);
  }
};

struct ComponentKey {
  int num_local_roots = 0;
  int solution_limit = 1;
  std::vector<std::array<int32_t, 6>> taco_clauses;
  std::vector<std::array<int32_t, 3>> trans_clauses;

  bool operator==(const ComponentKey& o) const = default;
  template <typename H>
  friend H AbslHashValue(H h, const ComponentKey& k) {
    return H::combine(std::move(h), k.num_local_roots, k.solution_limit,
                      k.taco_clauses, k.trans_clauses);
  }
};

struct ComponentCacheValue {
  // Number of valid solutions enumerated for this component (bounded by
  // `sol_limit` when positive and saturated at `INT64_MAX`). `0` means UNSAT.
  int64_t num_solutions = 0;
  // First valid assignment (`num_local_roots` bits, `0` or `1`) in the
  // canonical local-root orientation. Empty when `num_solutions == 0`.
  std::vector<uint8_t> first_solution;
};

// ============================================================================
// Tier-2 `SupportEntry` (M/V-Invariant Folded Arrangement & Compiled UF/Tacos)
// ============================================================================

struct UfTreeEdge {
  int neighbor_pair = 0;
  TortillaConstraintKind constraint_type =
      TortillaConstraintKind::kTacoTortilla;
  uint8_t num_faces = 0;
  std::array<int, 4> faces{};
};

struct OddTortillaCycle {
  int p0 = 0;
  int p1 = 0;
  TortillaConstraintKind constraint_type =
      TortillaConstraintKind::kTacoTortilla;
  std::vector<int> cycle_faces;
};

struct CompiledTacoTaco {
  std::array<int, 4> faces{};
  std::array<int, 6> pairs{};
  std::array<int, 6> roots{};
  std::array<uint8_t, 6> root_flips{};
};

struct RootMember {
  int pair_idx = 0;
  int u_face = 0;
  int v_face = 0;
  uint8_t xor_to_root = 0;
};

struct SupportEntry {
  internal::FoldedArrangementOutcome folded_outcome =
      internal::FoldedArrangementOutcome::kCouldNotDecide;
  std::string error_message;

  internal::UnfoldedMesh mesh;
  internal::FoldedArrangementData folded;

  int num_faces = 0;
  int num_cells = 0;
  int num_pairs = 0;
  int words = 0;

  std::vector<int> pair_idx_flat;    // `num_faces * num_faces`, `-1` if none
  std::vector<uint64_t> cell_masks;  // `num_cells * words`
  bool has_precomputed_pair_common = false;
  std::vector<uint64_t> pair_common_faces;  // `num_pairs * words` (if <= 4 MB)

  // Weighted parity Union-Find over `0 .. num_pairs - 1`.
  int num_roots = 0;
  std::vector<int> pair_root_idx;  // `0 .. num_roots - 1`
  std::vector<uint8_t> pair_xor_to_root;
  std::vector<int> root_rep_pair;  // `num_roots`
  std::vector<std::vector<RootMember>> root_members;
  std::vector<std::vector<UfTreeEdge>> uf_tree_adj;
  std::vector<OddTortillaCycle> odd_cycles;
  std::vector<int> root_odd_cycle_idx;  // `-1` or index into `odd_cycles`

  std::vector<CompiledTacoTaco> taco_taco;
  std::vector<std::vector<int>> taco_by_root;  // `num_roots` -> taco indices
  std::vector<std::vector<int>> taco_by_pair;  // `num_pairs` -> taco indices

  // Full tortilla constraint neighbor lists in canonical `constraint_table`
  // discovery order, plus taco-tortilla face connectivity components
  // (`face_constraints.build_face_graph_components`) for exact component
  // partitioning and tie-breaking parity with `solver.py`.
  std::vector<std::vector<int>> type1_neighbors;
  std::vector<std::vector<int>> type2_neighbors;
  std::vector<absl::flat_hash_map<int, int>> taco_tortilla_components;

  bool IsTransPrunedByFaceComponents(int f1, int f2, int f3) const {
    const auto& c1 = taco_tortilla_components[f1];
    if (auto it12 = c1.find(f2); it12 != c1.end()) {
      auto it13 = c1.find(f3);
      if (it13 != c1.end() && it12->second == it13->second) return true;
    }
    const auto& c2 = taco_tortilla_components[f2];
    if (auto it21 = c2.find(f1); it21 != c2.end()) {
      auto it23 = c2.find(f3);
      if (it23 != c2.end() && it21->second == it23->second) return true;
    }
    const auto& c3 = taco_tortilla_components[f3];
    if (auto it31 = c3.find(f1); it31 != c3.end()) {
      auto it32 = c3.find(f2);
      if (it32 != c3.end() && it31->second == it32->second) return true;
    }
    return false;
  }
};

template <typename T>
size_t NestedVectorBytes(const std::vector<std::vector<T>>& vecs) {
  size_t bytes = SaturatingMulSize(vecs.capacity(), sizeof(std::vector<T>));
  for (const auto& inner : vecs) {
    bytes = SaturatingAddSize(bytes,
                              SaturatingMulSize(inner.capacity(), sizeof(T)));
  }
  return bytes;
}

size_t EstimateSupportEntryBytes(const SupportEntry& sup) {
  auto add_vec = [](size_t acc, size_t cap, size_t elem_size) {
    return SaturatingAddSize(acc, SaturatingMulSize(cap, elem_size));
  };
  size_t bytes =
      SaturatingAddSize(sizeof(SupportEntry), sup.error_message.capacity());

  // `UnfoldedMesh` dynamic allocations.
  bytes = add_vec(bytes, sup.mesh.vertex_coords.capacity(),
                  sizeof(std::pair<double, double>));
  bytes = add_vec(bytes, sup.mesh.exact_vertex_coords.capacity(),
                  sizeof(std::pair<int64_t, int64_t>));
  bytes =
      SaturatingAddSize(bytes, NestedVectorBytes(sup.mesh.vertex_adjacencies));
  bytes = add_vec(bytes, sup.mesh.edge_vertices.capacity(),
                  sizeof(std::pair<int, int>));
  bytes = SaturatingAddSize(bytes, NestedVectorBytes(sup.mesh.edge_lines));
  bytes = add_vec(bytes, sup.mesh.edge_assignments.capacity(),
                  sizeof(CreaseAssignment));
  bytes = SaturatingAddSize(bytes, NestedVectorBytes(sup.mesh.edge_faces));
  bytes = SaturatingAddSize(bytes, NestedVectorBytes(sup.mesh.face_vertices));

  // `FoldedArrangementData` dynamic allocations.
  bytes = add_vec(bytes, sup.folded.folded_vertices.capacity(),
                  sizeof(std::pair<double, double>));
  const size_t orient_cap = sup.folded.face_orientations.capacity();
  bytes =
      SaturatingAddSize(bytes, orient_cap / 8 + (orient_cap % 8 != 0 ? 1 : 0));
  bytes = add_vec(bytes, sup.folded.cell_points.capacity(),
                  sizeof(std::pair<double, double>));
  bytes = SaturatingAddSize(bytes, NestedVectorBytes(sup.folded.cell_vertices));
  bytes = add_vec(bytes, sup.folded.segment_points.capacity(),
                  sizeof(std::pair<int, int>));
  bytes = SaturatingAddSize(bytes, NestedVectorBytes(sup.folded.segment_edges));
  bytes = SaturatingAddSize(bytes, NestedVectorBytes(sup.folded.segment_cells));
  bytes = SaturatingAddSize(bytes, NestedVectorBytes(sup.folded.cell_faces));
  bytes = SaturatingAddSize(bytes, NestedVectorBytes(sup.folded.face_cells));
  bytes = add_vec(bytes, sup.folded.face_pairs.capacity(),
                  sizeof(std::pair<int, int>));

  // Compiled solver tables.
  bytes = add_vec(bytes, sup.pair_idx_flat.capacity(), sizeof(int));
  bytes = add_vec(bytes, sup.cell_masks.capacity(), sizeof(uint64_t));
  bytes = add_vec(bytes, sup.pair_common_faces.capacity(), sizeof(uint64_t));
  bytes = add_vec(bytes, sup.pair_root_idx.capacity(), sizeof(int));
  bytes = add_vec(bytes, sup.pair_xor_to_root.capacity(), sizeof(uint8_t));
  bytes = add_vec(bytes, sup.root_rep_pair.capacity(), sizeof(int));
  bytes = SaturatingAddSize(bytes, NestedVectorBytes(sup.root_members));
  bytes = SaturatingAddSize(bytes, NestedVectorBytes(sup.uf_tree_adj));
  bytes = add_vec(bytes, sup.odd_cycles.capacity(), sizeof(OddTortillaCycle));
  for (const auto& oc : sup.odd_cycles) {
    bytes = add_vec(bytes, oc.cycle_faces.capacity(), sizeof(int));
  }
  bytes = add_vec(bytes, sup.root_odd_cycle_idx.capacity(), sizeof(int));
  bytes = add_vec(bytes, sup.taco_taco.capacity(), sizeof(CompiledTacoTaco));
  bytes = SaturatingAddSize(bytes, NestedVectorBytes(sup.taco_by_root));
  bytes = SaturatingAddSize(bytes, NestedVectorBytes(sup.taco_by_pair));
  bytes = SaturatingAddSize(bytes, NestedVectorBytes(sup.type1_neighbors));
  bytes = SaturatingAddSize(bytes, NestedVectorBytes(sup.type2_neighbors));
  bytes = add_vec(bytes, sup.taco_tortilla_components.capacity(),
                  sizeof(absl::flat_hash_map<int, int>));
  for (const auto& map : sup.taco_tortilla_components) {
    bytes =
        add_vec(bytes, map.capacity(), sizeof(std::pair<const int, int>) + 1);
  }
  return bytes;
}

// Extracts exact shortest-path faces in `uf_tree_adj` between `start_p` and
// `goal_p` (plus `extra_faces`), avoiding whole-component over-reporting.
std::vector<int> ExtractUfTreePathFaces(
    const std::vector<std::vector<UfTreeEdge>>& uf_tree_adj,
    absl::Span<const std::pair<int, int>> face_pairs, int start_p, int goal_p,
    absl::Span<const int> extra_faces) {
  std::vector<int> out(extra_faces.begin(), extra_faces.end());
  if (start_p >= 0 && start_p < static_cast<int>(face_pairs.size())) {
    out.push_back(face_pairs[start_p].first);
    out.push_back(face_pairs[start_p].second);
  }
  if (goal_p >= 0 && goal_p < static_cast<int>(face_pairs.size())) {
    out.push_back(face_pairs[goal_p].first);
    out.push_back(face_pairs[goal_p].second);
  }
  if (start_p == goal_p || start_p < 0 || goal_p < 0) {
    SortAndUnique(&out);
    return out;
  }

  absl::flat_hash_map<int, std::pair<int, const UfTreeEdge*>> parent;
  std::vector<int> bfs = {start_p};
  parent[start_p] = {-1, nullptr};
  bool found = false;
  for (size_t qi = 0; qi < bfs.size() && !found; ++qi) {
    int cur = bfs[qi];
    for (const auto& edge : uf_tree_adj[cur]) {
      if (parent.contains(edge.neighbor_pair)) continue;
      parent[edge.neighbor_pair] = {cur, &edge};
      if (edge.neighbor_pair == goal_p) {
        found = true;
        break;
      }
      bfs.push_back(edge.neighbor_pair);
    }
  }

  if (found) {
    int cur = goal_p;
    while (cur != start_p) {
      const auto& [prev, edge_ptr] = parent[cur];
      if (edge_ptr != nullptr) {
        for (int k = 0; k < edge_ptr->num_faces; ++k) {
          out.push_back(edge_ptr->faces[k]);
        }
      }
      cur = prev;
    }
  }
  SortAndUnique(&out);
  return out;
}

struct RawTacoTaco {
  std::array<int, 4> faces{};
  std::array<int, 6> pairs{};
  std::array<uint8_t, 6> flips{};
};

void InitPairLookupAndCellMasks(SupportEntry* sup) {
  sup->num_faces = static_cast<int>(sup->mesh.face_vertices.size());
  sup->num_cells = static_cast<int>(sup->folded.cell_vertices.size());
  sup->num_pairs = static_cast<int>(sup->folded.face_pairs.size());
  sup->words = (sup->num_faces + 63) / 64;

  const int num_faces = sup->num_faces;
  const int num_cells = sup->num_cells;
  const int num_pairs = sup->num_pairs;
  const int num_words = sup->words;

  sup->pair_idx_flat.assign(static_cast<size_t>(num_faces) * num_faces, -1);
  for (int p = 0; p < num_pairs; ++p) {
    auto [u, v] = sup->folded.face_pairs[p];
    sup->pair_idx_flat[static_cast<size_t>(u) * num_faces + v] = p;
    sup->pair_idx_flat[static_cast<size_t>(v) * num_faces + u] = p;
  }

  sup->cell_masks.assign(static_cast<size_t>(num_cells) * num_words, 0ULL);
  for (int ci = 0; ci < num_cells; ++ci) {
    uint64_t* cm = &sup->cell_masks[static_cast<size_t>(ci) * num_words];
    for (int fi : sup->folded.cell_faces[ci]) {
      cm[fi >> 6] |= (1ULL << (fi & 63));
    }
  }

  if (static_cast<size_t>(num_pairs) * num_words <= kMaxPairCommonMaskWords) {
    sup->has_precomputed_pair_common = true;
    sup->pair_common_faces.assign(static_cast<size_t>(num_pairs) * num_words,
                                  0ULL);
    for (int p = 0; p < num_pairs; ++p) {
      auto [f1, f2] = sup->folded.face_pairs[p];
      uint64_t* dst =
          &sup->pair_common_faces[static_cast<size_t>(p) * num_words];
      const auto& c1 = sup->folded.face_cells[f1];
      const auto& c2 = sup->folded.face_cells[f2];
      size_t i1 = 0;
      size_t i2 = 0;
      while (i1 < c1.size() && i2 < c2.size()) {
        if (c1[i1] == c2[i2]) {
          const uint64_t* cm =
              &sup->cell_masks[static_cast<size_t>(c1[i1]) * num_words];
          for (int wd = 0; wd < num_words; ++wd) {
            dst[wd] |= cm[wd];
          }
          ++i1;
          ++i2;
        } else if (c1[i1] < c2[i2]) {
          ++i1;
        } else {
          ++i2;
        }
      }
      dst[f1 >> 6] &= ~(1ULL << (f1 & 63));
      dst[f2 >> 6] &= ~(1ULL << (f2 & 63));
    }
  }
}

void BuildTacoTortillaFaceComponents(
    int num_faces,
    absl::Span<const std::vector<std::pair<int, int>>> edges_by_f3,
    std::vector<absl::flat_hash_map<int, int>>* components) {
  components->assign(num_faces, {});
  for (int f3 = 0; f3 < num_faces; ++f3) {
    const auto& edges = edges_by_f3[f3];
    if (edges.empty()) continue;
    absl::flat_hash_map<int, std::vector<int>> adj;
    for (auto [u, v] : edges) {
      adj[u].push_back(v);
      adj[v].push_back(u);
    }
    auto& comp_map = (*components)[f3];
    comp_map.reserve(adj.size());
    int comp_id = 0;
    std::vector<int> bfs;
    for (const auto& [start_f, _] : adj) {
      if (comp_map.contains(start_f)) continue;
      bfs.clear();
      bfs.push_back(start_f);
      comp_map[start_f] = comp_id;
      for (size_t qi = 0; qi < bfs.size(); ++qi) {
        int cur = bfs[qi];
        for (int nb : adj[cur]) {
          if (comp_map.emplace(nb, comp_id).second) {
            bfs.push_back(nb);
          }
        }
      }
      ++comp_id;
    }
  }
}

void FinalizeUnionFindRootsAndCompileTacos(
    absl::Span<const int> uf_parent, absl::Span<const uint8_t> uf_xor,
    absl::Span<const RawTacoTaco> raw_tacos, SupportEntry* sup) {
  const int num_pairs = sup->num_pairs;
  auto find_uf = [&](int u) -> std::pair<int, uint8_t> {
    uint8_t xr = 0;
    while (uf_parent[u] != u) {
      xr ^= uf_xor[u];
      u = uf_parent[u];
    }
    return {u, xr};
  };

  sup->pair_root_idx.assign(num_pairs, -1);
  sup->pair_xor_to_root.assign(num_pairs, 0);
  absl::flat_hash_map<int, int> raw_root_to_dense;
  raw_root_to_dense.reserve(num_pairs);
  for (int p = 0; p < num_pairs; ++p) {
    auto [raw_r, xr] = find_uf(p);
    auto [it, inserted] = raw_root_to_dense.emplace(raw_r, sup->num_roots);
    if (inserted) {
      ++sup->num_roots;
      sup->root_rep_pair.push_back(p);
      sup->root_members.emplace_back();
    }
    int dense_r = it->second;
    sup->pair_root_idx[p] = dense_r;
    sup->pair_xor_to_root[p] = xr;
  }

  std::vector<uint8_t> root_base_flip(sup->num_roots, 0);
  for (int r = 0; r < sup->num_roots; ++r) {
    root_base_flip[r] = sup->pair_xor_to_root[sup->root_rep_pair[r]];
  }
  for (int p = 0; p < num_pairs; ++p) {
    int r = sup->pair_root_idx[p];
    sup->pair_xor_to_root[p] ^= root_base_flip[r];
    sup->root_members[r].push_back(
        RootMember{.pair_idx = p,
                   .u_face = sup->folded.face_pairs[p].first,
                   .v_face = sup->folded.face_pairs[p].second,
                   .xor_to_root = sup->pair_xor_to_root[p]});
  }

  sup->root_odd_cycle_idx.assign(sup->num_roots, -1);
  for (int ci = 0; ci < static_cast<int>(sup->odd_cycles.size()); ++ci) {
    int r = sup->pair_root_idx[sup->odd_cycles[ci].p0];
    if (sup->root_odd_cycle_idx[r] < 0) {
      sup->root_odd_cycle_idx[r] = ci;
    }
  }

  sup->taco_taco.reserve(raw_tacos.size());
  sup->taco_by_root.assign(sup->num_roots, {});
  sup->taco_by_pair.assign(num_pairs, {});
  for (const auto& rt : raw_tacos) {
    CompiledTacoTaco ct{.faces = rt.faces, .pairs = rt.pairs};
    for (int s = 0; s < 6; ++s) {
      int p = rt.pairs[s];
      ct.roots[s] = sup->pair_root_idx[p];
      ct.root_flips[s] = rt.flips[s] ^ sup->pair_xor_to_root[p];
    }
    int tid = static_cast<int>(sup->taco_taco.size());
    sup->taco_taco.push_back(ct);
    for (int s = 0; s < 6; ++s) {
      int r = ct.roots[s];
      auto& rvec = sup->taco_by_root[r];
      if (absl::c_find(rvec, tid) == rvec.end()) {
        rvec.push_back(tid);
      }
      int p = ct.pairs[s];
      auto& pvec = sup->taco_by_pair[p];
      if (absl::c_find(pvec, tid) == pvec.end()) {
        pvec.push_back(tid);
      }
    }
  }
}

struct TortillaUfBuilder {
  explicit TortillaUfBuilder(SupportEntry* sup_in)
      : sup(sup_in),
        uf_parent(sup_in->num_pairs),
        uf_xor(sup_in->num_pairs, 0),
        uf_rank(sup_in->num_pairs, 0),
        taco_tortilla_edges_by_f3(sup_in->num_faces) {
    for (int p = 0; p < sup->num_pairs; ++p) uf_parent[p] = p;
    sup->uf_tree_adj.assign(sup->num_pairs, {});
    sup->type1_neighbors.assign(sup->num_pairs, {});
    sup->type2_neighbors.assign(sup->num_pairs, {});
  }

  std::pair<int, uint8_t> FindUf(int u) const {
    uint8_t xr = 0;
    while (uf_parent[u] != u) {
      xr ^= uf_xor[u];
      u = uf_parent[u];
    }
    return {u, xr};
  }

  std::pair<int, uint8_t> LookupPair(int fa, int fb) const {
    int p = sup->pair_idx_flat[static_cast<size_t>(fa) * sup->num_faces + fb];
    uint8_t flp = (fb < fa) ? 1 : 0;
    return {p, flp};
  }

  bool AddTortilla(TortillaConstraintKind kind, absl::Span<const int> faces,
                   int fa0, int fb0, int fa1, int fb1, uint8_t base_xor) {
    auto [p0, flp0] = LookupPair(fa0, fb0);
    auto [p1, flp1] = LookupPair(fa1, fb1);
    if (p0 < 0 || p1 < 0) {
      sup->folded_outcome = internal::FoldedArrangementOutcome::kCouldNotDecide;
      sup->error_message = "Constraint variable missing";
      return false;
    }
    if (kind == TortillaConstraintKind::kTacoTortilla) {
      sup->type1_neighbors[p0].push_back(p1);
      if (p1 != p0) sup->type1_neighbors[p1].push_back(p0);
      if (faces.size() == 3) {
        taco_tortilla_edges_by_f3[faces[2]].push_back({faces[0], faces[1]});
      }
    } else {
      sup->type2_neighbors[p0].push_back(p1);
      if (p1 != p0) sup->type2_neighbors[p1].push_back(p0);
    }
    uint8_t req_xor = base_xor ^ flp0 ^ flp1;
    auto [r0, x0] = FindUf(p0);
    auto [r1, x1] = FindUf(p1);
    if (r0 == r1) {
      if ((x0 ^ x1) != req_xor) {
        sup->odd_cycles.push_back(OddTortillaCycle{
            .p0 = p0,
            .p1 = p1,
            .constraint_type = kind,
            .cycle_faces = ExtractUfTreePathFaces(
                sup->uf_tree_adj, sup->folded.face_pairs, p0, p1, faces)});
      }
      return true;
    }
    uint8_t link_xor = x0 ^ x1 ^ req_xor;
    if (uf_rank[r0] < uf_rank[r1]) {
      uf_parent[r0] = r1;
      uf_xor[r0] = link_xor;
    } else {
      uf_parent[r1] = r0;
      uf_xor[r1] = link_xor;
      if (uf_rank[r0] == uf_rank[r1]) ++uf_rank[r0];
    }
    UfTreeEdge e01{.neighbor_pair = p1,
                   .constraint_type = kind,
                   .num_faces = static_cast<uint8_t>(faces.size())};
    UfTreeEdge e10{.neighbor_pair = p0,
                   .constraint_type = kind,
                   .num_faces = static_cast<uint8_t>(faces.size())};
    for (size_t k = 0; k < faces.size() && k < 4; ++k) {
      e01.faces[k] = faces[k];
      e10.faces[k] = faces[k];
    }
    sup->uf_tree_adj[p0].push_back(e01);
    sup->uf_tree_adj[p1].push_back(e10);
    return true;
  }

  SupportEntry* sup;
  std::vector<int> uf_parent;
  std::vector<uint8_t> uf_xor;
  std::vector<int> uf_rank;
  std::vector<std::vector<std::pair<int, int>>> taco_tortilla_edges_by_f3;
};

bool CompileEdgeEdgeOverlapConstraints(SupportEntry* sup, TortillaUfBuilder* uf,
                                       std::vector<RawTacoTaco>* raw_tacos) {
  const int num_faces = sup->num_faces;
  const int num_edges = static_cast<int>(sup->mesh.edge_faces.size());
  std::vector<std::vector<int>> ee_mates(num_edges);
  for (const auto& edges : sup->folded.segment_edges) {
    for (size_t j = 0; j < edges.size(); ++j) {
      for (size_t k = j + 1; k < edges.size(); ++k) {
        int e1 = edges[j];
        int e2 = edges[k];
        if (e2 < e1) std::swap(e1, e2);
        if (e1 >= 0 && e2 < num_edges) {
          ee_mates[e1].push_back(e2);
        }
      }
    }
  }

  absl::flat_hash_set<std::tuple<int, int, int, int>> seen_tacos;
  for (int e1 = 0; e1 < num_edges; ++e1) {
    if (sup->mesh.edge_faces[e1].size() != 2) continue;
    auto& mates = ee_mates[e1];
    if (mates.empty()) continue;
    SortAndUnique(&mates);

    int f1 = sup->mesh.edge_faces[e1][0];
    int f2 = sup->mesh.edge_faces[e1][1];
    int ov12 =
        (sup->pair_idx_flat[static_cast<size_t>(f1) * num_faces + f2] >= 0) ? 1
                                                                            : 0;

    for (int e2 : mates) {
      if (sup->mesh.edge_faces[e2].size() != 2) continue;
      int f3 = sup->mesh.edge_faces[e2][0];
      int f4 = sup->mesh.edge_faces[e2][1];
      int ov13 =
          (sup->pair_idx_flat[static_cast<size_t>(f1) * num_faces + f3] >= 0)
              ? 1
              : 0;
      int ov14 =
          (sup->pair_idx_flat[static_cast<size_t>(f1) * num_faces + f4] >= 0)
              ? 1
              : 0;
      const int choice = (ov12 << 2) | (ov13 << 1) | ov14;
      switch (choice) {
        case 0: {
          std::array<int, 3> fc = {f3, f4, f2};
          if (!uf->AddTortilla(TortillaConstraintKind::kTacoTortilla, fc, f3,
                               f2, f2, f4, 1)) {
            return false;
          }
          break;
        }
        case 1: {
          std::array<int, 4> fc = {f1, f2, f4, f3};
          if (!uf->AddTortilla(TortillaConstraintKind::kTortillaTortilla, fc,
                               f1, f4, f2, f3, 0)) {
            return false;
          }
          break;
        }
        case 2: {
          std::array<int, 4> fc = {f1, f2, f3, f4};
          if (!uf->AddTortilla(TortillaConstraintKind::kTortillaTortilla, fc,
                               f1, f3, f2, f4, 0)) {
            return false;
          }
          break;
        }
        case 3: {
          std::array<int, 3> fc = {f3, f4, f1};
          if (!uf->AddTortilla(TortillaConstraintKind::kTacoTortilla, fc, f3,
                               f1, f1, f4, 1)) {
            return false;
          }
          break;
        }
        case 4:
          break;
        case 5: {
          std::array<int, 3> fc = {f1, f2, f4};
          if (!uf->AddTortilla(TortillaConstraintKind::kTacoTortilla, fc, f1,
                               f4, f4, f2, 1)) {
            return false;
          }
          break;
        }
        case 6: {
          std::array<int, 3> fc = {f1, f2, f3};
          if (!uf->AddTortilla(TortillaConstraintKind::kTacoTortilla, fc, f1,
                               f3, f3, f2, 1)) {
            return false;
          }
          break;
        }
        case 7: {
          if (!seen_tacos.insert({f1, f2, f3, f4}).second) break;
          static constexpr std::array<std::pair<int, int>, 6> kSlotIdx = {
              {{0, 1}, {2, 3}, {2, 1}, {0, 3}, {0, 2}, {1, 3}}};
          RawTacoTaco rt{.faces = {f1, f2, f3, f4}};
          for (int s = 0; s < 6; ++s) {
            int fa = rt.faces[kSlotIdx[s].first];
            int fb = rt.faces[kSlotIdx[s].second];
            auto [p_idx, flp] = uf->LookupPair(fa, fb);
            if (p_idx < 0) {
              sup->folded_outcome =
                  internal::FoldedArrangementOutcome::kCouldNotDecide;
              sup->error_message = "Constraint variable missing";
              return false;
            }
            rt.pairs[s] = p_idx;
            rt.flips[s] = flp;
          }
          raw_tacos->push_back(rt);
          break;
        }
        default:
          ABSL_DCHECK(false) << "Unexpected 3-bit overlap choice: " << choice;
          break;
      }
    }
  }
  return true;
}

bool CompileEdgeFaceOverlapConstraints(SupportEntry* sup,
                                       TortillaUfBuilder* uf) {
  const int num_faces = sup->num_faces;
  const int num_words = sup->words;
  const int num_edges = static_cast<int>(sup->mesh.edge_faces.size());
  std::vector<std::vector<int>> ef_faces(num_edges);
  for (size_t si = 0; si < sup->folded.segment_cells.size(); ++si) {
    if (sup->folded.segment_cells[si].size() != 2) continue;
    int c1 = sup->folded.segment_cells[si][0];
    int c2 = sup->folded.segment_cells[si][1];
    const uint64_t* m1 = &sup->cell_masks[static_cast<size_t>(c1) * num_words];
    const uint64_t* m2 = &sup->cell_masks[static_cast<size_t>(c2) * num_words];
    std::vector<int> shared_faces;
    for (int wd = 0; wd < num_words; ++wd) {
      uint64_t both = m1[wd] & m2[wd];
      while (both != 0) {
        int bit = std::countr_zero(both);
        shared_faces.push_back((wd << 6) + bit);
        both &= (both - 1);
      }
    }
    if (shared_faces.empty()) continue;
    for (int ei : sup->folded.segment_edges[si]) {
      if (ei >= 0 && ei < num_edges) {
        ef_faces[ei].insert(ef_faces[ei].end(), shared_faces.begin(),
                            shared_faces.end());
      }
    }
  }

  for (int ei = 0; ei < num_edges; ++ei) {
    if (sup->mesh.edge_faces[ei].size() != 2 || ef_faces[ei].empty()) continue;
    auto& flist = ef_faces[ei];
    SortAndUnique(&flist);
    int f1 = sup->mesh.edge_faces[ei][0];
    int f2 = sup->mesh.edge_faces[ei][1];
    bool ov12 =
        (sup->pair_idx_flat[static_cast<size_t>(f1) * num_faces + f2] >= 0);
    for (int f3 : flist) {
      if (f3 == f1 || f3 == f2) continue;
      if (ov12) {
        std::array<int, 3> fc = {f1, f2, f3};
        if (!uf->AddTortilla(TortillaConstraintKind::kTacoTortilla, fc, f1, f3,
                             f3, f2, 1)) {
          return false;
        }
      } else {
        std::array<int, 4> fc = {f1, f2, f3, f3};
        if (!uf->AddTortilla(TortillaConstraintKind::kTortillaTortilla, fc, f1,
                             f3, f2, f3, 0)) {
          return false;
        }
      }
    }
  }
  return true;
}

void CompileSupportConstraints(SupportEntry* sup) {
  InitPairLookupAndCellMasks(sup);
  TortillaUfBuilder uf(sup);
  std::vector<RawTacoTaco> raw_tacos;
  if (!CompileEdgeEdgeOverlapConstraints(sup, &uf, &raw_tacos) ||
      !CompileEdgeFaceOverlapConstraints(sup, &uf)) {
    return;
  }
  BuildTacoTortillaFaceComponents(sup->num_faces, uf.taco_tortilla_edges_by_f3,
                                  &sup->taco_tortilla_components);
  FinalizeUnionFindRootsAndCompileTacos(uf.uf_parent, uf.uf_xor, raw_tacos,
                                        sup);
}

// ============================================================================
// Wave-1 Seed Face Discovery (`options.changed_segments`)
// ============================================================================

std::vector<int> FindSeedFaces(const SupportEntry& sup,
                               absl::Span<const SeedSegment> changed_segments) {
  std::vector<int> seed_faces;
  if (changed_segments.empty()) return seed_faces;

  const auto& vcoords = sup.mesh.vertex_coords;
  const int num_edges = static_cast<int>(sup.mesh.edge_vertices.size());
  absl::flat_hash_set<int> fset;

  auto point_on_seg = [](double px, double py, const SeedSegment& s) -> bool {
    double dx = s.x2 - s.x1;
    double dy = s.y2 - s.y1;
    double len2 = dx * dx + dy * dy;
    if (len2 <= kSeedDegenerateLenSq) {
      return std::hypot(px - s.x1, py - s.y1) <= kSeedPointOnSegTol;
    }
    double cross = (px - s.x1) * dy - (py - s.y1) * dx;
    if (std::abs(cross) > kSeedPointOnSegTol * std::sqrt(len2)) return false;
    double dot = (px - s.x1) * dx + (py - s.y1) * dy;
    return dot >= -kSeedPointOnSegTol * std::sqrt(len2) &&
           dot <= len2 + kSeedPointOnSegTol * std::sqrt(len2);
  };

  auto segs_collinear_overlap = [&](int u, int v,
                                    const SeedSegment& s) -> bool {
    auto [ux, uy] = vcoords[u];
    auto [vx, vy] = vcoords[v];
    double edx = vx - ux;
    double edy = vy - uy;
    double elen = std::hypot(edx, edy);
    double sdx = s.x2 - s.x1;
    double sdy = s.y2 - s.y1;
    double slen = std::hypot(sdx, sdy);
    if (elen <= kSeedMinEdgeLen || slen <= kSeedMinEdgeLen) return false;
    if (std::abs(edx * sdy - edy * sdx) >
        kSeedCollinearOverlapTol * elen * slen) {
      return false;
    }
    if (std::abs((s.x1 - ux) * edy - (s.y1 - uy) * edx) >
        kSeedCollinearOverlapTol * elen) {
      return false;
    }
    if (std::abs(edx) >= std::abs(edy)) {
      double a0 = std::min(ux, vx);
      double a1 = std::max(ux, vx);
      double b0 = std::min(s.x1, s.x2);
      double b1 = std::max(s.x1, s.x2);
      return std::max(a0, b0) < std::min(a1, b1) - kSeedCollinearOverlapTol;
    }
    double a0 = std::min(uy, vy);
    double a1 = std::max(uy, vy);
    double b0 = std::min(s.y1, s.y2);
    double b1 = std::max(s.y1, s.y2);
    return std::max(a0, b0) < std::min(a1, b1) - kSeedCollinearOverlapTol;
  };

  for (const auto& seg : changed_segments) {
    bool matched_edge = false;
    for (int ei = 0; ei < num_edges; ++ei) {
      auto [u, v] = sup.mesh.edge_vertices[ei];
      if (segs_collinear_overlap(u, v, seg)) {
        matched_edge = true;
        for (int fi : sup.mesh.edge_faces[ei]) fset.insert(fi);
      }
    }
    if (!matched_edge) {
      for (int ei = 0; ei < num_edges; ++ei) {
        auto [u, v] = sup.mesh.edge_vertices[ei];
        if (point_on_seg(vcoords[u].first, vcoords[u].second, seg) ||
            point_on_seg(vcoords[v].first, vcoords[v].second, seg)) {
          for (int fi : sup.mesh.edge_faces[ei]) fset.insert(fi);
        }
      }
    }
  }

  seed_faces.assign(fset.begin(), fset.end());
  absl::c_sort(seed_faces);
  return seed_faces;
}

// ============================================================================
// Causal Conflict Provenance (`PairProv`)
// ============================================================================

enum class ProvKind : uint8_t {
  kNone = 0,
  kCrease = 1,
  kUfParity = 2,
  kTacoTaco = 3,
  kTransitivity = 4,
};

struct PairProv {
  ProvKind kind = ProvKind::kNone;
  int a = -1;
  int b = -1;
};

std::vector<int> TraceConflictFaces(const SupportEntry& sup,
                                    absl::Span<const int8_t> pair_bit,
                                    absl::Span<const PairProv> prov,
                                    absl::Span<const int> seed_pairs,
                                    absl::Span<const int> seed_faces) {
  std::vector<int> faces_out(seed_faces.begin(), seed_faces.end());
  std::vector<uint8_t> seen_pair(sup.num_pairs, 0);
  std::vector<int> stack;
  stack.reserve(64);

  for (int p : seed_pairs) {
    if (p >= 0 && p < sup.num_pairs && !seen_pair[p]) {
      seen_pair[p] = 1;
      stack.push_back(p);
    }
  }

  while (!stack.empty()) {
    int p = stack.back();
    stack.pop_back();
    faces_out.push_back(sup.folded.face_pairs[p].first);
    faces_out.push_back(sup.folded.face_pairs[p].second);

    const PairProv& pv = prov[p];
    if (pv.kind == ProvKind::kUfParity) {
      int p_src = pv.a;
      auto path_faces = ExtractUfTreePathFaces(
          sup.uf_tree_adj, sup.folded.face_pairs, p_src, p, {});
      faces_out.insert(faces_out.end(), path_faces.begin(), path_faces.end());
      if (p_src >= 0 && p_src < sup.num_pairs && !seen_pair[p_src]) {
        seen_pair[p_src] = 1;
        stack.push_back(p_src);
      }
    } else if (pv.kind == ProvKind::kTacoTaco) {
      int tid = pv.a;
      if (tid >= 0 && tid < static_cast<int>(sup.taco_taco.size())) {
        const auto& ct = sup.taco_taco[tid];
        for (int f : ct.faces) faces_out.push_back(f);
        for (int slot = 0; slot < 6; ++slot) {
          if (pv.b >= 0 && ((pv.b >> slot) & 1) == 0) continue;
          int q = ct.pairs[slot];
          if (q >= 0 && q < sup.num_pairs && pair_bit[q] >= 0 &&
              !seen_pair[q]) {
            seen_pair[q] = 1;
            stack.push_back(q);
          }
        }
      }
    } else if (pv.kind == ProvKind::kTransitivity) {
      for (int q : {pv.a, pv.b}) {
        if (q >= 0 && q < sup.num_pairs && !seen_pair[q]) {
          seen_pair[q] = 1;
          stack.push_back(q);
        }
      }
    }
  }

  SortAndUnique(&faces_out);
  return faces_out;
}

TortillaConstraintKind FindTortillaPathConstraintType(const SupportEntry& sup,
                                                      int start_p, int goal_p) {
  if (start_p == goal_p || start_p < 0 || goal_p < 0) {
    return TortillaConstraintKind::kTacoTortilla;
  }
  absl::flat_hash_map<int, std::pair<int, TortillaConstraintKind>> parent;
  std::vector<int> bfs = {start_p};
  parent[start_p] = {-1, TortillaConstraintKind::kTacoTortilla};
  for (size_t qi = 0; qi < bfs.size(); ++qi) {
    int cur = bfs[qi];
    for (const auto& edge : sup.uf_tree_adj[cur]) {
      if (parent.contains(edge.neighbor_pair)) continue;
      parent[edge.neighbor_pair] = {cur, edge.constraint_type};
      if (edge.neighbor_pair == goal_p) {
        return edge.constraint_type;
      }
      bfs.push_back(edge.neighbor_pair);
    }
  }
  return TortillaConstraintKind::kTacoTortilla;
}

// ============================================================================
// Per-Cell Topological Sort (`face_constraints.sort_faces_in_cells_by_order`)
// ============================================================================

std::vector<std::vector<int>> SortFacesInCellsByOrder(
    const SupportEntry& sup, absl::Span<const int8_t> pair_bit) {
  const int num_faces = sup.num_faces;
  const int num_cells = sup.num_cells;
  std::vector<std::vector<int>> out(num_cells);

  for (int ci = 0; ci < num_cells; ++ci) {
    const auto& faces = sup.folded.cell_faces[ci];
    const int k = static_cast<int>(faces.size());
    if (k <= 1) {
      out[ci] = faces;
      continue;
    }
    std::vector<int> in_deg(k, 0);
    std::vector<std::vector<int>> adj(k);
    for (int i = 0; i < k; ++i) {
      int fi = faces[i];
      for (int j = i + 1; j < k; ++j) {
        int fj = faces[j];
        int p = sup.pair_idx_flat[static_cast<size_t>(fi) * num_faces + fj];
        if (p < 0 || pair_bit[p] < 0) continue;
        if (pair_bit[p] == 0) {
          adj[j].push_back(i);
          ++in_deg[i];
        } else {
          adj[i].push_back(j);
          ++in_deg[j];
        }
      }
    }

    std::vector<int> ready;
    ready.reserve(k);
    for (int i = 0; i < k; ++i) {
      if (in_deg[i] == 0) ready.push_back(i);
    }

    std::vector<int> sorted_cell;
    sorted_cell.reserve(k);
    while (!ready.empty()) {
      std::vector<int> next_ready;
      for (int u_local : ready) {
        sorted_cell.push_back(faces[u_local]);
        for (int v_local : adj[u_local]) {
          if (--in_deg[v_local] == 0) {
            next_ready.push_back(v_local);
          }
        }
      }
      ready = std::move(next_ready);
    }

    // Valid solutions have acyclic per-cell pairwise orders (`sorted_cell` has
    // length `k`); fallback to `faces` matches `face_constraints.py:678` if
    // called on a partial assignment.
    ABSL_DCHECK_EQ(static_cast<int>(sorted_cell.size()), k);
    if (static_cast<int>(sorted_cell.size()) == k) {
      out[ci] = std::move(sorted_cell);
    } else {
      out[ci] = faces;
    }
  }
  return out;
}

// ============================================================================
// Stage-1 + Stage-2 Unified Solver & Deduplicated Propagators
// ============================================================================

void PopulateGeometryIntoResult(const SupportEntry& sup, SolveResult* res) {
  res->vertex_coords = sup.mesh.vertex_coords;
  res->folded_vertices = sup.folded.folded_vertices;
  res->edge_vertices = sup.mesh.edge_vertices;
  res->edge_faces = sup.mesh.edge_faces;
  res->face_vertices = sup.mesh.face_vertices;
  res->face_orientations = sup.folded.face_orientations;
  res->cell_points = sup.folded.cell_points;
  res->cell_vertices = sup.folded.cell_vertices;
  res->segment_points = sup.folded.segment_points;
  res->segment_cells = sup.folded.segment_cells;
  res->segment_edges = sup.folded.segment_edges;
}

struct SolverWorkspace {
  SolverWorkspace(const SupportEntry& sup_in, const SolveOptions& opts_in,
                  int64_t t1_ns)
      : sup(sup_in),
        options(opts_in),
        num_faces(sup_in.num_faces),
        num_pairs(sup_in.num_pairs),
        num_roots(sup_in.num_roots),
        num_words(sup_in.words),
        t_stage1_start_ns(t1_ns),
        taco_table(GetTacoTacoTable()),
        root_val(num_roots, -1),
        root_src_pair(num_roots, -1),
        pair_bit(num_pairs, -1),
        prov(num_pairs),
        taco_dead(sup_in.taco_taco.size(), 0),
        use_coherent_groups(static_cast<int>(opts_in.face_group_ranks.size()) ==
                            num_faces) {}

  void InitStage2Bitsets() {
    below_bits.assign(static_cast<size_t>(num_faces) * num_words, 0ULL);
    above_bits.assign(static_cast<size_t>(num_faces) * num_words, 0ULL);
    if (!sup.has_precomputed_pair_common) {
      scratch_common.assign(num_words, 0ULL);
    }
  }

  void SetRootMembers(int r, uint8_t rval, int src_pair, PairProv src_prov) {
    ABSL_DCHECK_GE(r, 0);
    ABSL_DCHECK_LT(r, num_roots);
    root_val[r] = static_cast<int8_t>(rval);
    root_src_pair[r] = src_pair;
    for (const auto& m : sup.root_members[r]) {
      pair_bit[m.pair_idx] = static_cast<int8_t>(rval ^ m.xor_to_root);
      if (m.pair_idx == src_pair) {
        prov[m.pair_idx] = src_prov;
      } else {
        prov[m.pair_idx] =
            PairProv{.kind = ProvKind::kUfParity, .a = src_pair, .b = -1};
      }
    }
  }

  SolverConflict MakeOddCycleConflict(int r, ExitStage stage,
                                      bool seeded) const {
    const auto& oc = sup.odd_cycles[sup.root_odd_cycle_idx[r]];
    int src_p = root_src_pair[r];
    auto extra = ExtractUfTreePathFaces(sup.uf_tree_adj, sup.folded.face_pairs,
                                        src_p, oc.p0, oc.cycle_faces);
    std::array<int, 1> sp = {src_p};
    ConflictKind kind =
        (oc.constraint_type == TortillaConstraintKind::kTacoTortilla)
            ? ConflictKind::kTacoTortilla
            : ConflictKind::kTortillaTortilla;
    return SolverConflict{
        .stage = stage,
        .kind = kind,
        .conflict_faces = TraceConflictFaces(sup, pair_bit, prov, sp, extra),
        .seeded_hit = seeded,
    };
  }

  void SetPairDirectedBits(int p) {
    auto [f1, f2] = sup.folded.face_pairs[p];
    int u_bot = (pair_bit[p] == 1) ? f1 : f2;
    int v_top = (pair_bit[p] == 1) ? f2 : f1;
    below_bits[static_cast<size_t>(v_top) * num_words + (u_bot >> 6)] |=
        (1ULL << (u_bot & 63));
    above_bits[static_cast<size_t>(u_bot) * num_words + (v_top >> 6)] |=
        (1ULL << (v_top & 63));
  }

  void RecordGroupDelta(int p, int8_t bit, int delta) {
    if (!use_coherent_groups) return;
    auto [u, v] = sup.folded.face_pairs[p];
    int gu = options.face_group_ranks[u];
    int gv = options.face_group_ranks[v];
    if (gu == gv) return;
    uint64_t key =
        (bit == 0) ? EncodeDirectedPair(gu, gv) : EncodeDirectedPair(gv, gu);
    int& cnt = below_count[key];
    cnt += delta;
    if (cnt <= 0) below_count.erase(key);
  }

  const uint64_t* GetCommonFacesMask(int p) {
    if (sup.has_precomputed_pair_common) {
      return &sup.pair_common_faces[static_cast<size_t>(p) * num_words];
    }
    std::fill(scratch_common.begin(), scratch_common.end(), 0ULL);
    auto [f1, f2] = sup.folded.face_pairs[p];
    const auto& c1 = sup.folded.face_cells[f1];
    const auto& c2 = sup.folded.face_cells[f2];
    size_t i1 = 0;
    size_t i2 = 0;
    while (i1 < c1.size() && i2 < c2.size()) {
      if (c1[i1] == c2[i2]) {
        const uint64_t* cell_m =
            &sup.cell_masks[static_cast<size_t>(c1[i1]) * num_words];
        for (int wd = 0; wd < num_words; ++wd) scratch_common[wd] |= cell_m[wd];
        ++i1;
        ++i2;
      } else if (c1[i1] < c2[i2]) {
        ++i1;
      } else {
        ++i2;
      }
    }
    scratch_common[f1 >> 6] &= ~(1ULL << (f1 & 63));
    scratch_common[f2 >> 6] &= ~(1ULL << (f2 & 63));
    return scratch_common.data();
  }

  bool IsTransDead(int fa, int fb, int fc) const {
    int p0 = sup.pair_idx_flat[static_cast<size_t>(fa) * num_faces + fb];
    int p1 = sup.pair_idx_flat[static_cast<size_t>(fb) * num_faces + fc];
    int p2 = sup.pair_idx_flat[static_cast<size_t>(fc) * num_faces + fa];
    if (p0 < 0 || p1 < 0 || p2 < 0) return true;
    int d0 =
        (pair_bit[p0] < 0) ? 0 : (1 + (pair_bit[p0] ^ ((fb < fa) ? 1 : 0)));
    int d1 =
        (pair_bit[p1] < 0) ? 0 : (1 + (pair_bit[p1] ^ ((fc < fb) ? 1 : 0)));
    int d2 =
        (pair_bit[p2] < 0) ? 0 : (1 + (pair_bit[p2] ^ ((fa < fc) ? 1 : 0)));
    int zeros = (d0 == 0) + (d1 == 0) + (d2 == 0);
    if (zeros == 0) return true;
    if (zeros >= 2) return false;
    for (int c = 1; c <= 2; ++c) {
      int x0 = (d0 == 0) ? c : d0;
      int x1 = (d1 == 0) ? c : d1;
      int x2 = (d2 == 0) ? c : d2;
      if ((x0 == 1 && x1 == 1 && x2 == 1) || (x0 == 2 && x1 == 2 && x2 == 2)) {
        return false;
      }
    }
    return true;
  }

  const SupportEntry& sup;
  const SolveOptions& options;
  int num_faces = 0;
  int num_pairs = 0;
  int num_roots = 0;
  int num_words = 0;
  int64_t t_stage1_start_ns = 0;
  int64_t t_stage2_start_ns = 0;
  const std::array<TacoTacoTableEntry, 729>& taco_table;
  std::vector<int8_t> root_val;
  std::vector<int> root_src_pair;
  std::vector<int8_t> pair_bit;
  std::vector<PairProv> prov;
  std::vector<uint8_t> taco_dead;
  std::vector<uint64_t> below_bits;
  std::vector<uint64_t> above_bits;
  std::vector<uint64_t> scratch_common;
  bool use_coherent_groups = false;
  absl::flat_hash_map<uint64_t, int> below_count;
};

int ComputeTacoTableIndex(const SolverWorkspace& ws,
                          const CompiledTacoTaco& ct) {
  int idx = 0;
  for (int s = 0; s < 6; ++s) {
    int8_t rv = ws.root_val[ct.roots[s]];
    if (rv >= 0) {
      int digit = 1 + (static_cast<int>(rv) ^ ct.root_flips[s]);
      idx += digit * kPow3[s];
    }
  }
  return idx;
}

// Unified Taco-Taco clause evaluator shared across Stage 1, Stage 2 initial
// closure, and DPLL component search.
//   - `assign_root_fn(r_imp, req_rval, p_imp, antecedent_mask)` assigns an
//     unassigned root (passing the bitmask of pre-existing clause slots for
//     conflict provenance) and returns `false` on immediate odd-cycle conflict.
//   - `on_conflict_fn(seed_pairs, seed_faces)` records conflict provenance (if
//     needed) when the clause is unsatisfied or contradicts an existing root.
template <typename AssignRootFn, typename OnConflictFn>
bool PropagateTacoTacoClause(SolverWorkspace& ws, int tid, bool mark_dead,
                             AssignRootFn&& assign_root_fn,
                             OnConflictFn&& on_conflict_fn) {
  if (ws.taco_dead[tid]) return true;
  const auto& ct = ws.sup.taco_taco[tid];
  const auto& entry = ws.taco_table[ComputeTacoTableIndex(ws, ct)];
  if (entry.status == ImplicationStatus::kDead) {
    if (mark_dead) ws.taco_dead[tid] = 1;
    return true;
  }
  if (entry.status == ImplicationStatus::kAlive) return true;
  if (entry.status == ImplicationStatus::kConflict) {
    on_conflict_fn(absl::MakeConstSpan(ct.pairs),
                   absl::MakeConstSpan(ct.faces));
    return false;
  }
  // Record the exact slots already assigned before this implication fires so
  // conflict provenance never traces unrelated downstream assignments.
  int antecedent_mask = 0;
  for (int slot = 0; slot < 6; ++slot) {
    if (ws.root_val[ct.roots[slot]] >= 0) {
      antecedent_mask |= (1 << slot);
    }
  }
  for (uint8_t k = 0; k < entry.num_implied; ++k) {
    auto [slot, req_digit] = entry.implied[k];
    int r_imp = ct.roots[slot];
    int p_imp = ct.pairs[slot];
    uint8_t req_rval =
        static_cast<uint8_t>(req_digit - 1) ^ ct.root_flips[slot];
    if (ws.root_val[r_imp] < 0) {
      if (!assign_root_fn(r_imp, req_rval, p_imp, antecedent_mask)) {
        return false;
      }
    } else if (ws.root_val[r_imp] != static_cast<int8_t>(req_rval)) {
      std::array<int, 7> seed_p;
      int num_seeds = 0;
      for (int s = 0; s < 6; ++s) {
        if ((antecedent_mask >> s) & 1) seed_p[num_seeds++] = ct.pairs[s];
      }
      seed_p[num_seeds++] = p_imp;
      on_conflict_fn(absl::MakeConstSpan(seed_p.data(), num_seeds),
                     absl::MakeConstSpan(ct.faces));
      return false;
    }
  }
  return true;
}

// Unified bitset transitivity propagator for a newly assigned pair `p_uv`
// (`u_bot < v_top`), shared across Stage 2 initial closure and DPLL search.
template <typename AssignTransPairFn, typename OnConflictFn>
bool PropagateTransitivityForPair(SolverWorkspace& ws, int p_uv,
                                  AssignTransPairFn&& assign_pair_fn,
                                  OnConflictFn&& on_conflict_fn) {
  const auto& sup = ws.sup;
  const int num_faces = ws.num_faces;
  const int num_words = ws.num_words;
  auto [f1, f2] = sup.folded.face_pairs[p_uv];
  int u_bot = (ws.pair_bit[p_uv] == 1) ? f1 : f2;
  int v_top = (ws.pair_bit[p_uv] == 1) ? f2 : f1;
  const uint64_t* cm = ws.GetCommonFacesMask(p_uv);

  const uint64_t* below_u =
      &ws.below_bits[static_cast<size_t>(u_bot) * num_words];
  const uint64_t* above_u =
      &ws.above_bits[static_cast<size_t>(u_bot) * num_words];
  const uint64_t* below_v =
      &ws.below_bits[static_cast<size_t>(v_top) * num_words];
  const uint64_t* above_v =
      &ws.above_bits[static_cast<size_t>(v_top) * num_words];

  for (int wd = 0; wd < num_words; ++wd) {
    uint64_t mask = cm[wd];
    if (mask == 0ULL) continue;

    // Case A: `z < u_bot` and `u_bot < v_top` => `z < v_top`.
    uint64_t z_below_u = mask & below_u[wd];
    if (z_below_u != 0ULL) {
      uint64_t conflict_z = z_below_u & above_v[wd];
      if (conflict_z != 0ULL) {
        int z = (wd << 6) + std::countr_zero(conflict_z);
        int p_zu =
            sup.pair_idx_flat[static_cast<size_t>(z) * num_faces + u_bot];
        int p_vz =
            sup.pair_idx_flat[static_cast<size_t>(v_top) * num_faces + z];
        on_conflict_fn(std::array<int, 3>{p_zu, p_uv, p_vz},
                       std::array<int, 3>{z, u_bot, v_top});
        return false;
      }
      uint64_t implied_z = z_below_u & ~below_v[wd];
      while (implied_z != 0ULL) {
        int z = (wd << 6) + std::countr_zero(implied_z);
        implied_z &= (implied_z - 1);
        int p_zv =
            sup.pair_idx_flat[static_cast<size_t>(z) * num_faces + v_top];
        if (p_zv < 0) continue;
        uint8_t req_bit = (z < v_top) ? 1 : 0;
        int p_zu =
            sup.pair_idx_flat[static_cast<size_t>(z) * num_faces + u_bot];
        if (ws.pair_bit[p_zv] >= 0) {
          if (ws.pair_bit[p_zv] != static_cast<int8_t>(req_bit)) {
            on_conflict_fn(std::array<int, 3>{p_zu, p_uv, p_zv},
                           std::array<int, 3>{z, u_bot, v_top});
            return false;
          }
          continue;
        }
        if (!assign_pair_fn(p_zv, req_bit, p_zu, p_uv)) {
          return false;
        }
      }
    }

    // Case B: `u_bot < v_top` and `v_top < z` => `u_bot < z`.
    uint64_t z_above_v = mask & above_v[wd];
    if (z_above_v != 0ULL) {
      uint64_t conflict_z = z_above_v & below_u[wd];
      if (conflict_z != 0ULL) {
        int z = (wd << 6) + std::countr_zero(conflict_z);
        int p_vz =
            sup.pair_idx_flat[static_cast<size_t>(v_top) * num_faces + z];
        int p_zu =
            sup.pair_idx_flat[static_cast<size_t>(z) * num_faces + u_bot];
        on_conflict_fn(std::array<int, 3>{p_uv, p_vz, p_zu},
                       std::array<int, 3>{u_bot, v_top, z});
        return false;
      }
      uint64_t implied_z = z_above_v & ~above_u[wd];
      while (implied_z != 0ULL) {
        int z = (wd << 6) + std::countr_zero(implied_z);
        implied_z &= (implied_z - 1);
        int p_uz =
            sup.pair_idx_flat[static_cast<size_t>(u_bot) * num_faces + z];
        if (p_uz < 0) continue;
        uint8_t req_bit = (u_bot < z) ? 1 : 0;
        int p_vz =
            sup.pair_idx_flat[static_cast<size_t>(v_top) * num_faces + z];
        if (ws.pair_bit[p_uz] >= 0) {
          if (ws.pair_bit[p_uz] != static_cast<int8_t>(req_bit)) {
            on_conflict_fn(std::array<int, 3>{p_uv, p_vz, p_uz},
                           std::array<int, 3>{u_bot, v_top, z});
            return false;
          }
          continue;
        }
        if (!assign_pair_fn(p_uz, req_bit, p_uv, p_vz)) {
          return false;
        }
      }
    }
  }
  return true;
}

std::optional<SolverConflict> EvalTacoStage1(SolverWorkspace& ws, int tid,
                                             bool seeded_wave,
                                             std::vector<int>* next_roots) {
  std::optional<SolverConflict> conflict;
  bool ok = PropagateTacoTacoClause(
      ws, tid, /*mark_dead=*/true,
      [&](int r_imp, uint8_t req_rval, int p_imp, int antecedent_mask) -> bool {
        ws.SetRootMembers(
            r_imp, req_rval, p_imp,
            PairProv{
                .kind = ProvKind::kTacoTaco, .a = tid, .b = antecedent_mask});
        if (ws.sup.root_odd_cycle_idx[r_imp] >= 0) {
          conflict = ws.MakeOddCycleConflict(
              r_imp,
              seeded_wave ? ExitStage::kStage1TortillaSeeded
                          : ExitStage::kStage1Tortilla,
              seeded_wave);
          return false;
        }
        next_roots->push_back(r_imp);
        return true;
      },
      [&](absl::Span<const int> seed_p, absl::Span<const int> seed_f) {
        conflict = SolverConflict{
            .stage = seeded_wave ? ExitStage::kStage1TacoTacoSeeded
                                 : ExitStage::kStage1TacoTaco,
            .kind = ConflictKind::kTacoTaco,
            .conflict_faces = TraceConflictFaces(ws.sup, ws.pair_bit, ws.prov,
                                                 seed_p, seed_f),
            .seeded_hit = seeded_wave,
        };
      });
  if (!ok) return conflict;
  return std::nullopt;
}

std::optional<SolverConflict> RunStage1CreasesAndTaco(
    SolverWorkspace& ws, absl::Span<const CreaseAssignment> edge_assignments) {
  const auto& sup = ws.sup;
  for (size_t ei = 0; ei < sup.mesh.edge_faces.size(); ++ei) {
    CreaseAssignment as_code = edge_assignments[ei];
    if (as_code != CreaseAssignment::kMountain &&
        as_code != CreaseAssignment::kValley) {
      continue;
    }
    if (sup.mesh.edge_faces[ei].size() != 2) continue;
    int fa = sup.mesh.edge_faces[ei][0];
    int fb = sup.mesh.edge_faces[ei][1];
    if (fb < fa) std::swap(fa, fb);
    int p = sup.pair_idx_flat[static_cast<size_t>(fa) * ws.num_faces + fb];
    if (p < 0) continue;

    bool fa_flipped = sup.folded.face_orientations[fa];
    uint8_t req_bit =
        ((!fa_flipped && as_code == CreaseAssignment::kMountain) ||
         (fa_flipped && as_code == CreaseAssignment::kValley))
            ? 1
            : 0;
    int r = sup.pair_root_idx[p];
    uint8_t req_rval = req_bit ^ sup.pair_xor_to_root[p];

    if (ws.root_val[r] < 0) {
      ws.SetRootMembers(r, req_rval, p,
                        PairProv{.kind = ProvKind::kCrease, .a = -1, .b = -1});
      if (sup.root_odd_cycle_idx[r] >= 0) {
        return ws.MakeOddCycleConflict(r, ExitStage::kStage1TortillaOddCycle,
                                       /*seeded=*/false);
      }
    } else if (ws.root_val[r] != static_cast<int8_t>(req_rval)) {
      int prev_p = ws.root_src_pair[r];
      TortillaConstraintKind ctype =
          FindTortillaPathConstraintType(sup, prev_p, p);
      auto path_faces = ExtractUfTreePathFaces(
          sup.uf_tree_adj, sup.folded.face_pairs, prev_p, p, {});
      std::array<int, 1> sp = {prev_p};
      return SolverConflict{
          .stage = ExitStage::kStage1Tortilla,
          .kind = (ctype == TortillaConstraintKind::kTacoTortilla)
                      ? ConflictKind::kTacoTortilla
                      : ConflictKind::kTortillaTortilla,
          .conflict_faces =
              TraceConflictFaces(sup, ws.pair_bit, ws.prov, sp, path_faces),
      };
    }
  }

  // Expand Stage-1 Taco-Taco unit propagation level by level so direct shallow
  // contradictions are reported before deeper implication chains.
  auto run_taco_bfs = [&](std::vector<int>& frontier,
                          bool seeded_wave) -> std::optional<SolverConflict> {
    std::vector<int> next_frontier;
    std::vector<int> pending_implied;
    while (!frontier.empty()) {
      next_frontier.clear();
      pending_implied.clear();
      for (int r : frontier) {
        for (int tid : sup.taco_by_root[r]) {
          if (ws.taco_dead[tid]) continue;
          const auto& ct = sup.taco_taco[tid];
          const auto& entry = ws.taco_table[ComputeTacoTableIndex(ws, ct)];
          if (entry.status == ImplicationStatus::kDead) {
            ws.taco_dead[tid] = 1;
          } else if (entry.status == ImplicationStatus::kConflict) {
            return SolverConflict{
                .stage = seeded_wave ? ExitStage::kStage1TacoTacoSeeded
                                     : ExitStage::kStage1TacoTaco,
                .kind = ConflictKind::kTacoTaco,
                .conflict_faces = TraceConflictFaces(
                    ws.sup, ws.pair_bit, ws.prov, absl::MakeConstSpan(ct.pairs),
                    absl::MakeConstSpan(ct.faces)),
                .seeded_hit = seeded_wave,
            };
          } else if (entry.status != ImplicationStatus::kAlive) {
            pending_implied.push_back(tid);
          }
        }
      }
      for (int tid : pending_implied) {
        if (std::optional<SolverConflict> conflict =
                EvalTacoStage1(ws, tid, seeded_wave, &next_frontier);
            conflict.has_value()) {
          return conflict;
        }
      }
      frontier.swap(next_frontier);
    }
    return std::nullopt;
  };

  // Stage-1 Taco-Taco unit propagation (Wave-1 seeded -> Wave-2 full).
  std::vector<uint8_t> root_queued_wave1(ws.num_roots, 0);
  if (!ws.options.changed_segments.empty()) {
    std::vector<int> seed_faces =
        FindSeedFaces(sup, ws.options.changed_segments);
    if (!seed_faces.empty()) {
      std::vector<uint8_t> is_seed_face(ws.num_faces, 0);
      for (int fi : seed_faces) is_seed_face[fi] = 1;
      std::vector<int> wave1_q;
      for (int r = 0; r < ws.num_roots; ++r) {
        if (ws.root_val[r] < 0) continue;
        bool touches_seed = false;
        for (const auto& m : sup.root_members[r]) {
          if (is_seed_face[m.u_face] || is_seed_face[m.v_face]) {
            touches_seed = true;
            break;
          }
        }
        if (touches_seed) {
          root_queued_wave1[r] = 1;
          wave1_q.push_back(r);
        }
      }
      if (std::optional<SolverConflict> conflict =
              run_taco_bfs(wave1_q, /*seeded_wave=*/true);
          conflict.has_value()) {
        return conflict;
      }
    }
  }

  std::vector<int> wave2_q;
  wave2_q.reserve(ws.num_roots);
  for (int r = 0; r < ws.num_roots; ++r) {
    if (ws.root_val[r] >= 0 && !root_queued_wave1[r]) {
      wave2_q.push_back(r);
    }
  }
  return run_taco_bfs(wave2_q, /*seeded_wave=*/false);
}

std::optional<SolverConflict> AssignRootStage2(SolverWorkspace& ws, int r,
                                               uint8_t req_rval, int src_p,
                                               PairProv src_prov,
                                               std::vector<int>* trans_q,
                                               std::vector<int>* taco_q) {
  ws.SetRootMembers(r, req_rval, src_p, src_prov);
  for (const auto& m : ws.sup.root_members[r]) {
    ws.SetPairDirectedBits(m.pair_idx);
    trans_q->push_back(m.pair_idx);
  }
  if (ws.sup.root_odd_cycle_idx[r] >= 0) {
    return ws.MakeOddCycleConflict(r, ExitStage::kStage2Tortilla,
                                   /*seeded=*/false);
  }
  taco_q->push_back(r);
  return std::nullopt;
}

std::optional<SolverConflict> RunStage2InitialClosure(SolverWorkspace& ws) {
  std::vector<int> trans_q;
  trans_q.reserve(ws.num_pairs);
  for (int p = 0; p < ws.num_pairs; ++p) {
    if (ws.pair_bit[p] >= 0) {
      ws.SetPairDirectedBits(p);
      trans_q.push_back(p);
    }
  }

  std::vector<int> stage2_taco_q;
  size_t trans_head = 0;
  size_t taco_head = 0;

  while (trans_head < trans_q.size() || taco_head < stage2_taco_q.size()) {
    while (trans_head < trans_q.size()) {
      int p_uv = trans_q[trans_head++];
      std::optional<SolverConflict> conflict;
      bool ok = PropagateTransitivityForPair(
          ws, p_uv,
          [&](int p_imp, uint8_t req_bit, int prov_a, int prov_b) -> bool {
            int r_imp = ws.sup.pair_root_idx[p_imp];
            uint8_t req_rval = req_bit ^ ws.sup.pair_xor_to_root[p_imp];
            conflict = AssignRootStage2(
                ws, r_imp, req_rval, p_imp,
                PairProv{
                    .kind = ProvKind::kTransitivity, .a = prov_a, .b = prov_b},
                &trans_q, &stage2_taco_q);
            return !conflict.has_value();
          },
          [&](std::array<int, 3> sp, std::array<int, 3> sf) {
            conflict = SolverConflict{
                .stage = ExitStage::kStage2Transitivity,
                .kind = ConflictKind::kTransitivity,
                .conflict_faces =
                    TraceConflictFaces(ws.sup, ws.pair_bit, ws.prov, sp, sf),
            };
          });
      if (!ok) return conflict;
    }
    while (taco_head < stage2_taco_q.size()) {
      int r = stage2_taco_q[taco_head++];
      for (int tid : ws.sup.taco_by_root[r]) {
        std::optional<SolverConflict> conflict;
        bool ok = PropagateTacoTacoClause(
            ws, tid, /*mark_dead=*/true,
            [&](int r_imp, uint8_t req_rval, int p_imp,
                int antecedent_mask) -> bool {
              conflict = AssignRootStage2(ws, r_imp, req_rval, p_imp,
                                          PairProv{.kind = ProvKind::kTacoTaco,
                                                   .a = tid,
                                                   .b = antecedent_mask},
                                          &trans_q, &stage2_taco_q);
              return !conflict.has_value();
            },
            [&](absl::Span<const int> seed_p, absl::Span<const int> seed_f) {
              conflict = SolverConflict{
                  .stage = ExitStage::kStage2TacoTaco,
                  .kind = ConflictKind::kTacoTaco,
                  .conflict_faces = TraceConflictFaces(ws.sup, ws.pair_bit,
                                                       ws.prov, seed_p, seed_f),
              };
            });
        if (!ok) return conflict;
      }
    }
  }
  return std::nullopt;
}

std::vector<std::vector<int>> DiscoverUnassignedComponents(
    SolverWorkspace& ws) {
  const auto& sup = ws.sup;
  std::vector<uint8_t> seen_unassigned(ws.num_pairs, 0);
  std::vector<std::vector<int>> components;

  for (int start_p = 0; start_p < ws.num_pairs; ++start_p) {
    if (ws.pair_bit[start_p] >= 0 || seen_unassigned[start_p]) continue;
    std::vector<int> stack = {start_p};
    seen_unassigned[start_p] = 1;
    for (size_t si = 0; si < stack.size(); ++si) {
      int cur_p = stack[si];
      for (int tid : sup.taco_by_pair[cur_p]) {
        if (ws.taco_dead[tid]) continue;
        const auto& ct = sup.taco_taco[tid];
        if (ws.taco_table[ComputeTacoTableIndex(ws, ct)].status ==
            ImplicationStatus::kDead) {
          ws.taco_dead[tid] = 1;
          continue;
        }
        for (int q : ct.pairs) {
          if (ws.pair_bit[q] < 0 && !seen_unassigned[q]) {
            seen_unassigned[q] = 1;
            stack.push_back(q);
          }
        }
      }
      for (int nb : sup.type1_neighbors[cur_p]) {
        if (ws.pair_bit[nb] < 0 && !seen_unassigned[nb]) {
          seen_unassigned[nb] = 1;
          stack.push_back(nb);
        }
      }
      for (int nb : sup.type2_neighbors[cur_p]) {
        if (ws.pair_bit[nb] < 0 && !seen_unassigned[nb]) {
          seen_unassigned[nb] = 1;
          stack.push_back(nb);
        }
      }
      auto [f1, f2] = sup.folded.face_pairs[cur_p];
      const uint64_t* cm = ws.GetCommonFacesMask(cur_p);
      for (int wd = 0; wd < ws.num_words; ++wd) {
        uint64_t mask = cm[wd];
        while (mask != 0ULL) {
          int f3 = (wd << 6) + std::countr_zero(mask);
          mask &= (mask - 1);
          if (sup.IsTransPrunedByFaceComponents(f1, f2, f3)) continue;
          if (ws.IsTransDead(f1, f2, f3)) continue;
          for (int q :
               {sup.pair_idx_flat[static_cast<size_t>(f2) * ws.num_faces + f3],
                sup.pair_idx_flat[static_cast<size_t>(f1) * ws.num_faces +
                                  f3]}) {
            if (q >= 0 && ws.pair_bit[q] < 0 && !seen_unassigned[q]) {
              seen_unassigned[q] = 1;
              stack.push_back(q);
            }
          }
        }
      }
    }
    components.push_back(std::move(stack));
  }

  absl::c_stable_sort(components, [](const auto& a, const auto& b) {
    return a.size() < b.size();
  });
  return components;
}

ComponentKey BuildComponentKey(
    SolverWorkspace& ws, absl::Span<const int> comp_pairs,
    absl::Span<const int> comp_roots,
    absl::Span<const uint8_t> comp_root_base_flip,
    const absl::flat_hash_map<int, int>& global_to_local_root, int sol_limit) {
  const auto& sup = ws.sup;
  ComponentKey ckey;
  ckey.num_local_roots = static_cast<int>(comp_roots.size());
  ckey.solution_limit = sol_limit;

  absl::flat_hash_set<int> comp_tacos;
  for (int r : comp_roots) {
    for (int tid : sup.taco_by_root[r]) {
      if (!ws.taco_dead[tid]) comp_tacos.insert(tid);
    }
  }
  std::vector<int> sorted_tacos(comp_tacos.begin(), comp_tacos.end());
  absl::c_sort(sorted_tacos);
  for (int tid : sorted_tacos) {
    const auto& ct = sup.taco_taco[tid];
    std::array<int32_t, 6> clause{};
    for (int s = 0; s < 6; ++s) {
      int r = ct.roots[s];
      if (ws.root_val[r] >= 0) {
        clause[s] = 1 + (static_cast<int>(ws.root_val[r]) ^ ct.root_flips[s]);
      } else {
        auto it = global_to_local_root.find(r);
        ABSL_DCHECK(it != global_to_local_root.end());
        int lr = it->second;
        uint8_t rel = ct.root_flips[s] ^ comp_root_base_flip[lr];
        clause[s] = 3 + 2 * lr + rel;
      }
    }
    ckey.taco_clauses.push_back(clause);
  }

  absl::flat_hash_set<uint64_t> seen_triples;
  auto encode_slot = [&](int fa, int fb) -> int32_t {
    int p = sup.pair_idx_flat[static_cast<size_t>(fa) * ws.num_faces + fb];
    uint8_t flp = (fb < fa) ? 1 : 0;
    if (ws.pair_bit[p] >= 0) {
      return 1 + (static_cast<int>(ws.pair_bit[p]) ^ flp);
    }
    int r = sup.pair_root_idx[p];
    auto it = global_to_local_root.find(r);
    ABSL_DCHECK(it != global_to_local_root.end());
    int lr = it->second;
    uint8_t rel = sup.pair_xor_to_root[p] ^ flp ^ comp_root_base_flip[lr];
    return 3 + 2 * lr + rel;
  };

  for (int p : comp_pairs) {
    auto [f1, f2] = sup.folded.face_pairs[p];
    const uint64_t* cm = ws.GetCommonFacesMask(p);
    for (int wd = 0; wd < ws.num_words; ++wd) {
      uint64_t mask = cm[wd];
      while (mask != 0ULL) {
        int f3 = (wd << 6) + std::countr_zero(mask);
        mask &= (mask - 1);
        if (sup.IsTransPrunedByFaceComponents(f1, f2, f3)) continue;
        std::array<int, 3> tri = {f1, f2, f3};
        absl::c_sort(tri);
        ABSL_DCHECK_LT(tri[2], kMaxTriplePackedFaces);
        uint64_t tkey = (static_cast<uint64_t>(tri[0]) << 40) |
                        (static_cast<uint64_t>(tri[1]) << 20) |
                        static_cast<uint64_t>(tri[2]);
        if (!seen_triples.insert(tkey).second) continue;
        if (ws.IsTransDead(tri[0], tri[1], tri[2])) continue;
        ckey.trans_clauses.push_back({encode_slot(tri[0], tri[1]),
                                      encode_slot(tri[1], tri[2]),
                                      encode_slot(tri[2], tri[0])});
      }
    }
  }
  absl::c_sort(ckey.taco_clauses);
  absl::c_sort(ckey.trans_clauses);
  return ckey;
}

void UndoDpllRoots(SolverWorkspace& ws, const std::vector<int>& roots_to_undo) {
  for (auto it = roots_to_undo.rbegin(); it != roots_to_undo.rend(); ++it) {
    int r = *it;
    ws.root_val[r] = -1;
    for (const auto& m : ws.sup.root_members[r]) {
      int p = m.pair_idx;
      int8_t old_bit = ws.pair_bit[p];
      if (old_bit >= 0) {
        ws.RecordGroupDelta(p, old_bit, -1);
        auto [f1, f2] = ws.sup.folded.face_pairs[p];
        int u_bot = (old_bit == 1) ? f1 : f2;
        int v_top = (old_bit == 1) ? f2 : f1;
        ws.below_bits[static_cast<size_t>(v_top) * ws.num_words +
                      (u_bot >> 6)] &= ~(1ULL << (u_bot & 63));
        ws.above_bits[static_cast<size_t>(u_bot) * ws.num_words +
                      (v_top >> 6)] &= ~(1ULL << (v_top & 63));
        ws.pair_bit[p] = -1;
      }
    }
  }
}

bool PropagateDpllGuess(SolverWorkspace& ws, int guess_p, uint8_t guess_bit,
                        std::vector<int>* trail_roots) {
  const auto& sup = ws.sup;
  trail_roots->clear();
  std::vector<int> l_trans_q;
  std::vector<int> l_taco_q;

  auto push_root = [&](int r, uint8_t rval) -> bool {
    if (sup.root_odd_cycle_idx[r] >= 0) return false;
    ws.root_val[r] = static_cast<int8_t>(rval);
    trail_roots->push_back(r);
    for (const auto& m : sup.root_members[r]) {
      int p = m.pair_idx;
      int8_t pb = static_cast<int8_t>(rval ^ m.xor_to_root);
      ws.pair_bit[p] = pb;
      ws.SetPairDirectedBits(p);
      ws.RecordGroupDelta(p, pb, +1);
      l_trans_q.push_back(p);
    }
    l_taco_q.push_back(r);
    return true;
  };

  int gr = sup.pair_root_idx[guess_p];
  uint8_t gr_val = guess_bit ^ sup.pair_xor_to_root[guess_p];
  if (!push_root(gr, gr_val)) {
    UndoDpllRoots(ws, *trail_roots);
    return false;
  }

  size_t lt_head = 0;
  size_t ltc_head = 0;
  while (lt_head < l_trans_q.size() || ltc_head < l_taco_q.size()) {
    while (lt_head < l_trans_q.size()) {
      int p_uv = l_trans_q[lt_head++];
      bool ok = PropagateTransitivityForPair(
          ws, p_uv,
          [&](int p_imp, uint8_t req_bit, int /*prov_a*/,
              int /*prov_b*/) -> bool {
            int r_imp = sup.pair_root_idx[p_imp];
            return push_root(r_imp, req_bit ^ sup.pair_xor_to_root[p_imp]);
          },
          [](std::array<int, 3> /*sp*/, std::array<int, 3> /*sf*/) {});
      if (!ok) {
        UndoDpllRoots(ws, *trail_roots);
        return false;
      }
    }

    while (ltc_head < l_taco_q.size()) {
      int r = l_taco_q[ltc_head++];
      for (int tid : sup.taco_by_root[r]) {
        bool ok = PropagateTacoTacoClause(
            ws, tid, /*mark_dead=*/false,
            [&](int r_imp, uint8_t req_rval, int /*p_imp*/,
                int /*antecedent_mask*/) -> bool {
              return push_root(r_imp, req_rval);
            },
            [](absl::Span<const int> /*seed_p*/,
               absl::Span<const int> /*seed_f*/) {});
        if (!ok) {
          UndoDpllRoots(ws, *trail_roots);
          return false;
        }
      }
    }
  }
  return true;
}

std::pair<int, uint8_t> PickNextDpllVar(const SolverWorkspace& ws,
                                        absl::Span<const int> comp_pairs,
                                        bool has_inter_group_pair) {
  if (!ws.use_coherent_groups || !has_inter_group_pair) {
    for (int p : comp_pairs) {
      if (ws.pair_bit[p] < 0) return {p, 0};
    }
    return {-1, 0};
  }
  int best_p = -1;
  uint8_t best_bit = 0;
  std::pair<int, int> best_score{-1, -1};
  for (int p : comp_pairs) {
    if (ws.pair_bit[p] >= 0) continue;
    auto [f1, f2] = ws.sup.folded.face_pairs[p];
    int g1 = ws.options.face_group_ranks[f1];
    int g2 = ws.options.face_group_ranks[f2];
    std::pair<int, int> score{0, 0};
    uint8_t pref_bit = 0;
    if (g1 != g2) {
      auto it12 = ws.below_count.find(EncodeDirectedPair(g1, g2));
      auto it21 = ws.below_count.find(EncodeDirectedPair(g2, g1));
      int c12 = (it12 != ws.below_count.end()) ? it12->second : 0;
      int c21 = (it21 != ws.below_count.end()) ? it21->second : 0;
      if ((c12 > 0) != (c21 > 0)) {
        score = {3, std::max(c12, c21)};
        pref_bit = (c12 > c21) ? 0 : 1;
      } else if (c12 == 0 && c21 == 0) {
        score = {2, 0};
        pref_bit = (g1 <= g2) ? 0 : 1;
      } else {
        score = {1, std::abs(c12 - c21)};
        pref_bit = (c12 >= c21) ? 0 : 1;
      }
    }
    if (score > best_score) {
      best_score = score;
      best_p = p;
      best_bit = pref_bit;
    }
  }
  return {best_p, best_bit};
}

ComponentCacheValue SolveSingleComponentDpll(
    SolverWorkspace& ws, absl::Span<const int> comp_pairs,
    absl::Span<const int> comp_roots,
    absl::Span<const uint8_t> comp_root_base_flip, bool has_inter_group_pair,
    int sol_limit) {
  struct DfsFrame {
    int guess_p = -1;
    uint8_t first_bit = 0;
    bool tried_second = false;
    std::vector<int> trail_roots{};
  };
  const int num_lr = static_cast<int>(comp_roots.size());
  ComponentCacheValue result;
  std::vector<DfsFrame> dfs_stack;
  bool need_advance = true;

  while (true) {
    if (need_advance) {
      auto [next_p, pref_bit] =
          PickNextDpllVar(ws, comp_pairs, has_inter_group_pair);
      if (next_p < 0) {
        if (result.num_solutions == 0) {
          result.first_solution.assign(num_lr, 0);
          for (int lr = 0; lr < num_lr; ++lr) {
            result.first_solution[lr] =
                static_cast<uint8_t>(ws.root_val[comp_roots[lr]]) ^
                comp_root_base_flip[lr];
          }
        }
        if (result.num_solutions < std::numeric_limits<int64_t>::max()) {
          ++result.num_solutions;
        }
        if (sol_limit > 0 && result.num_solutions >= sol_limit) {
          while (!dfs_stack.empty()) {
            UndoDpllRoots(ws, dfs_stack.back().trail_roots);
            dfs_stack.pop_back();
          }
          break;
        }
        need_advance = false;
        continue;
      }
      DfsFrame frame{
          .guess_p = next_p, .first_bit = pref_bit, .tried_second = false};
      if (PropagateDpllGuess(ws, next_p, pref_bit, &frame.trail_roots)) {
        dfs_stack.push_back(std::move(frame));
        need_advance = true;
      } else {
        frame.tried_second = true;
        if (PropagateDpllGuess(ws, next_p, pref_bit ^ 1, &frame.trail_roots)) {
          dfs_stack.push_back(std::move(frame));
          need_advance = true;
        } else {
          need_advance = false;
        }
      }
    } else {
      if (dfs_stack.empty()) break;
      auto& top = dfs_stack.back();
      UndoDpllRoots(ws, top.trail_roots);
      top.trail_roots.clear();
      if (!top.tried_second) {
        top.tried_second = true;
        if (PropagateDpllGuess(ws, top.guess_p, top.first_bit ^ 1,
                               &top.trail_roots)) {
          need_advance = true;
          continue;
        }
      }
      dfs_stack.pop_back();
    }
  }
  return result;
}

void ApplyFirstComponentSolution(SolverWorkspace& ws,
                                 absl::Span<const int> comp_roots,
                                 absl::Span<const uint8_t> comp_root_base_flip,
                                 absl::Span<const uint8_t> first_solution) {
  for (size_t lr = 0; lr < comp_roots.size(); ++lr) {
    int r = comp_roots[lr];
    uint8_t rval = first_solution[lr] ^ comp_root_base_flip[lr];
    ws.root_val[r] = static_cast<int8_t>(rval);
    for (const auto& m : ws.sup.root_members[r]) {
      ws.pair_bit[m.pair_idx] = static_cast<int8_t>(rval ^ m.xor_to_root);
      ws.SetPairDirectedBits(m.pair_idx);
      ws.RecordGroupDelta(m.pair_idx, ws.pair_bit[m.pair_idx], +1);
    }
  }
}

int64_t SaturatingMultiply(int64_t a, int64_t b) {
  if (a <= 0 || b <= 0) return 0;
  if (a > std::numeric_limits<int64_t>::max() / b) {
    return std::numeric_limits<int64_t>::max();
  }
  return a * b;
}

std::optional<SolverConflict> SolveAllComponents(
    SolverWorkspace& ws, const std::vector<std::vector<int>>& components,
    LruCache<ComponentKey, ComponentCacheValue>* comp_cache, SolveResult* res,
    int64_t* total_states) {
  if (ws.use_coherent_groups) {
    for (int p = 0; p < ws.num_pairs; ++p) {
      if (ws.pair_bit[p] >= 0) ws.RecordGroupDelta(p, ws.pair_bit[p], +1);
    }
  }
  const int sol_limit =
      (ws.options.solution_limit > 0) ? ws.options.solution_limit : 0;

  for (int comp_idx = 0; comp_idx < static_cast<int>(components.size());
       ++comp_idx) {
    const auto& comp_pairs = components[comp_idx];
    bool has_odd_cycle = false;
    std::vector<int> comp_roots;
    std::vector<uint8_t> comp_root_base_flip;
    absl::flat_hash_map<int, int> global_to_local_root;
    bool has_inter_group_pair = false;

    for (int p : comp_pairs) {
      int r = ws.sup.pair_root_idx[p];
      if (ws.sup.root_odd_cycle_idx[r] >= 0) {
        has_odd_cycle = true;
        break;
      }
      if (ws.use_coherent_groups) {
        auto [u, v] = ws.sup.folded.face_pairs[p];
        if (ws.options.face_group_ranks[u] != ws.options.face_group_ranks[v]) {
          has_inter_group_pair = true;
        }
      }
      if (!global_to_local_root.contains(r)) {
        global_to_local_root[r] = static_cast<int>(comp_roots.size());
        comp_roots.push_back(r);
        comp_root_base_flip.push_back(ws.sup.pair_xor_to_root[p]);
      }
    }

    auto make_comp_conflict = [&]() {
      return SolverConflict{
          .stage = ExitStage::kStage2ComponentUnsat,
          .kind = ConflictKind::kComponentUnsat,
          .component_index = comp_idx + 1,
      };
    };

    if (has_odd_cycle) {
      return make_comp_conflict();
    }

    const int num_lr = static_cast<int>(comp_roots.size());
    const bool can_use_comp_cache =
        ws.options.use_component_cache && comp_cache != nullptr &&
        (!ws.use_coherent_groups || !has_inter_group_pair) &&
        num_lr <= kMaxCachedComponentRoots;

    ComponentKey ckey;
    if (can_use_comp_cache) {
      ckey = BuildComponentKey(ws, comp_pairs, comp_roots, comp_root_base_flip,
                               global_to_local_root, sol_limit);
      if (auto cached = comp_cache->Lookup(ckey)) {
        ++res->component_cache_hits;
        if (cached->num_solutions == 0) {
          return make_comp_conflict();
        }
        *total_states =
            SaturatingMultiply(*total_states, cached->num_solutions);
        ApplyFirstComponentSolution(ws, comp_roots, comp_root_base_flip,
                                    cached->first_solution);
        continue;
      }
      ++res->component_cache_misses;
    }

    ComponentCacheValue found = SolveSingleComponentDpll(
        ws, comp_pairs, comp_roots, comp_root_base_flip, has_inter_group_pair,
        sol_limit);

    if (can_use_comp_cache) {
      comp_cache->Insert(std::move(ckey),
                         std::make_shared<ComponentCacheValue>(found));
    }

    if (found.num_solutions == 0) {
      return make_comp_conflict();
    }

    *total_states = SaturatingMultiply(*total_states, found.num_solutions);
    ApplyFirstComponentSolution(ws, comp_roots, comp_root_base_flip,
                                found.first_solution);
  }
  return std::nullopt;
}

void PopulateLayerOrders(const SupportEntry& sup,
                         absl::Span<const int8_t> pair_bit, SolveResult* res) {
  const int64_t t_order_start = NowNanos();
  PopulateGeometryIntoResult(sup, res);
  res->ordered_face_pairs.clear();
  res->ordered_face_pairs.reserve(sup.num_pairs);
  for (int p = 0; p < sup.num_pairs; ++p) {
    ABSL_DCHECK_GE(pair_bit[p], 0);
    auto [f1, f2] = sup.folded.face_pairs[p];
    res->ordered_face_pairs.push_back(
        (pair_bit[p] == 0) ? std::make_pair(f1, f2) : std::make_pair(f2, f1));
  }
  absl::StatusOr<std::vector<int>> orders =
      ComputeAbsoluteFaceOrders(res->ordered_face_pairs, sup.num_faces);
  ABSL_DCHECK_OK(orders);
  if (orders.ok()) {
    res->face_orders = *std::move(orders);
  }
  res->cell_faces_ordered = SortFacesInCellsByOrder(sup, pair_bit);
  res->order_seconds = ElapsedSeconds(t_order_start, NowNanos());
}

enum class LocalCheckPolicy : uint8_t {
  kRun,
  kAlreadyPassed,
};

void PopulateLocalViolationFields(const internal::UnfoldedMesh& mesh,
                                  std::vector<std::string> local_violations,
                                  int64_t t_local_start, SolveResult* res) {
  res->used_exact_octilinear = mesh.used_exact_octilinear;
  res->num_faces = static_cast<int>(mesh.face_vertices.size());
  res->local_violations = std::move(local_violations);
  PopulateConflictIntoSolveResult(
      SolverConflict{.stage = ExitStage::kLocalViolation,
                     .kind = ConflictKind::kLocalViolation},
      res);
  res->stage1_seconds = ElapsedSeconds(t_local_start, NowNanos());
}

void SolveOnSupport(const SupportEntry& sup,
                    absl::Span<const CreaseAssignment> edge_assignments,
                    const SolveOptions& options,
                    LruCache<ComponentKey, ComponentCacheValue>* comp_cache,
                    LocalCheckPolicy local_check_policy, SolveResult* res) {
  const int64_t t_stage1_start = NowNanos();
  if (local_check_policy == LocalCheckPolicy::kRun) {
    std::vector<std::string> local_violations =
        internal::CheckLocalVertexFoldability(sup.mesh, edge_assignments,
                                              options.kawasaki_epsilon);
    if (!local_violations.empty()) {
      PopulateLocalViolationFields(sup.mesh, std::move(local_violations),
                                   t_stage1_start, res);
      return;
    }
  }

  res->used_exact_octilinear = sup.mesh.used_exact_octilinear;
  res->used_exact_folded_octilinear = sup.folded.used_exact_folded_octilinear;
  res->num_faces = sup.num_faces;
  res->num_cells = sup.num_cells;
  res->num_face_pairs = sup.num_pairs;

  if (sup.folded_outcome != internal::FoldedArrangementOutcome::kOk) {
    if (sup.folded_outcome ==
        internal::FoldedArrangementOutcome::kNotFoldableProof) {
      PopulateConflictIntoSolveResult(
          SolverConflict{.stage = ExitStage::kFoldedGeometryError,
                         .kind = ConflictKind::kFoldedGeometryError,
                         .custom_reason = sup.error_message},
          res);
    } else {
      PopulateConflictIntoSolveResult(
          SolverConflict{.stage = ExitStage::kCouldNotDecide,
                         .kind = ConflictKind::kCouldNotDecide,
                         .custom_reason = sup.error_message},
          res);
    }
    res->stage1_seconds = ElapsedSeconds(t_stage1_start, NowNanos());
    return;
  }

  SolverWorkspace ws(sup, options, t_stage1_start);
  if (std::optional<SolverConflict> stage1_conflict =
          RunStage1CreasesAndTaco(ws, edge_assignments);
      stage1_conflict.has_value()) {
    PopulateConflictIntoSolveResult(*stage1_conflict, res);
    res->stage1_seconds = ElapsedSeconds(t_stage1_start, NowNanos());
    return;
  }

  res->passed_taco_stage = true;
  res->stage1_seconds = ElapsedSeconds(t_stage1_start, NowNanos());

  if (options.taco_only) {
    res->flat_foldable = std::nullopt;
    res->states = 0;
    res->stage = ExitStage::kStage1Pass;
    return;
  }

  ws.t_stage2_start_ns = NowNanos();
  ws.InitStage2Bitsets();
  if (std::optional<SolverConflict> stage2_conflict =
          RunStage2InitialClosure(ws);
      stage2_conflict.has_value()) {
    PopulateConflictIntoSolveResult(*stage2_conflict, res);
    res->stage2_seconds = ElapsedSeconds(ws.t_stage2_start_ns, NowNanos());
    return;
  }

  int64_t total_states = 1;
  bool any_unassigned =
      absl::c_any_of(ws.pair_bit, [](int8_t b) { return b < 0; });
  if (any_unassigned) {
    auto components = DiscoverUnassignedComponents(ws);
    res->num_unassigned_components = static_cast<int>(components.size());
    if (std::optional<SolverConflict> comp_conflict =
            SolveAllComponents(ws, components, comp_cache, res, &total_states);
        comp_conflict.has_value()) {
      PopulateConflictIntoSolveResult(*comp_conflict, res);
      res->stage2_seconds = ElapsedSeconds(ws.t_stage2_start_ns, NowNanos());
      return;
    }
  }

  res->flat_foldable = true;
  res->states = total_states;
  res->stage = ExitStage::kSat;
  res->stage2_seconds = ElapsedSeconds(ws.t_stage2_start_ns, NowNanos());

  if (options.compute_layer_order) {
    PopulateLayerOrders(sup, ws.pair_bit, res);
  }
}

// ============================================================================
// Helper Builders for `SupportKey` / `MemoKey` & Support Construction
// ============================================================================

SupportKey MakeSupportKey(absl::Span<const InputSegment> lines,
                          FaceOrientationPolicy orientation_policy) {
  SupportKey key;
  key.flip_to_top_face_up =
      (orientation_policy == FaceOrientationPolicy::kFlipToTopFaceUp);
  key.segments.reserve(lines.size());
  for (const auto& ln : lines) {
    CreaseAssignment c = ln.assignment;
    CreaseAssignment mv_class =
        IsActiveCrease(c) ? CreaseAssignment::kMountain : c;
    auto [x1, y1, x2, y2] = CanonicalSegmentBits(ln.x1, ln.y1, ln.x2, ln.y2);
    key.segments.push_back(SegmentBitsKey{
        .x1 = x1, .y1 = y1, .x2 = x2, .y2 = y2, .code = mv_class});
  }
  return key;
}

MemoKey MakeMemoKey(absl::Span<const InputSegment> lines,
                    const SolveOptions& options) {
  MemoKey key;
  key.flip_to_top_face_up = options.flip_to_top_face_up;
  key.taco_only = options.taco_only;
  key.compute_layer_order = options.compute_layer_order;
  key.ignore_flat_creases = options.ignore_flat_creases;
  key.solution_limit = options.solution_limit;
  key.kawasaki_eps_bits = CoordBits(options.kawasaki_epsilon);
  key.face_group_ranks = options.face_group_ranks;
  key.changed_segments.reserve(options.changed_segments.size());
  for (const auto& seg : options.changed_segments) {
    auto [x1, y1, x2, y2] =
        CanonicalSegmentBits(seg.x1, seg.y1, seg.x2, seg.y2);
    key.changed_segments.push_back(
        SeedSegmentBitsKey{.x1 = x1, .y1 = y1, .x2 = x2, .y2 = y2});
  }
  absl::c_sort(key.changed_segments);
  key.changed_segments.erase(
      std::unique(key.changed_segments.begin(), key.changed_segments.end()),
      key.changed_segments.end());
  key.segments.reserve(lines.size());
  for (const auto& ln : lines) {
    CreaseAssignment c = ln.assignment;
    if (options.ignore_flat_creases && c == CreaseAssignment::kFlat) continue;
    auto [x1, y1, x2, y2] = CanonicalSegmentBits(ln.x1, ln.y1, ln.x2, ln.y2);
    key.segments.push_back(
        SegmentBitsKey{.x1 = x1, .y1 = y1, .x2 = x2, .y2 = y2, .code = c});
  }
  return key;
}

std::shared_ptr<SupportEntry> BuildSupportFromUnfoldedMesh(
    internal::UnfoldedMesh mesh) {
  auto sup = std::make_shared<SupportEntry>();
  sup->mesh = std::move(mesh);

  internal::FoldedArrangementResult folded =
      internal::BuildFoldedArrangement(sup->mesh);
  sup->folded_outcome = folded.outcome;
  if (folded.outcome != internal::FoldedArrangementOutcome::kOk) {
    sup->error_message = std::move(folded.error_message);
    return sup;
  }
  sup->folded = std::move(folded.data);
  CompileSupportConstraints(sup.get());
  return sup;
}

absl::StatusOr<std::vector<CreaseAssignment>> ExtractEdgeAssignmentsForLines(
    const SupportEntry& sup, absl::Span<const InputSegment> lines,
    FaceOrientationPolicy orientation_policy,
    absl::Span<const int> segment_line_numbers) {
  absl::StatusOr<std::vector<CreaseAssignment>> resolved =
      internal::ResolveEdgeAssignmentsFromLines(sup.mesh.edge_lines, lines,
                                                segment_line_numbers);
  if (!resolved.ok()) {
    return resolved.status();
  }
  const size_t num_e = sup.mesh.edge_vertices.size();
  std::vector<CreaseAssignment> edge_as(num_e, CreaseAssignment::kFlat);
  for (size_t ei = 0; ei < num_e; ++ei) {
    CreaseAssignment code =
        (ei < resolved->size()) ? (*resolved)[ei] : CreaseAssignment::kFlat;
    if (orientation_policy == FaceOrientationPolicy::kFlipToTopFaceUp) {
      code = FlipMountainValley(code);
    }
    if (sup.mesh.edge_faces[ei].size() == 1) {
      code = CreaseAssignment::kBoundary;
    }
    edge_as[ei] = code;
  }
  return edge_as;
}

struct ParsedCpDocument {
  std::vector<InputSegment> segments;
  std::vector<int> line_numbers;
};

absl::StatusOr<ParsedCpDocument> ParseCpDocumentWithLineNumbers(
    absl::string_view cp_doc, FlatCreasePolicy flat_policy) {
  ParsedCpDocument out;
  const bool ignore_flat_creases =
      (flat_policy == FlatCreasePolicy::kIgnoreFlatCreases);
  size_t pos = 0;
  const size_t n = cp_doc.size();
  int line_num = 0;
  while (pos < n) {
    ++line_num;
    size_t eol = cp_doc.find('\n', pos);
    absl::string_view line = (eol == absl::string_view::npos)
                                 ? cp_doc.substr(pos)
                                 : cp_doc.substr(pos, eol - pos);
    pos = (eol == absl::string_view::npos) ? n : eol + 1;

    std::array<absl::string_view, 5> toks;
    int t_cnt = 0;
    bool extra_tokens = false;
    size_t i = 0;
    while (i < line.size()) {
      while (i < line.size() &&
             (line[i] == ' ' || line[i] == '\t' || line[i] == '\r')) {
        ++i;
      }
      if (i >= line.size() || line[i] == '#') break;
      size_t start = i;
      while (i < line.size() && line[i] != ' ' && line[i] != '\t' &&
             line[i] != '\r' && line[i] != '#') {
        ++i;
      }
      if (t_cnt < 5) {
        toks[t_cnt++] = line.substr(start, i - start);
      } else {
        extra_tokens = true;
        break;
      }
    }
    if (t_cnt == 0) continue;
    if (t_cnt != 5 || extra_tokens) {
      return absl::InvalidArgumentError(absl::StrCat(
          "IO Parser failed at line ", line_num, ": expected 5 tokens"));
    }

    int raw_code = 0;
    double x1 = 0.0;
    double y1 = 0.0;
    double x2 = 0.0;
    double y2 = 0.0;
    if (!absl::SimpleAtoi(toks[0], &raw_code) ||
        !absl::SimpleAtod(toks[1], &x1) || !absl::SimpleAtod(toks[2], &y1) ||
        !absl::SimpleAtod(toks[3], &x2) || !absl::SimpleAtod(toks[4], &y2) ||
        !std::isfinite(x1) || !std::isfinite(y1) || !std::isfinite(x2) ||
        !std::isfinite(y2)) {
      return absl::InvalidArgumentError(
          absl::StrCat("IO Parser failed at line ", line_num,
                       ": invalid or non-finite numeric token"));
    }
    if (!IsValidCoordinate(x1) || !IsValidCoordinate(y1) ||
        !IsValidCoordinate(x2) || !IsValidCoordinate(y2)) {
      return absl::InvalidArgumentError(
          absl::StrCat("IO Parser failed at line ", line_num,
                       ": coordinate magnitude exceeds 1e100"));
    }
    std::optional<CreaseAssignment> norm_code =
        TryParseCreaseAssignment(raw_code);
    if (!norm_code.has_value()) {
      return absl::InvalidArgumentError(
          absl::StrCat("IO Parser failed at line ", line_num,
                       ": invalid crease code ", raw_code, " (expected 0..5)"));
    }
    if (x1 == x2 && y1 == y2) {
      return absl::InvalidArgumentError(absl::StrCat(
          "IO Parser failed at line ", line_num, ": zero-length segment"));
    }
    if (ignore_flat_creases && *norm_code == CreaseAssignment::kFlat) continue;
    out.segments.push_back(InputSegment{
        .x1 = x1, .y1 = y1, .x2 = x2, .y2 = y2, .assignment = *norm_code});
    out.line_numbers.push_back(line_num);
  }
  if (out.segments.empty()) {
    return absl::InvalidArgumentError("IO Parser failed: empty .cp document");
  }
  return out;
}

void PopulateParseErrorResult(absl::string_view reason, int64_t t_start_ns,
                              SolveResult* res) {
  PopulateConflictIntoSolveResult(
      SolverConflict{.stage = ExitStage::kParseError,
                     .kind = ConflictKind::kParseError,
                     .custom_reason = std::string(reason)},
      res);
  res->wall_seconds = ElapsedSeconds(t_start_ns, NowNanos());
}

void PopulateCouldNotDecideResult(absl::string_view reason, int64_t t_start_ns,
                                  SolveResult* res) {
  PopulateConflictIntoSolveResult(
      SolverConflict{.stage = ExitStage::kCouldNotDecide,
                     .kind = ConflictKind::kCouldNotDecide,
                     .custom_reason = std::string(reason)},
      res);
  res->wall_seconds = ElapsedSeconds(t_start_ns, NowNanos());
}

absl::Status ValidateOptionsAndSegments(absl::Span<const InputSegment> lines,
                                        const SolveOptions& options) {
  if (options.solution_limit < 0) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Invalid negative solution_limit: ", options.solution_limit,
        " (use 0 for unlimited)"));
  }
  if (!std::isfinite(options.kawasaki_epsilon) ||
      options.kawasaki_epsilon < 0.0) {
    return absl::InvalidArgumentError(
        absl::StrCat("Invalid kawasaki_epsilon: ", options.kawasaki_epsilon));
  }
  for (size_t i = 0; i < lines.size(); ++i) {
    const InputSegment& ln = lines[i];
    if (!std::isfinite(ln.x1) || !std::isfinite(ln.y1) ||
        !std::isfinite(ln.x2) || !std::isfinite(ln.y2)) {
      return absl::InvalidArgumentError("Non-finite input segment coordinate");
    }
    if (!IsValidCoordinate(ln.x1) || !IsValidCoordinate(ln.y1) ||
        !IsValidCoordinate(ln.x2) || !IsValidCoordinate(ln.y2)) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Input segment coordinate exceeds 1e100 magnitude bound at index ",
          i));
    }
    if (!IsValidCreaseAssignment(ln.assignment)) {
      return absl::InvalidArgumentError(
          absl::StrCat("Invalid crease assignment code ",
                       static_cast<int>(ln.assignment), " at index ", i));
    }
    if (ln.x1 == ln.x2 && ln.y1 == ln.y2) {
      return absl::InvalidArgumentError(
          absl::StrCat("Zero-length input segment at index ", i));
    }
  }
  for (const auto& seg : options.changed_segments) {
    if (!std::isfinite(seg.x1) || !std::isfinite(seg.y1) ||
        !std::isfinite(seg.x2) || !std::isfinite(seg.y2)) {
      return absl::InvalidArgumentError(
          "Non-finite changed_segments coordinate");
    }
    if (!IsValidCoordinate(seg.x1) || !IsValidCoordinate(seg.y1) ||
        !IsValidCoordinate(seg.x2) || !IsValidCoordinate(seg.y2)) {
      return absl::InvalidArgumentError(
          "changed_segments coordinate exceeds 1e100 magnitude bound");
    }
  }
  return absl::OkStatus();
}

absl::Status ValidateFaceGroupRanks(const SolveOptions& options,
                                    int num_faces) {
  if (!options.face_group_ranks.empty() &&
      options.face_group_ranks.size() != static_cast<size_t>(num_faces)) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Invalid face_group_ranks length ", options.face_group_ranks.size(),
        " (expected 0 or ", num_faces, ")"));
  }
  return absl::OkStatus();
}

}  // namespace

// ============================================================================
// Public Functions & `FastFlatFolderEngine`
// ============================================================================

absl::string_view ExitStageToString(ExitStage stage) {
  switch (stage) {
    case ExitStage::kUninitialized:
      return "";
    case ExitStage::kSat:
      return "sat";
    case ExitStage::kStage1Pass:
      return "stage1_pass";
    case ExitStage::kParseError:
      return "parse_error";
    case ExitStage::kLocalViolation:
      return "local_violation";
    case ExitStage::kFoldedGeometryError:
      return "folded_geometry_error";
    case ExitStage::kStage1TortillaOddCycle:
      return "stage1_tortilla_odd_cycle";
    case ExitStage::kStage1Tortilla:
      return "stage1_tortilla";
    case ExitStage::kStage1TortillaSeeded:
      return "stage1_tortilla_seeded";
    case ExitStage::kStage1TacoTaco:
      return "stage1_taco_taco";
    case ExitStage::kStage1TacoTacoSeeded:
      return "stage1_taco_taco_seeded";
    case ExitStage::kStage2Tortilla:
      return "stage2_tortilla";
    case ExitStage::kStage2Transitivity:
      return "stage2_transitivity";
    case ExitStage::kStage2TacoTaco:
      return "stage2_taco_taco";
    case ExitStage::kStage2ComponentUnsat:
      return "stage2_component_unsat";
    case ExitStage::kCouldNotDecide:
      return "could_not_decide";
  }
  return "parse_error";
}

absl::string_view FoldVerdictToString(FoldVerdict verdict) {
  switch (verdict) {
    case FoldVerdict::kCouldNotDecide:
      return "could_not_decide";
    case FoldVerdict::kFoldable:
      return "foldable";
    case FoldVerdict::kNotFoldable:
      return "not_foldable";
  }
  return "could_not_decide";
}

absl::StatusOr<std::vector<InputSegment>> ParseCpDocument(
    absl::string_view cp_doc, FlatCreasePolicy flat_policy) {
  absl::StatusOr<ParsedCpDocument> parsed =
      ParseCpDocumentWithLineNumbers(cp_doc, flat_policy);
  if (!parsed.ok()) {
    return parsed.status();
  }
  return std::move(parsed->segments);
}

absl::StatusOr<std::vector<int>> ComputeAbsoluteFaceOrders(
    absl::Span<const std::pair<int, int>> top_bottom_edges, int num_faces) {
  if (num_faces <= 0) return std::vector<int>{};
  if (num_faces > kMaxFaces) {
    return absl::InvalidArgumentError(absl::StrCat(
        "num_faces ", num_faces, " exceeds kMaxFaces (", kMaxFaces, ")"));
  }
  std::vector<std::vector<int>> adj(num_faces);
  std::vector<int> in_degree(num_faces, 0);
  for (auto [top_f, bot_f] : top_bottom_edges) {
    if (top_f >= 0 && top_f < num_faces && bot_f >= 0 && bot_f < num_faces) {
      adj[bot_f].push_back(top_f);
      ++in_degree[top_f];
    }
  }
  // Matches `fold_cp.get_absolute_face_orders_from_edges`: Kahn's longest-path
  // DAG relaxation from bottom (`in_degree == 0`, order `0`) to top. Any faces
  // participating in a directed cycle retain `in_degree > 0` and remain at `0`.
  std::vector<int> orders(num_faces, 0);
  std::vector<int> queue;
  queue.reserve(num_faces);
  for (int i = 0; i < num_faces; ++i) {
    if (in_degree[i] == 0) queue.push_back(i);
  }
  for (size_t head = 0; head < queue.size(); ++head) {
    int bot_f = queue[head];
    for (int top_f : adj[bot_f]) {
      orders[top_f] = std::max(orders[top_f], orders[bot_f] + 1);
      if (--in_degree[top_f] == 0) {
        queue.push_back(top_f);
      }
    }
  }
  return orders;
}

struct FastFlatFolderEngine::Impl {
  Impl(size_t max_memo, size_t max_sup, size_t max_comp, size_t max_sup_bytes)
      : memo_cache(max_memo),
        support_cache(max_sup, max_sup_bytes),
        comp_cache(max_comp) {}

  SolveResult AnalyzeLinesWithLineNumbers(
      absl::Span<const InputSegment> lines,
      absl::Span<const int> segment_line_numbers, const SolveOptions& options);

  LruCache<MemoKey, SolveResult> memo_cache;
  LruCache<SupportKey, SupportEntry> support_cache;
  LruCache<ComponentKey, ComponentCacheValue> comp_cache;
};

FastFlatFolderEngine::FastFlatFolderEngine(size_t max_memo_entries,
                                           size_t max_support_entries,
                                           size_t max_component_entries,
                                           size_t max_support_bytes)
    : impl_(std::make_unique<Impl>(max_memo_entries, max_support_entries,
                                   max_component_entries, max_support_bytes)) {}

FastFlatFolderEngine::~FastFlatFolderEngine() = default;

void FastFlatFolderEngine::ClearCaches() {
  impl_->memo_cache.Clear();
  impl_->support_cache.Clear();
  impl_->comp_cache.Clear();
}

EngineCacheStats FastFlatFolderEngine::GetCacheStats() const {
  const CacheCounters memo = impl_->memo_cache.SnapshotCounters();
  const CacheCounters support = impl_->support_cache.SnapshotCounters();
  const CacheCounters comp = impl_->comp_cache.SnapshotCounters();
  return EngineCacheStats{
      .memo_entries = memo.size,
      .support_entries = support.size,
      .support_bytes = support.total_bytes,
      .component_entries = comp.size,
      .memo_hits = memo.hits,
      .memo_misses = memo.misses,
      .support_hits = support.hits,
      .support_misses = support.misses,
      .component_hits = comp.hits,
      .component_misses = comp.misses,
  };
}

SolveResult FastFlatFolderEngine::Impl::AnalyzeLinesWithLineNumbers(
    absl::Span<const InputSegment> lines,
    absl::Span<const int> segment_line_numbers, const SolveOptions& options) {
  const int64_t t_wall_start = NowNanos();
  SolveResult res;
  if (absl::Status status = ValidateOptionsAndSegments(lines, options);
      !status.ok()) {
    PopulateParseErrorResult(status.message(), t_wall_start, &res);
    return res;
  }

  std::vector<InputSegment> filtered_storage;
  std::vector<int> filtered_line_numbers;
  absl::Span<const InputSegment> eff_lines = lines;
  absl::Span<const int> eff_line_numbers = segment_line_numbers;
  if (options.ignore_flat_creases) {
    filtered_storage.reserve(lines.size());
    const bool keep_line_numbers = !segment_line_numbers.empty() &&
                                   segment_line_numbers.size() == lines.size();
    if (keep_line_numbers) filtered_line_numbers.reserve(lines.size());
    for (size_t i = 0; i < lines.size(); ++i) {
      if (lines[i].assignment != CreaseAssignment::kFlat) {
        filtered_storage.push_back(lines[i]);
        if (keep_line_numbers) {
          filtered_line_numbers.push_back(segment_line_numbers[i]);
        }
      }
    }
    eff_lines = filtered_storage;
    if (keep_line_numbers) eff_line_numbers = filtered_line_numbers;
  }

  if (eff_lines.empty()) {
    PopulateParseErrorResult("IO Parser failed", t_wall_start, &res);
    return res;
  }

  MemoKey mkey;
  if (options.use_memo_cache) {
    mkey = MakeMemoKey(eff_lines, options);
    if (auto cached = memo_cache.Lookup(mkey)) {
      res = *cached;
      res.memo_cache_hit = true;
      res.support_cache_hit = false;
      res.component_cache_hits = 0;
      res.component_cache_misses = 0;
      res.parse_arrange_seconds = 0.0;
      res.stage1_seconds = 0.0;
      res.stage2_seconds = 0.0;
      res.order_seconds = 0.0;
      res.wall_seconds = ElapsedSeconds(t_wall_start, NowNanos());
      return res;
    }
  }

  const FaceOrientationPolicy orientation_policy =
      options.flip_to_top_face_up ? FaceOrientationPolicy::kFlipToTopFaceUp
                                  : FaceOrientationPolicy::kPreserveInput;
  const int64_t t_arr_start = NowNanos();
  std::shared_ptr<const SupportEntry> sup;
  SupportKey skey;
  if (options.use_support_cache) {
    skey = MakeSupportKey(eff_lines, orientation_policy);
    sup = support_cache.Lookup(skey);
    if (sup != nullptr) {
      res.support_cache_hit = true;
    }
  }

  LocalCheckPolicy local_check_policy = LocalCheckPolicy::kRun;
  if (sup == nullptr) {
    absl::StatusOr<internal::UnfoldedMesh> mesh =
        internal::BuildUnfoldedMeshFromLines(eff_lines, orientation_policy,
                                             eff_line_numbers);
    if (!mesh.ok()) {
      res.parse_arrange_seconds = ElapsedSeconds(t_arr_start, NowNanos());
      if (absl::IsInvalidArgument(mesh.status())) {
        PopulateParseErrorResult(mesh.status().message(), t_wall_start, &res);
      } else {
        PopulateCouldNotDecideResult(mesh.status().message(), t_wall_start,
                                     &res);
      }
      return res;
    }
    if (absl::Status rank_status = ValidateFaceGroupRanks(
            options, static_cast<int>(mesh->face_vertices.size()));
        !rank_status.ok()) {
      res.parse_arrange_seconds = ElapsedSeconds(t_arr_start, NowNanos());
      PopulateParseErrorResult(rank_status.message(), t_wall_start, &res);
      return res;
    }

    // Check local foldability (Maekawa/Kawasaki) before building the folded
    // arrangement: folding a locally invalid pattern can produce degenerate
    // folded edges that stall the float sweep line. Do not insert into
    // `support_cache` on failure because `SupportKey` does not include
    // `kawasaki_epsilon` (a later call with a looser tolerance must not reuse
    // a cached failure); `memo_cache` still caches the exact result.
    const int64_t t_local_start = NowNanos();
    std::vector<std::string> local_violations =
        internal::CheckLocalVertexFoldability(*mesh, mesh->edge_assignments,
                                              options.kawasaki_epsilon);
    if (!local_violations.empty()) {
      res.parse_arrange_seconds = ElapsedSeconds(t_arr_start, t_local_start);
      PopulateLocalViolationFields(*mesh, std::move(local_violations),
                                   t_local_start, &res);
      res.wall_seconds = ElapsedSeconds(t_wall_start, NowNanos());
      if (options.use_memo_cache) {
        memo_cache.Insert(std::move(mkey), std::make_shared<SolveResult>(res));
      }
      return res;
    }

    std::shared_ptr<SupportEntry> built =
        BuildSupportFromUnfoldedMesh(std::move(*mesh));
    if (options.use_support_cache) {
      size_t sup_bytes = EstimateSupportEntryBytes(*built);
      support_cache.Insert(std::move(skey), built, sup_bytes);
    }
    sup = std::move(built);
    local_check_policy = LocalCheckPolicy::kAlreadyPassed;
  } else if (absl::Status rank_status =
                 ValidateFaceGroupRanks(options, sup->num_faces);
             !rank_status.ok()) {
    res.parse_arrange_seconds = ElapsedSeconds(t_arr_start, NowNanos());
    PopulateParseErrorResult(rank_status.message(), t_wall_start, &res);
    return res;
  }
  res.parse_arrange_seconds = ElapsedSeconds(t_arr_start, NowNanos());

  std::vector<CreaseAssignment> edge_as;
  absl::Span<const CreaseAssignment> edge_as_span = sup->mesh.edge_assignments;
  if (res.support_cache_hit) {
    absl::StatusOr<std::vector<CreaseAssignment>> extracted =
        ExtractEdgeAssignmentsForLines(*sup, eff_lines, orientation_policy,
                                       eff_line_numbers);
    if (!extracted.ok()) {
      PopulateParseErrorResult(extracted.status().message(), t_wall_start,
                               &res);
      return res;
    }
    edge_as = std::move(*extracted);
    edge_as_span = edge_as;
  }

  SolveOnSupport(*sup, edge_as_span, options,
                 options.use_component_cache ? &comp_cache : nullptr,
                 local_check_policy, &res);
  res.wall_seconds = ElapsedSeconds(t_wall_start, NowNanos());

  if (options.use_memo_cache) {
    memo_cache.Insert(std::move(mkey), std::make_shared<SolveResult>(res));
  }
  return res;
}

SolveResult FastFlatFolderEngine::AnalyzeLines(
    absl::Span<const InputSegment> lines, const SolveOptions& options) {
  return impl_->AnalyzeLinesWithLineNumbers(lines, /*segment_line_numbers=*/{},
                                            options);
}

SolveResult FastFlatFolderEngine::AnalyzeCp(absl::string_view cp_doc,
                                            const SolveOptions& options) {
  const int64_t t_start = NowNanos();
  const FlatCreasePolicy flat_policy =
      options.ignore_flat_creases ? FlatCreasePolicy::kIgnoreFlatCreases
                                  : FlatCreasePolicy::kKeepFlatCreases;
  absl::StatusOr<ParsedCpDocument> parsed =
      ParseCpDocumentWithLineNumbers(cp_doc, flat_policy);
  if (!parsed.ok()) {
    SolveResult res;
    PopulateParseErrorResult(parsed.status().message(), t_start, &res);
    return res;
  }
  SolveOptions sub_opt = options;
  sub_opt.ignore_flat_creases = false;
  SolveResult res = impl_->AnalyzeLinesWithLineNumbers(
      parsed->segments, parsed->line_numbers, sub_opt);
  res.wall_seconds = ElapsedSeconds(t_start, NowNanos());
  return res;
}

SolveResult FastFlatFolderEngine::AnalyzeFoldGraph(
    absl::Span<const std::pair<double, double>> vertex_coords,
    absl::Span<const std::pair<int, int>> edge_vertices,
    absl::Span<const CreaseAssignment> edge_assignments,
    absl::Span<const std::vector<int>> optional_face_vertices,
    const SolveOptions& options) {
  const int64_t t_start = NowNanos();
  SolveResult res;
  if (absl::Status status = ValidateOptionsAndSegments(/*lines=*/{}, options);
      !status.ok()) {
    PopulateParseErrorResult(status.message(), t_start, &res);
    return res;
  }

  const FaceOrientationPolicy orientation_policy =
      options.flip_to_top_face_up ? FaceOrientationPolicy::kFlipToTopFaceUp
                                  : FaceOrientationPolicy::kPreserveInput;
  absl::StatusOr<internal::UnfoldedMesh> mesh =
      internal::BuildUnfoldedMeshFromFoldGraph(
          vertex_coords, edge_vertices, edge_assignments,
          optional_face_vertices, orientation_policy);
  if (!mesh.ok()) {
    PopulateParseErrorResult(mesh.status().message(), t_start, &res);
    return res;
  }
  if (absl::Status rank_status = ValidateFaceGroupRanks(
          options, static_cast<int>(mesh->face_vertices.size()));
      !rank_status.ok()) {
    PopulateParseErrorResult(rank_status.message(), t_start, &res);
    return res;
  }

  const int64_t t_local_start = NowNanos();
  std::vector<std::string> local_violations =
      internal::CheckLocalVertexFoldability(*mesh, mesh->edge_assignments,
                                            options.kawasaki_epsilon);
  if (!local_violations.empty()) {
    res.parse_arrange_seconds = ElapsedSeconds(t_start, t_local_start);
    PopulateLocalViolationFields(*mesh, std::move(local_violations),
                                 t_local_start, &res);
    res.wall_seconds = ElapsedSeconds(t_start, NowNanos());
    return res;
  }

  std::shared_ptr<SupportEntry> sup =
      BuildSupportFromUnfoldedMesh(std::move(*mesh));
  res.parse_arrange_seconds = ElapsedSeconds(t_start, NowNanos());
  SolveOnSupport(*sup, sup->mesh.edge_assignments, options,
                 options.use_component_cache ? &impl_->comp_cache : nullptr,
                 LocalCheckPolicy::kAlreadyPassed, &res);
  res.wall_seconds = ElapsedSeconds(t_start, NowNanos());
  return res;
}

absl::StatusOr<UnfoldedFaces> FastFlatFolderEngine::ExtractUnfoldedFacesCp(
    absl::string_view cp_doc, FaceOrientationPolicy orientation_policy) {
  absl::StatusOr<ParsedCpDocument> parsed = ParseCpDocumentWithLineNumbers(
      cp_doc, FlatCreasePolicy::kKeepFlatCreases);
  if (!parsed.ok()) {
    return parsed.status();
  }
  SupportKey skey = MakeSupportKey(parsed->segments, orientation_policy);
  if (auto sup = impl_->support_cache.Lookup(skey)) {
    if (absl::StatusOr<std::vector<CreaseAssignment>> resolved =
            internal::ResolveEdgeAssignmentsFromLines(
                sup->mesh.edge_lines, parsed->segments, parsed->line_numbers);
        !resolved.ok()) {
      return resolved.status();
    }
    return UnfoldedFaces{
        .vertex_coords = sup->mesh.vertex_coords,
        .face_vertices = sup->mesh.face_vertices,
    };
  }
  // Build only the unfolded mesh on a cache miss so locally invalid inputs
  // never invoke `BuildFoldedArrangement` or populate `support_cache`.
  absl::StatusOr<internal::UnfoldedMesh> mesh =
      internal::BuildUnfoldedMeshFromLines(parsed->segments, orientation_policy,
                                           parsed->line_numbers);
  if (!mesh.ok()) {
    return mesh.status();
  }
  return UnfoldedFaces{
      .vertex_coords = std::move(mesh->vertex_coords),
      .face_vertices = std::move(mesh->face_vertices),
  };
}

}  // namespace bp_pipeline_flat_folder
