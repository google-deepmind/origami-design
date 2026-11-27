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

// Unfolded and folded planar arrangement construction and local vertex
// flat-foldability verification (`Maekawa` and `Kawasaki` theorems).
//
// For octilinear crease patterns (`0`, `45`, `90`, `135` degrees) whose
// endpoints quantize cleanly onto a scaled integer lattice, both the unfolded
// arrangement (`BuildArrangementFromSegments`) and the rigid-body folded cell
// arrangement (`BuildFoldedArrangement`) execute in exact `int64_t` /
// `absl::int128` arithmetic, avoiding `1 / sqrt(2)` float64 reflection drift.
// Non-octilinear or out-of-range inputs fall back to the adaptive-epsilon
// `GeneralSweepRunner` matching `bp_pipeline.third_party.flat_folder`.

#include "fast_flat_folder_geometry.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numbers>
#include <optional>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/absl_check.h"
#include "absl/numeric/int128.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "fast_flat_folder_types.h"

namespace bp_pipeline_flat_folder::internal {
namespace {

constexpr double kPi = std::numbers::pi;
constexpr double kTwoPi = 2.0 * kPi;
constexpr double kFloatEpsilon = 1e-16;

// Bound on `|coord * exact_scale|` for exact octilinear quantization. Keeping
// scaled coordinates within `4.0e8` guarantees that `dx = x2 - x1`,
// `c_val = x +/- y` (`<= 8.0e8`), and `sum = c_a + c_b` (`<= 1.6e9`) in
// `IntersectOctiLines` cannot overflow `int64_t` (or even `int32_t`).
constexpr double kMaxExactScaledCoord = 4.0e8;
constexpr double kForcedScaleQuantizeTol = 1e-4;
constexpr double kAutoScaleQuantizeTol = 4e-6;
// Maximum magnitude of exact folded integer coordinates (`2^60`), keeping
// differences (`<= 2^61`) and `OctiLineConstant` sums safely inside `int64_t`.
constexpr int64_t kFoldedCoordLimit = int64_t{1} << 60;

// Reserved slot `0` in `GeneralSweepRunner::vertices_` holding `(-inf, -inf)`
// so `RemoveNextVertex(kSweepSentinelVertexIdx)` pops the minimum queued event.
constexpr int kSweepSentinelVertexIdx = 0;
// Reserved slot `0` in `GeneralSweepRunner::segment_*` used as a mutable
// sweep-line binary-search probe at the current event vertex.
constexpr int kSweepProbeSegmentIdx = 0;
constexpr int kMaxGeneralSweepExpIterations = 50;
constexpr int kTargetSweepStability = 3;

double MathAngle0To2Pi(double dx, double dy) {
  double angle = std::atan2(dy, dx);
  return angle + ((angle < 0.0) ? kTwoPi : 0.0);
}

double SafeAngle0To2Pi(double dx, double dy) {
  // Canonicalize `-0.0` to `+0.0` so `atan2` returns `0.0` (rather than `-0.0`
  // or `+/-pi`) consistently for axis-aligned vectors.
  if (dx == 0.0) dx = 0.0;
  if (dy == 0.0) dy = 0.0;
  double angle = std::atan2(dy, dx);
  if (angle < 0.0) angle += kTwoPi;
  if (angle >= kTwoPi) angle = 0.0;
  return angle;
}

double PolygonTwiceSignedAreaDouble(
    absl::Span<const int> poly,
    absl::Span<const std::pair<double, double>> coords) {
  if (poly.empty()) return 0.0;
  double area = 0.0;
  ABSL_DCHECK_GE(poly.back(), 0);
  ABSL_DCHECK_LT(poly.back(), static_cast<int>(coords.size()));
  auto [x1, y1] = coords[poly.back()];
  for (int v : poly) {
    ABSL_DCHECK_GE(v, 0);
    ABSL_DCHECK_LT(v, static_cast<int>(coords.size()));
    auto [x2, y2] = coords[v];
    area += (x1 + x2) * (y2 - y1);
    x1 = x2;
    y1 = y2;
  }
  return area;
}

// ============================================================================
// Exact `int64_t` Octilinear Planar Arrangement
// ============================================================================

enum class OctiDirection : uint8_t {
  kHorizontal = 0,  // Y = c
  kVertical = 1,    // X = c
  kDiagPos = 2,     // X - Y = c
  kDiagNeg = 3,     // X + Y = c
};

struct OctiInterval {
  int64_t lo = 0;
  int64_t hi = 0;
  int line_idx = 0;
};

struct OctiRun {
  OctiDirection dir = OctiDirection::kHorizontal;
  int64_t c_val = 0;
  // Parameter along line: `X` for `{kHorizontal, kDiagPos, kDiagNeg}`, `Y` for
  // `kVertical`.
  int64_t lo = 0;
  int64_t hi = 0;
  std::vector<OctiInterval> intervals;
  std::vector<int64_t> splits;
};

std::pair<int64_t, int64_t> OctiPointAt(OctiDirection dir, int64_t c_val,
                                        int64_t param) {
  switch (dir) {
    case OctiDirection::kHorizontal:
      return {param, c_val};
    case OctiDirection::kVertical:
      return {c_val, param};
    case OctiDirection::kDiagPos:
      return {param, param - c_val};
    case OctiDirection::kDiagNeg:
      return {param, c_val - param};
  }
  ABSL_DCHECK(false) << "Invalid OctiDirection: " << static_cast<int>(dir);
  return {param, c_val - param};
}

int64_t OctiParamOf(OctiDirection dir, int64_t x, int64_t y) {
  return (dir == OctiDirection::kVertical) ? y : x;
}

std::optional<OctiDirection> ClassifyOctilinearDelta(int64_t dx, int64_t dy) {
  if (dx == 0 && dy == 0) return std::nullopt;
  if (dy == 0) return OctiDirection::kHorizontal;
  if (dx == 0) return OctiDirection::kVertical;
  if (dx == dy) return OctiDirection::kDiagPos;
  if (dx == -dy) return OctiDirection::kDiagNeg;
  return std::nullopt;
}

int64_t OctiLineConstant(OctiDirection dir, int64_t x, int64_t y) {
  switch (dir) {
    case OctiDirection::kHorizontal:
      return y;
    case OctiDirection::kVertical:
      return x;
    case OctiDirection::kDiagPos:
      return x - y;
    case OctiDirection::kDiagNeg:
      return x + y;
  }
  ABSL_DCHECK(false) << "Invalid OctiDirection: " << static_cast<int>(dir);
  return x + y;
}

std::optional<std::pair<int64_t, int64_t>> IntersectOctiLines(
    OctiDirection dir_a, int64_t c_a, OctiDirection dir_b, int64_t c_b) {
  if (dir_a == dir_b) return std::nullopt;
  if (static_cast<uint8_t>(dir_a) > static_cast<uint8_t>(dir_b)) {
    return IntersectOctiLines(dir_b, c_b, dir_a, c_a);
  }
  if (dir_a == OctiDirection::kHorizontal &&
      dir_b == OctiDirection::kVertical) {
    return std::make_pair(c_b, c_a);
  }
  if (dir_a == OctiDirection::kHorizontal && dir_b == OctiDirection::kDiagPos) {
    return std::make_pair(c_a + c_b, c_a);
  }
  if (dir_a == OctiDirection::kHorizontal && dir_b == OctiDirection::kDiagNeg) {
    return std::make_pair(c_b - c_a, c_a);
  }
  if (dir_a == OctiDirection::kVertical && dir_b == OctiDirection::kDiagPos) {
    return std::make_pair(c_a, c_a - c_b);
  }
  if (dir_a == OctiDirection::kVertical && dir_b == OctiDirection::kDiagNeg) {
    return std::make_pair(c_a, c_b - c_a);
  }
  // `dir_a == kDiagPos` (`X - Y = c_a`) and `dir_b == kDiagNeg` (`X + Y = c_b`)
  int64_t sum = c_a + c_b;
  if ((sum & 1) != 0) return std::nullopt;
  return std::make_pair(sum / 2, (c_b - c_a) / 2);
}

std::optional<std::vector<OctiRun>> GroupOctilinearRuns(
    absl::Span<const std::tuple<int64_t, int64_t, int64_t, int64_t>> segs) {
  if (segs.empty()) return std::nullopt;
  absl::flat_hash_map<std::pair<uint8_t, int64_t>, std::vector<OctiInterval>>
      grouped;
  grouped.reserve(segs.size());

  for (int i = 0; i < static_cast<int>(segs.size()); ++i) {
    auto [x1, y1, x2, y2] = segs[i];
    std::optional<OctiDirection> dir =
        ClassifyOctilinearDelta(x2 - x1, y2 - y1);
    if (!dir.has_value()) return std::nullopt;
    int64_t c_val = OctiLineConstant(*dir, x1, y1);
    int64_t p1 = OctiParamOf(*dir, x1, y1);
    int64_t p2 = OctiParamOf(*dir, x2, y2);
    if (p2 < p1) std::swap(p1, p2);
    grouped[{static_cast<uint8_t>(*dir), c_val}].push_back(
        OctiInterval{.lo = p1, .hi = p2, .line_idx = i});
  }

  std::vector<OctiRun> runs;
  runs.reserve(grouped.size());
  for (auto& [key, intervals] : grouped) {
    absl::c_sort(intervals, [](const OctiInterval& a, const OctiInterval& b) {
      if (a.lo != b.lo) return a.lo < b.lo;
      if (a.hi != b.hi) return a.hi < b.hi;
      return a.line_idx < b.line_idx;
    });
    size_t idx = 0;
    while (idx < intervals.size()) {
      OctiRun run;
      run.dir = static_cast<OctiDirection>(key.first);
      run.c_val = key.second;
      run.lo = intervals[idx].lo;
      run.hi = intervals[idx].hi;
      run.intervals.push_back(intervals[idx]);
      ++idx;
      while (idx < intervals.size() && intervals[idx].lo <= run.hi) {
        run.hi = std::max(run.hi, intervals[idx].hi);
        run.intervals.push_back(intervals[idx]);
        ++idx;
      }
      run.splits.push_back(run.lo);
      run.splits.push_back(run.hi);
      runs.push_back(std::move(run));
    }
  }
  return runs;
}

std::optional<std::vector<std::pair<int64_t, int64_t>>>
SplitOctilinearRunsAtIntersections(std::vector<OctiRun>* runs) {
  const int num_runs = static_cast<int>(runs->size());
  for (int a = 0; a < num_runs; ++a) {
    for (int b = a + 1; b < num_runs; ++b) {
      if ((*runs)[a].dir == (*runs)[b].dir) continue;
      std::optional<std::pair<int64_t, int64_t>> inter = IntersectOctiLines(
          (*runs)[a].dir, (*runs)[a].c_val, (*runs)[b].dir, (*runs)[b].c_val);
      if (!inter.has_value()) return std::nullopt;
      auto [ix, iy] = *inter;
      int64_t ta = OctiParamOf((*runs)[a].dir, ix, iy);
      if (ta < (*runs)[a].lo || ta > (*runs)[a].hi) continue;
      int64_t tb = OctiParamOf((*runs)[b].dir, ix, iy);
      if (tb < (*runs)[b].lo || tb > (*runs)[b].hi) continue;
      (*runs)[a].splits.push_back(ta);
      (*runs)[b].splits.push_back(tb);
    }
  }

  std::vector<std::pair<int64_t, int64_t>> unique_pts;
  unique_pts.reserve(num_runs * 4);
  for (auto& run : *runs) {
    SortAndUnique(&run.splits);
    for (int64_t t : run.splits) {
      unique_pts.push_back(OctiPointAt(run.dir, run.c_val, t));
    }
  }

  // Match `_compare_points`: sort lexicographically by `(y, x)` ascending.
  absl::c_sort(unique_pts, [](const auto& a, const auto& b) {
    if (a.second != b.second) return a.second < b.second;
    return a.first < b.first;
  });
  unique_pts.erase(std::unique(unique_pts.begin(), unique_pts.end()),
                   unique_pts.end());
  return unique_pts;
}

std::optional<ArrangementGraph> BuildOctilinearFromExactSegments(
    absl::Span<const std::tuple<int64_t, int64_t, int64_t, int64_t>> segs,
    int64_t exact_scale) {
  std::optional<std::vector<OctiRun>> runs = GroupOctilinearRuns(segs);
  if (!runs.has_value()) return std::nullopt;
  std::optional<std::vector<std::pair<int64_t, int64_t>>> unique_pts =
      SplitOctilinearRunsAtIntersections(&*runs);
  if (!unique_pts.has_value()) return std::nullopt;

  ArrangementGraph out;
  absl::flat_hash_map<std::pair<int64_t, int64_t>, int> pt_to_vid;
  pt_to_vid.reserve(unique_pts->size() * 2);
  const double scale_d = static_cast<double>(exact_scale);
  out.vertices.resize(unique_pts->size());
  out.exact_vertices = *unique_pts;
  out.used_exact_octilinear = true;
  out.exact_scale = exact_scale;
  for (int vid = 0; vid < static_cast<int>(unique_pts->size()); ++vid) {
    pt_to_vid[(*unique_pts)[vid]] = vid;
    out.vertices[vid] = {
        static_cast<double>((*unique_pts)[vid].first) / scale_d,
        static_cast<double>((*unique_pts)[vid].second) / scale_d};
  }

  struct RawEdge {
    int u = 0;
    int v = 0;
    std::vector<int> lines{};
  };
  std::vector<RawEdge> raw_edges;
  raw_edges.reserve(unique_pts->size() * 2);

  for (const auto& run : *runs) {
    std::vector<OctiInterval> by_line = run.intervals;
    absl::c_sort(by_line, [](const OctiInterval& a, const OctiInterval& b) {
      return a.line_idx < b.line_idx;
    });
    for (size_t k = 0; k + 1 < run.splits.size(); ++k) {
      int64_t s0 = run.splits[k];
      int64_t s1 = run.splits[k + 1];
      auto it_u = pt_to_vid.find(OctiPointAt(run.dir, run.c_val, s0));
      auto it_v = pt_to_vid.find(OctiPointAt(run.dir, run.c_val, s1));
      ABSL_DCHECK(it_u != pt_to_vid.end());
      ABSL_DCHECK(it_v != pt_to_vid.end());
      int u = it_u->second;
      int v = it_v->second;
      if (u == v) continue;
      if (v < u) std::swap(u, v);
      RawEdge re{.u = u, .v = v};
      for (const auto& iv : by_line) {
        if (std::max(iv.lo, s0) < std::min(iv.hi, s1)) {
          if (re.lines.empty() || re.lines.back() != iv.line_idx) {
            re.lines.push_back(iv.line_idx);
          }
        }
      }
      raw_edges.push_back(std::move(re));
    }
  }

  absl::c_sort(raw_edges, [](const RawEdge& a, const RawEdge& b) {
    if (a.u != b.u) return a.u < b.u;
    return a.v < b.v;
  });

  out.edge_vertices.reserve(raw_edges.size());
  out.edge_lines.reserve(raw_edges.size());
  for (auto& re : raw_edges) {
    if (!out.edge_vertices.empty() && out.edge_vertices.back().first == re.u &&
        out.edge_vertices.back().second == re.v) {
      auto& dst = out.edge_lines.back();
      dst.insert(dst.end(), re.lines.begin(), re.lines.end());
      SortAndUnique(&dst);
    } else {
      out.edge_vertices.push_back({re.u, re.v});
      out.edge_lines.push_back(std::move(re.lines));
    }
  }

  // Defer graphs with dangling degree-1 vertices to `BuildGeneralArrangement`
  // so `_SweepLineRunner` rejects incomplete interior crease spikes identically
  // to the Python reference while keeping closed arrangements on the exact
  // integer path.
  std::vector<int> deg(unique_pts->size(), 0);
  for (auto [u, v] : out.edge_vertices) {
    ++deg[u];
    ++deg[v];
  }
  for (int d : deg) {
    if (d < 2) return std::nullopt;
  }
  return out;
}

bool HasUnanchoredCollinearOverlap(
    absl::Span<const std::tuple<int64_t, int64_t, int64_t, int64_t>> segs) {
  std::vector<std::tuple<uint8_t, int64_t, int64_t, int64_t>> intervals;
  intervals.reserve(segs.size());
  for (const auto& [x1, y1, x2, y2] : segs) {
    std::optional<OctiDirection> dir =
        ClassifyOctilinearDelta(x2 - x1, y2 - y1);
    ABSL_DCHECK(dir.has_value());
    int64_t c_val = OctiLineConstant(*dir, x1, y1);
    int64_t p1 = OctiParamOf(*dir, x1, y1);
    int64_t p2 = OctiParamOf(*dir, x2, y2);
    if (p2 < p1) std::swap(p1, p2);
    intervals.emplace_back(static_cast<uint8_t>(*dir), c_val, p1, p2);
  }
  absl::c_sort(intervals);
  bool has_overlap = false;
  for (size_t i = 1; i < intervals.size(); ++i) {
    const auto& [pd, pc, plo, phi] = intervals[i - 1];
    const auto& [d, c, lo, hi] = intervals[i];
    if (d == pd && c == pc && lo < phi && !(lo == plo && hi == phi)) {
      has_overlap = true;
      break;
    }
  }
  if (!has_overlap) return false;

  absl::flat_hash_map<std::pair<int64_t, int64_t>, std::pair<uint8_t, int64_t>>
      ep_line;
  absl::flat_hash_set<std::pair<int64_t, int64_t>> multi_line_eps;
  for (const auto& [x1, y1, x2, y2] : segs) {
    std::optional<OctiDirection> dir =
        ClassifyOctilinearDelta(x2 - x1, y2 - y1);
    uint8_t dir_code = static_cast<uint8_t>(*dir);
    int64_t c_val = OctiLineConstant(*dir, x1, y1);
    for (const auto& pt : {std::pair<int64_t, int64_t>{x1, y1},
                           std::pair<int64_t, int64_t>{x2, y2}}) {
      auto [it, inserted] =
          ep_line.emplace(pt, std::make_pair(dir_code, c_val));
      if (!inserted && it->second != std::make_pair(dir_code, c_val)) {
        multi_line_eps.insert(pt);
      }
    }
  }
  for (size_t i = 1; i < intervals.size(); ++i) {
    const auto& [pd, pc, plo, phi] = intervals[i - 1];
    const auto& [d, c, lo, hi] = intervals[i];
    if (d == pd && c == pc && lo < phi && !(lo == plo && hi == phi)) {
      int64_t span_lo = std::min(plo, lo);
      int64_t span_hi = std::max(phi, hi);
      OctiDirection dir = static_cast<OctiDirection>(d);
      for (int64_t param : {plo, phi, lo, hi}) {
        if (param > span_lo && param < span_hi &&
            !multi_line_eps.contains(OctiPointAt(dir, c, param))) {
          return true;
        }
      }
    }
  }
  return false;
}

enum class QuantizeScaleOutcome {
  kDidNotQuantize = 0,
  kQuantizedButDeferred = 1,
  kBuiltExact = 2,
};

QuantizeScaleOutcome TryQuantizeOctilinearAtScale(
    absl::Span<const InputSegment> lines, int64_t exact_scale, double tol,
    ArrangementGraph* out) {
  std::vector<std::tuple<int64_t, int64_t, int64_t, int64_t>> segs;
  segs.reserve(lines.size());
  const double scale_d = static_cast<double>(exact_scale);
  for (const auto& ln : lines) {
    for (double v : {ln.x1, ln.y1, ln.x2, ln.y2}) {
      if (!std::isfinite(v) || std::abs(v) * scale_d > kMaxExactScaledCoord) {
        return QuantizeScaleOutcome::kDidNotQuantize;
      }
      double scaled = v * scale_d;
      int64_t r = std::llround(scaled);
      if (std::abs(scaled - static_cast<double>(r)) > tol) {
        return QuantizeScaleOutcome::kDidNotQuantize;
      }
    }
    int64_t x1 = std::llround(ln.x1 * scale_d);
    int64_t y1 = std::llround(ln.y1 * scale_d);
    int64_t x2 = std::llround(ln.x2 * scale_d);
    int64_t y2 = std::llround(ln.y2 * scale_d);
    if (!ClassifyOctilinearDelta(x2 - x1, y2 - y1).has_value()) {
      return QuantizeScaleOutcome::kDidNotQuantize;
    }
    segs.emplace_back(x1, y1, x2, y2);
  }
  if (HasUnanchoredCollinearOverlap(segs)) {
    return QuantizeScaleOutcome::kQuantizedButDeferred;
  }
  if (std::optional<ArrangementGraph> built =
          BuildOctilinearFromExactSegments(segs, exact_scale);
      built.has_value()) {
    *out = std::move(*built);
    return QuantizeScaleOutcome::kBuiltExact;
  }
  return QuantizeScaleOutcome::kQuantizedButDeferred;
}

std::optional<ArrangementGraph> TryBuildOctilinearArrangement(
    absl::Span<const InputSegment> lines, int64_t forced_exact_scale) {
  if (lines.empty()) return std::nullopt;
  static constexpr std::array<int64_t, 28> kBaseScales = {
      1,  2,  4,  8,  16,  32,  64,  128, 256, 512,  1024, 2048, 4096,  10,
      20, 40, 50, 80, 100, 200, 400, 500, 800, 1000, 2000, 5000, 10000, 100000};

  ArrangementGraph out;
  if (forced_exact_scale > 0) {
    if (TryQuantizeOctilinearAtScale(lines, forced_exact_scale,
                                     kForcedScaleQuantizeTol, &out) ==
        QuantizeScaleOutcome::kBuiltExact) {
      return out;
    }
    return std::nullopt;
  }
  for (int64_t base_scale : kBaseScales) {
    // Multiply by 4 so unfolded diagonal crossings stay even integers and
    // folded D_4 diagonal crossings stay exact integers.
    QuantizeScaleOutcome status = TryQuantizeOctilinearAtScale(
        lines, 4 * base_scale, kAutoScaleQuantizeTol, &out);
    if (status == QuantizeScaleOutcome::kBuiltExact) {
      return out;
    }
    if (status == QuantizeScaleOutcome::kQuantizedButDeferred) {
      // Coordinates quantized cleanly to exact octilinear integers, so higher
      // scales produce the identical graph topology; defer to general sweep.
      return std::nullopt;
    }
  }
  return std::nullopt;
}

// ============================================================================
// General `double` Sweep-Line Planar Arrangement (`create_graph_from_lines`)
// ============================================================================

int ComparePointsEps(const std::pair<double, double>& p1,
                     const std::pair<double, double>& p2, double eps) {
  if (std::abs(p1.second - p2.second) > eps) {
    return (p1.second < p2.second) ? -1 : 1;
  }
  if (std::abs(p1.first - p2.first) > eps) {
    return (p1.first < p2.first) ? -1 : 1;
  }
  return 0;
}

std::optional<std::pair<double, double>> IntersectLinesEps(
    const std::pair<double, double>& p1, const std::pair<double, double>& p2,
    const std::pair<double, double>& p3, const std::pair<double, double>& p4,
    double eps) {
  double dx12 = p1.first - p2.first;
  double dy12 = p1.second - p2.second;
  double dx34 = p3.first - p4.first;
  double dy34 = p3.second - p4.second;
  double denom = dx12 * dy34 - dx34 * dy12;
  if (std::abs(denom) < eps * eps) return std::nullopt;
  double c12 = p1.first * p2.second - p1.second * p2.first;
  double c34 = p3.first * p4.second - p3.second * p4.first;
  return std::make_pair((c12 * dx34 - c34 * dx12) / denom,
                        (c12 * dy34 - c34 * dy12) / denom);
}

enum class SweepRunStatus : uint8_t {
  kOk = 0,
  kNumericallyUnstable = 1,
  kEventBudgetExceeded = 2,
};

struct SweepRunResult {
  SweepRunStatus status = SweepRunStatus::kNumericallyUnstable;
  ArrangementGraph graph{};
};

// Derived upper bound on Bentley-Ottmann sweep events for `n` segments:
//   - Initialization inserts at most `2n` segment endpoints.
//   - In a numerically consistent monotonic sweep, every sub-segment of input
//     line `i` lies on the same supporting line `L_i`; `n` supporting lines
//     have at most `n * (n - 1) / 2` distinct pairwise intersection points,
//     each strictly ahead of the current sweep vertex and deduplicated in
//     `vertex_queue_`, so each point is popped at most once.
// Total events popped in any correct run <= `2n + n * (n - 1) / 2` (no slack).
int64_t DerivedMaxSweepEvents(int64_t num_lines) {
  return 2 * num_lines + (num_lines * (num_lines - 1)) / 2;
}

class GeneralSweepRunner {
 public:
  GeneralSweepRunner(absl::Span<const InputSegment> lines, double eps)
      : lines_(lines), eps_(eps) {}

  SweepRunResult Run() {
    // Slot `kSweepSentinelVertexIdx == 0` holds the `(-inf, -inf)` sentinel
    // vertex; slot `kSweepProbeSegmentIdx == 0` holds the mutable probe segment
    // used when binary-searching `sweep_line_` at `cur_sweep_vertex_`.
    vertices_.push_back({-std::numeric_limits<double>::infinity(),
                         -std::numeric_limits<double>::infinity()});
    vertex_lines_.push_back({});
    segment_endpoints_.push_back({-1, -1});
    segment_unit_vectors_.push_back({-1.0, 0.0});
    segment_angles_.push_back(std::numeric_limits<double>::infinity());
    segment_distances_.push_back(0.0);
    segment_lines_.push_back({});

    for (int i = 0; i < static_cast<int>(lines_.size()); ++i) {
      std::pair<double, double> p1{lines_[i].x1, lines_[i].y1};
      std::pair<double, double> p2{lines_[i].x2, lines_[i].y2};
      if (ComparePointsEps(p1, p2, eps_) > 0) std::swap(p1, p2);
      int v1 = InsertOrFindEndpointVertex(p1);
      int v2 = InsertOrFindEndpointVertex(p2);
      line_vertices_.push_back({v1, v2});
      double dx = vertices_[v2].first - vertices_[v1].first;
      double dy = vertices_[v2].second - vertices_[v1].second;
      double mag = std::sqrt(dx * dx + dy * dy);
      double inv_mag = (mag != 0.0) ? (1.0 / mag) : 0.0;
      std::pair<double, double> uvec =
          (mag != 0.0) ? std::make_pair(dx * inv_mag, dy * inv_mag)
                       : std::make_pair(0.0, 0.0);
      line_unit_vectors_.push_back(uvec);
      line_angles_.push_back((dy < eps_) ? 0.0 : MathAngle0To2Pi(dx, dy));
      line_distances_.push_back(uvec.second * vertices_[v2].first +
                                (-uvec.first) * vertices_[v2].second);
      vertex_lines_[v1].push_back(i);
    }
    vert_to_inter_map_.assign(vertices_.size(), -1);

    const int64_t num_lines = static_cast<int64_t>(lines_.size());
    const int64_t max_events = DerivedMaxSweepEvents(num_lines);
    int64_t events_processed = 0;

    while (!vertex_queue_.empty()) {
      if (events_processed >= max_events) {
        return SweepRunResult{.status = SweepRunStatus::kEventBudgetExceeded};
      }
      ++events_processed;
      int v_idx = RemoveNextVertex(kSweepSentinelVertexIdx);
      if (v_idx < 0) return SweepRunResult{};
      cur_sweep_vertex_ = v_idx;
      auto vertex = vertices_[v_idx];

      std::optional<std::vector<int>> segs_opt = GetSegmentsAtVertex(v_idx);
      if (!segs_opt.has_value()) return SweepRunResult{};
      const std::vector<int>& segs_at_v = *segs_opt;

      if (vertex_lines_[v_idx].empty() && segs_at_v.size() < 2) {
        if (segs_at_v.empty()) continue;
        int seg_idx = segs_at_v[0];
        bool is_ep = false;
        for (int li : segment_lines_[seg_idx]) {
          if (ComparePointsEps(vertices_[line_vertices_[li].second], vertex,
                               eps_) <= 0) {
            is_ep = true;
            break;
          }
        }
        if (!is_ep && !InsertSweepSegment(seg_idx).has_value()) {
          return SweepRunResult{};
        }
        continue;
      }

      if (segs_at_v.size() == 1) {
        int seg0 = segs_at_v[0];
        bool all_on_line = true;
        for (int li : vertex_lines_[v_idx]) {
          if (!IsOnLine(line_vertices_[li].second, seg0)) {
            all_on_line = false;
            break;
          }
        }
        if (all_on_line) {
          if (!InsertSweepSegment(seg0).has_value()) return SweepRunResult{};
          // Pigeonhole bound: each of the `n` input lines can pass through one
          // active sweep segment at most once in a consistent run.
          if (segment_lines_[seg0].size() + vertex_lines_[v_idx].size() >
              lines_.size()) {
            return SweepRunResult{.status =
                                      SweepRunStatus::kEventBudgetExceeded};
          }
          segment_lines_[seg0].insert(segment_lines_[seg0].end(),
                                      vertex_lines_[v_idx].begin(),
                                      vertex_lines_[v_idx].end());
          continue;
        }
      }

      if (v_idx >= static_cast<int>(vert_to_inter_map_.size())) {
        vert_to_inter_map_.resize(v_idx + 1, -1);
      }
      vert_to_inter_map_[v_idx] = static_cast<int>(intersection_points_.size());
      intersection_points_.push_back(vertex);

      for (int seg_i : segs_at_v) {
        segment_endpoints_[seg_i].second = v_idx;
        for (int li : segment_lines_[seg_i]) {
          if (ComparePointsEps(vertices_[line_vertices_[li].second], vertex,
                               eps_) > 0) {
            vertex_lines_[v_idx].push_back(li);
            if (vertex_lines_[v_idx].size() > lines_.size()) {
              return SweepRunResult{.status =
                                        SweepRunStatus::kEventBudgetExceeded};
            }
          }
        }
      }

      auto& vlines = vertex_lines_[v_idx];
      absl::c_stable_sort(vlines, [&](int a, int b) {
        auto [ax, ay] = vertices_[line_vertices_[a].second];
        auto [bx, by] = vertices_[line_vertices_[b].second];
        double dax = ax - vertex.first;
        double day = ay - vertex.second;
        double dbx = bx - vertex.first;
        double dby = by - vertex.second;
        return (dax * dax + day * day) < (dbx * dbx + dby * dby);
      });

      for (int li : vlines) {
        int seg_idx = static_cast<int>(segment_endpoints_.size());
        segment_endpoints_.push_back({v_idx, line_vertices_[li].second});
        segment_unit_vectors_.push_back(line_unit_vectors_[li]);
        segment_angles_.push_back(line_angles_[li]);
        segment_distances_.push_back(line_distances_[li]);
        segment_lines_.push_back({li});
        std::optional<int> existing = InsertSweepSegment(seg_idx);
        if (!existing.has_value()) return SweepRunResult{};
        if (*existing >= 0) {
          segment_endpoints_.pop_back();
          segment_unit_vectors_.pop_back();
          segment_angles_.pop_back();
          segment_distances_.pop_back();
          segment_lines_.pop_back();
          if (segment_lines_[*existing].size() >= lines_.size()) {
            return SweepRunResult{.status =
                                      SweepRunStatus::kEventBudgetExceeded};
          }
          segment_lines_[*existing].push_back(li);
        } else {
          segment_endpoints_[seg_idx].second = -1;
        }
      }

      if (!FindAndAddIntersections(vertex)) return SweepRunResult{};
    }

    std::optional<ArrangementGraph> built = BuildFinalEdges();
    if (!built.has_value()) return SweepRunResult{};
    return SweepRunResult{.status = SweepRunStatus::kOk,
                          .graph = std::move(*built)};
  }

 private:
  int BisectLeftVertex(int v_idx) const {
    int lo = 0;
    int hi = static_cast<int>(vertex_queue_.size());
    while (lo < hi) {
      int mid = (lo + hi) / 2;
      if (ComparePointsEps(vertices_[v_idx], vertices_[vertex_queue_[mid]],
                           eps_) > 0) {
        lo = mid + 1;
      } else {
        hi = mid;
      }
    }
    return lo;
  }

  int BisectRightVertex(int v_idx) const {
    int lo = 0;
    int hi = static_cast<int>(vertex_queue_.size());
    while (lo < hi) {
      int mid = (lo + hi) / 2;
      if (ComparePointsEps(vertices_[v_idx], vertices_[vertex_queue_[mid]],
                           eps_) < 0) {
        hi = mid;
      } else {
        lo = mid + 1;
      }
    }
    return lo;
  }

  int InsertVertexQueue(int v_idx) {
    int idx = BisectLeftVertex(v_idx);
    if (idx < static_cast<int>(vertex_queue_.size()) &&
        ComparePointsEps(vertices_[v_idx], vertices_[vertex_queue_[idx]],
                         eps_) == 0) {
      return vertex_queue_[idx];
    }
    vertex_queue_.insert(vertex_queue_.begin() + idx, v_idx);
    return -1;
  }

  int RemoveNextVertex(int v_idx) {
    int idx = BisectRightVertex(v_idx);
    if (idx < static_cast<int>(vertex_queue_.size())) {
      int val = vertex_queue_[idx];
      vertex_queue_.erase(vertex_queue_.begin() + idx);
      return val;
    }
    return -1;
  }

  int InsertOrFindEndpointVertex(const std::pair<double, double>& pt) {
    int new_idx = static_cast<int>(vertices_.size());
    vertices_.push_back(pt);
    vertex_lines_.push_back({});
    int existing = InsertVertexQueue(new_idx);
    if (existing < 0) {
      return new_idx;
    }
    vertices_.pop_back();
    vertex_lines_.pop_back();
    return existing;
  }

  void AddIntersectionToQueue(
      const std::optional<std::pair<double, double>>& inter,
      const std::pair<double, double>& vertex) {
    if (!inter.has_value()) return;
    int cmp = ComparePointsEps(*inter, vertex, eps_);
    if (cmp == 0 || (cmp < 0 && inter->second <= vertex.second)) {
      return;
    }
    int new_idx = static_cast<int>(vertices_.size());
    vertices_.push_back(*inter);
    vertex_lines_.push_back({});
    vert_to_inter_map_.push_back(-1);
    int existing = InsertVertexQueue(new_idx);
    if (existing >= 0) {
      vertices_.pop_back();
      vertex_lines_.pop_back();
      vert_to_inter_map_.pop_back();
    }
  }

  double PointSegmentDist(int v_idx, int seg_idx) const {
    const auto& uvec = segment_unit_vectors_[seg_idx];
    const auto& pt = vertices_[v_idx];
    return segment_distances_[seg_idx] -
           (uvec.second * pt.first + (-uvec.first) * pt.second);
  }

  bool IsOnLine(int v_idx, int seg_idx) const {
    return std::abs(PointSegmentDist(v_idx, seg_idx)) < eps_;
  }

  std::optional<int> CompareSegments(int s1, int s2) const {
    if (!IsOnLine(cur_sweep_vertex_, s1)) return std::nullopt;
    double d2 = PointSegmentDist(cur_sweep_vertex_, s2);
    if (std::abs(d2) < eps_) {
      int ep1 = segment_endpoints_[s1].second;
      if (ep1 >= 0 && IsOnLine(ep1, s2)) {
        return 0;
      }
      return (segment_angles_[s2] - segment_angles_[s1] > 0.0) ? 1 : -1;
    }
    return (-d2 > 0.0) ? 1 : -1;
  }

  std::optional<int> BisectLeftSweep(int x) const {
    int lo = 0;
    int hi = static_cast<int>(sweep_line_.size());
    while (lo < hi) {
      int mid = (lo + hi) / 2;
      std::optional<int> cmp = CompareSegments(x, sweep_line_[mid]);
      if (!cmp.has_value()) return std::nullopt;
      if (*cmp > 0) {
        lo = mid + 1;
      } else {
        hi = mid;
      }
    }
    return lo;
  }

  std::optional<int> BisectRightSweep(int x) const {
    int lo = 0;
    int hi = static_cast<int>(sweep_line_.size());
    while (lo < hi) {
      int mid = (lo + hi) / 2;
      std::optional<int> cmp = CompareSegments(x, sweep_line_[mid]);
      if (!cmp.has_value()) return std::nullopt;
      if (*cmp < 0) {
        hi = mid;
      } else {
        lo = mid + 1;
      }
    }
    return lo;
  }

  // Inserts `seg_idx` into `sweep_line_`, returning `-1` if newly inserted, the
  // index of an existing collinear segment if merged, or `std::nullopt` on
  // numerical instability.
  std::optional<int> InsertSweepSegment(int seg_idx) {
    std::optional<int> idx = BisectLeftSweep(seg_idx);
    if (!idx.has_value()) return std::nullopt;
    if (*idx < static_cast<int>(sweep_line_.size())) {
      std::optional<int> cmp = CompareSegments(seg_idx, sweep_line_[*idx]);
      if (!cmp.has_value()) return std::nullopt;
      if (*cmp == 0) {
        return sweep_line_[*idx];
      }
    }
    sweep_line_.insert(sweep_line_.begin() + *idx, seg_idx);
    return -1;
  }

  std::optional<int> PrevSweep(int x) const {
    std::optional<int> idx = BisectLeftSweep(x);
    if (!idx.has_value()) return std::nullopt;
    return (*idx > 0) ? sweep_line_[*idx - 1] : -1;
  }

  std::optional<int> NextSweep(int x) const {
    std::optional<int> idx = BisectRightSweep(x);
    if (!idx.has_value()) return std::nullopt;
    return (*idx < static_cast<int>(sweep_line_.size())) ? sweep_line_[*idx]
                                                         : -1;
  }

  std::optional<int> RemovePrevSweep(int x) {
    std::optional<int> idx = BisectLeftSweep(x);
    if (!idx.has_value()) return std::nullopt;
    if (*idx > 0) {
      int removed = sweep_line_[*idx - 1];
      sweep_line_.erase(sweep_line_.begin() + *idx - 1);
      return removed;
    }
    return -1;
  }

  std::optional<int> RemoveNextSweep(int x) {
    std::optional<int> idx = BisectRightSweep(x);
    if (!idx.has_value()) return std::nullopt;
    if (*idx < static_cast<int>(sweep_line_.size())) {
      int removed = sweep_line_[*idx];
      sweep_line_.erase(sweep_line_.begin() + *idx);
      return removed;
    }
    return -1;
  }

  std::optional<std::vector<int>> GetSegmentsAtVertex(int v_idx) {
    std::vector<int> out;
    segment_endpoints_[kSweepProbeSegmentIdx].first = v_idx;
    const auto& u0 = segment_unit_vectors_[kSweepProbeSegmentIdx];
    segment_distances_[kSweepProbeSegmentIdx] =
        u0.second * vertices_[v_idx].first +
        (-u0.first) * vertices_[v_idx].second;
    std::optional<int> prev_seg = PrevSweep(kSweepProbeSegmentIdx);
    if (!prev_seg.has_value()) return std::nullopt;
    if (*prev_seg >= 0 && segment_angles_[*prev_seg] == 0.0) {
      std::optional<int> removed = RemovePrevSweep(kSweepProbeSegmentIdx);
      if (!removed.has_value()) return std::nullopt;
      if (*removed >= 0) out.push_back(*removed);
    }
    while (true) {
      std::optional<int> next_seg = NextSweep(kSweepProbeSegmentIdx);
      if (!next_seg.has_value()) return std::nullopt;
      if (*next_seg < 0 || !IsOnLine(v_idx, *next_seg)) break;
      std::optional<int> removed = RemoveNextSweep(kSweepProbeSegmentIdx);
      if (!removed.has_value()) return std::nullopt;
      if (*removed >= 0) out.push_back(*removed);
    }
    return out;
  }

  bool FindAndAddIntersections(const std::pair<double, double>& vertex) {
    std::optional<int> l1 = PrevSweep(kSweepProbeSegmentIdx);
    std::optional<int> r1 = NextSweep(kSweepProbeSegmentIdx);
    if (!l1.has_value() || !r1.has_value()) return false;
    segment_angles_[kSweepProbeSegmentIdx] =
        -std::numeric_limits<double>::infinity();
    std::optional<int> l2 = PrevSweep(kSweepProbeSegmentIdx);
    std::optional<int> r2 = NextSweep(kSweepProbeSegmentIdx);
    segment_angles_[kSweepProbeSegmentIdx] =
        std::numeric_limits<double>::infinity();
    if (!l2.has_value() || !r2.has_value()) return false;

    const std::array<std::pair<int, int>, 2> pairs = {{{*l1, *r1}, {*l2, *r2}}};
    for (auto [left_i, right_i] : pairs) {
      if (left_i < 0 || right_i < 0) continue;
      int p1_idx = segment_endpoints_[left_i].first;
      int p2_idx = segment_endpoints_[right_i].first;
      if (p1_idx < 0 || p2_idx < 0) return false;
      auto p1 = vertices_[p1_idx];
      auto u1 = segment_unit_vectors_[left_i];
      auto p1_end = std::make_pair(p1.first + u1.first, p1.second + u1.second);
      auto p2 = vertices_[p2_idx];
      auto u2 = segment_unit_vectors_[right_i];
      auto p2_end = std::make_pair(p2.first + u2.first, p2.second + u2.second);
      auto inter = IntersectLinesEps(p1, p1_end, p2, p2_end, eps_);
      AddIntersectionToQueue(inter, vertex);
    }
    return true;
  }

  std::optional<ArrangementGraph> BuildFinalEdges() {
    absl::flat_hash_map<uint64_t, int> edge_map;
    std::vector<std::pair<std::pair<int, int>, std::vector<int>>> edges;
    for (int s = 1; s < static_cast<int>(segment_endpoints_.size()); ++s) {
      auto [va, vb] = segment_endpoints_[s];
      if (va < 0 || vb < 0 ||
          va >= static_cast<int>(vert_to_inter_map_.size()) ||
          vb >= static_cast<int>(vert_to_inter_map_.size())) {
        return std::nullopt;
      }
      int p1 = vert_to_inter_map_[va];
      int p2 = vert_to_inter_map_[vb];
      if (p1 < 0 || p2 < 0 || p1 == p2) return std::nullopt;
      if (p2 < p1) std::swap(p1, p2);
      uint64_t key = EncodeOrderPair(p1, p2);
      auto it = edge_map.find(key);
      if (it == edge_map.end()) {
        int idx = static_cast<int>(edges.size());
        edge_map[key] = idx;
        edges.push_back({{p1, p2}, segment_lines_[s]});
      } else {
        auto& dst = edges[it->second].second;
        dst.insert(dst.end(), segment_lines_[s].begin(),
                   segment_lines_[s].end());
      }
    }
    absl::c_sort(
        edges, [](const auto& a, const auto& b) { return a.first < b.first; });
    ArrangementGraph out;
    out.vertices = intersection_points_;
    out.edge_vertices.reserve(edges.size());
    out.edge_lines.reserve(edges.size());
    for (auto& [ev, el] : edges) {
      SortAndUnique(&el);
      out.edge_vertices.push_back(ev);
      out.edge_lines.push_back(std::move(el));
    }
    return out;
  }

  absl::Span<const InputSegment> lines_;
  double eps_;
  int cur_sweep_vertex_ = 0;
  std::vector<std::pair<double, double>> vertices_;
  std::vector<std::vector<int>> vertex_lines_;
  std::vector<std::pair<int, int>> line_vertices_;
  std::vector<std::pair<double, double>> line_unit_vectors_;
  std::vector<double> line_angles_;
  std::vector<double> line_distances_;
  std::vector<int> vertex_queue_;

  std::vector<std::pair<int, int>> segment_endpoints_;
  std::vector<std::pair<double, double>> segment_unit_vectors_;
  std::vector<double> segment_angles_;
  std::vector<double> segment_distances_;
  std::vector<std::vector<int>> segment_lines_;
  std::vector<int> sweep_line_;

  std::vector<int> vert_to_inter_map_;
  std::vector<std::pair<double, double>> intersection_points_;
};

absl::StatusOr<ArrangementGraph> BuildGeneralArrangement(
    absl::Span<const InputSegment> lines, std::string_view stage_label) {
  ArrangementGraph out;
  if (lines.empty()) {
    return absl::InvalidArgumentError("IO Parser failed");
  }
  for (const auto& ln : lines) {
    if (!std::isfinite(ln.x1) || !std::isfinite(ln.y1) ||
        !std::isfinite(ln.x2) || !std::isfinite(ln.y2)) {
      return absl::InvalidArgumentError("Non-finite input segment coordinate");
    }
  }
  for (const auto& ln : lines) {
    if (!IsValidCoordinate(ln.x1) || !IsValidCoordinate(ln.y1) ||
        !IsValidCoordinate(ln.x2) || !IsValidCoordinate(ln.y2)) {
      return absl::InvalidArgumentError(
          "Input segment coordinate exceeds 1e100 magnitude bound");
    }
  }

  double min_len_sq = std::numeric_limits<double>::infinity();
  for (const auto& ln : lines) {
    double dx = ln.x2 - ln.x1;
    double dy = ln.y2 - ln.y1;
    min_len_sq = std::min(min_len_sq, dx * dx + dy * dy);
  }
  double min_len = (std::isfinite(min_len_sq) && min_len_sq > 0.0)
                       ? std::sqrt(min_len_sq)
                       : 1.0;

  size_t prev_v = 0;
  size_t prev_e = 0;
  int stability = 0;
  int max_stability = 1;
  bool any_event_budget_exceeded = false;
  for (int exp_i = 0; exp_i < kMaxGeneralSweepExpIterations; ++exp_i) {
    int exponent = exp_i + 3;
    double eps = std::ldexp(min_len, -exponent);
    if (eps < kFloatEpsilon) break;

    GeneralSweepRunner runner(lines, eps);
    SweepRunResult cand = runner.Run();
    if (cand.status != SweepRunStatus::kOk) {
      if (cand.status == SweepRunStatus::kEventBudgetExceeded) {
        any_event_budget_exceeded = true;
      }
      prev_v = 0;
      prev_e = 0;
      stability = 0;
      continue;
    }
    if (cand.graph.vertices.size() == prev_v &&
        cand.graph.edge_vertices.size() == prev_e) {
      ++stability;
    } else {
      prev_v = cand.graph.vertices.size();
      prev_e = cand.graph.edge_vertices.size();
      stability = 1;
    }
    if (stability >= max_stability) {
      out = std::move(cand.graph);
      max_stability = stability;
    }
    if (stability == kTargetSweepStability) break;
  }
  if (out.vertices.empty() ||
      (any_event_budget_exceeded && max_stability < kTargetSweepStability)) {
    if (any_event_budget_exceeded) {
      const int64_t num_lines = static_cast<int64_t>(lines.size());
      return absl::ResourceExhaustedError(absl::StrFormat(
          "Sweep event limit exceeded (%s arrangement: exceeded %d events for "
          "%d segments)",
          stage_label, DerivedMaxSweepEvents(num_lines), num_lines));
    }
    return absl::AbortedError("Precision Error: unstable graph");
  }
  return out;
}

// ============================================================================
// Graph Topology (`graph_to_adjacency_and_faces` & `build_edge_face_maps`)
// ============================================================================

void BuildAdjacencyAndFaces(
    absl::Span<const std::pair<double, double>> vertices,
    absl::Span<const std::pair<int, int>> edge_vertices,
    std::vector<std::vector<int>>* out_adj,
    std::vector<std::vector<int>>* out_faces) {
  const int num_v = static_cast<int>(vertices.size());
  std::vector<std::vector<int>> raw_adj(num_v);
  for (auto [u, v] : edge_vertices) {
    if (u >= 0 && u < num_v && v >= 0 && v < num_v && u != v) {
      raw_adj[u].push_back(v);
      raw_adj[v].push_back(u);
    }
  }

  out_adj->assign(num_v, {});
  for (int v = 0; v < num_v; ++v) {
    std::vector<std::pair<double, int>> ang_neighbors;
    ang_neighbors.reserve(raw_adj[v].size());
    for (int nb : raw_adj[v]) {
      double dx = vertices[nb].first - vertices[v].first;
      double dy = vertices[nb].second - vertices[v].second;
      ang_neighbors.push_back({SafeAngle0To2Pi(dx, dy), nb});
    }
    absl::c_stable_sort(ang_neighbors, [](const auto& a, const auto& b) {
      return a.first < b.first;
    });
    (*out_adj)[v].reserve(ang_neighbors.size());
    for (const auto& [_, nb] : ang_neighbors) {
      (*out_adj)[v].push_back(nb);
    }
  }

  absl::flat_hash_set<uint64_t> seen_half_edges;
  seen_half_edges.reserve(edge_vertices.size() * 4 + 8);
  out_faces->clear();

  for (int v1 = 0; v1 < num_v; ++v1) {
    for (int v2 : (*out_adj)[v1]) {
      uint64_t he_key = EncodeDirectedPair(v1, v2);
      if (!seen_half_edges.insert(he_key).second) continue;

      std::vector<int> face = {v1};
      int cur_v = v1;
      int next_v = v2;
      int guard = 0;
      // On malformed graphs with dangling degree-1 bridges, a half-edge walk
      // can traverse a bridge twice without closing a simple polygon; bound
      // the walk by total directed half-edges (`2 * |E| + 4`).
      const int max_steps = static_cast<int>(edge_vertices.size()) * 2 + 4;
      while (next_v != v1 && guard++ < max_steps) {
        face.push_back(next_v);
        const auto& nb_list = (*out_adj)[next_v];
        int prev_nb = -1;
        for (int i = 0; i < static_cast<int>(nb_list.size()); ++i) {
          if (nb_list[i] == cur_v) {
            prev_nb = (i == 0) ? nb_list.back() : nb_list[i - 1];
            break;
          }
        }
        if (prev_nb < 0) {
          face.clear();
          break;
        }
        seen_half_edges.insert(EncodeDirectedPair(next_v, prev_nb));
        cur_v = next_v;
        next_v = prev_nb;
      }
      if (face.size() > 2 && next_v == v1) {
        out_faces->push_back(std::move(face));
      }
    }
  }

  // Sort faces by signed area on the `double` vertex coordinates, matching
  // `math_utils.sort_faces`, and pop the clockwise outer boundary face.
  std::vector<std::pair<double, int>> areas(out_faces->size());
  for (int i = 0; i < static_cast<int>(out_faces->size()); ++i) {
    areas[i] = {PolygonTwiceSignedAreaDouble((*out_faces)[i], vertices), i};
  }
  absl::c_stable_sort(
      areas, [](const auto& a, const auto& b) { return a.first > b.first; });
  std::vector<std::vector<int>> sorted_faces;
  sorted_faces.reserve(out_faces->size());
  for (const auto& [_, idx] : areas) {
    sorted_faces.push_back(std::move((*out_faces)[idx]));
  }
  *out_faces = std::move(sorted_faces);

  if (!out_faces->empty()) {
    out_faces->pop_back();
  }
}

void BuildFacesToAdjacency(absl::Span<const std::pair<double, double>> vertices,
                           absl::Span<const std::vector<int>> face_vertices,
                           std::vector<std::vector<int>>* out_adj) {
  const int num_v = static_cast<int>(vertices.size());
  std::vector<absl::flat_hash_map<int, int>> next_nb(num_v);
  std::vector<absl::flat_hash_map<int, int>> prev_nb(num_v);
  std::vector<std::vector<int>> order_inserted(num_v);
  for (const auto& face : face_vertices) {
    if (face.size() < 2) continue;
    int v_prev = face[face.size() - 2];
    int v_curr = face.back();
    for (int v_next : face) {
      ABSL_DCHECK_GE(v_curr, 0);
      ABSL_DCHECK_LT(v_curr, num_v);
      if (!next_nb[v_curr].contains(v_prev)) {
        order_inserted[v_curr].push_back(v_prev);
      }
      next_nb[v_curr][v_prev] = v_next;
      prev_nb[v_curr][v_next] = v_prev;
      v_prev = v_curr;
      v_curr = v_next;
    }
  }
  out_adj->assign(num_v, {});
  for (int v = 0; v < num_v; ++v) {
    if (order_inserted[v].empty()) continue;
    int start_cand = order_inserted[v].front();
    int cur = start_cand;
    auto it_prev = prev_nb[v].find(start_cand);
    int guard = 0;
    while (it_prev != prev_nb[v].end() && it_prev->second != start_cand &&
           guard++ <= num_v) {
      cur = it_prev->second;
      it_prev = prev_nb[v].find(cur);
    }
    int start = cur;
    (*out_adj)[v].push_back(start);
    auto it_next = next_nb[v].find(cur);
    guard = 0;
    while (it_next != next_nb[v].end() && it_next->second != start &&
           guard++ <= num_v) {
      cur = it_next->second;
      (*out_adj)[v].push_back(cur);
      it_next = next_nb[v].find(cur);
    }
  }
}

void BuildEdgeFaceMaps(absl::Span<const std::pair<int, int>> edge_vertices,
                       absl::Span<const std::vector<int>> face_vertices,
                       std::vector<std::vector<int>>* out_edge_faces,
                       std::vector<std::vector<int>>* out_face_edges) {
  absl::flat_hash_map<uint64_t, int> edge_idx_map;
  edge_idx_map.reserve(edge_vertices.size() * 2 + 4);
  for (int i = 0; i < static_cast<int>(edge_vertices.size()); ++i) {
    edge_idx_map[EncodeOrderPair(edge_vertices[i].first,
                                 edge_vertices[i].second)] = i;
  }

  std::vector<std::array<int, 2>> pairs(edge_vertices.size(), {-1, -1});
  for (int fi = 0; fi < static_cast<int>(face_vertices.size()); ++fi) {
    const auto& face = face_vertices[fi];
    const int n = static_cast<int>(face.size());
    for (int j = 0; j < n; ++j) {
      int v1 = face[j];
      int v2 = face[(j + 1) % n];
      auto it = edge_idx_map.find(EncodeOrderPair(v1, v2));
      if (it != edge_idx_map.end()) {
        int slot = (v2 < v1) ? 0 : 1;
        pairs[it->second][slot] = fi;
      }
    }
  }

  out_edge_faces->assign(edge_vertices.size(), {});
  for (size_t i = 0; i < edge_vertices.size(); ++i) {
    if (pairs[i][0] >= 0) (*out_edge_faces)[i].push_back(pairs[i][0]);
    if (pairs[i][1] >= 0) (*out_edge_faces)[i].push_back(pairs[i][1]);
  }

  if (out_face_edges != nullptr) {
    out_face_edges->assign(face_vertices.size(), {});
    for (size_t fi = 0; fi < face_vertices.size(); ++fi) {
      const auto& face = face_vertices[fi];
      if (face.empty()) continue;
      int v1 = face.back();
      (*out_face_edges)[fi].reserve(face.size());
      for (int v2 : face) {
        auto it = edge_idx_map.find(EncodeOrderPair(v1, v2));
        (*out_face_edges)[fi].push_back((it != edge_idx_map.end()) ? it->second
                                                                   : -1);
        v1 = v2;
      }
    }
  }
}

absl::Status FinalizeParsedMesh(FaceOrientationPolicy orientation_policy,
                                bool has_explicit_faces, UnfoldedMesh* mesh) {
  const bool flip_to_top_face_up =
      (orientation_policy == FaceOrientationPolicy::kFlipToTopFaceUp);
  if (!has_explicit_faces) {
    if (flip_to_top_face_up) {
      for (CreaseAssignment& c : mesh->edge_assignments) {
        c = FlipMountainValley(c);
      }
    } else {
      // Matches `io_utils._flip_y`: reflects `y` across `[0, 1]` unit square.
      for (auto& [x, y] : mesh->vertex_coords) {
        y = -y + 1.0;
      }
      if (mesh->used_exact_octilinear) {
        for (auto& [x, y] : mesh->exact_vertex_coords) {
          y = -y + mesh->exact_scale;
        }
      }
    }
    BuildAdjacencyAndFaces(mesh->vertex_coords, mesh->edge_vertices,
                           &mesh->vertex_adjacencies, &mesh->face_vertices);
  } else {
    if (!mesh->face_vertices.empty() &&
        PolygonTwiceSignedAreaDouble(mesh->face_vertices[0],
                                     mesh->vertex_coords) < 0.0) {
      for (CreaseAssignment& c : mesh->edge_assignments) {
        c = FlipMountainValley(c);
      }
      for (auto& f : mesh->face_vertices) {
        std::reverse(f.begin(), f.end());
      }
    }
    if (!flip_to_top_face_up) {
      for (CreaseAssignment& c : mesh->edge_assignments) {
        c = FlipMountainValley(c);
      }
      for (auto& f : mesh->face_vertices) {
        std::reverse(f.begin(), f.end());
      }
      // Matches `io_utils._flip_y`: reflects `y` across `[0, 1]` unit square.
      for (auto& [x, y] : mesh->vertex_coords) {
        y = -y + 1.0;
      }
      mesh->used_exact_octilinear = false;
      mesh->exact_vertex_coords.clear();
    }
    BuildFacesToAdjacency(mesh->vertex_coords, mesh->face_vertices,
                          &mesh->vertex_adjacencies);
  }

  std::vector<std::vector<int>> face_edges;
  BuildEdgeFaceMaps(mesh->edge_vertices, mesh->face_vertices, &mesh->edge_faces,
                    &face_edges);

  if (mesh->face_vertices.size() > 1) {
    std::vector<uint8_t> is_hole(mesh->face_vertices.size(), 0);
    bool any_hole = false;
    for (size_t fi = 0; fi < mesh->face_vertices.size(); ++fi) {
      bool all_boundary = true;
      for (int ei : face_edges[fi]) {
        if (ei >= 0 &&
            mesh->edge_assignments[ei] != CreaseAssignment::kBoundary) {
          all_boundary = false;
          break;
        }
      }
      if (all_boundary) {
        is_hole[fi] = 1;
        any_hole = true;
      }
    }
    if (any_hole) {
      std::vector<std::vector<int>> kept_faces;
      kept_faces.reserve(mesh->face_vertices.size());
      for (size_t fi = 0; fi < mesh->face_vertices.size(); ++fi) {
        if (!is_hole[fi]) {
          kept_faces.push_back(std::move(mesh->face_vertices[fi]));
        }
      }
      mesh->face_vertices = std::move(kept_faces);
      BuildEdgeFaceMaps(mesh->edge_vertices, mesh->face_vertices,
                        &mesh->edge_faces, nullptr);
    }
  }

  for (size_t ei = 0; ei < mesh->edge_faces.size(); ++ei) {
    if (mesh->edge_faces[ei].size() == 1) {
      mesh->edge_assignments[ei] = CreaseAssignment::kBoundary;
    }
  }
  if (mesh->vertex_coords.empty() || mesh->face_vertices.empty()) {
    return absl::InvalidArgumentError("IO Parser failed");
  }
  if (mesh->face_vertices.size() > static_cast<size_t>(kMaxFaces)) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Face count %zu exceeds kMaxFaces (%d)",
                        mesh->face_vertices.size(), kMaxFaces));
  }
  return absl::OkStatus();
}

// ============================================================================
// Folded-State Helpers (`ComputeRigidFoldedVertices`, `BuildFoldedCellGraph`,
// `PropagateCellFaceContainment`, `ExtractTransitiveOverlappingFacePairs`)
// ============================================================================

struct RigidFoldedPlacement {
  std::vector<std::pair<double, double>> folded_vertices;
  std::vector<bool> face_orientations;
  std::vector<std::pair<int64_t, int64_t>> exact_folded;
  bool keep_exact = false;
};

struct RigidFoldedResult {
  FoldedArrangementOutcome outcome = FoldedArrangementOutcome::kCouldNotDecide;
  std::string error_message{};
  RigidFoldedPlacement placement{};
};

std::optional<std::pair<int64_t, int64_t>> TryPlaceExactFoldedVertex(
    const std::pair<int64_t, int64_t>& ep1,
    const std::pair<int64_t, int64_t>& ep2,
    const std::pair<int64_t, int64_t>& eq1,
    const std::pair<int64_t, int64_t>& eq2,
    const std::pair<int64_t, int64_t>& ev, bool is_face_up) {
  absl::int128 iux = absl::int128{ep2.first} - absl::int128{ep1.first};
  absl::int128 iuy = absl::int128{ep2.second} - absl::int128{ep1.second};
  absl::int128 ifx = absl::int128{eq2.first} - absl::int128{eq1.first};
  absl::int128 ify = absl::int128{eq2.second} - absl::int128{eq1.second};
  absl::int128 l2 = iux * iux + iuy * iuy;
  if (l2 <= 0 || l2 != ifx * ifx + ify * ify) return std::nullopt;
  absl::int128 ivx = absl::int128{ev.first} - absl::int128{ep1.first};
  absl::int128 ivy = absl::int128{ev.second} - absl::int128{ep1.second};
  absl::int128 dot_x = ivx * iux + ivy * iuy;
  absl::int128 dot_y = (ivx * iuy - ivy * iux) * (is_face_up ? 1 : -1);
  absl::int128 num_x = ifx * dot_x + ify * dot_y;
  absl::int128 num_y = ify * dot_x - ifx * dot_y;
  if (num_x % l2 != 0 || num_y % l2 != 0) return std::nullopt;
  absl::int128 out_x = absl::int128{eq1.first} + (num_x / l2);
  absl::int128 out_y = absl::int128{eq1.second} + (num_y / l2);
  constexpr absl::int128 kLimit = absl::int128{kFoldedCoordLimit};
  if (out_x < -kLimit || out_x > kLimit || out_y < -kLimit || out_y > kLimit) {
    return std::nullopt;
  }
  return std::make_pair(static_cast<int64_t>(out_x),
                        static_cast<int64_t>(out_y));
}

RigidFoldedResult ComputeRigidFoldedVertices(const UnfoldedMesh& mesh) {
  const int num_v = static_cast<int>(mesh.vertex_coords.size());
  const int num_f = static_cast<int>(mesh.face_vertices.size());
  if (num_v == 0 || num_f == 0 || mesh.face_vertices[0].size() < 2) {
    return RigidFoldedResult{
        .outcome = FoldedArrangementOutcome::kCouldNotDecide,
        .error_message = "Precision Error: unstable graph"};
  }

  absl::flat_hash_map<uint64_t, CreaseAssignment> assign_map;
  assign_map.reserve(mesh.edge_vertices.size() * 2 + 4);
  for (size_t ei = 0; ei < mesh.edge_vertices.size(); ++ei) {
    assign_map[EncodeOrderPair(mesh.edge_vertices[ei].first,
                               mesh.edge_vertices[ei].second)] =
        mesh.edge_assignments[ei];
  }

  absl::flat_hash_map<uint64_t, int> dir_edge_to_face;
  dir_edge_to_face.reserve(mesh.edge_vertices.size() * 4 + 8);
  for (int fi = 0; fi < num_f; ++fi) {
    const auto& face = mesh.face_vertices[fi];
    const int m = static_cast<int>(face.size());
    for (int j = 0; j < m; ++j) {
      dir_edge_to_face[EncodeDirectedPair(face[(j + 1) % m], face[j])] = fi;
    }
  }

  RigidFoldedPlacement out;
  std::vector<uint8_t> v_set(num_v, 0);
  out.folded_vertices.assign(num_v, {0.0, 0.0});
  out.exact_folded.assign(num_v, {0, 0});
  out.keep_exact = mesh.used_exact_octilinear &&
                   mesh.exact_vertex_coords.size() == mesh.vertex_coords.size();

  int v0_f0 = mesh.face_vertices[0][0];
  int v1_f0 = mesh.face_vertices[0][1];
  ABSL_DCHECK_GE(v0_f0, 0);
  ABSL_DCHECK_LT(v0_f0, num_v);
  ABSL_DCHECK_GE(v1_f0, 0);
  ABSL_DCHECK_LT(v1_f0, num_v);
  v_set[v0_f0] = 1;
  v_set[v1_f0] = 1;
  out.folded_vertices[v0_f0] = mesh.vertex_coords[v0_f0];
  out.folded_vertices[v1_f0] = mesh.vertex_coords[v1_f0];
  if (out.keep_exact) {
    out.exact_folded[v0_f0] = mesh.exact_vertex_coords[v0_f0];
    out.exact_folded[v1_f0] = mesh.exact_vertex_coords[v1_f0];
  }

  out.face_orientations.assign(num_f, false);
  std::vector<uint8_t> seen_faces(num_f, 0);
  seen_faces[0] = 1;
  // `out.face_orientations[0]` is `false` (`is_face_up == true`). Orientation
  // parity is stored at enqueue time so second-arrival checks see the first
  // arrival's parity even before that face is popped from `queue`.

  struct QueueEntry {
    int fi = 0;
    int p1_idx = 0;
    int p2_idx = 0;
    bool is_face_up = true;
  };
  std::vector<QueueEntry> queue = {QueueEntry{
      .fi = 0, .p1_idx = v0_f0, .p2_idx = v1_f0, .is_face_up = true}};

  for (size_t qi = 0; qi < queue.size(); ++qi) {
    auto [fi, p1_idx, p2_idx, is_face_up] = queue[qi];
    if (!v_set[p1_idx] || !v_set[p2_idx]) {
      return RigidFoldedResult{
          .outcome = FoldedArrangementOutcome::kCouldNotDecide,
          .error_message = "Folded vertex for face cannot be None."};
    }

    auto [p1x, p1y] = mesh.vertex_coords[p1_idx];
    auto [p2x, p2y] = mesh.vertex_coords[p2_idx];
    double ux = p2x - p1x;
    double uy = p2y - p1y;
    double umag = std::sqrt(ux * ux + uy * uy);
    double inv_umag = (umag != 0.0) ? (1.0 / umag) : 0.0;
    double xax_x = (umag != 0.0) ? (ux * inv_umag) : 0.0;
    double xax_y = (umag != 0.0) ? (uy * inv_umag) : 0.0;

    auto [q1x, q1y] = out.folded_vertices[p1_idx];
    auto [q2x, q2y] = out.folded_vertices[p2_idx];
    double fx = q2x - q1x;
    double fy = q2y - q1y;
    double fmag = std::sqrt(fx * fx + fy * fy);
    double inv_fmag = (fmag != 0.0) ? (1.0 / fmag) : 0.0;
    double xaf_x = (fmag != 0.0) ? (fx * inv_fmag) : 0.0;
    double xaf_y = (fmag != 0.0) ? (fy * inv_fmag) : 0.0;

    const auto& face = mesh.face_vertices[fi];
    // Pass 1: place unvisited vertices of `face` and, on the exact octilinear
    // path, verify exact folded-coordinate closure on every second arrival.
    for (int v_curr : face) {
      if (out.keep_exact) {
        std::optional<std::pair<int64_t, int64_t>> exact_pt =
            TryPlaceExactFoldedVertex(
                mesh.exact_vertex_coords[p1_idx],
                mesh.exact_vertex_coords[p2_idx], out.exact_folded[p1_idx],
                out.exact_folded[p2_idx], mesh.exact_vertex_coords[v_curr],
                is_face_up);
        if (!exact_pt.has_value()) {
          out.keep_exact = false;
        } else if (v_set[v_curr]) {
          if (out.exact_folded[v_curr] != *exact_pt) {
            return RigidFoldedResult{
                .outcome = FoldedArrangementOutcome::kNotFoldableProof,
                .error_message = absl::StrFormat(
                    "Folded geometry conflict: vertex %d closure mismatch on "
                    "face %d",
                    v_curr, fi)};
          }
        } else {
          out.exact_folded[v_curr] = *exact_pt;
        }
      }
      if (!v_set[v_curr]) {
        double vx = mesh.vertex_coords[v_curr].first - p1x;
        double vy = mesh.vertex_coords[v_curr].second - p1y;
        double dx_proj = vx * xax_x + vy * xax_y;
        double dy_proj = (vx * xax_y - vy * xax_x) * (is_face_up ? 1.0 : -1.0);
        out.folded_vertices[v_curr] = {
            (xaf_x * dx_proj + xaf_y * dy_proj) + q1x,
            (xaf_y * dx_proj - xaf_x * dy_proj) + q1y};
        v_set[v_curr] = 1;
      }
    }

    // Pass 2: traverse edges to enqueue unvisited neighbor faces and, on the
    // exact octilinear path, verify orientation parity on every second arrival
    // (float-path tolerance is intentionally omitted until derived).
    int v_prev = face.back();
    for (int v_curr : face) {
      auto it_nb = dir_edge_to_face.find(EncodeDirectedPair(v_prev, v_curr));
      if (it_nb != dir_edge_to_face.end()) {
        auto it_as = assign_map.find(EncodeOrderPair(v_prev, v_curr));
        CreaseAssignment as_code = (it_as != assign_map.end())
                                       ? it_as->second
                                       : CreaseAssignment::kFlat;
        bool next_up = IsActiveCrease(as_code) ? !is_face_up : is_face_up;
        int nb = it_nb->second;
        if (!seen_faces[nb]) {
          seen_faces[nb] = 1;
          out.face_orientations[nb] = !next_up;
          queue.push_back(QueueEntry{.fi = nb,
                                     .p1_idx = v_curr,
                                     .p2_idx = v_prev,
                                     .is_face_up = next_up});
        } else if (out.keep_exact && out.face_orientations[nb] != !next_up) {
          return RigidFoldedResult{
              .outcome = FoldedArrangementOutcome::kNotFoldableProof,
              .error_message = absl::StrFormat(
                  "Folded geometry conflict: face %d orientation parity "
                  "mismatch across edge (%d, %d)",
                  nb, v_prev, v_curr)};
        }
      }
      v_prev = v_curr;
    }
  }

  for (int vi = 0; vi < num_v; ++vi) {
    if (!v_set[vi]) {
      return RigidFoldedResult{
          .outcome = FoldedArrangementOutcome::kCouldNotDecide,
          .error_message = "Disconnected face encountered in folded state."};
    }
  }
  return RigidFoldedResult{.outcome = FoldedArrangementOutcome::kOk,
                           .placement = std::move(out)};
}

absl::Status BuildFoldedCellGraph(
    const UnfoldedMesh& mesh,
    absl::Span<const std::pair<int64_t, int64_t>> exact_folded, bool keep_exact,
    FoldedArrangementData* out) {
  ArrangementGraph folded_arr;
  bool built_folded = false;
  if (keep_exact && mesh.exact_scale > 0 &&
      mesh.exact_scale <= kFoldedCoordLimit) {
    std::vector<std::tuple<int64_t, int64_t, int64_t, int64_t>> f_segs;
    f_segs.reserve(mesh.edge_vertices.size());
    bool all_octilinear = true;
    auto coord_in_range = [](int64_t c) {
      return c >= -kFoldedCoordLimit && c <= kFoldedCoordLimit;
    };
    for (auto [u, v] : mesh.edge_vertices) {
      auto [x1, y1] = exact_folded[u];
      auto [x2, y2] = exact_folded[v];
      if (!coord_in_range(x1) || !coord_in_range(y1) || !coord_in_range(x2) ||
          !coord_in_range(y2) ||
          !ClassifyOctilinearDelta(x2 - x1, y2 - y1).has_value()) {
        all_octilinear = false;
        break;
      }
      f_segs.emplace_back(x1, y1, x2, y2);
    }
    if (all_octilinear) {
      if (std::optional<ArrangementGraph> exact_arr =
              BuildOctilinearFromExactSegments(f_segs, mesh.exact_scale);
          exact_arr.has_value()) {
        folded_arr = std::move(*exact_arr);
        built_folded = true;
      }
    }
  }
  out->used_exact_folded_octilinear = built_folded;
  if (!built_folded) {
    std::vector<InputSegment> f_lines;
    f_lines.reserve(mesh.edge_vertices.size());
    for (auto [u, v] : mesh.edge_vertices) {
      f_lines.push_back(
          InputSegment{.x1 = out->folded_vertices[u].first,
                       .y1 = out->folded_vertices[u].second,
                       .x2 = out->folded_vertices[v].first,
                       .y2 = out->folded_vertices[v].second,
                       .assignment = CreaseAssignment::kBoundary});
    }
    absl::StatusOr<ArrangementGraph> general_folded =
        BuildGeneralArrangement(f_lines, "folded");
    if (!general_folded.ok()) {
      return general_folded.status();
    }
    folded_arr = std::move(*general_folded);
  }

  if (folded_arr.vertices.empty()) {
    return absl::AbortedError("Precision Error: unstable graph");
  }

  out->cell_points = std::move(folded_arr.vertices);
  out->segment_points = std::move(folded_arr.edge_vertices);
  out->segment_edges = std::move(folded_arr.edge_lines);

  std::vector<std::vector<int>> cell_adj;
  BuildAdjacencyAndFaces(out->cell_points, out->segment_points, &cell_adj,
                         &out->cell_vertices);
  if (out->cell_vertices.empty()) {
    return absl::AbortedError("Precision Error: empty folded cell graph");
  }
  BuildEdgeFaceMaps(out->segment_points, out->cell_vertices,
                    &out->segment_cells, nullptr);
  return absl::OkStatus();
}

void PropagateCellFaceContainment(
    const UnfoldedMesh& mesh,
    const absl::flat_hash_map<uint64_t, std::vector<int>>& seg_face_map,
    const absl::flat_hash_map<uint64_t, int>& dir_seg_cell_map,
    FoldedArrangementData* out) {
  const int num_f = static_cast<int>(mesh.face_vertices.size());
  const int num_c = static_cast<int>(out->cell_vertices.size());
  out->cell_faces.assign(num_c, {});
  std::vector<uint8_t> seen_cells(num_c, 0);
  std::vector<int> cell_q;
  cell_q.reserve(num_c);

  for (size_t si = 0; si < out->segment_cells.size(); ++si) {
    if (out->segment_cells[si].size() == 1) {
      int ci = out->segment_cells[si][0];
      auto it = seg_face_map.find(EncodeOrderPair(
          out->segment_points[si].first, out->segment_points[si].second));
      if (it != seg_face_map.end()) {
        absl::flat_hash_set<int> fset(it->second.begin(), it->second.end());
        out->cell_faces[ci].assign(fset.begin(), fset.end());
      }
      seen_cells[ci] = 1;
      cell_q.push_back(ci);
      break;
    }
  }

  for (size_t qi = 0; qi < cell_q.size(); ++qi) {
    int ci = cell_q[qi];
    const auto& cell = out->cell_vertices[ci];
    if (cell.empty()) continue;
    int v1 = cell.back();
    for (int v2 : cell) {
      auto it_nb = dir_seg_cell_map.find(EncodeDirectedPair(v1, v2));
      if (it_nb != dir_seg_cell_map.end() && !seen_cells[it_nb->second]) {
        int cj = it_nb->second;
        seen_cells[cj] = 1;
        cell_q.push_back(cj);
        absl::flat_hash_set<int> fset(out->cell_faces[ci].begin(),
                                      out->cell_faces[ci].end());
        auto it_sf = seg_face_map.find(EncodeOrderPair(v1, v2));
        if (it_sf != seg_face_map.end()) {
          for (int fi : it_sf->second) {
            if (!fset.erase(fi)) fset.insert(fi);
          }
        }
        out->cell_faces[cj].assign(fset.begin(), fset.end());
      }
      v1 = v2;
    }
  }

  out->face_cells.assign(num_f, {});
  for (int ci = 0; ci < num_c; ++ci) {
    auto& cf = out->cell_faces[ci];
    SortAndUnique(&cf);
    for (int fi : cf) {
      out->face_cells[fi].push_back(ci);
    }
  }
}

void ExtractTransitiveOverlappingFacePairs(
    const absl::flat_hash_map<uint64_t, std::vector<int>>& seg_face_map,
    const absl::flat_hash_map<uint64_t, int>& dir_seg_cell_map,
    FoldedArrangementData* out) {
  const int num_c = static_cast<int>(out->cell_vertices.size());
  absl::flat_hash_set<uint64_t> face_pairs_set;
  if (num_c > 0 && !out->cell_faces[0].empty()) {
    const auto& faces0 = out->cell_faces[0];
    for (size_t j_idx = 1; j_idx < faces0.size(); ++j_idx) {
      for (size_t i_idx = 0; i_idx < j_idx; ++i_idx) {
        face_pairs_set.insert(EncodeOrderPair(faces0[i_idx], faces0[j_idx]));
      }
    }
  }
  std::vector<uint8_t> seen_tp_cells(num_c, 0);
  std::vector<int> tp_queue;
  if (num_c > 0) {
    seen_tp_cells[0] = 1;
    tp_queue.push_back(0);
  }
  for (size_t qi = 0; qi < tp_queue.size(); ++qi) {
    int ci = tp_queue[qi];
    if (ci < 0 || ci >= num_c) continue;
    const auto& cell = out->cell_vertices[ci];
    if (cell.empty()) continue;
    int v1 = cell.back();
    for (int v2 : cell) {
      auto it_nb = dir_seg_cell_map.find(EncodeDirectedPair(v1, v2));
      if (it_nb != dir_seg_cell_map.end() && !seen_tp_cells[it_nb->second]) {
        int cj = it_nb->second;
        tp_queue.push_back(cj);
        seen_tp_cells[cj] = 1;
        const auto& faces_i = out->cell_faces[ci];
        const auto& faces_j = out->cell_faces[cj];
        auto it_sf = seg_face_map.find(EncodeOrderPair(v1, v2));
        if (it_sf != seg_face_map.end()) {
          for (int face_i : it_sf->second) {
            if (std::binary_search(faces_i.begin(), faces_i.end(), face_i) ||
                !std::binary_search(faces_j.begin(), faces_j.end(), face_i)) {
              continue;
            }
            for (int face_j : faces_j) {
              if (face_i == face_j) continue;
              face_pairs_set.insert(EncodeOrderPair(face_i, face_j));
            }
          }
        }
      }
      v1 = v2;
    }
  }
  out->face_pairs.clear();
  out->face_pairs.reserve(face_pairs_set.size());
  for (uint64_t key : face_pairs_set) {
    out->face_pairs.push_back(
        {static_cast<int>(key >> 32), static_cast<int>(key & 0xFFFFFFFFULL)});
  }
  absl::c_sort(out->face_pairs);
}

void ComputeCellFaceMappingsAndOverlappingPairs(const UnfoldedMesh& mesh,
                                                FoldedArrangementData* out) {
  const int num_c = static_cast<int>(out->cell_vertices.size());
  absl::flat_hash_map<uint64_t, std::vector<int>> seg_face_map;
  seg_face_map.reserve(out->segment_points.size() * 2 + 4);
  for (size_t si = 0; si < out->segment_points.size(); ++si) {
    std::vector<int> faces;
    for (int ei : out->segment_edges[si]) {
      if (ei >= 0 && ei < static_cast<int>(mesh.edge_faces.size())) {
        for (int fi : mesh.edge_faces[ei]) {
          faces.push_back(fi);
        }
      }
    }
    seg_face_map[EncodeOrderPair(out->segment_points[si].first,
                                 out->segment_points[si].second)] =
        std::move(faces);
  }

  absl::flat_hash_map<uint64_t, int> dir_seg_cell_map;
  dir_seg_cell_map.reserve(out->segment_points.size() * 4 + 8);
  for (int ci = 0; ci < num_c; ++ci) {
    const auto& cell = out->cell_vertices[ci];
    if (cell.empty()) continue;
    int v1 = cell.back();
    for (int v2 : cell) {
      dir_seg_cell_map[EncodeDirectedPair(v2, v1)] = ci;
      v1 = v2;
    }
  }

  PropagateCellFaceContainment(mesh, seg_face_map, dir_seg_cell_map, out);
  ExtractTransitiveOverlappingFacePairs(seg_face_map, dir_seg_cell_map, out);
}

char CreaseCodeChar(CreaseAssignment code) {
  switch (code) {
    case CreaseAssignment::kBoundary:
      return 'B';
    case CreaseAssignment::kMountain:
      return 'M';
    case CreaseAssignment::kValley:
      return 'V';
    case CreaseAssignment::kFlat:
      return 'F';
    case CreaseAssignment::kUnassigned:
      return 'U';
  }
  return '?';
}

absl::StatusOr<ArrangementGraph> BuildArrangementFromSegmentsWithStatus(
    absl::Span<const InputSegment> lines, int64_t forced_exact_scale,
    std::string_view stage_label) {
  if (lines.empty()) {
    return absl::InvalidArgumentError("IO Parser failed");
  }
  for (const auto& ln : lines) {
    if (!std::isfinite(ln.x1) || !std::isfinite(ln.y1) ||
        !std::isfinite(ln.x2) || !std::isfinite(ln.y2)) {
      return absl::InvalidArgumentError("Non-finite input segment coordinate");
    }
  }
  for (const auto& ln : lines) {
    if (!IsValidCoordinate(ln.x1) || !IsValidCoordinate(ln.y1) ||
        !IsValidCoordinate(ln.x2) || !IsValidCoordinate(ln.y2)) {
      return absl::InvalidArgumentError(
          "Input segment coordinate exceeds 1e100 magnitude bound");
    }
  }
  if (std::optional<ArrangementGraph> exact_graph =
          TryBuildOctilinearArrangement(lines, forced_exact_scale);
      exact_graph.has_value()) {
    return *std::move(exact_graph);
  }
  return BuildGeneralArrangement(lines, stage_label);
}

}  // namespace

ArrangementGraph BuildArrangementFromSegments(
    absl::Span<const InputSegment> lines, int64_t forced_exact_scale) {
  absl::StatusOr<ArrangementGraph> arr = BuildArrangementFromSegmentsWithStatus(
      lines, forced_exact_scale, "unfolded");
  if (!arr.ok()) return ArrangementGraph{};
  return *std::move(arr);
}

absl::StatusOr<std::vector<CreaseAssignment>> ResolveEdgeAssignmentsFromLines(
    absl::Span<const std::vector<int>> edge_lines,
    absl::Span<const InputSegment> lines,
    absl::Span<const int> segment_line_numbers) {
  const bool has_line_numbers = !segment_line_numbers.empty() &&
                                segment_line_numbers.size() == lines.size();
  std::vector<CreaseAssignment> edge_assignments(edge_lines.size(),
                                                 CreaseAssignment::kFlat);
  for (size_t ei = 0; ei < edge_lines.size(); ++ei) {
    CreaseAssignment chosen = CreaseAssignment::kFlat;
    int chosen_idx = -1;
    for (int li : edge_lines[ei]) {
      if (li < 0 || li >= static_cast<int>(lines.size())) continue;
      CreaseAssignment c = lines[li].assignment;
      if (c == CreaseAssignment::kFlat) continue;
      if (chosen == CreaseAssignment::kFlat) {
        chosen = c;
        chosen_idx = li;
      } else if (c != chosen) {
        CreaseAssignment code_a = chosen;
        CreaseAssignment code_b = c;
        int loc_a =
            has_line_numbers ? segment_line_numbers[chosen_idx] : chosen_idx;
        int loc_b = has_line_numbers ? segment_line_numbers[li] : li;
        if (loc_a > loc_b) {
          std::swap(loc_a, loc_b);
          std::swap(code_a, code_b);
        }
        if (has_line_numbers) {
          return absl::InvalidArgumentError(absl::StrFormat(
              "Conflicting collinear crease codes (%c vs %c) at lines %d and "
              "%d",
              CreaseCodeChar(code_a), CreaseCodeChar(code_b), loc_a, loc_b));
        }
        return absl::InvalidArgumentError(absl::StrFormat(
            "Conflicting collinear crease codes (%c vs %c) on overlapping "
            "segments %d and %d",
            CreaseCodeChar(code_a), CreaseCodeChar(code_b), loc_a, loc_b));
      }
    }
    edge_assignments[ei] = chosen;
  }
  return edge_assignments;
}

absl::StatusOr<UnfoldedMesh> BuildUnfoldedMeshFromLines(
    absl::Span<const InputSegment> lines,
    FaceOrientationPolicy orientation_policy,
    absl::Span<const int> segment_line_numbers) {
  if (lines.empty()) {
    return absl::InvalidArgumentError("IO Parser failed");
  }
  absl::StatusOr<ArrangementGraph> arr_or =
      BuildArrangementFromSegmentsWithStatus(lines, /*forced_exact_scale=*/0,
                                             "unfolded");
  if (!arr_or.ok()) {
    return arr_or.status();
  }
  ArrangementGraph arr = *std::move(arr_or);
  if (arr.vertices.empty() || arr.edge_vertices.empty()) {
    return absl::InvalidArgumentError("IO Parser failed");
  }

  absl::StatusOr<std::vector<CreaseAssignment>> resolved_assignments =
      ResolveEdgeAssignmentsFromLines(arr.edge_lines, lines,
                                      segment_line_numbers);
  if (!resolved_assignments.ok()) {
    return resolved_assignments.status();
  }

  UnfoldedMesh mesh;
  mesh.used_exact_octilinear = arr.used_exact_octilinear;
  mesh.exact_scale = arr.exact_scale;
  mesh.vertex_coords = std::move(arr.vertices);
  mesh.exact_vertex_coords = std::move(arr.exact_vertices);
  mesh.edge_vertices = std::move(arr.edge_vertices);
  mesh.edge_assignments = std::move(*resolved_assignments);
  mesh.edge_lines = std::move(arr.edge_lines);

  if (absl::Status status = FinalizeParsedMesh(
          orientation_policy, /*has_explicit_faces=*/false, &mesh);
      !status.ok()) {
    return status;
  }
  return mesh;
}

absl::StatusOr<UnfoldedMesh> BuildUnfoldedMeshFromFoldGraph(
    absl::Span<const std::pair<double, double>> vertex_coords,
    absl::Span<const std::pair<int, int>> edge_vertices,
    absl::Span<const CreaseAssignment> edge_assignments,
    absl::Span<const std::vector<int>> optional_face_vertices,
    FaceOrientationPolicy orientation_policy) {
  if (vertex_coords.empty() || edge_vertices.empty() ||
      edge_assignments.size() != edge_vertices.size()) {
    return absl::InvalidArgumentError("IO Parser failed");
  }
  for (const auto& [x, y] : vertex_coords) {
    if (!std::isfinite(x) || !std::isfinite(y)) {
      return absl::InvalidArgumentError("Non-finite vertex coordinate");
    }
    if (!IsValidCoordinate(x) || !IsValidCoordinate(y)) {
      return absl::InvalidArgumentError(
          "Vertex coordinate exceeds 1e100 magnitude bound");
    }
  }
  for (size_t i = 0; i < edge_assignments.size(); ++i) {
    if (!IsValidCreaseAssignment(edge_assignments[i])) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Invalid crease assignment code %d at edge %u",
          static_cast<int>(edge_assignments[i]), static_cast<unsigned>(i)));
    }
  }
  const int num_vertices = static_cast<int>(vertex_coords.size());
  absl::flat_hash_set<uint64_t> declared_edges;
  declared_edges.reserve(edge_vertices.size());
  for (const auto& [u, v] : edge_vertices) {
    if (u < 0 || u >= num_vertices || v < 0 || v >= num_vertices || u == v) {
      return absl::InvalidArgumentError("Out-of-bounds edge vertex index");
    }
    if (!declared_edges.insert(EncodeOrderPair(u, v)).second) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Duplicate edge (%d, %d) in fold graph", u, v));
    }
  }
  if (optional_face_vertices.size() > static_cast<size_t>(kMaxFaces)) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Face count %zu exceeds kMaxFaces (%d)",
                        optional_face_vertices.size(), kMaxFaces));
  }
  absl::flat_hash_set<uint64_t> seen_directed_half_edges;
  if (!optional_face_vertices.empty()) {
    seen_directed_half_edges.reserve(edge_vertices.size() * 2);
  }
  absl::flat_hash_set<int> face_seen_vertices;
  for (size_t fi = 0; fi < optional_face_vertices.size(); ++fi) {
    const std::vector<int>& face = optional_face_vertices[fi];
    if (face.size() < 3) {
      return absl::InvalidArgumentError("Face must have at least 3 vertices");
    }
    face_seen_vertices.clear();
    face_seen_vertices.reserve(face.size());
    for (int v : face) {
      if (v < 0 || v >= num_vertices) {
        return absl::InvalidArgumentError("Out-of-bounds face vertex index");
      }
      if (!face_seen_vertices.insert(v).second) {
        return absl::InvalidArgumentError(
            absl::StrFormat("Repeated vertex %d in face %zu", v, fi));
      }
    }
    int v1 = face.back();
    for (int v2 : face) {
      if (!declared_edges.contains(EncodeOrderPair(v1, v2))) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "Face %zu references undeclared edge (%d, %d)", fi, v1, v2));
      }
      if (!seen_directed_half_edges.insert(EncodeDirectedPair(v1, v2)).second) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "Directed half-edge (%d, %d) used by multiple faces (face %zu)", v1,
            v2, fi));
      }
      v1 = v2;
    }
  }

  UnfoldedMesh mesh;
  mesh.vertex_coords.assign(vertex_coords.begin(), vertex_coords.end());
  mesh.edge_vertices.assign(edge_vertices.begin(), edge_vertices.end());
  mesh.edge_assignments.assign(edge_assignments.begin(),
                               edge_assignments.end());
  bool has_faces = !optional_face_vertices.empty();
  if (has_faces) {
    mesh.face_vertices.assign(optional_face_vertices.begin(),
                              optional_face_vertices.end());
  }
  if (absl::Status status =
          FinalizeParsedMesh(orientation_policy, has_faces, &mesh);
      !status.ok()) {
    return status;
  }
  return mesh;
}

std::vector<std::string> CheckLocalVertexFoldability(
    const UnfoldedMesh& mesh,
    absl::Span<const CreaseAssignment> edge_assignments,
    double kawasaki_epsilon) {
  ABSL_DCHECK_EQ(edge_assignments.size(), mesh.edge_vertices.size());
  std::vector<std::string> local_violations;
  absl::flat_hash_map<uint64_t, CreaseAssignment> assign_map;
  assign_map.reserve(mesh.edge_vertices.size() * 2 + 4);
  for (size_t ei = 0; ei < mesh.edge_vertices.size(); ++ei) {
    auto [u, v] = mesh.edge_vertices[ei];
    assign_map[EncodeOrderPair(u, v)] = edge_assignments[ei];
  }

  for (int vi = 0; vi < static_cast<int>(mesh.vertex_adjacencies.size());
       ++vi) {
    const auto& neighbors = mesh.vertex_adjacencies[vi];
    std::vector<int> internal_nbs;
    internal_nbs.reserve(neighbors.size());
    bool is_boundary = false;
    int m_cnt = 0;
    int v_cnt = 0;
    int u_cnt = 0;
    for (int nb : neighbors) {
      auto it = assign_map.find(EncodeOrderPair(vi, nb));
      CreaseAssignment c =
          (it != assign_map.end()) ? it->second : CreaseAssignment::kFlat;
      if (c == CreaseAssignment::kBoundary) {
        is_boundary = true;
        break;
      }
      if (IsActiveCrease(c)) {
        internal_nbs.push_back(nb);
        if (c == CreaseAssignment::kMountain) {
          ++m_cnt;
        } else if (c == CreaseAssignment::kValley) {
          ++v_cnt;
        } else {
          ++u_cnt;
        }
      }
    }
    if (is_boundary || internal_nbs.empty()) continue;

    int maekawa_dev = std::abs(m_cnt - v_cnt);
    if ((u_cnt == 0 && maekawa_dev != 2) || (internal_nbs.size() % 2 != 0)) {
      local_violations.push_back(
          absl::StrFormat("v%d Maekawa Error |M-V|=%d", vi, maekawa_dev));
      continue;
    }

    std::vector<double> angles;
    angles.reserve(internal_nbs.size());
    for (int nb : internal_nbs) {
      double dx = mesh.vertex_coords[nb].first - mesh.vertex_coords[vi].first;
      double dy = mesh.vertex_coords[nb].second - mesh.vertex_coords[vi].second;
      angles.push_back(SafeAngle0To2Pi(dx, dy));
    }
    absl::c_sort(angles);
    double kawasaki_sum = 0.0;
    for (size_t j = 0; j + 1 < angles.size(); j += 2) {
      kawasaki_sum += angles[j + 1] - angles[j];
    }
    double dev = std::abs(kawasaki_sum - kPi);
    if (dev > kawasaki_epsilon) {
      local_violations.push_back(
          absl::StrFormat("v%d Kawasaki Error dev=%.2e", vi, dev));
    }
  }
  return local_violations;
}

FoldedArrangementResult BuildFoldedArrangement(const UnfoldedMesh& mesh) {
  RigidFoldedResult placed = ComputeRigidFoldedVertices(mesh);
  if (placed.outcome != FoldedArrangementOutcome::kOk) {
    return FoldedArrangementResult{
        .outcome = placed.outcome,
        .error_message = std::move(placed.error_message)};
  }

  FoldedArrangementData out;
  out.folded_vertices = std::move(placed.placement.folded_vertices);
  out.face_orientations = std::move(placed.placement.face_orientations);
  if (absl::Status status =
          BuildFoldedCellGraph(mesh, placed.placement.exact_folded,
                               placed.placement.keep_exact, &out);
      !status.ok()) {
    // Every failure in folded cell-graph construction (including float-sweep
    // cap, instability, or non-finite/degenerate folded lines) is an engine
    // limitation, never a proof of non-foldability.
    return FoldedArrangementResult{
        .outcome = FoldedArrangementOutcome::kCouldNotDecide,
        .error_message = std::string(status.message())};
  }
  ComputeCellFaceMappingsAndOverlappingPairs(mesh, &out);
  return FoldedArrangementResult{.outcome = FoldedArrangementOutcome::kOk,
                                 .data = std::move(out)};
}

}  // namespace bp_pipeline_flat_folder::internal
