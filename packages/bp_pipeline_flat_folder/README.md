# `bp-pipeline-flat-folder`

Fast C++20 global flat-foldability checker and exact-rational SAT certificate
verifier for [`bp-pipeline`](https://github.com/google-deepmind/origami-design).

## What It Provides

`bp-pipeline-flat-folder` (`import bp_pipeline_flat_folder`) ships two
components with **no OR-Tools dependency** (depends only on `bp-pipeline`):

1.  **C++20 flat-foldability engine (`bp_pipeline_flat_folder.fast_flat_folder`,
    curated public interface re-exported at package root)**:
    *   `FlatFolderSession`: thread-safe stateful 3-tier incremental/memoized
        checker (Tier-1 canonical `.cp` result memo, Tier-2 M/V-invariant planar
        arrangement & folded cell overlap support cache, Tier-3 canonical
        unassigned component CSP solution cache) with exact `int64_t` /
        `absl::int128` octilinear geometry, derived Bentley-Ottmann float sweep
        event cap, seed-aware memoization, LRU-capped 256-stream `.cp` delta
        seeding, optional `backend='python_ref'`, and runtime shadow auditing.
    *   Drop-in `.cp`/`.fold`/`.svg`/`.opx` entry point (`analyze_fold_file`,
        returning `FlatFolderOutput`), stateless `.cp` checker (`check_cp`,
        returning three-valued `FastFoldResult` / `FoldVerdict`), and
        unfolded/folded geometry helpers
        (`FlatFolderSession.extract_unfolded_faces_cp`,
        `compute_folded_vertices_cp`, `solve_flat_folded_state`).
2.  **Exact-rational SAT certificate verifier
    (`bp_pipeline_flat_folder.sat_certificate_checker`)**:
    *   Independent pure-Python exact integer/rational checker verifying every
        accepted folded state against the raw `.cp` crease pattern
        (`verify_cpp_sat_certificate`, `verify_reference_sat_certificate`,
        `CertificateCheckResult`, `CertificateRejectionKind`,
        `REJECTION_REASON_TABLE`, `classify_rejection_reason`).

### Standalone Behavior (No Side Effects on `bp_pipeline`)

**Installing `bp-pipeline-flat-folder` alone changes nothing in `bp_pipeline`.**
It registers no `bp_pipeline.plugins` entry points, and `bp.pack`, `bp.solve`,
`bp.fold`, and `SolvedCreasePattern.flat_fold()` continue using `bp-pipeline`'s
built-in pure-Python flat-folder unchanged. Call the functions and
`FlatFolderSession` exported by `bp_pipeline_flat_folder` explicitly by name.
Making `bp_pipeline` use `bp-pipeline-flat-folder` automatically when installed
requires a hook in `bp_pipeline` itself and is deferred to a separate library
change.

## Public Interface

### Core Public API

<!-- mdformat off(reason: standard GFM table rendering on GitHub and PyPI) -->
| Symbol / Method | Purpose |
| :--- | :--- |
| `FlatFolderSession` | Thread-safe stateful C++20 flat-foldability session with Tier-1/2/3 caches. |
| `FlatFolderSession.check_cp(...)` | Checks global flat-foldability (and exact rational SAT certificate when `verify_certificate=True`), returning `FastFoldResult`. |
| `FlatFolderSession.extract_unfolded_faces_cp(cp_doc, *, flip_to_top_face_up=True)` | Extracts unfolded `(vertex_coords, face_vertices)` polygons from a `.cp` crease pattern. |
| `check_cp(...)` | Stateless global flat-foldability check returning `FastFoldResult`. |
| `analyze_fold_file(...)` | Drop-in `.cp`/`.fold`/`.svg`/`.opx` checker returning `FlatFolderOutput`. |
| `compute_folded_vertices_cp(cp_doc, *, session=None, kawasaki_epsilon=5e-2)` | Computes `(vertex_coords, folded_vertices)` pairs from a `.cp` crease pattern. |
| `solve_flat_folded_state(...)` | Computes a flat-folded state with optional face-group precedent tie-breaking (`FlatFoldSolution`). |
| `FastFoldResult` / `FoldVerdict` | Frozen result dataclass (`flat_foldable`, `verdict`, `states`, `conflict_reason`, `conflict_faces`, `local_violations`, `vertex_coords`, `folded_vertices`) and three-valued verdict (`FOLDABLE`, `NOT_FOLDABLE`, `COULD_NOT_DECIDE`). |
| `BackendName` | Type alias (`Literal['cpp', 'python_ref']`) for backend selection. |
| `sat_certificate_checker`, `fast_flat_folder` | Public submodules exposing exact-rational SAT certificate verification, three-way rejection classification, and diagnostic utilities. |
<!-- mdformat on -->

### Full Package Root Exports (`bp_pipeline_flat_folder.__all__`, 13 names)

*   **Session, verdict, result & exception types (7):** `BackendName`,
    `FastFlatFolderParityError`, `FastFoldResult`, `FlatFoldSolution`,
    `FlatFolderOutput`, `FlatFolderSession`, `FoldVerdict`.
*   **Top-level functions (4):** `analyze_fold_file`, `check_cp`,
    `compute_folded_vertices_cp`, `solve_flat_folded_state`.
*   **Submodules (2):** `fast_flat_folder`, `sat_certificate_checker`.

Advanced and diagnostic helpers (`ShadowAuditStats`, `engine_claim_rejections`,
`diff_cp_changed_segments`, `get_absolute_face_orders_from_edges`,
`FlatFolderFOLD`, `GUI_COLORS`, `visualize_folded_state`) remain available on
`bp_pipeline_flat_folder.fast_flat_folder`, and exact-certificate APIs
(`CertificateCheckResult`, `CertificateRejectionKind`, `REJECTION_REASON_TABLE`,
`classify_rejection_reason`, `verify_cpp_sat_certificate`,
`verify_reference_sat_certificate`) remain on
`bp_pipeline_flat_folder.sat_certificate_checker`.

## Requirements

*   **OS / Architecture:** Linux `x86_64` (verified; Windows and macOS are not
    yet supported).
*   **Python:** `>= 3.12` (with development headers for source builds; `uv` can
    manage Python versions automatically).
*   **Runtime dependencies:** `bp-pipeline >= 0.1.0` (no OR-Tools).
*   **Build dependencies (source install):** C++20 compiler (`g++ >= 11`,
    `clang++ >= 15`, or `ziglang`), `CMake >= 3.25`, `Ninja`,
    `pybind11 >= 3.0`, `scikit-build-core >= 0.11`, and `abseil-cpp`
    `>= 20250814` (fetched automatically at `20250814.1` via CMake
    `FetchContent` with SHA-256 verification, or supplied via
    `-DBP_USE_SYSTEM_ABSL=ON` / `-DBP_ABSEIL_URL=file:///path/to/archive`). A
    system Abseil must have been built with the same C++ standard library as
    your compiler (for example, a `libstdc++`-linked system Abseil with `g++`).

### CMake Targets and Options

`CMakeLists.txt` separates the Python-free C++20 engine (`fast_flat_folder_core`
static library) from the pybind11 extension (`_fast_flat_folder`), compiles
first-party targets with `-Wall -Wextra` (`/W4` on MSVC), and hides symbol
visibility by default for statically bundled Abseil:

*   `FAST_FLAT_FOLDER_BUILD_PYTHON` (default `ON`): builds the pybind11
    `_fast_flat_folder` Python extension module.
*   `FAST_FLAT_FOLDER_INSTALL_CPP` (default `ON` for standalone top-level CMake
    builds, `OFF` under `scikit-build-core` and `add_subdirectory`): installs
    `libfast_flat_folder_core.a`, public headers under
    `include/bp_pipeline_flat_folder/`, and CMake package config files
    exporting `bp_pipeline_flat_folder::fast_flat_folder_core`.
*   `FAST_FLAT_FOLDER_SANITIZERS` (default `""`): semicolon-separated sanitizer
    list (e.g. `address;undefined` or `thread`), applied consistently to both
    first-party targets and bundled Abseil.
*   `BP_USE_SYSTEM_ABSL` (default `OFF`): uses a pre-installed system Abseil
    instead of `FetchContent`. Any Abseil LTS release `>= 20250814` is
    accepted; configuration fails with a clear message on an older release.
*   `BP_ABSEIL_URL` / `BP_ABSEIL_URL_HASH`: source archive URL (including
    `file:///...` for offline builds) and expected SHA-256 hash when
    `BP_USE_SYSTEM_ABSL=OFF`.

## Installation

`bp-pipeline-flat-folder` depends on `bp-pipeline >= 0.1.0`, which is not yet
published on PyPI, so installing `packages/bp_pipeline_flat_folder` on its own
fails with `No matching distribution found for bp-pipeline>=0.1.0`. Install
both packages together from a clone of
[`origami-design`](https://github.com/google-deepmind/origami-design), running
the commands from the repository root: `.` builds `bp-pipeline` from the root
`pyproject.toml`, which satisfies the dependency of
`packages/bp_pipeline_flat_folder`.

### Standard Linux Environment (`uv` + System C++20 Compiler)

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python . packages/bp_pipeline_flat_folder pytest
source .venv/bin/activate
```

### Plain `pip` with System Python (`>= 3.12`)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install . packages/bp_pipeline_flat_folder pytest
```

### Offline / Firewalled Builds and System Abseil

By default (`BP_USE_SYSTEM_ABSL=OFF`), CMake downloads `abseil-cpp` `20250814.1`
and links it statically with hidden visibility. On offline or firewalled hosts,
either use a system Abseil (`>= 20250814`) or point `BP_ABSEIL_URL` at a local
tarball:

*   **System Abseil (with a matching C++ toolchain):**

    ```bash
    CMAKE_ARGS="-DBP_USE_SYSTEM_ABSL=ON" \
      pip install . packages/bp_pipeline_flat_folder pytest
    ```

    `CMAKE_ARGS` works identically with `uv pip install`. With `pip`, you can
    also pass the CMake definition directly via `--config-settings`:

    ```bash
    pip install --config-settings=cmake.define.BP_USE_SYSTEM_ABSL=ON \
      . packages/bp_pipeline_flat_folder pytest
    ```

*   **Custom Abseil prefix:** pass `-Dabsl_DIR=/path/to/lib/cmake/absl` inside
    `CMAKE_ARGS` alongside `-DBP_USE_SYSTEM_ABSL=ON`. Do **not** pass
    `-DCMAKE_PREFIX_PATH=...` inside `CMAKE_ARGS` — it overrides
    `scikit-build-core`'s own prefix path and prevents CMake from finding
    `pybind11`.

*   **Pre-downloaded local Abseil tarball (offline bundled build):**

    ```bash
    CMAKE_ARGS="-DBP_ABSEIL_URL=file:///path/to/abseil-cpp-20250814.1.tar.gz" \
      pip install . packages/bp_pipeline_flat_folder pytest
    ```

### Environments Without System Python 3.12+ or `g++` (`uv` + `ziglang`)

`ziglang` ships a Clang-based C/C++ compiler as a Python module but no `cc` /
`c++` executables, while CMake needs `CC` and `CXX` to each name a single
executable. Two small wrapper scripts provide them:

```bash
# Requires `uv` on PATH (https://docs.astral.sh/uv/getting-started/installation/).
uv python install 3.12
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python ziglang cmake ninja pytest

mkdir -p .zig-bin
printf '#!/bin/sh\nexec "%s" -m ziglang cc "$@"\n' "$PWD/.venv/bin/python" > .zig-bin/zig-cc
printf '#!/bin/sh\nexec "%s" -m ziglang c++ "$@"\n' "$PWD/.venv/bin/python" > .zig-bin/zig-cxx
chmod +x .zig-bin/zig-cc .zig-bin/zig-cxx

CC="$PWD/.zig-bin/zig-cc" CXX="$PWD/.zig-bin/zig-cxx" \
  PATH="$PWD/.venv/bin:$PATH" \
  uv pip install --python .venv/bin/python \
    . packages/bp_pipeline_flat_folder
rm -rf .zig-bin
source .venv/bin/activate
```

`zig c++` compiles and links against its bundled `libc++`, so use the default
bundled Abseil (`BP_USE_SYSTEM_ABSL=OFF`, or a local tarball via
`BP_ABSEIL_URL`) rather than a `libstdc++`-linked system Abseil.
(If pointing `zig c++` at a custom `libc++`-built Abseil prefix on a host that
also has `/usr/include/absl`, add `-DCMAKE_NO_SYSTEM_FROM_IMPORTED=ON` alongside
`-Dabsl_DIR=...`.) `.zig-bin/` is only needed during compilation and can be
deleted afterwards.

### Standalone C++20 Library (CMake, No Python)

To embed the C++ engine without Python, either include this directory via
`add_subdirectory` or install the static library, headers, and CMake config:

```cmake
# Option 1: add_subdirectory (with FAST_FLAT_FOLDER_BUILD_PYTHON=OFF)
set(FAST_FLAT_FOLDER_BUILD_PYTHON OFF CACHE BOOL "" FORCE)
add_subdirectory(packages/bp_pipeline_flat_folder)
target_link_libraries(my_app PRIVATE bp_pipeline_flat_folder::fast_flat_folder_core)
```

```bash
# Option 2: build and install, then consume via find_package
cmake -S packages/bp_pipeline_flat_folder -B build-cpp -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DFAST_FLAT_FOLDER_BUILD_PYTHON=OFF
cmake --build build-cpp
cmake --install build-cpp --prefix /path/to/prefix
```

Downstream projects link `bp_pipeline_flat_folder::fast_flat_folder_core` after
`find_package(bp_pipeline_flat_folder CONFIG REQUIRED)` (passing
`-DCMAKE_PREFIX_PATH=/path/to/prefix`) and include
`"bp_pipeline_flat_folder/fast_flat_folder_core.h"`. Pass
`-DBP_USE_SYSTEM_ABSL=ON` to link a pre-installed Abseil (`>= 20250814`) instead
of installing the bundled copy into `--prefix`.

### Running the Portable Test Suite

With `.venv` activated from any of the install steps above:

```bash
pytest -c packages/bp_pipeline_flat_folder/pyproject.toml \
  packages/bp_pipeline_flat_folder/tests/fast_flat_folder_test.py
```

## Quickstart

```python
import bp_pipeline_flat_folder as ff

session = ff.FlatFolderSession()
res = session.check_cp(cp_doc, verify_certificate=True)
print(
    f'verdict={res.verdict.value} flat_foldable={res.flat_foldable}'
    f' states={res.states}'
)
```

## Verification Defaults and Stated Differences vs. Reference

*   **Three explicit answers (`foldable`, `not_foldable`, `could_not_decide`):**
    `check_cp(...)`, `FlatFolderSession.check_cp(...)`, and `_fast_flat_folder`
    return a three-valued `FoldVerdict` (`FOLDABLE` / `flat_foldable=True`,
    `NOT_FOLDABLE` / `flat_foldable=False` with a conflict reason, or
    `COULD_NOT_DECIDE` / `flat_foldable=None` with a diagnostic reason and
    `exit_stage="could_not_decide"`). `COULD_NOT_DECIDE` is used uniformly for
    both (a) engine limitations or failures — including the derived
    Bentley-Ottmann float sweep event cap $2n + n(n-1)/2$ for $n$ segments (at
    most $2n$ endpoints plus $\binom{n}{2}$ pairwise line intersections,
    preventing non-terminating float-noise sweeps), unstable float arrangements,
    disconnected sheets in rigid-fold BFS, `taco_only=True` Stage-1 screens, and
    checker rejections of the engine's own claimed layer order or fields (or
    unknown reasons) — and (b) patterns outside the exact checker's scope (off
    the 45° lines `non_octilinear_cp`, `excessive_common_denominator`,
    `slit_or_dangling_edge`, `non_manifold_or_orphan_edge`,
    `disconnected_unfolded_faces`, or unparsable checker inputs).
*   **Default exact SAT certificate confirmation (`verify_certificate=True`):**
    `check_cp(...)` and `FlatFolderSession.check_cp(...)` confirm
    `flat_foldable=True` results with the independent exact-rational
    `sat_certificate_checker` by default (except when `taco_only=True` or
    `ignore_flat_creases=True`, reported by `certificate_verified`), protecting
    callers on both the exact octilinear and float paths. Under
    `verify_certificate=True`, a failed certificate check returns `NOT_FOLDABLE`
    only for exact pattern-level refutations (`orientation_parity_conflict`,
    `transform_holonomy_conflict`, `folded_vertex_closure_mismatch`, listed in
    `sat_certificate_checker.REJECTION_REASON_TABLE`); engine-claim failures
    (crossings in a claimed layer order, mismatched vertex/cell fields, and any
    unrecognised reason) return `COULD_NOT_DECIDE` naming the engine and are
    counted via `engine_claim_rejections`, while checker scope limits return
    `COULD_NOT_DECIDE` (`SAT certificate uncheckable: ...`). Iterative search
    loops can pass `verify_certificate=False` explicitly during candidate
    screening and certify final accepted solutions with
    `verify_certificate=True`. On the 280-pattern fixed-seed differential suite
    (`113` SAT, `167` UNSAT), default `verify_certificate=True` adds 0.9 to
    2.0 s total (7 to 17 ms/SAT; zero overhead on UNSATs). `NOT_FOLDABLE` from
    the layer-order search is the engine's own result and is not independently
    confirmed (only `FOLDABLE` is confirmed by `sat_certificate_checker`); the
    evidence for `NOT_FOLDABLE` is agreement with the reference folder on the
    differential suite.
*   **Exact-path second-arrival orientation-parity and vertex-closure checks:**
    On the exact octilinear path, `bp-pipeline-flat-folder` verifies both face
    orientation parity and exact folded vertex coordinates whenever BFS arrives
    at an already-visited face or vertex via another edge, returning
    `folded_geometry_error` (`flat_foldable=False`) on mismatch. This is a
    stated soundness difference from `bp_pipeline`'s reference Python
    flat-folder, which skips second-arrival checks and falsely returns
    `flat_foldable=True` (refuted by `sat_certificate_checker`) on:
    *   **Punctured sheet with non-zero rigid holonomy (`holonomy_cp`):**
        `[0, 6] × [0, 4]` minus `[2, 4] × [1, 3]` with parallel active creases
        (`M` at `x = 2`, `V` at `x = 3`) across `y ∈ [3, 4]` — exact checker
        reason: `transform_holonomy_conflict`.
    *   **Interior boundary-coded (`1`/`B`) segment (smallest case: 3×3
        alternating accordion square with `1 1 2 1 3`):** interior vertex
        `(1, 2)` touches a `B`-coded edge, bypassing local Maekawa/Kawasaki
        despite odd active degree — exact checker reason:
        `orientation_parity_conflict` (or exact vertex closure mismatch when two
        adjacent interior segments are coded `B`, e.g., `1 1 2 1 3` and
        `1 1 2 2 2`).
    *   On the 280-pattern fixed-seed differential parity suite (`SHA-256`
        `c2db17d45605b2a7...`), `0/280` cases hit this difference (`280/280`
        agree with Python and `113/113` SATs are confirmed by
        `sat_certificate_checker`).
*   **Malformed `.fold` graph topology validation (`analyze_fold_graph`):**
    `BuildUnfoldedMeshFromFoldGraph` validates graph topology at ingress and
    returns `parse_error` (`ok_parse=False`, `flat_foldable=False`) on duplicate
    undirected edges, repeated vertices within a face, face sides referencing
    undeclared edges, or any directed half-edge shared by multiple faces (which
    rejects duplicate faces, mixed-winding adjacent faces, and non-manifold
    edges shared by more than two faces). Well-formed CCW and CW 2-manifold
    `.fold` graphs (with or without explicit `faces_vertices`) are unaffected.
*   **Conflicting collinear crease codes (`analyze_cp` / `analyze_lines` and
    `sat_certificate_checker`):** Overlapping collinear segments carrying an
    auxiliary flat line (`F`, code `4`) together with one crease or boundary
    code (`M`, `V`, `U`, `B`) do not conflict — the non-flat code stands,
    matching `bp_pipeline`'s `io_utils.py`. Overlapping collinear segments with
    two distinct fold or boundary codes (`M`, `V`, `U`, `B`) on the same
    elementary edge are rejected as `parse_error` (citing 1-based `.cp` line
    numbers when available, on both cold and warm `support_cache` paths) and
    rejected identically by `sat_certificate_checker`
    (`conflicting_collinear_crease_codes`). This is an intentional strictness
    difference vs. `bp_pipeline`'s reference Python flat-folder, which takes the
    first non-flat code in input order.
*   **Ingress validation of crease codes, zero-length segments, and coordinate
    bounds (`analyze_cp` / `analyze_lines` / `analyze_fold_graph` and
    `sat_certificate_checker`):**
    *   **Zero-length segments:** Both `_fast_flat_folder` and
        `sat_certificate_checker` reject zero-length segments
        (`(x1, y1) == (x2, y2)`) at ingress with the 1-based line number
        (`parse_error` / `zero_length_segment_at_line_<N>`), so one `.cp` file
        means the same to both. Stated difference from the library's reference:
        the reference accepts a zero-length segment and ignores it; this package
        rejects the file with the line number.
    *   **Crease codes, coordinate magnitude, and limits:** Crease codes outside
        `0..5` (`0` and `5` both map to unassigned), coordinates that are
        non-finite or exceed `|coord| <= 1e100` (matching
        `sat_certificate_checker._MAX_COORD_MAGNITUDE`), negative
        `solution_limit`, wrong-length `face_group_ranks`, and graphs exceeding
        `kMaxFaces = 8192` are rejected cleanly with diagnostics.

## Known Limitations

*   `fast_flat_folder.py` imports
    `bp_pipeline.third_party.flat_folder.{fold_cp, io_utils}` for its
    `FlatFolderFOLD` / `FlatFolderOutput` result types, `.svg`/`.opx`/`.fold`
    parsers, and one internal helper (`fold_cp._create_svg_data` for front/back
    folded-state SVG rendering) until `bp-pipeline` exposes a public SVG
    renderer.
*   Float sweep degenerate-direction handling (`fast_flat_folder_geometry.cc`):
    `GeneralSweepRunner` can produce a wrong face polygon from a degenerate
    zero-direction segment; zero-length ingress checks reject malformed
    zero-length segments at entry until the float sweep path is refactored.

## License and Third-Party Notices

Copyright 2026 Google LLC.

All Google-authored software in this package is licensed under the Apache
License, Version 2.0 (https://www.apache.org/licenses/LICENSE-2.0; see the
repository root `LICENSE`).

The C++20 flat-folder engine (`src/bp_pipeline_flat_folder/`) re-implements the
global flat-foldability algorithm from Jason S. Ku's MIT-licensed
[Flat-Folder](https://github.com/origamimagiro/flat-folder) software (see
[`src/bp_pipeline_flat_folder/LICENSE`](src/bp_pipeline_flat_folder/LICENSE) and
the repository root `LICENSE`).

This is not an official Google product.
