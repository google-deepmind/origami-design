---
name: origami-trajectory-optimization
description: >-
  Evaluates and optimizes continuous two-stage 3D origami folding trajectories
  from flat crease patterns to shaped sculptures using an 8-metric physicality
  scorecard, signed-dihedral XPBD base collapse, and Short-Hop Radial Dijkstra
  shaping schedules. Use when adding a showcase model to COrigami Studio,
  generating or auditing `schedule.json`, diagnosing mid-fold paper stretching,
  volume expansion, seam tears, layer self-intersections, or velocity spikes,
  or restoring solved-base creases on `shaped_3d.fold`. Don't use for static
  stick-figure authoring (use `stick-figure-design`), packing/solving crease
  patterns (use `origami-solving-pipeline`), or static 3D pose sculpting and
  turnaround fidelity scoring (use `origami-shaping` and `origami-critique`).
---

# Two-Stage Origami Trajectory Metrics & Optimization

Static turnaround renders (`bp.evaluate_fidelity`) only measure the final
`t = 1` pose. Continuous folding animations from the flat `1x1` sheet (`t = 0`)
through the collapsed box-pleated base (`t = 0.37`) to the 3D sculpture
(`t = 1.0`) fail physically unless optimized as a constrained trajectory
problem:

- **Spanning-tree base collapse tears 2D box pleats**: Folding closed degree-4
  and degree-8 box-pleat cycles along a rigid face tree cuts every non-tree
  crease, opening `6g–17g` seam gaps and inflating mid-fold surface area by
  `20%–168%`.
- **Act I (`0 -> 0.37`) requires an unbroken triangulated sheet**: Use
  `docs/js/xpbd.js` and `docs/js/optimize_trajectory.mjs` to fold the sheet as
  one connected mesh with unwrapped `atan2` signed dihedral springs, soft `F`
  facet-diagonal compliance (`0.08` vs `0.44` on `M/V` creases), and axial bar
  inextensibility (`0.00g` seam tear).
- **Acts II–IV (`0.41 -> 1.00`) require Short-Hop Radial Dijkstra routing**:
  Route Stage-2 shaping from an interior root face with heavy penalties on
  unfolding creases (`32–60`), keep 180-degree narrowing folds on the table in
  Act II, pose limbs proximally-to-distally in Act III, and defer volumetric
  wing/plume/hood/crest fan-out to Act IV (`t >= 0.82`) so compressed paper
  never re-expands mid-fold.

## 1. Quick Start CLI (`docs/js/optimize_trajectory.mjs`)

Run from the repository root with Node (`docs/js/package.json` configures native
ES modules):

```bash
# 1. Benchmark all showcase models in docs/models/catalog.json (~26 s):
node docs/js/optimize_trajectory.mjs --mode=benchmark --models-dir=docs/models

# 2. Score a single model (with optional schedule.json) across M1..M8:
node docs/js/optimize_trajectory.mjs --mode=score \
  --fold=docs/models/golden-eagle/shaped_3d.fold \
  --stick-figure=docs/models/golden-eagle/stick_figure.json \
  --schedule=docs/models/golden-eagle/schedule.json

# 3. Optimize Act I (XPBD waves) + Stage 2 (Short-Hop SA) and write schedule:
node docs/js/optimize_trajectory.mjs --mode=optimize \
  --fold=docs/models/<slug>/shaped_3d.fold \
  --stick-figure=docs/models/<slug>/stick_figure.json \
  --out=docs/models/<slug>/schedule.json
```

Pass `--fast` for rapid smoke checks during iteration and `--json` for
machine-readable output.

## 2. Precondition: Verify Base Topology Before Optimizing

Before optimizing a trajectory, confirm that `shaped_3d.fold` preserves the
solved base's flat-foldable creases (`edges_bp:baseFoldAngle`):

1. **`edges_bp:shaping` / `edges_bp:baseFoldAngle` consistency**:
   - Every `M`/`V` edge must satisfy `shaping[e] == (baseFoldAngle[e] == 0)`.
   - Every border edge (`B`) and shaping-only edge (`shaping[e] == true`) must
     have `baseFoldAngle[e] == 0`.
2. **Interior Maekawa parity (`M1` hard gate)**:
   - Every interior vertex with nonzero base degree must satisfy
     `|M_base - V_base| == 2`.
   - Fresh `bp-pipeline` outputs satisfy this automatically. If an older
     snapshot lost solved-base subsegments when shaping split edges, overlay the
     missing solved-base segments via a 3D-preserving planar arrangement (linear
     3D interpolation along split shaped edges, affine triangle interpolation
     inside faces, `foldAngle = 0` and `assignment = 'F'` on base-only interior
     splits, `frame_attributes: ['3D']` on root and `['2D']` on
     `file_frames[0]`).

## 3. The 8-Metric Physicality Scorecard (`docs/js/scorecard.js`)

All 8 gates weight triangles by flat 2D area and edges by flat 2D length so
subdividing a mesh never biases the score. Full mathematical definitions,
thresholds, and `schedule.json` fields live in
[metrics_and_schema.md](references/metrics_and_schema.md).

<!-- disableFinding(LINE_OVER_80) -->

| Gate | Metric | Pass Gate | Common Failure Cause & Remedy |
| :--- | :--- | :--- | :--- |
| `M1` | Endpoint Fidelity | `0` Maekawa viol, `<= 0.8%` strain | Missing base subsegments (`baseMaekawaViol > 0`) or mismatched handoff tree (`posBase` must use `_shapingTree` at `rootFace`). |
| `M2` | Surface-Area Conservation | `<= 6.5%` (`maxSurfAreaDevPct`) | Rigid spanning-tree cuts or linear keyframe secant shrinkage. Use XPBD Act-I waves (`nK >= 29`) + continuous open-pleat bar polish. |
| `M3` | Warped Paper Area (`>15%`) | `<= 8.5%` of sheet | Rigid polygonal faces locking 2D box-pleat cycles. Keep triangulated `F` facet-diagonal gain soft (`0.08` vs `0.44` on `M/V`). |
| `M4` | Crease Inextensibility | `<= 12.0%` (`lw-p95`), `0.00g` seam | Insufficient bar sweeps per XPBD step (run `2` sweeps with `corr = 0.49` and decay dihedral gain on final `4–6` sub-iterations). |
| `M5` | Layer-Aware Piercings | `<= 10.0%` (`Stage 2`), minimized (`Act I`) | Counting coplanar stacked plies as hits (filter by one-sided `epsPly = 1.5e-3` and subtract `baselinePiercedFaces` in Stage 2), or locking pleats too early (prefer `Linear-Angle` or `Gentle Isometric` gather). |
| `M6` | Compression Monotonicity | `<= 2.0%` $R_g$ rebound (`t <= 0.825`) | Measuring axis-aligned bounding-box volume (rotations look like expansions) or firing wing/hood/plume fan-out before Act IV (`t < 0.80`). |
| `M7` | Table Accordion Height | Median `0.45g–6.5g`, peak `<= 6.5g` | Dead-flat `0g` pancake projection or `>10g` ballooning. Tune `zCalm` (`0.04–0.14`) and `tetherPow` (`2.3–3.6`), and frame camera against `tableRef`. |
| `M8` | Temporal Smoothness | `<= 3.0x` spike, `<= 14%` idle | Dead time gaps between phases or sudden late snaps. Apply `resampleKeyframesByArcLength` in Act I and `buildArcLengthTimeMap` (`s2TimeMap`) in Stage 2. |

<!-- enableFinding(LINE_OVER_80) -->

## 4. Non-Obvious Solver & Serialization Gotchas

- **Signed `atan2` dihedral vs unsigned `acos`**: Unsigned `acos(n1 . n2)`
  cannot distinguish Mountain (`-180` deg) from Valley (`+180` deg) and folds
  adjacent pleats onto the same side. Always compute
  `atan2(-(u . (n1 x n2)), n1 . n2)` along oriented shared edge `u` and unwrap
  by `+-2*pi` against `prevTh[h]` when `|th - pv| > 4.5` rad.
- **Zero-discontinuity Stage-2 handoff (`t = 0.37 -> 0.41`)**: Lock
  `keyframes[K - 1]` to `evalFolder.foldByHingesRaw(baseFoldAngles)` evaluated
  on `_shapingTree` at `schedule.rootFace`. Any root-face or tree mismatch
  between Act I's final frame and Stage 2's initial tree causes a visible jump
  across `[0.37, 0.41]`.
- **Compact 2-byte radix-41 `act1DeltaQ` encoding**:
  `quantizeAct1KeyframesDeltaQ` quantizes interior keyframes `1..K-2` (or
  `1..K-1` when `includesBaseFrame` is `true` for strained level-shifted bases)
  to `0..1680` (`41^2 - 1`) and encodes each coordinate in 2 non-letter
  printable ASCII bytes (`encodeAct1CoordsQ` / `decodeAct1CoordsQ` in
  `docs/js/fold.js`, `~2.5x` smaller than JSON integer arrays with `< 2 ms`
  hydration). Using a letter-free alphabet keeps the payload compact without
  embedding arbitrary alphanumeric sequences in checked-in JSON files.
- **Keep reproducible `.fold` snapshots intact**: If a showcase model ships
  `shaping_history.json` (tested by `showcase_replay_test`), optimize and ship
  `schedule.json` without rewriting its `shaped_3d.fold`. If mesh coordinates,
  edges, or faces change later, `hydrateSchedule` checks `(numVertices,
  numEdges, meshHash)` and falls back cleanly to non-blocking
  `buildFoldingSchedule` + background `xpbd_worker.js`.

## 5. Verification Checklist Before Committing

1. Run `node docs/js/optimize_trajectory.mjs --mode=score` on every added or
   updated model and verify `M1` (`0` Maekawa violations, `<= 0.8%` strain),
   `M2` (`<= 6.5%` Act-I area deviation), `M6` (`<= 2.0%` $R_g$ rebound), and
   `0.00g` Act-I seam gap.
2. Run `pytest tests/showcase_test.py tests/stick_figure_skill_test.py` (or the
   equivalent test runner) to verify FOLD invariants, Maekawa parity, and Node
   playback area conservation across the entire catalog.
