# 8-Metric Physicality Scorecard & `schedule.json` Schema Reference

## 1. Exact Metric Definitions (`docs/js/scorecard.js`)

Let the flat crease pattern occupy $[0, 1]^2$ with grid size $G$
(`gUnit = 1 / max(G, 12)`), triangulated flat areas $A_t^0$
($\sum_t A_t^0 = A_{\text{flat}} \approx 1$), vertex area weights
$w_v = \frac{1}{3} \sum_{t \ni v} A_t^0$, and flat crease lengths $\ell_e^0$
($\sum_e \ell_e^0 = L_{\text{flat}}$). Window scorecards sample $N = 19$
uniform steps across Act I ($t \in [0, t_{\text{base}}]$, default $[0, 0.37]$,
plus bridge audits across $(t_{\text{base}}, t_{\text{shape}}]$ when the base
handoff moves) and Stage 2 ($t \in [t_{\text{shape}}, 1.0]$, default
$[0.41, 1.0]$).

### `M1` · Endpoint Fidelity

- **Act I (`t = baseCompleteT`)**:
  - `baseMaekawaViol`: count of non-border vertices with active
    `baseFoldAngles` where $|M - V| \neq 2$ (`FAIL` if $> 0$).
  - `lwMeanEdgeStrainPct`: length-weighted mean edge-length error at
    $t_{\text{end}}$ (`PASS <= 0.8%`, `WARN <= 2.5%`).
  - `stripWidthG`: $\min(\Delta x, \Delta y) / \text{gUnit}$ diagnostic strip
    width in grid units.
- **Stage 2 (`t = 1.0`)**: `lwMeanEdgeStrainPct` (`PASS <= 0.8%`,
  `WARN <= 2.5%`).

### `M2` · Global Surface-Area Conservation

- Per interior frame $k \in \{1, \dots, N-2\}$:
  $$\Delta A_k = 100 \left|\frac{\sum_t A_t(t_k)}{A_{\text{flat}}} - 1\right|$$
- `maxSurfAreaDevPct` $= \max_k \Delta A_k$ (`PASS <= 6.5%`, `WARN <= 15.0%`).
  Catches global shrink-to-point interpolation and rigid-tree blowups.

### `M3` · Warped Paper Area (`> 15%` Triangle Area Strain)

- Per triangle $t$: $\varepsilon_t = 100 |A_t(t_k) / A_t^0 - 1|$.
- `warpedAreaPct`: flat-area share of triangles with $\varepsilon_t > 15\%$;
  `areaP95StrainPct` is the area-weighted 95th percentile of $\varepsilon_t$ via
  $O(N)$ 3-way quickselect.
- Gate on `maxWarpedAreaPct`: `PASS <= 8.5%`, `WARN <= 18.0%`.

### `M4` · Crease Inextensibility & Pre-Averaging Seam Gap

- `lwP95EdgeStrainPct`: length-weighted 95th percentile of
  $100 \left|\frac{\ell_e(t_k)}{\ell_e^0} - 1\right|$ across all edges
  (`PASS <= 12.0%`, `WARN <= 25.0%`).
- `p95SeamGapG`: area-weighted 95th percentile of per-vertex bounding-box spread
  across incident face rigid transforms before vertex averaging (`0.00g` by
  construction for single-mesh XPBD).

### `M5` · Layer-Aware Transient Pierced Sheet Area (`docs/js/intersections.js`)

- Adaptive widest-axis Sweep-and-Prune over triangles `tris`:
  1. Skip triangle pairs from the same FOLD face or sharing $\ge 2$ vertices.
  2. **Stacked-ply filter**: compute signed distances of each triangle's
     vertices to the other triangle's plane; if either triangle lies within
     one-sided tolerance `epsPly = 1.5e-3` ($\min d \ge -\epsilon_{\text{ply}}$
     or $\max d \le \epsilon_{\text{ply}}$), treat as valid parallel stacked
     paper layers and skip.
  3. Test strict interior Möller–Trumbore edge-pierces-triangle
     (`margin = 0.015`) on non-shared edges.
- In Stage 2, faces already intersecting in the static zero-thickness target
  pose at $t = 1.0$ (`baselinePiercedFaces`) are excluded so `M5` measures
  strictly **transient trajectory piercings**.
- `meanTransientPiercedAreaPct`: interior-frame mean of pierced flat face area
  (`PASS <= 10.0%`, `WARN <= 22.0%`).

### `M6` · Rotation-Invariant Spatial Compression Monotonicity ($R_g$)

- Area-weighted 3D radius of gyration around area-weighted centroid
  $\bar{\mathbf{p}}$:
  $$R_g(t) = \sqrt{\frac{1}{A_{\text{flat}}} \sum_v w_v \|\mathbf{p}_v(t) - \bar{\mathbf{p}}(t)\|^2}$$
- Evaluated across all of Act I (including the
  $[t_{\text{base}}, t_{\text{shape}}]$ handoff) and up to warped schedule time
  $t_{\text{warped}} \le 0.825$ in Stage 2 (prior to Act IV wing/plume fan-out).
  Tracks running minimum
  $R_g^{\min}(t) = \min_{\tau \le t} R_g(\tau)$ and maximum rebound
  $\max_t (R_g(t) - R_g^{\min}(t))$.
- `rgOvershootPct` $= 100 \cdot \max_t(R_g(t) - R_g^{\min}(t)) / R_g(0)$
  (`PASS <= 2.0%`, `WARN <= 5.0%`). Unlike bounding-box volume, $R_g$ is
  strictly invariant under global 3D `SO(3)` stand-up rotations.

### `M7` · Table-Grounded Accordion Height Profile

- Area-weighted 90th percentile of $(z_v - z_{\min}) / \text{gUnit}$ across
  $u \in [0.16, 0.84]$ in Act I:
  - `PASS`: median $p_{90}(z) \in [0.45g, 6.50g]$ and peak $p_{90}(z) \le 6.50g$.
  - `WARN`: median $\ge 0.22g$ and peak $\le 10.50g$; `FAIL` otherwise.

### `M8` · Temporal Velocity Smoothness

- Step RMS displacement:
  $$\Delta s_k = \sqrt{\frac{1}{A_{\text{flat}}} \sum_v w_v
    \|\mathbf{p}_v(t_k) - \mathbf{p}_v(t_{k-1})\|^2}$$
- `speedSpikeRatio` $= \max_k \Delta s_k / \overline{\Delta s}$
  (`PASS <= 3.00x`, `WARN <= 4.20x`).
- `frozenFramePct`: percentage of active steps with
  $\Delta s_k < 0.08 \overline{\Delta s}$ (`PASS <= 14%`, `WARN <= 26%`).

---

## 2. Composite `score100` & Optimization Loss (`scorecardLoss`)

- **Headline `score100` (`0..100`)**: Starts at `100` and subtracts bounded
  penalties on Maekawa violations (`10`), `maxSurfAreaDevPct > 2.0%` (up to
  `24`), `maxWarpedAreaPct > 2.5%` (up to `24`), `maxLwP95EdgeStrainPct > 4.0%`
  (up to `18`), `meanTransientPiercedAreaPct > 3.0%` (up to `16`),
  `rgOvershootPct` (up to `8`), Act-I height out-of-band (up to `10`), and speed
  spikes / frozen frames (up to `8`).
- **Continuous Optimizer Loss (`scorecardLoss(sc)`)**: Combines smooth linear
  slopes with hinge-loss kinks at each `PASS` threshold (`6.5%` area, `8.5%`
  warp, `12.0%` crease strain, `10.0%` piercing, `2.0%` $R_g$ rebound, `2.3x`
  speed spike) so Simulated Annealing and wave ranking continue improving
  trajectories even before or after discrete gate crossings.

---

## 3. `schedule.json` Serialization Schema (`hydrateSchedule` in `docs/js/fold.js`)

<!-- disableFinding(LINE_OVER_80) -->

| Field | Type | Purpose |
| :--- | :--- | :--- |
| `numVertices`, `numEdges`, `meshHash` | `number`, `string` | Mesh content and topology guard (`computeModelMeshHash`); if mismatched against `shaped_3d.fold`, `hydrateSchedule` returns `null` and Studio falls back to `buildFoldingSchedule` + background `xpbd_worker.js`. |
| `rootFace` | `number` | Anchor face index (`0 <= rootFace < faces.length`) shared by Act-I `posBase` and Stage-2 `_shapingTree`. |
| `baseCompleteT` | `number` | Schedule `t` where Act I completes (`0.37`). |
| `shapingStartT` | `number` | Schedule `t` where Stage 2 begins (`0.41`; `[0.37, 0.41]` is the collapsed base dwell). |
| `standupStartT` | `number` | Schedule `t` where `standupProgress` begins rotating the model upright (after Act II narrowing). |
| `includesBaseFrame` | `boolean \| undefined` | Optional flag emitted when level-shifter shaping split base creases so spanning-tree `posBase` has cycle strain; when `true`, `act1DeltaQ` includes the XPBD-relieved frame `K - 1` (`1..K - 1`) and `Folder.setSchedule` bridges any residual delta smoothly across `[baseCompleteT, shapingStartT]`. |
| `act1Bounds` | `[x0, y0, z0, x1, y1, z1]` | Quantization bounding box for serialized Act-I XPBD keyframes (`1..K - 2`, or `1..K - 1` when `includesBaseFrame` is `true`). |
| `act1DeltaQ` | `string` | 2-byte-per-coordinate letter-free radix-41 (`0..1680`) keyframe coordinates (`6 * numVertices * M` bytes). Frame `0` is reconstructed from `model.flat`, and frame `K - 1` from `_shapingTree(baseFoldAngles)` unless `includesBaseFrame` is `true`. |
| `s2TimeMap` | `number[] \| null` | Monotone `[0..1]` knot array from `buildArcLengthTimeMap` equalizing Stage-2 RMS velocity (`warpedT` via `warpScheduleProgress` / `unwarpScheduleProgress`). |
| `shapingHopCosts` | `number[] \| null` | Per-edge hop costs (`length === edges.length`) for Stage-2 `buildShortHopDijkstra`. |
| `phases` | `Phase[]` | Ordered Act I–IV stages (`id`, `act`, `stageGroup`, `label`, `tStart`, `tEnd`, `easeOut`, `power`, `edgeIndices`, `deltas`) driving Stage-2 kinematics, HUD captions, 3D crease highlights, and 2D diagram states. |
| `stats` | `Object` | Summary telemetry (`act1Score`, `s2Score`, `maxSurfAreaDevPct`, `warpedAreaPct`, `piercedAreaPct`, `avgActiveFlaps`, `maxActiveFlaps`, `totalFlaps`). |

<!-- enableFinding(LINE_OVER_80) -->
