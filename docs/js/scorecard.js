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

/**
 * @fileoverview 8-Metric Physicality Scorecard for Origami Folding Trajectories.
 *
 * Every primary gate (`M1..M8`) is area- or length-weighted over the physical
 * `1x1` flat paper sheet, rotation-invariant, and solver-agnostic (evaluating
 * spanning-tree hinge folders and single-mesh Bar-and-Hinge / XPBD solvers
 * identically):
 *
 *   M1. Endpoint Fidelity (`baseMaekawaViol`, `lwMeanEdgeStrainPct`, `stripWidthG`)
 *   M2. Surface-Area Conservation (`maxSurfAreaDevPct`)
 *   M3. Warped Paper Area (`maxWarpedAreaPct >15%`, `maxAreaP95StrainPct`)
 *   M4. Crease Inextensibility & Seam Gap (`maxLwP95EdgeStrainPct`, `maxP95SeamGapG`)
 *   M5. Layer-Aware Transient Piercings (`meanTransientPiercedAreaPct`, `sumFaceIX`)
 *   M6. Spatial Compression Monotonicity (`rgOvershootPct` — rotation-invariant R_g)
 *   M7. Table-Grounded Accordion Profile (`medianP90HeightG`, `maxP90HeightG`)
 *   M8. Temporal Velocity Smoothness (`speedSpikeRatio`, `frozenFramePct`)
 *
 * Free of DOM and three.js dependencies, so it runs identically in browsers
 * and under Node.
 */

import {countPhysicalIntersections} from './intersections.js';

/** Calibrated pass/warn thresholds anchored on the Stage-2 reference runs. */
export const SCORECARD_GATES = {
  endpointStrainPct: {pass: 0.8, warn: 2.5},
  surfAreaDevPct: {pass: 6.5, warn: 15.0},
  warpedAreaPct: {pass: 8.5, warn: 18.0},
  lwP95EdgeStrainPct: {pass: 12.0, warn: 25.0},
  transientPiercedAreaPct: {pass: 10.0, warn: 22.0},
  rgOvershootPct: {pass: 2.0, warn: 5.0},
  act1MinP90HeightG: {pass: 0.45, warn: 0.22},
  act1MaxP90HeightG: {pass: 6.50, warn: 10.50},
  speedSpikeRatio: {pass: 3.00, warn: 4.20},
  frozenFramePct: {pass: 14.0, warn: 26.0},
};

/**
 * Exact O(N) iterative 3-way weighted quickselect over `(values, weights)`.
 *
 * @param {!Float64Array} values
 * @param {!Float64Array} weights
 * @param {number} totalWeight
 * @param {!Int32Array} idxBuf
 * @param {number} n
 * @param {number} q Quantile in `[0, 1]`.
 * @return {number}
 */
function weightedQuantile(values, weights, totalWeight, idxBuf, n, q) {
  if (n <= 0 || totalWeight <= 0) return 0;
  for (let i = 0; i < n; i++) idxBuf[i] = i;
  let target = q * totalWeight;
  let lo = 0;
  let hi = n - 1;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    const pivotVal = values[idxBuf[mid]];
    let lt = lo;
    let i = lo;
    let gt = hi;
    let wLt = 0;
    let wEq = 0;
    while (i <= gt) {
      const idx = idxBuf[i];
      const v = values[idx];
      if (v < pivotVal) {
        wLt += weights[idx];
        idxBuf[i] = idxBuf[lt];
        idxBuf[lt] = idx;
        lt++;
        i++;
      } else if (v > pivotVal) {
        idxBuf[i] = idxBuf[gt];
        idxBuf[gt] = idx;
        gt--;
      } else {
        wEq += weights[idx];
        i++;
      }
    }
    if (target <= wLt) {
      hi = lt - 1;
    } else if (target <= wLt + wEq) {
      return pivotVal;
    } else {
      target -= (wLt + wEq);
      lo = gt + 1;
    }
  }
  return values[idxBuf[lo]];
}

/**
 * Precomputes static flat-sheet area/length weights, face areas, vertex areas,
 * and Maekawa counts once per model.
 *
 * @param {!Object} model Parsed FOLD `Model` from `fold.js`.
 * @param {!Object} collisionCtx Built by `buildCollisionContext(model)`.
 * @return {!Object}
 */
export function buildScorecardContext(model, collisionCtx) {
  const {flat, faces, edges, assignments, baseFoldAngles, shaping, gridSize} =
      model;
  const nV = flat.length / 2;
  const nF = faces.length;
  const gUnit = 1 / Math.max(gridSize || 28, 12);
  const {tris, numTris} = collisionCtx;

  const flatTriArea = new Float64Array(numTris);
  const flatFaceArea = new Float64Array(nF);
  const vertArea = new Float64Array(nV);
  let flatTotalArea = 0;

  for (let t = 0; t < numTris; t++) {
    const a = tris[4 * t];
    const b = tris[4 * t + 1];
    const c = tris[4 * t + 2];
    const fIdx = tris[4 * t + 3];
    const ax = flat[2 * b] - flat[2 * a];
    const ay = flat[2 * b + 1] - flat[2 * a + 1];
    const bx = flat[2 * c] - flat[2 * a];
    const by = flat[2 * c + 1] - flat[2 * a + 1];
    const ar = 0.5 * Math.abs(ax * by - ay * bx);
    flatTriArea[t] = ar;
    flatFaceArea[fIdx] += ar;
    vertArea[a] += ar / 3;
    vertArea[b] += ar / 3;
    vertArea[c] += ar / 3;
    flatTotalArea += ar;
  }
  flatTotalArea = Math.max(flatTotalArea, 1e-8);

  let cx0 = 0;
  let cy0 = 0;
  for (let v = 0; v < nV; v++) {
    cx0 += vertArea[v] * flat[2 * v];
    cy0 += vertArea[v] * flat[2 * v + 1];
  }
  cx0 /= flatTotalArea;
  cy0 /= flatTotalArea;
  let rgSq0 = 0;
  for (let v = 0; v < nV; v++) {
    const dx = flat[2 * v] - cx0;
    const dy = flat[2 * v + 1] - cy0;
    rgSq0 += vertArea[v] * (dx * dx + dy * dy);
  }
  const flatRg = Math.max(0.1, Math.sqrt(rgSq0 / flatTotalArea));

  const edgePairs = [];
  const edgeLens = [];
  let totalEdgeLen = 0;
  for (let e = 0; e < edges.length; e++) {
    const [u, v] = edges[e];
    const len =
        Math.hypot(flat[2 * u] - flat[2 * v], flat[2 * u + 1] - flat[2 * v + 1]);
    if (len > 1e-6) {
      edgePairs.push(u, v);
      edgeLens.push(len);
      totalEdgeLen += len;
    }
  }

  const vertFaces = Array.from({length: nV}, () => []);
  faces.forEach((fc, fIdx) => {
    for (const v of fc) vertFaces[v].push(fIdx);
  });

  const border = new Uint8Array(nV);
  const touchesShaping = new Uint8Array(nV);
  const mCnt = new Int32Array(nV);
  const vCnt = new Int32Array(nV);
  for (let v = 0; v < nV; v++) {
    const x = flat[2 * v];
    const y = flat[2 * v + 1];
    if (x <= 1e-5 || x >= 1 - 1e-5 || y <= 1e-5 || y >= 1 - 1e-5) {
      border[v] = 1;
    }
  }
  edges.forEach(([u, v], e) => {
    if (assignments[e] === 'B') {
      border[u] = 1;
      border[v] = 1;
      return;
    }
    if (shaping && shaping[e]) {
      touchesShaping[u] = 1;
      touchesShaping[v] = 1;
    }
    const a = baseFoldAngles[e];
    if (a < -1) {
      mCnt[u]++;
      mCnt[v]++;
    } else if (a > 1) {
      vCnt[u]++;
      vCnt[v]++;
    }
  });
  let baseMaekawaViol = 0;
  for (let v = 0; v < nV; v++) {
    if (border[v] || touchesShaping[v]) continue;
    if (mCnt[v] + vCnt[v] > 0 && Math.abs(mCnt[v] - vCnt[v]) !== 2) {
      baseMaekawaViol++;
    }
  }

  const maxN = Math.max(numTris, edgeLens.length, nV, 1);
  return {
    nV,
    nF,
    gUnit,
    gridSize: gridSize || 28,
    flatTotalArea,
    flatRg,
    flatTriArea,
    flatFaceArea,
    vertArea,
    edgePairs: Int32Array.from(edgePairs),
    edgeLens: Float64Array.from(edgeLens),
    totalEdgeLen: Math.max(totalEdgeLen, 1e-8),
    vertFaces,
    baseMaekawaViol,
    triStrainBuf: new Float64Array(Math.max(1, numTris)),
    edgeStrainBuf: new Float64Array(Math.max(1, edgeLens.length)),
    seamGapBuf: new Float64Array(nV),
    seamWeightBuf: new Float64Array(nV),
    zRelBuf: new Float64Array(nV),
    sortIdxBuf: new Int32Array(maxN),
  };
}

/**
 * Evaluates all single-frame physicality diagnostics on `positions`.
 *
 * @param {!Object} model
 * @param {!Float32Array|!Float64Array} positions
 * @param {!Object} collisionCtx
 * @param {!Object} scoreCtx
 * @param {?Object=} folder Optional `Folder` instance for spanning-tree seam
 *     diagnostics (`null` when single-mesh XPBD keyframes are active).
 * @param {boolean=} collectFaceSets Whether to populate `intersectingFaces` and
 *     `warpedFaces` sets.
 * @param {?Set<number>=} baselinePiercedFaces Optional set of faces already
 *     pierced in the static target endpoint (`t = 1.0` in Stage 2).
 * @return {!Object}
 */
export function evaluateFramePhysicality(
    model,
    positions,
    collisionCtx,
    scoreCtx,
    folder = null,
    collectFaceSets = false,
    baselinePiercedFaces = null) {
  const {flat} = model;
  const {tris, numTris} = collisionCtx;
  const {
    nV,
    gUnit,
    flatTotalArea,
    flatTriArea,
    flatFaceArea,
    vertArea,
    edgePairs,
    edgeLens,
    totalEdgeLen,
    vertFaces,
    triStrainBuf,
    edgeStrainBuf,
    seamGapBuf,
    seamWeightBuf,
    zRelBuf,
    sortIdxBuf,
  } = scoreCtx;

  let curTotalArea = 0;
  let warpedArea = 0;
  const warpedFaces = collectFaceSets ? new Set() : null;
  for (let t = 0; t < numTris; t++) {
    const a = tris[4 * t];
    const b = tris[4 * t + 1];
    const c = tris[4 * t + 2];
    const ux = positions[3 * b] - positions[3 * a];
    const uy = positions[3 * b + 1] - positions[3 * a + 1];
    const uz = positions[3 * b + 2] - positions[3 * a + 2];
    const vx = positions[3 * c] - positions[3 * a];
    const vy = positions[3 * c + 1] - positions[3 * a + 1];
    const vz = positions[3 * c + 2] - positions[3 * a + 2];
    const ar3 = 0.5 *
        Math.hypot(uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx);
    curTotalArea += ar3;
    const ar0 = flatTriArea[t];
    const strainPct = ar0 > 1e-9 ? 100 * Math.abs(ar3 / ar0 - 1) : 0;
    triStrainBuf[t] = strainPct;
    if (strainPct > 15.0) {
      warpedArea += ar0;
      if (warpedFaces) warpedFaces.add(tris[4 * t + 3]);
    }
  }
  const surfAreaRatio = curTotalArea / flatTotalArea;
  const surfAreaDevPct = 100 * Math.abs(surfAreaRatio - 1);
  const warpedAreaPct = 100 * (warpedArea / flatTotalArea);
  const areaP95StrainPct = weightedQuantile(
      triStrainBuf, flatTriArea, flatTotalArea, sortIdxBuf, numTris, 0.95);

  const nE = edgeLens.length;
  let lwStrainSum = 0;
  for (let i = 0; i < nE; i++) {
    const u = edgePairs[2 * i];
    const v = edgePairs[2 * i + 1];
    const len0 = edgeLens[i];
    const len1 = Math.hypot(
        positions[3 * u] - positions[3 * v],
        positions[3 * u + 1] - positions[3 * v + 1],
        positions[3 * u + 2] - positions[3 * v + 2]);
    const pct = 100 * Math.abs(len1 / len0 - 1);
    edgeStrainBuf[i] = pct;
    lwStrainSum += len0 * pct;
  }
  const lwMeanEdgeStrainPct = lwStrainSum / totalEdgeLen;
  const lwP95EdgeStrainPct = weightedQuantile(
      edgeStrainBuf, edgeLens, totalEdgeLen, sortIdxBuf, nE, 0.95);

  let p95SeamGapG = 0;
  const m = folder?._transforms ?? null;
  const reached = folder?._reached ?? null;
  if (m && reached) {
    let countV = 0;
    let wSum = 0;
    for (let v = 0; v < nV; v++) {
      const fList = vertFaces[v];
      if (fList.length <= 1) continue;
      const fx = flat[2 * v];
      const fy = flat[2 * v + 1];
      let x0 = 1e9;
      let x1 = -1e9;
      let y0 = 1e9;
      let y1 = -1e9;
      let z0 = 1e9;
      let z1 = -1e9;
      let seen = 0;
      for (let i = 0; i < fList.length; i++) {
        const f = fList[i];
        if (!reached[f]) continue;
        const t = 12 * f;
        const qx = m[t] * fx + m[t + 1] * fy + m[t + 3];
        const qy = m[t + 4] * fx + m[t + 5] * fy + m[t + 7];
        const qz = m[t + 8] * fx + m[t + 9] * fy + m[t + 11];
        if (qx < x0) x0 = qx;
        if (qx > x1) x1 = qx;
        if (qy < y0) y0 = qy;
        if (qy > y1) y1 = qy;
        if (qz < z0) z0 = qz;
        if (qz > z1) z1 = qz;
        seen++;
      }
      if (seen >= 2) {
        const w = Math.max(vertArea[v], 1e-8);
        seamGapBuf[countV] = Math.hypot(x1 - x0, y1 - y0, z1 - z0) / gUnit;
        seamWeightBuf[countV] = w;
        wSum += w;
        countV++;
      }
    }
    p95SeamGapG = weightedQuantile(
        seamGapBuf, seamWeightBuf, wSum, sortIdxBuf, countV, 0.95);
  }

  let xMin = 1e9;
  let xMax = -1e9;
  let yMin = 1e9;
  let yMax = -1e9;
  let zMin = 1e9;
  let zMax = -1e9;
  let cx = 0;
  let cy = 0;
  let cz = 0;
  for (let v = 0; v < nV; v++) {
    const px = positions[3 * v];
    const py = positions[3 * v + 1];
    const pz = positions[3 * v + 2];
    const w = vertArea[v];
    cx += w * px;
    cy += w * py;
    cz += w * pz;
    if (px < xMin) xMin = px;
    if (px > xMax) xMax = px;
    if (py < yMin) yMin = py;
    if (py > yMax) yMax = py;
    if (pz < zMin) zMin = pz;
    if (pz > zMax) zMax = pz;
  }
  cx /= flatTotalArea;
  cy /= flatTotalArea;
  cz /= flatTotalArea;

  let rgSq = 0;
  for (let v = 0; v < nV; v++) {
    const dx = positions[3 * v] - cx;
    const dy = positions[3 * v + 1] - cy;
    const dz = positions[3 * v + 2] - cz;
    rgSq += vertArea[v] * (dx * dx + dy * dy + dz * dz);
    zRelBuf[v] = (positions[3 * v + 2] - zMin) / gUnit;
  }
  const rg3D = Math.sqrt(rgSq / flatTotalArea);
  const p90HeightG = weightedQuantile(
      zRelBuf, vertArea, flatTotalArea, sortIdxBuf, nV, 0.90);
  const zSpanG = (zMax - zMin) / gUnit;

  const ix = countPhysicalIntersections(positions, collisionCtx, 1.5e-3, true);
  let piercedArea = 0;
  let transientPiercedArea = 0;
  for (const fIdx of ix.intersectingFaces) {
    const fa = flatFaceArea[fIdx] || 0;
    piercedArea += fa;
    if (!baselinePiercedFaces || !baselinePiercedFaces.has(fIdx)) {
      transientPiercedArea += fa;
    }
  }
  const piercedAreaPct = 100 * (piercedArea / flatTotalArea);
  const transientPiercedAreaPct = 100 * (transientPiercedArea / flatTotalArea);

  return {
    surfAreaRatio: +surfAreaRatio.toFixed(4),
    surfAreaDevPct: +surfAreaDevPct.toFixed(2),
    warpedAreaPct: +warpedAreaPct.toFixed(2),
    areaP95StrainPct: +areaP95StrainPct.toFixed(1),
    lwMeanEdgeStrainPct: +lwMeanEdgeStrainPct.toFixed(2),
    lwP95EdgeStrainPct: +lwP95EdgeStrainPct.toFixed(1),
    p95SeamGapG: +p95SeamGapG.toFixed(2),
    rg3D: +rg3D.toFixed(4),
    p90HeightG: +p90HeightG.toFixed(2),
    zSpanG: +zSpanG.toFixed(2),
    stripWidthG: +(Math.min(xMax - xMin, yMax - yMin) / gUnit).toFixed(2),
    facePairHits: ix.facePairHits,
    piercedAreaPct: +piercedAreaPct.toFixed(2),
    transientPiercedAreaPct: +transientPiercedAreaPct.toFixed(2),
    intersectingFaces: collectFaceSets ? ix.intersectingFaces : null,
    warpedFaces,
  };
}

/**
 * Grades an upper-bounded metric against pass/warn thresholds.
 *
 * @param {number} val
 * @param {{pass: number, warn: number}} gate
 * @return {string}
 */
function gradeUpper(val, gate) {
  if (val <= gate.pass) return 'PASS';
  if (val <= gate.warn) return 'WARN';
  return 'FAIL';
}

/**
 * Evaluates the 8-Metric Physicality Scorecard across a trajectory window
 * (`stageKind === 'act1'` for `[0, baseCompleteT]`, `'stage2'` for
 * `[shapingStartT, 1.0]`).
 *
 * @param {!Object} model
 * @param {!Object} folder `Folder` instance with schedule attached.
 * @param {!Object} collisionCtx
 * @param {!Object} scoreCtx
 * @param {string=} stageKind `'act1'` or `'stage2'`.
 * @param {number=} nSamples Uniform sample count across the window.
 * @param {number=} baseCompleteT
 * @param {number=} shapingStartT
 * @return {!Object}
 */
export function evaluateWindowScorecard(
    model,
    folder,
    collisionCtx,
    scoreCtx,
    stageKind = 'act1',
    nSamples = 19,
    baseCompleteT = 0.37,
    shapingStartT = 0.41) {
  const isAct1 = stageKind === 'act1';
  const tStart = isAct1 ? 0.0 : shapingStartT;
  const tEnd = isAct1 ? baseCompleteT : 1.0;

  const endRefPos = Float32Array.from(folder.positionsAt(tEnd, true));
  const baselinePiercedFaces = isAct1 ?
      null :
      countPhysicalIntersections(endRefPos, collisionCtx, 1.5e-3, true)
          .intersectingFaces;

  const singleMeshAct1 = Boolean(
      folder._skipSeamDiag ||
      (isAct1 && folder._act1Keyframes && folder._act1Keyframes.length >= 2));

  let maxSurfAreaDevPct = 0;
  let worstSurfAreaT = tStart;
  let maxWarpedAreaPct = 0;
  let maxAreaP95StrainPct = 0;
  let worstWarpedT = tStart;
  let maxLwP95EdgeStrainPct = 0;
  let maxP95SeamGapG = 0;
  let worstCreaseT = tStart;
  let sumFaceIX = 0;
  let maxTransientPiercedPct = 0;
  let sumTransientPiercedPct = 0;
  let worstPiercingT = tStart;

  let minRgSeen = Infinity;
  let maxRgRebound = 0;
  let worstEnvT = tStart;

  const activeHeightsG = [];
  let maxP90HeightG = 0;
  let worstHeightT = tStart;

  const stepRms = [];
  const stepTs = [];
  let prevPos = null;
  let endFrame = null;

  for (let k = 0; k < nSamples; k++) {
    const u = k / (nSamples - 1);
    const t = tStart + u * (tEnd - tStart);
    const pos = Float32Array.from(folder.positionsAt(t, true));
    const fr = evaluateFramePhysicality(
        model, pos, collisionCtx, scoreCtx,
        singleMeshAct1 ? null : folder, false, baselinePiercedFaces);

    if (k === nSamples - 1) {
      endFrame = fr;
    }

    if (k > 0 && k < nSamples - 1) {
      if (fr.surfAreaDevPct > maxSurfAreaDevPct) {
        maxSurfAreaDevPct = fr.surfAreaDevPct;
        worstSurfAreaT = t;
      }
      if (fr.warpedAreaPct > maxWarpedAreaPct) {
        maxWarpedAreaPct = fr.warpedAreaPct;
        worstWarpedT = t;
      }
      if (fr.areaP95StrainPct > maxAreaP95StrainPct) {
        maxAreaP95StrainPct = fr.areaP95StrainPct;
      }
      if (fr.lwP95EdgeStrainPct > maxLwP95EdgeStrainPct) {
        maxLwP95EdgeStrainPct = fr.lwP95EdgeStrainPct;
        worstCreaseT = t;
      }
      if (fr.p95SeamGapG > maxP95SeamGapG) {
        maxP95SeamGapG = fr.p95SeamGapG;
      }

      sumFaceIX += fr.facePairHits;
      sumTransientPiercedPct += fr.transientPiercedAreaPct;
      if (fr.transientPiercedAreaPct > maxTransientPiercedPct) {
        maxTransientPiercedPct = fr.transientPiercedAreaPct;
        worstPiercingT = t;
      }

      if (isAct1 && u >= 0.16 && u <= 0.84) {
        activeHeightsG.push(fr.p90HeightG);
        if (fr.p90HeightG > maxP90HeightG) {
          maxP90HeightG = fr.p90HeightG;
          worstHeightT = t;
        }
      } else if (!isAct1 && fr.p90HeightG > maxP90HeightG) {
        maxP90HeightG = fr.p90HeightG;
        worstHeightT = t;
      }
    }

    // Rotation-invariant spatial compression monotonicity before Act IV fan-out:
    if (isAct1 || (folder.lastState?.warpedT ?? t) <= 0.825) {
      if (fr.rg3D < minRgSeen) {
        minRgSeen = fr.rg3D;
      } else if (fr.rg3D - minRgSeen > maxRgRebound) {
        maxRgRebound = fr.rg3D - minRgSeen;
        worstEnvT = t;
      }
    }

    if (prevPos) {
      let wSq = 0;
      for (let v = 0; v < scoreCtx.nV; v++) {
        const dx = pos[3 * v] - prevPos[3 * v];
        const dy = pos[3 * v + 1] - prevPos[3 * v + 1];
        const dz = pos[3 * v + 2] - prevPos[3 * v + 2];
        wSq += scoreCtx.vertArea[v] * (dx * dx + dy * dy + dz * dz);
      }
      stepRms.push(Math.sqrt(wSq / scoreCtx.flatTotalArea));
      stepTs.push(t);
    }
    prevPos = pos;
  }

  // Also audit the `[baseCompleteT, shapingStartT]` handoff bridge during Act I
  // so linear `_baseBridgeDelta` crushes on strained bases cannot slip through
  // the gap between the Act-I and Stage-2 windows.
  if (isAct1 && shapingStartT > baseCompleteT + 1e-6) {
    for (const frac of [0.25, 0.5, 0.75, 1.0]) {
      const tBridge = baseCompleteT + frac * (shapingStartT - baseCompleteT);
      const pos = Float32Array.from(folder.positionsAt(tBridge, true));
      let changed = false;
      for (let i = 0; i < pos.length; i++) {
        if (Math.abs(pos[i] - endRefPos[i]) > 1e-6) {
          changed = true;
          break;
        }
      }
      if (!changed) continue;
      const fr = evaluateFramePhysicality(
          model, pos, collisionCtx, scoreCtx,
          singleMeshAct1 ? null : folder, false, baselinePiercedFaces);
      if (fr.surfAreaDevPct > maxSurfAreaDevPct) {
        maxSurfAreaDevPct = fr.surfAreaDevPct;
        worstSurfAreaT = tBridge;
      }
      if (fr.warpedAreaPct > maxWarpedAreaPct) {
        maxWarpedAreaPct = fr.warpedAreaPct;
        worstWarpedT = tBridge;
      }
      if (fr.areaP95StrainPct > maxAreaP95StrainPct) {
        maxAreaP95StrainPct = fr.areaP95StrainPct;
      }
      if (fr.lwP95EdgeStrainPct > maxLwP95EdgeStrainPct) {
        maxLwP95EdgeStrainPct = fr.lwP95EdgeStrainPct;
        worstCreaseT = tBridge;
      }
      if (fr.p95SeamGapG > maxP95SeamGapG) {
        maxP95SeamGapG = fr.p95SeamGapG;
      }
      if (fr.transientPiercedAreaPct > maxTransientPiercedPct) {
        maxTransientPiercedPct = fr.transientPiercedAreaPct;
        worstPiercingT = tBridge;
      }
      if (fr.rg3D < minRgSeen) {
        minRgSeen = fr.rg3D;
      } else if (fr.rg3D - minRgSeen > maxRgRebound) {
        maxRgRebound = fr.rg3D - minRgSeen;
        worstEnvT = tBridge;
      }
    }
  }

  const interiorCount = Math.max(1, nSamples - 2);
  const meanTransientPiercedAreaPct = sumTransientPiercedPct / interiorCount;
  const rgOvershootPct = 100 * (maxRgRebound / scoreCtx.flatRg);

  activeHeightsG.sort((a, b) => a - b);
  const medianP90HeightG = activeHeightsG.length > 0 ?
      activeHeightsG[Math.floor(0.5 * activeHeightsG.length)] :
      maxP90HeightG;

  const meanStep =
      stepRms.reduce((a, b) => a + b, 0) / Math.max(1, stepRms.length);
  let maxStep = 0;
  let worstSpeedT = tStart;
  let frozenSteps = 0;
  const activeStepLim = isAct1 ? Math.max(1, stepRms.length - 2) : stepRms.length;
  for (let i = 0; i < stepRms.length; i++) {
    if (stepRms[i] > maxStep) {
      maxStep = stepRms[i];
      worstSpeedT = stepTs[i];
    }
    if (i < activeStepLim && stepRms[i] < 0.08 * meanStep) {
      frozenSteps++;
    }
  }
  const speedSpikeRatio = maxStep / Math.max(meanStep, 1e-9);
  const frozenFramePct = 100 * (frozenSteps / Math.max(1, activeStepLim));

  const m1Status = (isAct1 && scoreCtx.baseMaekawaViol > 0) ?
      'FAIL' :
      gradeUpper(
          endFrame.lwMeanEdgeStrainPct, SCORECARD_GATES.endpointStrainPct);
  const m2Status = gradeUpper(maxSurfAreaDevPct, SCORECARD_GATES.surfAreaDevPct);
  const m3Status = gradeUpper(maxWarpedAreaPct, SCORECARD_GATES.warpedAreaPct);
  const m4Status =
      gradeUpper(maxLwP95EdgeStrainPct, SCORECARD_GATES.lwP95EdgeStrainPct);
  const m5Status = gradeUpper(
      meanTransientPiercedAreaPct, SCORECARD_GATES.transientPiercedAreaPct);
  const m6Status = gradeUpper(rgOvershootPct, SCORECARD_GATES.rgOvershootPct);

  let m7Status = 'PASS';
  if (isAct1) {
    if (medianP90HeightG < SCORECARD_GATES.act1MinP90HeightG.warn ||
        maxP90HeightG > SCORECARD_GATES.act1MaxP90HeightG.warn) {
      m7Status = 'FAIL';
    } else if (
        medianP90HeightG < SCORECARD_GATES.act1MinP90HeightG.pass ||
        maxP90HeightG > SCORECARD_GATES.act1MaxP90HeightG.pass) {
      m7Status = 'WARN';
    }
  }

  const sStatus = gradeUpper(speedSpikeRatio, SCORECARD_GATES.speedSpikeRatio);
  const fStatus = gradeUpper(frozenFramePct, SCORECARD_GATES.frozenFramePct);
  const m8Status = (sStatus === 'FAIL' || fStatus === 'FAIL') ?
      'FAIL' :
      ((sStatus === 'WARN' || fStatus === 'WARN') ? 'WARN' : 'PASS');

  const seamNote =
      maxP95SeamGapG > 0.05 ? ` (seam ${maxP95SeamGapG.toFixed(1)}g)` : '';

  const rows = [
    {
      id: 'M1',
      name: isAct1 ? 'Base Endpoint Fidelity' : 'Final Pose Fidelity',
      value: isAct1 ?
          `${endFrame.stripWidthG}g · ${scoreCtx.baseMaekawaViol} viol · ${endFrame.lwMeanEdgeStrainPct.toFixed(2)}% str` :
          `${endFrame.lwMeanEdgeStrainPct.toFixed(2)}% str`,
      gate: isAct1 ? '0 viol, <=0.8% str' : '<=0.8% str',
      status: m1Status,
      worstT: +tEnd.toFixed(3),
    },
    {
      id: 'M2',
      name: 'Surface-Area Conservation',
      value: `±${maxSurfAreaDevPct.toFixed(1)}%`,
      gate: `<= ${SCORECARD_GATES.surfAreaDevPct.pass}%`,
      status: m2Status,
      worstT: +worstSurfAreaT.toFixed(3),
    },
    {
      id: 'M3',
      name: 'Warped Paper Area (>15%)',
      value: `${maxWarpedAreaPct.toFixed(1)}% sheet · p95 ${maxAreaP95StrainPct.toFixed(0)}%`,
      gate: `<= ${SCORECARD_GATES.warpedAreaPct.pass}% sheet`,
      status: m3Status,
      worstT: +worstWarpedT.toFixed(3),
    },
    {
      id: 'M4',
      name: 'Crease Inextensibility (p95)',
      value: `p95 ${maxLwP95EdgeStrainPct.toFixed(1)}%${seamNote}`,
      gate: `<= ${SCORECARD_GATES.lwP95EdgeStrainPct.pass}%`,
      status: m4Status,
      worstT: +worstCreaseT.toFixed(3),
    },
    {
      id: 'M5',
      name: 'Transient Pierced Area',
      value: `${meanTransientPiercedAreaPct.toFixed(1)}% sheet (pk ${maxTransientPiercedPct.toFixed(0)}%)`,
      gate: `<= ${SCORECARD_GATES.transientPiercedAreaPct.pass}% sheet`,
      status: m5Status,
      worstT: +worstPiercingT.toFixed(3),
    },
    {
      id: 'M6',
      name: 'Compression Monotonicity',
      value: `${rgOvershootPct.toFixed(2)}% R_g rebound`,
      gate: `<= ${SCORECARD_GATES.rgOvershootPct.pass}%`,
      status: m6Status,
      worstT: +worstEnvT.toFixed(3),
    },
    {
      id: 'M7',
      name: isAct1 ? 'Table Accordion Height' : '3D Sculpted Height',
      value: isAct1 ?
          `med ${medianP90HeightG.toFixed(2)}g · pk ${maxP90HeightG.toFixed(1)}g` :
          `pk ${maxP90HeightG.toFixed(1)}g`,
      gate: isAct1 ? 'med 0.45–6.5g, pk<=6.5g' : '3D pose',
      status: m7Status,
      worstT: +worstHeightT.toFixed(3),
    },
    {
      id: 'M8',
      name: 'Temporal Smoothness',
      value: `${speedSpikeRatio.toFixed(2)}x spike · ${frozenFramePct.toFixed(0)}% idle`,
      gate: `<= ${SCORECARD_GATES.speedSpikeRatio.pass}x, <=14% idle`,
      status: m8Status,
      worstT: +worstSpeedT.toFixed(3),
    },
  ];

  const passCount = rows.filter((r) => r.status === 'PASS').length;
  const warnCount = rows.filter((r) => r.status === 'WARN').length;
  const failCount = rows.filter((r) => r.status === 'FAIL').length;

  const maekawaPenalty =
      (isAct1 && scoreCtx.baseMaekawaViol > 0) ? 10 : 0;
  const heightPenalty = isAct1 ?
      Math.min(
          10,
          Math.max(0, maxP90HeightG - 6.5) * 0.8 +
              Math.max(0, 0.45 - medianP90HeightG) * 16) :
      0;

  const score100 = Math.max(
      0,
      Math.min(
          100,
          Math.round(
              100 - maekawaPenalty -
              Math.min(24, Math.max(0, maxSurfAreaDevPct - 2.0) * 0.55) -
              Math.min(24, Math.max(0, maxWarpedAreaPct - 2.5) * 0.65) -
              Math.min(18, Math.max(0, maxLwP95EdgeStrainPct - 4.0) * 0.30) -
              Math.min(16, Math.max(0, meanTransientPiercedAreaPct - 3.0) * 0.45) -
              Math.min(8, rgOvershootPct * 1.1) -
              heightPenalty -
              Math.min(
                  8,
                  Math.max(0, speedSpikeRatio - 2.2) * 2.2 +
                      Math.max(0, frozenFramePct - 8.0) * 0.2))));

  return {
    stageKind,
    passCount,
    warnCount,
    failCount,
    score100,
    summary: {
      endpointStrainPct: endFrame.lwMeanEdgeStrainPct,
      stripWidthG: endFrame.stripWidthG,
      baseMaekawaViol: scoreCtx.baseMaekawaViol,
      maxSurfAreaDevPct: +maxSurfAreaDevPct.toFixed(2),
      maxWarpedAreaPct: +maxWarpedAreaPct.toFixed(2),
      maxAreaP95StrainPct: +maxAreaP95StrainPct.toFixed(1),
      maxLwP95EdgeStrainPct: +maxLwP95EdgeStrainPct.toFixed(1),
      maxP95SeamGapG: +maxP95SeamGapG.toFixed(2),
      sumFaceIX,
      meanTransientPiercedAreaPct: +meanTransientPiercedAreaPct.toFixed(2),
      maxTransientPiercedPct: +maxTransientPiercedPct.toFixed(1),
      rgOvershootPct: +rgOvershootPct.toFixed(2),
      medianP90HeightG: +medianP90HeightG.toFixed(2),
      maxP90HeightG: +maxP90HeightG.toFixed(2),
      speedSpikeRatio: +speedSpikeRatio.toFixed(2),
      frozenFramePct: +frozenFramePct.toFixed(1),
    },
    rows,
  };
}

/**
 * Smooth multi-objective scalar loss minimizing all 8 physicality defects
 * (lower is better; used by Stage-1 XPBD wave screening and Stage-2 SA).
 *
 * @param {!Object} sc Result of `evaluateWindowScorecard`.
 * @return {number}
 */
export function scorecardLoss(sc) {
  const s = sc.summary;
  const isAct1 = sc.stageKind === 'act1';
  let loss = 0;
  loss += s.endpointStrainPct * 15.0 +
      (s.baseMaekawaViol > 0 && isAct1 ? 500 : 0);
  loss += s.maxSurfAreaDevPct * 2.4 +
      Math.max(0, s.maxSurfAreaDevPct - 6.5) * 3.8;
  loss += s.maxWarpedAreaPct * 2.2 +
      Math.max(0, s.maxWarpedAreaPct - 8.5) * 3.5;
  loss += s.maxLwP95EdgeStrainPct * 0.42 +
      Math.max(0, s.maxLwP95EdgeStrainPct - 12.0) * 0.65;
  loss += s.meanTransientPiercedAreaPct * 3.6 +
      s.maxTransientPiercedPct * 0.9 +
      Math.max(0, s.meanTransientPiercedAreaPct - 10.0) * 4.5;
  loss += s.rgOvershootPct * 6.5 + Math.max(0, s.rgOvershootPct - 2.0) * 12.0;
  if (isAct1) {
    loss += Math.max(0, s.maxP90HeightG - 6.5) * 3.8;
    loss += Math.max(0, 0.45 - s.medianP90HeightG) * 45.0;
  }
  loss += Math.max(0, s.speedSpikeRatio - 2.3) * 8.5 +
      Math.max(0, s.frozenFramePct - 8.0) * 1.4;
  loss -= sc.score100 * 2.5 + sc.passCount * 8.0 - sc.failCount * 5.0;
  return loss;
}
