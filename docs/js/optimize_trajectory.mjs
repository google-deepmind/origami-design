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
 * @fileoverview Deterministic two-stage origami folding trajectory evaluator
 * and optimizer (`M1..M8` Physicality Scorecard + Signed-Dihedral XPBD Act-I
 * wave search + Short-Hop Radial Dijkstra Stage-2 Simulated Annealing).
 *
 * CLI usage (from the repository root):
 *   node docs/js/optimize_trajectory.mjs --mode=benchmark --models-dir=docs/models
 *   node docs/js/optimize_trajectory.mjs --mode=score \
 *     --fold=docs/models/golden-eagle/shaped_3d.fold \
 *     --stick-figure=docs/models/golden-eagle/stick_figure.json \
 *     --schedule=docs/models/golden-eagle/schedule.json
 *   node docs/js/optimize_trajectory.mjs --mode=optimize \
 *     --fold=docs/models/golden-eagle/shaped_3d.fold \
 *     --stick-figure=docs/models/golden-eagle/stick_figure.json \
 *     --out=docs/models/golden-eagle/schedule.json
 */

import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';

import {
  Folder,
  buildShortHopDijkstra,
  computeModelMeshHash,
  encodeAct1CoordsQ,
  hydrateSchedule,
  nearestFaceToCenter,
  parseFold,
} from './fold.js';
import {buildCollisionContext, countMovingFlaps} from './intersections.js';
import {
  DEFAULT_BASE_COMPLETE_T,
  DEFAULT_SHAPING_START_T,
  DEFAULT_STANDUP_START_T,
  buildFoldingSchedule,
  smootherstep,
} from './schedule.js';
import {
  buildScorecardContext,
  evaluateWindowScorecard,
  scorecardLoss,
} from './scorecard.js';
import {buildSignedXPBDSystem, relaxBasePositions} from './xpbd.js';

function createRng(seed = 1234567) {
  let s = seed | 0;
  return () => {
    s = (s + 0x6d2b79f5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function classifyAxis(flat, [u, v]) {
  const dx = Math.abs(flat[2 * v] - flat[2 * u]);
  const dy = Math.abs(flat[2 * v + 1] - flat[2 * u + 1]);
  const len = Math.hypot(dx, dy) || 1;
  if (Math.abs(dx / len - dy / len) < 0.35 && Math.min(dx, dy) / len > 0.25) {
    return 'ridge';
  }
  return dy <= dx ? 'horiz' : 'vert';
}

/**
 * Quantizes interior Act-I XPBD keyframes (`1..K-2`, or `1..K-1` when
 * `includesBaseFrame` is true for strained level-shifted bases) to `0..1680`
 * (`41^2 - 1`) and encodes each coordinate in 2 letter-free radix-41 ASCII
 * bytes (`act1Bounds` + `act1DeltaQ`), matching `hydrateSchedule` in `fold.js`.
 *
 * @param {!Array<!Float32Array>} keyframes Full `[0..K-1]` keyframe buffers.
 * @param {boolean=} includesBaseFrame Whether to include `keyframes[K - 1]`.
 * @return {{act1Bounds: ?Array<number>, act1DeltaQ: ?string}}
 */
export function quantizeAct1KeyframesDeltaQ(
    keyframes, includesBaseFrame = false) {
  if (!keyframes || keyframes.length < 3) {
    return {act1Bounds: null, act1DeltaQ: null};
  }
  const interior =
      includesBaseFrame ? keyframes.slice(1) : keyframes.slice(1, -1);
  const nFloats = interior[0].length;
  let x0 = Infinity;
  let y0 = Infinity;
  let z0 = Infinity;
  let x1 = -Infinity;
  let y1 = -Infinity;
  let z1 = -Infinity;
  for (const fr of interior) {
    for (let i = 0; i < nFloats; i += 3) {
      if (fr[i] < x0) x0 = fr[i];
      if (fr[i] > x1) x1 = fr[i];
      if (fr[i + 1] < y0) y0 = fr[i + 1];
      if (fr[i + 1] > y1) y1 = fr[i + 1];
      if (fr[i + 2] < z0) z0 = fr[i + 2];
      if (fr[i + 2] > z1) z1 = fr[i + 2];
    }
  }
  const bMinX = +x0.toFixed(7);
  const bMinY = +y0.toFixed(7);
  const bMinZ = +z0.toFixed(7);
  const bMaxX = +Math.max(x1, bMinX + 1e-5).toFixed(7);
  const bMaxY = +Math.max(y1, bMinY + 1e-5).toFixed(7);
  const bMaxZ = +Math.max(z1, bMinZ + 1e-5).toFixed(7);

  const invX = 1680 / (bMaxX - bMinX);
  const invY = 1680 / (bMaxY - bMinY);
  const invZ = 1680 / (bMaxZ - bMinZ);

  const qCoords = new Uint16Array(interior.length * nFloats);
  let p = 0;
  for (let k = 0; k < interior.length; k++) {
    const fr = interior[k];
    for (let i = 0; i < nFloats; i += 3) {
      qCoords[p++] =
          Math.max(0, Math.min(1680, Math.round((fr[i] - bMinX) * invX)));
      qCoords[p++] =
          Math.max(0, Math.min(1680, Math.round((fr[i + 1] - bMinY) * invY)));
      qCoords[p++] =
          Math.max(0, Math.min(1680, Math.round((fr[i + 2] - bMinZ) * invZ)));
    }
  }
  return {
    act1Bounds: [bMinX, bMinY, bMinZ, bMaxX, bMaxY, bMaxZ],
    act1DeltaQ: encodeAct1CoordsQ(qCoords),
  };
}

/**
 * Computes a monotone arc-length velocity-equalization knot table (`0..1`)
 * across `[tStart, tEnd]` so area-weighted RMS paper speed stays smooth (`M8`).
 *
 * @param {!Folder} folder
 * @param {!Object} scoreCtx
 * @param {number} tStart
 * @param {number} tEnd
 * @param {number=} nDense
 * @param {number=} nKnots
 * @return {?Float64Array}
 */
export function buildArcLengthTimeMap(
    folder, scoreCtx, tStart, tEnd, nDense = 44, nKnots = 22) {
  const cum = new Float64Array(nDense + 1);
  const prev = Float32Array.from(folder.positionsAt(tStart, true));
  for (let i = 1; i <= nDense; i++) {
    const u = i / nDense;
    const t = tStart + u * (tEnd - tStart);
    const pos = folder.positionsAt(t, true);
    let wSq = 0;
    for (let v = 0; v < scoreCtx.nV; v++) {
      const dx = pos[3 * v] - prev[3 * v];
      const dy = pos[3 * v + 1] - prev[3 * v + 1];
      const dz = pos[3 * v + 2] - prev[3 * v + 2];
      wSq += scoreCtx.vertArea[v] * (dx * dx + dy * dy + dz * dz);
    }
    cum[i] = cum[i - 1] + Math.sqrt(wSq / scoreCtx.flatTotalArea) + 1e-4 / nDense;
    prev.set(pos);
  }
  const totalLen = cum[nDense];
  if (totalLen <= 1e-6) return null;

  const knots = new Float64Array(nKnots + 1);
  knots[0] = 0;
  knots[nKnots] = 1;
  for (let k = 1; k < nKnots; k++) {
    const qLin = k / nKnots;
    const qSmooth = qLin * qLin * (3 - 2 * qLin);
    const targetS = (0.82 * qLin + 0.18 * qSmooth) * totalLen;
    let lo = 0;
    let hi = nDense;
    while (lo + 1 < hi) {
      const mid = (lo + hi) >> 1;
      if (cum[mid] < targetS) lo = mid;
      else hi = mid;
    }
    const frac = (targetS - cum[lo]) / Math.max(1e-9, cum[hi] - cum[lo]);
    knots[k] = (lo + frac) / nDense;
  }
  return knots;
}

function resampleKeyframesByArcLength(rawKeyframes, scoreCtx, sys, nOut) {
  const nIn = rawKeyframes.length;
  const nFloats = rawKeyframes[0].length;
  const cum = new Float64Array(nIn);
  for (let k = 1; k < nIn; k++) {
    const a = rawKeyframes[k - 1];
    const b = rawKeyframes[k];
    let wSq = 0;
    for (let v = 0; v < scoreCtx.nV; v++) {
      const dx = b[3 * v] - a[3 * v];
      const dy = b[3 * v + 1] - a[3 * v + 1];
      const dz = b[3 * v + 2] - a[3 * v + 2];
      wSq += scoreCtx.vertArea[v] * (dx * dx + dy * dy + dz * dz);
    }
    cum[k] = cum[k - 1] + Math.sqrt(wSq / scoreCtx.flatTotalArea) + 1e-5;
  }
  const totalLen = cum[nIn - 1];
  const out = new Array(nOut);
  out[0] = rawKeyframes[0];
  out[nOut - 1] = rawKeyframes[nIn - 1];

  const {barPairs, barLens} = sys;
  const nB = barLens.length;
  for (let j = 1; j < nOut - 1; j++) {
    const q = j / (nOut - 1);
    const qSmooth = q * q * (3 - 2 * q);
    const targetS = (0.84 * q + 0.16 * qSmooth) * totalLen;
    let lo = 0;
    let hi = nIn - 1;
    while (lo + 1 < hi) {
      const mid = (lo + hi) >> 1;
      if (cum[mid] < targetS) lo = mid;
      else hi = mid;
    }
    const frac = (targetS - cum[lo]) / Math.max(1e-9, cum[hi] - cum[lo]);
    const a = rawKeyframes[lo];
    const b = rawKeyframes[hi];
    const frame = new Float32Array(nFloats);
    for (let i = 0; i < nFloats; i++) {
      frame[i] = (1 - frac) * a[i] + frac * b[i];
    }
    const qFade = 1 - smootherstep((q - 0.64) / 0.22);
    const wBar = 4 * frac * (1 - frac) * qFade;
    if (wBar > 1e-5) {
      const gain = 0.44 * wBar;
      for (let sw = 0; sw < 3; sw++) {
        for (let bi = 0; bi < nB; bi++) {
          const i0 = 3 * barPairs[2 * bi];
          const i1 = 3 * barPairs[2 * bi + 1];
          const dx = frame[i1] - frame[i0];
          const dy = frame[i1 + 1] - frame[i0 + 1];
          const dz = frame[i1 + 2] - frame[i0 + 2];
          const d = Math.hypot(dx, dy, dz) || 1e-9;
          const c = gain * (d - barLens[bi]) / d;
          frame[i0] += c * dx;
          frame[i0 + 1] += c * dy;
          frame[i0 + 2] += c * dz;
          frame[i1] -= c * dx;
          frame[i1 + 1] -= c * dy;
          frame[i1 + 2] -= c * dz;
        }
      }
    }
    out[j] = frame;
  }
  return out;
}

function buildWaveAct1Phases(model, waveSpec) {
  const {flat, edges, assignments, baseFoldAngles} = model;
  const hOut = [];
  const hIn = [];
  const vOut = [];
  const vIn = [];
  const rAll = [];
  for (let e = 0; e < edges.length; e++) {
    if (assignments[e] === 'B' || Math.abs(baseFoldAngles[e]) < 1) continue;
    const [u, v] = edges[e];
    const rad = Math.min(
        1,
        Math.hypot(
            0.5 * (flat[2 * u] + flat[2 * v]) - 0.5,
            0.5 * (flat[2 * u + 1] + flat[2 * v + 1]) - 0.5) /
            0.5);
    const out = rad >= 0.48;
    const ax = classifyAxis(flat, edges[e]);
    if (ax === 'horiz') (out ? hOut : hIn).push(e);
    else if (ax === 'vert') (out ? vOut : vIn).push(e);
    else rAll.push(e);
  }
  const T = DEFAULT_BASE_COMPLETE_T;
  const shiftOut = 0.25 * waveSpec.rimLead;
  const shiftIn = 0.75 * waveSpec.rimLead;
  const rawList = [
    ['Outer Horizontal Pleats', hOut, waveSpec.hWin[0] + shiftOut, waveSpec.hWin[1]],
    ['Core Horizontal Pleats', hIn, waveSpec.hWin[0] + shiftIn, waveSpec.hWin[1]],
    ['Outer Vertical Pleats', vOut, waveSpec.vWin[0] + shiftOut, waveSpec.vWin[1]],
    ['Core Vertical Pleats', vIn, waveSpec.vWin[0] + shiftIn, waveSpec.vWin[1]],
    ['45° Miter Ridges & Base Lock', rAll, waveSpec.rWin[0], waveSpec.rWin[1]],
  ].filter(([, es]) => es.length > 0);

  rawList.sort((a, b) => a[2] - b[2]);
  return rawList.map(([title, es, u0, u1], idx) => {
    const t0 = Math.max(0, Math.min(T - 0.04, u0 * T));
    const t1 = idx === rawList.length - 1 ?
        T :
        Math.max(t0 + 0.04, Math.min(T, u1 * T));
    return {
      id: `xpbd_act1_${idx}`,
      act: 'I',
      stageGroup: 'collapse',
      label: `Act I.${idx + 1} · Gather ${title}`,
      tStart: +t0.toFixed(4),
      tEnd: +t1.toFixed(4),
      easeOut: 0.35,
      power: 1.0,
      edgeIndices: es,
      deltas: Float64Array.from(es, (e) => baseFoldAngles[e]),
    };
  });
}

function solveCandidateXPBD(
    model, guideFolder, guideTree, posBase, sys, waveSpec, nK, subIters,
    targetBase = posBase) {
  const {nV, barPairs, barLens, hingeVerts, hingeEdgeIdx, numHinges} = sys;
  const nBars = barLens.length;
  const {flat, edges, baseFoldAngles} = model;
  const numEdges = edges.length;
  const hasRelaxedBase = targetBase !== posBase;

  const eStart = new Float64Array(numEdges);
  const eEnd = new Float64Array(numEdges);
  const linearAngles = Boolean(waveSpec.linearAngles);
  const useScheduleGuide = Boolean(waveSpec.useScheduleGuide);

  if (!linearAngles && !useScheduleGuide) {
    for (let e = 0; e < numEdges; e++) {
      const [u, v] = edges[e];
      const rad = Math.min(
          1,
          Math.hypot(
              0.5 * (flat[2 * u] + flat[2 * v]) - 0.5,
              0.5 * (flat[2 * u + 1] + flat[2 * v + 1]) - 0.5) /
              0.5);
      const ax = classifyAxis(flat, edges[e]);
      const win = ax === 'horiz' ?
          waveSpec.hWin :
          (ax === 'vert' ? waveSpec.vWin : waveSpec.rWin);
      const shift = waveSpec.rimLead * (1 - rad);
      eStart[e] = Math.max(0, Math.min(0.72, win[0] + shift));
      eEnd[e] = Math.max(eStart[e] + 0.22, Math.min(1.0, win[1] + 0.45 * shift));
    }
  }

  let offX = 0;
  let offY = 0;
  let offZ = 0;
  if (!useScheduleGuide) {
    guideFolder.adoptTree(guideTree);
    const rawGuideEnd = guideFolder.foldByHingesRaw(baseFoldAngles);
    for (let v = 0; v < nV; v++) {
      offX += (targetBase[3 * v] - rawGuideEnd[3 * v]) / nV;
      offY += (targetBase[3 * v + 1] - rawGuideEnd[3 * v + 1]) / nV;
      offZ += (targetBase[3 * v + 2] - rawGuideEnd[3 * v + 2]) / nV;
    }
  }

  const curAngles = new Float64Array(numEdges);
  const keyframes = new Array(nK);
  const p = new Float64Array(3 * nV);
  for (let v = 0; v < nV; v++) {
    p[3 * v] = flat[2 * v];
    p[3 * v + 1] = flat[2 * v + 1];
    p[3 * v + 2] = 0;
  }
  keyframes[0] = Float32Array.from(p);
  const prevTh = new Float64Array(numHinges);

  for (let k = 1; k < nK - 1; k++) {
    const u = k / (nK - 1);
    let posGuide;
    let anglesRef;
    if (useScheduleGuide) {
      posGuide = guideFolder.positionsAt(u * DEFAULT_BASE_COMPLETE_T, true);
      anglesRef = guideFolder._angles;
    } else {
      for (let e = 0; e < numEdges; e++) {
        if (Math.abs(baseFoldAngles[e]) < 1) {
          curAngles[e] = 0;
        } else if (linearAngles) {
          curAngles[e] = u * baseFoldAngles[e];
        } else {
          const eu = smootherstep(
              (u - eStart[e]) / Math.max(1e-5, eEnd[e] - eStart[e]));
          curAngles[e] = eu * baseFoldAngles[e];
        }
      }
      guideFolder.adoptTree(guideTree);
      posGuide = guideFolder.foldByHingesRaw(curAngles);
      anglesRef = curAngles;
    }

    const uLate = Math.max(
        0,
        (u - waveSpec.tetherStart) / Math.max(1e-4, 1 - waveSpec.tetherStart));
    const xyTether = waveSpec.baseTether +
        (waveSpec.tetherScale ?? 0.78) * Math.pow(uLate, waveSpec.tetherPow);
    const zTether = waveSpec.equalXYZ ?
        xyTether :
        ((0.5 * waveSpec.baseTether) +
         (waveSpec.tetherScale ?? 0.78) *
             Math.pow(uLate, waveSpec.tetherPow + 0.6));
    const zCalm = waveSpec.zCalm * (1 - uLate);

    for (let v = 0; v < nV; v++) {
      const dxR = hasRelaxedBase ? u * (targetBase[3 * v] - posBase[3 * v]) : 0;
      const dyR =
          hasRelaxedBase ? u * (targetBase[3 * v + 1] - posBase[3 * v + 1]) : 0;
      const dzR =
          hasRelaxedBase ? u * (targetBase[3 * v + 2] - posBase[3 * v + 2]) : 0;
      const gx = posGuide[3 * v] + (useScheduleGuide ? 0 : u * offX) + dxR;
      const gy = posGuide[3 * v + 1] + (useScheduleGuide ? 0 : u * offY) + dyR;
      const gz = posGuide[3 * v + 2] + (useScheduleGuide ? 0 : u * offZ) + dzR;
      p[3 * v] = (1 - xyTether) * p[3 * v] + xyTether * gx;
      p[3 * v + 1] = (1 - xyTether) * p[3 * v + 1] + xyTether * gy;
      p[3 * v + 2] = ((1 - zTether) * p[3 * v + 2] + zTether * gz) * (1 - zCalm);
    }

    for (let it = 0; it < subIters; it++) {
      const hAlpha = it < (subIters * 0.65) ? 0.025 : 0.09;
      const hDecay = it < (subIters - 6) ? 1.0 : 0.15;
      for (let h = 0; h < numHinges; h++) {
        const eIdx = hingeEdgeIdx[h];
        const targetDeg = eIdx >= 0 ? anglesRef[eIdx] : 0;
        const isFacet = eIdx < 0 || Math.abs(baseFoldAngles[eIdx]) < 1;
        const targetRad = Math.max(
            -0.985 * Math.PI,
            Math.min(0.985 * Math.PI, targetDeg * Math.PI / 180));

        const v3 = hingeVerts[4 * h];
        const v4 = hingeVerts[4 * h + 1];
        const v1 = hingeVerts[4 * h + 2];
        const v2 = hingeVerts[4 * h + 3];
        const i3 = 3 * v3;
        const i4 = 3 * v4;
        const i1 = 3 * v1;
        const i2 = 3 * v2;
        const ex = p[i4] - p[i3];
        const ey = p[i4 + 1] - p[i3 + 1];
        const ez = p[i4 + 2] - p[i3 + 2];
        const L = Math.hypot(ex, ey, ez);
        if (L <= 1e-8) continue;
        const invL = 1 / L;
        const ux = ex * invL;
        const uy = ey * invL;
        const uz = ez * invL;
        const a1x = p[i1] - p[i3];
        const a1y = p[i1 + 1] - p[i3 + 1];
        const a1z = p[i1 + 2] - p[i3 + 2];
        const a2x = p[i2] - p[i4];
        const a2y = p[i2 + 1] - p[i4 + 1];
        const a2z = p[i2 + 2] - p[i4 + 2];

        let n1x = ey * a1z - ez * a1y;
        let n1y = ez * a1x - ex * a1z;
        let n1z = ex * a1y - ey * a1x;
        let n2x = -ey * a2z + ez * a2y;
        let n2y = -ez * a2x + ex * a2z;
        let n2z = -ex * a2y + ey * a2x;
        const A1 = Math.hypot(n1x, n1y, n1z);
        const A2 = Math.hypot(n2x, n2y, n2z);
        if (A1 <= 1e-9 || A2 <= 1e-9) continue;
        n1x /= A1;
        n1y /= A1;
        n1z /= A1;
        n2x /= A2;
        n2y /= A2;
        n2z /= A2;
        const h1 = A1 * invL;
        const h2 = A2 * invL;

        const cx = n1y * n2z - n1z * n2y;
        const cy = n1z * n2x - n1x * n2z;
        const cz = n1x * n2y - n1y * n2x;
        let th = Math.atan2(
            -(ux * cx + uy * cy + uz * cz),
            n1x * n2x + n1y * n2y + n1z * n2z);
        const pv = prevTh[h];
        if (th - pv > 4.5) th -= 2 * Math.PI;
        else if (th - pv < -4.5) th += 2 * Math.PI;
        prevTh[h] = th;
        const err = th - targetRad;
        if (Math.abs(err) <= 1e-4) continue;

        const s1 = (a1x * ux + a1y * uy + a1z * uz) * invL;
        const s2 =
            ((p[i2] - p[i3]) * ux + (p[i2 + 1] - p[i3 + 1]) * uy +
             (p[i2 + 2] - p[i3 + 2]) * uz) *
            invL;
        const invH1 = 1 / Math.max(h1, 2e-4);
        const invH2 = 1 / Math.max(h2, 2e-4);
        const g1x = n1x * invH1;
        const g1y = n1y * invH1;
        const g1z = n1z * invH1;
        const g2x = n2x * invH2;
        const g2y = n2y * invH2;
        const g2z = n2z * invH2;
        const c31 = -(1 - s1);
        const c32 = -(1 - s2);
        const g3x = c31 * g1x + c32 * g2x;
        const g3y = c31 * g1y + c32 * g2y;
        const g3z = c31 * g1z + c32 * g2z;
        const g4x = -s1 * g1x - s2 * g2x;
        const g4y = -s1 * g1y - s2 * g2y;
        const g4z = -s1 * g1z - s2 * g2z;

        const normSq = invH1 * invH1 + invH2 * invH2 +
            (g3x * g3x + g3y * g3y + g3z * g3z) +
            (g4x * g4x + g4y * g4y + g4z * g4z);
        const gain = hDecay * (isFacet ? 0.08 : 0.44);
        const dLam = -gain * err / (normSq + hAlpha);

        p[i1] += dLam * g1x;
        p[i1 + 1] += dLam * g1y;
        p[i1 + 2] += dLam * g1z;
        p[i2] += dLam * g2x;
        p[i2 + 1] += dLam * g2y;
        p[i2 + 2] += dLam * g2z;
        p[i3] += dLam * g3x;
        p[i3 + 1] += dLam * g3y;
        p[i3 + 2] += dLam * g3z;
        p[i4] += dLam * g4x;
        p[i4 + 1] += dLam * g4y;
        p[i4 + 2] += dLam * g4z;
      }
      for (let sw = 0; sw < 2; sw++) {
        for (let b = 0; b < nBars; b++) {
          const i0 = 3 * barPairs[2 * b];
          const i1 = 3 * barPairs[2 * b + 1];
          const dx = p[i1] - p[i0];
          const dy = p[i1 + 1] - p[i0 + 1];
          const dz = p[i1 + 2] - p[i0 + 2];
          const d = Math.hypot(dx, dy, dz) || 1e-9;
          const corr = 0.49 * (d - barLens[b]) / d;
          p[i0] += corr * dx;
          p[i0 + 1] += corr * dy;
          p[i0 + 2] += corr * dz;
          p[i1] -= corr * dx;
          p[i1 + 1] -= corr * dy;
          p[i1 + 2] -= corr * dz;
        }
      }
    }
    keyframes[k] = Float32Array.from(p);
  }
  keyframes[nK - 1] = targetBase;
  return keyframes;
}

function candidateRootFaces(model) {
  const {flat, faces, faceRegions, regions} = model;
  const center = nearestFaceToCenter(flat, faces);
  const roots = [center];
  const targets = [
    [0.5, 0.30], [0.5, 0.70], [0.30, 0.5], [0.70, 0.5],
    [0.40, 0.40], [0.60, 0.60],
  ];
  for (const [tx, ty] of targets) {
    let bestF = center;
    let bestD = 1e9;
    faces.forEach((fc, f) => {
      let cx = 0;
      let cy = 0;
      for (const v of fc) {
        cx += flat[2 * v];
        cy += flat[2 * v + 1];
      }
      cx /= fc.length;
      cy /= fc.length;
      const isRiver = regions[faceRegions[f]]?.kind === 'river';
      const d = Math.hypot(cx - tx, cy - ty) + (isRiver ? -0.035 : 0);
      if (d < bestD) {
        bestD = d;
        bestF = f;
      }
    });
    if (!roots.includes(bestF)) roots.push(bestF);
  }
  return roots;
}

function serializeScheduleJson(
    model, sched, keyframes, includesBaseFrame = false) {
  const {act1Bounds, act1DeltaQ} =
      quantizeAct1KeyframesDeltaQ(keyframes, includesBaseFrame);
  return {
    numVertices: model.flat.length / 2,
    numEdges: model.edges.length,
    meshHash: computeModelMeshHash(model),
    rootFace: sched.rootFace,
    ...(includesBaseFrame ? {includesBaseFrame: true} : {}),
    baseCompleteT: sched.baseCompleteT ?? DEFAULT_BASE_COMPLETE_T,
    shapingStartT: sched.shapingStartT ?? DEFAULT_SHAPING_START_T,
    standupStartT: +(sched.standupStartT ?? DEFAULT_STANDUP_START_T).toFixed(4),
    act1Bounds,
    act1DeltaQ,
    s2TimeMap: sched.s2TimeMap ?
        Array.from(sched.s2TimeMap, (v) => +v.toFixed(5)) :
        null,
    shapingHopCosts: sched.shapingHopCosts ?
        Array.from(sched.shapingHopCosts, (w) => +w.toFixed(4)) :
        null,
    stats: sched.stats ?? null,
    phases: sched.phases.map((ph) => ({
      id: ph.id,
      act: ph.act,
      stageGroup: ph.stageGroup,
      label: ph.label,
      tStart: +ph.tStart.toFixed(4),
      tEnd: +ph.tEnd.toFixed(4),
      easeOut: +(ph.easeOut ?? 0.35).toFixed(3),
      power: +(ph.power ?? 1.0).toFixed(3),
      edgeIndices: Array.from(ph.edgeIndices),
      deltas: Array.from(ph.deltas, (d) => +d.toFixed(3)),
    })),
  };
}

/**
 * Evaluates both Act-I (`[0, baseCompleteT]`) and Stage-2 (`[shapingStartT, 1]`)
 * 8-Metric Physicality Scorecards on `foldJson`.
 *
 * @param {!Object} foldJson Parsed `shaped_3d.fold` JSON.
 * @param {?Object=} stickFigureJson Optional `stick_figure.json`.
 * @param {?Object=} rawScheduleJson Optional `schedule.json` payload.
 * @param {number=} nSamples Uniform evaluation samples per window.
 * @return {{
 *   act1: !Object,
 *   stage2: !Object,
 *   stats: !Object,
 *   scheduleSource: string,
 * }}
 */
export function scoreModelTrajectory(
    foldJson, stickFigureJson = null, rawScheduleJson = null, nSamples = 19) {
  const model = parseFold(foldJson);
  const hydrated = hydrateSchedule(rawScheduleJson, model);
  const sched =
      hydrated ?? buildFoldingSchedule(model, stickFigureJson, {solveXpbd: true});
  const folder = new Folder(model);
  folder.setSchedule(sched);
  const cctx = buildCollisionContext(model);
  const sctx = buildScorecardContext(model, cctx);
  const act1 = evaluateWindowScorecard(
      model, folder, cctx, sctx, 'act1', nSamples, sched.baseCompleteT,
      sched.shapingStartT);
  const stage2 = evaluateWindowScorecard(
      model, folder, cctx, sctx, 'stage2', nSamples, sched.baseCompleteT,
      sched.shapingStartT);

  let sumFlaps = 0;
  let maxFlaps = 0;
  let flapSteps = 0;
  const prev = new Float32Array(3 * sctx.nV);
  for (let k = 0; k < nSamples; k++) {
    const pos = folder.positionsAt((k / (nSamples - 1)) * sched.baseCompleteT, true);
    if (k > 0) {
      const fl = countMovingFlaps(prev, pos, cctx, 0.005);
      if (fl.activeFlaps > 0) {
        sumFlaps += fl.activeFlaps;
        flapSteps++;
        if (fl.activeFlaps > maxFlaps) maxFlaps = fl.activeFlaps;
      }
    }
    prev.set(pos);
  }
  return {
    act1,
    stage2,
    scheduleSource: hydrated ? 'schedule.json' : 'live-fallback',
    stats: {
      act1Score: act1.score100,
      s2Score: stage2.score100,
      maxSurfAreaDevPct: act1.summary.maxSurfAreaDevPct,
      warpedAreaPct: act1.summary.maxWarpedAreaPct,
      piercedAreaPct: act1.summary.meanTransientPiercedAreaPct,
      avgActiveFlaps: +(flapSteps ? sumFlaps / flapSteps : 0).toFixed(2),
      maxActiveFlaps: maxFlaps,
      totalFlaps: cctx.flapVertices.length,
    },
  };
}

/**
 * Optimizes both Stage-2 shaping (Short-Hop Radial Dijkstra root + hop-cost
 * search + phase-window Simulated Annealing + arc-length velocity equalization)
 * and Act-I collapse (multi-wave Tree-Tethered Signed-Dihedral + Facet-Crease
 * XPBD search + `act1DeltaQ` `Uint16` quantization), verifying exact hydrated
 * `M1..M8` scorecards.
 *
 * @param {!Object} foldJson Parsed `shaped_3d.fold` JSON.
 * @param {?Object=} stickFigureJson Optional `stick_figure.json`.
 * @param {{fast: (boolean|undefined), seed: (number|undefined)}=} options
 * @return {{
 *   scheduleJson: !Object,
 *   act1: !Object,
 *   stage2: !Object,
 *   baselineAct1: !Object,
 *   baselineStage2: !Object,
 *   winningWave: string,
 * }}
 */
export function optimizeModelTrajectory(
    foldJson, stickFigureJson = null, options = {}) {
  const fast = Boolean(options.fast);
  const model = parseFold(foldJson);
  const {flat, faces, edges, baseFoldAngles, foldAngles, faceRegions} = model;
  const nV = flat.length / 2;
  const numEdges = edges.length;
  const rng = createRng(options.seed ?? (nV * 104729 + numEdges * 53));

  const evalFolder = new Folder(model);
  const guideFolder = new Folder(model);
  const cctx = buildCollisionContext(model);
  const sctx = buildScorecardContext(model, cctx);
  const sys = buildSignedXPBDSystem(model);

  const liveInit =
      buildFoldingSchedule(model, stickFigureJson, {solveXpbd: true});
  evalFolder.setSchedule(liveInit);
  const baselineAct1 = evaluateWindowScorecard(
      model, evalFolder, cctx, sctx, 'act1', 19, liveInit.baseCompleteT,
      liveInit.shapingStartT);
  const baselineStage2 = evaluateWindowScorecard(
      model, evalFolder, cctx, sctx, 'stage2', 19, liveInit.baseCompleteT,
      liveInit.shapingStartT);

  const edgeAxis = edges.map((uv) => classifyAxis(flat, uv));
  const edgeByKey =
      new Map(edges.map(([u, v], e) => [u < v ? u * nV + v : v * nV + u, e]));
  const edgeRegCross = new Uint8Array(numEdges);
  const edgeSeenReg = new Int32Array(numEdges).fill(-2);
  faces.forEach((fc, f) => {
    const r = faceRegions[f] ?? -1;
    for (let i = 0; i < fc.length; i++) {
      const a = fc[i];
      const b = fc[(i + 1) % fc.length];
      const e = edgeByKey.get(a < b ? a * nV + b : b * nV + a);
      if (e === undefined) continue;
      if (edgeSeenReg[e] === -2) edgeSeenReg[e] = r;
      else if (edgeSeenReg[e] !== r) edgeRegCross[e] = 1;
    }
  });

  // ---------------------------------------------------------------------------
  // Stage A: Short-Hop Radial Dijkstra Root + Hop-Cost + Phase-Window SA (S2)
  // ---------------------------------------------------------------------------
  const isHuge = cctx.numTris > 3200;
  const nCoarseS2 = fast ? 9 : (isHuge ? 9 : 12);
  evalFolder._skipSeamDiag = true;

  const s1Init = liveInit.phases.filter((p) => p.stageGroup === 'collapse');
  const roots = candidateRootFaces(model).slice(0, fast ? 2 : (isHuge ? 3 : 6));
  const s2TreeConfigs = fast ? [
    [0.55, 1.0, 0.015, 0.30, 0.20, 35.0],
    [0.30, 0.8, 0.028, 0.65, 0.45, 45.0],
  ] : [
    [0.55, 1.0, 0.015, 0.30, 0.20, 35.0],
    [0.30, 0.8, 0.028, 0.65, 0.45, 45.0],
    [0.85, 1.0, 0.008, 0.15, 0.05, 32.0],
    [0.45, 1.0, 0.040, 1.20, 0.80, 55.0],
    [1.20, 1.0, 0.000, 0.00, 0.00, 32.0],
    [0.25, 0.6, 0.050, 0.80, 0.20, 60.0],
  ];

  const s2Seeds = [{loss: scorecardLoss(baselineStage2), sched: liveInit}];
  for (const rf of roots) {
    for (const [cLocked, cFlat, dScale, rPen, regPen, unfPen] of s2TreeConfigs) {
      const hc = new Float64Array(numEdges);
      for (let e = 0; e < numEdges; e++) {
        const b = baseFoldAngles[e];
        const f =
            (Math.abs(b) >= 178 && Math.abs(foldAngles[e]) >= 178) ? b : foldAngles[e];
        const d = Math.abs(f - b);
        const unfolds =
            Math.abs(b) >= 10 && (Math.abs(f) < Math.abs(b) - 10 || b * f < -10);
        const baseCost = Math.abs(b) < 1 ? cFlat : (d < 1.5 ? cLocked : 1.05);
        hc[e] = baseCost + dScale * d + (edgeAxis[e] === 'ridge' ? rPen : 0) +
            (edgeRegCross[e] ? regPen : 0) + (unfolds ? unfPen : 0);
      }
      const cand = {...liveInit, rootFace: rf, shapingHopCosts: hc};
      evalFolder.setSchedule(cand);
      const sc = evaluateWindowScorecard(
          model, evalFolder, cctx, sctx, 'stage2', nCoarseS2, 0.37, 0.41);
      s2Seeds.push({loss: scorecardLoss(sc), sched: cand});
    }
  }
  s2Seeds.sort((a, b) => a.loss - b.loss);

  let bestS2Sched = liveInit;
  let bestS2Loss19 = scorecardLoss(baselineStage2);

  for (const seed of s2Seeds.slice(0, fast ? 1 : 3)) {
    evalFolder.setSchedule(seed.sched);
    let curPh = seed.sched.phases
        .filter((p) => p.stageGroup === 'shaping')
        .map((p) => ({...p}));
    let curLoss = scorecardLoss(evaluateWindowScorecard(
        model, evalFolder, cctx, sctx, 'stage2', nCoarseS2, 0.37, 0.41));

    const s2Steps = fast ? 16 : (faces.length > 2200 ? 45 : 75);
    for (let st = 0; st < s2Steps; st++) {
      if (!curPh.length) break;
      const nextP = curPh.map((p) => ({...p}));
      const idx = (rng() * nextP.length) | 0;
      const p = nextP[idx];
      const minS = p.act === 'IV' ? 0.79 : 0.41;
      const maxE =
          (p.act === 'III' && nextP.some((q) => q.act === 'IV')) ? 0.85 : 0.996;
      const ns =
          Math.max(minS, Math.min(maxE - 0.04, p.tStart + (rng() - 0.5) * 0.12));
      const ne =
          Math.max(ns + 0.035, Math.min(maxE, p.tEnd + (rng() - 0.5) * 0.12));
      const npw =
          Math.max(0.55, Math.min(1.85, (p.power ?? 1.0) + (rng() - 0.5) * 0.35));
      nextP[idx] = {...p, tStart: +ns.toFixed(4), tEnd: +ne.toFixed(4), power: +npw.toFixed(3)};
      evalFolder._schedule = {...seed.sched, phases: [...s1Init, ...nextP]};
      const sc = evaluateWindowScorecard(
          model, evalFolder, cctx, sctx, 'stage2', nCoarseS2, 0.37, 0.41);
      const l = scorecardLoss(sc);
      const temp = 1.6 * Math.pow(0.08, st / Math.max(1, s2Steps));
      if (l < curLoss || rng() < Math.exp((curLoss - l) / Math.max(0.05, temp))) {
        curLoss = l;
        curPh = nextP;
      }
    }

    const candFull = {...seed.sched, phases: [...s1Init, ...curPh]};
    evalFolder.setSchedule(candFull);
    const sc19 = evaluateWindowScorecard(
        model, evalFolder, cctx, sctx, 'stage2', 19, 0.37, 0.41);
    const map2 = buildArcLengthTimeMap(evalFolder, sctx, 0.41, 1.0, 44, 22);
    for (const s2Map of [null, map2]) {
      const withMap = {...candFull, s2TimeMap: s2Map};
      evalFolder.setSchedule(withMap);
      const scM = evaluateWindowScorecard(
          model, evalFolder, cctx, sctx, 'stage2', 19, 0.37, 0.41);
      const lM = scorecardLoss(scM);
      if (lM < bestS2Loss19 - 1e-6) {
        bestS2Loss19 = lM;
        bestS2Sched = withMap;
      }
    }
    if (scorecardLoss(sc19) < bestS2Loss19 - 1e-6) {
      bestS2Loss19 = scorecardLoss(sc19);
      bestS2Sched = candFull;
    }
  }
  evalFolder._skipSeamDiag = false;
  evalFolder.setSchedule(bestS2Sched);
  guideFolder.setSchedule({...bestS2Sched, act1Keyframes: null});

  // ---------------------------------------------------------------------------
  // Stage B: Multi-Wave Signed-Dihedral + Facet-Crease XPBD Act-I Search
  // ---------------------------------------------------------------------------
  evalFolder.adoptTree(evalFolder._shapingTree);
  const posBase = Float32Array.from(evalFolder.foldByHingesRaw(baseFoldAngles));
  const rf = evalFolder._shapingTree.rootFace;
  let maxBaseBarErr = 0;
  for (let b = 0; b < sys.barLens.length; b++) {
    const i0 = 3 * sys.barPairs[2 * b];
    const i1 = 3 * sys.barPairs[2 * b + 1];
    const d = Math.hypot(
        posBase[i1] - posBase[i0],
        posBase[i1 + 1] - posBase[i0 + 1],
        posBase[i1 + 2] - posBase[i0 + 2]);
    const rel = Math.abs(d - sys.barLens[b]) / sys.barLens[b];
    if (rel > maxBaseBarErr) maxBaseBarErr = rel;
  }
  const strainedBase = maxBaseBarErr > 1e-3;
  const targetBase =
      strainedBase ? relaxBasePositions(posBase, sys, 120) : posBase;
  const s2Phases = bestS2Sched.phases.filter((p) => p.stageGroup === 'shaping');
  const s1SchedulePhases =
      bestS2Sched.phases.filter((p) => p.stageGroup === 'collapse');

  const mkDijk = (wH, wV, wR) => buildShortHopDijkstra(
      model, rf,
      Float64Array.from(edges, (_, e) => Math.abs(baseFoldAngles[e]) < 1 ?
          0.22 :
          (edgeAxis[e] === 'horiz' ? wH : (edgeAxis[e] === 'vert' ? wV : wR))));

  const guideTrees = fast ?
      [['Shaping Short-Hop', evalFolder._shapingTree]] :
      [
        ['Shaping Short-Hop', evalFolder._shapingTree],
        ['Short-Hop Balanced', mkDijk(1.05, 1.05, 1.45)],
        ['Short-Hop Horiz-Spine', mkDijk(0.75, 1.35, 1.85)],
        ['Short-Hop Vert-Spine', mkDijk(1.35, 0.75, 1.85)],
      ];

  const allWaves = [
    {
      name: 'Schedule-Guided Gather',
      useScheduleGuide: true,
      hWin: [0.00, 1.00],
      vWin: [0.00, 1.00],
      rWin: [0.00, 1.00],
      rimLead: 0.00,
      baseTether: 0.020,
      tetherScale: 0.68,
      tetherStart: 0.00,
      tetherPow: 2.8,
      zCalm: 0.00,
      equalXYZ: true,
    },
    {
      name: 'Linear-Angle Isometric Gather',
      linearAngles: true,
      hWin: [0.00, 1.00],
      vWin: [0.00, 1.00],
      rWin: [0.00, 1.00],
      rimLead: 0.00,
      baseTether: 0.020,
      tetherScale: 0.78,
      tetherStart: 0.00,
      tetherPow: 3.6,
      zCalm: 0.00,
      equalXYZ: true,
    },
    {
      name: 'Gentle Isometric Gather',
      hWin: [0.00, 1.00],
      vWin: [0.00, 1.00],
      rWin: [0.00, 1.00],
      rimLead: 0.06,
      baseTether: 0.018,
      tetherScale: 0.74,
      tetherStart: 0.12,
      tetherPow: 3.0,
      zCalm: 0.04,
      equalXYZ: true,
    },
    {
      name: 'Rim-to-Core Accordion',
      hWin: [0.00, 0.92],
      vWin: [0.00, 0.92],
      rWin: [0.02, 0.96],
      rimLead: 0.18,
      baseTether: 0.014,
      tetherStart: 0.36,
      tetherPow: 2.3,
      zCalm: 0.14,
    },
    {
      name: 'Vert-Lead Coupled Pleats',
      hWin: [0.07, 0.98],
      vWin: [0.00, 0.85],
      rWin: [0.02, 0.98],
      rimLead: 0.07,
      baseTether: 0.016,
      tetherScale: 0.76,
      tetherStart: 0.18,
      tetherPow: 2.6,
      zCalm: 0.05,
      equalXYZ: true,
    },
    {
      name: 'Horiz-Lead Coupled Pleats',
      hWin: [0.00, 0.85],
      vWin: [0.07, 0.98],
      rWin: [0.02, 0.98],
      rimLead: 0.07,
      baseTether: 0.016,
      tetherScale: 0.76,
      tetherStart: 0.18,
      tetherPow: 2.6,
      zCalm: 0.05,
      equalXYZ: true,
    },
  ];
  const waveSpecs = fast ? allWaves.slice(0, 3) : allWaves;

  const screenNK = fast ? 17 : (isHuge ? 17 : 21);
  const screenIters = fast ? 26 : (isHuge ? 30 : 38);

  const screened = [];
  for (const spec of waveSpecs) {
    const treesToTry = spec.useScheduleGuide ?
        [['Schedule Guide', evalFolder._shapingTree]] :
        guideTrees;
    for (const [tName, gTree] of treesToTry) {
      if (spec.useScheduleGuide) {
        guideFolder.setSchedule({...bestS2Sched, act1Keyframes: null});
      }
      const rawKfs = solveCandidateXPBD(
          model, guideFolder, gTree, posBase, sys, spec, screenNK, screenIters,
          targetBase);
      const eqKfs =
          resampleKeyframesByArcLength(rawKfs, sctx, sys, screenNK);
      const act1Ph =
          spec.useScheduleGuide ? s1SchedulePhases : buildWaveAct1Phases(model, spec);
      for (const kfs of [eqKfs, rawKfs]) {
        const candSched = {
          ...bestS2Sched,
          act1Keyframes: kfs,
          phases: [...act1Ph, ...s2Phases],
        };
        evalFolder.setSchedule(candSched);
        const sc = evaluateWindowScorecard(
            model, evalFolder, cctx, sctx, 'act1', 19, 0.37, 0.41);
        screened.push({loss: scorecardLoss(sc), spec, gTree, tName});
      }
    }
  }
  screened.sort((a, b) => a.loss - b.loss);

  const topDistinct = [];
  const seenSpec = new Set();
  for (const item of screened) {
    if (!seenSpec.has(item.spec.name)) {
      seenSpec.add(item.spec.name);
      topDistinct.push(item);
      if (topDistinct.length >= (fast ? 2 : 3)) break;
    }
  }

  const hiNK = fast ? 23 : 31;
  const hiIters = fast ? 34 : 50;
  let bestOut = null;

  for (const item of topDistinct) {
    if (item.spec.useScheduleGuide) {
      guideFolder.setSchedule({...bestS2Sched, act1Keyframes: null});
    }
    const hiRaw = solveCandidateXPBD(
        model, guideFolder, item.gTree, posBase, sys, item.spec, hiNK, hiIters,
        targetBase);
    const hiEq = resampleKeyframesByArcLength(hiRaw, sctx, sys, hiNK);
    const act1Ph = item.spec.useScheduleGuide ?
        s1SchedulePhases :
        buildWaveAct1Phases(model, item.spec);

    for (const kfs of [hiEq, hiRaw]) {
      const draftSched = {
        ...bestS2Sched,
        phases: [...act1Ph, ...s2Phases],
      };
      const serialized =
          serializeScheduleJson(model, draftSched, kfs, strainedBase);
      const scored = scoreModelTrajectory(foldJson, stickFigureJson, serialized, 19);
      serialized.stats = scored.stats;
      const totalLoss =
          scorecardLoss(scored.act1) + 0.35 * scorecardLoss(scored.stage2);
      if (!bestOut || totalLoss < bestOut.totalLoss) {
        bestOut = {
          totalLoss,
          scheduleJson: serialized,
          act1: scored.act1,
          stage2: scored.stage2,
          winningWave: `${item.spec.name} (${item.tName})`,
        };
      }
    }
  }

  return {
    scheduleJson: bestOut.scheduleJson,
    act1: bestOut.act1,
    stage2: bestOut.stage2,
    baselineAct1,
    baselineStage2,
    winningWave: bestOut.winningWave,
  };
}

function parseCliArgs(argv) {
  const out = {mode: 'score', json: false, fast: false};
  for (const arg of argv) {
    if (arg === '--json') out.json = true;
    else if (arg === '--fast') out.fast = true;
    else if (arg.startsWith('--')) {
      const eq = arg.indexOf('=');
      if (eq > 2) {
        out[arg.slice(2, eq)] = arg.slice(eq + 1);
      } else {
        out[arg.slice(2)] = true;
      }
    }
  }
  return out;
}

function formatScorecardBlock(label, sc) {
  const lines = [
    `${label}: ${sc.score100}/100 (${sc.passCount} PASS · ${sc.warnCount} WARN · ${sc.failCount} FAIL)`,
  ];
  for (const r of sc.rows) {
    lines.push(
        `  [${r.status.padEnd(4)}] ${r.id} ${r.name.padEnd(28)} | ` +
        `${r.value.padEnd(32)} | gate ${r.gate.padEnd(20)} | worst t=${r.worstT}`);
  }
  return lines.join('\n');
}

function runCli() {
  const args = parseCliArgs(process.argv.slice(2));
  if (args.mode === 'benchmark') {
    const modelsDir = path.resolve(args['models-dir'] || 'docs/models');
    const catalog = JSON.parse(
        fs.readFileSync(path.join(modelsDir, 'catalog.json'), 'utf8'));
    const rows = [];
    for (const entry of catalog) {
      const dir = path.join(modelsDir, entry.slug);
      const foldJson = JSON.parse(
          fs.readFileSync(path.join(dir, 'shaped_3d.fold'), 'utf8'));
      const sfPath = path.join(dir, 'stick_figure.json');
      const sfJson = fs.existsSync(sfPath) ?
          JSON.parse(fs.readFileSync(sfPath, 'utf8')) :
          null;
      const schedPath = path.join(dir, 'schedule.json');
      const rawSched = fs.existsSync(schedPath) ?
          JSON.parse(fs.readFileSync(schedPath, 'utf8')) :
          null;
      const res = scoreModelTrajectory(foldJson, sfJson, rawSched, 19);
      rows.push({slug: entry.slug, ...res});
    }
    if (args.json) {
      console.log(JSON.stringify(rows, null, 2));
      return;
    }
    let sumA1 = 0;
    let sumS2 = 0;
    console.log(
        'SLUG                  | SOURCE        | ACT I           | STAGE 2         | dAreaMax | lwP95Str | pierce% | RgReb');
    console.log(
        '----------------------+---------------+-----------------+-----------------+----------+----------+---------+------');
    for (const r of rows) {
      sumA1 += r.act1.score100 / rows.length;
      sumS2 += r.stage2.score100 / rows.length;
      const s = r.act1.summary;
      console.log(
          `${r.slug.padEnd(21)} | ${r.scheduleSource.padEnd(13)} | ` +
          `${String(r.act1.score100).padStart(3)}/100 (${r.act1.passCount}P/${r.act1.failCount}F) | ` +
          `${String(r.stage2.score100).padStart(3)}/100 (${r.stage2.passCount}P/${r.stage2.failCount}F) | ` +
          `${(s.maxSurfAreaDevPct.toFixed(1) + '%').padStart(8)} | ` +
          `${(s.maxLwP95EdgeStrainPct.toFixed(1) + '%').padStart(8)} | ` +
          `${(s.meanTransientPiercedAreaPct.toFixed(1) + '%').padStart(7)} | ` +
          `${s.rgOvershootPct.toFixed(2)}%`);
    }
    console.log(
        `\nMean across ${rows.length} models: Act I = ${sumA1.toFixed(1)}/100, Stage 2 = ${sumS2.toFixed(1)}/100`);
    return;
  }

  if (!args.fold) {
    console.error(
        'Usage: node docs/js/optimize_trajectory.mjs --mode=score|optimize|benchmark --fold=PATH [--stick-figure=PATH] [--schedule=PATH] [--out=PATH] [--fast] [--json]');
    process.exit(2);
  }
  const foldJson = JSON.parse(fs.readFileSync(path.resolve(args.fold), 'utf8'));
  const sfJson = args['stick-figure'] ?
      JSON.parse(fs.readFileSync(path.resolve(args['stick-figure']), 'utf8')) :
      null;

  if (args.mode === 'score') {
    const rawSched = args.schedule ?
        JSON.parse(fs.readFileSync(path.resolve(args.schedule), 'utf8')) :
        null;
    const res = scoreModelTrajectory(foldJson, sfJson, rawSched, 19);
    if (args.json) {
      console.log(JSON.stringify(res, null, 2));
    } else {
      console.log(`Source: ${res.scheduleSource}`);
      console.log(formatScorecardBlock('Act I (0 -> baseCompleteT)', res.act1));
      console.log(formatScorecardBlock('Stage 2 (shapingStartT -> 1)', res.stage2));
    }
    return;
  }

  if (args.mode === 'optimize') {
    const res = optimizeModelTrajectory(foldJson, sfJson, {fast: args.fast});
    if (args.out) {
      fs.writeFileSync(
          path.resolve(args.out), JSON.stringify(res.scheduleJson) + '\n');
    }
    if (args.json) {
      console.log(JSON.stringify({
        winningWave: res.winningWave,
        act1: res.act1,
        stage2: res.stage2,
        out: args.out || null,
      }, null, 2));
    } else {
      console.log(`Winning wave: ${res.winningWave}`);
      console.log(
          `Act I: ${res.baselineAct1.score100}/100 -> ${res.act1.score100}/100 | ` +
          `Stage 2: ${res.baselineStage2.score100}/100 -> ${res.stage2.score100}/100`);
      console.log(formatScorecardBlock('Optimized Act I', res.act1));
      console.log(formatScorecardBlock('Optimized Stage 2', res.stage2));
      if (args.out) console.log(`Wrote ${path.resolve(args.out)}`);
    }
    return;
  }

  console.error(`Unknown --mode=${args.mode} (expected score, optimize, or benchmark).`);
  process.exit(2);
}

const isMainEntry = process.argv[1] &&
    path.resolve(process.argv[1]) === fileURLToPath(import.meta.url);
if (isMainEntry) {
  runCli();
}
