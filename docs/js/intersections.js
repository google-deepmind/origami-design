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
 * @fileoverview Sweep-and-Prune layer-aware 3D self-intersection and
 * flap-locality metrics (zero Map/Set allocations in the inner loop).
 *
 * Distinguishes valid folded paper layers (coplanar or parallel stacked plies
 * staying within `epsPly` on one side of each other's planes) from physically
 * impossible paper-through-paper piercings.
 *
 * Free of DOM and three.js dependencies, so it runs identically in browsers
 * and under Node.
 */

import {earClipFace} from './xpbd.js';

/**
 * Precomputes flat triangle topology and scratch typed arrays for zero-GC
 * Sweep-and-Prune intersection and flap-locality queries on `model`.
 *
 * @param {!Object} model Parsed `Model` from `fold.js`.
 * @return {!Object}
 */
export function buildCollisionContext(model) {
  const {flat, faces, regions = [], faceRegions = []} = model;
  const rawTris = [];
  faces.forEach((fc, fIdx) => {
    for (const [v0, v1, v2] of earClipFace(fc, flat)) {
      rawTris.push(v0, v1, v2, fIdx);
    }
  });
  const numTris = rawTris.length / 4;
  const tris = Int32Array.from(rawTris);

  const flapVertices = [];
  let maxFlapVerts = 0;
  regions.forEach((reg, rIdx) => {
    if (reg.kind !== 'flap') return;
    const vs = new Set();
    faces.forEach((fc, fIdx) => {
      if (faceRegions[fIdx] === rIdx) {
        for (const v of fc) vs.add(v);
      }
    });
    if (vs.size > 0) {
      const arr = Int32Array.from([...vs].sort((a, b) => a - b));
      flapVertices.push(arr);
      if (arr.length > maxFlapVerts) maxFlapVerts = arr.length;
    }
  });

  return {
    numTris,
    numFaces: faces.length,
    tris,
    flapVertices,
    flapDistScratch: new Float64Array(Math.max(maxFlapVerts, 1)),
    normals: new Float64Array(3 * numTris),
    bboxes: new Float64Array(6 * numTris),
    order: new Int32Array(numTris),
    seenFaceKeys: new Int32Array(Math.max(numTris * 4, 64)),
  };
}

/**
 * Möller–Trumbore strict interior segment-triangle piercing test.
 *
 * @param {number} px
 * @param {number} py
 * @param {number} pz
 * @param {number} qx
 * @param {number} qy
 * @param {number} qz
 * @param {number} ax
 * @param {number} ay
 * @param {number} az
 * @param {number} bx
 * @param {number} by
 * @param {number} bz
 * @param {number} cx
 * @param {number} cy
 * @param {number} cz
 * @param {number=} margin Interior barycentric/ray margin.
 * @return {boolean}
 */
function segPiercesTri(
    px, py, pz, qx, qy, qz, ax, ay, az, bx, by, bz, cx, cy, cz,
    margin = 0.015) {
  const dx = qx - px;
  const dy = qy - py;
  const dz = qz - pz;
  const abx = bx - ax;
  const aby = by - ay;
  const abz = bz - az;
  const acx = cx - ax;
  const acy = cy - ay;
  const acz = cz - az;
  const hx = dy * acz - dz * acy;
  const hy = dz * acx - dx * acz;
  const hz = dx * acy - dy * acx;
  const det = abx * hx + aby * hy + abz * hz;
  if (Math.abs(det) <= 1e-12) return false;
  const inv = 1 / det;
  const sx = px - ax;
  const sy = py - ay;
  const sz = pz - az;
  const u = inv * (sx * hx + sy * hy + sz * hz);
  if (u <= margin || u >= 1 - margin) return false;
  const rx = sy * abz - sz * aby;
  const ry = sz * abx - sx * abz;
  const rz = sx * aby - sy * abx;
  const v = inv * (dx * rx + dy * ry + dz * rz);
  if (v <= margin || u + v >= 1 - margin) return false;
  const t = inv * (acx * rx + acy * ry + acz * rz);
  return t > margin && t < 1 - margin;
}

/**
 * Counts physically impossible 3D paper self-intersections in `positions` via
 * adaptive-axis Sweep-and-Prune, ignoring valid stacked paper layers.
 *
 * @param {!Float32Array|!Float64Array} positions
 * @param {!Object} ctx Built by `buildCollisionContext(model)`.
 * @param {number=} epsPly One-sided stacked-ply plane tolerance.
 * @param {boolean=} collectFaces Whether to return `intersectingFaces` set.
 * @return {{
 *   triHits: number,
 *   facePairHits: number,
 *   severity: number,
 *   intersectingFaces: ?Set<number>
 * }}
 */
export function countPhysicalIntersections(
    positions, ctx, epsPly = 1.5e-3, collectFaces = false) {
  const {numTris, numFaces, tris, normals, bboxes, order} = ctx;
  if (numTris < 2) {
    return {
      triHits: 0,
      facePairHits: 0,
      severity: 0,
      intersectingFaces: collectFaces ? new Set() : null,
    };
  }

  let gx0 = 1e9;
  let gy0 = 1e9;
  let gz0 = 1e9;
  let gx1 = -1e9;
  let gy1 = -1e9;
  let gz1 = -1e9;
  let validCount = 0;

  for (let t = 0; t < numTris; t++) {
    const v0 = 3 * tris[4 * t];
    const v1 = 3 * tris[4 * t + 1];
    const v2 = 3 * tris[4 * t + 2];
    const ax = positions[v0];
    const ay = positions[v0 + 1];
    const az = positions[v0 + 2];
    const bx = positions[v1];
    const by = positions[v1 + 1];
    const bz = positions[v1 + 2];
    const cx = positions[v2];
    const cy = positions[v2 + 1];
    const cz = positions[v2 + 2];

    const e1x = bx - ax;
    const e1y = by - ay;
    const e1z = bz - az;
    const e2x = cx - ax;
    const e2y = cy - ay;
    const e2z = cz - az;
    const nx = e1y * e2z - e1z * e2y;
    const ny = e1z * e2x - e1x * e2z;
    const nz = e1x * e2y - e1y * e2x;
    const ln = Math.hypot(nx, ny, nz);
    if (ln <= 1e-10) continue;

    const inv = 1 / ln;
    normals[3 * t] = nx * inv;
    normals[3 * t + 1] = ny * inv;
    normals[3 * t + 2] = nz * inv;

    const mnx = ax < bx ? (ax < cx ? ax : cx) : (bx < cx ? bx : cx);
    const mny = ay < by ? (ay < cy ? ay : cy) : (by < cy ? by : cy);
    const mnz = az < bz ? (az < cz ? az : cz) : (bz < cz ? bz : cz);
    const mxx = ax > bx ? (ax > cx ? ax : cx) : (bx > cx ? bx : cx);
    const mxy = ay > by ? (ay > cy ? ay : cy) : (by > cy ? by : cy);
    const mxz = az > bz ? (az > cz ? az : cz) : (bz > cz ? bz : cz);
    const b6 = 6 * t;
    bboxes[b6] = mnx;
    bboxes[b6 + 1] = mny;
    bboxes[b6 + 2] = mnz;
    bboxes[b6 + 3] = mxx;
    bboxes[b6 + 4] = mxy;
    bboxes[b6 + 5] = mxz;
    if (mnx < gx0) gx0 = mnx;
    if (mny < gy0) gy0 = mny;
    if (mnz < gz0) gz0 = mnz;
    if (mxx > gx1) gx1 = mxx;
    if (mxy > gy1) gy1 = mxy;
    if (mxz > gz1) gz1 = mxz;
    order[validCount++] = t;
  }

  // Sweep along the widest world axis so narrow collapsed strips prune in O(N).
  const spanX = gx1 - gx0;
  const spanY = gy1 - gy0;
  const spanZ = gz1 - gz0;
  const sweepAxis =
      spanX >= spanY && spanX >= spanZ ? 0 : (spanY >= spanZ ? 1 : 2);
  const aOther1 = sweepAxis === 0 ? 1 : 0;
  const aOther2 = sweepAxis === 2 ? 1 : 2;

  const activeSlice = order.subarray(0, validCount);
  activeSlice.sort(
      (a, b) => bboxes[6 * a + sweepAxis] - bboxes[6 * b + sweepAxis]);

  let triHits = 0;
  let severity = 0;
  let faceKeyCount = 0;
  let keysBuf = ctx.seenFaceKeys;
  const intersectingFaces = collectFaces ? new Set() : null;

  for (let i = 0; i < validCount; i++) {
    const ia = activeSlice[i];
    const ab6 = 6 * ia;
    const maxSweepA = bboxes[ab6 + 3 + sweepAxis] - epsPly;
    const aMin1 = bboxes[ab6 + aOther1];
    const aMax1 = bboxes[ab6 + 3 + aOther1] - epsPly;
    const aMin2 = bboxes[ab6 + aOther2];
    const aMax2 = bboxes[ab6 + 3 + aOther2] - epsPly;

    const ta4 = 4 * ia;
    const av0 = tris[ta4];
    const av1 = tris[ta4 + 1];
    const av2 = tris[ta4 + 2];
    const af = tris[ta4 + 3];

    for (let j = i + 1; j < validCount; j++) {
      const ib = activeSlice[j];
      const bb6 = 6 * ib;
      if (bboxes[bb6 + sweepAxis] > maxSweepA) break;
      if (bboxes[bb6 + aOther1] > aMax1 ||
          bboxes[bb6 + 3 + aOther1] < aMin1 + epsPly ||
          bboxes[bb6 + aOther2] > aMax2 ||
          bboxes[bb6 + 3 + aOther2] < aMin2 + epsPly) {
        continue;
      }

      const tb4 = 4 * ib;
      const bf = tris[tb4 + 3];
      if (af === bf) continue;

      const bv0 = tris[tb4];
      const bv1 = tris[tb4 + 1];
      const bv2 = tris[tb4 + 2];
      let sh = 0;
      if (av0 === bv0 || av0 === bv1 || av0 === bv2) sh++;
      if (av1 === bv0 || av1 === bv1 || av1 === bv2) sh++;
      if (av2 === bv0 || av2 === bv1 || av2 === bv2) sh++;
      if (sh >= 2) continue;

      const a0x = positions[3 * av0];
      const a0y = positions[3 * av0 + 1];
      const a0z = positions[3 * av0 + 2];
      const a1x = positions[3 * av1];
      const a1y = positions[3 * av1 + 1];
      const a1z = positions[3 * av1 + 2];
      const a2x = positions[3 * av2];
      const a2y = positions[3 * av2 + 1];
      const a2z = positions[3 * av2 + 2];

      const b0x = positions[3 * bv0];
      const b0y = positions[3 * bv0 + 1];
      const b0z = positions[3 * bv0 + 2];
      const b1x = positions[3 * bv1];
      const b1y = positions[3 * bv1 + 1];
      const b1z = positions[3 * bv1 + 2];
      const b2x = positions[3 * bv2];
      const b2y = positions[3 * bv2 + 1];
      const b2z = positions[3 * bv2 + 2];

      const nbx = normals[3 * ib];
      const nby = normals[3 * ib + 1];
      const nbz = normals[3 * ib + 2];
      const da0 = (a0x - b0x) * nbx + (a0y - b0y) * nby + (a0z - b0z) * nbz;
      const da1 = (a1x - b0x) * nbx + (a1y - b0y) * nby + (a1z - b0z) * nbz;
      const da2 = (a2x - b0x) * nbx + (a2y - b0y) * nby + (a2z - b0z) * nbz;
      const mnDa = da0 < da1 ? (da0 < da2 ? da0 : da2) : (da1 < da2 ? da1 : da2);
      const mxDa = da0 > da1 ? (da0 > da2 ? da0 : da2) : (da1 > da2 ? da1 : da2);
      if (mnDa >= -epsPly || mxDa <= epsPly) continue;

      const nax = normals[3 * ia];
      const nay = normals[3 * ia + 1];
      const naz = normals[3 * ia + 2];
      const db0 = (b0x - a0x) * nax + (b0y - a0y) * nay + (b0z - a0z) * naz;
      const db1 = (b1x - a0x) * nax + (b1y - a0y) * nay + (b1z - a0z) * naz;
      const db2 = (b2x - a0x) * nax + (b2y - a0y) * nay + (b2z - a0z) * naz;
      const mnDb = db0 < db1 ? (db0 < db2 ? db0 : db2) : (db1 < db2 ? db1 : db2);
      const mxDb = db0 > db1 ? (db0 > db2 ? db0 : db2) : (db1 > db2 ? db1 : db2);
      if (mnDb >= -epsPly || mxDb <= epsPly) continue;

      let pierced = false;
      if (sh === 0) {
        pierced =
            segPiercesTri(
                a0x, a0y, a0z, a1x, a1y, a1z, b0x, b0y, b0z, b1x, b1y, b1z,
                b2x, b2y, b2z) ||
            segPiercesTri(
                a1x, a1y, a1z, a2x, a2y, a2z, b0x, b0y, b0z, b1x, b1y, b1z,
                b2x, b2y, b2z) ||
            segPiercesTri(
                a2x, a2y, a2z, a0x, a0y, a0z, b0x, b0y, b0z, b1x, b1y, b1z,
                b2x, b2y, b2z) ||
            segPiercesTri(
                b0x, b0y, b0z, b1x, b1y, b1z, a0x, a0y, a0z, a1x, a1y, a1z,
                a2x, a2y, a2z) ||
            segPiercesTri(
                b1x, b1y, b1z, b2x, b2y, b2z, a0x, a0y, a0z, a1x, a1y, a1z,
                a2x, a2y, a2z) ||
            segPiercesTri(
                b2x, b2y, b2z, b0x, b0y, b0z, a0x, a0y, a0z, a1x, a1y, a1z,
                a2x, a2y, a2z);
      } else {
        const s0 = av0 === bv0 || av0 === bv1 || av0 === bv2;
        const s1 = av1 === bv0 || av1 === bv1 || av1 === bv2;
        if (!s0 && !s1) {
          pierced = segPiercesTri(
              a0x, a0y, a0z, a1x, a1y, a1z, b0x, b0y, b0z, b1x, b1y, b1z, b2x,
              b2y, b2z);
        } else if (!s1) {
          pierced = segPiercesTri(
              a1x, a1y, a1z, a2x, a2y, a2z, b0x, b0y, b0z, b1x, b1y, b1z, b2x,
              b2y, b2z);
        } else {
          pierced = segPiercesTri(
              a2x, a2y, a2z, a0x, a0y, a0z, b0x, b0y, b0z, b1x, b1y, b1z, b2x,
              b2y, b2z);
        }
        if (!pierced) {
          const bs0 = bv0 === av0 || bv0 === av1 || bv0 === av2;
          const bs1 = bv1 === av0 || bv1 === av1 || bv1 === av2;
          if (!bs0 && !bs1) {
            pierced = segPiercesTri(
                b0x, b0y, b0z, b1x, b1y, b1z, a0x, a0y, a0z, a1x, a1y, a1z,
                a2x, a2y, a2z);
          } else if (!bs1) {
            pierced = segPiercesTri(
                b1x, b1y, b1z, b2x, b2y, b2z, a0x, a0y, a0z, a1x, a1y, a1z,
                a2x, a2y, a2z);
          } else {
            pierced = segPiercesTri(
                b2x, b2y, b2z, b0x, b0y, b0z, a0x, a0y, a0z, a1x, a1y, a1z,
                a2x, a2y, a2z);
          }
        }
      }

      if (pierced) {
        triHits++;
        const fLo = af < bf ? af : bf;
        const fHi = af < bf ? bf : af;
        if (faceKeyCount >= keysBuf.length) {
          const grown = new Int32Array(keysBuf.length * 2);
          grown.set(keysBuf);
          keysBuf = grown;
          ctx.seenFaceKeys = grown;
        }
        keysBuf[faceKeyCount++] = fLo * numFaces + fHi;
        if (intersectingFaces) {
          intersectingFaces.add(af);
          intersectingFaces.add(bf);
        }
        severity += Math.min(-mnDa, mxDa, -mnDb, mxDb);
      }
    }
  }

  let facePairHits = 0;
  if (faceKeyCount > 0) {
    const sub = keysBuf.subarray(0, faceKeyCount);
    sub.sort();
    facePairHits = 1;
    for (let k = 1; k < faceKeyCount; k++) {
      if (sub[k] !== sub[k - 1]) facePairHits++;
    }
  }

  return {
    triHits,
    facePairHits,
    severity,
    intersectingFaces,
  };
}

/**
 * Zero-allocation active-flap count between two frames.
 *
 * @param {!Float32Array|!Float64Array} posA
 * @param {!Float32Array|!Float64Array} posB
 * @param {!Object} ctx Built by `buildCollisionContext(model)`.
 * @param {number=} threshold 80th-percentile vertex displacement threshold.
 * @return {{activeFlaps: number, totalFlaps: number}}
 */
export function countMovingFlaps(posA, posB, ctx, threshold = 0.004) {
  const {flapVertices, flapDistScratch} = ctx;
  let active = 0;
  for (const fv of flapVertices) {
    const n = fv.length;
    const sub = flapDistScratch.subarray(0, n);
    for (let i = 0; i < n; i++) {
      const v3 = 3 * fv[i];
      sub[i] = Math.hypot(
          posB[v3] - posA[v3],
          posB[v3 + 1] - posA[v3 + 1],
          posB[v3 + 2] - posA[v3 + 2]);
    }
    sub.sort();
    if (sub[Math.min(n - 1, (0.8 * n) | 0)] > threshold) active++;
  }
  return {activeFlaps: active, totalFlaps: flapVertices.length};
}
