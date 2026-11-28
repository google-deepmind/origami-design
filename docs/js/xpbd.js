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
 * @fileoverview Triangulated Signed-Dihedral + Facet-Crease XPBD solver for
 * unbroken single-sheet Act-I box-pleat collapse.
 *
 * Triangulates polygonal faces with compliant zero-target facet diagonals so
 * closed degree-4/8 box-pleat cycles can flex elastically in mid-fold without
 * cutting seams, while enforcing unwrapped `atan2` signed Mountain/Valley
 * dihedral angles and axial bar inextensibility (`SSS` triangle congruence).
 *
 * Free of DOM and three.js dependencies, so it also runs under Node.
 */

/**
 * Ear-clips a simple 2D polygon (`face` vertex indices into `flat`) into
 * counter-clockwise triangles.
 *
 * @param {!Array<number>} face Vertex indices around one face.
 * @param {!Float64Array} flat Flat `(x, y)` coordinates per vertex.
 * @return {!Array<!Array<number>>} Triples `[a, b, c]` of vertex indices.
 */
export function earClipFace(face, flat) {
  const n = face.length;
  if (n < 3) return [];
  if (n === 3) return [[face[0], face[1], face[2]]];

  let area2 = 0;
  for (let i = 0; i < n; i++) {
    const u = face[i];
    const v = face[(i + 1) % n];
    area2 += flat[2 * u] * flat[2 * v + 1] - flat[2 * v] * flat[2 * u + 1];
  }
  const ring = area2 >= 0 ? [...face] : [...face].reverse();
  const tris = [];
  let guard = 0;
  while (ring.length > 3 && guard < 4 * n * n) {
    guard++;
    let clipped = false;
    const m = ring.length;
    for (let i = 0; i < m; i++) {
      const a = ring[(i + m - 1) % m];
      const b = ring[i];
      const c = ring[(i + 1) % m];
      const ax = flat[2 * a];
      const ay = flat[2 * a + 1];
      const bx = flat[2 * b];
      const by = flat[2 * b + 1];
      const cx = flat[2 * c];
      const cy = flat[2 * c + 1];
      const cross = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
      if (cross <= 1e-12) continue;

      let containsVertex = false;
      for (let j = 0; j < m; j++) {
        const p = ring[j];
        if (p === a || p === b || p === c) continue;
        const px = flat[2 * p];
        const py = flat[2 * p + 1];
        const c0 = (bx - ax) * (py - ay) - (by - ay) * (px - ax);
        const c1 = (cx - bx) * (py - by) - (cy - by) * (px - bx);
        const c2 = (ax - cx) * (py - cy) - (ay - cy) * (px - cx);
        if (c0 >= -1e-12 && c1 >= -1e-12 && c2 >= -1e-12) {
          containsVertex = true;
          break;
        }
      }
      if (!containsVertex) {
        tris.push([a, b, c]);
        ring.splice(i, 1);
        clipped = true;
        break;
      }
    }
    if (!clipped) {
      for (let k = 1; k < ring.length - 1; k++) {
        tris.push([ring[0], ring[k], ring[k + 1]]);
      }
      return tris;
    }
  }
  if (ring.length === 3) {
    tris.push([ring[0], ring[1], ring[2]]);
  }
  return tris;
}

/**
 * Builds axial inextensibility bars and oriented interior dihedral hinges
 * (including compliant `F` facet diagonals across triangulated polygons).
 *
 * @param {!Object} model Parsed `Model` from `fold.js`.
 * @return {{
 *   nV: number,
 *   barPairs: !Int32Array,
 *   barLens: !Float64Array,
 *   hingeVerts: !Int32Array,
 *   hingeEdgeIdx: !Int32Array,
 *   numHinges: number,
 * }}
 */
export function buildSignedXPBDSystem(model) {
  const {flat, faces, edges} = model;
  const nV = flat.length / 2;
  const origEdgeMap = new Map();
  edges.forEach(([u, v], e) => {
    origEdgeMap.set(u < v ? u * nV + v : v * nV + u, e);
  });

  const barMap = new Map();
  const barPairs = [];
  const barLens = [];
  const edgeAdj = new Map();

  for (let f = 0; f < faces.length; f++) {
    const tris = earClipFace(faces[f], flat);
    for (let t = 0; t < tris.length; t++) {
      const [a, b, c] = tris[t];
      for (const [p, q, opp] of [[a, b, c], [b, c, a], [c, a, b]]) {
        const k = p < q ? p * nV + q : q * nV + p;
        if (!barMap.has(k)) {
          const len = Math.hypot(
              flat[2 * p] - flat[2 * q], flat[2 * p + 1] - flat[2 * q + 1]);
          if (len > 1e-8) {
            barMap.set(k, barLens.length);
            barPairs.push(p, q);
            barLens.push(len);
          }
        }
        if (!edgeAdj.has(k)) edgeAdj.set(k, []);
        edgeAdj.get(k).push({a: p, b: q, opp});
      }
    }
  }
  for (let e = 0; e < edges.length; e++) {
    const [p, q] = edges[e];
    const k = p < q ? p * nV + q : q * nV + p;
    if (!barMap.has(k)) {
      const len = Math.hypot(
          flat[2 * p] - flat[2 * q], flat[2 * p + 1] - flat[2 * q + 1]);
      if (len > 1e-8) {
        barMap.set(k, barLens.length);
        barPairs.push(p, q);
        barLens.push(len);
      }
    }
  }

  const hingeVerts = [];
  const hingeEdgeIdx = [];
  for (const [k, list] of edgeAdj.entries()) {
    if (list.length !== 2) continue;
    const t1 = list[0];
    const t2 = list[1];
    if (t1.a === t2.b && t1.b === t2.a) {
      hingeVerts.push(t1.a, t1.b, t1.opp, t2.opp);
      hingeEdgeIdx.push(origEdgeMap.get(k) ?? -1);
    }
  }

  return {
    nV,
    barPairs: Int32Array.from(barPairs),
    barLens: Float64Array.from(barLens),
    hingeVerts: Int32Array.from(hingeVerts),
    hingeEdgeIdx: Int32Array.from(hingeEdgeIdx),
    numHinges: hingeEdgeIdx.length,
  };
}

/**
 * Relaxes axial bar inextensibility strain on a collapsed base pose (for
 * example, level-shifted models whose spanning-tree `posBase` exhibits
 * non-tree cycle averaging strain) while staying aligned with `posBase`.
 *
 * @param {!Float32Array} posBase Spanning-tree collapsed base vertices.
 * @param {!Object} sys System from `buildSignedXPBDSystem(model)`.
 * @param {number=} sweeps Number of axial Gauss-Seidel relaxation sweeps.
 * @return {!Float32Array}
 */
export function relaxBasePositions(posBase, sys, sweeps = 120) {
  const {barPairs, barLens} = sys;
  const nBars = barLens.length;
  const p = Float64Array.from(posBase);
  for (let sw = 0; sw < sweeps; sw++) {
    for (let b = 0; b < nBars; b++) {
      const i0 = 3 * barPairs[2 * b];
      const i1 = 3 * barPairs[2 * b + 1];
      const dx = p[i1] - p[i0];
      const dy = p[i1 + 1] - p[i0 + 1];
      const dz = p[i1 + 2] - p[i0 + 2];
      const d = Math.hypot(dx, dy, dz) || 1e-9;
      const corr = 0.495 * (d - barLens[b]) / d;
      p[i0] += corr * dx;
      p[i0 + 1] += corr * dy;
      p[i0 + 2] += corr * dz;
      p[i1] -= corr * dx;
      p[i1 + 1] -= corr * dy;
      p[i1 + 2] -= corr * dz;
    }
  }
  return Float32Array.from(p);
}

/**
 * Solves an unbroken single-mesh Act-I collapse trajectory from flat (`u = 0`)
 * to `posBase` (`u = 1`) via Tree-Tethered Signed-Dihedral + Bar XPBD.
 *
 * Used automatically when folding local `.fold` files that do not ship a
 * precomputed `schedule.json`.
 *
 * @param {!Object} model Parsed `Model` from `fold.js`.
 * @param {!Float32Array} posBase Exact collapsed base vertices at `t = baseT`.
 * @param {function(!Float64Array): !Float32Array} foldGuideHinges Callback
 *     evaluating the guide spanning tree at intermediate `curAngles`.
 * @param {!Object=} sys Prebuilt system from `buildSignedXPBDSystem(model)`.
 * @return {!Array<!Float32Array>} Keyframe vertex buffers across `[0, baseT]`.
 */
export function solveLiveSignedXPBD(
    model, posBase, foldGuideHinges, sys = null) {
  const xpbd = sys ?? buildSignedXPBDSystem(model);
  const {nV, barPairs, barLens, hingeVerts, hingeEdgeIdx, numHinges} = xpbd;
  const nBars = barLens.length;
  const {flat, baseFoldAngles, edges} = model;

  // Adaptive resolution balances collapse fidelity against solve latency
  // (~0.6-1.4 s on 600-2,000-vertex models; browser callers run this off the
  // main thread via `xpbd_worker.js` or opt in via `{solveXpbd: true}`).
  const nK = nV > 1800 ? 19 : (nV > 1100 ? 23 : 29);
  const subIters = nV > 1800 ? 24 : (nV > 1100 ? 32 : 42);

  // Detect whether rigid spanning-tree `posBase` itself has non-tree cycle
  // averaging strain (e.g. level-shifter or squash-split `.fold` files without
  // reconstructed base subsegments). When `posBase` is strained, relax its
  // axial bars (~120 sweeps) so Act I tethers normally to an unstrained,
  // base-aligned endpoint and `_baseBridgeDelta` remains tiny across
  // `[baseT, shapeT]`.
  let maxBaseBarErr = 0;
  for (let b = 0; b < nBars; b++) {
    const i0 = 3 * barPairs[2 * b];
    const i1 = 3 * barPairs[2 * b + 1];
    const d = Math.hypot(
        posBase[i1] - posBase[i0],
        posBase[i1 + 1] - posBase[i0 + 1],
        posBase[i1 + 2] - posBase[i0 + 2]);
    const rel = Math.abs(d - barLens[b]) / barLens[b];
    if (rel > maxBaseBarErr) maxBaseBarErr = rel;
  }
  const strainedBase = maxBaseBarErr > 1e-3;
  const targetBase =
      strainedBase ? relaxBasePositions(posBase, xpbd, 120) : posBase;

  const keyframes = new Array(nK);
  const p = new Float64Array(3 * nV);
  for (let v = 0; v < nV; v++) {
    p[3 * v] = flat[2 * v];
    p[3 * v + 1] = flat[2 * v + 1];
    p[3 * v + 2] = 0;
  }
  keyframes[0] = Float32Array.from(p);

  const prevTh = new Float64Array(numHinges);
  const curAngles = new Float64Array(edges.length);

  for (let k = 1; k < nK - 1; k++) {
    const u = k / (nK - 1);
    for (let e = 0; e < curAngles.length; e++) {
      curAngles[e] = u * baseFoldAngles[e];
    }
    const posGuide = foldGuideHinges(curAngles);
    const tether = 0.02 + 0.76 * Math.pow(u, 3.4);
    for (let i = 0; i < p.length; i++) {
      const guide = strainedBase ?
          posGuide[i] + u * (targetBase[i] - posBase[i]) :
          posGuide[i];
      p[i] = (1 - tether) * p[i] + tether * guide;
    }

    for (let it = 0; it < subIters; it++) {
      const hAlpha = it < Math.floor(0.65 * subIters) ? 0.025 : 0.09;
      const hDecay = it < subIters - 4 ? 1.0 : 0.18;
      for (let h = 0; h < numHinges; h++) {
        const eIdx = hingeEdgeIdx[h];
        const targetDeg = eIdx >= 0 ? curAngles[eIdx] : 0;
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
        const gain = hDecay * (isFacet ? 0.08 : 0.42);
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
