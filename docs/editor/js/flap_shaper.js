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
 * @fileoverview Pure crease-backed 3D flap, sub-flap, and river shaping on FOLD meshes.
 *
 * Every shaping action (`bend`, `narrow`, `crimp`, `curve`, `pleat`) is
 * strictly represented by real planar Mountain/Valley creases added to both the
 * 2D `creasePattern` frame and the 3D `foldedForm` frame:
 *   - No creaseless vertex warping is ever performed; if a candidate cut plane
 *     does not intersect the target region to create real creases, zero
 *     vertices move.
 *   - Sub-flap cuts slice long faces spanning `[0, 1]` at the exact `[fracStart,
 *     fracEnd]` interval of the target stick along the merged chain.
 *   - River cuts (`bend`, `crimp`, `curve`, `pleat`) slice across the river
 *     strip and rigidly carry the disconnected downstream sub-tree of faces
 *     (`0.00%` extra strain on carried flaps).
 *   - Flat crimps (`crimp`) and flat narrowing tucks (`narrow`) use exact
 *     isometric 3D plane reflections (`R_Pi`), preserving 100% of edge lengths.
 */

/**
 * Mirrors a left/right stick token (`_L` <-> `_R`, `left` <-> `right`).
 *
 * @param {string} name
 * @return {?string}
 */
export function mirrorStickToken(name) {
  if (!name) return null;
  const rules = [
    [/_L$/, '_R'],
    [/_R$/, '_L'],
    [/_l$/, '_r'],
    [/_r$/, '_l'],
    [/_L_/g, '_R_'],
    [/_R_/g, '_L_'],
    [/(^|_)left($|_)/i, '$1right$2'],
    [/(^|_)right($|_)/i, '$1left$2'],
  ];
  for (const [pattern, replacement] of rules) {
    if (pattern.test(name)) {
      const candidate = name.replace(pattern, replacement);
      if (candidate !== name) return candidate;
    }
  }
  return null;
}

/**
 * Finds the bilateral L/R mirror region index in `regions`, if any.
 *
 * @param {!Array<!Object>} regions
 * @param {number} regionIdx
 * @return {?number}
 */
export function findMirroredRegionIndex(regions, regionIdx) {
  if (regionIdx == null || regionIdx < 0 || regionIdx >= regions.length) {
    return null;
  }
  const target = regions[regionIdx];
  if (!target?.sticks?.length || target.kind === 'reserved') return null;
  const mirroredSticks = target.sticks.map(mirrorStickToken);
  if (mirroredSticks.every((m) => m === null)) return null;

  for (let r = 0; r < regions.length; r++) {
    if (r === regionIdx || regions[r].kind !== target.kind) continue;
    const sticks = regions[r].sticks || [];
    if (sticks.length === target.sticks.length &&
        sticks.every((s, i) => s === (mirroredSticks[i] ?? target.sticks[i]))) {
      return r;
    }
  }
  const tipMirror = mirroredSticks.at(-1);
  if (tipMirror) {
    for (let r = 0; r < regions.length; r++) {
      if (r === regionIdx || regions[r].kind !== target.kind) continue;
      if (regions[r].sticks?.at(-1) === tipMirror) return r;
    }
  }
  return null;
}

/**
 * Finds the mirrored sub-flap ID in `subFlaps`, if any.
 *
 * @param {!Array<!Object>} subFlaps
 * @param {?number} subFlapId
 * @return {?number}
 */
export function findMirroredSubFlapId(subFlaps, subFlapId) {
  if (subFlapId == null || subFlapId < 0 || subFlapId >= subFlaps.length) {
    return null;
  }
  const target = subFlaps[subFlapId];
  const mirrorName = mirrorStickToken(target.stickName);
  if (mirrorName) {
    const exact = subFlaps.find(
        (sf) => sf.id !== subFlapId && sf.kind === target.kind &&
            sf.stickName === mirrorName);
    if (exact) return exact.id;
  }
  if (target.mirrorRegionIdx !== null && target.mirrorRegionIdx !== undefined) {
    const byPos = subFlaps.find(
        (sf) => sf.regionIdx === target.mirrorRegionIdx &&
            sf.stickIndex === target.stickIndex);
    if (byPos) return byPos.id;
  }
  return null;
}

/**
 * Formats a human-readable region label.
 *
 * @param {!Array<!Object>} regions
 * @param {number} regionIdx
 * @return {string}
 */
export function formatRegionName(regions, regionIdx) {
  if (regionIdx == null || regionIdx < 0 || regionIdx >= regions.length) {
    return '';
  }
  const region = regions[regionIdx];
  const tip = region?.sticks?.at(-1) || `region_${regionIdx}`;
  return tip.replaceAll('_', ' ');
}

/**
 * Chebyshev L-infinity distance from `(px, py)` to segment `(ax, ay)-(bx, by)`.
 *
 * @param {number} px
 * @param {number} py
 * @param {number} ax
 * @param {number} ay
 * @param {number} bx
 * @param {number} by
 * @return {number}
 */
function pointSegmentLinf(px, py, ax, ay, bx, by) {
  const dx = bx - ax;
  const dy = by - ay;
  const lenSq = dx * dx + dy * dy;
  const t = lenSq < 1e-15 ?
      0 :
      Math.max(0, Math.min(1, ((px - ax) * dx + (py - ay) * dy) / lenSq));
  return Math.max(Math.abs(px - (ax + t * dx)), Math.abs(py - (ay + t * dy)));
}

/**
 * Partitions `faces - rFaces` into connected components and returns the anchor
 * component (containing the sheet-center face) vs. any disconnected downstream
 * components (`carryFaces`) on the far side of `regionIdx`.
 *
 * For a terminal flap, `carryFaces` is empty; for a river (or internal spacer),
 * `carryFaces` holds all downstream flaps/rivers attached past that river.
 *
 * @param {!Array<!Array<number>>} flat
 * @param {!Array<!Array<number>>} faces
 * @param {!Array<?number>} faceRegions
 * @param {number} regionIdx
 * @return {{
 *   rFaces: !Set<number>,
 *   anchorFaces: !Set<number>,
 *   carryFaces: !Set<number>,
 *   carryComps: !Array<!Set<number>>,
 * }}
 */
function partitionSheetAcrossRegion(flat, faces, faceRegions, regionIdx) {
  const isTyped = ArrayBuffer.isView(flat);
  const getX = (v) => (isTyped ? flat[2 * v] : flat[v][0]);
  const getY = (v) => (isTyped ? flat[2 * v + 1] : flat[v][1]);

  const rFaces = new Set();
  for (let f = 0; f < faces.length; f++) {
    if (faceRegions[f] === regionIdx) rFaces.add(f);
  }
  if (!rFaces.size || rFaces.size === faces.length) {
    return {rFaces, anchorFaces: new Set(), carryFaces: new Set(), carryComps: []};
  }

  // Build edge adjacency among faces outside `rFaces`.
  const edgeToRem = new Map();
  let rootCand = -1;
  let rootDist = Infinity;
  for (let f = 0; f < faces.length; f++) {
    if (rFaces.has(f)) continue;
    const poly = faces[f];
    let cx = 0;
    let cy = 0;
    for (const v of poly) {
      cx += getX(v) / poly.length;
      cy += getY(v) / poly.length;
    }
    const dCenter = Math.hypot(cx - 0.5, cy - 0.5);
    if (dCenter < rootDist) {
      rootDist = dCenter;
      rootCand = f;
    }
    for (let i = 0; i < poly.length; i++) {
      const a = poly[i];
      const b = poly[(i + 1) % poly.length];
      const key = a < b ? `${a}_${b}` : `${b}_${a}`;
      if (!edgeToRem.has(key)) edgeToRem.set(key, []);
      edgeToRem.get(key).push(f);
    }
  }

  const adj = new Map();
  for (const inc of edgeToRem.values()) {
    if (inc.length < 2) continue;
    for (const f1 of inc) {
      if (!adj.has(f1)) adj.set(f1, []);
      for (const f2 of inc) {
        if (f1 !== f2) adj.get(f1).push(f2);
      }
    }
  }

  const seen = new Set();
  const comps = [];
  for (let f = 0; f < faces.length; f++) {
    if (rFaces.has(f) || seen.has(f)) continue;
    const comp = new Set();
    const stack = [f];
    seen.add(f);
    while (stack.length > 0) {
      const cur = stack.pop();
      comp.add(cur);
      for (const nb of adj.get(cur) || []) {
        if (!seen.has(nb)) {
          seen.add(nb);
          stack.push(nb);
        }
      }
    }
    comps.push(comp);
  }

  if (comps.length <= 1) {
    return {
      rFaces,
      anchorFaces: comps[0] ?? new Set(),
      carryFaces: new Set(),
      carryComps: [],
    };
  }

  // Pick the largest component (breaking ties by `rootCand`) as stationary anchor.
  comps.sort((a, b) => {
    if (b.size !== a.size) return b.size - a.size;
    return (b.has(rootCand) ? 1 : 0) - (a.has(rootCand) ? 1 : 0);
  });
  const anchorFaces = comps[0];
  const carryComps = comps.slice(1);
  const carryFaces = new Set();
  for (const comp of carryComps) {
    for (const f of comp) carryFaces.add(f);
  }
  return {rFaces, anchorFaces, carryFaces, carryComps};
}

/**
 * Computes intrinsic `[0, 1]` radial progression `tau` from the upstream seam
 * (`anchorFaces`) across `regionIdx`, returning both per-face and per-vertex maps.
 *
 * @param {!Float64Array|!Array<!Array<number>>} flat2d
 * @param {!Array<!Array<number>>} faces
 * @param {!Array<?number>} faceRegions
 * @param {number} regionIdx
 * @param {?Map<string, !Array<number>>=} edgeToFaces
 * @param {?Set<number>=} anchorFacesOpt
 * @return {{
 *   tauByFace: !Map<number, number>,
 *   tauByVert: !Map<number, number>,
 * }}
 */
function computeRegionIntrinsicTauMaps(
    flat2d, faces, faceRegions, regionIdx, edgeToFaces = null,
    anchorFacesOpt = null) {
  const isTyped = ArrayBuffer.isView(flat2d);
  const getX = (v) => (isTyped ? flat2d[2 * v] : flat2d[v][0]);
  const getY = (v) => (isTyped ? flat2d[2 * v + 1] : flat2d[v][1]);

  const anchorFaces = anchorFacesOpt ?? (() => {
    const s = new Set();
    for (let f = 0; f < faces.length; f++) {
      if (faceRegions[f] !== regionIdx) s.add(f);
    }
    return s;
  })();

  const rFaces = [];
  const rVerts = new Set();
  const anchorVerts = new Set();
  for (let f = 0; f < faces.length; f++) {
    if (faceRegions[f] === regionIdx) {
      rFaces.push(f);
      for (const v of faces[f]) rVerts.add(v);
    } else if (anchorFaces.has(f)) {
      for (const v of faces[f]) anchorVerts.add(v);
    }
  }
  const tauByFace = new Map();
  const tauByVert = new Map();
  if (!rFaces.length) return {tauByFace, tauByVert};

  const sharedVerts = new Set();
  for (const v of rVerts) {
    if (anchorVerts.has(v)) sharedVerts.add(v);
  }

  let ef = edgeToFaces;
  if (!ef) {
    ef = new Map();
    for (let f = 0; f < faces.length; f++) {
      const poly = faces[f];
      for (let i = 0; i < poly.length; i++) {
        const a = poly[i];
        const b = poly[(i + 1) % poly.length];
        const key = a < b ? `${a}_${b}` : `${b}_${a}`;
        if (!ef.has(key)) ef.set(key, []);
        ef.get(key).push(f);
      }
    }
  }

  const seamSegs = [];
  for (const f of rFaces) {
    const poly = faces[f];
    for (let i = 0; i < poly.length; i++) {
      const a = poly[i];
      const b = poly[(i + 1) % poly.length];
      const key = a < b ? `${a}_${b}` : `${b}_${a}`;
      const inc = ef.get(key) || [];
      if (inc.some((fi) => anchorFaces.has(fi))) {
        seamSegs.push([getX(a), getY(a), getX(b), getY(b)]);
      }
    }
  }

  const dVert = new Map();
  let dMax = 0;
  for (const v of rVerts) {
    let dist = 0;
    if (!sharedVerts.has(v)) {
      const px = getX(v);
      const py = getY(v);
      if (seamSegs.length > 0) {
        dist = Infinity;
        for (const [ax, ay, bx, by] of seamSegs) {
          const cand = pointSegmentLinf(px, py, ax, ay, bx, by);
          if (cand < dist) dist = cand;
        }
      } else if (sharedVerts.size > 0) {
        dist = Infinity;
        for (const sv of sharedVerts) {
          const cand = Math.max(Math.abs(px - getX(sv)), Math.abs(py - getY(sv)));
          if (cand < dist) dist = cand;
        }
      } else {
        dist = py;
      }
    }
    dVert.set(v, dist);
    if (dist > dMax) dMax = dist;
  }

  const invMax = dMax > 1e-9 ? 1 / dMax : 0;
  for (const [v, d] of dVert.entries()) {
    tauByVert.set(v, d * invMax);
  }
  for (const f of rFaces) {
    const poly = faces[f];
    let sum = 0;
    for (const v of poly) sum += tauByVert.get(v) ?? 0;
    tauByFace.set(f, sum / poly.length);
  }
  return {tauByFace, tauByVert};
}

/**
 * Computes per-face intrinsic `[0, 1]` radial coordinates on `regionIdx`.
 *
 * @param {!Float64Array|!Array<!Array<number>>} flat2d
 * @param {!Array<!Array<number>>} faces
 * @param {!Array<?number>} faceRegions
 * @param {number} regionIdx
 * @param {?Map<string, !Array<number>>=} edgeToFaces
 * @return {!Map<number, number>}
 */
export function computeRegionIntrinsicTau(
    flat2d, faces, faceRegions, regionIdx, edgeToFaces = null) {
  const {anchorFaces} =
      partitionSheetAcrossRegion(flat2d, faces, faceRegions, regionIdx);
  return computeRegionIntrinsicTauMaps(
      flat2d, faces, faceRegions, regionIdx, edgeToFaces, anchorFaces).tauByFace;
}

/**
 * Builds the catalog of unmerged sub-flaps and rivers for 3D/2D picking.
 *
 * @param {!Object} model
 * @param {?Object=} stickFigure
 * @return {{
 *   subFlaps: !Array<!Object>,
 *   faceSubFlap: !Int32Array,
 *   stickLengths: !Object<string, number>,
 * }}
 */
export function buildSubFlapCatalog(model, stickFigure = null) {
  const {flat, faces, faceRegions, regions} = model;
  const stickLengths = {};
  for (const s of stickFigure?.children ?? []) {
    if (s?.name) {
      stickLengths[s.name] = Math.max(0.25, Number(s.length) || 1);
    }
  }

  const edgeToFaces = new Map();
  for (let f = 0; f < faces.length; f++) {
    const poly = faces[f];
    for (let i = 0; i < poly.length; i++) {
      const a = poly[i];
      const b = poly[(i + 1) % poly.length];
      const key = a < b ? `${a}_${b}` : `${b}_${a}`;
      if (!edgeToFaces.has(key)) edgeToFaces.set(key, []);
      edgeToFaces.get(key).push(f);
    }
  }

  const subFlaps = [];
  const faceSubFlap = new Int32Array(faces.length).fill(-1);

  regions.forEach((region, r) => {
    if (region.kind === 'reserved') return;
    const sticks = region.sticks?.length ? region.sticks : [`region_${r}`];
    const mirrorRegionIdx = findMirroredRegionIndex(regions, r);
    const {rFaces, anchorFaces} =
        partitionSheetAcrossRegion(flat, faces, faceRegions, r);
    if (!rFaces.size) return;

    // Exclude enclosed internal seam pockets that have zero free vertices
    // relative to the stationary sheet anchor (`anchorFaces`).
    if (region.kind === 'river') {
      const anchorVerts = new Set();
      for (const af of anchorFaces) {
        for (const v of faces[af]) anchorVerts.add(v);
      }
      let hasFreeVert = false;
      for (const rf of rFaces) {
        for (const v of faces[rf]) {
          if (!anchorVerts.has(v)) {
            hasFreeVert = true;
            break;
          }
        }
        if (hasFreeVert) break;
      }
      if (!hasFreeVert) return;
    }

    if (sticks.length === 1) {
      const id = subFlaps.length;
      const faceIndices = [];
      for (let f = 0; f < faces.length; f++) {
        if (faceRegions[f] === r) {
          faceIndices.push(f);
          faceSubFlap[f] = id;
        }
      }
      subFlaps.push({
        id,
        regionIdx: r,
        mirrorRegionIdx,
        stickIndex: 0,
        stickCount: 1,
        stickName: sticks[0],
        label: sticks[0].replaceAll('_', ' '),
        kind: region.kind,
        fracStart: 0,
        fracEnd: 1,
        chainSticks: [...sticks],
        chainSubFlapIds: [id],
        faceIndices,
      });
      return;
    }

    const {tauByFace} = computeRegionIntrinsicTauMaps(
        flat, faces, faceRegions, r, edgeToFaces, anchorFaces);
    const lengths = sticks.map((s) => stickLengths[s] ?? 1);
    const totalLen = lengths.reduce((acc, v) => acc + v, 0) || sticks.length;
    let cum = 0;
    const chainIds = [];
    const baseId = subFlaps.length;
    for (let j = 0; j < sticks.length; j++) chainIds.push(baseId + j);

    const created = sticks.map((stickName, j) => {
      const fracStart = cum / totalLen;
      cum += lengths[j];
      const fracEnd = j === sticks.length - 1 ? 1 : cum / totalLen;
      const entry = {
        id: baseId + j,
        regionIdx: r,
        mirrorRegionIdx,
        stickIndex: j,
        stickCount: sticks.length,
        stickName,
        label: stickName.replaceAll('_', ' '),
        kind: region.kind,
        fracStart,
        fracEnd,
        chainSticks: [...sticks],
        chainSubFlapIds: chainIds,
        faceIndices: [],
      };
      subFlaps.push(entry);
      return entry;
    });

    for (const [f, tau] of tauByFace.entries()) {
      let chosen = created.length - 1;
      for (let j = 0; j < created.length; j++) {
        if (tau <= created[j].fracEnd + 1e-9) {
          chosen = j;
          break;
        }
      }
      created[chosen].faceIndices.push(f);
      faceSubFlap[f] = created[chosen].id;
    }

    const allRFaces = [...tauByFace.keys()];
    if (allRFaces.length > 0) {
      for (const entry of created) {
        if (entry.faceIndices.length === 0) {
          const mid = 0.5 * (entry.fracStart + entry.fracEnd);
          let nearestF = allRFaces[0];
          let bestDist = Infinity;
          for (const f of allRFaces) {
            const d = Math.abs((tauByFace.get(f) ?? 0) - mid);
            if (d < bestDist) {
              bestDist = d;
              nearestF = f;
            }
          }
          entry.faceIndices.push(nearestF);
        }
      }
    }
  });

  return {subFlaps, faceSubFlap, stickLengths};
}

/**
 * @param {!Array<number>} a
 * @param {!Array<number>} b
 * @return {number}
 */
function dot3(a, b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

/**
 * @param {!Array<number>} a
 * @param {!Array<number>} b
 * @return {!Array<number>}
 */
function cross3(a, b) {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}

/**
 * @param {!Array<number>} v
 * @return {number}
 */
function norm3(v) {
  return Math.hypot(v[0], v[1], v[2]);
}

/**
 * @param {!Array<number>} v
 * @return {!Array<number>}
 */
function normalize3(v) {
  const len = norm3(v);
  return len > 1e-12 ? [v[0] / len, v[1] / len, v[2] / len] : [0, 0, 1];
}

/**
 * @param {!Array<number>} p
 * @param {!Array<number>} pivot
 * @param {!Array<number>} omega
 * @return {!Array<number>}
 */
function rotatePointAround(p, pivot, omega) {
  const angle = norm3(omega);
  if (angle < 1e-9) return [p[0], p[1], p[2]];
  const k = [omega[0] / angle, omega[1] / angle, omega[2] / angle];
  const vx = p[0] - pivot[0];
  const vy = p[1] - pivot[1];
  const vz = p[2] - pivot[2];
  const cos = Math.cos(angle);
  const sin = Math.sin(angle);
  const kv = k[0] * vx + k[1] * vy + k[2] * vz;
  const cx = k[1] * vz - k[2] * vy;
  const cy = k[2] * vx - k[0] * vz;
  const cz = k[0] * vy - k[1] * vx;
  return [
    pivot[0] + vx * cos + cx * sin + k[0] * kv * (1 - cos),
    pivot[1] + vy * cos + cy * sin + k[1] * kv * (1 - cos),
    pivot[2] + vz * cos + cz * sin + k[2] * kv * (1 - cos),
  ];
}

/**
 * @param {!Array<number>} p
 * @param {!Array<number>} planeNormal
 * @param {number} planeOffset
 * @return {!Array<number>}
 */
function reflectPointAcrossPlane(p, planeNormal, planeOffset) {
  const d = dot3(p, planeNormal) - planeOffset;
  return [
    p[0] - 2 * d * planeNormal[0],
    p[1] - 2 * d * planeNormal[1],
    p[2] - 2 * d * planeNormal[2],
  ];
}

/**
 * @param {!Array<number>} v
 * @param {!Array<number>} planeNormal
 * @return {!Array<number>}
 */
function reflectVectorAcrossPlane(v, planeNormal) {
  const d = dot3(v, planeNormal);
  return [
    v[0] - 2 * d * planeNormal[0],
    v[1] - 2 * d * planeNormal[1],
    v[2] - 2 * d * planeNormal[2],
  ];
}

/**
 * @param {!Object} foldDict
 * @return {!Object}
 */
function unpackFoldMesh(foldDict) {
  const childFrames = foldDict.file_frames ?? [];
  const frames = [
    foldDict,
    ...childFrames.map(
        (c) => (c.frame_inherit ? {...foldDict, ...c} : c)),
  ];
  const dim = (f) => f.vertices_coords?.[0]?.length;
  const flatFrame =
      frames.find((f) => f.frame_classes?.includes('creasePattern')) ??
      frames.find((f) => dim(f) === 2);
  const foldedFrame =
      frames.find((f) => f !== flatFrame && dim(f) === 3) ??
      (dim(foldDict) === 3 ? foldDict : null);

  if (!flatFrame?.vertices_coords || !foldedFrame?.vertices_coords) {
    throw new Error('Model requires both 2D flat and 3D folded frames.');
  }

  const flat = flatFrame.vertices_coords.map((p) => [p[0], p[1]]);
  const folded = foldedFrame.vertices_coords.map((p) => [p[0], p[1], p[2]]);
  const faces = flatFrame.faces_vertices.map((f) => {
    let area2 = 0;
    for (let i = 0; i < f.length; i++) {
      const a = flat[f[i]];
      const b = flat[f[(i + 1) % f.length]];
      area2 += a[0] * b[1] - b[0] * a[1];
    }
    return area2 < 0 ? [...f].reverse() : [...f];
  });
  const faceRegions = [...(flatFrame['faces_bp:region'] ?? faces.map(() => null))];
  const regions = (flatFrame['file_bp:regions'] ?? []).map(
      (r) => ({kind: r.kind, sticks: [...(r.sticks ?? [])]}));
  const edges = flatFrame.edges_vertices.map(([u, v]) => [u, v]);
  const assignments = [...(flatFrame.edges_assignment ?? edges.map(() => 'U'))];
  const foldAngles = [...(flatFrame.edges_foldAngle ?? assignments.map(
      (a) => (a === 'M' ? -180 : (a === 'V' ? 180 : 0))))];
  const shaping = flatFrame['edges_bp:shaping'] ?
      [...flatFrame['edges_bp:shaping']] :
      edges.map(() => false);
  const userShaping = flatFrame['edges_bp:userShaping'] ?
      [...flatFrame['edges_bp:userShaping']] :
      edges.map(() => false);
  const baseFoldAngles = flatFrame['edges_bp:baseFoldAngle'] ?
      [...flatFrame['edges_bp:baseFoldAngle']] :
      foldAngles.map((ang, e) => (
        shaping[e] || (assignments[e] !== 'M' && assignments[e] !== 'V') ? 0 : ang
      ));

  return {
    meta: {
      file_spec: foldDict.file_spec ?? 1.2,
      file_creator: foldDict.file_creator ?? 'bp_pipeline',
      file_title: foldDict.file_title ?? 'Shaped Origami Model',
      file_classes: foldDict.file_classes ?? ['singleModel'],
      gridSize: flatFrame['file_bp:gridSize'] ?? foldDict['file_bp:gridSize'] ?? null,
    },
    flat,
    folded,
    faces,
    faceRegions,
    regions,
    edges,
    assignments,
    foldAngles,
    baseFoldAngles,
    shaping,
    userShaping,
    addedUserCreases: 0,
    regionCutCount: 0,
  };
}

/**
 * @param {!Object} state
 * @return {!Object}
 */
function packFoldMesh(state) {
  const out = {
    file_spec: state.meta.file_spec,
    file_creator: state.meta.file_creator,
    file_title: state.meta.file_title,
    file_classes: state.meta.file_classes,
    frame_classes: ['foldedForm'],
    frame_attributes: ['3D'],
    vertices_coords: state.folded,
    edges_vertices: state.edges,
    edges_assignment: state.assignments,
    edges_foldAngle: state.foldAngles,
    'edges_bp:shaping': state.shaping,
    'edges_bp:userShaping': state.userShaping,
    'edges_bp:baseFoldAngle': state.baseFoldAngles,
    faces_vertices: state.faces,
    'faces_bp:region': state.faceRegions,
    'file_bp:regions': state.regions,
    file_frames: [
      {
        frame_classes: ['creasePattern'],
        frame_attributes: ['2D'],
        frame_inherit: true,
        frame_parent: 0,
        vertices_coords: state.flat,
      },
    ],
  };
  if (state.meta.gridSize !== null) {
    out['file_bp:gridSize'] = state.meta.gridSize;
  }
  return out;
}

/**
 * @param {number} u
 * @param {number} v
 * @return {string}
 */
function undirectedKey(u, v) {
  return u < v ? `${u}_${v}` : `${v}_${u}`;
}

/**
 * @param {!Array<number>} poly
 * @param {!Array<!Array<number>>} flat
 * @return {!Array<!Array<number>>}
 */
function earClipPolygon2D(poly, flat) {
  if (poly.length <= 3) return [poly];
  const rem = [...poly];
  const tris = [];
  const cross2 = (ia, ib, ic) => {
    const a = flat[ia];
    const b = flat[ib];
    const c = flat[ic];
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
  };
  const inTri = (ip, ia, ib, ic) => {
    return cross2(ia, ib, ip) >= -1e-11 &&
        cross2(ib, ic, ip) >= -1e-11 &&
        cross2(ic, ia, ip) >= -1e-11;
  };
  let guard = rem.length * rem.length + 8;
  while (rem.length > 3 && guard-- > 0) {
    let clipped = false;
    for (let i = 0; i < rem.length; i++) {
      const prev = rem[(i - 1 + rem.length) % rem.length];
      const curr = rem[i];
      const next = rem[(i + 1) % rem.length];
      if (cross2(prev, curr, next) <= 1e-12) continue;
      let hasInside = false;
      for (let j = 0; j < rem.length; j++) {
        const cand = rem[j];
        if (cand === prev || cand === curr || cand === next) continue;
        if (inTri(cand, prev, curr, next)) {
          hasInside = true;
          break;
        }
      }
      if (!hasInside) {
        tris.push([prev, curr, next]);
        rem.splice(i, 1);
        clipped = true;
        break;
      }
    }
    if (!clipped) {
      for (let i = 1; i + 1 < rem.length; i++) {
        tris.push([rem[0], rem[i], rem[i + 1]]);
      }
      return tris;
    }
  }
  if (rem.length === 3) tris.push(rem);
  return tris;
}

/**
 * @param {!Array<number>} augPoly
 * @param {!Array<number>} d
 * @return {!Array<!Array<number>>}
 */
function splitTwoCrossingPolygon(augPoly, d) {
  const zeroIndices = [];
  for (let i = 0; i < augPoly.length; i++) {
    if (d[augPoly[i]] === 0) zeroIndices.push(i);
  }
  if (zeroIndices.length !== 2) return [augPoly];
  const [i0, i1] = zeroIndices;
  const polyA = [];
  for (let i = i0; i <= i1; i++) polyA.push(augPoly[i]);
  const polyB = [];
  for (let i = i1; i < augPoly.length; i++) polyB.push(augPoly[i]);
  for (let i = 0; i <= i0; i++) polyB.push(augPoly[i]);

  const hasSign = (p, sign) => p.some((v) => sign * d[v] > 0);
  if ((hasSign(polyA, 1) && hasSign(polyB, -1)) ||
      (hasSign(polyA, -1) && hasSign(polyB, 1))) {
    return [polyA, polyB].filter((p) => p.length >= 3);
  }
  return [augPoly];
}

/**
 * Computes the 3D free span, sliceable faces, stationary locked vertices, and
 * any downstream carried vertices (for rivers) for `(regionIdx, fracStart, fracEnd)`.
 *
 * @param {!Object} state
 * @param {number} regionIdx
 * @param {number=} fracStart
 * @param {number=} fracEnd
 * @param {?Array<number>=} preferredU
 * @return {?Object}
 */
function findFlapFreeSpan(
    state, regionIdx, fracStart = 0, fracEnd = 1, preferredU = null) {
  const {flat, folded, faces, faceRegions} = state;
  const {rFaces, anchorFaces, carryFaces, carryComps} =
      partitionSheetAcrossRegion(flat, faces, faceRegions, regionIdx);
  if (!rFaces.size) return null;

  const {tauByVert} = computeRegionIntrinsicTauMaps(
      flat, faces, faceRegions, regionIdx, null, anchorFaces);
  const useSubRange = fracStart > 1e-4 || fracEnd < 1 - 1e-4;

  // Include every face in `regionIdx` that reaches past `fracStart`!
  // (Even a single long triangle spanning `0 -> 1` has `maxTau = 1 > fracStart`
  // so it gets cleanly sliced at `[fracStart, fracEnd]`!)
  const downstreamFaces = new Set();
  const subSegmentFaces = new Set();
  for (const f of rFaces) {
    if (!useSubRange) {
      downstreamFaces.add(f);
      subSegmentFaces.add(f);
      continue;
    }
    let minTau = Infinity;
    let maxTau = -Infinity;
    for (const v of faces[f]) {
      const tv = tauByVert.get(v) ?? 0;
      if (tv < minTau) minTau = tv;
      if (tv > maxTau) maxTau = tv;
    }
    if (maxTau > fracStart + 1e-4) {
      downstreamFaces.add(f);
      if (minTau < fracEnd - 1e-4 || maxTau <= fracEnd + 1e-4) {
        subSegmentFaces.add(f);
      }
    }
  }
  if (!downstreamFaces.size) {
    for (const f of rFaces) downstreamFaces.add(f);
  }
  if (!subSegmentFaces.size) {
    for (const f of downstreamFaces) subSegmentFaces.add(f);
  }

  // Stationary vertices: any vertex belonging to `anchorFaces` OR belonging to
  // an upstream face of `rFaces` omitted from `downstreamFaces`.
  const lockedOutsideVerts = new Set();
  for (let f = 0; f < faces.length; f++) {
    if (anchorFaces.has(f) || (rFaces.has(f) && !downstreamFaces.has(f))) {
      for (const v of faces[f]) lockedOutsideVerts.add(v);
    }
  }

  // Carry vertices: vertices in disconnected downstream `carryFaces` (for rivers)
  // that are not locked to `anchorFaces`. Any vertex shared between
  // `downstreamFaces` and `carryFaces` (`distalSeamVerts`) must move rigidly
  // WITH `carryFaces` (`dist > 0`) so river cuts never stretch the distal seam!
  const carryOnlyVerts = new Set();
  const distalSeamVerts = new Set();
  for (const f of carryFaces) {
    for (const v of faces[f]) {
      if (!lockedOutsideVerts.has(v)) carryOnlyVerts.add(v);
    }
  }

  const activeVerts = new Set();
  for (const f of downstreamFaces) {
    for (const v of faces[f]) {
      activeVerts.add(v);
      if (carryOnlyVerts.has(v)) distalSeamVerts.add(v);
    }
  }

  const sharedVerts = new Set();
  const freeVerts = new Set();
  for (const v of activeVerts) {
    if (lockedOutsideVerts.has(v)) sharedVerts.add(v);
    else freeVerts.add(v);
  }
  if (!freeVerts.size) return null;

  // If `fracStart > 0` and there are vertices strictly above `fracStart` AND
  // intermediate vertices strictly in `(0, fracStart)`, also lock those upstream
  // sub-flap vertices so upstream sticks stay motionless.
  if (useSubRange && fracStart > 1e-4) {
    const aboveStart = [];
    const belowStart = [];
    for (const v of freeVerts) {
      const tv = tauByVert.get(v) ?? 1;
      if (tv >= fracStart - 1e-4) aboveStart.push(v);
      else belowStart.push(v);
    }
    if (aboveStart.length > 0 && belowStart.length > 0) {
      for (const v of belowStart) {
        freeVerts.delete(v);
        sharedVerts.add(v);
        lockedOutsideVerts.add(v);
      }
    }
  }

  const subFreeVerts = new Set();
  for (const f of subSegmentFaces) {
    for (const v of faces[f]) {
      if (freeVerts.has(v)) subFreeVerts.add(v);
    }
  }
  const targetFree = subFreeVerts.size > 0 ? subFreeVerts : freeVerts;

  if (!sharedVerts.size) {
    const u = preferredU ? normalize3(preferredU) : [0, 1, 0];
    let sMin = Infinity;
    let sMax = -Infinity;
    for (const v of freeVerts) {
      const s = dot3(folded[v], u);
      sMin = Math.min(sMin, s);
      sMax = Math.max(sMax, s);
    }
    const sSpanStart = sMin + fracStart * (sMax - sMin);
    const sSpanEnd = sMin + fracEnd * (sMax - sMin);
    return {
      u,
      sSharedMax: sSpanStart,
      sSubMax: sSpanEnd,
      sFreeMax: sMax,
      gap: Math.max(1e-5, sSpanEnd - sSpanStart),
      downstreamFaces,
      subSegmentFaces,
      freeVerts,
      targetFree,
      sharedVerts,
      lockedOutsideVerts,
      carryOnlyVerts,
      distalSeamVerts,
      carryFaces,
      carryComps,
    };
  }

  const sc = [0, 0, 0];
  for (const v of sharedVerts) {
    sc[0] += folded[v][0] / sharedVerts.size;
    sc[1] += folded[v][1] / sharedVerts.size;
    sc[2] += folded[v][2] / sharedVerts.size;
  }
  const fc = [0, 0, 0];
  for (const v of targetFree) {
    fc[0] += folded[v][0] / targetFree.size;
    fc[1] += folded[v][1] / targetFree.size;
    fc[2] += folded[v][2] / targetFree.size;
  }
  let tipV = -1;
  let tipDist = -1;
  for (const v of targetFree) {
    const d = Math.hypot(
        folded[v][0] - sc[0], folded[v][1] - sc[1], folded[v][2] - sc[2]);
    if (d > tipDist) {
      tipDist = d;
      tipV = v;
    }
  }

  // Always prefer directions `u` where `rawTop > shMax + 2e-4` (`isClean = true`,
  // zero locked-vertex overhang) before falling back to `shMean`.
  const evalDir = (dir) => {
    const u = normalize3(dir);
    let shMax = -Infinity;
    let shMean = 0;
    for (const v of sharedVerts) {
      const sv = dot3(folded[v], u);
      shMax = Math.max(shMax, sv);
      shMean += sv / sharedVerts.size;
    }
    let subMax = -Infinity;
    for (const v of targetFree) subMax = Math.max(subMax, dot3(folded[v], u));
    let frMax = subMax;
    for (const v of freeVerts) frMax = Math.max(frMax, dot3(folded[v], u));

    const rawTop = subMax > shMax + 1e-4 ? subMax : frMax;
    const isClean = rawTop > shMax + 2e-4;
    const baseRef = isClean ? shMax : shMean;
    const fullSpan = rawTop - baseRef;

    let sStart = baseRef;
    let sEnd = rawTop;
    if (useSubRange && fullSpan > 1e-4) {
      let minFreeTau = Infinity;
      for (const v of freeVerts) {
        minFreeTau = Math.min(minFreeTau, tauByVert.get(v) ?? 1);
      }
      if (minFreeTau > fracStart + 0.15) {
        sStart = baseRef + fracStart * fullSpan;
        sEnd = baseRef + fracEnd * fullSpan;
      }
    }
    return {
      u,
      isClean,
      sSharedMax: sStart,
      sSubMax: sEnd,
      sFreeMax: frMax,
      gap: sEnd - sStart,
    };
  };

  const candidates = [
    [folded[tipV][0] - sc[0], folded[tipV][1] - sc[1], folded[vOrZero(tipV, folded)][2] - sc[2]],
    [fc[0] - sc[0], fc[1] - sc[1], fc[2] - sc[2]],
  ];
  for (const dx of [-1, 0, 1]) {
    for (const dy of [-1, 0, 1]) {
      for (const dz of [-1, 0, 1]) {
        if (dx !== 0 || dy !== 0 || dz !== 0) candidates.push([dx, dy, dz]);
      }
    }
  }

  let best = null;
  for (const c of candidates) {
    if (norm3(c) < 1e-9) continue;
    const res = evalDir(c);
    if (!best) {
      best = res;
    } else if (res.isClean !== best.isClean) {
      if (res.isClean) best = res;
    } else if (res.gap > best.gap) {
      best = res;
    }
  }
  if (!best || best.gap <= 1e-4) return null;

  if (preferredU) {
    const pref = evalDir(preferredU);
    if (pref.isClean === best.isClean &&
        pref.gap >= Math.max(1e-3, 0.55 * best.gap)) {
      best = pref;
    }
  }
  return {
    ...best,
    downstreamFaces,
    subSegmentFaces,
    freeVerts,
    targetFree,
    sharedVerts,
    lockedOutsideVerts,
    carryOnlyVerts,
    distalSeamVerts,
    carryFaces,
    carryComps,
  };
}

/**
 * @param {number} idx
 * @param {!Array<!Array<number>>} arr
 * @return {number}
 */
function vOrZero(idx, arr) {
  return idx >= 0 && idx < arr.length ? idx : 0;
}

/**
 * @param {!Object} state
 * @param {!Set<number>} faceSet
 * @param {!Array<number>} u
 * @return {!Array<number>}
 */
function computeDominantSheetNormal(state, faceSet, u) {
  const {folded, faces} = state;
  let refNormal = null;
  let bestArea2 = 0;
  const faceVectors = [];

  for (const f of faceSet) {
    const poly = faces[f];
    if (!poly) continue;
    const n = [0, 0, 0];
    for (let i = 0; i < poly.length; i++) {
      const p = folded[poly[i]];
      const q = folded[poly[(i + 1) % poly.length]];
      n[0] += (p[1] - q[1]) * (p[2] + q[2]);
      n[1] += (p[2] - q[2]) * (p[0] + q[0]);
      n[2] += (p[0] - q[0]) * (p[1] + q[1]);
    }
    const len = norm3(n);
    if (len <= 1e-12) continue;
    faceVectors.push(n);
    if (len > bestArea2) {
      bestArea2 = len;
      refNormal = [n[0] / len, n[1] / len, n[2] / len];
    }
  }

  let sheetN = null;
  if (refNormal) {
    const sum = [0, 0, 0];
    for (const n of faceVectors) {
      const sign = dot3(n, refNormal) >= 0 ? 1 : -1;
      sum[0] += sign * n[0];
      sum[1] += sign * n[1];
      sum[2] += sign * n[2];
    }
    const proj = dot3(sum, u);
    const ortho = [sum[0] - proj * u[0], sum[1] - proj * u[1], sum[2] - proj * u[2]];
    if (norm3(ortho) > 1e-8) {
      sheetN = normalize3(ortho);
    }
  }

  if (!sheetN) {
    const worldUp = [0, 1, 0];
    const worldFront = [0, 0, 1];
    const ref = Math.abs(dot3(u, worldUp)) < 0.9 ? worldUp : worldFront;
    const pitchAxis = normalize3(cross3(ref, u));
    sheetN = normalize3(cross3(pitchAxis, u));
  }

  const prefUp = Math.abs(sheetN[1]) > 0.15 ? [0, 1, 0] : [0, 0, 1];
  if (dot3(sheetN, prefUp) < 0) {
    sheetN = [-sheetN[0], -sheetN[1], -sheetN[2]];
  }
  return sheetN;
}

/**
 * Recomputes signed dihedral fold angles and `'M'` / `'V'` assignments on
 * `targetEdges` from adjacent 3D face normals across the mesh.
 *
 * @param {!Object} state
 * @param {!Set<number>} targetEdges
 * @param {?string=} fallbackAssign
 */
function updateRegionDihedralAngles(
    state, targetEdges, fallbackAssign = null) {
  if (!targetEdges.size) return;
  const {folded, faces} = state;

  const targetVerts = new Set();
  for (const e of targetEdges) {
    targetVerts.add(state.edges[e][0]);
    targetVerts.add(state.edges[e][1]);
  }

  const faceNormals = new Map();
  const leftFace = new Map();
  for (let f = 0; f < faces.length; f++) {
    const poly = faces[f];
    if (!poly.some((v) => targetVerts.has(v))) continue;
    const n = [0, 0, 0];
    for (let i = 0; i < poly.length; i++) {
      const p = folded[poly[i]];
      const q = folded[poly[(i + 1) % poly.length]];
      n[0] += p[1] * q[2] - p[2] * q[1];
      n[1] += p[2] * q[0] - p[0] * q[2];
      n[2] += p[0] * q[1] - p[1] * q[0];
    }
    const len = norm3(n);
    if (len <= 1e-12) continue;
    faceNormals.set(f, [n[0] / len, n[1] / len, n[2] / len]);
    for (let i = 0; i < poly.length; i++) {
      const prev = poly[(i - 1 + poly.length) % poly.length];
      const curr = poly[i];
      leftFace.set(`${prev}_${curr}`, f);
    }
  }

  for (const e of targetEdges) {
    if (state.assignments[e] === 'B') continue;
    const [start, end] = state.edges[e];
    const lf = leftFace.get(`${start}_${end}`);
    const rf = leftFace.get(`${end}_${start}`);
    const left = lf !== undefined ? faceNormals.get(lf) : null;
    const right = rf !== undefined ? faceNormals.get(rf) : null;
    if (left && right) {
      const dir = normalize3([
        folded[end][0] - folded[start][0],
        folded[end][1] - folded[start][1],
        folded[end][2] - folded[start][2],
      ]);
      const cr = cross3(left, right);
      const deg = Math.atan2(-dot3(cr, dir), dot3(left, right)) * 180 / Math.PI;
      if (Math.abs(deg) >= 1.0) {
        state.foldAngles[e] = deg;
        state.assignments[e] = deg < 0 ? 'M' : 'V';
        continue;
      }
    }
    if (fallbackAssign) {
      state.assignments[e] = fallbackAssign;
      state.foldAngles[e] = fallbackAssign === 'M' ? -35 : 35;
    } else {
      state.foldAngles[e] = 0;
      state.assignments[e] = 'F';
    }
  }
}

/**
 * Classifies each disconnected downstream component (`carryComps`) of a river
 * relative to a candidate cut plane `(planeNormal, offset)`:
 *   - Components entirely on `dist > 0` (`sMin > offset`) move rigidly with the
 *     distal half (`carryVerts`), adding zero creases to those flaps.
 *   - Components entirely on `dist <= 0` (`sMax <= offset`) stay stationary
 *     (`lockedForSlice`), so side limbs upstream of a mid-river cut never tear.
 *   - Components whose seam straddles `offset` (when `offset` cannot be nudged
 *     outside `[sMin, sMax]` within `[safeMin, safeMax]`) are added to
 *     `facesToSlice` so the planar cut continues cleanly across the tilted seam
 *     with `0.00%` seam strain.
 *
 * @param {!Object} state
 * @param {!Set<number>} candidateFaces
 * @param {!Set<number>} lockedVerts
 * @param {!Array<!Set<number>>} carryComps
 * @param {!Array<number>} planeNormal
 * @param {number} offset
 * @param {number} safeMin
 * @param {number} safeMax
 * @return {{
 *   facesToSlice: !Set<number>,
 *   lockedForSlice: !Set<number>,
 *   carryVerts: !Set<number>,
 *   adjustedOffset: number,
 * }}
 */
function prepareRiverCarryAndOffset(
    state, candidateFaces, lockedVerts, carryComps, planeNormal, offset,
    safeMin, safeMax) {
  const facesToSlice = new Set(candidateFaces);
  const lockedForSlice = new Set(lockedVerts);
  const carryVerts = new Set();
  let adjustedOffset = offset;
  if (!carryComps?.length) {
    return {facesToSlice, lockedForSlice, carryVerts, adjustedOffset};
  }

  const candVerts = new Set();
  for (const f of candidateFaces) {
    for (const v of state.faces[f]) candVerts.add(v);
  }

  const compInfos = carryComps.map((comp) => {
    const verts = new Set();
    const seamVerts = [];
    let hasLocked = false;
    for (const f of comp) {
      for (const v of state.faces[f]) {
        verts.add(v);
        if (lockedVerts.has(v)) hasLocked = true;
      }
    }
    for (const v of verts) {
      if (candVerts.has(v) || lockedVerts.has(v)) seamVerts.push(v);
    }
    let sMin = Infinity;
    let sMax = -Infinity;
    for (const v of seamVerts) {
      const s = dot3(state.folded[v], planeNormal);
      if (s < sMin) sMin = s;
      if (s > sMax) sMax = s;
    }
    return {comp, verts, seamVerts, hasLocked, sMin, sMax};
  });

  for (const info of compInfos) {
    if (!Number.isFinite(info.sMin) || info.hasLocked) continue;
    if (adjustedOffset >= info.sMin - 1.5e-4 &&
        adjustedOffset <= info.sMax + 1.5e-4) {
      if (adjustedOffset <= 0.5 * (info.sMin + info.sMax) &&
          info.sMin - 2e-4 >= safeMin) {
        adjustedOffset = info.sMin - 2e-4;
      } else if (info.sMax + 2e-4 <= safeMax) {
        adjustedOffset = info.sMax + 2e-4;
      }
    }
  }

  for (const info of compInfos) {
    if (!Number.isFinite(info.sMin)) continue;
    if (!info.hasLocked && info.sMin > adjustedOffset + 1e-6) {
      for (const v of info.verts) {
        if (!lockedForSlice.has(v)) carryVerts.add(v);
      }
    } else if (info.sMax <= adjustedOffset - 1e-6) {
      for (const v of info.verts) lockedForSlice.add(v);
    } else {
      for (const f of info.comp) facesToSlice.add(f);
    }
  }

  return {facesToSlice, lockedForSlice, carryVerts, adjustedOffset};
}

/**
 * Shared planar mesh slicer: splits `candidateFaces` along `x · planeNormal =
 * planeOffset` at identical barycentric `t` in both 2D `flat` and 3D `folded`
 * (`0.000%` splitting strain), keeping all `lockedVerts` strictly stationary
 * and preserving each face's original region index.
 *
 * @param {!Object} state
 * @param {!Set<number>} candidateFaces
 * @param {!Array<number>} planeNormal
 * @param {number} planeOffset
 * @param {!Set<number>} lockedVerts
 * @return {?{
 *   candVerts: !Set<number>,
 *   dist: !Array<number>,
 *   cutEdgeIndices: !Array<number>,
 *   proximalFaces: !Set<number>,
 *   distalFaces: !Set<number>,
 *   effectiveOffset: number,
 * }}
 */
function sliceMeshByPlaneCore(
    state, candidateFaces, planeNormal, planeOffset, lockedVerts) {
  if (!candidateFaces.size) return null;
  const {flat, folded, faces, faceRegions} = state;

  const candVerts = new Set();
  for (const f of candidateFaces) {
    for (const v of faces[f]) candVerts.add(v);
  }

  let maxLockedProj = -Infinity;
  let maxFreeProj = -Infinity;
  for (const v of candVerts) {
    const p = dot3(folded[v], planeNormal);
    if (lockedVerts.has(v)) {
      if (p > maxLockedProj) maxLockedProj = p;
    } else if (p > maxFreeProj) {
      maxFreeProj = p;
    }
  }
  if (!Number.isFinite(maxFreeProj)) return null;

  let effectiveOffset = planeOffset;
  if (Number.isFinite(maxLockedProj)) {
    if (maxLockedProj + 1e-4 < maxFreeProj - 2e-5) {
      effectiveOffset = Math.max(planeOffset, maxLockedProj + 1e-4);
    } else if (state.regionCutCount > 0) {
      // Secondary cut on this region (e.g. Cut 2 of 2-sided narrow or crimp)
      // would cross a locked seam/crease: keep Cut 1's clean creases intact.
      return null;
    }
  }

  const dist = new Array(flat.length).fill(-1);
  let hasPositive = false;
  for (const v of candVerts) {
    const raw = dot3(folded[v], planeNormal) - effectiveOffset;
    if (lockedVerts.has(v)) {
      dist[v] = Math.min(-1e-5, raw);
    } else {
      dist[v] = Math.abs(raw) < 1e-6 ? 0 : raw;
      if (dist[v] > 0) hasPositive = true;
    }
  }
  if (!hasPositive) return null;

  const edgeMap = new Map();
  state.edges.forEach(([a, b], idx) => edgeMap.set(undirectedKey(a, b), idx));
  const addEdgeRecord = (a, b, assign, ang, baseAng, isShap, isUserShap = false) => {
    if (a === b) return;
    const key = undirectedKey(a, b);
    if (edgeMap.has(key)) return;
    edgeMap.set(key, state.edges.length);
    state.edges.push([a, b]);
    state.assignments.push(assign);
    state.foldAngles.push(ang);
    state.baseFoldAngles.push(baseAng);
    state.shaping.push(isShap);
    state.userShaping.push(isUserShap);
  };

  const origFaceCount = faces.length;
  for (let f = 0; f < origFaceCount; f++) {
    if (!candidateFaces.has(f)) continue;
    const poly = faces[f];
    if (poly.length <= 3) continue;
    let signChanges = 0;
    let prevSign = 0;
    for (let i = 0; i <= poly.length; i++) {
      const dv = dist[poly[i % poly.length]];
      const s = dv > 0 ? 1 : (dv < 0 ? -1 : 0);
      if (s !== 0) {
        if (prevSign !== 0 && s !== prevSign) signChanges++;
        prevSign = s;
      }
    }
    if (signChanges > 2) {
      const tris = earClipPolygon2D(poly, flat);
      if (tris.length > 1) {
        const fReg = faceRegions[f];
        faces[f] = tris[0];
        for (let t = 1; t < tris.length; t++) {
          candidateFaces.add(faces.length);
          faces.push(tris[t]);
          faceRegions.push(fReg);
        }
        for (const tri of tris) {
          for (let k = 0; k < 3; k++) {
            addEdgeRecord(tri[k], tri[(k + 1) % 3], 'F', 0, 0, true, false);
          }
        }
      }
    }
  }

  let anyCrossing = false;
  for (let e = 0; e < state.edges.length; e++) {
    const [a, b] = state.edges[e];
    if (candVerts.has(a) && candVerts.has(b) &&
        (dist[a] * dist[b] < 0 || (dist[a] === 0 && dist[b] === 0))) {
      anyCrossing = true;
      break;
    }
  }
  if (!anyCrossing) return null;

  const splitVertexByEdge = new Map();
  const newEdges = [];
  const newAssignments = [];
  const newFoldAngles = [];
  const newBaseFoldAngles = [];
  const newShaping = [];
  const newUserShaping = [];

  for (let e = 0; e < state.edges.length; e++) {
    const [a, b] = state.edges[e];
    if (!candVerts.has(a) || !candVerts.has(b) || dist[a] * dist[b] >= 0) {
      newEdges.push([a, b]);
      newAssignments.push(state.assignments[e]);
      newFoldAngles.push(state.foldAngles[e]);
      newBaseFoldAngles.push(state.baseFoldAngles[e]);
      newShaping.push(state.shaping[e]);
      newUserShaping.push(state.userShaping[e]);
      continue;
    }
    const t = Math.max(1e-4, Math.min(1 - 1e-4, dist[a] / (dist[a] - dist[b])));
    const w = flat.length;
    flat.push([
      flat[a][0] + t * (flat[b][0] - flat[a][0]),
      flat[a][1] + t * (flat[b][1] - flat[a][1]),
    ]);
    folded.push([
      folded[a][0] + t * (folded[b][0] - folded[a][0]),
      folded[a][1] + t * (folded[b][1] - folded[a][1]),
      folded[a][2] + t * (folded[b][2] - folded[a][2]),
    ]);
    dist.push(0);
    candVerts.add(w);
    splitVertexByEdge.set(undirectedKey(a, b), w);

    for (const [u0, v0] of [[a, w], [w, b]]) {
      newEdges.push([u0, v0]);
      newAssignments.push(state.assignments[e]);
      newFoldAngles.push(state.foldAngles[e]);
      newBaseFoldAngles.push(state.baseFoldAngles[e]);
      newShaping.push(state.shaping[e]);
      newUserShaping.push(state.userShaping[e]);
    }
  }

  state.edges = newEdges;
  state.assignments = newAssignments;
  state.foldAngles = newFoldAngles;
  state.baseFoldAngles = newBaseFoldAngles;
  state.shaping = newShaping;
  state.userShaping = newUserShaping;

  edgeMap.clear();
  state.edges.forEach(([a, b], idx) => edgeMap.set(undirectedKey(a, b), idx));

  const nextFaces = [];
  const nextFaceRegions = [];
  const proximalFaces = new Set();
  const distalFaces = new Set();
  const cutEdgeIndices = [];

  for (let f = 0; f < faces.length; f++) {
    if (!candidateFaces.has(f)) {
      nextFaces.push(faces[f]);
      nextFaceRegions.push(faceRegions[f]);
      continue;
    }
    const fReg = faceRegions[f];
    const poly = faces[f];
    const aug = [];
    for (let i = 0; i < poly.length; i++) {
      const a = poly[i];
      const b = poly[(i + 1) % poly.length];
      aug.push(a);
      const mid = splitVertexByEdge.get(undirectedKey(a, b));
      if (mid !== undefined) aug.push(mid);
    }
    const subPolys = splitTwoCrossingPolygon(aug, dist);
    const zeros = aug.filter((v) => dist[v] === 0);
    if (zeros.length === 2 && zeros[0] !== zeros[1]) {
      const key = undirectedKey(zeros[0], zeros[1]);
      let cIdx = edgeMap.get(key);
      if (cIdx === undefined) {
        cIdx = state.edges.length;
        edgeMap.set(key, cIdx);
        state.edges.push([zeros[0], zeros[1]]);
        state.assignments.push('V');
        state.foldAngles.push(0);
        state.baseFoldAngles.push(0);
        state.shaping.push(true);
        state.userShaping.push(true);
      } else if (state.assignments[cIdx] !== 'B') {
        state.shaping[cIdx] = true;
        state.userShaping[cIdx] = true;
        state.baseFoldAngles[cIdx] = 0;
      }
      if (state.assignments[cIdx] !== 'B') {
        cutEdgeIndices.push(cIdx);
      }
    }

    for (const sp of subPolys) {
      const outIdx = nextFaces.length;
      nextFaces.push(sp);
      nextFaceRegions.push(fReg);
      if (sp.some((v) => dist[v] > 0)) {
        distalFaces.add(outIdx);
      } else {
        proximalFaces.add(outIdx);
      }
    }
  }

  state.faces = nextFaces;
  state.faceRegions = nextFaceRegions;

  if (!cutEdgeIndices.length) return null;
  state.addedUserCreases += cutEdgeIndices.length;
  state.regionCutCount++;
  return {
    candVerts,
    dist,
    cutEdgeIndices,
    proximalFaces,
    distalFaces,
    effectiveOffset,
  };
}

/**
 * Exact zero-strain 3D plane reflection (`R_Pi`) across `x · planeNormal =
 * effectiveOffset`, carrying any `carryVerts` rigidly.
 *
 * @param {!Object} state
 * @param {!Set<number>} candidateFaces
 * @param {!Array<number>} planeNormal
 * @param {number} planeOffset
 * @param {string} assignChar
 * @param {!Set<number>} lockedVerts
 * @param {!Set<number>=} carryVerts
 * @param {?Array<number>=} biasDir
 * @return {{
 *   addedCreases: number,
 *   proximalFaces: !Set<number>,
 *   distalFaces: !Set<number>,
 * }}
 */
function sliceReflectFold(
    state, candidateFaces, planeNormal, planeOffset, assignChar,
    lockedVerts, carryVerts = new Set(), biasDir = null) {
  const sliced = sliceMeshByPlaneCore(
      state, candidateFaces, planeNormal, planeOffset, lockedVerts);
  if (!sliced) {
    return {
      addedCreases: 0,
      proximalFaces: new Set(candidateFaces),
      distalFaces: new Set(),
    };
  }
  const {
    candVerts,
    dist,
    cutEdgeIndices,
    proximalFaces,
    distalFaces,
    effectiveOffset,
  } = sliced;
  const {folded} = state;

  const movedVerts = new Set();
  for (const v of candVerts) {
    if (dist[v] > 0) movedVerts.add(v);
  }
  for (const v of carryVerts) {
    if (!candVerts.has(v)) movedVerts.add(v);
  }

  for (const v of movedVerts) {
    const pRef = reflectPointAcrossPlane(folded[v], planeNormal, effectiveOffset);
    if (biasDir) {
      const d = Math.max(0, dot3(folded[v], planeNormal) - effectiveOffset);
      const b = 0.0012 * Math.min(1, d / 0.015);
      pRef[0] += b * biasDir[0];
      pRef[1] += b * biasDir[1];
      pRef[2] += b * biasDir[2];
    }
    folded[v] = pRef;
  }

  const cutSet = new Set(cutEdgeIndices);
  for (let e = 0; e < state.edges.length; e++) {
    if (cutSet.has(e) || state.assignments[e] === 'B') continue;
    const [a, b] = state.edges[e];
    if (movedVerts.has(a) && movedVerts.has(b)) {
      state.foldAngles[e] = -state.foldAngles[e];
      if (state.assignments[e] === 'M') state.assignments[e] = 'V';
      else if (state.assignments[e] === 'V') state.assignments[e] = 'M';
    }
  }

  for (const cIdx of cutEdgeIndices) {
    state.assignments[cIdx] = assignChar;
    state.foldAngles[cIdx] = assignChar === 'M' ? -178 : 178;
  }

  return {
    addedCreases: cutEdgeIndices.length,
    proximalFaces,
    distalFaces,
  };
}

/**
 * Slices `candidateFaces` and folds `dist > 0` vertices (plus any downstream
 * `carryVerts`) around the sheet-tangent hinge `sheetNormal × planeNormal`
 * with globally continuous collar blending across river-to-flap seams.
 *
 * @param {!Object} state
 * @param {!Set<number>} candidateFaces
 * @param {!Array<number>} planeNormal
 * @param {number} planeOffset
 * @param {number} foldRad
 * @param {!Array<number>} sheetNormal
 * @param {!Set<number>} lockedVerts
 * @param {!Set<number>=} carryVerts
 * @return {{
 *   addedCreases: number,
 *   pivot: !Array<number>,
 *   hingeAxis: !Array<number>,
 *   proximalFaces: !Set<number>,
 *   distalFaces: !Set<number>,
 * }}
 */
function sliceHingeFold(
    state, candidateFaces, planeNormal, planeOffset, foldRad,
    sheetNormal, lockedVerts, carryVerts = new Set()) {
  const rawHinge = cross3(sheetNormal, planeNormal);
  const hingeAxis = norm3(rawHinge) > 1e-8 ?
      normalize3(rawHinge) :
      normalize3(cross3([0, 1, 0], planeNormal));

  if (Math.abs(foldRad) < 1e-6) {
    return {
      addedCreases: 0,
      pivot: [0, 0, 0],
      hingeAxis,
      proximalFaces: new Set(candidateFaces),
      distalFaces: new Set(),
    };
  }

  const sliced = sliceMeshByPlaneCore(
      state, candidateFaces, planeNormal, planeOffset, lockedVerts);
  if (!sliced) {
    return {
      addedCreases: 0,
      pivot: [0, 0, 0],
      hingeAxis,
      proximalFaces: new Set(candidateFaces),
      distalFaces: new Set(),
    };
  }
  const {
    candVerts,
    dist,
    cutEdgeIndices,
    proximalFaces,
    distalFaces,
    effectiveOffset,
  } = sliced;
  const {folded} = state;

  const zeroVerts = [];
  for (const v of candVerts) {
    if (dist[v] === 0) zeroVerts.push(v);
  }
  const pivot = [0, 0, 0];
  for (const v of zeroVerts) {
    pivot[0] += folded[v][0] / zeroVerts.length;
    pivot[1] += folded[v][1] / zeroVerts.length;
    pivot[2] += folded[v][2] / zeroVerts.length;
  }

  let maxOffAxisH = 0;
  for (const v of zeroVerts) {
    const rx = folded[v][0] - pivot[0];
    const ry = folded[v][1] - pivot[1];
    const rz = folded[v][2] - pivot[2];
    const along = rx * hingeAxis[0] + ry * hingeAxis[1] + rz * hingeAxis[2];
    const h = Math.hypot(
        rx - along * hingeAxis[0],
        ry - along * hingeAxis[1],
        rz - along * hingeAxis[2]);
    if (h > maxOffAxisH) maxOffAxisH = h;
  }
  const collar = maxOffAxisH > 1e-4 ? 2.5 * maxOffAxisH : 0;

  for (const v of candVerts) {
    if (dist[v] > 0) {
      const blend = collar > 1e-5 ? Math.min(1, dist[v] / collar) : 1;
      const eff = foldRad * blend;
      folded[v] = rotatePointAround(folded[v], pivot, [
        hingeAxis[0] * eff,
        hingeAxis[1] * eff,
        hingeAxis[2] * eff,
      ]);
    }
  }
  for (const v of carryVerts) {
    if (!candVerts.has(v)) {
      const d = Math.max(0, dot3(folded[v], planeNormal) - effectiveOffset);
      const blend = collar > 1e-5 ? Math.min(1, d / collar) : 1;
      const eff = foldRad * blend;
      folded[v] = rotatePointAround(folded[v], pivot, [
        hingeAxis[0] * eff,
        hingeAxis[1] * eff,
        hingeAxis[2] * eff,
      ]);
    }
  }

  const fallbackAssign = foldRad > 0 ? 'M' : 'V';
  updateRegionDihedralAngles(
      state, new Set(cutEdgeIndices), fallbackAssign);

  return {
    addedCreases: cutEdgeIndices.length,
    pivot,
    hingeAxis,
    proximalFaces,
    distalFaces,
  };
}

/**
 * Diagonal side-tuck / taper fold (`narrow`) guaranteed to intersect free flap
 * vertices across rectangular, triangular, and pointed flaps.
 *
 * @param {!Object} state
 * @param {number} regionIdx
 * @param {number} fracStart
 * @param {number} fracEnd
 * @param {number} narrowPct
 * @param {number} tuckDeg
 * @param {number} depth
 * @param {boolean} bothSides
 * @param {?Array<number>=} preferredU
 * @param {boolean=} isMirror
 */
function narrowFlapRegion(
    state, regionIdx, fracStart, fracEnd, narrowPct, tuckDeg, depth, bothSides,
    preferredU = null, isMirror = false) {
  // Rivers and leftover reserved pockets cannot be side-tuck narrowed.
  if (state.regions[regionIdx]?.kind !== 'flap') return;
  state.regionCutCount = 0;

  const clampedNarrow = Math.max(0, Math.min(85, narrowPct));
  if (clampedNarrow < 1 || Math.abs(tuckDeg) < 2) return;

  const span =
      findFlapFreeSpan(state, regionIdx, fracStart, fracEnd, preferredU);
  if (!span || span.gap <= 1e-4) return;

  const u = span.u;
  const sheetN = computeDominantSheetNormal(state, span.subSegmentFaces, u);
  const w = normalize3(cross3(sheetN, u));

  const {folded} = state;
  const freeTarget = span.targetFree;
  if (!freeTarget.size) return;

  const spinePt = [0, 0, 0];
  for (const v of freeTarget) {
    spinePt[0] += folded[v][0] / freeTarget.size;
    spinePt[1] += folded[v][1] / freeTarget.size;
    spinePt[2] += folded[v][2] / freeTarget.size;
  }

  let yMin = Infinity;
  let yMax = -Infinity;
  for (const v of freeTarget) {
    const y = dot3([
      folded[v][0] - spinePt[0],
      folded[v][1] - spinePt[1],
      folded[v][2] - spinePt[2],
    ], w);
    yMin = Math.min(yMin, y);
    yMax = Math.max(yMax, y);
  }
  const width = Math.max(1e-3, yMax - yMin);
  const halfW = 0.5 * width;

  const depthClamped = Math.max(0.05, Math.min(0.75, depth));
  const axialLen = Math.max(1e-3, (1 - 0.55 * depthClamped) * span.gap);
  const narrowFrac = Math.min(0.84, clampedNarrow / 100);
  const tanAlpha = Math.max(0.12, Math.min(1.35, (narrowFrac * halfW) / axialLen));
  const alpha = Math.atan(tanAlpha);

  const useFlatReflection = Math.abs(tuckDeg) >= 145;
  const tuckSign = tuckDeg >= 0 ? 1 : -1;
  const tuckRad = Math.max(-172, Math.min(172, tuckDeg)) * Math.PI / 180;
  let activeFaces = new Set(span.downstreamFaces);
  const baseLocked = new Set(span.lockedOutsideVerts);

  const sides = bothSides ? [1, -1] : [isMirror ? -1 : 1];
  for (const sideSign of sides) {
    const lockedSet = new Set(baseLocked);

    // Choose diagonal normal `m` (tilting toward `u` if a wide locked root seam
    // sticks out sideways) so `lockedMax < mMax` whenever possible.
    let m = null;
    let mMax = -Infinity;
    let mMin = Infinity;
    let lockedMax = -Infinity;
    for (const tiltBoost of [0, 0.35, 0.65, 0.85, 0.96]) {
      const effAlpha = alpha + tiltBoost * (0.5 * Math.PI - alpha);
      const cA = Math.cos(effAlpha);
      const sA = Math.sin(effAlpha);
      const candM = normalize3([
        sideSign * cA * w[0] + sA * u[0],
        sideSign * cA * w[1] + sA * u[1],
        sideSign * cA * w[2] + sA * u[2],
      ]);
      let hi = -Infinity;
      let lo = Infinity;
      let lHi = -Infinity;
      for (const f of activeFaces) {
        for (const v of state.faces[f]) {
          const val = dot3(state.folded[v], candM);
          if (lockedSet.has(v)) {
            if (val > lHi) lHi = val;
          } else {
            if (val > hi) hi = val;
            if (val < lo) lo = val;
          }
        }
      }
      m = candM;
      mMax = hi;
      mMin = lo;
      lockedMax = lHi;
      if (!Number.isFinite(lHi) || hi > lHi + 3e-4) break;
    }
    if (!m || !Number.isFinite(mMax)) break;

    const mSpine = dot3(spinePt, m);
    let refFloor = Math.min(mMax - 2e-4, Math.max(mMin, mSpine));
    if (Number.isFinite(lockedMax) && lockedMax < mMax - 3e-4) {
      refFloor = Math.max(refFloor, lockedMax + 1e-4);
    }
    const spanM = Math.max(2e-4, mMax - refFloor);
    const effFrac = Math.max(
        0.08, Math.min(0.72, narrowFrac * (1 - 0.3 * depthClamped)));
    const offset = mMax - effFrac * spanM;

    const hasLockOverhang = Number.isFinite(lockedMax) && lockedMax >= offset - 1e-5;
    const res = (useFlatReflection && !hasLockOverhang) ?
        sliceReflectFold(
            state, activeFaces, m, offset,
            tuckSign > 0 ? 'M' : 'V', lockedSet, new Set(),
            [-tuckSign * sheetN[0], -tuckSign * sheetN[1], -tuckSign * sheetN[2]]) :
        sliceHingeFold(
            state, activeFaces, m, offset, tuckRad, sheetN,
            lockedSet, new Set());
    if (res.addedCreases > 0 && res.proximalFaces.size > 0) {
      for (const df of res.distalFaces) {
        for (const v of state.faces[df]) baseLocked.add(v);
      }
      activeFaces = res.proximalFaces;
    }
  }
}

/**
 * Exact Cartan–Dieudonné double-reflection 2-crease Crimp (`R_Pi2 o R_Pi1`):
 * `0.00%` additional strain and proper `SO(3)` (`det = +1`) orientation.
 *
 * @param {!Object} state
 * @param {number} regionIdx
 * @param {number} fracStart
 * @param {number} fracEnd
 * @param {number} turnDeg
 * @param {number} depth
 * @param {?Array<number>=} preferredU
 * @param {boolean=} isMirror
 */
function crimpTurnFlapRegion(
    state, regionIdx, fracStart, fracEnd, turnDeg, depth,
    preferredU = null, isMirror = false) {
  const effectiveTurn = isMirror ? -turnDeg : turnDeg;
  if (Math.abs(effectiveTurn) < 1) return;
  state.regionCutCount = 0;

  const span =
      findFlapFreeSpan(state, regionIdx, fracStart, fracEnd, preferredU);
  if (!span || span.gap <= 1e-4) return;

  const u = span.u;
  const sheetN = computeDominantSheetNormal(state, span.subSegmentFaces, u);
  const w = normalize3(cross3(sheetN, u));

  // Choose `beta <= effectiveTurn / 4` such that `m1` keeps `m1SharedMax < m1FreeMax`
  // (zero locked-vertex overhang -> exact `0.000%` isometric reflection!).
  let beta = (effectiveTurn * Math.PI / 180) / 4;
  let m1 = u;
  let m2Orig = u;
  let m1SharedMax = -Infinity;
  let m1SharedMean = 0;
  let m1FreeMax = -Infinity;
  let m1FreeMin = Infinity;
  for (const scale of [1.0, 0.65, 0.35, 0.15, 0.05]) {
    const b = beta * scale;
    const cosB = Math.cos(b);
    const sinB = Math.sin(b);
    const candM1 = normalize3([
      cosB * u[0] + sinB * w[0],
      cosB * u[1] + sinB * w[1],
      cosB * u[2] + sinB * w[2],
    ]);
    let sMax = -Infinity;
    let sMean = 0;
    for (const v of span.sharedVerts) {
      const val = dot3(state.folded[v], candM1);
      if (val > sMax) sMax = val;
      sMean += val / Math.max(1, span.sharedVerts.size);
    }
    let fMax = -Infinity;
    let fMin = Infinity;
    for (const v of span.freeVerts) {
      const val = dot3(state.folded[v], candM1);
      if (val > fMax) fMax = val;
      if (val < fMin) fMin = val;
    }
    m1 = candM1;
    m2Orig = normalize3([
      cosB * u[0] - sinB * w[0],
      cosB * u[1] - sinB * w[1],
      cosB * u[2] - sinB * w[2],
    ]);
    m1SharedMax = sMax;
    m1SharedMean = sMean;
    m1FreeMax = fMax;
    m1FreeMin = fMin;
    if (!Number.isFinite(sMax) || fMax > sMax + 4e-4) break;
  }

  const base1 = Number.isFinite(m1SharedMax) ?
      (m1SharedMax < m1FreeMax - 4e-4 ?
          m1SharedMax + 1.5e-4 :
          Math.min(m1SharedMean, m1FreeMax - 4e-4)) :
      m1FreeMin;
  const safeMin1 = base1;
  const safeMax1 = m1FreeMax - 1.5e-4;
  if (safeMax1 <= safeMin1) return;

  const clampedDepth = Math.max(0.10, Math.min(0.68, depth));
  const rawOffset1 = safeMin1 + clampedDepth * (safeMax1 - safeMin1);
  const prep1 = prepareRiverCarryAndOffset(
      state, span.downstreamFaces, span.lockedOutsideVerts, span.carryComps,
      m1, rawOffset1, safeMin1, safeMax1);

  const cut1 = sliceReflectFold(
      state, prep1.facesToSlice, m1, prep1.adjustedOffset, 'M',
      prep1.lockedForSlice, prep1.carryVerts, null);
  if (cut1.addedCreases === 0 || !cut1.distalFaces.size) return;

  // Re-partition river carry components on the updated mesh before Cut 2 so
  // any face split in Cut 1 has fresh vertex indices and exact seam bounds!
  const post1CarryComps = span.carryComps.length > 0 ?
      partitionSheetAcrossRegion(
          state.flat, state.faces, state.faceRegions, regionIdx).carryComps :
      [];

  const rawM2Ref = normalize3(reflectVectorAcrossPlane(m2Orig, m1));
  const negM1 = [-m1[0], -m1[1], -m1[2]];
  const lockedForCut2 = new Set(span.lockedOutsideVerts);
  for (const f of cut1.proximalFaces) {
    for (const v of state.faces[f]) lockedForCut2.add(v);
  }
  const distalRegionFaces = new Set();
  for (const f of cut1.distalFaces) {
    if (state.faceRegions[f] === regionIdx || !span.carryComps.length) {
      distalRegionFaces.add(f);
    }
  }
  const baseFaces2 = distalRegionFaces.size > 0 ? distalRegionFaces : cut1.distalFaces;

  // Interpolate `m2Ref` toward `-m1` if needed so `d2LockedMax < d2FreeMax`
  // (at `-m1`, every Cut-1 distal vertex strictly exceeds `-offset1 = d2LockedMax`).
  let m2Ref = rawM2Ref;
  let d2LockedMax = -Infinity;
  let d2LockedMean = 0;
  let d2LockedCount = 0;
  let d2FreeMax = -Infinity;
  let d2FreeMin = Infinity;
  for (const blendToNegM1 of [0, 0.35, 0.65, 0.85, 1.0]) {
    const candM2 = normalize3([
      (1 - blendToNegM1) * rawM2Ref[0] + blendToNegM1 * negM1[0],
      (1 - blendToNegM1) * rawM2Ref[1] + blendToNegM1 * negM1[1],
      (1 - blendToNegM1) * rawM2Ref[2] + blendToNegM1 * negM1[2],
    ]);
    let lMax = -Infinity;
    let lMean = 0;
    let lCnt = 0;
    let fMax = -Infinity;
    let fMin = Infinity;
    for (const f of baseFaces2) {
      for (const v of state.faces[f]) {
        const val = dot3(state.folded[v], candM2);
        if (lockedForCut2.has(v)) {
          if (val > lMax) lMax = val;
          lMean += val;
          lCnt++;
        } else {
          if (val > fMax) fMax = val;
          if (val < fMin) fMin = val;
        }
      }
    }
    m2Ref = candM2;
    d2LockedMax = lMax;
    d2LockedMean = lMean;
    d2LockedCount = lCnt;
    d2FreeMax = fMax;
    d2FreeMin = fMin;
    if (lCnt === 0 || fMax > lMax + 3e-4) break;
  }
  if (!Number.isFinite(d2FreeMax)) return;
  const lo2 = d2LockedCount > 0 ?
      (d2LockedMax < d2FreeMax - 3e-4 ?
          d2LockedMax + 1e-4 :
          Math.min(d2LockedMean / d2LockedCount, d2FreeMax - 3e-4)) :
      d2FreeMin;
  const safeMax2 = d2FreeMax - 1.5e-4;
  if (safeMax2 <= lo2) return;

  const rawOffset2 = lo2 + 0.32 * (safeMax2 - lo2);
  const prep2 = prepareRiverCarryAndOffset(
      state, baseFaces2, lockedForCut2, post1CarryComps,
      m2Ref, rawOffset2, lo2, safeMax2);

  sliceReflectFold(
      state, prep2.facesToSlice, m2Ref, prep2.adjustedOffset, 'V',
      prep2.lockedForSlice, prep2.carryVerts, null);
}

/**
 * Applies a pure crease-backed shaping action (`bend`, `narrow`, `crimp`,
 * `curve`, `pleat`) to `region` (and its L/R mirror when `mirror === true`).
 *
 * @param {!Object} foldDict
 * @param {!Object} params
 * @return {{
 *   foldDict: !Object,
 *   appliedRegions: !Array<number>,
 *   mirrorRegion: ?number,
 *   addedCreases: number,
 * }}
 */
export function applyFlapShaping(foldDict, params) {
  const {
    region,
    fracStart = 0,
    fracEnd = 1,
    mirror = true,
    mode = 'bend',
    pitchDeg = 0,
    turnDeg = 0,
    narrowPct = 0,
    tuckDeg = 165,
    bothSides = true,
    depth = 0.3,
    steps = 3,
  } = params;

  const state = unpackFoldMesh(foldDict);
  if (region == null || region < 0 || region >= state.regions.length ||
      state.regions[region]?.kind === 'reserved') {
    return {
      foldDict: packFoldMesh(state),
      appliedRegions: [],
      mirrorRegion: null,
      addedCreases: 0,
    };
  }

  const mirrorRegion = mirror ?
      findMirroredRegionIndex(state.regions, region) :
      null;
  const targets = [region];
  if (mirrorRegion !== null && mirrorRegion !== region) {
    targets.push(mirrorRegion);
  }

  const primarySpan = findFlapFreeSpan(state, region, fracStart, fracEnd);
  if (!primarySpan) {
    return {
      foldDict: packFoldMesh(state),
      appliedRegions: [],
      mirrorRegion,
      addedCreases: 0,
    };
  }

  const mirrorPrefU = [-primarySpan.u[0], primarySpan.u[1], primarySpan.u[2]];

  if (mode === 'narrow') {
    narrowFlapRegion(
        state, region, fracStart, fracEnd, narrowPct, tuckDeg, depth,
        bothSides, null, false);
    if (mirrorRegion !== null && mirrorRegion !== region) {
      narrowFlapRegion(
          state, mirrorRegion, fracStart, fracEnd, narrowPct, tuckDeg, depth,
          bothSides, mirrorPrefU, true);
    }
    return {
      foldDict: packFoldMesh(state),
      appliedRegions: targets,
      mirrorRegion,
      addedCreases: state.addedUserCreases,
    };
  }

  if (mode === 'crimp') {
    crimpTurnFlapRegion(
        state, region, fracStart, fracEnd, turnDeg, depth, null, false);
    if (mirrorRegion !== null && mirrorRegion !== region) {
      crimpTurnFlapRegion(
          state, mirrorRegion, fracStart, fracEnd, turnDeg, depth,
          mirrorPrefU, true);
    }
    return {
      foldDict: packFoldMesh(state),
      appliedRegions: targets,
      mirrorRegion,
      addedCreases: state.addedUserCreases,
    };
  }

  const clampedDepth = Math.max(0.05, Math.min(0.86, depth));
  const clampedPitch = Math.max(-170, Math.min(170, pitchDeg));

  const buildCutSteps = () => {
    const foldRad = -(clampedPitch * Math.PI / 180);
    if (mode === 'curve') {
      const n = Math.max(2, Math.min(6, Math.round(steps)));
      const list = [];
      for (let k = 0; k < n; k++) {
        const frac = clampedDepth + (k / n) * (0.9 - clampedDepth);
        list.push({frac, foldRad: foldRad / n});
      }
      return list;
    }
    if (mode === 'pleat') {
      const pairs = Math.max(1, Math.min(4, Math.round(steps)));
      const totalCuts = 2 * pairs;
      const list = [];
      for (let k = 0; k < totalCuts; k++) {
        const sign = k % 2 === 0 ? 1 : -1;
        const frac = clampedDepth +
            (k / Math.max(1, totalCuts)) * Math.min(0.55, 0.92 - clampedDepth);
        list.push({frac, foldRad: sign * foldRad});
      }
      return list;
    }
    return [{frac: clampedDepth, foldRad}];
  };

  const cutSteps = buildCutSteps();

  const applyStepsToRegion = (rIdx, isMirror) => {
    state.regionCutCount = 0;
    const prefU = isMirror ? mirrorPrefU : primarySpan.u;
    const initialSpan = findFlapFreeSpan(state, rIdx, fracStart, fracEnd, prefU);
    if (!initialSpan) return;

    let curU = [...initialSpan.u];
    let curSheetN =
        computeDominantSheetNormal(state, initialSpan.subSegmentFaces, curU);

    const baseS = initialSpan.sSharedMax + Math.max(1e-4, 0.04 * initialSpan.gap);
    const tipS = initialSpan.sSubMax - 0.03 * initialSpan.gap;
    if (tipS <= baseS) return;

    const refCentroid = [0, 0, 0];
    for (const v of initialSpan.freeVerts) {
      refCentroid[0] += state.folded[v][0] / initialSpan.freeVerts.size;
      refCentroid[1] += state.folded[v][1] / initialSpan.freeVerts.size;
      refCentroid[2] += state.folded[v][2] / initialSpan.freeVerts.size;
    }
    const cDot = dot3(refCentroid, curU);
    const axisOrigin = [
      refCentroid[0] - cDot * curU[0],
      refCentroid[1] - cDot * curU[1],
      refCentroid[2] - cDot * curU[2],
    ];
    const stepAnchors = cutSteps.map(({frac}) => {
      const s = baseS + frac * (tipS - baseS);
      return [
        axisOrigin[0] + s * curU[0],
        axisOrigin[1] + s * curU[1],
        axisOrigin[2] + s * curU[2],
      ];
    });

    let activeFaces = new Set(initialSpan.downstreamFaces);
    const lockedSet = new Set(initialSpan.lockedOutsideVerts);
    const isRiverRegion = initialSpan.carryComps.length > 0;

    for (let i = 0; i < cutSteps.length; i++) {
      const {foldRad} = cutSteps[i];
      if (Math.abs(foldRad) < 1e-6 || !activeFaces.size) continue;

      let lockMax = -Infinity;
      let lockMean = 0;
      let lockCount = 0;
      let distMax = -Infinity;
      let distMin = Infinity;
      for (const f of activeFaces) {
        for (const v of state.faces[f]) {
          const s = dot3(state.folded[v], curU);
          if (lockedSet.has(v)) {
            if (s > lockMax) lockMax = s;
            lockMean += s;
            lockCount++;
          } else {
            if (s > distMax) distMax = s;
            if (s < distMin) distMin = s;
          }
        }
      }
      if (!Number.isFinite(distMax)) break;
      const baseFloor = lockCount > 0 ?
          (lockMax < distMax - 3e-4 ?
              lockMax :
              Math.min(lockMean / lockCount, distMax - 3e-4)) :
          distMin;
      const safeMin = baseFloor + 1.5e-4;
      const safeMax = distMax - 1.5e-4;
      if (safeMax <= safeMin) break;

      const sCut = Math.max(safeMin, Math.min(safeMax, dot3(stepAnchors[i], curU)));
      const liveCarryComps = isRiverRegion ?
          partitionSheetAcrossRegion(
              state.flat, state.faces, state.faceRegions, rIdx).carryComps :
          [];
      const prep = prepareRiverCarryAndOffset(
          state, activeFaces, lockedSet, liveCarryComps,
          curU, sCut, safeMin, safeMax);

      const res = sliceHingeFold(
          state, prep.facesToSlice, curU, prep.adjustedOffset, foldRad,
          curSheetN, prep.lockedForSlice, prep.carryVerts);
      if (res.addedCreases === 0 || !res.distalFaces.size) break;

      for (const pf of res.proximalFaces) {
        for (const v of state.faces[pf]) lockedSet.add(v);
      }

      const omega = [
        res.hingeAxis[0] * foldRad,
        res.hingeAxis[1] * foldRad,
        res.hingeAxis[2] * foldRad,
      ];
      for (let j = i + 1; j < stepAnchors.length; j++) {
        stepAnchors[j] = rotatePointAround(stepAnchors[j], res.pivot, omega);
      }
      curU = normalize3(rotatePointAround(curU, [0, 0, 0], omega));
      curSheetN = normalize3(rotatePointAround(curSheetN, [0, 0, 0], omega));

      if (isRiverRegion) {
        const nextActive = new Set();
        for (const df of res.distalFaces) {
          if (state.faceRegions[df] === rIdx) nextActive.add(df);
        }
        activeFaces = nextActive;
      } else {
        activeFaces = res.distalFaces;
      }
    }
  };

  applyStepsToRegion(region, false);
  if (mirrorRegion !== null && mirrorRegion !== region) {
    applyStepsToRegion(mirrorRegion, true);
  }

  return {
    foldDict: packFoldMesh(state),
    appliedRegions: targets,
    mirrorRegion,
    addedCreases: state.addedUserCreases,
  };
}
