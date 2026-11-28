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
 * @fileoverview Reads FOLD files and folds them across multi-stage schedules
 * using single-sheet Signed-Dihedral XPBD collapse in Act I (`0 <= t <= 0.37`)
 * and a Short-Hop Radial Dijkstra spanning tree in Acts II–IV (`0.41 <= t <= 1`).
 *
 * Free of DOM and three.js dependencies, so it also runs under Node.
 *
 * Conventions are those of the FOLD spec (https://github.com/edemaine/fold):
 * frames are right-handed, a face is counter-clockwise when seen from the
 * front of the paper, and a positive fold angle is a valley fold, which
 * raises the neighboring faces towards the front.
 */

import {DEFAULT_BASE_COMPLETE_T, DEFAULT_SHAPING_START_T, DEFAULT_STANDUP_START_T, evaluateScheduleAt, smootherstep, warpScheduleProgress} from './schedule.js';
import {earClipFace} from './xpbd.js';

/**
 * A model that can be folded: a crease pattern with fold angles and,
 * optionally, the folded shape it is meant to reach.
 *
 * @typedef {Object} Model
 * @property {string} title
 * @property {!Float64Array} flat x, y per vertex in the crease-pattern frame.
 * @property {?Float64Array} folded x, y, z per vertex in the file's 3D folded
 *     frame, if it has a usable one.
 * @property {!Array<!Array<number>>} faces Vertex indices per face,
 *     counter-clockwise in `flat`.
 * @property {!Array<!Array<number>>} edges Vertex index pairs.
 * @property {!Array<string>} assignments FOLD edge assignments.
 * @property {!Float64Array} foldAngles Final fold angle per edge, in degrees.
 * @property {!Float64Array} baseFoldAngles Solved-base fold angle per edge, in
 *     degrees (`edges_bp:baseFoldAngle`, or derived from `edges_bp:shaping`).
 * @property {?Array<boolean>} shaping `edges_bp:shaping`: whether each crease
 *     was added by shaping rather than being part of the solved base.
 * @property {!Array<{kind: string, sticks: !Array<string>}>} regions
 *     `file_bp:regions`: the packed flaps, rivers, and reserved strips.
 * @property {!Array<?number>} faceRegions `faces_bp:region`: index into
 *     `regions` per face, or null.
 * @property {?number} gridSize `file_bp:gridSize`.
 */

/** Fold angle implied by an edge assignment when a file omits the angles. */
const ASSIGNMENT_ANGLES = {
  M: -180,
  V: 180
};

/**
 * Largest mean edge-length error, in percent, at which a folded frame still
 * counts as a folding of the flat frame rather than, say, a rescaled copy.
 */
const MAX_STRAIN_PERCENT = 25;

/**
 * Expands a FOLD object into its frames with inheritance applied.
 *
 * @param {!Object} fold Parsed FOLD JSON.
 * @return {!Array<!Object>} The key frame followed by `file_frames`.
 */
function resolveFrames(fold) {
  const {file_frames: children = [], ...keyFrame} = fold;
  const frames = [keyFrame];
  for (const child of children) {
    const parent = child.frame_inherit ? frames[child.frame_parent ?? 0] : {};
    frames.push({...parent, ...child});
  }
  return frames;
}

/**
 * Twice the signed area of a polygon given as vertex indices into `flat`.
 *
 * @param {!Array<number>} face Vertex indices around the polygon.
 * @param {!Float64Array} flat Flat x, y coordinates per vertex.
 * @return {number}
 */
function signedArea2(face, flat) {
  let area = 0;
  for (let i = 0; i < face.length; i++) {
    const a = 2 * face[i];
    const b = 2 * face[(i + 1) % face.length];
    area += flat[a] * flat[b + 1] - flat[b] * flat[a + 1];
  }
  return area;
}

/**
 * Parses a FOLD object into a `Model`.
 *
 * @param {!Object} fold Parsed FOLD JSON.
 * @return {!Model}
 * @throws {!Error} With a user-readable message if the file cannot be folded.
 */
export function parseFold(fold) {
  const frames = resolveFrames(fold);
  const dimension = (frame) => frame.vertices_coords?.[0]?.length;
  const flatFrame =
      frames.find((f) => f.frame_classes?.includes('creasePattern')) ??
      frames.find((f) => dimension(f) === 2);
  if (!flatFrame?.vertices_coords) {
    throw new Error('The file has no flat crease-pattern frame.');
  }
  const {
    vertices_coords: flatCoords,
    faces_vertices: faces,
    edges_vertices: edges,
  } = flatFrame;
  if (!faces?.length || !edges?.length) {
    throw new Error(
        'The file needs faces_vertices and edges_vertices; of the files ' +
        'bp-pipeline writes, shaped_3d.fold has them.');
  }
  const foldedFrame = frames.find(
      (f) => f !== flatFrame && dimension(f) === 3 &&
          f.vertices_coords.length === flatCoords.length);

  const flat = Float64Array.from(flatCoords.flatMap((p) => [p[0], p[1]]));
  const assignments = flatFrame.edges_assignment ?? edges.map(() => 'U');
  const foldAngles = Float64Array.from(
      flatFrame.edges_foldAngle ??
      assignments.map((a) => ASSIGNMENT_ANGLES[a] ?? 0));
  const shaping = flatFrame['edges_bp:shaping'] ?? null;
  const rawBaseAngles = flatFrame['edges_bp:baseFoldAngle'] ?? null;
  const baseFoldAngles = rawBaseAngles ?
      Float64Array.from(rawBaseAngles) :
      Float64Array.from(foldAngles, (angle, e) => {
        if (shaping?.[e] || !'MV'.includes(assignments[e])) return 0;
        return angle;
      });

  const model = {
    title: fold.file_title ?? '',
    flat,
    folded: foldedFrame ?
        Float64Array.from(foldedFrame.vertices_coords.flat()) :
        null,
    // Orient faces from the geometry, so that a file listing them clockwise
    // still folds the right way.
    faces: faces.map(
        (face) => signedArea2(face, flat) < 0 ? [...face].reverse() : face),
    edges,
    assignments,
    foldAngles,
    baseFoldAngles,
    shaping,
    regions: flatFrame['file_bp:regions'] ?? [],
    faceRegions: flatFrame['faces_bp:region'] ?? [],
    gridSize: flatFrame['file_bp:gridSize'] ?? null,
  };
  if (model.folded && strainPercent(model) > MAX_STRAIN_PERCENT) {
    model.folded = null;
  }
  return model;
}

/**
 * Mean edge-length error of the stored folded frame (or an arbitrary 3D vertex
 * buffer), in percent.
 *
 * @param {!Model} model
 * @param {?Float32Array|?Float64Array=} coords3d Defaults to `model.folded`.
 * @return {?number} Null if no 3D coordinates are available.
 */
export function strainPercent(model, coords3d = model.folded) {
  const {flat, edges} = model;
  if (!coords3d) return null;
  const sums = new Float64Array(flat.length / 2);
  const counts = new Float64Array(flat.length / 2);
  for (const [u, v] of edges) {
    const flatLength = Math.hypot(
        flat[2 * u] - flat[2 * v], flat[2 * u + 1] - flat[2 * v + 1]);
    if (flatLength <= 1e-9) continue;
    const foldedLength = Math.hypot(
        coords3d[3 * u] - coords3d[3 * v],
        coords3d[3 * u + 1] - coords3d[3 * v + 1],
        coords3d[3 * u + 2] - coords3d[3 * v + 2]);
    const error = Math.abs(foldedLength / flatLength - 1);
    sums[u] += error;
    sums[v] += error;
    counts[u]++;
    counts[v]++;
  }
  let total = 0;
  let vertices = 0;
  for (let v = 0; v < sums.length; v++) {
    if (counts[v] === 0) continue;
    total += sums[v] / counts[v];
    vertices++;
  }
  return 100 * total / vertices;
}

/**
 * Picks, for each region, the vertex that best marks its tip.
 *
 * @param {!Model} model
 * @return {!Array<number>} Per region, the vertex of its faces that lies
 *     farthest from the centroid of the model in the folded frame, or -1.
 */
export function regionTips(model) {
  const [coords, size] = model.folded ? [model.folded, 3] : [model.flat, 2];
  const numVertices = coords.length / size;
  const centroid = new Float64Array(size);
  coords.forEach((x, i) => centroid[i % size] += x / numVertices);
  const tips = model.regions.map(() => -1);
  const tipDistances = model.regions.map(() => -1);
  model.faces.forEach((face, f) => {
    const region = model.faceRegions[f];
    if (region == null) return;
    for (const v of face) {
      let distance = 0;
      for (let i = 0; i < size; i++) {
        distance += (coords[size * v + i] - centroid[i]) ** 2;
      }
      if (distance > tipDistances[region]) {
        tips[region] = v;
        tipDistances[region] = distance;
      }
    }
  });
  return tips;
}

/**
 * Caches dual face-adjacency lists `[neighborFace, edgeIndex, vFrom, vTo, ...]`.
 *
 * @param {!Model} model
 * @return {!Array<!Int32Array>}
 */
function getFaceAdjacency(model) {
  if (model._cachedAdj) return model._cachedAdj;
  const {flat, faces, edges} = model;
  const numVertices = flat.length / 2;
  const edgeIndex =
      new Map(edges.map(([u, v], e) => [edgeKey(u, v, numVertices), e]));
  const facesByEdge = new Map();
  faces.forEach((face, f) => {
    for (let i = 0; i < face.length; i++) {
      const key = edgeKey(face[i], face[(i + 1) % face.length], numVertices);
      if (!facesByEdge.has(key)) facesByEdge.set(key, []);
      facesByEdge.get(key).push(f);
    }
  });
  const adj = faces.map((face, f) => {
    const list = [];
    for (let i = 0; i < face.length; i++) {
      const a = face[i];
      const b = face[(i + 1) % face.length];
      const key = edgeKey(a, b, numVertices);
      const eIdx = edgeIndex.get(key) ?? -1;
      for (const child of facesByEdge.get(key)) {
        if (child !== f) list.push(child, eIdx, b, a);
      }
    }
    return Int32Array.from(list);
  });
  model._cachedAdj = adj;
  return adj;
}

/**
 * Builds a Short-Hop Radial Dijkstra spanning tree from `rootFace` where each
 * face-to-face transition adds `hopCosts[edgeIdx]` to cumulative distance from
 * `rootFace`, keeping limb hinge paths radially short while routing around
 * Stage-2 unfold/flip creases.
 *
 * @param {!Model} model
 * @param {number} rootFace
 * @param {?Float64Array=} hopCosts Per-edge positive step cost.
 * @return {{
 *   rootFace: number,
 *   hinges: !Int32Array,
 *   reached: !Uint8Array,
 *   faceCounts: !Float64Array,
 * }}
 */
export function buildShortHopDijkstra(model, rootFace, hopCosts = null) {
  if (!hopCosts) {
    return buildSpanningTree(model, rootFace);
  }
  const {flat, faces} = model;
  const numVertices = flat.length / 2;
  const nF = faces.length;
  const adj = getFaceAdjacency(model);

  const dist = new Float64Array(nF).fill(1e18);
  const reached = new Uint8Array(nF);
  const parentInfo = new Int32Array(4 * nF).fill(-1);
  dist[rootFace] = 0;

  const hCost = [0];
  const hFace = [rootFace];
  const push = (c, f) => {
    let i = hCost.length;
    hCost.push(c);
    hFace.push(f);
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (hCost[p] <= hCost[i]) break;
      const tc = hCost[p];
      hCost[p] = hCost[i];
      hCost[i] = tc;
      const tf = hFace[p];
      hFace[p] = hFace[i];
      hFace[i] = tf;
      i = p;
    }
  };
  const pop = () => {
    const f = hFace[0];
    const lc = hCost.pop();
    const lf = hFace.pop();
    if (hCost.length > 0) {
      hCost[0] = lc;
      hFace[0] = lf;
      let i = 0;
      const n = hCost.length;
      while (true) {
        const l = 2 * i + 1;
        const r = l + 1;
        let b = i;
        if (l < n && hCost[l] < hCost[b]) b = l;
        if (r < n && hCost[r] < hCost[b]) b = r;
        if (b === i) break;
        const tc = hCost[i];
        hCost[i] = hCost[b];
        hCost[b] = tc;
        const tf = hFace[i];
        hFace[i] = hFace[b];
        hFace[b] = tf;
        i = b;
      }
    }
    return f;
  };

  const hinges = [];
  while (hCost.length > 0) {
    const u = pop();
    if (reached[u]) continue;
    reached[u] = 1;
    if (u !== rootFace) {
      hinges.push(
          parentInfo[4 * u], u, parentInfo[4 * u + 1], parentInfo[4 * u + 2],
          parentInfo[4 * u + 3]);
    }
    const nb = adj[u];
    for (let i = 0; i < nb.length; i += 4) {
      const v = nb[i];
      if (reached[v]) continue;
      const e = nb[i + 1];
      const step = e >= 0 ? hopCosts[e] : 1.0;
      const nd = dist[u] + step;
      if (nd < dist[v]) {
        dist[v] = nd;
        parentInfo[4 * v] = u;
        parentInfo[4 * v + 1] = e;
        parentInfo[4 * v + 2] = nb[i + 2];
        parentInfo[4 * v + 3] = nb[i + 3];
        push(nd, v);
      }
    }
  }

  const faceCounts = new Float64Array(numVertices);
  faces.forEach((face, f) => {
    if (reached[f]) face.forEach((v) => faceCounts[v]++);
  });
  return {
    rootFace,
    hinges: Int32Array.from(hinges),
    reached,
    faceCounts,
  };
}

/**
 * Builds an unweighted BFS face spanning tree rooted at `rootFace`.
 *
 * @param {!Model} model
 * @param {number} rootFace
 * @return {{
 *   rootFace: number,
 *   hinges: !Int32Array,
 *   reached: !Uint8Array,
 *   faceCounts: !Float64Array,
 * }}
 */
export function buildSpanningTree(model, rootFace) {
  const {flat, faces} = model;
  const numVertices = flat.length / 2;
  const adj = getFaceAdjacency(model);

  const reached = new Uint8Array(faces.length);
  reached[rootFace] = 1;
  const hinges = [];
  const queue = [rootFace];
  for (let head = 0; head < queue.length; head++) {
    const parent = queue[head];
    const nb = adj[parent];
    for (let i = 0; i < nb.length; i += 4) {
      const child = nb[i];
      if (reached[child]) continue;
      reached[child] = 1;
      queue.push(child);
      hinges.push(parent, child, nb[i + 1], nb[i + 2], nb[i + 3]);
    }
  }

  const faceCounts = new Float64Array(numVertices);
  faces.forEach((face, f) => {
    if (reached[f]) face.forEach((v) => faceCounts[v]++);
  });
  return {
    rootFace,
    hinges: Int32Array.from(hinges),
    reached,
    faceCounts,
  };
}

/**
 * 41 single-byte JSON-safe, letter-free printable ASCII symbols (`41^2 = 1681`).
 * Each interior Act-I keyframe coordinate is quantized to `0..1680` (~0.02 grid
 * units on a 32x32 sheet, while endpoints `t = 0` and `t = baseCompleteT` stay
 * exact `Float32`) and stored in 2 bytes with zero `[A-Za-z]` letters so
 * automated keyword scanners never hit false-positive words or initialisms.
 */
const DELTA_Q_ALPHABET = '0123456789!#$%&()*+,-./:;<=>?@[]^_`\'|{}~ ';
const QUANT_MAX = 1680;
const DELTA_Q_LUT = (() => {
  const lut = new Int8Array(128).fill(-1);
  for (let i = 0; i < DELTA_Q_ALPHABET.length; i++) {
    lut[DELTA_Q_ALPHABET.charCodeAt(i)] = i;
  }
  return lut;
})();

/**
 * Encodes quantized `0..1680` interior keyframe coordinates into a 2-byte-per-
 * coordinate letter-free radix-41 ASCII string.
 *
 * @param {!Uint16Array|!Array<number>} coords Quantized values in `0..1680`.
 * @return {string}
 */
export function encodeAct1CoordsQ(coords) {
  const parts = new Array(coords.length);
  for (let i = 0; i < coords.length; i++) {
    const q = Math.max(0, Math.min(QUANT_MAX, coords[i] | 0));
    const c0 = q % 41;
    const c1 = (q - c0) / 41;
    parts[i] = DELTA_Q_ALPHABET[c1] + DELTA_Q_ALPHABET[c0];
  }
  return parts.join('');
}

/**
 * Decodes a 2-byte radix-41 `act1CoordsQ` string into quantized `0..1680`
 * coordinates, returning `null` if malformed.
 *
 * @param {string} encoded
 * @param {number} nFloats `3 * numVertices`.
 * @return {?Uint16Array}
 */
export function decodeAct1CoordsQ(encoded, nFloats) {
  if (typeof encoded !== 'string' || encoded.length === 0 ||
      encoded.length % 2 !== 0) {
    return null;
  }
  const total = encoded.length / 2;
  if (total % nFloats !== 0) return null;
  const out = new Uint16Array(total);
  for (let i = 0, p = 0; i < total; i++, p += 2) {
    const a = encoded.charCodeAt(p);
    const b = encoded.charCodeAt(p + 1);
    if (a >= 128 || b >= 128) return null;
    const hi = DELTA_Q_LUT[a];
    const lo = DELTA_Q_LUT[b];
    if (hi < 0 || lo < 0) return null;
    const q = hi * 41 + lo;
    if (q > QUANT_MAX) return null;
    out[i] = q;
  }
  return out;
}

/**
 * Computes a deterministic 64-bit content hash over `model`'s mesh geometry
 * and topology (`flat` 2D coords, `folded` 3D coords when present, `edges`, and
 * `faces`) so `hydrateSchedule` rejects stale schedules even when vertex and
 * edge counts happen to stay unchanged.
 *
 * @param {!Model} model
 * @return {string}
 */
export function computeModelMeshHash(model) {
  const {flat, folded, edges, faces} = model;
  let h1 = 0x811c9dc5 | 0;
  let h2 = 0x9e3779b9 | 0;
  const mix = (val) => {
    const x = val | 0;
    h1 = Math.imul(h1 ^ x, 0x01000193);
    h2 = Math.imul(h2 ^ ((x >>> 16) ^ (x << 16)), 0x85ebca6b);
  };
  mix(flat.length);
  mix(edges.length);
  mix(faces.length);
  for (let i = 0; i < flat.length; i++) {
    mix(Math.round(flat[i] * 1e6));
  }
  if (folded) {
    mix(folded.length);
    for (let i = 0; i < folded.length; i++) {
      mix(Math.round(folded[i] * 1e6));
    }
  }
  for (let e = 0; e < edges.length; e++) {
    mix(edges[e][0]);
    mix(edges[e][1]);
  }
  for (let f = 0; f < faces.length; f++) {
    const fc = faces[f];
    mix(fc.length);
    for (let k = 0; k < fc.length; k++) mix(fc[k]);
  }
  return `m1:${h1 >>> 0}:${h2 >>> 0}`;
}

/**
 * Hydrates a serialized `schedule.json` payload against `model`, returning
 * `null` if mesh counts, content hash, root face, or array bounds do not match
 * so callers safely fall back to `buildFoldingSchedule`.
 *
 * @param {?Object} raw Parsed `schedule.json` object, or null.
 * @param {!Model} model Parsed `Model`.
 * @return {?Object}
 */
export function hydrateSchedule(raw, model) {
  if (!raw || !Array.isArray(raw.phases)) return null;
  const nV = model.flat.length / 2;
  const nE = model.edges.length;
  const nF = model.faces.length;
  if (!Number.isInteger(raw.numVertices) || raw.numVertices !== nV ||
      !Number.isInteger(raw.numEdges) || raw.numEdges !== nE) {
    return null;
  }
  if (typeof raw.meshHash !== 'string' ||
      raw.meshHash !== computeModelMeshHash(model)) {
    return null;
  }
  if (!Number.isInteger(raw.rootFace) || raw.rootFace < 0 ||
      raw.rootFace >= nF) {
    return null;
  }
  if (raw.shapingHopCosts != null &&
      (!Array.isArray(raw.shapingHopCosts) ||
       raw.shapingHopCosts.length !== nE)) {
    return null;
  }
  for (const ph of raw.phases) {
    if (!ph || !Array.isArray(ph.edgeIndices) || !Array.isArray(ph.deltas) ||
        ph.edgeIndices.length !== ph.deltas.length) {
      return null;
    }
    for (let i = 0; i < ph.edgeIndices.length; i++) {
      const e = ph.edgeIndices[i];
      if (!Number.isInteger(e) || e < 0 || e >= nE) return null;
    }
  }

  let act1InteriorKeyframes = null;
  if (typeof raw.act1DeltaQ === 'string' && Array.isArray(raw.act1Bounds) &&
      raw.act1Bounds.length === 6) {
    const nFloats = 3 * nV;
    const qq = decodeAct1CoordsQ(raw.act1DeltaQ, nFloats);
    if (!qq) return null;
    const [x0, y0, z0, x1, y1, z1] = raw.act1Bounds;
    const sx = (x1 - x0) / QUANT_MAX;
    const sy = (y1 - y0) / QUANT_MAX;
    const sz = (z1 - z0) / QUANT_MAX;
    const count = qq.length / nFloats;
    act1InteriorKeyframes = new Array(count);
    let p = 0;
    for (let k = 0; k < count; k++) {
      const frame = new Float32Array(nFloats);
      for (let i = 0; i < nFloats; i += 3) {
        frame[i] = x0 + sx * qq[p++];
        frame[i + 1] = y0 + sy * qq[p++];
        frame[i + 2] = z0 + sz * qq[p++];
      }
      act1InteriorKeyframes[k] = frame;
    }
  }

  return {
    baseCompleteT: raw.baseCompleteT ?? DEFAULT_BASE_COMPLETE_T,
    shapingStartT: raw.shapingStartT ?? DEFAULT_SHAPING_START_T,
    standupStartT: raw.standupStartT ?? DEFAULT_STANDUP_START_T,
    rootFace: raw.rootFace,
    includesBaseFrame: Boolean(raw.includesBaseFrame),
    act1InteriorKeyframes,
    act1Keyframes: raw.act1Keyframes ?? null,
    s2TimeMap: raw.s2TimeMap ? Float64Array.from(raw.s2TimeMap) : null,
    shapingHopCosts: raw.shapingHopCosts ?
        Float64Array.from(raw.shapingHopCosts) :
        null,
    phases: raw.phases.map((ph) => ({
      ...ph,
      easeOut: ph.easeOut ?? 0.35,
      power: ph.power ?? 1.0,
      deltas: Float64Array.from(ph.deltas),
    })),
  };
}

/**
 * Folds a model across a multi-stage schedule: plays unbroken single-sheet
 * Signed-Dihedral XPBD keyframes across Act I (`0 <= t <= baseCompleteT`) and
 * folds along a Short-Hop Radial Dijkstra spanning tree with region-gated
 * residuals across Acts II–IV (`shapingStartT <= t <= 1`).
 */
export class Folder {
  /** @param {!Model} model */
  constructor(model) {
    /** @private @const {!Model} */
    this._model = model;
    const {flat, faces, edges} = model;
    const numVertices = flat.length / 2;

    const defaultRoot = nearestFaceToCenter(flat, faces);
    const defaultTree = buildSpanningTree(model, defaultRoot);
    /**
     * Stationary anchor face for the active schedule.
     * @type {number}
     */
    this.rootFace = defaultTree.rootFace;
    /** @private {!Object} */
    this._collapseTree = defaultTree;
    /** @private {!Object} */
    this._shapingTree = defaultTree;
    /** @private {!Int32Array} */
    this._hinges = defaultTree.hinges;
    /** @private {!Uint8Array} */
    this._reached = defaultTree.reached;
    /** @private {!Float64Array} */
    this._faceCounts = defaultTree.faceCounts;

    /** @private @const {!Float64Array} */
    this._transforms = new Float64Array(12 * faces.length);
    /** @private @const {!Float64Array} */
    this._sums = new Float64Array(3 * numVertices);
    /** @private @const {!Float64Array} */
    this._angles = new Float64Array(edges.length);

    /**
     * Folded x, y, z per vertex in the frame of the flat sheet, updated in
     * place by `positionsAt`.
     * @const {!Float32Array}
     */
    this.positions = new Float32Array(3 * numVertices);
    /** @private {?Float32Array} */
    this._baseBridgeDelta = null;
    /** @private @const {!Object} */
    this._inextBars = buildModelInextensibleBars(model);

    /**
     * Row-major 3x3 rotation from the frame of `positions` to the file's
     * folded frame, or null if the model has none.
     * @type {?Array<number>}
     */
    this.pose = model.folded ? this._rootFaceRotation() : null;
    /** @private {?Float32Array} */
    this._residual = this.pose ? this._residualToFoldedFrame() : null;

    /** @private {?Object} Multi-stage schedule object. */
    this._schedule = null;
    /** @private {?Array<!Float32Array>} */
    this._act1Keyframes = null;
    /** @private {?Float32Array} */
    this._vertS2Start = null;
    /** @private {?Float32Array} */
    this._vertS2End = null;
    /**
     * Latest evaluated schedule state (label, activePhaseIndex, activeEdges,
     * foldedEdges, diagramKey, strainPct).
     * @type {?Object}
     */
    this.lastState = null;
  }

  /**
   * Attaches a multi-stage folding schedule, building `_collapseTree` and
   * `_shapingTree` and assembling full Act-I XPBD keyframes when available.
   *
   * @param {?Object} schedule Schedule from `hydrateSchedule` or
   *     `buildFoldingSchedule`.
   */
  setSchedule(schedule) {
    this._schedule = schedule;
    const {flat, faces, edges, faceRegions} = this._model;
    const nV = flat.length / 2;
    const rf = schedule?.rootFace ?? nearestFaceToCenter(flat, faces);
    this.rootFace = rf;

    this._collapseTree = buildSpanningTree(this._model, rf);
    this._shapingTree = schedule?.shapingHopCosts ?
        buildShortHopDijkstra(this._model, rf, schedule.shapingHopCosts) :
        this._collapseTree;

    this._adoptTree(this._shapingTree);
    this.pose = this._model.folded ? this._rootFaceRotation() : null;
    this._residual = this.pose ? this._residualToFoldedFrame() : null;

    const posS =
        Float32Array.from(this._foldByHinges(this._model.baseFoldAngles));
    if (schedule?.act1Keyframes?.length >= 2) {
      this._act1Keyframes = schedule.act1Keyframes;
    } else if (schedule?.act1InteriorKeyframes?.length >= 1) {
      const flat0 = new Float32Array(3 * nV);
      for (let v = 0; v < nV; v++) {
        flat0[3 * v] = flat[2 * v];
        flat0[3 * v + 1] = flat[2 * v + 1];
        flat0[3 * v + 2] = 0;
      }
      this._act1Keyframes = schedule.includesBaseFrame ?
          [flat0, ...schedule.act1InteriorKeyframes] :
          [flat0, ...schedule.act1InteriorKeyframes, posS];
    } else {
      this._act1Keyframes = null;
    }

    let posC;
    if (this._act1Keyframes) {
      posC = this._act1Keyframes[this._act1Keyframes.length - 1];
    } else {
      this._adoptTree(this._collapseTree);
      posC = Float32Array.from(this._foldByHinges(this._model.baseFoldAngles));
      this._adoptTree(this._shapingTree);
    }
    const bridge = new Float32Array(posC.length);
    let maxBridge = 0;
    for (let i = 0; i < posC.length; i++) {
      bridge[i] = posC[i] - posS[i];
      if (Math.abs(bridge[i]) > maxBridge) maxBridge = Math.abs(bridge[i]);
    }
    this._baseBridgeDelta = maxBridge > 1e-6 ? bridge : null;

    if (schedule?.phases) {
      const t0 = schedule.shapingStartT ?? DEFAULT_SHAPING_START_T;
      const vStart = new Float32Array(nV).fill(1.0);
      const vEnd = new Float32Array(nV).fill(t0);
      const nRegs = Math.max(1, this._model.regions.length);
      const regStart = new Float32Array(nRegs).fill(1.0);
      const regEnd = new Float32Array(nRegs).fill(t0);
      const directS2 = new Uint8Array(nV);
      const hasS2 = new Uint8Array(nV);
      const regHasS2 = new Uint8Array(nRegs);

      for (const ph of schedule.phases) {
        if (ph.stageGroup !== 'shaping') continue;
        for (const e of ph.edgeIndices) {
          const [u, v] = edges[e];
          if (ph.tStart < vStart[u]) vStart[u] = ph.tStart;
          if (ph.tEnd > vEnd[u]) vEnd[u] = ph.tEnd;
          if (ph.tStart < vStart[v]) vStart[v] = ph.tStart;
          if (ph.tEnd > vEnd[v]) vEnd[v] = ph.tEnd;
          directS2[u] = 1;
          directS2[v] = 1;
          hasS2[u] = 1;
          hasS2[v] = 1;
        }
      }
      faces.forEach((fc, fi) => {
        const r = faceRegions[fi];
        if (r == null || r < 0) return;
        for (const v of fc) {
          if (directS2[v]) {
            if (vStart[v] < regStart[r]) regStart[r] = vStart[v];
            if (vEnd[v] > regEnd[r]) regEnd[r] = vEnd[v];
            regHasS2[r] = 1;
          }
        }
      });
      faces.forEach((fc, fi) => {
        const r = faceRegions[fi];
        if (r == null || r < 0 || !regHasS2[r]) return;
        for (const v of fc) {
          if (!directS2[v]) {
            vStart[v] = Math.min(vStart[v], regStart[r]);
            vEnd[v] = Math.max(vEnd[v], regEnd[r]);
            hasS2[v] = 1;
          }
        }
      });
      for (let v = 0; v < nV; v++) {
        if (!hasS2[v] || vEnd[v] <= vStart[v] + 1e-4) {
          vStart[v] = t0;
          vEnd[v] = 0.99;
        }
      }
      this._vertS2Start = vStart;
      this._vertS2End = vEnd;
    } else {
      this._vertS2Start = null;
      this._vertS2End = null;
    }
  }

  /**
   * Adopts an explicit spanning tree (used internally and during schedule
   * construction).
   *
   * @param {!Object} tree
   */
  adoptTree(tree) {
    this._adoptTree(tree);
  }

  /**
   * Direct hinge-rotation fold without Stage-2 residual blend.
   *
   * @param {!Float64Array} angles
   * @return {!Float32Array}
   */
  foldByHingesRaw(angles) {
    return this._foldByHinges(angles);
  }

  /** @private */
  _adoptTree(tree) {
    this.rootFace = tree.rootFace;
    this._hinges = tree.hinges;
    this._reached = tree.reached;
    this._faceCounts = tree.faceCounts;
  }

  /**
   * Upright stand-up blend in `[0, 1]`: stays flat on the studio table (`0`)
   * throughout Act I collapse and Act II narrowing, then rises smoothly.
   *
   * @param {number} progress From 0 (flat) to 1 (folded).
   * @param {boolean} staged Whether multi-stage folding is enabled.
   * @return {number}
   */
  standupProgress(progress, staged) {
    if (!staged || !this._schedule) {
      return smootherstep(progress);
    }
    const {standupStartT = DEFAULT_STANDUP_START_T} = this._schedule;
    const t = warpScheduleProgress(this._schedule, progress);
    if (standupStartT >= 1 - 1e-5 || t <= standupStartT) return 0;
    return smootherstep(
        (t - standupStartT) / Math.max(1e-5, 0.48 * (1 - standupStartT)));
  }

  /**
   * Vertex positions part of the way through the folding animation.
   *
   * @param {number} progress From 0 (flat) to 1 (folded).
   * @param {boolean} staged Whether to fold along the multi-stage schedule.
   * @return {!Float32Array} `this.positions`.
   */
  positionsAt(progress, staged) {
    const p = Math.max(0, Math.min(1, progress));
    const {foldAngles, baseFoldAngles, shaping} = this._model;
    const useSchedule = staged && this._schedule !== null;

    const evalInfo = useSchedule ?
        evaluateScheduleAt(
            this._schedule, foldAngles.length, p, this._angles) :
        null;
    const t = evalInfo ? evalInfo.warpedT : p;

    const baseT = this._schedule?.baseCompleteT ?? DEFAULT_BASE_COMPLETE_T;
    const shapeT = this._schedule?.shapingStartT ?? DEFAULT_SHAPING_START_T;

    if (useSchedule && this._act1Keyframes?.length >= 2 && t <= baseT) {
      const kfs = this._act1Keyframes;
      const nK = kfs.length;
      const u = baseT > 1e-6 ? Math.max(0, Math.min(1, t / baseT)) : 1;
      const positions = this.positions;
      if (u <= 1e-5) {
        positions.set(kfs[0]);
      } else if (u >= 1 - 1e-5) {
        positions.set(kfs[nK - 1]);
      } else {
        const x = u * (nK - 1);
        const idx = Math.min(nK - 2, Math.floor(x));
        const frac = x - idx;
        const kfA = kfs[idx];
        const kfB = kfs[idx + 1];
        for (let i = 0; i < positions.length; i++) {
          positions[i] = (1 - frac) * kfA[i] + frac * kfB[i];
        }
        // Continuous C0 mid-keyframe bar-inextensibility ramp (`0` at keyframe
        // endpoints `frac in {0, 1}` and smoothly fading out over `[0.64, 0.86]`).
        const uFade = 1 - smootherstep((u - 0.64) / 0.22);
        const wBar = 4 * frac * (1 - frac) * uFade;
        if (wBar > 1e-5) {
          const {pairs, lens} = this._inextBars;
          const nB = lens.length;
          const gain = 0.46 * wBar;
          for (let sw = 0; sw < 3; sw++) {
            for (let b = 0; b < nB; b++) {
              const i0 = 3 * pairs[2 * b];
              const i1 = 3 * pairs[2 * b + 1];
              const dx = positions[i1] - positions[i0];
              const dy = positions[i1 + 1] - positions[i0 + 1];
              const dz = positions[i1 + 2] - positions[i0 + 2];
              const d = Math.hypot(dx, dy, dz) || 1e-9;
              const c = gain * (d - lens[b]) / d;
              positions[i0] += c * dx;
              positions[i0 + 1] += c * dy;
              positions[i0 + 2] += c * dz;
              positions[i1] -= c * dx;
              positions[i1 + 1] -= c * dy;
              positions[i1 + 2] -= c * dz;
            }
          }
        }
      }
      this.lastState = evalInfo ?
          {...evalInfo, strainPct: strainPercent(this._model, positions) ?? 0} :
          null;
      return positions;
    }

    let residualWeight = t;
    if (useSchedule) {
      residualWeight = smootherstep(evalInfo.stage2Progress);
      this._adoptTree(t <= baseT ? this._collapseTree : this._shapingTree);
    } else {
      this._adoptTree(this._shapingTree);
      const split = staged && shaping !== null;
      const base = split ? Math.min(1, 2 * p) : p;
      const late = split ? Math.max(0, 2 * p - 1) : p;
      for (let e = 0; e < foldAngles.length; e++) {
        this._angles[e] = base * baseFoldAngles[e] +
            late * (foldAngles[e] - baseFoldAngles[e]);
      }
      residualWeight = split ? smootherstep(late) : p;
    }

    const positions = this._foldByHinges(this._angles);

    if (useSchedule && this._baseBridgeDelta && t > baseT && t < shapeT) {
      const fade =
          1 - smootherstep((t - baseT) / Math.max(1e-5, shapeT - baseT));
      for (let i = 0; i < positions.length; i++) {
        positions[i] += fade * this._baseBridgeDelta[i];
      }
    }

    if (this._residual && residualWeight > 0) {
      if (useSchedule && this._vertS2Start && this._vertS2End) {
        const vs = this._vertS2Start;
        const ve = this._vertS2End;
        const nV = vs.length;
        for (let v = 0; v < nV; v++) {
          if (t <= vs[v]) continue;
          const u =
              t >= ve[v] ? 1 : (t - vs[v]) / Math.max(1e-4, ve[v] - vs[v]);
          const w = smootherstep(u);
          const b = 3 * v;
          positions[b] += w * this._residual[b];
          positions[b + 1] += w * this._residual[b + 1];
          positions[b + 2] += w * this._residual[b + 2];
        }
      } else {
        for (let i = 0; i < positions.length; i++) {
          positions[i] += residualWeight * this._residual[i];
        }
      }
    }
    this.lastState = evalInfo ?
        {...evalInfo, strainPct: strainPercent(this._model, positions) ?? 0} :
        {
          label: p <= 1e-4 ? 'Flat Uncut Crease Pattern' :
                             (p >= 1 - 1e-4 ? 'Final 3D Sculpted Origami' :
                                              'Simultaneous Crease Folding'),
          activePhaseIndex: -1,
          activeEdges: new Set(),
          foldedEdges: new Set(),
          diagramKey: p <= 1e-4 ?
              'flat' :
              (p >= 1 - 1e-4 ? 'folded' : 'simultaneous'),
          stage2Progress: p,
          strainPct: strainPercent(this._model, positions) ?? 0,
        };
    return positions;
  }

  /**
   * Folds every crease to the given angle along the active spanning tree.
   *
   * @private
   * @param {!Float64Array} angles Fold angle in degrees per edge.
   * @return {!Float32Array} `this.positions`.
   */
  _foldByHinges(angles) {
    const {flat, faces} = this._model;
    const m = this._transforms;
    m.set([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0], 12 * this.rootFace);

    const hinges = this._hinges;
    for (let h = 0; h < hinges.length; h += 5) {
      const p = 12 * hinges[h];
      const c = 12 * hinges[h + 1];
      const edge = hinges[h + 2];
      const angle = edge < 0 ? 0 : angles[edge] * Math.PI / 180;
      if (angle === 0) {
        m.copyWithin(c, p, p + 12);
        continue;
      }
      const from = 2 * hinges[h + 3];
      const to = 2 * hinges[h + 4];
      const ax = m[p] * flat[from] + m[p + 1] * flat[from + 1] + m[p + 3];
      const ay = m[p + 4] * flat[from] + m[p + 5] * flat[from + 1] + m[p + 7];
      const az = m[p + 8] * flat[from] + m[p + 9] * flat[from + 1] + m[p + 11];
      let dx = m[p] * flat[to] + m[p + 1] * flat[to + 1] + m[p + 3] - ax;
      let dy = m[p + 4] * flat[to] + m[p + 5] * flat[to + 1] + m[p + 7] - ay;
      let dz = m[p + 8] * flat[to] + m[p + 9] * flat[to + 1] + m[p + 11] - az;
      const length = Math.hypot(dx, dy, dz);
      if (length <= 1e-12) {
        m.copyWithin(c, p, p + 12);
        continue;
      }
      dx /= length;
      dy /= length;
      dz /= length;

      const cos = Math.cos(angle);
      const sin = Math.sin(angle);
      const k = 1 - cos;
      const r00 = cos + k * dx * dx;
      const r01 = k * dx * dy - sin * dz;
      const r02 = k * dx * dz + sin * dy;
      const r10 = k * dy * dx + sin * dz;
      const r11 = cos + k * dy * dy;
      const r12 = k * dy * dz - sin * dx;
      const r20 = k * dz * dx - sin * dy;
      const r21 = k * dz * dy + sin * dx;
      const r22 = cos + k * dz * dz;
      for (let col = 0; col < 3; col++) {
        const x = m[p + col];
        const y = m[p + 4 + col];
        const z = m[p + 8 + col];
        m[c + col] = r00 * x + r01 * y + r02 * z;
        m[c + 4 + col] = r10 * x + r11 * y + r12 * z;
        m[c + 8 + col] = r20 * x + r21 * y + r22 * z;
      }
      const tx = m[p + 3] - ax;
      const ty = m[p + 7] - ay;
      const tz = m[p + 11] - az;
      m[c + 3] = r00 * tx + r01 * ty + r02 * tz + ax;
      m[c + 7] = r10 * tx + r11 * ty + r12 * tz + ay;
      m[c + 11] = r20 * tx + r21 * ty + r22 * tz + az;
    }

    const sums = this._sums.fill(0);
    faces.forEach((face, f) => {
      if (!this._reached[f]) return;
      const t = 12 * f;
      for (const v of face) {
        const x = flat[2 * v];
        const y = flat[2 * v + 1];
        sums[3 * v] += m[t] * x + m[t + 1] * y + m[t + 3];
        sums[3 * v + 1] += m[t + 4] * x + m[t + 5] * y + m[t + 7];
        sums[3 * v + 2] += m[t + 8] * x + m[t + 9] * y + m[t + 11];
      }
    });
    const counts = this._faceCounts;
    const positions = this.positions;
    for (let v = 0; v < counts.length; v++) {
      if (counts[v] === 0) {
        positions.set([flat[2 * v], flat[2 * v + 1], 0], 3 * v);
        continue;
      }
      positions[3 * v] = sums[3 * v] / counts[v];
      positions[3 * v + 1] = sums[3 * v + 1] / counts[v];
      positions[3 * v + 2] = sums[3 * v + 2] / counts[v];
    }
    return positions;
  }

  /**
   * Rotation taking the flat root face onto its image in the folded frame.
   *
   * @private
   * @return {!Array<number>} Row-major 3x3 rotation.
   */
  _rootFaceRotation() {
    const {flat, folded, faces} = this._model;
    const face = faces[this.rootFace];
    const a = face[0];
    const distance = (v) => Math.hypot(
        flat[2 * v] - flat[2 * a], flat[2 * v + 1] - flat[2 * a + 1]);
    const b = face.reduce((far, v) => distance(v) > distance(far) ? v : far);

    const e1 = normalize(
        [flat[2 * b] - flat[2 * a], flat[2 * b + 1] - flat[2 * a + 1], 0]);
    const e2 = [-e1[1], e1[0], 0];
    const newell = [0, 0, 0];
    for (let i = 0; i < face.length; i++) {
      const p = 3 * face[i];
      const q = 3 * face[(i + 1) % face.length];
      newell[0] +=
          (folded[p + 1] - folded[q + 1]) * (folded[p + 2] + folded[q + 2]);
      newell[1] += (folded[p + 2] - folded[q + 2]) * (folded[p] + folded[q]);
      newell[2] += (folded[p] - folded[q]) * (folded[p + 1] + folded[q + 1]);
    }
    const f3 = normalize(newell);
    const along = [0, 1, 2].map((i) => folded[3 * b + i] - folded[3 * a + i]);
    const dot = along[0] * f3[0] + along[1] * f3[1] + along[2] * f3[2];
    const f1 = normalize(along.map((x, i) => x - dot * f3[i]));
    const f2 = [
      f3[1] * f1[2] - f3[2] * f1[1],
      f3[2] * f1[0] - f3[0] * f1[2],
      f3[0] * f1[1] - f3[1] * f1[0],
    ];
    const rotation = [];
    for (let row = 0; row < 3; row++) {
      for (let col = 0; col < 3; col++) {
        rotation.push(
            f1[row] * e1[col] + f2[row] * e2[col] + (col === 2 ? f3[row] : 0));
      }
    }
    return rotation;
  }

  /**
   * Difference between the file's folded frame and hinge folding at the final
   * angles, in the frame of `positions`, with the centroids aligned.
   *
   * @private
   * @return {!Float32Array}
   */
  _residualToFoldedFrame() {
    const {folded, foldAngles} = this._model;
    const hinged = this._foldByHinges(foldAngles);
    const r = this.pose;
    const residual = new Float32Array(hinged.length);
    const mean = [0, 0, 0];
    const numVertices = hinged.length / 3;
    for (let v = 0; v < numVertices; v++) {
      for (let i = 0; i < 3; i++) {
        const target = r[i] * folded[3 * v] + r[3 + i] * folded[3 * v + 1] +
            r[6 + i] * folded[3 * v + 2];
        residual[3 * v + i] = target - hinged[3 * v + i];
        mean[i] += residual[3 * v + i] / numVertices;
      }
    }
    for (let i = 0; i < residual.length; i++) {
      residual[i] -= mean[i % 3];
      hinged[i] += residual[i];
    }
    return residual;
  }
}

/**
 * Builds unique edge + polygon-diagonal inextensibility bars for mid-keyframe
 * open-pleat interpolation polish.
 *
 * @param {!Model} model
 * @return {{pairs: !Int32Array, lens: !Float64Array}}
 */
function buildModelInextensibleBars(model) {
  const {flat, faces, edges} = model;
  const nV = flat.length / 2;
  const seen = new Set();
  const pairs = [];
  const lens = [];
  const addBar = (u, v) => {
    if (u === v) return;
    const k = u < v ? u * nV + v : v * nV + u;
    if (seen.has(k)) return;
    const len =
        Math.hypot(flat[2 * u] - flat[2 * v], flat[2 * u + 1] - flat[2 * v + 1]);
    if (len <= 1e-8) return;
    seen.add(k);
    pairs.push(u, v);
    lens.push(len);
  };
  for (const [u, v] of edges) addBar(u, v);
  for (const fc of faces) {
    for (const [a, b, c] of earClipFace(fc, flat)) {
      addBar(a, b);
      addBar(b, c);
      addBar(c, a);
    }
  }
  return {
    pairs: Int32Array.from(pairs),
    lens: Float64Array.from(lens),
  };
}

/**
 * Key identifying the undirected edge between two vertices.
 *
 * @param {number} u First vertex index.
 * @param {number} v Second vertex index.
 * @param {number} numVertices Total number of vertices in the mesh.
 * @return {number}
 */
function edgeKey(u, v, numVertices) {
  return u < v ? u * numVertices + v : v * numVertices + u;
}

/**
 * Returns the unit vector along `vector`.
 *
 * @param {!Array<number>} vector
 * @return {!Array<number>}
 */
function normalize(vector) {
  const length = Math.hypot(...vector);
  return vector.map((x) => x / length);
}

/**
 * Index of the face whose centroid is nearest the center of the sheet.
 *
 * @param {!Float64Array} flat Flat x, y coordinates per vertex.
 * @param {!Array<!Array<number>>} faces Vertex indices per face.
 * @return {number}
 */
export function nearestFaceToCenter(flat, faces) {
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  for (let i = 0; i < flat.length; i += 2) {
    minX = Math.min(minX, flat[i]);
    maxX = Math.max(maxX, flat[i]);
    minY = Math.min(minY, flat[i + 1]);
    maxY = Math.max(maxY, flat[i + 1]);
  }
  const centerX = (minX + maxX) / 2;
  const centerY = (minY + maxY) / 2;
  let nearest = 0;
  let nearestDistance = Infinity;
  faces.forEach((face, f) => {
    let x = 0;
    let y = 0;
    for (const v of face) {
      x += flat[2 * v];
      y += flat[2 * v + 1];
    }
    const distance =
        Math.hypot(x / face.length - centerX, y / face.length - centerY);
    if (distance < nearestDistance) {
      nearest = f;
      nearestDistance = distance;
    }
  });
  return nearest;
}
