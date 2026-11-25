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
 * by BFS hinge rotation.
 *
 * Free of DOM and three.js dependencies, so it also runs under Node.
 *
 * Conventions are those of the FOLD spec (https://github.com/edemaine/fold):
 * frames are right-handed, a face is counter-clockwise when seen from the
 * front of the paper, and a positive fold angle is a valley fold, which
 * raises the neighboring faces towards the front.
 */

import {DEFAULT_STANDUP_START_T, evaluateScheduleAt, smootherstep} from './schedule.js';

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
 *     See `bp_pipeline.render_3d.mesh_to_fold_dict` for the canonical schema
 *     of all `bp:` FOLD keys.
 * @property {!Array<{kind: string, sticks: !Array<string>}>} regions
 *     `file_bp:regions`: the packed flaps and rivers.
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
 * The relative length error of each edge is averaged per vertex, then over
 * vertices.
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
 *     farthest from the centroid of the model in the folded frame (or in the
 *     flat frame without one), or -1 if the region has no faces.
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
 * Folds a model across a multi-stage schedule: rotates faces around a BFS
 * spanning tree of hinges from `rootFace`, averages each vertex across its
 * incident faces, and blends the Stage-2 residual to `model.folded`.
 */
export class Folder {
  /** @param {!Model} model */
  constructor(model) {
    /** @private @const {!Model} */
    this._model = model;
    const {flat, faces, edges} = model;
    const numVertices = flat.length / 2;

    /**
     * The face that stays put: the one nearest the middle of the sheet.
     * @const {number}
     */
    this.rootFace = nearestFaceToCenter(flat, faces);

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
    const hinges = [];
    /** @private @const {!Uint8Array} */
    this._reached = new Uint8Array(faces.length);
    this._reached[this.rootFace] = 1;
    const queue = [this.rootFace];
    for (let head = 0; head < queue.length; head++) {
      const parent = queue[head];
      const face = faces[parent];
      for (let i = 0; i < face.length; i++) {
        const a = face[i];
        const b = face[(i + 1) % face.length];
        const key = edgeKey(a, b, numVertices);
        for (const child of facesByEdge.get(key)) {
          if (this._reached[child]) continue;
          this._reached[child] = 1;
          queue.push(child);
          hinges.push(parent, child, edgeIndex.get(key) ?? -1, b, a);
        }
      }
    }
    /** @private @const {!Int32Array} */
    this._hinges = Int32Array.from(hinges);

    /** @private @const {!Float64Array} */
    this._faceCounts = new Float64Array(numVertices);
    faces.forEach((face, f) => {
      if (this._reached[f]) face.forEach((v) => this._faceCounts[v]++);
    });
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

    /**
     * Row-major 3x3 rotation from the frame of `positions` to the file's
     * folded frame, or null if the model has none.
     * @const {?Array<number>}
     */
    this.pose = model.folded ? this._rootFaceRotation() : null;
    /** @private @const {?Float32Array} */
    this._residual = this.pose ? this._residualToFoldedFrame() : null;

    /** @private {?Object} Multi-stage schedule object. */
    this._schedule = null;
    /**
     * Latest evaluated schedule state (label, activePhaseIndex, activeEdges,
     * foldedEdges, diagramKey, strainPct).
     * @type {?Object}
     */
    this.lastState = null;
  }

  /**
   * Attaches a multi-stage folding schedule.
   *
   * @param {?Object} schedule Schedule built by `buildFoldingSchedule`.
   */
  setSchedule(schedule) {
    this._schedule = schedule;
  }

  /**
   * Upright stand-up blend in `[0, 1]`: stays flat on the studio table (`0`)
   * throughout Act I collapse and Act II flat-clipping, then rises smoothly.
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
    if (standupStartT >= 1 - 1e-5 || progress <= standupStartT) return 0;
    return smootherstep(
        (progress - standupStartT) / Math.max(1e-5, 0.48 * (1 - standupStartT)));
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

    let residualWeight = p;
    if (useSchedule) {
      residualWeight = smootherstep(evalInfo.stage2Progress);
    } else {
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
    if (this._residual && residualWeight > 0) {
      for (let i = 0; i < positions.length; i++) {
        positions[i] += residualWeight * this._residual[i];
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
   * Folds every crease to the given angle.
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
function nearestFaceToCenter(flat, faces) {
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
