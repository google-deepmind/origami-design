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
 * @fileoverview Planar crease-pattern graph operations, Kawasaki/Maekawa
 * flat-foldability diagnostics, and instant client-side FOLD mesh construction.
 *
 * Free of DOM and three.js dependencies so it runs identically in browsers and
 * under Node tests.
 */

import {Folder, parseFold} from '../../js/fold.js';

const ASSIGNMENT_PRECEDENCE = {
  M: 3,
  V: 3,
  B: 2,
  F: 1,
  U: 0,
};

export const DEFAULT_ASSIGNMENT_ANGLES = {
  M: -180,
  V: 180,
  B: 0,
  F: 0,
  U: 0,
};

/**
 * Builds an editable planar graph object from a parsed Studio `Model`.
 *
 * @param {!Object} model Parsed model from `fold.js`.
 * @return {!Object}
 */
export function graphFromModel(model) {
  const vertices = [];
  for (let i = 0; i < model.flat.length; i += 2) {
    vertices.push([model.flat[i], model.flat[i + 1]]);
  }
  const edges = model.edges.map(([u, v]) => [u, v]);
  const assignments = model.assignments.map(
      (a) => (a in DEFAULT_ASSIGNMENT_ANGLES ? a : 'F'));
  const foldAngles = Array.from(model.foldAngles);
  const baseFoldAngles = model.baseFoldAngles ?
      Array.from(model.baseFoldAngles) :
      foldAngles.slice();
  const shaping = model.shaping ? model.shaping.slice() : null;
  return {
    vertices,
    edges,
    assignments,
    foldAngles,
    baseFoldAngles,
    shaping,
    gridSize: model.gridSize || 16,
  };
}

/**
 * Deep-clones a planar graph state for undo/redo stacks.
 *
 * @param {!Object} graph
 * @return {!Object}
 */
export function cloneGraph(graph) {
  return {
    vertices: graph.vertices.map(([x, y]) => [x, y]),
    edges: graph.edges.map(([u, v]) => [u, v]),
    assignments: graph.assignments.slice(),
    foldAngles: graph.foldAngles.slice(),
    baseFoldAngles: graph.baseFoldAngles ? graph.baseFoldAngles.slice() : null,
    shaping: graph.shaping ? graph.shaping.slice() : null,
    gridSize: graph.gridSize || 16,
  };
}

/**
 * Creates a clean blank square sheet `[0, 1]^2` with 4 border edges.
 *
 * @param {number=} gridSize
 * @return {!Object}
 */
export function createBlankSquareGraph(gridSize = 16) {
  return {
    vertices: [[0, 0], [1, 0], [1, 1], [0, 1]],
    edges: [[0, 1], [1, 2], [2, 3], [3, 0]],
    assignments: ['B', 'B', 'B', 'B'],
    foldAngles: [0, 0, 0, 0],
    baseFoldAngles: [0, 0, 0, 0],
    shaping: null,
    gridSize,
  };
}

/**
 * Snaps `(x, y)` according to `snapMode` and optional angle lock from `anchor`.
 *
 * @param {number} x
 * @param {number} y
 * @param {!Object} graph
 * @param {!Object=} options
 * @return {!Array<number>} `[sx, sy]` clamped to `[0, 1]^2`.
 */
export function snapPoint(x, y, graph, options = {}) {
  const {snapMode = 'grid', angleLock = 'off', anchor = null} = options;
  let sx = Math.max(0, Math.min(1, x));
  let sy = Math.max(0, Math.min(1, y));
  const g = Math.max(2, graph.gridSize || 16);

  // Vertex snap always wins when within 0.35 grid units of an existing vertex.
  const vertexSnapRadius = Math.min(0.025, 0.38 / g);
  if (snapMode !== 'free') {
    let bestD = vertexSnapRadius;
    let bestV = null;
    for (const [vx, vy] of graph.vertices) {
      const d = Math.hypot(vx - sx, vy - sy);
      if (d < bestD) {
        bestD = d;
        bestV = [vx, vy];
      }
    }
    if (bestV && (!anchor || Math.hypot(bestV[0] - anchor[0], bestV[1] - anchor[1]) > 1e-6)) {
      return [bestV[0], bestV[1]];
    }
  }

  if (snapMode === 'grid') {
    sx = Math.round(sx * g) / g;
    sy = Math.round(sy * g) / g;
  } else if (snapMode === 'half') {
    sx = Math.round(sx * 2 * g) / (2 * g);
    sy = Math.round(sy * 2 * g) / (2 * g);
  }

  if (anchor && angleLock !== 'off') {
    const dx = sx - anchor[0];
    const dy = sy - anchor[1];
    const dist = Math.hypot(dx, dy);
    if (dist > 1e-6) {
      const stepDeg = angleLock === '22.5' ? 22.5 : 45;
      const stepRad = (stepDeg * Math.PI) / 180;
      const snappedAngle = Math.round(Math.atan2(dy, dx) / stepRad) * stepRad;
      const ux = Math.cos(snappedAngle);
      const uy = Math.sin(snappedAngle);
      // Clip ray onto [0, 1]^2 and snap distance if axial/45°.
      let maxT = dist;
      for (const [axisCoord, dir] of [[anchor[0], ux], [anchor[1], uy]]) {
        if (dir > 1e-9) maxT = Math.min(maxT, (1 - axisCoord) / dir);
        else if (dir < -1e-9) maxT = Math.min(maxT, -axisCoord / dir);
      }
      sx = anchor[0] + maxT * ux;
      sy = anchor[1] + maxT * uy;
      if (snapMode === 'grid' || snapMode === 'half') {
        const div = snapMode === 'half' ? 2 * g : g;
        if (Math.abs(ux) < 1e-6) sy = Math.round(sy * div) / div;
        else if (Math.abs(uy) < 1e-6) sx = Math.round(sx * div) / div;
        else if (Math.abs(Math.abs(ux) - Math.abs(uy)) < 1e-4) {
          const delta = Math.round((maxT * Math.abs(ux)) * div) / div;
          sx = anchor[0] + Math.sign(ux) * delta;
          sy = anchor[1] + Math.sign(uy) * delta;
        }
      }
    }
  }
  return [
    +Math.max(0, Math.min(1, sx)).toFixed(6),
    +Math.max(0, Math.min(1, sy)).toFixed(6),
  ];
}

/**
 * Splits all crossing and T-junction segments, merges duplicate vertices/edges,
 * and prunes zero-length segments (`cp-editor`-style planar graph invariant).
 *
 * @param {!Object} rawGraph
 * @param {number=} tol
 * @return {!Object} Normalized planar graph.
 */
export function normalizePlanarGraph(rawGraph, tol = 1e-5) {
  const verts = [];
  const addVertex = (x, y) => {
    for (let i = 0; i < verts.length; i++) {
      if (Math.hypot(verts[i][0] - x, verts[i][1] - y) <= tol) return i;
    }
    verts.push([+x.toFixed(6), +y.toFixed(6)]);
    return verts.length - 1;
  };

  const segs = [];
  for (let e = 0; e < rawGraph.edges.length; e++) {
    const [u0, v0] = rawGraph.edges[e];
    const p1 = rawGraph.vertices[u0];
    const p2 = rawGraph.vertices[v0];
    if (!p1 || !p2) continue;
    if (Math.hypot(p2[0] - p1[0], p2[1] - p1[1]) <= tol) continue;
    const u = addVertex(p1[0], p1[1]);
    const v = addVertex(p2[0], p2[1]);
    if (u === v) continue;
    const assignment = rawGraph.assignments[e] || 'F';
    const foldAngle = rawGraph.foldAngles?.[e] ??
        (DEFAULT_ASSIGNMENT_ANGLES[assignment] ?? 0);
    segs.push({u, v, assignment, foldAngle});
  }

  // Pairwise proper intersection points.
  const nInit = segs.length;
  const splitParams = Array.from({length: nInit}, () => [0, 1]);
  for (let i = 0; i < nInit; i++) {
    const [ax, ay] = verts[segs[i].u];
    const [bx, by] = verts[segs[i].v];
    const abx = bx - ax;
    const aby = by - ay;
    for (let j = i + 1; j < nInit; j++) {
      const [cx, cy] = verts[segs[j].u];
      const [dx, dy] = verts[segs[j].v];
      const cdx = dx - cx;
      const cdy = dy - cy;
      const denom = abx * cdy - aby * cdx;
      if (Math.abs(denom) <= 1e-10) continue;
      const acx = cx - ax;
      const acy = cy - ay;
      const t = (acx * cdy - acy * cdx) / denom;
      const s = (acx * aby - acy * abx) / denom;
      if (t > -1e-6 && t < 1 + 1e-6 && s > -1e-6 && s < 1 + 1e-6) {
        const tc = Math.max(0, Math.min(1, t));
        const sc = Math.max(0, Math.min(1, s));
        const ix = ax + tc * abx;
        const iy = ay + tc * aby;
        addVertex(ix, iy);
        if (tc > 1e-5 && tc < 1 - 1e-5) splitParams[i].push(tc);
        if (sc > 1e-5 && sc < 1 - 1e-5) splitParams[j].push(sc);
      }
    }
  }

  // Also split segments at any vertex lying on their interior (T-junctions /
  // collinear overlaps).
  for (let i = 0; i < nInit; i++) {
    const [ax, ay] = verts[segs[i].u];
    const [bx, by] = verts[segs[i].v];
    const abx = bx - ax;
    const aby = by - ay;
    const lenSq = abx * abx + aby * aby;
    if (lenSq <= tol * tol) continue;
    const len = Math.sqrt(lenSq);
    for (let vIdx = 0; vIdx < verts.length; vIdx++) {
      if (vIdx === segs[i].u || vIdx === segs[i].v) continue;
      const [px, py] = verts[vIdx];
      const apx = px - ax;
      const apy = py - ay;
      const dist = Math.abs(abx * apy - aby * apx) / len;
      if (dist <= tol) {
        const t = (abx * apx + aby * apy) / lenSq;
        if (t > tol / len && t < 1 - tol / len) splitParams[i].push(t);
      }
    }
  }

  const byKey = new Map();
  for (let i = 0; i < nInit; i++) {
    const {u, v, assignment, foldAngle} = segs[i];
    const [ax, ay] = verts[u];
    const [bx, by] = verts[v];
    const ts = splitParams[i].sort((a, b) => a - b);
    const uniqueVerts = [];
    for (const t of ts) {
      const vid = t <= 1e-7 ? u :
                  (t >= 1 - 1e-7 ? v :
                   addVertex(ax + t * (bx - ax), ay + t * (by - ay)));
      if (uniqueVerts.at(-1) !== vid) uniqueVerts.push(vid);
    }
    for (let k = 0; k < uniqueVerts.length - 1; k++) {
      const va = uniqueVerts[k];
      const vb = uniqueVerts[k + 1];
      if (va === vb) continue;
      const key = va < vb ? `${va}:${vb}` : `${vb}:${va}`;
      const prev = byKey.get(key);
      // Newer drawn edges sit later in `segs`, so `>=` lets user edits overwrite.
      if (!prev ||
          (ASSIGNMENT_PRECEDENCE[assignment] ?? 0) >=
              (ASSIGNMENT_PRECEDENCE[prev.assignment] ?? 0) ||
          assignment !== 'F') {
        byKey.set(key, {
          u: Math.min(va, vb),
          v: Math.max(va, vb),
          assignment,
          foldAngle,
        });
      }
    }
  }

  // Drop isolated vertices with degree 0.
  const used = new Int32Array(verts.length).fill(-1);
  const compactVerts = [];
  const edges = [];
  const assignments = [];
  const foldAngles = [];
  for (const rec of byKey.values()) {
    if (used[rec.u] < 0) {
      used[rec.u] = compactVerts.length;
      compactVerts.push(verts[rec.u]);
    }
    if (used[rec.v] < 0) {
      used[rec.v] = compactVerts.length;
      compactVerts.push(verts[rec.v]);
    }
    edges.push([used[rec.u], used[rec.v]]);
    assignments.push(rec.assignment);
    foldAngles.push(rec.foldAngle);
  }

  return {
    vertices: compactVerts,
    edges,
    assignments,
    foldAngles,
    baseFoldAngles: null,
    shaping: null,
    gridSize: rawGraph.gridSize || 16,
  };
}

/**
 * Adds a crease segment `(p1 -> p2)` (and its `x = 0.5` bilateral mirror when
 * `mirror === true`) and re-normalizes the planar graph.
 *
 * @param {!Object} graph
 * @param {!Array<number>} p1
 * @param {!Array<number>} p2
 * @param {string} assignment
 * @param {number} foldAngle
 * @param {boolean=} mirror
 * @return {!Object}
 */
export function addCreaseSegment(
    graph, p1, p2, assignment, foldAngle, mirror = false) {
  if (Math.hypot(p2[0] - p1[0], p2[1] - p1[1]) <= 1e-5) return graph;
  const next = cloneGraph(graph);
  const appendSeg = (a, b) => {
    const u = next.vertices.length;
    next.vertices.push([a[0], a[1]], [b[0], b[1]]);
    next.edges.push([u, u + 1]);
    next.assignments.push(assignment);
    next.foldAngles.push(foldAngle);
  };
  appendSeg(p1, p2);
  if (mirror) {
    const mp1 = [+(1 - p1[0]).toFixed(6), p1[1]];
    const mp2 = [+(1 - p2[0]).toFixed(6), p2[1]];
    const sameLine =
        (Math.hypot(mp1[0] - p1[0], mp1[1] - p1[1]) < 1e-5 &&
         Math.hypot(mp2[0] - p2[0], mp2[1] - p2[1]) < 1e-5) ||
        (Math.hypot(mp1[0] - p2[0], mp1[1] - p2[1]) < 1e-5 &&
         Math.hypot(mp2[0] - p1[0], mp2[1] - p1[1]) < 1e-5);
    if (!sameLine) appendSeg(mp1, mp2);
  }
  return normalizePlanarGraph(next);
}

/**
 * Extracts bounded counter-clockwise faces in FOLD flat coordinates (`+y` up).
 *
 * @param {!Array<!Array<number>>} vertices
 * @param {!Array<!Array<number>>} edges
 * @return {!Array<!Array<number>>}
 */
export function extractPlanarFaces(vertices, edges) {
  const n = vertices.length;
  const adj = Array.from({length: n}, () => []);
  for (const [u, v] of edges) {
    if (u !== v) {
      adj[u].push(v);
      adj[v].push(u);
    }
  }
  // Prune degree-1 dangling filaments iteratively so interior face loops stay clean.
  const deg = adj.map((nb) => nb.length);
  const dead = new Uint8Array(n);
  const queue = [];
  for (let i = 0; i < n; i++) {
    if (deg[i] <= 1) queue.push(i);
  }
  while (queue.length > 0) {
    const v = queue.pop();
    if (dead[v]) continue;
    dead[v] = 1;
    for (const nb of adj[v]) {
      if (!dead[nb] && --deg[nb] <= 1) queue.push(nb);
    }
  }

  const nextHalf = new Map();
  const hk = (a, b) => a * n + b;
  for (let v = 0; v < n; v++) {
    if (dead[v]) continue;
    const [vx, vy] = vertices[v];
    const nbs = [...new Set(adj[v].filter((u) => !dead[u]))].sort(
        (a, b) => Math.atan2(vertices[a][1] - vy, vertices[a][0] - vx) -
                  Math.atan2(vertices[b][1] - vy, vertices[b][0] - vx));
    if (nbs.length < 2) continue;
    for (let i = 0; i < nbs.length; i++) {
      // To walk CCW minimal faces on (+x right, +y up), arriving along (u -> v)
      // leaves along the clockwise-previous neighbor around v.
      const u = nbs[i];
      const nextW = nbs[(i - 1 + nbs.length) % nbs.length];
      nextHalf.set(hk(u, v), nextW);
    }
  }

  const visited = new Set();
  const faces = [];
  const maxSteps = nextHalf.size + 4;
  for (const startKey of nextHalf.keys()) {
    if (visited.has(startKey)) continue;
    let curKey = startKey;
    const loop = [];
    let valid = true;
    while (!visited.has(curKey)) {
      visited.add(curKey);
      const u = Math.floor(curKey / n);
      const v = curKey % n;
      loop.push(u);
      const w = nextHalf.get(curKey);
      if (w === undefined || loop.length > maxSteps) {
        valid = false;
        break;
      }
      curKey = hk(v, w);
    }
    if (!valid || curKey !== startKey || loop.length < 3) continue;
    let area2 = 0;
    for (let i = 0; i < loop.length; i++) {
      const [x1, y1] = vertices[loop[i]];
      const [x2, y2] = vertices[loop[(i + 1) % loop.length]];
      area2 += x1 * y2 - x2 * y1;
    }
    // Bounded interior faces are CCW (area2 > 0); the unbounded outer face is CW (area2 < 0).
    if (area2 > 1e-10) {
      faces.push(loop);
    }
  }
  return faces;
}

/**
 * Evaluates Maekawa and Kawasaki flat-foldability conditions at every interior
 * vertex of `graph`.
 *
 * @param {!Object} graph
 * @return {!Array<{
 *   vertex: number,
 *   x: number,
 *   y: number,
 *   kind: string,
 *   message: string
 * }>}
 */
export function analyzeVertices(graph) {
  const {vertices, edges, assignments, foldAngles} = graph;
  const n = vertices.length;
  const isBorder = new Uint8Array(n);
  const incident = Array.from({length: n}, () => []);

  for (let v = 0; v < n; v++) {
    const [x, y] = vertices[v];
    if (x <= 1e-4 || x >= 1 - 1e-4 || y <= 1e-4 || y >= 1 - 1e-4) {
      isBorder[v] = 1;
    }
  }

  for (let e = 0; e < edges.length; e++) {
    const [u, v] = edges[e];
    const a = assignments[e];
    if (a === 'B') {
      isBorder[u] = 1;
      isBorder[v] = 1;
      continue;
    }
    const ang = foldAngles?.[e] ?? (DEFAULT_ASSIGNMENT_ANGLES[a] ?? 0);
    if ((a !== 'M' && a !== 'V') || Math.abs(ang) < 1) continue;
    incident[u].push({nb: v, a, absAng: Math.abs(ang)});
    incident[v].push({nb: u, a, absAng: Math.abs(ang)});
  }

  const violations = [];
  for (let v = 0; v < n; v++) {
    if (isBorder[v]) continue;
    const inc = incident[v];
    if (inc.length === 0) continue;
    const [vx, vy] = vertices[v];
    // Only enforce 2D flat-foldability theorems when all incident folds are ~180°.
    const allFlat180 = inc.every((it) => Math.abs(it.absAng - 180) <= 3.5);
    if (!allFlat180) continue;

    if (inc.length < 4 || inc.length % 2 !== 0) {
      violations.push({
        vertex: v,
        x: vx,
        y: vy,
        kind: 'degree',
        message: `Vertex #${v}: odd/dangling degree ${inc.length} (flat-foldable interior vertices need even degree ≥ 4)`,
      });
      continue;
    }

    let mCount = 0;
    let vCount = 0;
    for (const it of inc) {
      if (it.a === 'M') mCount++;
      else if (it.a === 'V') vCount++;
    }
    if (Math.abs(mCount - vCount) !== 2) {
      violations.push({
        vertex: v,
        x: vx,
        y: vy,
        kind: 'maekawa',
        message: `Vertex #${v} Maekawa violation: M=${mCount}, V=${vCount} (|M − V| = ${Math.abs(mCount - vCount)}, expected 2)`,
      });
      continue;
    }

    const thetas = inc
        .map(({nb}) => Math.atan2(vertices[nb][1] - vy, vertices[nb][0] - vx))
        .sort((a, b) => a - b);
    let evenSum = 0;
    for (let i = 0; i < thetas.length; i += 2) {
      let sector = (i + 1 < thetas.length ? thetas[i + 1] : thetas[0] + 2 * Math.PI) - thetas[i];
      if (sector < 0) sector += 2 * Math.PI;
      evenSum += (sector * 180) / Math.PI;
    }
    const kawasakiErr = Math.abs(evenSum - 180);
    if (kawasakiErr > 2.5) {
      violations.push({
        vertex: v,
        x: vx,
        y: vy,
        kind: 'kawasaki',
        message: `Vertex #${v} Kawasaki violation: alternating angles sum to ${evenSum.toFixed(1)}° (off by ${kawasakiErr.toFixed(1)}° from 180°)`,
      });
    }
  }
  return violations;
}

/**
 * Applies a sheet-level geometric transform (`mirrorH`, `mirrorV`, `rotCW`,
 * `rotCCW`, `invertMV`) to `graph`.
 *
 * @param {!Object} graph
 * @param {string} kind
 * @return {!Object}
 */
export function transformGraph(graph, kind) {
  const next = cloneGraph(graph);
  if (kind === 'invertMV') {
    for (let e = 0; e < next.edges.length; e++) {
      if (next.assignments[e] === 'M') next.assignments[e] = 'V';
      else if (next.assignments[e] === 'V') next.assignments[e] = 'M';
      next.foldAngles[e] = -next.foldAngles[e];
    }
    return next;
  }
  next.vertices = next.vertices.map(([x, y]) => {
    switch (kind) {
      case 'mirrorH': return [+(1 - x).toFixed(6), y];
      case 'mirrorV': return [x, +(1 - y).toFixed(6)];
      case 'rotCW': return [y, +(1 - x).toFixed(6)];
      case 'rotCCW': return [+(1 - y).toFixed(6), x];
      default: return [x, y];
    }
  });
  return normalizePlanarGraph(next);
}

/**
 * Builds a complete Studio `Model` and FOLD v1.1 dictionary in pure JS (<2ms)
 * from an edited planar `graph`, using `Folder` for exact 3D hinge coordinates.
 *
 * @param {!Object} graph Normalized planar graph.
 * @param {string=} title
 * @return {{foldDict: !Object, model: !Object}}
 */
export function buildFoldFromCpGraph(graph, title = 'Edited Crease Pattern') {
  const norm = normalizePlanarGraph(graph);
  const faces = extractPlanarFaces(norm.vertices, norm.edges);
  if (faces.length === 0) {
    throw new Error('No closed sheet faces found — ensure the outer boundary is closed.');
  }

  const draftFold = {
    file_spec: 1.1,
    file_creator: 'COrigami Studio Editor',
    file_title: title,
    file_classes: ['singleModel'],
    frame_classes: ['creasePattern'],
    frame_attributes: ['2D'],
    vertices_coords: norm.vertices,
    faces_vertices: faces,
    edges_vertices: norm.edges,
    edges_assignment: norm.assignments,
    edges_foldAngle: norm.foldAngles,
    'file_bp:gridSize': norm.gridSize || 16,
  };

  const tempModel = parseFold(draftFold);
  const folder = new Folder(tempModel);
  const pos3d = folder.positionsAt(1, false);
  const nVerts = pos3d.length / 3;
  let cx = 0;
  let cy = 0;
  let cz = 0;
  for (let i = 0; i < pos3d.length; i += 3) {
    cx += pos3d[i] / nVerts;
    cy += pos3d[i + 2] / nVerts;
    cz += -pos3d[i + 1] / nVerts;
  }
  const vertices3d = [];
  for (let i = 0; i < pos3d.length; i += 3) {
    vertices3d.push([
      +(pos3d[i] - cx).toFixed(6),
      +(pos3d[i + 2] - cy).toFixed(6),
      +(-pos3d[i + 1] - cz).toFixed(6),
    ]);
  }

  const foldDict = {
    file_spec: 1.1,
    file_creator: 'COrigami Studio Editor',
    file_title: title,
    file_classes: ['singleModel'],
    frame_classes: ['foldedForm'],
    frame_attributes: ['3D'],
    vertices_coords: vertices3d,
    faces_vertices: faces,
    edges_vertices: norm.edges,
    edges_assignment: norm.assignments,
    edges_foldAngle: norm.foldAngles.map((a) => +a.toFixed(4)),
    'file_bp:gridSize': norm.gridSize || 16,
    file_frames: [{
      frame_classes: ['creasePattern'],
      frame_attributes: ['2D'],
      frame_inherit: true,
      frame_parent: 0,
      vertices_coords: norm.vertices,
    }],
  };

  return {foldDict, model: parseFold(foldDict), normalizedGraph: norm};
}
