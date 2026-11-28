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
 * @fileoverview Multi-stage unbroken box-pleat collapse and proximal-to-distal
 * 3D shaping schedules.
 *
 * Partitions folding into four physical acts:
 *   - Act I (Collapse): Unbroken single-sheet Signed-Dihedral + Facet-Crease
 *     XPBD gather collapsing horizontal, vertical, and 45-degree miter creases
 *     onto the compact box-pleated base without seam tears.
 *   - Act II (Narrow Flap): Pure 180-degree flap-narrowing clip folds on the
 *     studio table.
 *   - Act III (Pose Limb): Proximal-to-distal skeletal river bends, limb-root
 *     hinges, and coupled crimps routed over a Short-Hop Radial Dijkstra tree.
 *   - Act IV (Fan Out): Volumetric wing, plume, hood, crest, and tip fan-out
 *     locked to the final stage so collapsed flaps stay compressed until the
 *     end.
 *
 * Free of DOM and three.js dependencies, so it also runs under Node.
 */

import {Folder, buildShortHopDijkstra, nearestFaceToCenter} from './fold.js';
import {buildSignedXPBDSystem, solveLiveSignedXPBD} from './xpbd.js';

/** Default schedule progress `t` where Act I base collapse completes. */
export const DEFAULT_BASE_COMPLETE_T = 0.37;

/** Default schedule progress `t` where Stage 2 shaping begins. */
export const DEFAULT_SHAPING_START_T = 0.41;

/** Default schedule progress `t` where the sheet begins standing upright. */
export const DEFAULT_STANDUP_START_T = 0.41;

/** Playback clock `u` where `warpFoldingTime` reaches `baseCompleteT`. */
export const WARP_STAGE1_END_U = 0.335;

/** Playback clock `u` where `warpFoldingTime` finishes the base hold. */
export const WARP_HOLD_END_U = 0.380;

const BILATERAL_SUFFIX = /(_left|_right|_l|_r|left|right)$/i;
const FAN_OUT_TOKEN =
    /(^|_)(wing|wings|fan|plume|tail_[lr]\d+|tail_center|hood|crest)(_|$)/i;

/**
 * C2-continuous quintic ease-in-out on `[0, 1]`.
 *
 * @param {number} x
 * @return {number}
 */
export function smootherstep(x) {
  const c = Math.max(0, Math.min(1, x));
  return c * c * c * (c * (c * 6 - 15) + 10);
}

/**
 * Evaluates C2 phase progress with optional power warp and final deceleration.
 *
 * @param {number} x Normalized phase progress in `[0, 1]`.
 * @param {number=} easeOut Deceleration strength in `[0, 1]`.
 * @param {number=} power Monotone power exponent (`> 0`).
 * @return {number}
 */
export function easePhaseProgress(x, easeOut = 0, power = 1) {
  let c = Math.max(0, Math.min(1, x));
  if (power > 0 && Math.abs(power - 1) > 1e-4) {
    c = Math.pow(c, power);
  }
  if (easeOut <= 1e-6) return smootherstep(c);
  const blend = Math.max(0, Math.min(1, easeOut));
  const decel = 1 - Math.pow(1 - c, 1.75);
  return smootherstep((1 - blend) * c + blend * decel);
}

/**
 * Maps normalized playback clock `u in [0, 1]` to schedule `t in [0, 1]`,
 * decelerating smoothly approaching both the collapsed base and 100% folded.
 *
 * @param {number} uLinear Normalized playback time in `[0, 1]`.
 * @param {number=} baseCompleteT Schedule `t` where Act I collapse finishes.
 * @param {number=} shapingStartT Schedule `t` where Stage 2 shaping starts.
 * @return {number}
 */
export function warpFoldingTime(
    uLinear,
    baseCompleteT = DEFAULT_BASE_COMPLETE_T,
    shapingStartT = DEFAULT_SHAPING_START_T) {
  const u = Math.max(0, Math.min(1, uLinear));
  if (u <= WARP_STAGE1_END_U) {
    const s = u / WARP_STAGE1_END_U;
    const sWarp = 1 - (1 - s) * (1 - 0.58 * s * s);
    return baseCompleteT * Math.max(0, Math.min(1, sWarp));
  }
  if (u <= WARP_HOLD_END_U) {
    const s = (u - WARP_STAGE1_END_U) / (WARP_HOLD_END_U - WARP_STAGE1_END_U);
    return baseCompleteT + (shapingStartT - baseCompleteT) * s;
  }
  const s = (u - WARP_HOLD_END_U) / (1 - WARP_HOLD_END_U);
  const sWarp = 1 - (1 - s) * (1 - 0.60 * s * s * s);
  return shapingStartT + (1 - shapingStartT) * Math.max(0, Math.min(1, sWarp));
}

/**
 * Monotonic inverse of `warpFoldingTime` via 28-step bisection.
 *
 * @param {number} tTarget Schedule `t` in `[0, 1]`.
 * @param {number=} baseCompleteT
 * @param {number=} shapingStartT
 * @return {number} Normalized playback `u` in `[0, 1]`.
 */
export function unwarpFoldingTime(
    tTarget,
    baseCompleteT = DEFAULT_BASE_COMPLETE_T,
    shapingStartT = DEFAULT_SHAPING_START_T) {
  const t = Math.max(0, Math.min(1, tTarget));
  if (t <= 0) return 0;
  if (t >= 1) return 1;
  if (Math.abs(t - baseCompleteT) <= 1e-12) return WARP_STAGE1_END_U;
  let lo = 0;
  let hi = 1;
  for (let i = 0; i < 28; i++) {
    const mid = 0.5 * (lo + hi);
    if (warpFoldingTime(mid, baseCompleteT, shapingStartT) < t) {
      lo = mid;
    } else {
      hi = mid;
    }
  }
  return 0.5 * (lo + hi);
}

/**
 * Computes stick hop distances from `stickFigure.root`.
 *
 * @param {?Object} stickFigure Parsed stick figure JSON, or null.
 * @return {!Map<string, number>}
 */
function stickHopDistances(stickFigure) {
  const distances = new Map();
  if (!stickFigure?.children?.length) return distances;
  const sticks = stickFigure.children;
  const rootName = stickFigure.root ?? sticks[0].name;
  const byJoint = new Map();
  for (const s of sticks) {
    if (!byJoint.has(s.start)) byJoint.set(s.start, []);
    if (!byJoint.has(s.end)) byJoint.set(s.end, []);
    byJoint.get(s.start).push(s.name);
    byJoint.get(s.end).push(s.name);
  }
  const byName = new Map(sticks.map((s) => [s.name, s]));
  distances.set(rootName, 0);
  const queue = [rootName];
  for (let head = 0; head < queue.length; head++) {
    const cur = queue[head];
    const s = byName.get(cur);
    if (!s) continue;
    const nextDist = distances.get(cur) + 1;
    for (const joint of [s.start, s.end]) {
      for (const nb of byJoint.get(joint) ?? []) {
        if (!distances.has(nb)) {
          distances.set(nb, nextDist);
          queue.push(nb);
        }
      }
    }
  }
  return distances;
}

/**
 * Formats a region's terminal stick into a clean bilateral title.
 *
 * @param {!Object} region Entry from `model.regions`.
 * @return {string}
 */
function cleanRegionTitle(region) {
  const raw = region?.sticks?.at(-1) ?? 'trunk';
  const stripped = raw.replace(BILATERAL_SUFFIX, '').replace(/^_+|_+$/g, '');
  const tokens = (stripped || raw).split('_').filter(Boolean);
  const chosen = tokens.length > 3 ? tokens.slice(-2) : tokens;
  return chosen.map((w) => w[0].toUpperCase() + w.slice(1)).join(' ') ||
      'Trunk';
}

/**
 * Classifies an edge as horizontal axial, vertical axial, or 45-degree ridge.
 *
 * @param {!Float64Array} flat
 * @param {!Array<number>} pair
 * @return {string}
 */
function classifyEdgeAxis(flat, [u, v]) {
  const dx = Math.abs(flat[2 * v] - flat[2 * u]);
  const dy = Math.abs(flat[2 * v + 1] - flat[2 * u + 1]);
  const len = Math.hypot(dx, dy) || 1;
  const ux = dx / len;
  const uy = dy / len;
  if (Math.abs(ux - uy) < 0.35 && Math.min(ux, uy) > 0.25) return 'ridge';
  return dy <= dx ? 'horiz' : 'vert';
}

/**
 * Avoids spurious 360-degree wraps when both base and final angles are ±180.
 *
 * @param {number} baseAngle
 * @param {number} finalAngle
 * @return {number}
 */
function effectiveTargetAngle(baseAngle, finalAngle) {
  if (Math.abs(baseAngle) >= 178 && Math.abs(finalAngle) >= 178) {
    return baseAngle;
  }
  return finalAngle;
}

/**
 * Builds a 4-Act multi-stage folding schedule directly from a parsed `Model`,
 * computing a Short-Hop Radial Dijkstra schedule for Acts II–IV and (when
 * `options.solveXpbd` is true) solving an unbroken Signed-Dihedral XPBD gather
 * for Act I. Interactive UI callers leave `solveXpbd` false (`< 2 ms`) and
 * solve Act-I XPBD asynchronously in `xpbd_worker.js`.
 *
 * @param {!Object} model Parsed `Model` from `fold.js`.
 * @param {?Object=} stickFigure Optional stick figure JSON.
 * @param {{solveXpbd: (boolean|undefined)}=} options Optional solver flags.
 * @return {!Object}
 */
export function buildFoldingSchedule(
    model, stickFigure = null, options = {}) {
  const {solveXpbd = false} = options;
  const {
    flat,
    faces,
    edges,
    assignments,
    foldAngles,
    baseFoldAngles,
    regions,
    faceRegions,
  } = model;
  const numEdges = edges.length;
  const nV = flat.length / 2;
  const rf = nearestFaceToCenter(flat, faces);
  const hops = stickHopDistances(stickFigure);

  const regionMeta = regions.map((r, idx) => {
    const sticks = r.sticks ?? [];
    const rawName = sticks.at(-1) ?? '';
    const minHop = sticks.length ?
        Math.min(...sticks.map((s) => hops.get(s) ?? 1)) :
        1;
    const isRiver = r.kind === 'river';
    const isNamedFan =
        r.kind === 'flap' && sticks.some((s) => FAN_OUT_TOKEN.test(s));
    return {
      id: idx,
      kind: r.kind,
      rawName,
      isNamedFan,
      hop: isRiver && minHop === 0 ? 0 : minHop,
      title: cleanRegionTitle(r),
    };
  });

  const edgeRegions = Array.from({length: numEdges}, () => new Set());
  const edgeByKey = new Map();
  edges.forEach(([u, v], e) => {
    edgeByKey.set(u < v ? u * nV + v : v * nV + u, e);
  });
  faces.forEach((face, f) => {
    const rid = faceRegions[f];
    if (rid == null || rid < 0) return;
    for (let i = 0; i < face.length; i++) {
      const a = face[i];
      const b = face[(i + 1) % face.length];
      const e = edgeByKey.get(a < b ? a * nV + b : b * nV + a);
      if (e !== undefined) edgeRegions[e].add(rid);
    }
  });

  const s1Edges = [];
  for (let e = 0; e < numEdges; e++) {
    if (assignments[e] !== 'B' && Math.abs(baseFoldAngles[e]) >= 1) {
      s1Edges.push(e);
    }
  }
  const hasS2 = foldAngles.some(
      (a, e) => Math.abs(effectiveTargetAngle(baseFoldAngles[e], a) -
                         baseFoldAngles[e]) >= 1);

  if (s1Edges.length === 0 || !hasS2) {
    const active = [];
    const deltas = [];
    for (let e = 0; e < numEdges; e++) {
      if (assignments[e] !== 'B' && Math.abs(foldAngles[e]) >= 1) {
        active.push(e);
        deltas.push(foldAngles[e]);
      }
    }
    return {
      baseCompleteT: 1.0,
      shapingStartT: 1.0,
      standupStartT: 0.0,
      rootFace: rf,
      phases: [{
        id: 'fold_all',
        act: 'I',
        stageGroup: 'collapse',
        label: 'Act I.1 · Fold Crease Pattern',
        tStart: 0.0,
        tEnd: 1.0,
        easeOut: 0.6,
        power: 1.0,
        edgeIndices: active,
        deltas: Float64Array.from(deltas),
      }],
    };
  }

  const tBase = DEFAULT_BASE_COMPLETE_T;
  const tShapeStart = DEFAULT_SHAPING_START_T;

  // Act I semantic wave phases (coupled outer/core horizontal, vertical, and
  // 45-degree miter ridges).
  const hOut = [];
  const hIn = [];
  const vOut = [];
  const vIn = [];
  const rAll = [];
  for (const e of s1Edges) {
    const [u, v] = edges[e];
    const rad = Math.hypot(
        0.5 * (flat[2 * u] + flat[2 * v]) - 0.5,
        0.5 * (flat[2 * u + 1] + flat[2 * v + 1]) - 0.5) / 0.5;
    const outer = rad >= 0.48;
    const ax = classifyEdgeAxis(flat, edges[e]);
    if (ax === 'horiz') (outer ? hOut : hIn).push(e);
    else if (ax === 'vert') (outer ? vOut : vIn).push(e);
    else rAll.push(e);
  }
  const act1Specs = [
    ['Outer Horizontal Pleats', hOut, 0.00, 0.96],
    ['Outer Vertical Pleats', vOut, 0.02, 0.96],
    ['Core Horizontal Pleats', hIn, 0.05, 0.98],
    ['Core Vertical Pleats', vIn, 0.07, 0.98],
    ['45° Miter Ridges & Base Lock', rAll, 0.04, 1.00],
  ].filter(([, es]) => es.length > 0);

  const phases = act1Specs.map(([title, es, u0, u1], idx) => ({
    id: `s1_w${idx}`,
    act: 'I',
    stageGroup: 'collapse',
    label: `Act I.${idx + 1} · Gather ${title}`,
    tStart: +(u0 * tBase).toFixed(4),
    tEnd: +(idx === act1Specs.length - 1 ? tBase : u1 * tBase).toFixed(4),
    easeOut: 0.35,
    power: 1.0,
    edgeIndices: es,
    deltas: Float64Array.from(es, (e) => baseFoldAngles[e]),
  }));

  // Stage-2 Short-Hop Radial Dijkstra step costs avoiding unfold/flip creases.
  const effFinal = new Float64Array(numEdges);
  const shapingHopCosts = new Float64Array(numEdges);
  const s2Active = [];
  const nonFlatVerts = new Set();
  const vAdj = new Map();

  for (let e = 0; e < numEdges; e++) {
    const b = baseFoldAngles[e];
    const f = effectiveTargetAngle(b, foldAngles[e]);
    effFinal[e] = f;
    const d = f - b;
    const unfolds =
        Math.abs(b) >= 10 && (Math.abs(f) < Math.abs(b) - 10 || b * f < -10);
    shapingHopCosts[e] =
        (Math.abs(b) < 1 ? 1.0 : 1.15) + (unfolds ? 32.0 : 0) +
        Math.abs(d) / 90.0;
    if (Math.abs(d) < 1) continue;
    s2Active.push(e);
    const [u, v] = edges[e];
    if (!vAdj.has(u)) vAdj.set(u, []);
    if (!vAdj.has(v)) vAdj.set(v, []);
    vAdj.get(u).push([v, e]);
    vAdj.get(v).push([u, e]);
    const flipped = Math.abs(b) >= 1 && (b * f < -1 || Math.abs(d) > 5);
    const absSh = Math.abs(f);
    if (flipped || (absSh > 4 && absSh < 176)) {
      nonFlatVerts.add(u);
      nonFlatVerts.add(v);
    }
  }

  const coupled3d = new Set();
  const bfsV = [...nonFlatVerts];
  const seenV = new Set(nonFlatVerts);
  for (let h = 0; h < bfsV.length; h++) {
    for (const [nb, ei] of vAdj.get(bfsV[h]) ?? []) {
      coupled3d.add(ei);
      if (!seenV.has(nb)) {
        seenV.add(nb);
        bfsV.push(nb);
      }
    }
  }

  const hasAnyNamedSticks = regionMeta.some((rm) => rm.rawName.length > 0);
  const regPosPart = new Int32Array(regionMeta.length);
  const regNegPart = new Int32Array(regionMeta.length);
  if (!hasAnyNamedSticks) {
    for (const e of s2Active) {
      const b = baseFoldAngles[e];
      const f = effFinal[e];
      if (Math.abs(b) < 5 && Math.abs(f) >= 25 && Math.abs(f) <= 168) {
        for (const r of edgeRegions[e]) {
          if (regionMeta[r]?.kind === 'flap') {
            if (f > 0) regPosPart[r]++;
            else regNegPart[r]++;
          }
        }
      }
    }
  }
  const isFanRegion = (r) => {
    const rm = regionMeta[r];
    if (!rm || rm.kind !== 'flap') return false;
    if (rm.isNamedFan) return true;
    if (hasAnyNamedSticks) return false;
    return regPosPart[r] >= 4 && regNegPart[r] >= 4;
  };

  const tier1 = [];
  const tierPose = [];
  const tierFan = [];
  for (const e of s2Active) {
    if ([...edgeRegions[e]].some(isFanRegion)) {
      tierFan.push(e);
    } else if (!coupled3d.has(e)) {
      tier1.push(e);
    } else {
      tierPose.push(e);
    }
  }

  const edgeRegionName = (e, preferFan = false) => {
    if (preferFan) {
      for (const r of edgeRegions[e]) {
        if (isFanRegion(r)) return regionMeta[r].title;
      }
    }
    for (const r of edgeRegions[e]) {
      if (regionMeta[r] && regionMeta[r].kind === 'flap') {
        return regionMeta[r].title;
      }
    }
    return 'Spine & Trunk';
  };
  const edgeHop = (e) => {
    let h = Infinity;
    for (const r of edgeRegions[e]) {
      if (regionMeta[r] && regionMeta[r].kind === 'flap') {
        h = Math.min(h, regionMeta[r].hop);
      }
    }
    return Number.isFinite(h) ? h : 0;
  };

  const buildGroupedWaves = (list, maxWaves, preferFan) => {
    if (list.length === 0) return [];
    const byReg = new Map();
    for (const e of list) {
      const rn = edgeRegionName(e, preferFan);
      if (!byReg.has(rn)) byReg.set(rn, []);
      byReg.get(rn).push(e);
    }
    const waves = [...byReg.entries()]
                      .map(([name, es]) => ({
                             names: [name],
                             hop: Math.min(...es.map(edgeHop)),
                             edges: es,
                           }))
                      .sort(
                          (a, b) => a.hop - b.hop ||
                              a.names[0].localeCompare(b.names[0]));
    while (waves.length > maxWaves) {
      let best = 0;
      for (let i = 1; i < waves.length - 1; i++) {
        if (waves[i].edges.length + waves[i + 1].edges.length <
            waves[best].edges.length + waves[best + 1].edges.length) {
          best = i;
        }
      }
      waves.splice(best, 2, {
        names: [...new Set([...waves[best].names, ...waves[best + 1].names])],
        hop: Math.min(waves[best].hop, waves[best + 1].hop),
        edges: [...waves[best].edges, ...waves[best + 1].edges],
      });
    }
    return waves;
  };

  const emitWavePhases = (waves, tA, tB, actRoman, verb, overlapFrac, isFinal) => {
    if (waves.length === 0 || tB <= tA + 1e-5) return;
    const weights = waves.map((w) => Math.sqrt(Math.max(2, w.edges.length)));
    if (isFinal && weights.length >= 2) {
      weights[weights.length - 1] *= 1.3;
      weights[weights.length - 2] *= 1.15;
    }
    const wSum = weights.reduce((a, b) => a + b, 0) || 1;
    let cursor = tA;
    waves.forEach((w, idx) => {
      const seg = (tB - tA) * (weights[idx] / wSum);
      const wt0 = idx > 0 ? Math.max(tA, cursor - overlapFrac * seg) : tA;
      const wt1 = idx === waves.length - 1 ?
          tB :
          Math.min(tB, cursor + seg * (1 + 0.5 * overlapFrac));
      cursor += seg;
      const desc = w.names.slice(0, 2).join(' & ') +
          (w.names.length > 2 ? ` +${w.names.length - 2}` : '');
      phases.push({
        id: `s2_${actRoman}_${idx}`,
        act: actRoman,
        stageGroup: 'shaping',
        label: `Act ${actRoman}.${idx + 1} · ${verb}: ${desc}`,
        tStart: +wt0.toFixed(4),
        tEnd: +wt1.toFixed(4),
        easeOut: isFinal && idx === waves.length - 1 ? 0.9 : 0.28,
        power: 1.0,
        edgeIndices: w.edges,
        deltas: Float64Array.from(
            w.edges, (e) => effFinal[e] - baseFoldAngles[e]),
      });
    });
  };

  const t1Waves = buildGroupedWaves(tier1, 6, false);
  const poseWaves = buildGroupedWaves(tierPose, 8, false);
  const fanWaves = buildGroupedWaves(tierFan, 6, true);

  const tFinish = 0.990;
  let curT = tShapeStart;
  let standupT = tShapeStart;
  if (t1Waves.length > 0 && (poseWaves.length > 0 || fanWaves.length > 0)) {
    const tAct2End = curT + (tFinish - tShapeStart) * 0.18;
    emitWavePhases(t1Waves, curT, tAct2End - 0.006, 'II', 'Narrow Flap', 0.03, false);
    standupT = tAct2End;
    curT = tAct2End;
  } else if (t1Waves.length > 0) {
    emitWavePhases(t1Waves, curT, tFinish, 'II', 'Narrow Flap', 0.03, true);
    standupT = 1.0;
    curT = tFinish;
  }

  if (poseWaves.length > 0 && fanWaves.length > 0) {
    const poseEndT = Math.max(curT + 0.18, 0.82);
    emitWavePhases(poseWaves, curT, poseEndT - 0.006, 'III', 'Pose Limb', 0.06, false);
    emitWavePhases(fanWaves, poseEndT, tFinish, 'IV', 'Fan Out', 0.05, true);
  } else if (poseWaves.length > 0) {
    emitWavePhases(poseWaves, curT, tFinish, 'III', 'Pose Limb', 0.06, true);
  } else if (fanWaves.length > 0) {
    emitWavePhases(fanWaves, curT, tFinish, 'IV', 'Fan Out', 0.05, true);
  }

  let act1Keyframes = null;
  if (solveXpbd) {
    const guideFolder = new Folder(model);
    const s2Tree = buildShortHopDijkstra(model, rf, shapingHopCosts);
    guideFolder.adoptTree(s2Tree);
    const posBase =
        Float32Array.from(guideFolder.foldByHingesRaw(baseFoldAngles));
    const xpbdSys = buildSignedXPBDSystem(model);
    act1Keyframes = solveLiveSignedXPBD(
        model, posBase,
        (curAngles) => {
          guideFolder.adoptTree(s2Tree);
          return guideFolder.foldByHingesRaw(curAngles);
        },
        xpbdSys);
  }

  return {
    baseCompleteT: tBase,
    shapingStartT: tShapeStart,
    standupStartT: +standupT.toFixed(4),
    rootFace: rf,
    shapingHopCosts,
    act1Keyframes,
    phases,
  };
}

/**
 * Evaluates a piecewise-linear monotone velocity-equalization knot array.
 *
 * @param {number} u Normalized window progress in `[0, 1]`.
 * @param {?Float64Array|?Array<number>} knots Monotone knot values in `[0, 1]`.
 * @return {number}
 */
function evalPiecewiseTimeMap(u, knots) {
  if (!knots || knots.length < 2) return u;
  const c = Math.max(0, Math.min(1, u));
  const m = knots.length - 1;
  const x = c * m;
  const idx = Math.min(m - 1, Math.floor(x));
  const frac = x - idx;
  return knots[idx] + frac * (knots[idx + 1] - knots[idx]);
}

/**
 * Exact inverse of `evalPiecewiseTimeMap` on a strictly monotone knot array.
 *
 * @param {number} w Warped window progress in `[0, 1]`.
 * @param {?Float64Array|?Array<number>} knots Monotone knot values in `[0, 1]`.
 * @return {number}
 */
function invertPiecewiseTimeMap(w, knots) {
  if (!knots || knots.length < 2) return w;
  const m = knots.length - 1;
  if (w <= knots[0]) return 0;
  if (w >= knots[m]) return 1;
  let lo = 0;
  let hi = m;
  while (lo + 1 < hi) {
    const mid = (lo + hi) >> 1;
    if (knots[mid] <= w) lo = mid;
    else hi = mid;
  }
  const span = knots[hi] - knots[lo];
  const frac = span > 1e-12 ? (w - knots[lo]) / span : 0;
  return (lo + frac) / m;
}

/**
 * Maps external slider/playback progress `t in [0, 1]` through `s2TimeMap`
 * into internal schedule phase time `warpedT in [0, 1]`.
 *
 * @param {?Object} schedule Schedule object, or null.
 * @param {number} t Unwarped progress in `[0, 1]`.
 * @return {number}
 */
export function warpScheduleProgress(schedule, t) {
  const tRaw = Math.max(0, Math.min(1, t));
  if (!schedule?.s2TimeMap) return tRaw;
  const sT = schedule.shapingStartT ?? DEFAULT_SHAPING_START_T;
  if (tRaw > sT && tRaw < 1) {
    return sT +
        (1 - sT) *
            evalPiecewiseTimeMap((tRaw - sT) / (1 - sT), schedule.s2TimeMap);
  }
  return tRaw;
}

/**
 * Exact inverse of `warpScheduleProgress`, mapping internal schedule phase time
 * `warpedT in [0, 1]` back to external slider/playback progress `t in [0, 1]`.
 *
 * @param {?Object} schedule Schedule object, or null.
 * @param {number} warpedT Internal schedule time in `[0, 1]`.
 * @return {number}
 */
export function unwarpScheduleProgress(schedule, warpedT) {
  const tClamped = Math.max(0, Math.min(1, warpedT));
  if (!schedule?.s2TimeMap) return tClamped;
  const sT = schedule.shapingStartT ?? DEFAULT_SHAPING_START_T;
  if (tClamped > sT && tClamped < 1) {
    return sT +
        (1 - sT) *
            invertPiecewiseTimeMap(
                (tClamped - sT) / (1 - sT), schedule.s2TimeMap);
  }
  return tClamped;
}

/**
 * Evaluates a schedule at progress `t in [0, 1]`, returning per-edge angles,
 * warped schedule time `warpedT`, active & folded edge sets, a phase-transition
 * `diagramKey`, dominant phase index, and HUD caption.
 *
 * @param {!Object} schedule Schedule from `hydrateSchedule` or
 *     `buildFoldingSchedule`.
 * @param {number} numEdges Total number of edges in `model`.
 * @param {number} t Schedule progress in `[0, 1]`.
 * @param {?Float64Array=} outAngles Optional preallocated `(numEdges)` buffer.
 * @return {{
 *   angles: !Float64Array,
 *   warpedT: number,
 *   label: string,
 *   activePhaseIndex: number,
 *   activeEdges: !Set<number>,
 *   foldedEdges: !Set<number>,
 *   diagramKey: string,
 *   stage2Progress: number,
 * }}
 */
export function evaluateScheduleAt(schedule, numEdges, t, outAngles = null) {
  const tClamped = warpScheduleProgress(schedule, t);

  const angles = outAngles ? outAngles.fill(0) : new Float64Array(numEdges);
  const activeEdges = new Set();
  const foldedEdges = new Set();
  const activePhases = [];
  const foldedPhases = [];
  let bestWeight = -1;
  let activePhaseIndex = -1;
  let bestLabel = '';

  schedule.phases.forEach((ph, idx) => {
    if (tClamped <= ph.tStart) return;
    const span = ph.tEnd - ph.tStart;
    let u = 1;
    if (tClamped < ph.tEnd && span > 1e-9) {
      u = easePhaseProgress(
          (tClamped - ph.tStart) / span, ph.easeOut ?? 0, ph.power ?? 1);
      if (u > 0.01 && u < 0.99) {
        activePhases.push(idx);
        for (const e of ph.edgeIndices) activeEdges.add(e);
        const w = 1 - Math.abs(u - 0.5);
        if (w > bestWeight) {
          bestWeight = w;
          activePhaseIndex = idx;
          bestLabel = ph.label;
        }
      } else if (u >= 0.99) {
        foldedPhases.push(idx);
        for (const e of ph.edgeIndices) foldedEdges.add(e);
      }
    } else {
      foldedPhases.push(idx);
      for (const e of ph.edgeIndices) foldedEdges.add(e);
    }
    const deltas = ph.deltas;
    if (deltas) {
      for (let i = 0; i < ph.edgeIndices.length; i++) {
        angles[ph.edgeIndices[i]] += u * deltas[i];
      }
    }
  });

  for (const e of activeEdges) foldedEdges.delete(e);

  const {
    baseCompleteT = DEFAULT_BASE_COMPLETE_T,
    shapingStartT = DEFAULT_SHAPING_START_T,
    standupStartT = DEFAULT_STANDUP_START_T,
  } = schedule;
  let label = bestLabel;
  if (tClamped <= 1e-4) {
    label = 'Flat Uncut Crease Pattern';
  } else if (tClamped >= baseCompleteT && tClamped <= shapingStartT + 1e-4) {
    label = 'Act I Complete · Razor-Flat Box-Pleated Base';
  } else if (
      standupStartT > shapingStartT + 0.02 &&
      Math.abs(tClamped - standupStartT) <= 0.008) {
    label = 'Act II Complete · Narrowed Flaps Ready for 3D Pose';
  } else if (tClamped >= 1 - 1e-4) {
    label = 'Final 3D Sculpted Origami';
  } else if (!label) {
    label = tClamped < baseCompleteT ?
        'Act I · Box-Pleated Base Collapse' :
        'Acts II–IV · 3D Shaping & Fan-Out';
  }

  const diagramKey = tClamped <= 1e-4 ?
      'flat' :
      (tClamped >= 1 - 1e-4 ?
           'folded' :
           `a:${activePhases.join(',')}|f:${foldedPhases.join(',')}`);

  const stage2Progress = tClamped <= shapingStartT ?
      0 :
      (shapingStartT >= 1 - 1e-6 ?
           1 :
           Math.max(
               0, Math.min(1, (tClamped - shapingStartT) / (1 - shapingStartT))));

  return {
    angles,
    warpedT: tClamped,
    label,
    activePhaseIndex,
    activeEdges,
    foldedEdges,
    diagramKey,
    stage2Progress,
  };
}
