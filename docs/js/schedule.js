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
 * @fileoverview Multi-stage axial-first collapse and anatomical 3D shaping
 * schedules.
 *
 * Partitions folding into four physical acts:
 *   - Act I (Collapse): Coupled horizontal and vertical axial creases with
 *     45-degree diagonal ridges, collapsing progressively outward from the
 *     spine to the flat box-pleated base.
 *   - Act II (Flat-Clip): Pure 180-degree narrowing/clip folds on the table.
 *   - Act III (3D Pose): Skeletal river bends, limb-root hinges, and coupled
 *     crimps ordered outward from the trunk, one bilateral region at a time.
 *   - Act IV (3D Sculpt): Volumetric feather fans, squashes, and tip details.
 *
 * Free of DOM and three.js dependencies, so it also runs under Node.
 */

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
 * Evaluates C2 phase progress with optional final hinge deceleration.
 *
 * @param {number} x Normalized phase progress in `[0, 1]`.
 * @param {number=} easeOut Deceleration strength in `[0, 1]`.
 * @return {number}
 */
export function easePhaseProgress(x, easeOut = 0) {
  const c = Math.max(0, Math.min(1, x));
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
 * Builds a 4-Act multi-stage folding schedule directly from a parsed `Model`.
 *
 * @param {!Object} model Parsed `Model` from `fold.js`.
 * @param {?Object=} stickFigure Optional stick figure JSON.
 * @return {!Object}
 */
export function buildFoldingSchedule(model, stickFigure = null) {
  const {
    flat,
    faces,
    edges,
    assignments,
    foldAngles,
    baseFoldAngles,
    regions,
    faceRegions,
    gridSize,
  } = model;
  const numEdges = edges.length;
  const hops = stickHopDistances(stickFigure);
  const regionMeta = regions.map((r, idx) => {
    const minHop = Math.min(
        ...(r.sticks ?? []).map((s) => hops.get(s) ?? 1), 1);
    const isRiver = r.kind === 'river';
    return {
      id: idx,
      kind: r.kind,
      hop: isRiver && minHop === 0 ? 0 : minHop,
      title: cleanRegionTitle(r),
    };
  });

  // Map edges to incident regions.
  const edgeRegions = Array.from({length: numEdges}, () => new Set());
  const edgeByKey = new Map();
  const nV = flat.length / 2;
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

  const tBase = DEFAULT_BASE_COMPLETE_T;
  const tShapeStart = DEFAULT_SHAPING_START_T;
  const s1Edges = [];
  const roles = new Array(numEdges).fill('shaping');

  for (let e = 0; e < numEdges; e++) {
    if (assignments[e] === 'B') {
      roles[e] = 'border';
      continue;
    }
    if (Math.abs(baseFoldAngles[e]) < 1) continue;
    s1Edges.push(e);
    const [u, v] = edges[e];
    const dx = Math.abs(flat[2 * v] - flat[2 * u]);
    const dy = Math.abs(flat[2 * v + 1] - flat[2 * u + 1]);
    const len = Math.hypot(dx, dy) || 1;
    const ux = dx / len;
    const uy = dy / len;
    if (Math.abs(ux - uy) < 0.35 && Math.min(ux, uy) > 0.25) {
      roles[e] = 'ridge';
    } else if (edgeRegions[e].size > 1) {
      roles[e] = 'hinge';
    } else {
      roles[e] = 'axial';
    }
  }

  // If the file has no separate base/shaping split, build a single-act schedule.
  const hasS2 = foldAngles.some((a, e) => Math.abs(a - baseFoldAngles[e]) >= 1);
  if (s1Edges.length === 0 || !hasS2) {
    const active = [];
    const deltas = [];
    for (let e = 0; e < numEdges; e++) {
      if (Math.abs(foldAngles[e]) >= 1) {
        active.push(e);
        deltas.push(foldAngles[e]);
      }
    }
    return {
      baseCompleteT: 1.0,
      shapingStartT: 1.0,
      standupStartT: 0.0,
      phases: [{
        id: 'fold_all',
        act: 'I',
        stageGroup: 'collapse',
        label: 'Act I.1 · Fold Crease Pattern',
        tStart: 0.0,
        tEnd: 1.0,
        easeOut: 0.6,
        edgeIndices: active,
        deltas: Float64Array.from(deltas),
      }],
    };
  }

  const phases = [];
  const remaining = Float64Array.from(baseFoldAngles);

  // Couple collinear & opposite-sign M+V axial creases and bucket by region.
  const gridUnit = 1 / Math.max(gridSize || 28, 12);
  const uf = s1Edges.map((_, i) => i);
  const find = (x) => {
    while (uf[x] !== x) {
      uf[x] = uf[uf[x]];
      x = uf[x];
    }
    return x;
  };
  const union = (a, b) => {
    const ra = find(a);
    const rb = find(b);
    if (ra !== rb) uf[rb] = ra;
  };

  const hAxials = [];
  const vAxials = [];
  s1Edges.forEach((e, k) => {
    if (roles[e] === 'ridge') return;
    const [u, v] = edges[e];
    const x1 = flat[2 * u];
    const y1 = flat[2 * u + 1];
    const x2 = flat[2 * v];
    const y2 = flat[2 * v + 1];
    const sgn = remaining[e] >= 0 ? 1 : -1;
    if (Math.abs(y2 - y1) <= 1e-3 && Math.abs(x2 - x1) > 1e-4) {
      hAxials.push([k, 0.5 * (y1 + y2), Math.min(x1, x2), Math.max(x1, x2), sgn]);
    } else if (Math.abs(x2 - x1) <= 1e-3 && Math.abs(y2 - y1) > 1e-4) {
      vAxials.push([k, 0.5 * (x1 + x2), Math.min(y1, y2), Math.max(y1, y2), sgn]);
    }
  });

  const coupleAxials = (list) => {
    for (let i = 0; i < list.length; i++) {
      const [ki, ci, loI, hiI, si] = list[i];
      for (let j = i + 1; j < list.length; j++) {
        const [kj, cj, loJ, hiJ, sj] = list[j];
        if (si === sj && Math.abs(ci - cj) <= 0.15 * gridUnit &&
            Math.min(hiI, hiJ) - Math.max(loI, loJ) >= -0.25 * gridUnit) {
          union(ki, kj);
        }
      }
    }
    const paired = new Set();
    for (let i = 0; i < list.length; i++) {
      const [ki, ci, loI, hiI, si] = list[i];
      if (paired.has(ki)) continue;
      let bestJ = -1;
      let bestDc = 2.15 * gridUnit;
      for (let j = 0; j < list.length; j++) {
        const [, cj, loJ, hiJ, sj] = list[j];
        if (si * sj >= 0) continue;
        const dc = Math.abs(ci - cj);
        if (dc >= 0.4 * gridUnit && dc < bestDc) {
          const ov = Math.min(hiI, hiJ) - Math.max(loI, loJ);
          if (ov >= 0.35 * Math.min(hiI - loI, hiJ - loJ)) {
            bestDc = dc;
            bestJ = j;
          }
        }
      }
      if (bestJ >= 0) {
        const kj = list[bestJ][0];
        union(ki, kj);
        paired.add(ki);
        paired.add(kj);
      }
    }
  };
  coupleAxials(hAxials);
  coupleAxials(vAxials);

  const packets = new Map();
  s1Edges.forEach((e, k) => {
    const r = find(k);
    if (!packets.has(r)) packets.set(r, []);
    packets.get(r).push(e);
  });

  const spineEdges = [];
  const flapGroups = new Map();
  for (const pkt of packets.values()) {
    let best = null;
    for (const e of pkt) {
      for (const rid of edgeRegions[e]) {
        const rm = regionMeta[rid];
        if (!rm) continue;
        const isRiv = rm.hop === 0 || rm.kind === 'river';
        const cand = {hop: isRiv ? 0 : rm.hop, isRiv, title: rm.title};
        if (!best || cand.hop < best.hop ||
            (cand.hop === best.hop && cand.isRiv && !best.isRiv)) {
          best = cand;
        }
      }
    }
    if (!best || best.isRiv) {
      spineEdges.push(...pkt);
    } else {
      const key = `${best.hop}|${best.title}`;
      if (!flapGroups.has(key)) {
        flapGroups.set(key, {hop: best.hop, title: best.title, edges: []});
      }
      flapGroups.get(key).edges.push(...pkt);
    }
  }

  const orderedS1 = [...flapGroups.values()].sort(
      (a, b) => a.hop - b.hop || b.edges.length - a.edges.length ||
          a.title.localeCompare(b.title));
  const buckets = [];
  if (spineEdges.length > 0) {
    buckets.push({hop: 0, names: ['Spine & Rivers'], edges: spineEdges});
  }
  for (const g of orderedS1) {
    const last = buckets.at(-1);
    if (last && last.hop > 0 &&
        ((last.hop === g.hop && (last.edges.length < 18 || g.edges.length < 18)) ||
         buckets.length >= 7)) {
      if (!last.names.includes(g.title)) last.names.push(g.title);
      last.edges.push(...g.edges);
    } else {
      buckets.push({hop: g.hop, names: [g.title], edges: [...g.edges]});
    }
  }

  // Keep axial creases and 45-degree ridges coupled inside each bucket and
  // overlap buckets across Act I so spanning-tree hinge folding collapses
  // degree-4 junctions synchronously without rolling the sheet.
  const tActiveEnd = tBase * 0.985;
  buckets.forEach((b, idx) => {
    const frac = buckets.length > 1 ? idx / (buckets.length - 1) : 0;
    const t0 = 0.12 * frac * tActiveEnd;
    const t1 = (0.88 + 0.12 * frac) * tActiveEnd;
    const ease = idx === buckets.length - 1 ? 0.85 : 0.45;
    const desc = b.names.slice(0, 2).join(', ') +
        (b.names.length > 2 ? ` +${b.names.length - 2}` : '');
    const beat = idx + 1;
    const title = b.hop === 0 ?
        `Act I.${beat} · Collapse ${desc}` :
        `Act I.${beat} · Collapse Flaps: ${desc}`;
    phases.push({
      id: `s1_b${idx}`,
      act: 'I',
      stageGroup: 'collapse',
      label: title,
      tStart: +t0.toFixed(4),
      tEnd: +t1.toFixed(4),
      easeOut: ease,
      edgeIndices: b.edges,
      deltas: Float64Array.from(b.edges, (e) => remaining[e]),
    });
  });

  // Classify Stage-2 shaping edges into Act II (flat 180 clip), Act III (pose),
  // and Act IV (volumetric sculpt).
  const s2Active = [];
  const nonFlatVerts = new Set();
  const flippedVerts = new Set();
  const vAdj = new Map();
  for (let e = 0; e < numEdges; e++) {
    const d = foldAngles[e] - baseFoldAngles[e];
    if (Math.abs(d) < 1) continue;
    s2Active.push(e);
    const [u, v] = edges[e];
    if (!vAdj.has(u)) vAdj.set(u, []);
    if (!vAdj.has(v)) vAdj.set(v, []);
    vAdj.get(u).push([v, e]);
    vAdj.get(v).push([u, e]);
    const flipped = Math.abs(baseFoldAngles[e]) >= 1 &&
        (baseFoldAngles[e] * foldAngles[e] < -1 || Math.abs(d) > 5);
    const absSh = Math.abs(foldAngles[e]);
    if (flipped) {
      flippedVerts.add(u);
      flippedVerts.add(v);
      nonFlatVerts.add(u);
      nonFlatVerts.add(v);
    } else if (absSh > 4 && absSh < 176) {
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

  const tier1 = [];
  const tier2 = [];
  const tier3 = [];
  for (const e of s2Active) {
    if (!coupled3d.has(e)) {
      tier1.push(e);
      continue;
    }
    const [u, v] = edges[e];
    const rids = [...edgeRegions[e]];
    const touchesRiver = rids.some((r) => regionMeta[r]?.kind === 'river');
    if (flippedVerts.has(u) || flippedVerts.has(v) || touchesRiver ||
        rids.length >= 2) {
      tier2.push(e);
    } else {
      tier3.push(e);
    }
  }

  const edgeRegionName = (e) => {
    for (const r of edgeRegions[e]) {
      if (regionMeta[r] && regionMeta[r].kind !== 'river') {
        return regionMeta[r].title;
      }
    }
    return 'Spine & Trunk';
  };
  const edgeHop = (e) => {
    let h = 1;
    for (const r of edgeRegions[e]) {
      if (regionMeta[r]) h = Math.min(h, regionMeta[r].hop);
    }
    return h;
  };

  const buildActWaves = (list, tA, tB, actRoman, verb, maxWaves, isFinal) => {
    if (list.length === 0) return;
    const byReg = new Map();
    for (const e of list) {
      const rn = edgeRegionName(e);
      if (!byReg.has(rn)) byReg.set(rn, []);
      byReg.get(rn).push(e);
    }
    let waves = [...byReg.entries()]
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
        names: [...waves[best].names, ...waves[best + 1].names],
        hop: Math.min(waves[best].hop, waves[best + 1].hop),
        edges: [...waves[best].edges, ...waves[best + 1].edges],
      });
    }
    const weights = waves.map((w) => Math.sqrt(Math.max(2, w.edges.length)));
    if (isFinal && weights.length >= 2) {
      weights[weights.length - 1] *= 1.35;
      weights[weights.length - 2] *= 1.18;
    }
    const wSum = weights.reduce((a, b) => a + b, 0) || 1;
    let cursor = tA;
    waves.forEach((w, idx) => {
      const seg = (tB - tA) * (weights[idx] / wSum);
      const wt0 = idx > 0 ? Math.max(tA, cursor - 0.03 * seg) : tA;
      const wt1 = idx === waves.length - 1 ? tB : Math.min(tB, cursor + seg * 1.02);
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
        easeOut: isFinal && idx === waves.length - 1 ? 0.9 : 0.25,
        edgeIndices: w.edges,
        deltas: Float64Array.from(
            w.edges, (e) => foldAngles[e] - baseFoldAngles[e]),
      });
    });
  };

  const tFinish = 0.988;
  let curT = tShapeStart;
  let standupT = tShapeStart;
  const totS2 = Math.max(1, tier1.length + tier2.length + tier3.length);
  if (tier1.length > 0 && (tier2.length > 0 || tier3.length > 0)) {
    const frac1 = Math.max(0.15, Math.min(0.26, tier1.length / totS2 + 0.06));
    const tAct2End = curT + (tFinish - curT) * frac1;
    buildActWaves(tier1, curT, tAct2End - 0.008, 'II', 'Flat-Clip', 6, false);
    standupT = tAct2End;
    curT = tAct2End;
  } else if (tier1.length > 0) {
    buildActWaves(tier1, curT, tFinish, 'II', 'Flat-Clip', 8, true);
    standupT = 1.0;
    curT = tFinish;
  }

  if (tier2.length > 0 && tier3.length > 0) {
    const frac2 = Math.max(
        0.40,
        Math.min(0.56, tier2.length / Math.max(1, tier2.length + tier3.length)));
    const tAct3End = curT + (tFinish - curT) * frac2;
    buildActWaves(tier2, curT, tAct3End, 'III', '3D Pose', 8, false);
    buildActWaves(
        tier3, Math.max(curT, tAct3End - 0.006), tFinish, 'IV', '3D Sculpt', 8,
        true);
  } else if (tier2.length > 0) {
    buildActWaves(tier2, curT, tFinish, 'III', '3D Pose', 10, true);
  } else if (tier3.length > 0) {
    buildActWaves(tier3, curT, tFinish, 'IV', '3D Sculpt', 10, true);
  }

  return {
    baseCompleteT: tBase,
    shapingStartT: tShapeStart,
    standupStartT: +standupT.toFixed(4),
    phases,
  };
}

/**
 * Evaluates a schedule at progress `t in [0, 1]`, returning per-edge angles,
 * active & folded edge sets, a phase-transition `diagramKey`, dominant phase
 * index, and HUD caption.
 *
 * @param {!Object} schedule Schedule built by `buildFoldingSchedule`.
 * @param {number} numEdges Total number of edges in `model`.
 * @param {number} t Schedule progress in `[0, 1]`.
 * @param {?Float64Array=} outAngles Optional preallocated `(numEdges)` buffer.
 * @return {{
 *   angles: !Float64Array,
 *   label: string,
 *   activePhaseIndex: number,
 *   activeEdges: !Set<number>,
 *   foldedEdges: !Set<number>,
 *   diagramKey: string,
 *   stage2Progress: number
 * }}
 */
export function evaluateScheduleAt(schedule, numEdges, t, outAngles = null) {
  const tClamped = Math.max(0, Math.min(1, t));
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
      u = easePhaseProgress((tClamped - ph.tStart) / span, ph.easeOut ?? 0);
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
    label = 'Act II Complete · Flat-Clipped Base Ready for 3D Pose';
  } else if (tClamped >= 1 - 1e-4) {
    label = 'Final 3D Sculpted Origami';
  } else if (!label) {
    label = tClamped < baseCompleteT ?
        'Act I · Box-Pleated Base Collapse' :
        'Acts II–IV · 3D Shaping';
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
           Math.max(0, Math.min(1, (tClamped - shapingStartT) / (1 - shapingStartT))));

  return {
    angles,
    label,
    activePhaseIndex,
    activeEdges,
    foldedEdges,
    diagramKey,
    stage2Progress,
  };
}
