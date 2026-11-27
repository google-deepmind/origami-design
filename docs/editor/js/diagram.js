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
 * @fileoverview Draws crease patterns, packings and stick figures as SVG.
 *
 * Everything is built with DOM calls and explicit presentation attributes
 * rather than markup strings and CSS, so the drawings are safe for untrusted
 * files and stay correct when serialized to a standalone `.svg`.
 */

const SVG_NS = 'http://www.w3.org/2000/svg';

/** Stroke per FOLD edge assignment; anything else is drawn as a flat crease. */
const CREASE_STROKES = {
  B: '#111827',
  M: '#dc2626',
  V: '#2563eb'
};
const FLAT_CREASE_STROKE = '#9ca3af';
const PAPER_FILL = '#fffdf7';
/** Fill per kind of packed region; `style.css` repeats these in its legend. */
const REGION_FILLS = {
  flap: '#ead9b5',
  river: '#b9d7ea',
  reserved: '#d8d4cc'
};
const HIGHLIGHT_FILL = '#fcd34d';

/**
 * Creates an SVG element with the given attributes.
 *
 * @param {string} name Tag name in the SVG namespace.
 * @param {!Object<string, *>=} attributes Presentation or geometry attributes.
 * @return {!SVGElement}
 */
function svgElement(name, attributes = {}) {
  const element = document.createElementNS(SVG_NS, name);
  for (const [key, value] of Object.entries(attributes)) {
    element.setAttribute(key, value);
  }
  return element;
}

/**
 * Starts a drawing of a model's flat sheet, replacing the content of `svg`.
 *
 * FOLD frames have y pointing up, so the drawing flips the y axis of SVG.
 *
 * @param {!SVGSVGElement} svg
 * @param {!Object} model Parsed model from `fold.js`.
 * @param {?{minX: number, minY: number, maxX: number, maxY: number}=} focusBox
 *     Optional zoomed viewport rectangle in flat-sheet coordinates.
 * @return {!SVGGElement} A group in the coordinates of the model's flat
 *     frame, holding the blank paper.
 */
function startSheet(svg, {flat}, focusBox = null) {
  let [minX, minY, maxX, maxY] = [Infinity, Infinity, -Infinity, -Infinity];
  for (let i = 0; i < flat.length; i += 2) {
    minX = Math.min(minX, flat[i]);
    maxX = Math.max(maxX, flat[i]);
    minY = Math.min(minY, flat[i + 1]);
    maxY = Math.max(maxY, flat[i + 1]);
  }
  const width = maxX - minX;
  const height = maxY - minY;
  const margin = 0.01 * Math.max(width, height);
  let viewBox =
      [minX - margin, -maxY - margin, width + 2 * margin, height + 2 * margin];
  if (focusBox) {
    const fw = Math.max(1e-4, focusBox.maxX - focusBox.minX);
    const fh = Math.max(1e-4, focusBox.maxY - focusBox.minY);
    viewBox = [focusBox.minX, -focusBox.maxY, fw, fh];
  }
  svg.setAttribute('viewBox', viewBox.join(' '));
  const sheet = svgElement(
      'g', {transform: 'scale(1 -1)', fill: 'none', 'stroke-linecap': 'round'});
  sheet.append(
      svgElement('rect', {x: minX, y: minY, width, height, fill: PAPER_FILL}));
  svg.replaceChildren(sheet);
  return sheet;
}

/**
 * Returns a function that formats a vertex of `flat` for SVG path data.
 *
 * @param {!Float64Array} flat Flat x, y coordinates per vertex.
 * @return {function(number): string}
 */
const pathPoint = (flat) => (v) => `${flat[2 * v]} ${flat[2 * v + 1]}`;

/**
 * A filled path covering some faces of a model.
 *
 * @param {!Object} model Parsed model from `fold.js`.
 * @param {!Array<!Array<number>>} faces Vertex indices per face.
 * @param {string} fill
 * @return {!SVGPathElement}
 */
function facesPath({flat}, faces, fill) {
  const point = pathPoint(flat);
  return svgElement('path', {
    d: faces.map((face) => `M${face.map(point).join('L')}Z`).join(''),
    fill,
    'shape-rendering': 'crispEdges',
  });
}

/**
 * A stroked path along some edges of a model.
 *
 * @param {!Object} model Parsed model from `fold.js`.
 * @param {!Array<!Array<number>>} edges Vertex index pairs.
 * @param {string} stroke
 * @param {number} width Line width in screen pixels.
 * @param {number=} opacity Optional stroke opacity.
 * @return {!SVGPathElement}
 */
function edgesPath({flat}, edges, stroke, width, opacity = 1) {
  const point = pathPoint(flat);
  const attrs = {
    d: edges.map(([u, v]) => `M${point(u)}L${point(v)}`).join(''),
    stroke,
    'stroke-width': width,
    'vector-effect': 'non-scaling-stroke',
  };
  if (opacity < 1) attrs['stroke-opacity'] = opacity;
  return svgElement('path', attrs);
}

/**
 * Checks whether region `r` is included in `highlight`.
 *
 * @param {?number|!Array<number>|!Set<number>} highlight
 * @param {number} r
 * @return {boolean}
 */
function isHighlighted(highlight, r) {
  if (highlight === null || highlight === undefined) return false;
  if (typeof highlight === 'number') return r === highlight;
  if (highlight instanceof Set) return highlight.has(r);
  if (Array.isArray(highlight)) return highlight.includes(r);
  return false;
}

/**
 * The faces of a model that belong to the given region or set of regions.
 *
 * @param {!Object} model Parsed model from `fold.js`.
 * @param {number|!Array<number>|!Set<number>} region Index/indices into
 *     `model.regions`.
 * @return {!Array<!Array<number>>}
 */
function regionFaces({faces, faceRegions}, region) {
  return faces.filter((_, f) => isHighlighted(region, faceRegions[f]));
}

/**
 * The outline of a set of faces: the edges that lie on exactly one of them.
 *
 * @param {!Array<!Array<number>>} faces Vertex indices per face.
 * @return {!Array<!Array<number>>} Vertex index pairs.
 */
function outlineEdges(faces) {
  const outline = new Map();
  for (const face of faces) {
    face.forEach((u, i) => {
      const v = face[(i + 1) % face.length];
      const key = u < v ? `${u} ${v}` : `${v} ${u}`;
      if (!outline.delete(key)) outline.set(key, [u, v]);
    });
  }
  return [...outline.values()];
}

/**
 * Draws a model's crease pattern, replacing the content of `svg`.
 *
 * @param {!SVGSVGElement} svg
 * @param {!Object} model Parsed model from `fold.js`.
 * @param {!Object=} options
 */
export function drawCreasePattern(
    svg, model,
    {baseOnly = false, highlight = null, highlightFaces = null,
     focusFaces = null, activeEdges = null, foldedEdges = null,
     userShaping = null, subFlaps = null} = {}) {
  let focusBox = null;
  if (focusFaces && focusFaces.size > 0) {
    let [sMinX, sMinY, sMaxX, sMaxY] = [Infinity, Infinity, -Infinity, -Infinity];
    for (let i = 0; i < model.flat.length; i += 2) {
      sMinX = Math.min(sMinX, model.flat[i]);
      sMaxX = Math.max(sMaxX, model.flat[i]);
      sMinY = Math.min(sMinY, model.flat[i + 1]);
      sMaxY = Math.max(sMaxY, model.flat[i + 1]);
    }
    let [fMinX, fMinY, fMaxX, fMaxY] = [Infinity, Infinity, -Infinity, -Infinity];
    for (const fi of focusFaces) {
      for (const v of model.faces[fi] ?? []) {
        const vx = model.flat[2 * v];
        const vy = model.flat[2 * v + 1];
        fMinX = Math.min(fMinX, vx);
        fMaxX = Math.max(fMaxX, vx);
        fMinY = Math.min(fMinY, vy);
        fMaxY = Math.max(fMaxY, vy);
      }
    }
    if (Number.isFinite(fMinX)) {
      const sheetSpan = Math.max(1e-4, Math.max(sMaxX - sMinX, sMaxY - sMinY));
      const cx = 0.5 * (fMinX + fMaxX);
      const cy = 0.5 * (fMinY + fMaxY);
      const span = Math.min(
          sheetSpan * 1.02,
          Math.max(sheetSpan * 0.24, 1.32 * Math.max(fMaxX - fMinX, fMaxY - fMinY)));
      const half = 0.5 * span;
      const margin = 0.015 * sheetSpan;
      let minX = cx - half;
      let maxX = cx + half;
      let minY = cy - half;
      let maxY = cy + half;
      if (minX < sMinX - margin) {
        maxX += (sMinX - margin) - minX;
        minX = sMinX - margin;
      }
      if (maxX > sMaxX + margin) {
        minX -= maxX - (sMaxX + margin);
        maxX = sMaxX + margin;
      }
      if (minY < sMinY - margin) {
        maxY += (sMinY - margin) - minY;
        minY = sMinY - margin;
      }
      if (maxY > sMaxY + margin) {
        minY -= maxY - (sMaxY + margin);
        maxY = sMaxY + margin;
      }
      focusBox = {
        minX: Math.max(sMinX - margin, minX),
        maxX: Math.min(sMaxX + margin, maxX),
        minY: Math.max(sMinY - margin, minY),
        maxY: Math.min(sMaxY + margin, maxY),
      };
    }
  }

  const sheet = startSheet(svg, model, focusBox);
  const selectedVerts = new Set();
  if (highlightFaces && highlightFaces.size > 0) {
    const pickedFaces = [];
    model.faces.forEach((face, f) => {
      if (highlightFaces.has(f)) {
        pickedFaces.push(face);
        for (const v of face) selectedVerts.add(v);
      }
    });
    if (pickedFaces.length > 0) {
      sheet.append(facesPath(model, pickedFaces, HIGHLIGHT_FILL));
    }
  } else if (highlight !== null) {
    const rFaces = regionFaces(model, highlight);
    for (const face of rFaces) {
      for (const v of face) selectedVerts.add(v);
    }
    sheet.append(facesPath(model, rFaces, HIGHLIGHT_FILL));
  }

  const hasSelection = selectedVerts.size > 0;
  const showLiveStage = !baseOnly && activeEdges && activeEdges.size > 0;
  const edges = {flat: [], B: [], M: [], V: []};
  const focusEdges = {M: [], V: []};
  const userAdded = {M: [], V: []};
  const upcoming = {M: [], V: []};
  const active = {M: [], V: []};

  model.edges.forEach((edge, e) => {
    let assignment = model.assignments[e];
    if (baseOnly) {
      if (model.baseFoldAngles) {
        const baseAngle = model.baseFoldAngles[e];
        if (assignment !== 'B') {
          assignment = baseAngle < -1 ? 'M' : (baseAngle > 1 ? 'V' : 'F');
          if (assignment === 'F' && model.shaping?.[e]) return;
        }
      } else if (model.shaping?.[e]) {
        return;
      }
    }
    if (!(assignment in CREASE_STROKES)) {
      edges.flat.push(edge);
    } else if (assignment === 'B') {
      edges.B.push(edge);
    } else if (!baseOnly && userShaping?.[e]) {
      userAdded[assignment].push(edge);
    } else if (showLiveStage) {
      if (activeEdges.has(e)) {
        active[assignment].push(edge);
      } else if (foldedEdges?.has(e)) {
        edges[assignment].push(edge);
      } else {
        upcoming[assignment].push(edge);
      }
    } else if (hasSelection &&
        selectedVerts.has(edge[0]) && selectedVerts.has(edge[1])) {
      focusEdges[assignment].push(edge);
    } else {
      edges[assignment].push(edge);
    }
  });

  sheet.append(edgesPath(
      model, edges.flat, FLAT_CREASE_STROKE, 0.5, hasSelection ? 0.35 : 1));
  if (showLiveStage) {
    sheet.append(edgesPath(model, upcoming.M, CREASE_STROKES.M, 0.65, 0.28));
    sheet.append(edgesPath(model, upcoming.V, CREASE_STROKES.V, 0.65, 0.28));
    sheet.append(edgesPath(model, edges.M, CREASE_STROKES.M, 1, 0.62));
    sheet.append(edgesPath(model, edges.V, CREASE_STROKES.V, 1, 0.62));
    sheet.append(edgesPath(model, edges.B, CREASE_STROKES.B, 1.15));
    sheet.append(edgesPath(model, active.M, CREASE_STROKES.M, 2.2));
    sheet.append(edgesPath(model, active.V, CREASE_STROKES.V, 2.2));
  } else if (hasSelection) {
    sheet.append(edgesPath(model, edges.M, CREASE_STROKES.M, 0.75, 0.26));
    sheet.append(edgesPath(model, edges.V, CREASE_STROKES.V, 0.75, 0.26));
    sheet.append(edgesPath(model, edges.B, CREASE_STROKES.B, 1.1));
    sheet.append(edgesPath(model, focusEdges.M, CREASE_STROKES.M, 1.25, 0.92));
    sheet.append(edgesPath(model, focusEdges.V, CREASE_STROKES.V, 1.25, 0.92));
  } else {
    for (const [assignment, stroke] of Object.entries(CREASE_STROKES)) {
      sheet.append(edgesPath(model, edges[assignment], stroke, 0.9));
    }
  }
  if (userAdded.M.length > 0 || userAdded.V.length > 0) {
    const allUser = [...userAdded.M, ...userAdded.V];
    sheet.append(edgesPath(model, allUser, '#ffffff', 5.0, 0.92));
    sheet.append(edgesPath(model, userAdded.M, '#dc2626', 3.0));
    sheet.append(edgesPath(model, userAdded.V, '#2563eb', 3.0));
  }
  if (subFlaps && subFlaps.length > 0) {
    for (const sf of subFlaps) {
      const faces = sf.faceIndices.map((fi) => model.faces[fi]).filter(Boolean);
      if (!faces.length) continue;
      const hit = facesPath(model, faces, 'transparent');
      hit.dataset.subflap = sf.id;
      const title = svgElement('title');
      title.textContent = `Click to shape: ${sf.label}`;
      hit.append(title);
      sheet.append(hit);
    }
  }
}

/**
 * Serializes a model's crease pattern to a standalone SVG document.
 *
 * @param {!Object} model Parsed model from `fold.js`.
 * @return {string}
 */
export function creasePatternSvg(model) {
  const svg = svgElement('svg');
  drawCreasePattern(svg, model);
  return new XMLSerializer().serializeToString(svg);
}

/**
 * Draws how a model's flaps and rivers tile the sheet, replacing the content
 * of `svg`.
 *
 * @param {!SVGSVGElement} svg
 * @param {!Object} model Parsed model from `fold.js`.
 * @param {?(number|!Set<number>)=} highlight Region or set to highlight.
 * @param {!Object=} options
 */
export function drawPacking(
    svg, model, highlight = null, {subFlaps = null} = {}) {
  const sheet = startSheet(svg, model);
  const outlines = model.edges.filter((_, e) => model.assignments[e] === 'B');
  const subDividers = [];

  if (subFlaps && subFlaps.length > 0) {
    for (const sf of subFlaps) {
      const faces = sf.faceIndices.map((fi) => model.faces[fi]).filter(Boolean);
      if (!faces.length) continue;
      const fill = isHighlighted(highlight, sf.id) ?
          HIGHLIGHT_FILL :
          REGION_FILLS[sf.kind] ?? PAPER_FILL;
      const area = facesPath(model, faces, fill);
      area.dataset.subflap = sf.id;
      const title = svgElement('title');
      title.textContent = sf.stickCount > 1 ?
          `${sf.label} (${sf.stickIndex + 1}/${sf.stickCount} of ${sf.chainSticks.join(' → ')})` :
          sf.label;
      area.append(title);
      sheet.append(area);
      if (sf.stickCount > 1) {
        subDividers.push(...outlineEdges(faces));
      }
    }
    model.regions.forEach((region, r) => {
      const rFaces = regionFaces(model, r);
      if (region.kind === 'reserved' && rFaces.length > 0) {
        const area = facesPath(model, rFaces, REGION_FILLS.reserved);
        const title = svgElement('title');
        title.textContent = 'Reserved';
        area.append(title);
        sheet.append(area);
      }
      outlines.push(...outlineEdges(rFaces));
    });
    if (subDividers.length > 0) {
      const dashed = edgesPath(model, subDividers, '#64748b', 0.75, 0.55);
      dashed.setAttribute('stroke-dasharray', '3 3');
      sheet.append(dashed);
    }
  } else {
    model.regions.forEach((region, r) => {
      const faces = regionFaces(model, r);
      const fill = (region.kind !== 'reserved' && isHighlighted(highlight, r)) ?
          HIGHLIGHT_FILL :
          REGION_FILLS[region.kind] ?? PAPER_FILL;
      const area = facesPath(model, faces, fill);
      if (region.kind !== 'reserved') {
        area.dataset.region = r;
      }
      sheet.append(area);
      outlines.push(...outlineEdges(faces));
    });
  }
  sheet.append(edgesPath(model, outlines, CREASE_STROKES.B, 1));
}

/**
 * Positions of the joints of a `bp_pipeline` stick figure.
 *
 * @param {!Object} stickFigure Parsed stick figure JSON.
 * @return {!Map<string, !Array<number>>}
 */
export function jointPositions(stickFigure) {
  const sticks = stickFigure.children;
  const root = sticks.find((s) => s.name === stickFigure.root) ?? sticks[0];
  const positions = new Map([[root.start, [0, 0, 0]]]);
  let placed;
  do {
    placed = false;
    for (const stick of sticks) {
      const known = positions.has(stick.start);
      if (known === positions.has(stick.end)) continue;
      const azimuth = stick.azimuth_angle * Math.PI / 180;
      const elevation = stick.elevation_angle * Math.PI / 180;
      const sign = known ? 1 : -1;
      const along = [
        Math.cos(elevation) * Math.sin(azimuth),
        Math.sin(elevation),
        Math.cos(elevation) * Math.cos(azimuth),
      ];
      const from = positions.get(known ? stick.start : stick.end);
      positions.set(
          known ? stick.end : stick.start,
          from.map((x, i) => x + sign * stick.length * along[i]));
      placed = true;
    }
  } while (placed);
  return positions;
}

/**
 * Draws a stick figure as seen by a camera, replacing the content of `svg`.
 *
 * @param {!SVGSVGElement} svg
 * @param {!Object} stickFigure Parsed stick figure JSON.
 * @param {!Array<number>} view Row-major 3x3 rotation from world axes to
 *     camera axes (x right, y up, z towards the viewer).
 */
export function drawStickFigure(svg, stickFigure, view) {
  const joints = jointPositions(stickFigure);
  const points = [...joints.values()];
  const center = [0, 1, 2].map(
      (i) => points.reduce((sum, p) => sum + p[i], 0) / points.length);
  const radius = Math.max(
      ...points.map((p) => Math.hypot(...p.map((x, i) => x - center[i]))));
  const project = (p) => [0, 1, 2].map(
      (row) => (row === 1 ? -1 : 1) *
          p.reduce(
              (sum, x, i) => sum + view[3 * row + i] * (x - center[i]), 0));

  const size = 1.15 * radius;
  svg.setAttribute('viewBox', [-size, -size, 2 * size, 2 * size].join(' '));
  const sticks = stickFigure.children
                     .filter((s) => joints.has(s.start) && joints.has(s.end))
                     .map((stick) => ({
                            stick,
                            start: project(joints.get(stick.start)),
                            end: project(joints.get(stick.end)),
                          }));
  sticks.sort((a, b) => a.start[2] + a.end[2] - b.start[2] - b.end[2]);
  const figure = svgElement('g', {
    stroke: 'currentColor',
    fill: 'currentColor',
    'stroke-linecap': 'round',
  });
  for (const {stick, start, end} of sticks) {
    const depth = (start[2] + end[2]) / (2 * radius);
    const line = svgElement('line', {
      x1: start[0],
      y1: start[1],
      x2: end[0],
      y2: end[1],
      'stroke-width': Math.max(0.02 * radius, 0.12 * stick.width),
      opacity: 0.6 + 0.3 * depth,
    });
    const title = svgElement('title');
    title.textContent = stick.name;
    line.append(title);
    figure.append(line);
  }
  for (const joint of points.map(project)) {
    figure.append(svgElement(
        'circle',
        {cx: joint[0], cy: joint[1], r: 0.03 * radius, stroke: 'none'}));
  }
  svg.replaceChildren(figure);
}
