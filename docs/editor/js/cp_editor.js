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
 * @fileoverview Interactive SVG Crease Pattern & Box-Pleating Packing Editor.
 *
 * Strictly gated by `this.active` so pointer/wheel listeners NEVER fire when
 * the shared `<svg id="diagram">` is displaying View-mode diagrams or the
 * Stick Figure editor.
 *
 * Combines:
 * - `edemaine/cp-editor` planar crease drawing, assignment painting, erasing,
 *   vertex dragging, fold-angle stepping, bilateral mirror drawing, and live
 *   Kawasaki/Maekawa vertex diagnostics.
 * - `bpstudio.abstreamace.com` + COrigami Packing Crease Pattern editor:
 *   renders the real Packing CP (mountain/valley ridges + green hinges +
 *   radius circles) in exact `(1 - x/G, 1 - y/G)` alignment with Studio's 3D
 *   stage, lets you drag flap circle-centers freely on the integer grid without
 *   snap-resetting, and shows live L_infinity tree-distance contacts/conflicts
 *   plus uncovered grid cells.
 */

import {
  DEFAULT_ASSIGNMENT_ANGLES,
  addCreaseSegment,
  analyzeVertices,
  cloneGraph,
  createBlankSquareGraph,
  normalizePlanarGraph,
  snapPoint,
  transformGraph,
} from './cp_geometry.js';

const SVG_NS = 'http://www.w3.org/2000/svg';

const STROKE_COLORS = {
  B: '#111827',
  M: '#dc2626',
  V: '#2563eb',
  H: '#16a34a',
  F: '#9ca3af',
  U: '#16a34a',
};

const REGION_FILLS = {
  flap: '#ead9b5',
  river: '#b9d7ea',
};

/**
 * @param {string} tag
 * @param {!Object<string, *>=} attrs
 * @return {!SVGElement}
 */
function svgEl(tag, attrs = {}) {
  const el = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs)) {
    el.setAttribute(k, String(v));
  }
  return el;
}

/**
 * @param {number} px
 * @param {number} py
 * @param {number} x1
 * @param {number} y1
 * @param {number} x2
 * @param {number} y2
 * @return {number}
 */
function pointToSegmentDist(px, py, x1, y1, x2, y2) {
  const dx = x2 - x1;
  const dy = y2 - y1;
  const lenSq = dx * dx + dy * dy;
  if (lenSq <= 1e-12) return Math.hypot(px - x1, py - y1);
  const t = Math.max(0, Math.min(1, ((px - x1) * dx + (py - y1) * dy) / lenSq));
  return Math.hypot(px - (x1 + t * dx), py - (y1 + t * dy));
}

export class CreasePatternEditor {
  /**
   * @param {!SVGSVGElement} svg
   * @param {!Object=} callbacks
   */
  constructor(svg, callbacks = {}) {
    /** @private @const {!SVGSVGElement} */
    this._svg = svg;
    /** @private @const {!Object} */
    this._cb = callbacks;

    /** @type {boolean} Whether this editor currently owns `#diagram` events. */
    this.active = false;
    /** @type {string} 'cp' | 'packing' */
    this.mode = 'cp';
    /** @type {string} 'select' | 'draw' | 'assign' | 'erase' | 'move' */
    this.tool = 'select';
    /** @type {string} 'M' | 'V' | 'F' | 'B' */
    this.activeAssignment = 'M';
    /** @type {string} 'grid' | 'half' | 'free' */
    this.snapMode = 'grid';
    /** @type {string} 'off' | '45' | '22.5' */
    this.angleLock = '45';
    /** @type {boolean} */
    this.mirrorDraw = false;
    /** @type {boolean} */
    this.showGrid = true;
    /** @type {boolean} */
    this.showViolations = false;
    /** @type {boolean} */
    this.autoRefold = true;

    /** @type {boolean} Packing CP layer toggles. */
    this.showPackingCreases = true;
    /** @type {boolean} */
    this.showPackingCircles = true;
    /** @type {boolean} */
    this.showPackingLabels = true;

    /** @private {!Object} */
    this._graph = createBlankSquareGraph(16);
    /** @private {?Object} */
    this._packing = null;
    /** @private {!Array<!Object>} */
    this._undoStack = [];
    /** @private {!Array<!Object>} */
    this._redoStack = [];
    /** @private {number} */
    this._selectedEdge = -1;
    /** @private {number} */
    this._hoveredEdge = -1;
    /** @private {number} */
    this._hoveredVertex = -1;
    /** @private {?Object} Hovered or dragged flap handle in packing mode. */
    this._hoveredFlap = null;
    /** @private {?Array<number>} */
    this._cursorSnap = null;
    /** @private {?Array<number>} */
    this._drawStart = null;
    /** @private {number} */
    this._dragVertex = -1;
    /** @private {?Object} */
    this._dragFlap = null;
    /** @private {boolean} */
    this._isPaintingAssign = false;

    /**
     * Viewport pan/zoom in FOLD sheet space (`x in [0,1]`, `y_svg = -y_fold in [-1,0]`).
     * @private {{cx: number, cy: number, span: number}}
     */
    this._camera = {cx: 0.5, cy: 0.5, span: 1.08};
    /** @private {?{startX: number, startY: number, cx: number, cy: number}} */
    this._panDrag = null;

    this._bindEvents();
  }

  /**
   * Enables or disables pointer/wheel handling on the shared SVG element.
   *
   * @param {boolean} enabled
   */
  setActive(enabled) {
    this.active = Boolean(enabled);
    if (!this.active) {
      this._panDrag = null;
      this._drawStart = null;
      this._dragVertex = -1;
      this._dragFlap = null;
      this._isPaintingAssign = false;
      this._cursorSnap = null;
      this._hoveredEdge = -1;
      this._hoveredVertex = -1;
      this._hoveredFlap = null;
    } else {
      this._svg.style.touchAction = 'none';
    }
  }

  /** @return {!Object} Current planar graph. */
  get graph() {
    return this._graph;
  }

  /** @return {?Object} Current packing state. */
  get packing() {
    return this._packing;
  }

  /** @return {number} Currently selected edge index, or -1. */
  get selectedEdge() {
    return this._selectedEdge;
  }

  /**
   * Loads a planar graph into the editor without pushing an undo entry.
   *
   * @param {!Object} graph
   * @param {boolean=} resetCamera
   */
  setGraph(graph, resetCamera = false) {
    this._graph = cloneGraph(graph);
    this._undoStack = [];
    this._redoStack = [];
    this._selectedEdge = -1;
    this._drawStart = null;
    if (resetCamera) this._camera = {cx: 0.5, cy: 0.5, span: 1.08};
    if (this.active && this.mode === 'cp') this.render();
    this._emitSelection();
  }

  /**
   * Loads a `PackedCreasePattern` payload from the WASM worker for packing edits.
   *
   * @param {?Object} packing
   */
  setPacking(packing) {
    this._packing = packing ? JSON.parse(JSON.stringify(packing)) : null;
    if (this._packing) this._packing.dirty = false;
    if (this.active && this.mode === 'packing') this.render();
  }

  /** Resets zoom/pan to frame `[0, 1]^2`. */
  resetZoom() {
    this._camera = {cx: 0.5, cy: 0.5, span: 1.08};
    if (this.active) this.render();
  }

  /** Adjusts the underlying integer grid resolution. */
  setGridSize(gridSize) {
    const g = Math.max(2, Math.min(128, Math.round(Number(gridSize) || 16)));
    this._graph.gridSize = g;
    if (this._packing) {
      this._packing.gridSize = g;
      this._packing.dirty = true;
    }
    if (this.active) this.render();
  }

  /** Clears interior creases back to an uncut square `[0, 1]^2`. */
  clearToBlankSheet() {
    this._pushUndo();
    this._graph = createBlankSquareGraph(this._graph.gridSize || 16);
    this._selectedEdge = -1;
    if (this.active) this.render();
    this._emitSelection();
    this._notifyGraphChanged(true);
  }

  /** Applies a sheet-wide geometric or M/V transform. */
  applyTransform(kind) {
    this._pushUndo();
    this._graph = transformGraph(this._graph, kind);
    this._selectedEdge = -1;
    if (this.active) this.render();
    this._emitSelection();
    this._notifyGraphChanged(true);
  }

  /**
   * Adjusts the selected edge's fold angle by `deltaDeg` (or sets an exact angle).
   *
   * @param {number} deltaDeg
   * @param {?number=} exactDeg
   */
  adjustSelectedFoldAngle(deltaDeg, exactDeg = null) {
    if (this._selectedEdge < 0 || this._selectedEdge >= this._graph.edges.length) {
      return;
    }
    const e = this._selectedEdge;
    if (this._graph.assignments[e] === 'B') return;
    this._pushUndo();
    const cur = this._graph.foldAngles[e] ?? 0;
    const next = Math.max(
        -180,
        Math.min(180, exactDeg !== null ? exactDeg : cur + deltaDeg));
    this._graph.foldAngles[e] = +next.toFixed(2);
    if (next < -0.5) this._graph.assignments[e] = 'M';
    else if (next > 0.5) this._graph.assignments[e] = 'V';
    else this._graph.assignments[e] = 'F';
    if (this.active) this.render();
    this._emitSelection();
    this._notifyGraphChanged(true);
  }

  /** Sets the selected edge's assignment (`M`, `V`, `F`, `B`). */
  setSelectedAssignment(assignment) {
    if (this._selectedEdge < 0 || this._selectedEdge >= this._graph.edges.length) {
      return;
    }
    this._pushUndo();
    this._graph.assignments[this._selectedEdge] = assignment;
    this._graph.foldAngles[this._selectedEdge] =
        DEFAULT_ASSIGNMENT_ANGLES[assignment] ?? 0;
    if (this.active) this.render();
    this._emitSelection();
    this._notifyGraphChanged(true);
  }

  /** Deletes the currently selected edge. */
  deleteSelectedEdge() {
    if (this._selectedEdge < 0 || this._selectedEdge >= this._graph.edges.length) {
      return;
    }
    this._pushUndo();
    this._eraseEdgeAt(this._selectedEdge);
    this._selectedEdge = -1;
    if (this.active) this.render();
    this._emitSelection();
    this._notifyGraphChanged(true);
  }

  /** Restores the previous graph snapshot. */
  undo() {
    if (!this._undoStack.length) return;
    this._redoStack.push(cloneGraph(this._graph));
    this._graph = this._undoStack.pop();
    this._selectedEdge = -1;
    if (this.active) this.render();
    this._emitSelection();
    this._notifyGraphChanged(true);
  }

  /** Reapplies an undone graph snapshot. */
  redo() {
    if (!this._redoStack.length) return;
    this._undoStack.push(cloneGraph(this._graph));
    this._graph = this._redoStack.pop();
    this._selectedEdge = -1;
    if (this.active) this.render();
    this._emitSelection();
    this._notifyGraphChanged(true);
  }

  /** @private */
  _pushUndo() {
    this._undoStack.push(cloneGraph(this._graph));
    if (this._undoStack.length > 60) this._undoStack.shift();
    this._redoStack = [];
  }

  /** @private */
  _eraseEdgeAt(edgeIndex) {
    const next = cloneGraph(this._graph);
    next.edges.splice(edgeIndex, 1);
    next.assignments.splice(edgeIndex, 1);
    next.foldAngles.splice(edgeIndex, 1);
    this._graph = normalizePlanarGraph(next);
  }

  /** @private */
  _notifyGraphChanged(structural) {
    this._cb.onGraphChanged?.(this._graph, structural);
  }

  /** @private */
  _emitSelection() {
    if (!this._cb.onSelectionChanged) return;
    const e = this._selectedEdge;
    if (e < 0 || e >= this._graph.edges.length) {
      this._cb.onSelectionChanged(null);
      return;
    }
    const [u, v] = this._graph.edges[e];
    this._cb.onSelectionChanged({
      edge: e,
      u,
      v,
      p1: this._graph.vertices[u],
      p2: this._graph.vertices[v],
      assignment: this._graph.assignments[e],
      foldAngle: this._graph.foldAngles[e],
    });
  }

  /**
   * Converts client pointer coordinates into FOLD flat-frame `(x, y)` (`+y` up).
   *
   * @private
   * @param {!PointerEvent|!WheelEvent} evt
   * @return {!Array<number>}
   */
  _clientToSheet(evt) {
    const rect = this._svg.getBoundingClientRect();
    const ux = (evt.clientX - rect.left) / Math.max(1, rect.width);
    const uy = (evt.clientY - rect.top) / Math.max(1, rect.height);
    const {cx, cy, span} = this._camera;
    const x = cx - span / 2 + ux * span;
    const y = cy + span / 2 - uy * span;
    return [x, y];
  }

  /**
   * @param {number} x
   * @param {number} y
   * @return {number}
   * @private
   */
  _findNearestEdge(x, y) {
    const threshold = 0.022 * this._camera.span;
    let bestE = -1;
    let bestD = threshold;
    const {vertices, edges} = this._graph;
    for (let e = 0; e < edges.length; e++) {
      const [u, v] = edges[e];
      const d = pointToSegmentDist(
          x, y, vertices[u][0], vertices[u][1], vertices[v][0], vertices[v][1]);
      if (d < bestD) {
        bestD = d;
        bestE = e;
      }
    }
    return bestE;
  }

  /**
   * @param {number} x
   * @param {number} y
   * @return {number}
   * @private
   */
  _findNearestVertex(x, y) {
    const threshold = 0.026 * this._camera.span;
    let bestV = -1;
    let bestD = threshold;
    for (let v = 0; v < this._graph.vertices.length; v++) {
      const [vx, vy] = this._graph.vertices[v];
      const d = Math.hypot(vx - x, vy - y);
      if (d < bestD) {
        bestD = d;
        bestV = v;
      }
    }
    return bestV;
  }

  /**
   * @param {number} rx
   * @param {number} ry
   * @return {?Object}
   * @private
   */
  _findNearestFlapHandle(rx, ry) {
    if (!this._packing?.flaps?.length) return null;
    const g = this._packing.gridSize || 16;
    // FOLD flat frame is (1 - gx/g, 1 - gy/g), so gx = (1 - rx)*g, gy = (1 - ry)*g.
    const gx = (1 - rx) * g;
    const gy = (1 - ry) * g;
    let best = null;
    let bestD = Math.max(0.85, 0.032 * this._camera.span * g);
    for (const flap of this._packing.flaps) {
      for (let cIdx = 0; cIdx < (flap.circle_centers || []).length; cIdx++) {
        const [cx, cy] = flap.circle_centers[cIdx];
        const d = Math.hypot(cx - gx, cy - gy);
        if (d < bestD) {
          bestD = d;
          best = {flap, cIdx, cx, cy};
        }
      }
    }
    return best;
  }

  /** @private */
  _bindEvents() {
    this._svg.addEventListener('wheel', (evt) => {
      if (!this.active) return;
      evt.preventDefault();
      const [px, py] = this._clientToSheet(evt);
      const factor = evt.deltaY > 0 ? 1.14 : 1 / 1.14;
      const nextSpan = Math.max(0.12, Math.min(2.4, this._camera.span * factor));
      const ratio = nextSpan / this._camera.span;
      this._camera.cx = px + (this._camera.cx - px) * ratio;
      this._camera.cy = py + (this._camera.cy - py) * ratio;
      this._camera.span = nextSpan;
      this.render();
    }, {passive: false});

    this._svg.addEventListener('pointerdown', (evt) => {
      if (!this.active) return;
      this._svg.setPointerCapture(evt.pointerId);
      if (evt.button === 1 || evt.button === 2 || evt.altKey) {
        this._panDrag = {
          startX: evt.clientX,
          startY: evt.clientY,
          cx: this._camera.cx,
          cy: this._camera.cy,
        };
        return;
      }

      const [rx, ry] = this._clientToSheet(evt);
      if (this.mode === 'packing') {
        this._startPackingDrag(rx, ry, evt);
        return;
      }

      if (this.tool === 'draw') {
        const snapped = snapPoint(rx, ry, this._graph, {
          snapMode: this.snapMode,
          angleLock: 'off',
        });
        this._drawStart = snapped;
        this._cursorSnap = snapped;
        this.render();
      } else if (this.tool === 'select') {
        this._selectedEdge = this._findNearestEdge(rx, ry);
        this.render();
        this._emitSelection();
      } else if (this.tool === 'assign') {
        const hit = this._findNearestEdge(rx, ry);
        if (hit >= 0) {
          this._pushUndo();
          this._isPaintingAssign = true;
          this._paintEdgeAssignment(hit, evt.shiftKey);
        }
      } else if (this.tool === 'erase') {
        const hit = this._findNearestEdge(rx, ry);
        if (hit >= 0) {
          this._pushUndo();
          this._eraseEdgeAt(hit);
          this._selectedEdge = -1;
          this.render();
          this._emitSelection();
          this._notifyGraphChanged(true);
        }
      } else if (this.tool === 'move') {
        const v = this._findNearestVertex(rx, ry);
        if (v >= 0) {
          this._pushUndo();
          this._dragVertex = v;
        }
      }
    });

    this._svg.addEventListener('pointermove', (evt) => {
      if (!this.active) return;
      if (this._panDrag) {
        const rect = this._svg.getBoundingClientRect();
        const dx = ((evt.clientX - this._panDrag.startX) / Math.max(1, rect.width)) *
            this._camera.span;
        const dy = ((evt.clientY - this._panDrag.startY) / Math.max(1, rect.height)) *
            this._camera.span;
        this._camera.cx = this._panDrag.cx - dx;
        this._camera.cy = this._panDrag.cy + dy;
        this.render();
        return;
      }

      const [rx, ry] = this._clientToSheet(evt);
      if (this.mode === 'packing') {
        if (this._dragFlap) {
          this._movePackingDrag(rx, ry);
        } else {
          const hovered = this._findNearestFlapHandle(rx, ry);
          const prevId = this._hoveredFlap ? `${this._hoveredFlap.flap.id}:${this._hoveredFlap.cIdx}` : '';
          const nextId = hovered ? `${hovered.flap.id}:${hovered.cIdx}` : '';
          if (prevId !== nextId) {
            this._hoveredFlap = hovered;
            this.render();
          }
        }
        return;
      }

      if (this._dragVertex >= 0) {
        const pt = snapPoint(rx, ry, this._graph, {
          snapMode: this.snapMode,
          angleLock: 'off',
        });
        this._graph.vertices[this._dragVertex] = pt;
        this.render();
        return;
      }

      if (this._isPaintingAssign && this.tool === 'assign') {
        const hit = this._findNearestEdge(rx, ry);
        if (hit >= 0 && this._graph.assignments[hit] !== this.activeAssignment) {
          this._paintEdgeAssignment(hit, false);
        }
        return;
      }

      this._hoveredVertex = this._findNearestVertex(rx, ry);
      this._hoveredEdge = this._findNearestEdge(rx, ry);
      if (this.tool === 'draw') {
        const lock = evt.shiftKey ? '45' : this.angleLock;
        this._cursorSnap = snapPoint(rx, ry, this._graph, {
          snapMode: this.snapMode,
          angleLock: this._drawStart ? lock : 'off',
          anchor: this._drawStart,
        });
      } else {
        this._cursorSnap = null;
      }
      this.render();
    });

    const finishPointer = (evt) => {
      if (!this.active) return;
      if (this._panDrag) {
        this._panDrag = null;
        return;
      }
      if (this.mode === 'packing' && this._dragFlap) {
        this._finishPackingDrag();
        return;
      }
      if (this._dragVertex >= 0) {
        this._dragVertex = -1;
        this._graph = normalizePlanarGraph(this._graph);
        this.render();
        this._notifyGraphChanged(true);
        return;
      }
      if (this._isPaintingAssign) {
        this._isPaintingAssign = false;
        this._notifyGraphChanged(false);
        return;
      }
      if (this.tool === 'draw' && this._drawStart) {
        const [rx, ry] = this._clientToSheet(evt);
        const lock = evt.shiftKey ? '45' : this.angleLock;
        const endPt = snapPoint(rx, ry, this._graph, {
          snapMode: this.snapMode,
          angleLock: lock,
          anchor: this._drawStart,
        });
        const startPt = this._drawStart;
        this._drawStart = null;
        if (Math.hypot(endPt[0] - startPt[0], endPt[1] - startPt[1]) > 1e-4) {
          this._pushUndo();
          const ang = DEFAULT_ASSIGNMENT_ANGLES[this.activeAssignment] ?? 0;
          this._graph = addCreaseSegment(
              this._graph, startPt, endPt, this.activeAssignment, ang,
              this.mirrorDraw);
          this._notifyGraphChanged(true);
        }
        this.render();
      }
    };

    this._svg.addEventListener('pointerup', finishPointer);
    this._svg.addEventListener('contextmenu', (e) => {
      if (this.active) e.preventDefault();
    });
  }

  /** @private */
  _paintEdgeAssignment(edgeIndex, cycle) {
    let nextAssign = this.activeAssignment;
    if (cycle) {
      const order = ['M', 'V', 'F', 'B'];
      const cur = this._graph.assignments[edgeIndex];
      nextAssign = order[(order.indexOf(cur) + 1) % order.length];
    }
    this._graph.assignments[edgeIndex] = nextAssign;
    this._graph.foldAngles[edgeIndex] =
        DEFAULT_ASSIGNMENT_ANGLES[nextAssign] ?? 0;
    if (this.mirrorDraw) {
      const [u, v] = this._graph.edges[edgeIndex];
      const [x1, y1] = this._graph.vertices[u];
      const [x2, y2] = this._graph.vertices[v];
      const mx1 = 1 - x1;
      const mx2 = 1 - x2;
      for (let e = 0; e < this._graph.edges.length; e++) {
        if (e === edgeIndex) continue;
        const [eu, ev] = this._graph.edges[e];
        const [ax, ay] = this._graph.vertices[eu];
        const [bx, by] = this._graph.vertices[ev];
        const matchFwd = Math.hypot(ax - mx1, ay - y1) < 1e-3 &&
            Math.hypot(bx - mx2, by - y2) < 1e-3;
        const matchRev = Math.hypot(ax - mx2, ay - y2) < 1e-3 &&
            Math.hypot(bx - mx1, by - y1) < 1e-3;
        if (matchFwd || matchRev) {
          this._graph.assignments[e] = nextAssign;
          this._graph.foldAngles[e] = DEFAULT_ASSIGNMENT_ANGLES[nextAssign] ?? 0;
        }
      }
    }
    this.render();
  }

  /** @private */
  _startPackingDrag(rx, ry, evt) {
    const hit = this._findNearestFlapHandle(rx, ry);
    if (!hit) {
      // Background drag on Packing canvas pans when zoomed in!
      if (this._camera.span < 1.05) {
        this._panDrag = {
          startX: evt.clientX,
          startY: evt.clientY,
          cx: this._camera.cx,
          cy: this._camera.cy,
        };
      }
      return;
    }
    const g = this._packing.gridSize || 16;
    const {flap, cIdx, cx, cy} = hit;
    let mirrorTarget = null;
    if (this.mirrorDraw && cx !== g - cx) {
      const mirOldX = g - cx;
      for (const cand of this._packing.flaps) {
        if (cand === flap) continue;
        for (let j = 0; j < (cand.circle_centers || []).length; j++) {
          const [mcx, mcy] = cand.circle_centers[j];
          if (mcx === mirOldX && mcy === cy) {
            mirrorTarget = {flap: cand, cIdx: j, origX: mcx, origY: mcy};
            break;
          }
        }
        if (mirrorTarget) break;
      }
    }
    this._dragFlap = {
      flap,
      cIdx,
      origX: cx,
      origY: cy,
      curX: cx,
      curY: cy,
      mirrorTarget,
    };
    this.render();
  }

  /** @private */
  _recomputeFlapRect(flap, g) {
    const centers = flap.circle_centers || [];
    if (!centers.length) return;
    const r = Math.max(1, flap.radius || 1);
    let minX = Infinity;
    let maxX = -Infinity;
    let minY = Infinity;
    let maxY = -Infinity;
    for (const [cx, cy] of centers) {
      minX = Math.min(minX, cx);
      maxX = Math.max(maxX, cx);
      minY = Math.min(minY, cy);
      maxY = Math.max(maxY, cy);
    }
    const x0 = Math.max(0, minX - r);
    const x1 = Math.min(g, maxX + r);
    const y0 = Math.max(0, minY - r);
    const y1 = Math.min(g, maxY + r);
    flap.x = x0;
    flap.y = y0;
    flap.width = Math.max(1, x1 - x0);
    flap.height = Math.max(1, y1 - y0);
  }

  /** @private */
  _movePackingDrag(rx, ry) {
    const drag = this._dragFlap;
    if (!drag || !this._packing) return;
    const g = this._packing.gridSize || 16;
    const nx = Math.max(0, Math.min(g, Math.round((1 - rx) * g)));
    const ny = Math.max(0, Math.min(g, Math.round((1 - ry) * g)));
    if (nx === drag.curX && ny === drag.curY) return;
    drag.curX = nx;
    drag.curY = ny;
    drag.flap.circle_centers[drag.cIdx] = [nx, ny];
    this._recomputeFlapRect(drag.flap, g);
    if (drag.mirrorTarget) {
      const mnx = Math.max(0, Math.min(g, g - nx));
      drag.mirrorTarget.flap.circle_centers[drag.mirrorTarget.cIdx] = [mnx, ny];
      this._recomputeFlapRect(drag.mirrorTarget.flap, g);
    }
    this._packing.dirty = true;
    this.render();
  }

  /** @private */
  _finishPackingDrag() {
    const drag = this._dragFlap;
    this._dragFlap = null;
    if (!drag || !this._packing) return;
    this.render();
    if (drag.curX === drag.origX && drag.curY === drag.origY) return;
    const moves = [{
      oldX: drag.origX,
      oldY: drag.origY,
      newX: drag.curX,
      newY: drag.curY,
    }];
    if (drag.mirrorTarget) {
      const g = this._packing.gridSize || 16;
      moves.push({
        oldX: drag.mirrorTarget.origX,
        oldY: drag.mirrorTarget.origY,
        newX: g - drag.curX,
        newY: drag.curY,
      });
    }
    const diag = this._evaluatePackingDiagnostics();
    this._cb.onPackingDiagnostics?.(diag);
    this._cb.onPackingMoved?.(moves, this._packing.gridSize || 16, diag);
  }

  /**
   * Evaluates live BP Studio circle-distance constraints and uncovered grid area.
   *
   * @private
   * @return {{conflicts: !Array<!Object>, contacts: !Array<!Object>, uncoveredCells: number}}
   */
  _evaluatePackingDiagnostics() {
    const pk = this._packing;
    if (!pk) return {conflicts: [], contacts: [], uncoveredCells: 0};
    const g = Math.max(2, pk.gridSize || 16);
    const byStick = new Map();
    for (const fl of pk.flaps || []) {
      if (fl.circle_centers?.length) {
        byStick.set(fl.stick || fl.id, {flap: fl, center: fl.circle_centers[0]});
      }
    }
    const conflicts = [];
    const contacts = [];
    for (const pair of pk.leafDistances || []) {
      const a = byStick.get(pair.a);
      const b = byStick.get(pair.b);
      if (!a || !b) continue;
      const dx = Math.abs(a.center[0] - b.center[0]);
      const dy = Math.abs(a.center[1] - b.center[1]);
      const linf = Math.max(dx, dy);
      if (linf < pair.dist - 1e-6) {
        conflicts.push({
          a: pair.a,
          b: pair.b,
          c1: a.center,
          c2: b.center,
          actual: linf,
          required: pair.dist,
        });
      } else if (Math.abs(linf - pair.dist) < 1e-6) {
        contacts.push({c1: a.center, c2: b.center});
      }
    }

    const covered = new Uint8Array(g * g);
    for (const fl of pk.flaps || []) {
      const x0 = Math.max(0, Math.min(g, fl.x));
      const x1 = Math.max(0, Math.min(g, fl.x + fl.width));
      const y0 = Math.max(0, Math.min(g, fl.y));
      const y1 = Math.max(0, Math.min(g, fl.y + fl.height));
      for (let y = y0; y < y1; y++) {
        for (let x = x0; x < x1; x++) covered[y * g + x] = 1;
      }
    }
    if (!pk.dirty) {
      for (const rv of pk.rivers || []) {
        for (const rr of rv.rects || []) {
          const x0 = Math.max(0, Math.min(g, rr.x));
          const x1 = Math.max(0, Math.min(g, rr.x + rr.width));
          const y0 = Math.max(0, Math.min(g, rr.y));
          const y1 = Math.max(0, Math.min(g, rr.y + rr.height));
          for (let y = y0; y < y1; y++) {
            for (let x = x0; x < x1; x++) covered[y * g + x] = 1;
          }
        }
      }
    }
    let uncoveredCells = 0;
    for (let i = 0; i < covered.length; i++) {
      if (!covered[i]) uncoveredCells++;
    }
    return {conflicts, contacts, uncoveredCells, covered};
  }

  /** Renders either the interactive CP canvas or the Packing CP canvas. */
  render() {
    const {cx, cy, span} = this._camera;
    const viewBox = [cx - span / 2, -(cy + span / 2), span, span];
    this._svg.setAttribute('viewBox', viewBox.join(' '));

    if (this.mode === 'packing' && this._packing) {
      this._renderPacking();
      return;
    }
    this._renderCp();
  }

  /** @private */
  _renderCp() {
    const g = Math.max(2, this._graph.gridSize || 16);
    const root = svgEl('g', {
      transform: 'scale(1 -1)',
      fill: 'none',
      'stroke-linecap': 'round',
    });

    root.append(svgEl('rect', {
      x: 0,
      y: 0,
      width: 1,
      height: 1,
      fill: '#fffdf7',
      stroke: '#d1d5db',
      'stroke-width': 1,
      'vector-effect': 'non-scaling-stroke',
    }));

    if (this.showGrid && g <= 96) {
      const minorCmds = [];
      const majorCmds = [];
      for (let i = 1; i < g; i++) {
        const c = +(i / g).toFixed(6);
        const bucket = i % 4 === 0 ? majorCmds : minorCmds;
        bucket.push(`M${c} 0L${c} 1M0 ${c}L1 ${c}`);
      }
      if (minorCmds.length) {
        root.append(svgEl('path', {
          d: minorCmds.join(''),
          stroke: '#e5e7eb',
          'stroke-width': 0.65,
          'vector-effect': 'non-scaling-stroke',
        }));
      }
      if (majorCmds.length) {
        root.append(svgEl('path', {
          d: majorCmds.join(''),
          stroke: '#cbd5e1',
          'stroke-width': 0.9,
          'vector-effect': 'non-scaling-stroke',
        }));
      }
      if (this.mirrorDraw) {
        root.append(svgEl('line', {
          x1: 0.5,
          y1: 0,
          x2: 0.5,
          y2: 1,
          stroke: '#d97706',
          'stroke-width': 1.2,
          'stroke-dasharray': '4 3',
          'vector-effect': 'non-scaling-stroke',
        }));
      }
    }

    const {vertices, edges, assignments, foldAngles} = this._graph;
    const byAssign = {F: [], U: [], B: [], M: [], V: []};
    for (let e = 0; e < edges.length; e++) {
      const a = assignments[e] in byAssign ? assignments[e] : 'F';
      byAssign[a].push(e);
    }

    for (const [assign, list] of Object.entries(byAssign)) {
      if (!list.length) continue;
      const cmds = list.map((e) => {
        const [u, v] = edges[e];
        return `M${vertices[u][0]} ${vertices[u][1]}L${vertices[v][0]} ${vertices[v][1]}`;
      }).join('');
      root.append(svgEl('path', {
        d: cmds,
        stroke: STROKE_COLORS[assign],
        'stroke-width': assign === 'B' ? 1.5 : (assign === 'F' ? 0.85 : 1.3),
        'vector-effect': 'non-scaling-stroke',
      }));
    }

    for (let e = 0; e < edges.length; e++) {
      const a = assignments[e];
      if (a !== 'M' && a !== 'V') continue;
      const absAng = Math.abs(foldAngles[e] ?? 180);
      if (absAng > 1 && absAng < 176) {
        const [u, v] = edges[e];
        root.append(svgEl('line', {
          x1: vertices[u][0],
          y1: vertices[u][1],
          x2: vertices[v][0],
          y2: vertices[v][1],
          stroke: '#ffffff',
          'stroke-width': 1.1,
          'stroke-dasharray': '2 3',
          'vector-effect': 'non-scaling-stroke',
          opacity: 0.75,
        }));
      }
    }

    for (const [eIdx, color, width] of [
      [this._hoveredEdge, '#f59e0b', 3.2],
      [this._selectedEdge, '#7c3aed', 3.8],
    ]) {
      if (eIdx >= 0 && eIdx < edges.length) {
        const [u, v] = edges[eIdx];
        root.append(svgEl('line', {
          x1: vertices[u][0],
          y1: vertices[u][1],
          x2: vertices[v][0],
          y2: vertices[v][1],
          stroke: color,
          'stroke-width': width,
          'vector-effect': 'non-scaling-stroke',
          opacity: 0.8,
        }));
      }
    }

    if (this.showViolations) {
      const rBadge = 0.011 * this._camera.span;
      for (const viol of analyzeVertices(this._graph)) {
        const badge = svgEl('circle', {
          cx: viol.x,
          cy: viol.y,
          r: rBadge,
          fill: viol.kind === 'kawasaki' ? '#f59e0b' : '#ef4444',
          stroke: '#ffffff',
          'stroke-width': 1.2,
          'vector-effect': 'non-scaling-stroke',
        });
        const title = svgEl('title');
        title.textContent = viol.message;
        badge.append(title);
        root.append(badge);
      }
    }

    if (this._hoveredVertex >= 0 && this._hoveredVertex < vertices.length) {
      const [vx, vy] = vertices[this._hoveredVertex];
      root.append(svgEl('circle', {
        cx: vx,
        cy: vy,
        r: 0.011 * this._camera.span,
        fill: '#2563eb',
        stroke: '#ffffff',
        'stroke-width': 1.2,
        'vector-effect': 'non-scaling-stroke',
      }));
    }

    if (this.tool === 'draw' && this._cursorSnap) {
      const [cx, cy] = this._cursorSnap;
      root.append(svgEl('circle', {
        cx,
        cy,
        r: 0.008 * this._camera.span,
        fill: STROKE_COLORS[this.activeAssignment] || '#111827',
      }));
      if (this._drawStart) {
        const [sx, sy] = this._drawStart;
        root.append(svgEl('line', {
          x1: sx,
          y1: sy,
          x2: cx,
          y2: cy,
          stroke: STROKE_COLORS[this.activeAssignment] || '#111827',
          'stroke-width': 2.2,
          'stroke-dasharray': '4 3',
          'vector-effect': 'non-scaling-stroke',
        }));
        if (this.mirrorDraw) {
          root.append(svgEl('line', {
            x1: 1 - sx,
            y1: sy,
            x2: 1 - cx,
            y2: cy,
            stroke: STROKE_COLORS[this.activeAssignment] || '#111827',
            'stroke-width': 2.0,
            'stroke-dasharray': '3 3',
            'vector-effect': 'non-scaling-stroke',
            opacity: 0.65,
          }));
        }
      }
    }

    this._svg.replaceChildren(root);
  }

  /** @private */
  _renderPacking() {
    const pk = this._packing;
    const g = Math.max(2, pk.gridSize || 16);
    const diag = this._evaluatePackingDiagnostics();
    const root = svgEl('g', {
      transform: 'scale(1 -1)',
      fill: 'none',
      'stroke-linecap': 'round',
    });

    // 1. Sheet background in FOLD flat coordinates (x_fold = 1 - gx/g, y_fold = 1 - gy/g).
    const clipId = 'corigami-pack-sheet-clip';
    const defs = svgEl('defs');
    const clip = svgEl('clipPath', {id: clipId});
    clip.append(svgEl('rect', {x: 0, y: 0, width: 1, height: 1}));
    defs.append(clip);
    root.append(defs);

    root.append(svgEl('rect', {
      x: 0,
      y: 0,
      width: 1,
      height: 1,
      fill: '#fffdf7',
    }));

    // 2. River & Flap region fills matching Studio's legend colors (#b9d7ea, #ead9b5).
    if (!pk.dirty) {
      for (const rv of pk.rivers || []) {
        for (const rr of rv.rects || []) {
          const el = svgEl('rect', {
            x: 1 - (rr.x + rr.width) / g,
            y: 1 - (rr.y + rr.height) / g,
            width: rr.width / g,
            height: rr.height / g,
            fill: REGION_FILLS.river,
            opacity: 0.72,
          });
          const t = svgEl('title');
          t.textContent = `River: ${(rv.stick || rv.id).replaceAll('_', ' ')} (width ${rv.width})`;
          el.append(t);
          root.append(el);
        }
      }
    }

    for (const fl of pk.flaps || []) {
      const fx = 1 - (fl.x + fl.width) / g;
      const fy = 1 - (fl.y + fl.height) / g;
      const fw = fl.width / g;
      const fh = fl.height / g;
      const rectEl = svgEl('rect', {
        x: fx,
        y: fy,
        width: fw,
        height: fh,
        fill: REGION_FILLS.flap,
        opacity: pk.dirty ? 0.55 : 0.76,
        stroke: pk.dirty ? '#b45309' : 'none',
        'stroke-width': pk.dirty ? 1.1 : 0,
        'vector-effect': 'non-scaling-stroke',
      });
      const t = svgEl('title');
      t.textContent =
          `Flap: ${(fl.stick || fl.id).replaceAll('_', ' ')} (radius ${fl.radius || '?'}) — drag red center dot to move`;
      rectEl.append(t);
      root.append(rectEl);
    }

    // 3. When dragging (`pk.dirty`), shade any uncovered grid cells so tiling gaps are obvious.
    if (pk.dirty && diag.uncoveredCells > 0 && g <= 64) {
      const gapCmds = [];
      const u = 1 / g;
      for (let gy = 0; gy < g; gy++) {
        for (let gx = 0; gx < g; gx++) {
          if (!diag.covered[gy * g + gx]) {
            const x = +(1 - (gx + 1) / g).toFixed(6);
            const y = +(1 - (gy + 1) / g).toFixed(6);
            gapCmds.push(`M${x} ${y}h${u.toFixed(6)}v${u.toFixed(6)}h-${u.toFixed(6)}Z`);
          }
        }
      }
      if (gapCmds.length) {
        root.append(svgEl('path', {
          d: gapCmds.join(''),
          fill: 'rgba(239, 68, 68, 0.14)',
        }));
      }
    }

    // 4. Integer grid lines + symmetry centerline.
    if (g <= 96) {
      const minor = [];
      const major = [];
      for (let i = 1; i < g; i++) {
        const c = +(i / g).toFixed(6);
        (i % 4 === 0 ? major : minor).push(`M${c} 0L${c} 1M0 ${c}L1 ${c}`);
      }
      if (minor.length) {
        root.append(svgEl('path', {
          d: minor.join(''),
          stroke: 'rgba(15, 23, 42, 0.08)',
          'stroke-width': 0.65,
          'vector-effect': 'non-scaling-stroke',
        }));
      }
      if (major.length) {
        root.append(svgEl('path', {
          d: major.join(''),
          stroke: 'rgba(15, 23, 42, 0.17)',
          'stroke-width': 0.9,
          'vector-effect': 'non-scaling-stroke',
        }));
      }
    }
    root.append(svgEl('line', {
      x1: 0.5,
      y1: 0,
      x2: 0.5,
      y2: 1,
      stroke: '#64748b',
      'stroke-width': 1,
      'stroke-dasharray': '4 3',
      'vector-effect': 'non-scaling-stroke',
      opacity: 0.55,
    }));

    // 5. BP Studio-style radius circles (clipped to the [0, 1]^2 sheet).
    if (this.showPackingCircles) {
      const circleGroup = svgEl('g', {'clip-path': `url(#${clipId})`});
      for (const fl of pk.flaps || []) {
        const rad = (fl.radius || 1) / g;
        for (const [cx, cy] of fl.circle_centers || []) {
          circleGroup.append(svgEl('circle', {
            cx: 1 - cx / g,
            cy: 1 - cy / g,
            r: rad,
            stroke: '#92400e',
            'stroke-width': 0.95,
            'stroke-dasharray': '3 3',
            'vector-effect': 'non-scaling-stroke',
            opacity: 0.42,
          }));
        }
      }
      root.append(circleGroup);
    }

    // 6. Real COrigami Packing Crease Pattern lines (axial/miter ridges + green hinges)!
    if (!pk.dirty && this.showPackingCreases && pk.creases?.length) {
      const buckets = {F: [], M: [], V: [], H: [], B: []};
      for (const c of pk.creases) {
        const k = c.fold in buckets ? c.fold : 'F';
        buckets[k].push(
            `M${+(1 - c.x1).toFixed(6)} ${+(1 - c.y1).toFixed(6)}` +
            `L${+(1 - c.x2).toFixed(6)} ${+(1 - c.y2).toFixed(6)}`);
      }
      for (const [code, width] of [
        ['F', 0.85],
        ['M', 1.2],
        ['V', 1.2],
        ['H', 1.9],
        ['B', 1.6],
      ]) {
        if (!buckets[code].length) continue;
        root.append(svgEl('path', {
          d: buckets[code].join(''),
          stroke: STROKE_COLORS[code],
          'stroke-width': width,
          'vector-effect': 'non-scaling-stroke',
        }));
      }
    } else {
      root.append(svgEl('rect', {
        x: 0,
        y: 0,
        width: 1,
        height: 1,
        stroke: '#111827',
        'stroke-width': 1.5,
        'vector-effect': 'non-scaling-stroke',
      }));
    }

    // 7. Live BP Studio tree-distance contact & conflict lines during interactive drag!
    if (pk.dirty) {
      for (const ct of diag.contacts) {
        root.append(svgEl('line', {
          x1: 1 - ct.c1[0] / g,
          y1: 1 - ct.c1[1] / g,
          x2: 1 - ct.c2[0] / g,
          y2: 1 - ct.c2[1] / g,
          stroke: '#16a34a',
          'stroke-width': 1.4,
          'vector-effect': 'non-scaling-stroke',
          opacity: 0.65,
        }));
      }
      for (const cf of diag.conflicts) {
        const line = svgEl('line', {
          x1: 1 - cf.c1[0] / g,
          y1: 1 - cf.c1[1] / g,
          x2: 1 - cf.c2[0] / g,
          y2: 1 - cf.c2[1] / g,
          stroke: '#dc2626',
          'stroke-width': 2.4,
          'stroke-dasharray': '4 3',
          'vector-effect': 'non-scaling-stroke',
        });
        const tt = svgEl('title');
        tt.textContent =
            `Overlap: ${cf.a} ↔ ${cf.b} (L∞ distance ${cf.actual} < required ${cf.required})`;
        line.append(tt);
        root.append(line);
      }
    }

    // 8. Draggable circle-center handles + clean compact labels.
    const span = this._camera.span;
    const fontSize = Math.max(0.015, Math.min(0.023, 0.021 * span));
    for (const fl of pk.flaps || []) {
      const fx = 1 - (fl.x + fl.width) / g;
      const fy = 1 - (fl.y + fl.height) / g;
      const fw = fl.width / g;
      const fh = fl.height / g;

      if (this.showPackingLabels) {
        const lx = Math.max(0.06, Math.min(0.94, fx + fw / 2));
        const ly = Math.max(0.04, Math.min(0.96, fy + fh / 2));
        const txt = svgEl('text', {
          x: lx,
          y: -ly,
          transform: 'scale(1 -1)',
          fill: '#1e293b',
          stroke: '#fffdf7',
          'stroke-width': 2.6,
          'paint-order': 'stroke',
          'font-size': fontSize,
          'font-weight': '600',
          'text-anchor': 'middle',
          'dominant-baseline': 'central',
          'pointer-events': 'none',
        });
        txt.textContent = String(fl.stick || fl.id).replaceAll('_', ' ');
        root.append(txt);
      }

      for (let cIdx = 0; cIdx < (fl.circle_centers || []).length; cIdx++) {
        const [cx, cy] = fl.circle_centers[cIdx];
        const hx = 1 - cx / g;
        const hy = 1 - cy / g;
        const isActive =
            (this._dragFlap && this._dragFlap.flap === fl && this._dragFlap.cIdx === cIdx) ||
            (this._hoveredFlap && this._hoveredFlap.flap === fl && this._hoveredFlap.cIdx === cIdx);
        if (isActive) {
          root.append(svgEl('circle', {
            cx: hx,
            cy: hy,
            r: 0.021 * span,
            fill: 'rgba(220, 38, 38, 0.22)',
            'pointer-events': 'none',
          }));
        }
        const handle = svgEl('circle', {
          cx: hx,
          cy: hy,
          r: (isActive ? 0.015 : 0.0115) * span,
          fill: '#dc2626',
          stroke: '#ffffff',
          'stroke-width': 1.6,
          'vector-effect': 'non-scaling-stroke',
          style: 'cursor: grab;',
        });
        const ht = svgEl('title');
        ht.textContent =
            `${(fl.stick || fl.id).replaceAll('_', ' ')} center (${cx}, ${cy}) · radius ${fl.radius || '?'} — drag to move on ${g}×${g} grid`;
        handle.append(ht);
        root.append(handle);

        if (isActive) {
          const coordLbl = svgEl('text', {
            x: Math.max(0.08, Math.min(0.92, hx)),
            y: -(Math.min(0.97, hy + 0.032 * span)),
            transform: 'scale(1 -1)',
            fill: '#991b1b',
            stroke: '#ffffff',
            'stroke-width': 3,
            'paint-order': 'stroke',
            'font-size': fontSize * 0.95,
            'font-weight': '700',
            'text-anchor': 'middle',
            'pointer-events': 'none',
          });
          coordLbl.textContent = `${(fl.stick || fl.id).replaceAll('_', ' ')} (${cx}, ${cy})`;
          root.append(coordLbl);
        }
      }
    }

    this._svg.replaceChildren(root);
  }
}
