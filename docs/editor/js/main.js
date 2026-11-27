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
 * @fileoverview COrigami Editor controller:
 *   - Single-screen (`100dvh`) 5-mode workspace (`Shape 3D`, `Creases`,
 *     `Packing`, `Skeleton`, `Stages & Export`) with zero nested scroll traps
 *   - Direct 1-click 3D & 2D unmerged sub-flap selection + live 60 FPS
 *     crease-backed shaping (`flap_shaper.js`)
 *   - Universal zero-upload local file drop (`.fold`, `.svg`, `.cp`, `.bps`,
 *     `stick_figure.json`, `.zip`)
 */

import {parseFold, regionTips, strainPercent} from '../../js/fold.js';
import {DEFAULT_BASE_COMPLETE_T, DEFAULT_SHAPING_START_T, buildFoldingSchedule, unwarpFoldingTime, warpFoldingTime} from '../../js/schedule.js';
import {VIDEO_PRESETS, pickSupportedVideoFormat, recordFoldingVideo} from '../../js/video.js';
import {CreasePatternEditor} from './cp_editor.js';
import {buildFoldFromCpGraph, graphFromModel} from './cp_geometry.js';
import {creasePatternSvg, drawCreasePattern, drawPacking} from './diagram.js';
import {applyFlapShaping, buildSubFlapCatalog, findMirroredSubFlapId} from './flap_shaper.js';
import {Stage} from './stage.js';
import {StickFigureEditor, createStarterStickFigure} from './stick_editor.js';
import {WasmPipelineClient} from './wasm_client.js';

const FOLD_SECONDS_STAGED = 14;
const FOLD_SECONDS_SIMULTANEOUS = 6;
const LOOP_DWELL_SECONDS = 0.55;
const BASE_DWELL_SECONDS = 0.6;
const SPEEDS = [1, 0.5, 2];

const DEFAULT_COLORS = {
  top: '#9e3329',
  bottom: '#f3e7ce',
};

const WASHI_PRESETS = [
  {name: 'Vermilion & Kozo Cream', top: '#9e3329', bottom: '#f3e7ce'},
  {name: 'Aizome Indigo & Ivory', top: '#24476e', bottom: '#f4efe4'},
  {name: 'Matcha Jade & Warm Sand', top: '#2f6a52', bottom: '#efe4c8'},
  {name: 'Kitsune Ochre & Sumi', top: '#c4832b', bottom: '#2b2d33'},
  {name: 'Ume Plum & Sakura Mist', top: '#7a2e52', bottom: '#f6e6ea'},
  {name: 'Suzu Slate & Snow Washi', top: '#475569', bottom: '#f8fafc'},
];

const FALLBACK_SHAPING_TOOLS = [
  {
    name: 'simple_fold',
    label: 'Simple Fold (2D Zigzag)',
    params: [
      {key: 'element_name', type: 'element', label: 'Flap / River'},
      {key: 'angle', type: 'number', label: 'Angle (°)', default: 45, min: -175, max: 175, step: 5},
      {key: 'depth', type: 'number', label: 'Depth', default: 0.25, min: 0, max: 0.9, step: 0.05},
      {key: 'fold_percent', type: 'number', label: 'Fold %', default: 0.75, min: 0.1, max: 1.0, step: 0.05},
    ],
  },
  {
    name: 'crimp_fold',
    label: 'Crimp Fold',
    params: [
      {key: 'element_name', type: 'element', label: 'Flap / River'},
      {key: 'angle', type: 'number', label: 'Angle (°)', default: 35, min: -160, max: 160, step: 5},
      {key: 'depth', type: 'number', label: 'Depth', default: 0.3, min: 0, max: 0.9, step: 0.05},
      {key: 'fold_order', type: 'select', label: 'Order', default: 'MV', options: ['MV', 'VM']},
    ],
  },
  {
    name: 'reverse_fold',
    label: 'Inside / Outside Reverse Fold',
    params: [
      {key: 'element_name', type: 'element', label: 'Flap'},
      {key: 'angle', type: 'number', label: 'Angle (°)', default: 60, min: -170, max: 170, step: 5},
      {key: 'depth', type: 'number', label: 'Depth', default: 0.35, min: 0, max: 0.9, step: 0.05},
    ],
  },
  {
    name: 'fan_out_fold',
    label: 'Fan Out Wing / Tail',
    params: [
      {key: 'element_name', type: 'element', label: 'Flap'},
      {key: 'num_pleats', type: 'number', label: 'Pleats', default: 3, min: 1, max: 8, step: 1},
      {key: 'fold_percent', type: 'number', label: 'Fold %', default: 0.6, min: 0.1, max: 1.0, step: 0.05},
    ],
  },
  {
    name: 'squash_fold',
    label: 'Squash Fold',
    params: [
      {key: 'element_name', type: 'element', label: 'Flap'},
      {key: 'fold_percent', type: 'number', label: 'Fold %', default: 0.85, min: 0.1, max: 1.0, step: 0.05},
    ],
  },
  {
    name: 'narrow_central_flap',
    label: 'Narrow Central Flap (Taper)',
    params: [
      {key: 'element_name', type: 'element', label: 'Flap'},
      {key: 'tip_width', type: 'number', label: 'Tip Width', default: 0.25, min: 0.05, max: 0.85, step: 0.05},
      {key: 'start_offset', type: 'number', label: 'Start Offset', default: 0.0, min: 0.0, max: 0.7, step: 0.05},
      {key: 'fold_type', type: 'select', label: 'Polarity', default: 'auto', options: ['auto', 'mountain', 'valley']},
    ],
  },
];

const WORKSPACE_ORDER = ['shape', 'pattern', 'packing', 'sticks', 'export'];

/**
 * @param {string} id
 * @return {?HTMLElement}
 */
const byId = (id) => document.getElementById(id);

/** @type {?Stage} */
let stage = null;

/** @type {?CreasePatternEditor} */
let cpEditor = null;

/** @type {?StickFigureEditor} */
let stickEditor = null;

/** @type {?WasmPipelineClient} */
let wasm = null;

let speedIndex = 0;
let shapingTools = FALLBACK_SHAPING_TOOLS;
let requested = null;

/** Active model and single-screen workspace state. */
const shown = {
  /** @type {?Object} Pristine loaded `.fold` dict for full reset. */
  originalFoldDict: null,
  /** @type {?Object} Committed `.fold` dict before current live slider drag. */
  committedFoldDict: null,
  /** @type {!Array<!Object>} Stack of prior committed `.fold` dicts for Undo. */
  flapUndoStack: [],
  /** @type {boolean} Whether current flap sliders have uncommitted live edits. */
  activeFlapDirty: false,
  /** @type {string} Active flap shaping mode: 'bend', 'narrow', 'crimp', 'curve', 'pleat'. */
  flapMode: 'bend',
  /** @type {string} Companion map shown under Shape 3D: 'packing' | 'pattern'. */
  shapeMapMode: 'pattern',
  /** @type {string} Sub-mode in Creases workspace: 'shaped' | 'base' | 'edit'. */
  cpSubmode: 'shaped',

  /** @type {!Array<!Object>} Unmerged sub-flap entries from `buildSubFlapCatalog`. */
  subFlaps: [],
  /** @type {?Int32Array} Per-face sub-flap ID map for 3D raycasting & highlights. */
  faceSubFlap: null,
  /** @type {?number} Selected sub-flap ID (`0 .. subFlaps.length - 1`), or null. */
  subFlapId: null,
  /** @type {boolean} Whether the whole merged chain is active (`All` in stepper). */
  wholeChainSelected: false,

  /** @type {?Object} Current displayed FOLD JSON dict. */
  rawFoldDict: null,
  /** @type {?Object} Parsed model from `fold.js`. */
  model: null,
  /** @type {?Object} Parsed stick figure JSON. */
  stickFigure: null,
  /** @type {?Object} Packing data from Python WASM (`flaps`, `creases`, etc.). */
  packingData: null,
  /** @type {boolean} Whether a solved base exists in the WASM worker. */
  hasWasmSolved: false,
  /** @type {!Array<!Object>} Available shaping elements from WASM. */
  shapingElements: [],
  /** @type {!Array<!Object>} Applied WASM shaping actuator history. */
  shapingHistory: [],

  /** @type {?Object} Multi-stage folding schedule. */
  schedule: null,
  /** @type {?Object} Latest evaluated stage info from `Stage.setFold`. */
  lastState: null,
  /** @type {?string} Key of the last rendered crease-pattern SVG state. */
  lastDiagramKey: null,
  /** @type {?Object} Catalog entry; a local file has one without a `slug`. */
  entry: null,
  /** @type {number} Folding progress from 0 (flat) to 1 (folded). */
  fold: 1,
  /** @type {string} Active workspace tab: 'shape' | 'pattern' | 'packing' | 'sticks' | 'export'. */
  tab: 'shape',
  /** @type {boolean} Whether Packing workspace is in interactive circle-drag mode. */
  editingPacking: false,
  /** @type {boolean} Whether Base solver options drawer is open in Creases -> Base. */
  editingBase: false,
  /** @type {!Array<number>} Camera rotation matrix from Stage. */
  view: [1, 0, 0, 0, 1, 0, 0, 0, 1],
  /** @type {number} `requestAnimationFrame` handle while folding animation plays. */
  animation: 0,
  /** @type {?AbortController} Controller while 1080p video recording runs. */
  recordingAbort: null,
};

/**
 * @param {string} tag
 * @param {!Object<string, *>=} properties
 * @param {...*} children
 * @return {!HTMLElement}
 */
function html(tag, properties = {}, ...children) {
  const element = Object.assign(document.createElement(tag), properties);
  element.append(...children);
  return element;
}

/**
 * @param {string} stickName
 * @return {string}
 */
const words = (stickName) => stickName.replaceAll('_', ' ');

/**
 * @param {string} url
 * @return {!Promise<*>}
 */
async function fetchJson(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url}: HTTP ${response.status}`);
  return response.json();
}

/**
 * @param {string} label
 * @return {string}
 */
function shortStageTitle(label) {
  return label.replace(/^Act\s+[IVX]+(?:\.\d+)?\s*·\s*/i, '');
}

/**
 * @param {string} toolName
 */
function setCpToolUI(toolName) {
  if (!cpEditor) return;
  cpEditor.tool = toolName;
  for (const b of document.querySelectorAll('[data-cp-tool]')) {
    b.classList.toggle('active', b.dataset.cpTool === toolName);
  }
}

/**
 * @param {*} content
 * @param {string} filename
 * @param {string=} mimeType
 */
function triggerBlobDownload(content, filename, mimeType = 'application/octet-stream') {
  const blob = new Blob([content], {type: mimeType});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 2000);
}

/**
 * Closes or toggles the showcase model picker popover.
 *
 * @param {boolean=} forceOpen
 */
function setGalleryOpen(forceOpen) {
  const pop = byId('gallery-popover');
  const btn = byId('toggle-gallery');
  const open = forceOpen === undefined ? pop.hidden : Boolean(forceOpen);
  pop.hidden = !open;
  btn.setAttribute('aria-expanded', String(open));
}

/**
 * Returns the active sub-flap IDs and exact face indices currently highlighted
 * (including bilateral L/R mirror partners when `Mirror L↔R` is checked).
 *
 * @return {{subFlapIds: !Array<number>, faceSet: !Set<number>}}
 */
function activeHighlightedSelection() {
  const subFlapIds = [];
  const faceSet = new Set();
  const {subFlapId, subFlaps, wholeChainSelected, model} = shown;
  if (subFlapId === null || !model || !subFlaps[subFlapId]) {
    return {subFlapIds, faceSet};
  }
  const primary = subFlaps[subFlapId];
  const mirrorOn = byId('flap-mirror').checked;

  if (wholeChainSelected) {
    for (const id of primary.chainSubFlapIds) subFlapIds.push(id);
    if (mirrorOn && primary.mirrorRegionIdx !== null) {
      for (const sf of subFlaps) {
        if (sf.regionIdx === primary.mirrorRegionIdx) subFlapIds.push(sf.id);
      }
    }
  } else {
    subFlapIds.push(primary.id);
    if (mirrorOn) {
      const mirrorId = findMirroredSubFlapId(subFlaps, primary.id);
      if (mirrorId !== null && mirrorId !== primary.id) subFlapIds.push(mirrorId);
    }
  }

  for (const id of subFlapIds) {
    for (const f of subFlaps[id]?.faceIndices ?? []) faceSet.add(f);
  }
  return {subFlapIds, faceSet};
}

/**
 * @param {number=} defaultDepth
 */
function resetFlapSlidersUI(defaultDepth = 30) {
  byId('flap-pitch').value = 0;
  byId('flap-pitch-val').textContent = '0°';
  byId('flap-depth').value = defaultDepth;
  byId('flap-depth-val').textContent = `${defaultDepth}%`;
  byId('flap-narrow').value = 0;
  byId('flap-narrow-val').textContent = '0%';
  byId('flap-tuck-angle').value = 165;
  byId('flap-tuck-val').textContent = '+165°';
  byId('flap-narrow-depth').value = 12;
  byId('flap-narrow-depth-val').textContent = '12%';
  byId('flap-turn').value = 0;
  byId('flap-turn-val').textContent = '0°';
  byId('flap-crimp-depth').value = Math.min(70, defaultDepth);
  byId('flap-crimp-depth-val').textContent = `${Math.min(70, defaultDepth)}%`;
  shown.activeFlapDirty = false;
}

/** Updates Undo / Reset button enabled states. */
function syncFlapHistoryUI() {
  const canUndo = shown.activeFlapDirty || shown.flapUndoStack.length > 0;
  byId('flap-undo-btn').disabled = !canUndo;
  byId('flap-reset-btn').disabled = !canUndo;
  byId('flap-top-undo').disabled = !canUndo;
  byId('flap-top-reset').disabled = !canUndo;
}

/** Commits any live slider edit onto `shown.flapUndoStack`. */
function commitPendingFlapEdit() {
  if (!shown.activeFlapDirty || !shown.rawFoldDict) return;
  if (shown.committedFoldDict) {
    shown.flapUndoStack.push(shown.committedFoldDict);
  }
  shown.committedFoldDict = shown.rawFoldDict;
  shown.activeFlapDirty = false;
  syncFlapHistoryUI();
}

/**
 * @param {string} mode
 */
function setFlapModeUI(mode) {
  if (shown.activeFlapDirty) {
    commitPendingFlapEdit();
    resetFlapSlidersUI(Number(byId('flap-depth').value) || 30);
  }
  shown.flapMode = mode;
  for (const btn of document.querySelectorAll('[data-flap-mode]')) {
    btn.classList.toggle('active', btn.dataset.flapMode === mode);
  }
  const isBendLike = mode === 'bend' || mode === 'curve' || mode === 'pleat';
  byId('flap-bend-controls').hidden = !isBendLike;
  byId('flap-narrow-controls').hidden = mode !== 'narrow';
  byId('flap-crimp-controls').hidden = mode !== 'crimp';

  const pitchLabels = {
    bend: 'Fold Angle',
    curve: 'Total Arc',
    pleat: 'Pleat Angle',
  };
  if (pitchLabels[mode]) {
    byId('flap-pitch-label').textContent = pitchLabels[mode];
  }

  const showSteps = mode === 'curve' || mode === 'pleat';
  byId('flap-steps-row').hidden = !showSteps;
  if (mode === 'curve') {
    byId('flap-steps-label').textContent = 'Crease Steps';
    byId('flap-steps').min = 2;
    byId('flap-steps').max = 5;
    if (Number(byId('flap-steps').value) < 2) byId('flap-steps').value = 3;
  } else if (mode === 'pleat') {
    byId('flap-steps-label').textContent = 'Pleat Pairs';
    byId('flap-steps').min = 1;
    byId('flap-steps').max = 4;
    if (Number(byId('flap-steps').value) > 4) byId('flap-steps').value = 2;
  }
  byId('flap-steps-val').textContent = byId('flap-steps').value;
}

/** Renders the `[All] [1. thigh] [2. shin] [3. claw]` stepper when `stickCount > 1`. */
function renderChainStepperUI() {
  const bar = byId('flap-chain-bar');
  const container = byId('flap-chain-pills');
  const sf = shown.subFlapId !== null ? shown.subFlaps[shown.subFlapId] : null;
  if (!sf || sf.stickCount <= 1) {
    bar.hidden = true;
    container.replaceChildren();
    return;
  }
  bar.hidden = false;
  const allBtn = html(
      'button',
      {
        type: 'button',
        className: `tool-btn${shown.wholeChainSelected ? ' active' : ''}`,
        title: `Shape all ${sf.stickCount} merged sticks together`,
      },
      `All (${sf.stickCount})`);
  allBtn.addEventListener('click', () => {
    selectSubFlap(sf.id, /* wholeChain= */ true);
  });

  const stepBtns = sf.chainSubFlapIds.map((cid, idx) => {
    const item = shown.subFlaps[cid];
    const isActive = !shown.wholeChainSelected && cid === shown.subFlapId;
    const btn = html(
        'button',
        {
          type: 'button',
          className: `tool-btn${isActive ? ' active' : ''}`,
          title: `Select sub-flap ${idx + 1}/${sf.stickCount}: ${item.label} (locks upstream sticks, carries downstream tip)`,
        },
        `${idx + 1}. ${item.label}`);
    btn.addEventListener('click', () => {
      selectSubFlap(cid, /* wholeChain= */ false);
    });
    return btn;
  });
  container.replaceChildren(allBtn, ...stepBtns);
}

/** Updates title + mirror badge + river mode availability inside `#flap-shaper-card`. */
function syncShaperHeaderUI() {
  const sf = shown.subFlapId !== null ? shown.subFlaps[shown.subFlapId] : null;
  if (!sf || !shown.model) return;
  const titleText = shown.wholeChainSelected && sf.stickCount > 1 ?
      `${sf.chainSticks.map(words).join(' + ')}` :
      sf.label;
  byId('flap-selected-title').textContent = titleText;

  const isRiver = sf.kind === 'river';
  const narrowModeBtn = document.querySelector('[data-flap-mode="narrow"]');
  const narrowPresetBtn = document.querySelector('[data-flap-preset="narrow"]');
  if (narrowModeBtn) {
    narrowModeBtn.disabled = isRiver;
    narrowModeBtn.title = isRiver ?
        'Rivers connect flaps on both sides; use Bend, Crimp, Curve, or Pleat' :
        'Diagonal side-tuck / taper creases to narrow flap width';
  }
  if (narrowPresetBtn) {
    narrowPresetBtn.disabled = isRiver;
  }
  if (isRiver && shown.flapMode === 'narrow') {
    setFlapModeUI('bend');
  }

  const mirrorId = findMirroredSubFlapId(shown.subFlaps, sf.id);
  const hasMirror = mirrorId !== null || sf.mirrorRegionIdx !== null;
  const showBadge = hasMirror && byId('flap-mirror').checked;
  byId('flap-mirror-badge').hidden = !showBadge;
  if (showBadge) {
    if (shown.wholeChainSelected && sf.mirrorRegionIdx !== null) {
      const mReg = shown.model.regions[sf.mirrorRegionIdx];
      byId('flap-mirror-badge').textContent =
          `↔ ${mReg.sticks.map(words).join(' + ')}`;
    } else if (mirrorId !== null) {
      byId('flap-mirror-badge').textContent =
          `↔ ${shown.subFlaps[mirrorId].label}`;
    }
  }
}

/** Updates the `+N creases` badge from `edges_bp:userShaping`. */
function syncUserCreaseBadgeUI() {
  const badge = byId('flap-crease-badge');
  const flags = shown.rawFoldDict?.['edges_bp:userShaping'];
  const count = Array.isArray(flags) ? flags.filter(Boolean).length : 0;
  badge.hidden = count <= 0;
  if (count > 0) {
    badge.textContent = `+${count} crease${count === 1 ? '' : 's'}`;
  }
}

/**
 * Selects (or deselects) an unmerged sub-flap (or its whole merged chain),
 * highlights it + its mirror partner in 3D and 2D, and opens the Shaper Card.
 *
 * @param {?number} subFlapId
 * @param {boolean=} wholeChain
 */
function selectSubFlap(subFlapId, wholeChain = false) {
  const valid = (subFlapId !== null && subFlapId >= 0 &&
      subFlapId < shown.subFlaps.length) ?
      subFlapId :
      null;

  if ((shown.subFlapId !== valid || shown.wholeChainSelected !== wholeChain) &&
      shown.activeFlapDirty) {
    commitPendingFlapEdit();
  }
  shown.subFlapId = valid;
  shown.wholeChainSelected = Boolean(wholeChain);
  resetFlapSlidersUI(30);

  const {subFlapIds} = activeHighlightedSelection();
  stage?.setHighlight(subFlapIds.length ? subFlapIds : null);

  const card = byId('flap-shaper-card');
  const emptyState = byId('flap-empty-state');
  const sel = byId('flap-select');
  if (valid === null || !shown.model) {
    card.hidden = true;
    emptyState.hidden = shown.subFlaps.length === 0;
    sel.value = '';
  } else {
    // If the user clicked a flap on the 3D model while in another workspace,
    // keep them in `shape` unless they're actively editing CP/Packing/Sticks.
    if (shown.tab === 'export') {
      shown.tab = 'shape';
      syncWorkspaceUI();
    }
    card.hidden = false;
    emptyState.hidden = true;
    sel.value = String(valid);
    syncShaperHeaderUI();
    renderChainStepperUI();

    const sf = shown.subFlaps[valid];
    const elemSel = document.querySelector(
        '#actuator-params [data-param-key="element_name"]');
    if (elemSel && sf) {
      const cands = [sf.stickName, sf.chainSticks.at(-1)];
      for (const cand of cands) {
        const match = [...elemSel.options].find((o) => o.value === cand);
        if (match) {
          elemSel.value = cand;
          break;
        }
      }
    }
    if (shown.fold < 0.25) {
      setFold(1);
    }
  }

  syncFlapHistoryUI();
  shown.lastDiagramKey = null;
  drawDiagram();
}

/** Populates `<select id="flap-select">` and 5 featured quick-pick buttons. */
function renderSubFlapSelector() {
  const showRivers = byId('show-river-chips').checked;
  const flaps = shown.subFlaps.filter((sf) => sf.kind === 'flap');
  const rivers = shown.subFlaps.filter((sf) => sf.kind === 'river');

  const sel = byId('flap-select');
  const placeholder = html(
      'option', {value: ''},
      flaps.length ?
          `— Choose flap / sub-flap (${flaps.length} flaps) —` :
          '— No named flaps in this file —');

  const formatOpt = (sf) => {
    const suffix = sf.stickCount > 1 ?
        ` [${sf.stickIndex + 1}/${sf.stickCount} of ${sf.chainSticks.map(words).join(' → ')}]` :
        '';
    return html('option', {value: String(sf.id)}, `${sf.label}${suffix}`);
  };

  const children = [placeholder];
  if (flaps.length) {
    const grp = html('optgroup', {label: `Flaps (${flaps.length})`});
    grp.append(...flaps.map(formatOpt));
    children.push(grp);
  }
  if (showRivers && rivers.length) {
    const grp = html('optgroup', {label: `Rivers (${rivers.length})`});
    grp.append(...rivers.map(formatOpt));
    children.push(grp);
  }
  sel.replaceChildren(...children);
  if (shown.subFlapId !== null) {
    sel.value = String(shown.subFlapId);
  }

  // Pick up to 5 distinct representative flaps (preferring left/center sticks)
  // for 1-click starter selection in `#quick-flap-pills` without scrolling.
  const seenMirror = new Set();
  const featured = [];
  for (const sf of flaps) {
    if (seenMirror.has(sf.id)) continue;
    const mirrorId = findMirroredSubFlapId(shown.subFlaps, sf.id);
    if (mirrorId !== null) seenMirror.add(mirrorId);
    featured.push(sf);
    if (featured.length >= 5) break;
  }
  byId('quick-flap-pills').replaceChildren(...featured.map((sf) => {
    const btn = html(
        'button',
        {
          type: 'button',
          className: 'mini-btn',
          title: `Select ${sf.label} to shape`,
        },
        sf.label);
    btn.addEventListener('click', () => selectSubFlap(sf.id, false));
    return btn;
  }));
}

/**
 * Evaluates live Flap Shaper sliders against `shown.committedFoldDict` for the
 * selected sub-flap (or whole merged chain) and updates 3D + 2D in place.
 */
function applyLiveFlapSliders() {
  const sf = shown.subFlapId !== null ? shown.subFlaps[shown.subFlapId] : null;
  if (!sf || !shown.committedFoldDict) return;

  const pitchDeg = Number(byId('flap-pitch').value);
  const depthPct = Number(byId('flap-depth').value);
  const steps = Number(byId('flap-steps').value);
  const narrowPct = Number(byId('flap-narrow').value);
  const tuckDeg = Number(byId('flap-tuck-angle').value);
  const narrowDepthPct = Number(byId('flap-narrow-depth').value);
  const bothSides = byId('flap-both-sides').checked;
  const turnDeg = Number(byId('flap-turn').value);
  const crimpDepthPct = Number(byId('flap-crimp-depth').value);

  byId('flap-pitch-val').textContent = `${pitchDeg > 0 ? '+' : ''}${pitchDeg}°`;
  byId('flap-depth-val').textContent = `${depthPct}%`;
  byId('flap-steps-val').textContent = String(steps);
  byId('flap-narrow-val').textContent = `${narrowPct}%`;
  byId('flap-tuck-val').textContent = `${tuckDeg > 0 ? '+' : ''}${tuckDeg}°`;
  byId('flap-narrow-depth-val').textContent = `${narrowDepthPct}%`;
  byId('flap-turn-val').textContent = `${turnDeg > 0 ? '+' : ''}${turnDeg}°`;
  byId('flap-crimp-depth-val').textContent = `${crimpDepthPct}%`;

  let isNeutral = false;
  if (shown.flapMode === 'narrow') {
    isNeutral = narrowPct < 1 || Math.abs(tuckDeg) < 2;
  } else if (shown.flapMode === 'crimp') {
    isNeutral = Math.abs(turnDeg) < 1;
  } else {
    isNeutral = Math.abs(pitchDeg) < 1;
  }

  if (isNeutral) {
    shown.activeFlapDirty = false;
    applyFoldDictInPlace(shown.committedFoldDict);
    syncFlapHistoryUI();
    return;
  }

  const fracStart = shown.wholeChainSelected ? 0 : sf.fracStart;
  const fracEnd = shown.wholeChainSelected ? 1 : sf.fracEnd;
  const effectiveDepth = shown.flapMode === 'narrow' ?
      narrowDepthPct / 100 :
      (shown.flapMode === 'crimp' ? crimpDepthPct / 100 : depthPct / 100);

  // Switch companion map to Live Creases automatically when adding new creases!
  if (shown.tab === 'shape' && shown.shapeMapMode !== 'pattern') {
    setShapeMapMode('pattern');
  }

  const {foldDict} = applyFlapShaping(shown.committedFoldDict, {
    region: sf.regionIdx,
    fracStart,
    fracEnd,
    mirror: byId('flap-mirror').checked,
    mode: shown.flapMode,
    pitchDeg,
    turnDeg,
    narrowPct,
    tuckDeg,
    bothSides,
    depth: effectiveDepth,
    steps,
  });
  foldDict._edited = true;
  shown.activeFlapDirty = true;
  applyFoldDictInPlace(foldDict);
  syncFlapHistoryUI();
}

/**
 * @param {!Object} foldDict
 */
function applyFoldDictInPlace(foldDict) {
  const model = parseFold(foldDict);
  const schedule = buildFoldingSchedule(model, shown.stickFigure);
  const {subFlaps, faceSubFlap} = buildSubFlapCatalog(model, shown.stickFigure);

  shown.rawFoldDict = foldDict;
  shown.model = model;
  shown.schedule = schedule;
  shown.subFlaps = subFlaps;
  shown.faceSubFlap = faceSubFlap;
  shown.lastDiagramKey = null;

  stage.setModel(
      model, shown.entry?.camera, schedule, true, faceSubFlap, subFlaps.length,
      foldDict?.['edges_bp:userShaping'] ?? null);
  const tips = regionTips(model);
  stage.setPins(model.regions.flatMap(
      (region, r) => region.kind === 'flap' && tips[r] >= 0 ?
          [{vertex: tips[r], label: words(region.sticks.at(-1))}] :
          []));
  applyOptions();
  renderScheduleUI(schedule);
  setFold(shown.fold);

  const {subFlapIds} = activeHighlightedSelection();
  stage.setHighlight(subFlapIds.length ? subFlapIds : null);

  const creases = model.assignments.filter((a) => a === 'M' || a === 'V');
  byId('crease-count').textContent = creases.length.toLocaleString('en');
  syncUserCreaseBadgeUI();
  updateStrainMetric();
  refreshDownloadLinks();
  drawDiagram();
}

/** Undoes the last flap shaping step. */
function undoFlapShaping() {
  if (shown.activeFlapDirty && shown.committedFoldDict) {
    resetFlapSlidersUI(30);
    applyFoldDictInPlace(shown.committedFoldDict);
    syncFlapHistoryUI();
    byId('status').textContent = 'Reverted current flap slider.';
    return;
  }
  if (shown.flapUndoStack.length > 0) {
    const prev = shown.flapUndoStack.pop();
    shown.committedFoldDict = prev;
    resetFlapSlidersUI(30);
    applyFoldDictInPlace(prev);
    syncFlapHistoryUI();
    byId('status').textContent = 'Undid previous flap fold.';
  }
}

/** Resets all flap shaping back to `shown.originalFoldDict`. */
function resetAllFlapShaping() {
  if (!shown.originalFoldDict) return;
  shown.flapUndoStack = [];
  shown.committedFoldDict = shown.originalFoldDict;
  resetFlapSlidersUI(30);
  applyFoldDictInPlace(shown.originalFoldDict);
  syncFlapHistoryUI();
  byId('status').textContent = 'Reset model to original fold.';
}

/**
 * @param {!Object} schedule
 */
function renderScheduleUI(schedule) {
  const baseBtn = byId('jump-base');
  const hasStages = schedule.phases.length > 1;
  baseBtn.hidden = !hasStages;
  if (hasStages) {
    baseBtn.dataset.jump = String(schedule.baseCompleteT);
  }

  const track = byId('phase-track');
  track.hidden = !hasStages;
  if (hasStages) {
    const segs = schedule.phases.map((ph, idx) => {
      const seg = html('div', {
        className: 'phase-seg',
        title: `${ph.label} (${Math.round(100 * ph.tStart)}–${Math.round(100 * ph.tEnd)}%)`,
      });
      seg.dataset.act = ph.act ?? 'I';
      seg.dataset.phase = idx;
      seg.style.left = `${(100 * ph.tStart).toFixed(2)}%`;
      seg.style.width = `${Math.max(0.8, 100 * (ph.tEnd - ph.tStart)).toFixed(2)}%`;
      return seg;
    });
    const baseTick = html('div', {
      className: 'base-tick',
      title: `Completed Flat Base (${Math.round(100 * schedule.baseCompleteT)}%)`,
    });
    baseTick.style.left = `${(100 * schedule.baseCompleteT).toFixed(2)}%`;
    track.replaceChildren(...segs, baseTick);
  } else {
    track.replaceChildren();
  }

  const actsPresent = new Set(schedule.phases.map((p) => p.act));
  const actDefs = [
    ['I', 'Act I · Grid'],
    ['II', 'Act II · Base'],
    ['III', 'Act III · Shaping'],
    ['IV', 'Act IV · Standup'],
  ].filter(([code]) => actsPresent.has(code));

  const pills = actDefs.map(([code, text]) => {
    const el = html('span', {className: 'act-pill'}, text);
    el.dataset.act = code;
    return el;
  });
  byId('schedule-meta').replaceChildren(
      html('div', {className: 'act-pills'}, ...pills),
      html('span', {}, `${schedule.phases.length} stages`));

  const list = byId('phase-list');
  list.replaceChildren(...schedule.phases.map((ph, idx) => {
    const act = ph.act ?? 'I';
    const badge = html('span', {className: 'act-pill'}, act);
    badge.dataset.act = act;
    const rangeText =
        `${Math.round(100 * ph.tStart)}–${Math.round(100 * ph.tEnd)}%`;
    const row = html(
        'button',
        {
          type: 'button',
          className: 'phase-item',
          role: 'listitem',
          title: `${ph.label} (${ph.edgeIndices.length} creases, ${rangeText})`,
        },
        badge,
        html('span', {className: 'phase-title'}, shortStageTitle(ph.label)),
        html('span', {className: 'phase-range'}, rangeText));
    row.dataset.phase = idx;
    row.addEventListener('click', () => {
      stopAnimation();
      byId('staged').checked = true;
      setFold(0.5 * (ph.tStart + ph.tEnd));
    });
    return row;
  }));
}

/** Updates local `.fold`, `.svg`, `.json`, and `.bps` download links. */
function refreshDownloadLinks() {
  const {entry, rawFoldDict, model, stickFigure, packingData} = shown;
  const slug = entry?.slug || (entry?.title || 'origami_model')
      .toLowerCase().replace(/[^a-z0-9]+/g, '_');

  const foldLink = byId('download-fold');
  if (foldLink.dataset.blobUrl) URL.revokeObjectURL(foldLink.dataset.blobUrl);
  if (entry?.slug && !rawFoldDict?._edited) {
    foldLink.href = `../models/${entry.slug}/shaped_3d.fold`;
    foldLink.download = `${entry.slug}.fold`;
    delete foldLink.dataset.blobUrl;
  } else if (rawFoldDict) {
    const cleanFold = {...rawFoldDict};
    delete cleanFold._edited;
    const url = URL.createObjectURL(
        new Blob([JSON.stringify(cleanFold, null, 2)], {type: 'application/json'}));
    foldLink.href = url;
    foldLink.download = `${slug}.fold`;
    foldLink.dataset.blobUrl = url;
  } else {
    foldLink.removeAttribute('href');
  }

  const sticksLink = byId('download-sticks');
  if (sticksLink.dataset.blobUrl) URL.revokeObjectURL(sticksLink.dataset.blobUrl);
  if (entry?.slug && stickFigure && !stickFigure._edited) {
    sticksLink.href = `../models/${entry.slug}/stick_figure.json`;
    sticksLink.download = `${entry.slug}_stick_figure.json`;
    delete sticksLink.dataset.blobUrl;
  } else if (stickFigure) {
    const cleanSf = {...stickFigure};
    delete cleanSf._edited;
    const url = URL.createObjectURL(
        new Blob([JSON.stringify(cleanSf, null, 2)], {type: 'application/json'}));
    sticksLink.href = url;
    sticksLink.download = `${slug}_stick_figure.json`;
    sticksLink.dataset.blobUrl = url;
  } else {
    sticksLink.removeAttribute('href');
  }

  if (model) {
    const svgLink = byId('download-svg');
    if (svgLink.href) URL.revokeObjectURL(svgLink.href);
    svgLink.href = URL.createObjectURL(
        new Blob([creasePatternSvg(model)], {type: 'image/svg+xml'}));
    svgLink.download = `${slug}_crease_pattern.svg`;
  }

  byId('download-bps-btn').hidden = !(packingData || stickFigure);
}

/**
 * Sets the companion 2D map mode inside `Shape 3D` (`packing` | `pattern`).
 *
 * @param {string} mode
 */
function setShapeMapMode(mode) {
  shown.shapeMapMode = mode;
  shown.lastDiagramKey = null;
  for (const btn of document.querySelectorAll('[data-shape-map]')) {
    btn.classList.toggle('active', btn.dataset.shapeMap === mode);
  }
  syncFooterLegend();
}

/**
 * Sets the sub-mode inside `Creases` (`shaped` | `base` | `edit`).
 *
 * @param {string} submode
 */
function setCpSubmode(submode) {
  shown.cpSubmode = submode;
  shown.lastDiagramKey = null;
  if (submode === 'edit' && shown.model && cpEditor) {
    cpEditor.setGraph(graphFromModel(shown.model), false);
  }
  syncWorkspaceUI();
  drawDiagram();
}

/** Updates the 1-line context legend at the bottom of the right workspace. */
function syncFooterLegend() {
  const {tab, shapeMapMode} = shown;
  let activeLegend = 'cp';
  if (tab === 'shape') {
    activeLegend = shapeMapMode === 'packing' ? 'packing' : 'cp';
  } else if (tab === 'packing') {
    activeLegend = 'packing';
  } else if (tab === 'sticks') {
    activeLegend = 'sticks';
  } else if (tab === 'export') {
    activeLegend = 'export';
  }
  for (const el of document.querySelectorAll('[data-legend]')) {
    el.hidden = el.dataset.legend !== activeLegend;
  }
}

/** Synchronizes availability of the 5 workspace tabs and sub-controls. */
function syncTabAvailability() {
  const {model, stickFigure, packingData} = shown;
  const hasPacking = Boolean(model?.regions?.length > 0 || packingData || stickFigure);
  const hasBase = Boolean(model?.shaping !== null || shown.hasWasmSolved || stickFigure);

  const available = {
    shape: true,
    pattern: true,
    packing: hasPacking,
    sticks: true,
    export: true,
  };
  for (const tab of byId('tabs').children) {
    tab.disabled = !available[tab.dataset.tab];
  }
  byId('cp-submode-base-btn').disabled = !hasBase;
  if (!hasBase && shown.cpSubmode === 'base') {
    shown.cpSubmode = 'shaped';
  }
  // If a raw crease pattern without named flaps is loaded, default Shape's map to `pattern`.
  if (shown.subFlaps.length === 0 && shown.shapeMapMode === 'packing') {
    setShapeMapMode('pattern');
  }
  if (!available[shown.tab]) {
    selectTab('pattern');
  }
}

/** Synchronizes all workspace headers, toolbars, and active editors. */
function syncWorkspaceUI() {
  const name = shown.tab;
  for (const tab of byId('tabs').children) {
    const selected = tab.dataset.tab === name;
    tab.ariaSelected = String(selected);
  }

  const isShape = name === 'shape';
  const isPattern = name === 'pattern';
  const isPacking = name === 'packing';
  const isSticks = name === 'sticks';
  const isExport = name === 'export';

  byId('flap-picker-box').hidden = !isShape;
  byId('pattern-header').hidden = !isPattern;
  byId('packing-header').hidden = !isPacking;

  // Sub-mode pills inside Creases
  for (const btn of document.querySelectorAll('[data-cp-submode]')) {
    btn.classList.toggle('active', btn.dataset.cpSubmode === shown.cpSubmode);
  }
  const cpEditActive = isPattern && shown.cpSubmode === 'edit';
  const baseViewActive = isPattern && shown.cpSubmode === 'base';
  byId('base-resolve-inline-btn').hidden = !baseViewActive;
  byId('base-toolbar').hidden = !(baseViewActive && shown.editingBase);
  byId('cp-toolbar').hidden = !cpEditActive;

  // Sub-mode pills inside Packing
  const packEditActive = isPacking && shown.editingPacking && Boolean(shown.packingData);
  for (const btn of document.querySelectorAll('[data-pack-submode]')) {
    const wantEdit = btn.dataset.packSubmode === 'edit';
    btn.classList.toggle('active', wantEdit ? packEditActive : !packEditActive);
  }
  byId('packing-toolbar').hidden = !packEditActive;
  byId('pack-validate-btn').hidden = !(packEditActive && cpEditor?.packing?.dirty);

  // Skeleton mode activates the interactive 3D StickFigureEditor immediately!
  byId('sticks-toolbar').hidden = !isSticks;
  byId('stick-stage-wrap').hidden = !isSticks;

  // Stage body visibility
  byId('diagram').hidden = isSticks || isExport;
  byId('export-stage').hidden = !isExport;

  if (cpEditor) {
    cpEditor.mode = isPacking ? 'packing' : 'cp';
    cpEditor.setActive(cpEditActive || packEditActive);
  }
  if (stickEditor) {
    if (isSticks) {
      if (!shown.stickFigure) {
        shown.stickFigure = createStarterStickFigure();
      }
      stickEditor.setStickFigure(shown.stickFigure);
      syncStickInspectorInputs(stickEditor.selectedStick);
      ensureWasmMetadata().catch(() => {});
    }
    stickEditor.setActive(isSticks);
  }
  if (!(cpEditActive || packEditActive)) {
    byId('diagram').style.touchAction = '';
  }
  if (!shown.packingData && shown.stickFigure?.children?.length > 22) {
    byId('pack-method').value = 'stack';
  }

  syncFooterLegend();
}

/**
 * Displays a parsed model and synchronizes all 5 single-screen workspaces.
 *
 * @param {!Object} model
 * @param {?Object} stickFigure
 * @param {!Object} entry
 * @param {?Object=} rawFoldDict
 * @param {!Object=} opts
 */
function show(model, stickFigure, entry, rawFoldDict = null, opts = {}) {
  const schedule = buildFoldingSchedule(model, stickFigure);
  const {subFlaps, faceSubFlap} = buildSubFlapCatalog(model, stickFigure);
  const effectiveFold = rawFoldDict ?? shown.rawFoldDict;

  Object.assign(shown, {
    rawFoldDict: effectiveFold,
    committedFoldDict: effectiveFold,
    originalFoldDict: opts.keepUndoHistory ?
        (shown.originalFoldDict ?? effectiveFold) :
        effectiveFold,
    flapUndoStack: opts.keepUndoHistory ? shown.flapUndoStack : [],
    activeFlapDirty: false,
    subFlaps,
    faceSubFlap,
    subFlapId: null,
    wholeChainSelected: false,
    model,
    stickFigure,
    schedule,
    lastState: null,
    lastDiagramKey: null,
    entry,
  });
  stopAnimation();
  byId('flap-shaper-card').hidden = true;
  byId('flap-empty-state').hidden = subFlaps.length === 0;
  syncFlapHistoryUI();

  stage.setModel(
      model, entry.camera, schedule, Boolean(opts.preserveCamera),
      faceSubFlap, subFlaps.length,
      effectiveFold?.['edges_bp:userShaping'] ?? null);
  const tips = regionTips(model);
  stage.setPins(model.regions.flatMap(
      (region, r) => region.kind === 'flap' && tips[r] >= 0 ?
          [{vertex: tips[r], label: words(region.sticks.at(-1))}] :
          []));
  if (!opts.preserveCamera) {
    byId('color-top').value = entry.colorTop ?? DEFAULT_COLORS.top;
    byId('color-bottom').value = entry.colorBottom ?? DEFAULT_COLORS.bottom;
  }
  applyOptions();
  byId('staged').disabled = schedule.phases.length <= 1 && model.shaping === null;
  renderScheduleUI(schedule);
  setFold(opts.preserveCamera ? shown.fold : 1);

  byId('title').textContent = entry.title;
  byId('header-model-name').textContent = entry.title;
  byId('subtitle').textContent = entry.subtitle ?? '';
  byId('summary').textContent = entry.summary ?? '';
  byId('techniques')
      .replaceChildren(
          ...(entry.techniques ?? []).map((text) => html('li', {}, text)));
  byId('status').textContent = '';
  document.title = `${entry.title} – COrigami Editor`;
  for (const card of byId('gallery').children) {
    card.ariaCurrent = card.dataset.slug === entry.slug ? 'page' : null;
  }

  const creases = model.assignments.filter((a) => a === 'M' || a === 'V');
  byId('grid').textContent =
      model.gridSize ? `${model.gridSize}×${model.gridSize}` : '–';
  byId('stick-count').textContent = stickFigure?.children?.length ?? '–';
  byId('crease-count').textContent = creases.length.toLocaleString('en');
  updateStrainMetric();

  renderSubFlapSelector();
  syncUserCreaseBadgeUI();
  if (!opts.preserveCamera) {
    setShapeMapMode('pattern');
  }

  if (!opts.skipEditorSync && cpEditor) {
    const g = graphFromModel(model);
    if (g.edges.length > 20 && !opts.preserveCamera) {
      setCpToolUI('select');
      const isShaped = Boolean(model.shaping !== null);
      cpEditor.showViolations = !isShaped;
      byId('cp-violations').checked = !isShaped;
    }
    cpEditor.setGraph(g, !opts.preserveCamera);
    byId('cp-grid-size').value = g.gridSize || 16;
  }
  if (!opts.skipStickSync && stickEditor && stickFigure?.children?.length) {
    stickEditor.setStickFigure(stickFigure);
  }

  byId('wasm-actuator-note').textContent = shown.hasWasmSolved ?
      'Applies exact bp_pipeline 2D crease-pattern actuators to the solved base.' :
      'Applies exact bp_pipeline 2D actuators (re-solves showcase base in WASM on first run).';

  renderActuatorParams();
  refreshDownloadLinks();

  if (!opts.preserveCamera) {
    const videoLink = byId('download-video');
    if (videoLink.href) URL.revokeObjectURL(videoLink.href);
    videoLink.removeAttribute('href');
    videoLink.hidden = true;
  }

  syncTabAvailability();
  selectTab(shown.tab);
}

/** Updates the live edge-strain readout in the stage header. */
function updateStrainMetric() {
  if (!shown.model) return;
  const live = shown.lastState?.strainPct;
  const fallback = strainPercent(shown.model);
  const value = (shown.fold > 0.001 && shown.fold < 0.999 && live != null) ?
      live :
      fallback;
  byId('strain').textContent = value == null ? '–' : `${value.toFixed(2)}%`;
}

/**
 * Switches to one of the 5 top-level workspaces (`shape`, `pattern`, `packing`,
 * `sticks`, `export`, plus legacy aliases `base` and `schedule`).
 *
 * @param {string} rawName
 */
function selectTab(rawName) {
  let name = rawName;
  if (rawName === 'base') {
    name = 'pattern';
    shown.cpSubmode = 'base';
  } else if (rawName === 'schedule') {
    name = 'export';
  }
  if (!WORKSPACE_ORDER.includes(name)) {
    name = 'shape';
  }
  shown.tab = name;
  shown.lastDiagramKey = null;

  syncWorkspaceUI();

  if (name === 'export') {
    syncActiveStageHighlight();
  } else if (name !== 'sticks') {
    drawDiagram();
  }
}

/** Redraws the 2D crease pattern only when active folding edges change. */
function maybeDrawPatternDiagram() {
  const showingPattern =
      (shown.tab === 'shape' && shown.shapeMapMode === 'pattern') ||
      (shown.tab === 'pattern' && shown.cpSubmode === 'shaped');
  if (!showingPattern || !shown.model) return;
  const phaseKey = shown.lastState?.diagramKey ?? '';
  const {subFlapIds} = activeHighlightedSelection();
  const key = `${subFlapIds.join(',')}|${phaseKey}`;
  if (key === shown.lastDiagramKey) return;
  shown.lastDiagramKey = key;
  drawDiagram();
}

/** Renders `<svg id="diagram">` for `shape`, `pattern`, or `packing`. */
function drawDiagram() {
  if (shown.tab === 'sticks' || shown.tab === 'export') return;
  const svg = byId('diagram');
  const {model, lastState, packingData, subFlaps} = shown;
  const {subFlapIds, faceSet} = activeHighlightedSelection();

  if (shown.tab === 'pattern' && shown.cpSubmode === 'edit') {
    cpEditor.mode = 'cp';
    cpEditor.render();
    return;
  }
  if (shown.tab === 'packing' && shown.editingPacking && packingData) {
    cpEditor.mode = 'packing';
    cpEditor.render();
    return;
  }

  if (!model) return;

  if (shown.tab === 'packing' ||
      (shown.tab === 'shape' && shown.shapeMapMode === 'packing' && subFlaps.length > 0)) {
    drawPacking(
        svg, model, subFlapIds.length ? subFlapIds : null, {subFlaps});
    return;
  }

  const userShaping = shown.rawFoldDict?.['edges_bp:userShaping'] ?? null;

  if (shown.tab === 'pattern' && shown.cpSubmode === 'base') {
    drawCreasePattern(svg, model, {
      baseOnly: true,
      highlightFaces: faceSet.size ? faceSet : null,
      subFlaps,
    });
    return;
  }

  let focusFaces = null;
  if (shown.tab === 'shape' && shown.subFlapId !== null &&
      byId('zoom-selected-flap').checked) {
    const primary = subFlaps[shown.subFlapId];
    const targetIds = shown.wholeChainSelected ?
        (primary?.chainSubFlapIds ?? []) :
        (primary ? [primary.id] : []);
    const fSet = new Set();
    for (const tid of targetIds) {
      for (const fi of subFlaps[tid]?.faceIndices ?? []) fSet.add(fi);
    }
    if (fSet.size > 0) focusFaces = fSet;
  }

  drawCreasePattern(svg, model, {
    highlightFaces: faceSet.size ? faceSet : null,
    focusFaces,
    activeEdges: lastState?.activeEdges ?? null,
    foldedEdges: lastState?.foldedEdges ?? null,
    userShaping,
    subFlaps,
  });
}

/** Synchronizes multi-stage progress bar & active phase item. */
function syncActiveStageHighlight() {
  const {schedule, lastState, fold} = shown;
  if (!schedule) return;
  const caption = lastState?.label ?? 'Final 3D Sculpted Origami';
  byId('stage-caption').textContent = caption;

  const activeIdx = lastState?.activePhaseIndex ?? -1;
  for (const seg of byId('phase-track').querySelectorAll('.phase-seg')) {
    const idx = Number(seg.dataset.phase);
    const ph = schedule.phases[idx];
    const active = idx === activeIdx ||
        (ph && fold >= ph.tEnd - 1e-6 && activeIdx < 0 && fold > 0.01);
    seg.classList.toggle('active', Boolean(active));
  }
  for (const row of byId('phase-list').querySelectorAll('.phase-item')) {
    const idx = Number(row.dataset.phase);
    if (idx === activeIdx) {
      row.setAttribute('aria-current', 'step');
    } else {
      row.removeAttribute('aria-current');
    }
  }
}

/** Applies paper colors and 3D stage display toggles. */
function applyOptions() {
  const top = byId('color-top').value;
  const bottom = byId('color-bottom').value;
  stage.setColors(top, bottom);
  stage.setCreasesVisible(byId('creases').checked);
  stage.setXray(byId('xray').checked);
  stage.autoRotate = byId('rotate').checked;
  byId('pins').hidden = !byId('show-pins').checked;
  syncSwatches(top, bottom);
}

/**
 * @param {string} top
 * @param {string} bottom
 */
function syncSwatches(top, bottom) {
  const t = top.toLowerCase();
  const b = bottom.toLowerCase();
  for (const swatch of byId('swatches').children) {
    const match = swatch.dataset.top === t && swatch.dataset.bottom === b;
    swatch.setAttribute('aria-pressed', String(match));
  }
}

/**
 * @param {number} progress
 */
function setFold(progress) {
  const clamped = Math.min(1, Math.max(0, progress));
  shown.fold = clamped;
  byId('fold').value = 100 * clamped;
  byId('fold-value').textContent = `${Math.round(100 * clamped)}%`;
  if (!shown.animation) {
    byId('play').textContent = clamped < 0.99 ? 'Fold' : 'Unfold';
  }
  const staged = byId('staged').checked && !byId('staged').disabled;
  shown.lastState = stage?.setFold(clamped, staged) ?? null;
  syncActiveStageHighlight();
  updateStrainMetric();
  maybeDrawPatternDiagram();
}

/** Starts folding / unfolding playback along the schedule. */
function playAnimation() {
  stopAnimation();
  const staged = byId('staged').checked && !byId('staged').disabled;
  const baseCompleteT =
      shown.schedule?.baseCompleteT ?? DEFAULT_BASE_COMPLETE_T;
  const shapingStartT =
      shown.schedule?.shapingStartT ?? DEFAULT_SHAPING_START_T;
  const baseSeconds = staged ? FOLD_SECONDS_STAGED : FOLD_SECONDS_SIMULTANEOUS;
  let direction = shown.fold < 0.99 ? 1 : -1;
  let linear = staged ?
      unwarpFoldingTime(shown.fold, baseCompleteT, shapingStartT) :
      shown.fold;
  let dwellUntil = 0;
  let pausedAtBase =
      (direction > 0 && linear >= baseCompleteT - 0.005) ||
      (direction < 0 && linear <= baseCompleteT + 0.005);
  let previous = performance.now();
  byId('play').textContent = 'Pause';

  const frame = (now) => {
    const dt = Math.min(0.1, (now - previous) / 1000);
    previous = now;
    if (now < dwellUntil) {
      shown.animation = requestAnimationFrame(frame);
      return;
    }
    const speed = SPEEDS[speedIndex];
    const prevLinear = linear;
    linear += direction * (dt * speed) / baseSeconds;

    if (staged && !pausedAtBase) {
      const crossedForward =
          direction > 0 && prevLinear < baseCompleteT && linear >= baseCompleteT;
      const crossedBackward =
          direction < 0 && prevLinear > baseCompleteT && linear <= baseCompleteT;
      if (crossedForward || crossedBackward) {
        linear = baseCompleteT;
        pausedAtBase = true;
        dwellUntil = now + (1000 * BASE_DWELL_SECONDS) / speed;
        setFold(baseCompleteT);
        shown.animation = requestAnimationFrame(frame);
        return;
      }
    }

    if (linear >= 1 || linear <= 0) {
      const end = linear >= 1 ? 1 : 0;
      linear = end;
      setFold(end);
      if (byId('loop').checked) {
        direction = end >= 1 ? -1 : 1;
        pausedAtBase = false;
        dwellUntil = now + 1000 * LOOP_DWELL_SECONDS;
        shown.animation = requestAnimationFrame(frame);
      } else {
        shown.animation = 0;
        byId('play').textContent = end < 0.99 ? 'Fold' : 'Unfold';
      }
      return;
    }

    const progress = staged ?
        warpFoldingTime(linear, baseCompleteT, shapingStartT) :
        linear;
    setFold(progress);
    shown.animation = requestAnimationFrame(frame);
  };
  shown.animation = requestAnimationFrame(frame);
}

/** Stops any active folding animation. */
function stopAnimation() {
  cancelAnimationFrame(shown.animation);
  shown.animation = 0;
}

/**
 * @return {boolean}
 */
function isDarkTheme() {
  const explicit = document.documentElement.dataset.theme;
  if (explicit) return explicit === 'dark';
  return window.matchMedia('(prefers-color-scheme: dark)').matches;
}

/**
 * Records or cancels a 1080p folding video.
 *
 * @return {!Promise<void>}
 */
async function toggleVideoRecording() {
  const recordBtn = byId('record-video');
  const quickBtn = byId('quick-video');
  if (shown.recordingAbort) {
    shown.recordingAbort.abort();
    return;
  }
  if (!shown.model) return;

  stopAnimation();
  const abortController = new AbortController();
  shown.recordingAbort = abortController;

  const prevFold = shown.fold;
  const prevRotate = byId('rotate').checked;
  const staged = byId('staged').checked && !byId('staged').disabled;
  const presetId = byId('video-preset').value;
  const includeCaptions = byId('video-captions').checked;

  byId('rotate').checked = false;
  stage.autoRotate = false;
  recordBtn.textContent = 'Cancel';
  quickBtn.textContent = 'Cancel';
  byId('status').textContent = 'Preparing 1080p video capture…';

  try {
    const result = await recordFoldingVideo({
      stage,
      presetId,
      staged,
      schedule: shown.schedule,
      heroCamera: shown.entry?.camera,
      initialFold: prevFold,
      title: shown.entry?.title ?? 'Origami',
      subtitle: shown.entry?.subtitle ?? '',
      includeCaptions,
      darkTheme: isDarkTheme(),
      signal: abortController.signal,
      onProgress: (fraction, sec, totalSec) => {
        const pct = Math.round(100 * fraction);
        recordBtn.textContent = `Cancel (${pct}%)`;
        quickBtn.textContent = `Cancel (${pct}%)`;
        byId('status').textContent =
            `Recording 1080p video… ${pct}% (${sec.toFixed(1)}s / ${totalSec.toFixed(0)}s)`;
      },
    });

    const slug = shown.entry?.slug ?? 'origami';
    const filename = `${slug}-${presetId}.${result.extension}`;
    const url = URL.createObjectURL(result.blob);
    const saveLink = byId('download-video');
    if (saveLink.href) URL.revokeObjectURL(saveLink.href);
    saveLink.href = url;
    saveLink.download = filename;
    saveLink.textContent = `Save .${result.extension}`;
    saveLink.hidden = false;
    saveLink.click();
    byId('status').textContent = `Saved 1080p video (${filename}).`;
  } catch (error) {
    if (error.name === 'AbortError') {
      byId('status').textContent = 'Video recording cancelled.';
    } else {
      byId('status').textContent = `Could not record video: ${error.message}`;
    }
  } finally {
    shown.recordingAbort = null;
    recordBtn.textContent = 'Record 1080p';
    quickBtn.textContent = '🎥 Video';
    byId('rotate').checked = prevRotate;
    stage.autoRotate = prevRotate;
    setFold(prevFold);
  }
}

/**
 * Loads a showcase model from `../models/<slug>/`.
 *
 * @param {!Object} entry
 * @return {!Promise<void>}
 */
async function showCatalogModel(entry) {
  requested = entry;
  byId('status').textContent = `Loading ${entry.title}…`;
  try {
    const [fold, stickFigure] = await Promise.all([
      fetchJson(`../models/${entry.slug}/shaped_3d.fold`),
      fetchJson(`../models/${entry.slug}/stick_figure.json`),
    ]);
    if (requested !== entry) return;
    shown.packingData = null;
    shown.hasWasmSolved = false;
    shown.shapingElements = [];
    shown.shapingHistory = [];
    shown.editingPacking = false;
    cpEditor?.setPacking(null);
    byId('fidelity-badge').hidden = true;
    renderShapingHistory();

    show(parseFold(fold), stickFigure, entry, fold);
    if (wasm?.isReady) {
      wasm.call('sync_showcase', {
        title: entry.title,
        foldDict: fold,
        stickFigure,
      }).catch(() => {});
    }
  } catch (error) {
    if (requested === entry) {
      byId('status').textContent =
          `Could not load ${entry.title}: ${error.message}`;
    }
  }
}

/**
 * Applies a state snapshot returned from `py_worker.js`.
 *
 * @param {!Object} snap
 * @param {!Object=} opts
 */
function applyWorkerSnapshot(snap, opts = {}) {
  shown.packingData = snap.packing || null;
  shown.hasWasmSolved = Boolean(snap.hasSolved);
  shown.shapingElements = snap.shapingElements || [];
  shown.shapingHistory = snap.shapingHistory || [];

  if (snap.stickFigure) {
    shown.stickFigure = snap.stickFigure;
    stickEditor?.setStickFigure(snap.stickFigure);
  }
  if (snap.packing) {
    cpEditor?.setPacking(snap.packing);
    byId('pack-grid-size').value = snap.packing.gridSize;
  }

  const note = byId('pack-validation');
  if (snap.bpsValidationErrors?.length) {
    note.hidden = false;
    note.textContent = `⚠️ ${snap.bpsValidationErrors[0]}`;
  } else if (snap.packing) {
    note.hidden = false;
    note.textContent = `✓ Valid ${snap.packing.gridSize}×${snap.packing.gridSize} packing (${snap.packing.flaps.length} flaps, ${snap.packing.rivers.length} rivers).`;
  } else {
    note.hidden = true;
  }

  const fidBadge = byId('fidelity-badge');
  if (snap.fidelity?.overall != null) {
    fidBadge.hidden = false;
    fidBadge.textContent = `Score ${snap.fidelity.overall}/10`;
    fidBadge.title = `Pose: ${snap.fidelity.pose}, Width: ${snap.fidelity.width}, Separation: ${snap.fidelity.separation}`;
  } else {
    fidBadge.hidden = true;
  }
  renderShapingHistory();

  if (opts.switchTab) {
    if (opts.switchTab === 'base') {
      shown.tab = 'pattern';
      shown.cpSubmode = 'base';
    } else {
      shown.tab = opts.switchTab;
    }
  }

  if (snap.foldDict) {
    snap.foldDict._edited = true;
    const parsed = parseFold(snap.foldDict);
    const entry = {
      slug: null,
      title: snap.title || shown.entry?.title || parsed.title || 'Custom Design',
      subtitle: snap.hasShaped ?
          '3D Shaped Origami (Python WASM)' :
          (snap.hasSolved ? 'Flat-Foldable Solved Base (Python WASM)' : 'Local Origami Model'),
      summary: 'Designed and folded locally in your browser via COrigami Python WebAssembly.',
      techniques: [
        ...(snap.solveStats ? [`${snap.solveStats.gridSize}×${snap.solveStats.gridSize} Grid`] : []),
        ...(snap.solveStats?.globallyFlatFoldable ? ['Globally Flat-Foldable'] : []),
        ...(snap.hasShaped ? ['3D Shaped'] : []),
      ],
      colorTop: byId('color-top').value,
      colorBottom: byId('color-bottom').value,
      camera: shown.entry?.camera,
    };
    show(parsed, shown.stickFigure, entry, snap.foldDict, {
      preserveCamera: Boolean(opts.preserveCamera),
      skipStickSync: true,
      keepUndoHistory: Boolean(opts.keepUndoHistory),
    });
  } else {
    syncTabAvailability();
    selectTab(shown.tab);
    refreshDownloadLinks();
  }
}

/**
 * Opens or folds any dropped/selected local file (`.fold`, `.svg`, `.cp`,
 * `.bps`, `stick_figure.json`, `.zip`) strictly in browser memory.
 *
 * @param {!Array<!File>} files
 * @return {!Promise<void>}
 */
async function showLocalFiles(files) {
  if (!files.length) return;
  requested = null;
  setGalleryOpen(false);

  if (files.every((f) => /\.(fold|json)$/i.test(f.name))) {
    try {
      const texts = await Promise.all(files.map((f) => f.text()));
      const docs = texts.map((t) => JSON.parse(t));
      const foldIdx = docs.findIndex(
          (d) => Array.isArray(d?.faces_vertices) && d.faces_vertices.length > 0 &&
              (Array.isArray(d?.vertices_coords) || Array.isArray(d?.file_frames)));
      const sfDoc = docs.find((d) => Array.isArray(d?.children) && d?.root);
      if (foldIdx >= 0) {
        const foldDoc = docs[foldIdx];
        const parsed = parseFold(foldDoc);
        if (parsed.flat && parsed.faces.length > 0) {
          shown.packingData = null;
          shown.hasWasmSolved = false;
          shown.editingPacking = false;
          cpEditor?.setPacking(null);
          show(parsed, sfDoc ?? null, {
            title: parsed.title || files[foldIdx].name.replace(/\.fold$/i, ''),
            subtitle: 'Local .fold file (folded in browser)',
            summary: 'Loaded locally in browser memory — zero data uploaded.',
            colorTop: byId('color-top').value,
            colorBottom: byId('color-bottom').value,
          }, foldDoc);
          return;
        }
      }
    } catch (_) {
      // Fall through to Python WASM universal parser.
    }
  }

  try {
    byId('status').textContent = 'Folding file locally in Python WASM…';
    await ensureWasmMetadata();
    const serialized = [];
    for (const file of files) {
      if (file.name.toLowerCase().endsWith('.zip')) {
        const buf = new Uint8Array(await file.arrayBuffer());
        let binary = '';
        for (let i = 0; i < buf.length; i++) binary += String.fromCharCode(buf[i]);
        serialized.push({name: file.name, base64: btoa(binary)});
      } else {
        serialized.push({name: file.name, text: await file.text()});
      }
    }
    const snap = await wasm.call('drop_files', {files: serialized, autoPipeline: true});
    const isPackingOnly = snap.droppedKind === 'bps' && !snap.hasSolved;
    shown.editingPacking = isPackingOnly;
    applyWorkerSnapshot(snap, {switchTab: isPackingOnly ? 'packing' : 'shape'});
    byId('status').textContent =
        `Loaded & folded ${files.map((f) => f.name).join(', ')} locally.`;
  } catch (error) {
    if (!error.cancelled) {
      byId('status').textContent = `Could not fold file: ${error.message}`;
    }
  }
}

/**
 * Boots Pyodide metadata (`examples` + `tools`) on demand.
 *
 * @return {!Promise<void>}
 */
async function ensureWasmMetadata() {
  const info = await wasm.ensureInitialized();
  if (info?.examples?.length) {
    const sel = byId('example-tree-select');
    const current = sel.value;
    sel.replaceChildren(
        html('option', {value: ''}, `— Load built-in stick figure (${info.examples.length} trees) —`),
        ...info.examples.map((name) => html('option', {value: name}, name)));
    sel.value = current;
  }
  if (info?.tools?.length) {
    shapingTools = info.tools;
    populateActuatorDropdown();
  }
}

/** Populates `<select id="actuator-select">` with `bp_pipeline` actuators. */
function populateActuatorDropdown() {
  const sel = byId('actuator-select');
  const prev = sel.value;
  sel.replaceChildren(...shapingTools.map(
      (t) => html('option', {value: t.name}, t.label)));
  if (prev && shapingTools.some((t) => t.name === prev)) sel.value = prev;
  renderActuatorParams();
}

/**
 * @return {!Array<string>}
 */
function activeElementChoices() {
  if (shown.shapingElements.length) {
    return shown.shapingElements.map((e) => e.name);
  }
  if (shown.stickFigure?.children?.length) {
    return shown.stickFigure.children.map((s) => s.name);
  }
  if (shown.subFlaps.length) {
    return shown.subFlaps.map((sf) => sf.stickName);
  }
  return [];
}

/** Renders dynamic parameter inputs for the selected WASM actuator. */
function renderActuatorParams() {
  const toolName = byId('actuator-select').value;
  const spec = shapingTools.find((t) => t.name === toolName) || shapingTools[0];
  const container = byId('actuator-params');
  if (!spec) {
    container.replaceChildren();
    return;
  }
  const elements = activeElementChoices();
  const preferredStick = shown.subFlapId !== null ?
      shown.subFlaps[shown.subFlapId]?.stickName :
      null;
  const controls = spec.params.map((p) => {
    let input;
    if (p.type === 'element') {
      input = html('select', {className: 'mini-select'});
      const opts = elements.length ? elements : ['spine'];
      input.append(...opts.map((name) => html('option', {value: name}, words(name))));
      if (preferredStick && opts.includes(preferredStick)) {
        input.value = preferredStick;
      }
    } else if (p.type === 'select') {
      input = html('select', {className: 'mini-select'});
      input.append(...(p.options || []).map((opt) => html('option', {value: opt}, opt)));
      input.value = p.default;
    } else {
      input = html('input', {
        type: 'number',
        className: 'mini-num',
        min: p.min ?? -180,
        max: p.max ?? 180,
        step: p.step ?? 1,
        value: p.default ?? 0,
      });
    }
    input.dataset.paramKey = p.key;
    input.dataset.paramType = p.type;
    return html('label', {className: 'mini-field'}, `${p.label} `, input);
  });
  container.replaceChildren(...controls);
}

/** Renders recorded WASM actuator history. */
function renderShapingHistory() {
  const list = byId('shaping-history-list');
  if (!shown.shapingHistory.length) {
    list.hidden = true;
    list.replaceChildren();
    return;
  }
  list.hidden = false;
  list.replaceChildren(...shown.shapingHistory.map((item, idx) =>
    html('div', {className: 'history-item'},
        `${idx + 1}. ${item.tool}(${JSON.stringify(item.arguments)})`)));
}

/**
 * @return {?string}
 */
function slugInUrl() {
  const fromQuery = new URLSearchParams(location.search).get('model');
  if (fromQuery) return fromQuery;
  const fromHash = location.hash.replace(/^#/, '');
  return fromHash || null;
}

/**
 * @param {?Object} stick
 */
function syncStickInspectorInputs(stick) {
  const panel = byId('sf-stick-controls');
  if (!stick) {
    panel.hidden = true;
    return;
  }
  panel.hidden = false;
  const az = Math.round(stick.azimuth_angle ?? 0);
  const el = Math.round(stick.elevation_angle ?? 0);
  byId('sf-selected-name').textContent = words(stick.name);
  byId('sf-azimuth').value = az;
  byId('sf-azimuth-range').value = az;
  byId('sf-elevation').value = el;
  byId('sf-elevation-range').value = el;
  byId('sf-length').value = stick.length ?? 1;
  byId('sf-width').value = stick.width ?? 1;
  byId('sf-shaping-flap').checked = Boolean(stick.shaping_flap);
}

/**
 * Initializes the single-screen COrigami Editor.
 *
 * @return {!Promise<void>}
 */
async function main() {
  wasm = new WasmPipelineClient(({busy, ready, label}) => {
    const pill = byId('wasm-status');
    const cancelBtn = byId('wasm-cancel');
    pill.hidden = !busy && !ready;
    pill.textContent = busy ? `⏳ ${label}` : '⚡ Python WASM ready';
    pill.classList.toggle('busy', busy);
    cancelBtn.hidden = !busy;
  });

  byId('wasm-cancel').addEventListener('click', () => {
    wasm.cancel();
    byId('status').textContent = 'Cancelled Python WASM task.';
  });

  cpEditor = new CreasePatternEditor(byId('diagram'), {
    onGraphChanged: (graph) => {
      if (!byId('cp-auto-fold').checked) return;
      try {
        const title = shown.entry?.title || 'Edited Crease Pattern';
        const {foldDict, model} = buildFoldFromCpGraph(graph, title);
        foldDict._edited = true;
        show(model, shown.stickFigure, {
          ...shown.entry,
          slug: null,
          title,
          subtitle: 'Interactive Crease Pattern Edit (Local)',
        }, foldDict, {preserveCamera: true, skipEditorSync: true});
      } catch (err) {
        byId('status').textContent = `Preview note: ${err.message}`;
      }
    },
    onPackingDiagnostics: (diag) => {
      const note = byId('pack-validation');
      note.hidden = false;
      byId('pack-validate-btn').hidden = false;
      if (diag.conflicts.length > 0) {
        const c = diag.conflicts[0];
        note.textContent =
            `⚠️ Overlap: ${words(c.a)} ↔ ${words(c.b)} (distance ${c.actual} < required ${c.required}).`;
      } else if (diag.uncoveredCells > 0) {
        note.textContent =
            `ℹ️ Distances valid! ${diag.uncoveredCells} grid cells uncovered.`;
      } else {
        note.textContent = '✓ Full grid covered & distances valid!';
      }
    },
    onPackingMoved: async (moves, gridSize, diag) => {
      if (diag && (diag.conflicts.length > 0 || diag.uncoveredCells > 0)) {
        return;
      }
      try {
        await ensureWasmMetadata();
        const snap = await wasm.call('update_packing', {flapMoves: moves, gridSize});
        applyWorkerSnapshot(snap, {preserveCamera: true});
      } catch (err) {
        if (!err.cancelled) byId('status').textContent = err.message;
      }
    },
    onSelectionChanged: (sel) => {
      const box = byId('cp-edge-inspector');
      if (!sel) {
        box.hidden = true;
        return;
      }
      box.hidden = false;
      byId('cp-edge-label').textContent =
          `Edge #${sel.edge} (${sel.assignment})`;
      byId('cp-edge-angle').value = Math.round(sel.foldAngle ?? 0);
    },
  });

  stickEditor = new StickFigureEditor(byId('stick-stage'), {
    onChange: (sfDoc, committed) => {
      sfDoc._edited = true;
      shown.stickFigure = sfDoc;
      byId('stick-count').textContent = sfDoc.children.length;
      if (committed) refreshDownloadLinks();
    },
    onSelectStick: (stick) => syncStickInspectorInputs(stick),
  });

  populateActuatorDropdown();

  const [loadedStage, catalog] = await Promise.all([
    Stage.create(byId('canvas'), byId('pins')),
    fetchJson('../models/catalog.json'),
  ]);
  stage = loadedStage;

  stage.onRegionHover?.((subId) => {
    const pill = byId('flap-hover-pill');
    const sf = subId !== null ? shown.subFlaps[subId] : null;
    if (!sf) {
      pill.hidden = true;
      return;
    }
    const chainTag = sf.stickCount > 1 ?
        ` (${sf.stickIndex + 1}/${sf.stickCount} of ${words(sf.chainSticks.at(-1))})` :
        '';
    pill.hidden = false;
    pill.textContent = subId === shown.subFlapId ?
        `✦ Selected ${sf.kind}: ${sf.label}${chainTag}` :
        `👆 Click to shape: ${sf.label}${chainTag}`;
  });
  stage.onRegionClick?.((subId) => {
    if (subId === null) {
      if (shown.subFlapId !== null) selectSubFlap(null);
      return;
    }
    selectSubFlap(subId, false);
  });

  byId('flap-select').addEventListener('change', (e) => {
    const val = e.target.value;
    selectSubFlap(val === '' ? null : Number(val), false);
  });
  byId('show-river-chips').addEventListener('change', renderSubFlapSelector);

  for (const btn of document.querySelectorAll('[data-shape-map]')) {
    btn.addEventListener('click', () => {
      setShapeMapMode(btn.dataset.shapeMap);
      drawDiagram();
    });
  }
  byId('zoom-selected-flap').addEventListener('change', () => {
    shown.lastDiagramKey = null;
    drawDiagram();
  });
  byId('jump-full-cp-btn').addEventListener('click', () => {
    shown.cpSubmode = 'shaped';
    selectTab('pattern');
  });

  for (const btn of document.querySelectorAll('[data-flap-mode]')) {
    btn.addEventListener('click', () => setFlapModeUI(btn.dataset.flapMode));
  }
  for (const id of [
    'flap-pitch',
    'flap-depth',
    'flap-steps',
    'flap-narrow',
    'flap-tuck-angle',
    'flap-narrow-depth',
    'flap-turn',
    'flap-crimp-depth',
  ]) {
    byId(id).addEventListener('input', applyLiveFlapSliders);
  }
  byId('flap-both-sides').addEventListener('change', applyLiveFlapSliders);
  byId('flap-mirror').addEventListener('change', () => {
    if (shown.subFlapId === null || !shown.model) return;
    syncShaperHeaderUI();
    if (shown.activeFlapDirty) {
      applyLiveFlapSliders();
    } else {
      const {subFlapIds} = activeHighlightedSelection();
      stage.setHighlight(subFlapIds.length ? subFlapIds : null);
      drawDiagram();
    }
  });

  for (const btn of document.querySelectorAll('[data-flap-preset]')) {
    btn.addEventListener('click', () => {
      const p = btn.dataset.flapPreset;
      if (p === 'lift') {
        setFlapModeUI('bend');
        byId('flap-pitch').value = 40;
      } else if (p === 'droop') {
        setFlapModeUI('bend');
        byId('flap-pitch').value = -40;
      } else if (p === 'narrow') {
        setFlapModeUI('narrow');
        byId('flap-narrow').value = 55;
        byId('flap-tuck-angle').value = 165;
        byId('flap-both-sides').checked = true;
      } else if (p === 'crimp') {
        setFlapModeUI('crimp');
        byId('flap-turn').value = 35;
        byId('flap-crimp-depth').value = 25;
      } else if (p === 'curl') {
        setFlapModeUI('curve');
        byId('flap-pitch').value = 70;
        byId('flap-depth').value = 20;
        byId('flap-steps').value = 3;
      }
      applyLiveFlapSliders();
    });
  }

  byId('flap-stack-btn').addEventListener('click', () => {
    commitPendingFlapEdit();
    const nextDepth = Math.min(80, Number(byId('flap-depth').value) + 25);
    resetFlapSlidersUI(nextDepth);
    byId('status').textContent =
        'Fold locked in! Drag sliders again or pick the next sub-flap along the chain.';
  });
  byId('flap-done-btn').addEventListener('click', () => selectSubFlap(null));
  byId('flap-undo-btn').addEventListener('click', undoFlapShaping);
  byId('flap-top-undo').addEventListener('click', undoFlapShaping);
  byId('flap-reset-btn').addEventListener('click', resetAllFlapShaping);
  byId('flap-top-reset').addEventListener('click', resetAllFlapShaping);

  const showSlug = (slug) => showCatalogModel(
      catalog.find((entry) => entry.slug === slug) ?? catalog[0]);

  byId('gallery').replaceChildren(...catalog.map((entry) => {
    const card = html(
        'a', {href: `?model=${entry.slug}`}, html('img', {
          src: `../models/${entry.slug}/preview.png`,
          alt: '',
          loading: 'lazy',
        }),
        html('span', {}, entry.title));
    card.dataset.slug = entry.slug;
    card.addEventListener('click', (event) => {
      event.preventDefault();
      setGalleryOpen(false);
      history.pushState(null, '', card.href);
      showSlug(entry.slug);
    });
    return card;
  }));
  window.addEventListener('popstate', () => showSlug(slugInUrl()));
  window.addEventListener('hashchange', () => showSlug(slugInUrl()));

  byId('toggle-gallery').addEventListener('click', (evt) => {
    evt.stopPropagation();
    setGalleryOpen();
  });
  byId('close-gallery').addEventListener('click', () => setGalleryOpen(false));
  document.addEventListener('click', (evt) => {
    const pop = byId('gallery-popover');
    if (!pop.hidden && !pop.contains(evt.target) &&
        !byId('toggle-gallery').contains(evt.target)) {
      setGalleryOpen(false);
    }
  });

  // Creases sub-mode switcher (`Shaped CP` | `Solved Base` | `Draw / Edit CP`).
  for (const btn of document.querySelectorAll('[data-cp-submode]')) {
    btn.addEventListener('click', () => {
      commitPendingFlapEdit();
      setCpSubmode(btn.dataset.cpSubmode);
    });
  }

  const openBlankSheetEditor = () => {
    commitPendingFlapEdit();
    shown.tab = 'pattern';
    shown.cpSubmode = 'edit';
    syncWorkspaceUI();
    setCpToolUI('draw');
    cpEditor.showViolations = true;
    byId('cp-violations').checked = true;
    cpEditor.clearToBlankSheet();
    setFold(0.85);
    byId('status').textContent =
        'Blank square ready! Click-drag on the grid to draw Mountain/Valley creases.';
  };
  byId('cp-quick-blank-btn').addEventListener('click', openBlankSheetEditor);
  byId('cp-clear').addEventListener('click', openBlankSheetEditor);

  byId('base-resolve-inline-btn').addEventListener('click', () => {
    shown.editingBase = !shown.editingBase;
    syncWorkspaceUI();
  });

  for (const btn of document.querySelectorAll('[data-cp-tool]')) {
    btn.addEventListener('click', () => {
      setCpToolUI(btn.dataset.cpTool);
      cpEditor.render();
    });
  }
  for (const btn of document.querySelectorAll('[data-cp-assign]')) {
    btn.addEventListener('click', () => {
      const assign = btn.dataset.cpAssign;
      cpEditor.activeAssignment = assign;
      for (const b of document.querySelectorAll('[data-cp-assign]')) {
        b.classList.toggle('active', b === btn);
      }
      if (cpEditor.selectedEdge >= 0) {
        cpEditor.setSelectedAssignment(assign);
      }
    });
  }
  byId('cp-snap').addEventListener('change', (e) => {
    cpEditor.snapMode = e.target.value;
  });
  byId('cp-angle-lock').addEventListener('change', (e) => {
    cpEditor.angleLock = e.target.value;
  });
  byId('cp-grid-size').addEventListener('change', (e) => {
    cpEditor.setGridSize(e.target.value);
  });
  byId('cp-mirror').addEventListener('change', (e) => {
    cpEditor.mirrorDraw = e.target.checked;
    cpEditor.render();
  });
  byId('cp-violations').addEventListener('change', (e) => {
    cpEditor.showViolations = e.target.checked;
    cpEditor.render();
  });
  byId('cp-auto-fold').addEventListener('change', (e) => {
    cpEditor.autoRefold = e.target.checked;
  });
  byId('cp-zoom-reset').addEventListener('click', () => cpEditor.resetZoom());
  byId('cp-undo').addEventListener('click', () => cpEditor.undo());
  byId('cp-redo').addEventListener('click', () => cpEditor.redo());

  for (const btn of document.querySelectorAll('[data-cp-transform]')) {
    btn.addEventListener('click', () => cpEditor.applyTransform(btn.dataset.cpTransform));
  }
  for (const btn of document.querySelectorAll('[data-angle-step]')) {
    btn.addEventListener('click', () => {
      cpEditor.adjustSelectedFoldAngle(Number(btn.dataset.angleStep));
    });
  }
  byId('cp-edge-angle').addEventListener('change', (e) => {
    cpEditor.adjustSelectedFoldAngle(0, Number(e.target.value));
  });
  byId('cp-edge-delete').addEventListener('click', () => cpEditor.deleteSelectedEdge());

  byId('cp-check-ff').addEventListener('click', async () => {
    const readout = byId('cp-ff-readout');
    readout.hidden = false;
    readout.textContent = 'Running global flat-foldability check in Python WASM…';
    try {
      await ensureWasmMetadata();
      const g = cpEditor.graph;
      const res = await wasm.call('check_global_ff', {
        graph: {
          vertices: g.vertices,
          edges: g.edges,
          assignments: g.assignments,
          foldAngles: g.foldAngles,
        },
      });
      readout.textContent = res.flatFoldable ?
          `✓ Globally flat-foldable! (Local vertex FF: ${res.localFfPercent}%)` :
          `✗ Not globally flat-foldable: ${res.conflictReason || 'layer conflict'} (Local FF: ${res.localFfPercent}%)`;
    } catch (err) {
      if (!err.cancelled) readout.textContent = `Check failed: ${err.message}`;
    }
  });

  byId('cp-refold-wasm').addEventListener('click', async () => {
    try {
      await ensureWasmMetadata();
      const g = cpEditor.graph;
      const snap = await wasm.call('fold_cp_graph', {
        vertices: g.vertices,
        edges: g.edges,
        assignments: g.assignments,
        foldAngles: g.foldAngles,
        gridSize: g.gridSize,
        title: shown.entry?.title || 'Edited Crease Pattern',
      });
      applyWorkerSnapshot(snap, {preserveCamera: true});
    } catch (err) {
      if (!err.cancelled) byId('status').textContent = `Fold error: ${err.message}`;
    }
  });

  byId('example-tree-select').addEventListener('change', async (e) => {
    const name = e.target.value;
    if (!name) return;
    try {
      await ensureWasmMetadata();
      const snap = await wasm.call('load_example_tree', {name});
      shown.packingData = null;
      shown.hasWasmSolved = false;
      cpEditor.setPacking(null);
      applyWorkerSnapshot(snap, {switchTab: 'sticks'});
      syncStickInspectorInputs(stickEditor.selectedStick);
      byId('pack-method').value = snap.treeStats.numSticks > 20 ? 'stack' : 'pack';
      byId('status').textContent =
          `Loaded "${name}" (${snap.treeStats.numSticks} sticks). Drag joints in 3D or click "⚡ Fold Creature"!`;
    } catch (err) {
      if (!err.cancelled) byId('status').textContent = err.message;
    }
  });
  byId('new-tree-btn').addEventListener('click', () => {
    const starter = createStarterStickFigure();
    shown.stickFigure = starter;
    shown.packingData = null;
    shown.hasWasmSolved = false;
    cpEditor.setPacking(null);
    stickEditor.setStickFigure(starter);
    stickEditor.render();
    syncStickInspectorInputs(stickEditor.selectedStick);
    byId('pack-method').value = 'pack';
    refreshDownloadLinks();
    byId('status').textContent =
        'Created fresh 7-stick creature! Drag blue joints in 3D or click "⚡ Fold Creature".';
  });
  byId('sf-mirror').addEventListener('change', (e) => {
    stickEditor.mirrorPose = e.target.checked;
  });
  byId('sf-show-labels').addEventListener('change', (e) => {
    stickEditor.showLabels = e.target.checked;
    stickEditor.render();
  });
  byId('sf-show-axes').addEventListener('change', (e) => {
    stickEditor.showAxes = e.target.checked;
    stickEditor.render();
  });
  byId('sf-reset-cam').addEventListener('click', () => stickEditor.resetCamera());
  byId('sf-add-child').addEventListener('click', () => stickEditor.addChildStick(false));
  byId('sf-add-pair').addEventListener('click', () => stickEditor.addChildStick(true));
  byId('sf-delete').addEventListener('click', () => stickEditor.deleteSelectedStick());

  byId('sf-azimuth-range').addEventListener('input', (e) => {
    const v = Number(e.target.value);
    byId('sf-azimuth').value = v;
    stickEditor.updateSelectedStick({azimuth_angle: v});
  });
  byId('sf-elevation-range').addEventListener('input', (e) => {
    const v = Number(e.target.value);
    byId('sf-elevation').value = v;
    stickEditor.updateSelectedStick({elevation_angle: v});
  });
  for (const [id, key] of [
    ['sf-azimuth', 'azimuth_angle'],
    ['sf-elevation', 'elevation_angle'],
    ['sf-length', 'length'],
    ['sf-width', 'width'],
  ]) {
    byId(id).addEventListener('change', (e) => {
      stickEditor.updateSelectedStick({[key]: Number(e.target.value)});
    });
  }
  byId('sf-shaping-flap').addEventListener('change', (e) => {
    stickEditor.updateSelectedStick({shaping_flap: e.target.checked});
  });

  const runPackFromStickEditor = async () => {
    await ensureWasmMetadata();
    const sfDoc = shown.stickFigure || stickEditor.stickFigure;
    return wasm.call('pack', {
      stickFigure: sfDoc,
      config: {
        packer: byId('pack-method').value,
        maxRetries: Number(byId('pack-retries').value),
        timeoutSeconds: Number(byId('pack-timeout').value),
        allowPairedAxis: byId('pack-paired-axis').checked,
      },
    });
  };

  const handlePackAction = async () => {
    try {
      byId('status').textContent =
          'Packing stick figure onto integer box-pleating grid in Python WASM…';
      const snap = await runPackFromStickEditor();
      shown.editingPacking = true;
      applyWorkerSnapshot(snap, {switchTab: 'packing'});
      setFold(0);
      byId('status').textContent =
          `Packed ${snap.packing.gridSize}×${snap.packing.gridSize} grid! Drag red centers to adjust or click "⚡ Solve Base →".`;
    } catch (err) {
      if (!err.cancelled) {
        byId('status').textContent =
            `Pack note: ${err.message} (Tip: switch Engine to "Fast Stacking" for large trees!)`;
      }
    }
  };

  for (const btn of document.querySelectorAll('[data-pack-submode]')) {
    btn.addEventListener('click', () => {
      if (btn.dataset.packSubmode === 'map') {
        shown.editingPacking = false;
        selectTab('packing');
      } else if (shown.packingData) {
        shown.editingPacking = true;
        selectTab('packing');
      } else {
        handlePackAction();
      }
    });
  }
  byId('sf-pack-btn').addEventListener('click', handlePackAction);
  byId('pack-wasm-btn').addEventListener('click', handlePackAction);

  byId('pack-validate-btn').addEventListener('click', async () => {
    if (!cpEditor.packing) return;
    try {
      await ensureWasmMetadata();
      const snap = await wasm.call('update_packing', {
        flapMoves: [],
        gridSize: cpEditor.packing.gridSize,
      });
      applyWorkerSnapshot(snap, {preserveCamera: true});
    } catch (err) {
      if (!err.cancelled) byId('status').textContent = err.message;
    }
  });

  byId('sf-full-pipeline-btn').addEventListener('click', async () => {
    try {
      byId('status').textContent = 'Step 1/3 · Packing stick figure…';
      const packSnap = await runPackFromStickEditor();
      applyWorkerSnapshot(packSnap);
      byId('status').textContent = 'Step 2/3 · Solving flat-foldable base…';
      const solveSnap = await wasm.call('solve', {
        config: {
          checkLocalFf: byId('sol-local-ff').checked,
          checkGlobalFf: byId('sol-global-ff').checked,
          removeCentralCrease: byId('sol-remove-cc').checked,
          usePartitionOrder: byId('sol-partition').checked,
          disallowTrapping: byId('sol-no-trap').checked,
          maxSteps: Number(byId('sol-steps').value),
          maxTime: Number(byId('sol-time').value),
        },
      });
      applyWorkerSnapshot(solveSnap);
      byId('status').textContent = 'Step 3/3 · 3D posing & shaping…';
      const shapeSnap = await wasm.call('shape', {
        config: {method: 'pose', poseMode: 'figure', narrow: true},
      });
      applyWorkerSnapshot(shapeSnap, {switchTab: 'shape'});
      setFold(1);
      byId('status').textContent =
          '✓ Folded creature in 3D! Click any sub-flap on the 3D model to fine-tune its shape.';
    } catch (err) {
      if (!err.cancelled) byId('status').textContent = `Pipeline error: ${err.message}`;
    }
  });

  byId('pack-mirror').addEventListener('change', (e) => {
    cpEditor.mirrorDraw = e.target.checked;
  });
  byId('pack-show-creases').addEventListener('change', (e) => {
    cpEditor.showPackingCreases = e.target.checked;
    cpEditor.render();
  });
  byId('pack-show-circles').addEventListener('change', (e) => {
    cpEditor.showPackingCircles = e.target.checked;
    cpEditor.render();
  });
  byId('pack-show-labels').addEventListener('change', (e) => {
    cpEditor.showPackingLabels = e.target.checked;
    cpEditor.render();
  });
  byId('pack-grid-size').addEventListener('change', (e) => {
    cpEditor.setGridSize(Number(e.target.value));
    syncWorkspaceUI();
  });

  const runSolveWasm = async () => {
    try {
      byId('status').textContent = 'Solving flat-foldable Mountain/Valley base in Python WASM…';
      await ensureWasmMetadata();
      if (!shown.packingData && shown.stickFigure) {
        await wasm.call('set_stick_figure', {stickFigure: shown.stickFigure});
      }
      const snap = await wasm.call('solve', {
        config: {
          checkLocalFf: byId('sol-local-ff').checked,
          checkGlobalFf: byId('sol-global-ff').checked,
          removeCentralCrease: byId('sol-remove-cc').checked,
          usePartitionOrder: byId('sol-partition').checked,
          disallowTrapping: byId('sol-no-trap').checked,
          maxSteps: Number(byId('sol-steps').value),
          maxTime: Number(byId('sol-time').value),
        },
      });
      shown.editingPacking = false;
      applyWorkerSnapshot(snap, {switchTab: 'base'});
      setFold(1);
      byId('status').textContent =
          `✓ Solved ${snap.solveStats.gridSize}×${snap.solveStats.gridSize} flat-foldable base (${snap.solveStats.numCreases} creases).`;
    } catch (err) {
      if (!err.cancelled) byId('status').textContent = `Solve error: ${err.message}`;
    }
  };
  byId('pack-solve-btn').addEventListener('click', runSolveWasm);
  byId('solve-wasm-btn').addEventListener('click', runSolveWasm);

  byId('base-to-cp-btn').addEventListener('click', () => {
    if (!shown.model) return;
    const g = graphFromModel(shown.model);
    if (shown.model.baseFoldAngles) {
      for (let e = 0; e < g.edges.length; e++) {
        if (g.assignments[e] === 'B') continue;
        const ba = shown.model.baseFoldAngles[e];
        g.foldAngles[e] = ba;
        g.assignments[e] = ba < -1 ? 'M' : (ba > 1 ? 'V' : 'F');
      }
    }
    cpEditor.showViolations = true;
    byId('cp-violations').checked = true;
    cpEditor.setGraph(g, false);
    shown.cpSubmode = 'edit';
    selectTab('pattern');
  });

  byId('actuator-select').addEventListener('change', renderActuatorParams);
  byId('run-reshape-btn').addEventListener('click', async () => {
    try {
      byId('status').textContent = 'Running 3D posing in Python WASM…';
      await ensureWasmMetadata();
      if (!shown.packingData && shown.stickFigure) {
        await wasm.call('set_stick_figure', {stickFigure: shown.stickFigure});
      }
      const snap = await wasm.call('shape', {
        config: {method: 'pose', poseMode: 'figure', narrow: true},
      });
      applyWorkerSnapshot(snap, {preserveCamera: true});
      setFold(1);
      byId('status').textContent = '✓ Auto-posed 3D model locally in Python WASM.';
    } catch (err) {
      if (!err.cancelled) byId('status').textContent = `Shaping error: ${err.message}`;
    }
  });

  byId('apply-actuator-btn').addEventListener('click', async () => {
    try {
      commitPendingFlapEdit();
      await ensureWasmMetadata();
      if (!shown.packingData && shown.stickFigure) {
        await wasm.call('set_stick_figure', {stickFigure: shown.stickFigure});
      }
      const tool = byId('actuator-select').value;
      const args = {};
      for (const el of document.querySelectorAll('#actuator-params [data-param-key]')) {
        const k = el.dataset.paramKey;
        args[k] = el.dataset.paramType === 'number' ? Number(el.value) : el.value;
      }
      const snap = await wasm.call('apply_tool', {
        tool,
        args,
        symmetric: byId('actuator-sym').checked,
      });
      applyWorkerSnapshot(snap, {preserveCamera: true});
      setFold(1);
      byId('status').textContent = snap.toolResult?.message || `Applied ${tool}.`;
    } catch (err) {
      if (!err.cancelled) byId('status').textContent = `Tool error: ${err.message}`;
    }
  });

  byId('download-bps-btn').addEventListener('click', async () => {
    try {
      await ensureWasmMetadata();
      const res = await wasm.call('export_files');
      if (res.bpsJson) {
        triggerBlobDownload(res.bpsJson, `${shown.entry?.title || 'design'}.bps.json`, 'application/json');
      }
    } catch (err) {
      byId('status').textContent = err.message;
    }
  });
  byId('download-zip-btn').addEventListener('click', async () => {
    try {
      commitPendingFlapEdit();
      await ensureWasmMetadata();
      if (shown.rawFoldDict && !shown.packingData) {
        await wasm.call('sync_showcase', {
          title: shown.entry?.title || 'Model',
          foldDict: shown.rawFoldDict,
          stickFigure: shown.stickFigure,
        });
      }
      const res = await wasm.call('export_files');
      if (res.zipBase64) {
        const bin = atob(res.zipBase64);
        const bytes = new Uint8Array(bin.length);
        for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
        const slug = (shown.entry?.title || 'corigami_bundle')
            .toLowerCase().replace(/[^a-z0-9]+/g, '_');
        triggerBlobDownload(bytes, `${slug}.zip`, 'application/zip');
      }
    } catch (err) {
      byId('status').textContent = err.message;
    }
  });

  window.addEventListener('keydown', (evt) => {
    const tag = document.activeElement?.tagName;
    if (tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA') {
      if (evt.key === 'Escape') document.activeElement.blur();
      return;
    }
    if (evt.key === 'Escape') {
      if (!byId('gallery-popover').hidden) {
        setGalleryOpen(false);
        return;
      }
      if (shown.subFlapId !== null) {
        selectSubFlap(null);
        return;
      }
    }
    // Keys 1–5 switch top-level workspaces cleanly when no modifier key is held.
    if (!evt.ctrlKey && !evt.metaKey && !evt.altKey &&
        evt.key >= '1' && evt.key <= '5') {
      const targetTab = WORKSPACE_ORDER[Number(evt.key) - 1];
      const tabBtn = byId('tabs').querySelector(`[data-tab="${targetTab}"]`);
      if (tabBtn && !tabBtn.disabled) {
        selectTab(targetTab);
        return;
      }
    }
    if (shown.tab === 'pattern' && shown.cpSubmode === 'edit') {
      if ((evt.ctrlKey || evt.metaKey) && evt.key.toLowerCase() === 'z') {
        evt.preventDefault();
        if (evt.shiftKey) cpEditor.redo();
        else cpEditor.undo();
        return;
      }
      if ((evt.ctrlKey || evt.metaKey) && evt.key.toLowerCase() === 'y') {
        evt.preventDefault();
        cpEditor.redo();
        return;
      }
      const keyMap = {d: 'draw', a: 'assign', s: 'select', v: 'move', e: 'erase'};
      const mapped = keyMap[evt.key.toLowerCase()];
      if (mapped) {
        setCpToolUI(mapped);
        cpEditor.render();
      } else if (evt.key === 'Delete' || evt.key === 'Backspace') {
        cpEditor.deleteSelectedEdge();
      }
      return;
    }
    if ((evt.ctrlKey || evt.metaKey) && evt.key.toLowerCase() === 'z') {
      if (shown.activeFlapDirty || shown.flapUndoStack.length > 0) {
        evt.preventDefault();
        undoFlapShaping();
      }
    }
  });

  byId('swatches').replaceChildren(...WASHI_PRESETS.map((preset) => {
    const button = html('button', {
      type: 'button',
      className: 'swatch',
      title: preset.name,
      ariaLabel: preset.name,
      ariaPressed: 'false',
    });
    button.dataset.top = preset.top.toLowerCase();
    button.dataset.bottom = preset.bottom.toLowerCase();
    button.style.setProperty('--swatch-top', preset.top);
    button.style.setProperty('--swatch-bottom', preset.bottom);
    button.addEventListener('click', () => {
      byId('color-top').value = preset.top;
      byId('color-bottom').value = preset.bottom;
      applyOptions();
    });
    return button;
  }));

  byId('fold').addEventListener('input', (event) => {
    stopAnimation();
    setFold(event.target.value / 100);
  });
  byId('play').addEventListener('click', () => {
    if (shown.animation) {
      stopAnimation();
      setFold(shown.fold);
    } else {
      playAnimation();
    }
  });
  byId('loop').addEventListener('change', () => {
    if (byId('loop').checked && !shown.animation) {
      playAnimation();
    }
  });
  byId('staged').addEventListener('change', () => setFold(shown.fold));
  byId('stage-caption').addEventListener('click', () => selectTab('export'));
  for (const button of document.querySelectorAll('[data-jump]')) {
    button.addEventListener('click', () => {
      stopAnimation();
      setFold(Number(button.dataset.jump));
    });
  }
  byId('speed').addEventListener('click', (event) => {
    speedIndex = (speedIndex + 1) % SPEEDS.length;
    event.currentTarget.textContent = `${SPEEDS[speedIndex]}×`;
  });

  const presetSelect = byId('video-preset');
  presetSelect.replaceChildren(...VIDEO_PRESETS.map(
      (preset) => html(
          'option', {value: preset.id, title: preset.description},
          preset.label)));
  const syncPresetTooltip = () => {
    const active =
        VIDEO_PRESETS.find((p) => p.id === presetSelect.value) ??
        VIDEO_PRESETS[0];
    presetSelect.title = active.description;
  };
  syncPresetTooltip();
  presetSelect.addEventListener('change', syncPresetTooltip);
  const canRecordVideo = pickSupportedVideoFormat() !== null;
  byId('record-video').disabled = !canRecordVideo;
  byId('quick-video').disabled = !canRecordVideo;
  byId('record-video').addEventListener('click', toggleVideoRecording);
  byId('quick-video').addEventListener('click', toggleVideoRecording);

  const displayOptions =
      ['color-top', 'color-bottom', 'creases', 'xray', 'rotate', 'show-pins'];
  for (const id of displayOptions) {
    byId(id).addEventListener('input', applyOptions);
  }
  for (const button of document.querySelectorAll('[data-view]')) {
    button.addEventListener('click', () => stage.setView(button.dataset.view));
  }
  stage.onInteraction(() => {
    byId('rotate').checked = false;
    applyOptions();
  });
  stage.onViewChange((view) => {
    shown.view = view;
  });

  for (const tab of byId('tabs').children) {
    tab.addEventListener('click', () => selectTab(tab.dataset.tab));
  }
  byId('diagram').addEventListener('click', (event) => {
    if (cpEditor?.active || stickEditor?.active) return;
    const {subflap, region} = event.target.dataset;
    if (subflap !== undefined) {
      const idx = Number(subflap);
      const toggleOff = shown.subFlapId === idx && !shown.wholeChainSelected;
      selectSubFlap(toggleOff ? null : idx, false);
    } else if (region !== undefined) {
      const rIdx = Number(region);
      const firstSub = shown.subFlaps.find((sf) => sf.regionIdx === rIdx);
      if (firstSub) selectSubFlap(firstSub.id, true);
    }
  });
  byId('theme').addEventListener('click', () => {
    document.documentElement.dataset.theme = isDarkTheme() ? 'light' : 'dark';
  });

  byId('file').addEventListener('change', (event) => {
    showLocalFiles([...event.target.files]);
    event.target.value = '';
  });

  let dragDepth = 0;
  window.addEventListener('dragenter', (event) => {
    if (event.dataTransfer?.types?.includes('Files')) {
      dragDepth++;
      byId('drop-overlay').hidden = false;
    }
  });
  window.addEventListener('dragover', (event) => event.preventDefault());
  window.addEventListener('dragleave', () => {
    dragDepth = Math.max(0, dragDepth - 1);
    if (dragDepth === 0) byId('drop-overlay').hidden = true;
  });
  window.addEventListener('drop', (event) => {
    event.preventDefault();
    dragDepth = 0;
    byId('drop-overlay').hidden = true;
    showLocalFiles([...event.dataTransfer.files]);
  });

  await showSlug(slugInUrl());
}

main().catch((error) => {
  byId('status').textContent = `Could not start: ${error.message}`;
});
