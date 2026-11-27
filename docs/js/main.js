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
 * @fileoverview The studio page: loads models and wires up the controls.
 *
 * A showcase model is a directory `models/<slug>/` holding `shaped_3d.fold`,
 * `stick_figure.json` (plus `shaping_history.json` on `castor-and-pollux` and
 * `spinosaurus`) and `preview.png`, plus an entry in `models/catalog.json` with
 * descriptive text, colors and camera angle. Both showcase models and local
 * `.fold` files dropped onto the page fold through an automatic 4-Act
 * Multi-Stage Folding Schedule.
 */

import {creasePatternSvg, drawCreasePattern, drawPacking, drawStickFigure} from './diagram.js';
import {parseFold, regionTips, strainPercent} from './fold.js';
import {DEFAULT_BASE_COMPLETE_T, DEFAULT_SHAPING_START_T, buildFoldingSchedule, unwarpFoldingTime, warpFoldingTime} from './schedule.js';
import {Stage} from './stage.js';
import {VIDEO_PRESETS, pickSupportedVideoFormat, recordFoldingVideo} from './video.js';

/** Base duration of a full multi-stage fold at 1x speed, in seconds. */
const FOLD_SECONDS_STAGED = 14;

/** Duration of a simultaneous unstaged fold at 1x speed, in seconds. */
const FOLD_SECONDS_SIMULTANEOUS = 6;

/** Pause at each end of a continuous back-and-forth fold loop, in seconds. */
const LOOP_DWELL_SECONDS = 0.55;

/** Brief inspection dwell on the completed razor-flat base during playback. */
const BASE_DWELL_SECONDS = 0.6;

/** Selectable playback speed multipliers. */
const SPEEDS = [1, 0.5, 2];

/** Paper colors of a model that does not bring its own. */
const DEFAULT_COLORS = {
  top: '#9e3329',
  bottom: '#f3e7ce'
};

/** Curated duo-tone Japanese washi paper presets. */
const WASHI_PRESETS = [
  {name: 'Vermilion & Kozo Cream', top: '#9e3329', bottom: '#f3e7ce'},
  {name: 'Aizome Indigo & Ivory', top: '#24476e', bottom: '#f4efe4'},
  {name: 'Matcha Jade & Warm Sand', top: '#2f6a52', bottom: '#efe4c8'},
  {name: 'Kitsune Ochre & Sumi', top: '#c4832b', bottom: '#2b2d33'},
  {name: 'Ume Plum & Sakura Mist', top: '#7a2e52', bottom: '#f6e6ea'},
  {name: 'Suzu Slate & Snow Washi', top: '#475569', bottom: '#f8fafc'},
];

/**
 * Looks up a DOM element by its ID.
 *
 * @param {string} id
 * @return {?HTMLElement}
 */
const byId = (id) => document.getElementById(id);

/** @type {?Stage} */
let stage = null;

/** Current index into `SPEEDS`. */
let speedIndex = 0;

/** The model on screen and how it is shown. */
const shown = {
  /** @type {?Object} Parsed model from `fold.js`. */
  model: null,
  /** @type {?Object} Parsed stick figure JSON. */
  stickFigure: null,
  /** @type {?Object} Multi-stage folding schedule. */
  schedule: null,
  /** @type {?Object} Latest evaluated stage info from `Stage.setFold`. */
  lastState: null,
  /** @type {?string} Key of the last rendered crease-pattern SVG state. */
  lastDiagramKey: null,
  /** @type {?Object} Catalog entry; a local file has one without a `slug`. */
  entry: null,
  /** Folding progress from 0 (flat) to 1 (folded). */
  fold: 1,
  /** Highlighted region of the model, or null. */
  region: null,
  /** Selected inspector tab: 'pattern', 'base', 'packing', 'sticks', 'schedule'. */
  tab: 'pattern',
  /** Camera rotation of the stage, shared with the stick figure drawing. */
  view: [1, 0, 0, 0, 1, 0, 0, 0, 1],
  /** `requestAnimationFrame` handle while the folding animation plays. */
  animation: 0,
  /** `AbortController` while a 1080p video recording is in progress. */
  recordingAbort: null,
};

/**
 * The catalog entry asked for last, so that a model which loads slowly cannot
 * replace one that was asked for after it.
 */
let requested = null;

/**
 * Creates an HTML element with properties and children.
 *
 * @param {string} tag HTML tag name.
 * @param {!Object<string, *>=} properties DOM properties to assign.
 * @param {...(string|!Node)} children Child nodes or strings.
 * @return {!HTMLElement}
 */
function html(tag, properties = {}, ...children) {
  const element = Object.assign(document.createElement(tag), properties);
  element.append(...children);
  return element;
}

/**
 * Turns a stick name such as `beak_hook` into words.
 *
 * @param {string} stickName
 * @return {string}
 */
const words = (stickName) => stickName.replaceAll('_', ' ');

/**
 * Fetches and parses a JSON document.
 *
 * @param {string} url
 * @return {!Promise<!Object>}
 */
async function fetchJson(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url}: HTTP ${response.status}`);
  return response.json();
}

/**
 * Strips the leading `Act X.Y · ` prefix when displaying a row next to its
 * dedicated Act pill.
 *
 * @param {string} label
 * @return {string}
 */
function shortStageTitle(label) {
  return label.replace(/^Act\s+[IVX]+(?:\.\d+)?\s*·\s*/i, '');
}

/**
 * Renders the segmented timeline bar and the interactive Multi-Stage Folding
 * Schedule list for `schedule`.
 *
 * @param {!Object} schedule
 */
function renderScheduleUI(schedule) {
  const track = byId('phase-track');
  const segments = schedule.phases.map((ph, idx) => {
    const seg = html('div', {className: 'phase-seg'});
    seg.dataset.act = ph.act ?? (ph.stageGroup === 'collapse' ? 'I' : 'III');
    seg.dataset.phase = idx;
    seg.style.left = `${100 * ph.tStart}%`;
    seg.style.width = `${Math.max(0.8, 100 * (ph.tEnd - ph.tStart))}%`;
    return seg;
  });
  if (schedule.baseCompleteT < 0.99) {
    const baseTick = html('div', {className: 'base-tick'});
    baseTick.style.left = `${100 * schedule.baseCompleteT}%`;
    segments.push(baseTick);
  }
  track.replaceChildren(...segments);

  const jumpBase = byId('jump-base');
  const hasBaseSplit = schedule.baseCompleteT < 0.99;
  jumpBase.hidden = !hasBaseSplit;
  jumpBase.dataset.jump =
      String(schedule.baseCompleteT ?? DEFAULT_BASE_COMPLETE_T);

  const actCounts = new Map();
  for (const ph of schedule.phases) {
    const act = ph.act ?? 'I';
    actCounts.set(act, (actCounts.get(act) ?? 0) + 1);
  }
  const actLabels = {
    I: 'Act I Collapse',
    II: 'Act II Flat-Clip',
    III: 'Act III 3D Pose',
    IV: 'Act IV Sculpt',
  };
  const pills = [...actCounts.keys()].map((act) => {
    const pill = html('span', {className: 'act-pill'}, actLabels[act] ?? `Act ${act}`);
    pill.dataset.act = act;
    return pill;
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

/**
 * Shows a model and everything derived from it.
 *
 * @param {!Object} model Parsed model from `fold.js`.
 * @param {?Object} stickFigure Parsed stick figure JSON, if known.
 * @param {!Object} entry Catalog entry, or its stand-in for a local file.
 */
function show(model, stickFigure, entry) {
  const schedule = buildFoldingSchedule(model, stickFigure);
  Object.assign(shown, {
    model,
    stickFigure,
    schedule,
    lastState: null,
    lastDiagramKey: null,
    entry,
    region: null,
  });
  stopAnimation();

  stage.setModel(model, entry.camera, schedule);
  const tips = regionTips(model);
  stage.setPins(model.regions.flatMap(
      (region, r) => region.kind === 'flap' && tips[r] >= 0 ?
          [{vertex: tips[r], label: words(region.sticks.at(-1))}] :
          []));
  byId('color-top').value = entry.colorTop ?? DEFAULT_COLORS.top;
  byId('color-bottom').value = entry.colorBottom ?? DEFAULT_COLORS.bottom;
  applyOptions();
  byId('staged').disabled = schedule.phases.length <= 1 && model.shaping === null;
  renderScheduleUI(schedule);
  setFold(1);

  byId('title').textContent = entry.title;
  byId('subtitle').textContent = entry.subtitle ?? '';
  byId('summary').textContent = entry.summary ?? '';
  byId('techniques')
      .replaceChildren(
          ...(entry.techniques ?? []).map((text) => html('li', {}, text)));
  byId('status').textContent = '';
  document.title = `${entry.title} – COrigami Studio`;
  for (const card of byId('gallery').children) {
    card.ariaCurrent = card.dataset.slug === entry.slug ? 'page' : null;
  }

  const creases = model.assignments.filter((a) => a === 'M' || a === 'V');
  byId('grid').textContent =
      model.gridSize ? `${model.gridSize}×${model.gridSize}` : '–';
  byId('stick-count').textContent = stickFigure?.children.length ?? '–';
  byId('crease-count').textContent = creases.length.toLocaleString('en');
  updateStrainMetric();

  byId('regions').replaceChildren(...model.regions.map((region, r) => {
    const hasSticks = region.sticks.length > 0;
    const chip = html(
        'button', {
          type: 'button',
          className: region.kind,
          title: hasSticks ?
              `${region.kind}: ${region.sticks.map(words).join(' + ')}` :
              region.kind,
          ariaPressed: 'false',
        },
        hasSticks ? words(region.sticks.at(-1)) : region.kind);
    chip.addEventListener('click', () => toggleHighlight(r));
    return chip;
  }));

  const downloads = {
    'download-fold': 'shaped_3d.fold',
    'download-sticks': 'stick_figure.json',
  };
  for (const [id, file] of Object.entries(downloads)) {
    if (entry.slug) {
      byId(id).href = `models/${entry.slug}/${file}`;
    } else {
      byId(id).removeAttribute('href');
    }
  }
  const svgLink = byId('download-svg');
  URL.revokeObjectURL(svgLink.href);
  svgLink.href = URL.createObjectURL(
      new Blob([creasePatternSvg(model)], {type: 'image/svg+xml'}));
  const videoLink = byId('download-video');
  if (videoLink.href) URL.revokeObjectURL(videoLink.href);
  videoLink.removeAttribute('href');
  videoLink.hidden = true;

  const available = {
    pattern: true,
    base: model.shaping !== null,
    packing: model.regions.length > 0,
    sticks: stickFigure !== null,
    schedule: schedule.phases.length > 0,
  };
  for (const tab of byId('tabs').children) {
    tab.disabled = !available[tab.dataset.tab];
  }
  selectTab(available[shown.tab] ? shown.tab : 'pattern');
}

/** Updates the Strain metric readout from the current or final pose. */
function updateStrainMetric() {
  if (!shown.model) return;
  const live = shown.lastState?.strainPct;
  const fallback = strainPercent(shown.model);
  const value = (shown.fold > 0.001 && shown.fold < 0.999 && live != null) ?
      live :
      fallback;
  byId('strain').textContent = value == null ? '–' : `${value.toFixed(2)}%`;
}

/** Switches the inspector panel to the tab called `name`. */
function selectTab(name) {
  shown.tab = name;
  shown.lastDiagramKey = null;
  for (const tab of byId('tabs').children) {
    const selected = tab.dataset.tab === name;
    tab.ariaSelected = String(selected);
    if (selected) byId('diagram').ariaLabel = tab.textContent;
  }
  const isSchedule = name === 'schedule';
  byId('diagram').hidden = isSchedule;
  byId('schedule-panel').hidden = !isSchedule;
  for (const legend of document.querySelectorAll('[data-tabs]')) {
    legend.hidden = !legend.dataset.tabs.split(' ').includes(name);
  }
  if (!isSchedule) {
    if (name === 'pattern') {
      maybeDrawPatternDiagram();
    } else {
      drawDiagram();
    }
  } else {
    syncActiveStageHighlight();
  }
}

/** Redraws the crease-pattern SVG only when active/folded phases changed. */
function maybeDrawPatternDiagram() {
  if (shown.tab !== 'pattern' || !shown.model) return;
  const phaseKey = shown.lastState?.diagramKey ?? '';
  const key = `${shown.region ?? 'none'}|${phaseKey}`;
  if (key === shown.lastDiagramKey) return;
  shown.lastDiagramKey = key;
  drawDiagram();
}

/** Draws the SVG diagram of the selected tab. */
function drawDiagram() {
  if (shown.tab === 'schedule' || !shown.model) return;
  const svg = byId('diagram');
  const {model, stickFigure, region, view, lastState} = shown;
  switch (shown.tab) {
    case 'base':
      drawCreasePattern(svg, model, {baseOnly: true, highlight: region});
      break;
    case 'packing':
      drawPacking(svg, model, region);
      break;
    case 'sticks':
      drawStickFigure(svg, stickFigure, view);
      break;
    default:
      drawCreasePattern(svg, model, {
        highlight: region,
        activeEdges: lastState?.activeEdges ?? null,
        foldedEdges: lastState?.foldedEdges ?? null,
      });
  }
}

/**
 * Syncs active stage highlighting across the caption button, timeline bar,
 * and scrollable Multi-Stage Folding Schedule list.
 */
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
        (ph && fold >= ph.tStart - 0.002 && fold <= ph.tEnd + 0.002 &&
         fold > 0.002 && fold < 0.998);
    seg.classList.toggle('active', Boolean(active));
  }

  const list = byId('phase-list');
  let activeRow = null;
  for (const row of list.children) {
    const idx = Number(row.dataset.phase);
    const ph = schedule.phases[idx];
    const current = idx === activeIdx ||
        (activeIdx < 0 && ph && fold > 0.002 && fold < 0.998 &&
         fold >= ph.tStart && fold <= ph.tEnd);
    if (current) {
      row.setAttribute('aria-current', 'step');
      if (!activeRow) activeRow = row;
    } else {
      row.removeAttribute('aria-current');
    }
  }
  if (activeRow && !byId('schedule-panel').hidden) {
    const rowTop = activeRow.offsetTop - list.offsetTop;
    const rowBottom = rowTop + activeRow.offsetHeight;
    if (rowTop < list.scrollTop) {
      list.scrollTop = Math.max(0, rowTop - 8);
    } else if (rowBottom > list.scrollTop + list.clientHeight) {
      list.scrollTop = rowBottom - list.clientHeight + 8;
    }
  }
}

/**
 * Picks out a region on the 3D model and the diagram, or stops picking it
 * out if it already is.
 *
 * @param {number} region Index into the regions of the model.
 */
function toggleHighlight(region) {
  shown.region = shown.region === region ? null : region;
  shown.lastDiagramKey = null;
  stage.setHighlight(shown.region);
  if (shown.tab === 'pattern') {
    maybeDrawPatternDiagram();
  } else {
    drawDiagram();
  }
  [...byId('regions').children].forEach((chip, r) => {
    chip.ariaPressed = String(r === shown.region);
  });
}

/** Applies the paper colors and the display toggles to the stage. */
function applyOptions() {
  const top = byId('color-top').value;
  const bottom = byId('color-bottom').value;
  stage.setColors(top, bottom);
  syncSwatches(top, bottom);
  stage.setCreasesVisible(byId('creases').checked);
  stage.setXray(byId('xray').checked);
  stage.autoRotate = byId('rotate').checked;
  byId('pins').hidden = !byId('show-pins').checked;
}

/**
 * Updates `aria-pressed` on duo-tone paper preset buttons.
 *
 * @param {string} top Current display-side hex color.
 * @param {string} bottom Current reverse-side hex color.
 */
function syncSwatches(top, bottom) {
  const normTop = top.toLowerCase();
  const normBottom = bottom.toLowerCase();
  for (const button of byId('swatches').children) {
    const match = button.dataset.top === normTop &&
        button.dataset.bottom === normBottom;
    button.ariaPressed = String(match);
  }
}

/** Folds the model to `progress`, from 0 (flat) to 1 (folded). */
function setFold(progress) {
  const clamped = Math.max(0, Math.min(1, progress));
  shown.fold = clamped;
  byId('fold').value = 100 * clamped;
  byId('fold-value').textContent = `${Math.round(100 * clamped)}%`;
  const direction = clamped > 0.5 ? 'Unfold' : 'Fold';
  byId('play').textContent = shown.animation ? 'Pause' : direction;
  shown.lastState = stage.setFold(clamped, byId('staged').checked);
  syncActiveStageHighlight();
  updateStrainMetric();
  maybeDrawPatternDiagram();
}

/**
 * Smoothly animates folding towards whichever end is farther away, using
 * decelerated milestone time-warping in multi-stage mode so both the
 * collapsed base and final shaped pose settle calmly.
 */
function playAnimation() {
  let target = shown.fold > 0.5 ? 0 : 1;
  const staged = byId('staged').checked && shown.schedule !== null;
  const baseT = shown.schedule?.baseCompleteT ?? DEFAULT_BASE_COMPLETE_T;
  const shapeStartT = shown.schedule?.shapingStartT ?? DEFAULT_SHAPING_START_T;

  const toClock = (foldVal) => staged ?
      unwarpFoldingTime(foldVal, baseT, shapeStartT) :
      Math.acos(1 - 2 * Math.min(1, Math.max(0, foldVal))) / Math.PI;
  const fromClock = (uVal) => staged ?
      warpFoldingTime(uVal, baseT, shapeStartT) :
      0.5 - 0.5 * Math.cos(Math.PI * uVal);

  let u = toClock(shown.fold);
  let pausedAtBase = target === 1 ? shown.fold >= baseT - 1e-3 :
                                    shown.fold <= shapeStartT + 1e-3;
  let dwellUntil = 0;
  let last = performance.now();

  const step = (now) => {
    const dt = Math.min(0.05, (now - last) / 1000);
    last = now;
    if (now < dwellUntil) {
      shown.animation = requestAnimationFrame(step);
      return;
    }
    const useStaged = byId('staged').checked && shown.schedule !== null;
    const duration =
        (useStaged ? FOLD_SECONDS_STAGED : FOLD_SECONDS_SIMULTANEOUS) /
        SPEEDS[speedIndex];
    const delta = dt / duration;
    const nextU = target === 1 ? Math.min(1, u + delta) : Math.max(0, u - delta);
    const nextFold = fromClock(nextU);

    // Pause briefly on the razor-flat base during forward multi-stage folding.
    if (useStaged && target === 1 && !pausedAtBase && baseT < 0.95 &&
        shown.fold < baseT && nextFold >= baseT) {
      pausedAtBase = true;
      u = unwarpFoldingTime(baseT, baseT, shapeStartT);
      dwellUntil = now + 1000 * BASE_DWELL_SECONDS;
      shown.animation = requestAnimationFrame(step);
      setFold(baseT);
      return;
    }

    u = nextU;
    if (u === target) {
      if (byId('loop').checked) {
        target = 1 - target;
        pausedAtBase = false;
        dwellUntil = now + 1000 * LOOP_DWELL_SECONDS;
        shown.animation = requestAnimationFrame(step);
      } else {
        shown.animation = 0;
      }
    } else {
      shown.animation = requestAnimationFrame(step);
    }
    setFold(nextFold);
  };
  shown.animation = requestAnimationFrame(step);
  setFold(shown.fold);
}

/** Stops any running folding animation. */
function stopAnimation() {
  cancelAnimationFrame(shown.animation);
  shown.animation = 0;
}

/**
 * Returns whether the page is currently displaying the dark color scheme.
 *
 * @return {boolean}
 */
function isDarkTheme() {
  const root = document.documentElement;
  return root.dataset.theme ?
      root.dataset.theme === 'dark' :
      matchMedia('(prefers-color-scheme: dark)').matches;
}

/**
 * Starts recording a 1080p folding & showcase video (or cancels an active
 * recording if one is already running).
 *
 * @return {!Promise<void>}
 */
async function toggleVideoRecording() {
  if (shown.recordingAbort) {
    shown.recordingAbort.abort();
    return;
  }
  if (!shown.model || !stage) return;

  stopAnimation();
  const abort = new AbortController();
  shown.recordingAbort = abort;

  const recordBtn = byId('record-video');
  const quickBtn = byId('quick-video');
  const presetSelect = byId('video-preset');
  const presetId = presetSelect.value || 'guided';
  const staged = byId('staged').checked && shown.schedule !== null;
  const includeCaptions = byId('video-captions').checked;
  const initialFold = shown.fold;

  const setRecordingLabels = (pct) => {
    const label = `Cancel (${pct}%)`;
    recordBtn.textContent = label;
    quickBtn.textContent = label;
    byId('status').textContent = `Recording 1080p video… ${pct}%`;
  };
  setRecordingLabels(0);

  try {
    const result = await recordFoldingVideo({
      stage,
      presetId,
      staged,
      includeCaptions,
      dark: isDarkTheme(),
      title: shown.entry?.title ?? 'Origami Model',
      subtitle: shown.entry?.subtitle ?? '',
      slug: shown.entry?.slug ?? shown.entry?.title ?? 'origami',
      schedule: shown.schedule,
      initialFold,
      signal: abort.signal,
      onProgress: (frac, foldVal, state) => {
        setRecordingLabels(Math.round(100 * frac));
        shown.fold = foldVal;
        byId('fold').value = 100 * foldVal;
        byId('fold-value').textContent = `${Math.round(100 * foldVal)}%`;
        shown.lastState = state;
        syncActiveStageHighlight();
        updateStrainMetric();
        maybeDrawPatternDiagram();
      },
    });
    if (result) {
      const videoLink = byId('download-video');
      if (videoLink.href) URL.revokeObjectURL(videoLink.href);
      videoLink.href = URL.createObjectURL(result.blob);
      videoLink.download = result.filename;
      videoLink.textContent = `Save .${result.ext}`;
      videoLink.hidden = false;
      videoLink.click();
      const mb = (result.blob.size / (1024 * 1024)).toFixed(1);
      byId('status').textContent = `Exported ${result.filename} (${mb} MB)`;
    } else {
      byId('status').textContent = 'Video recording cancelled.';
    }
  } catch (error) {
    byId('status').textContent = `Video export failed: ${error.message}`;
  } finally {
    shown.recordingAbort = null;
    recordBtn.textContent = 'Record 1080p';
    quickBtn.textContent = 'Record video';
    setFold(initialFold);
  }
}

/**
 * Loads a showcase model by its catalog entry.
 *
 * @param {!Object} entry Catalog metadata entry.
 * @return {!Promise<void>}
 */
async function showCatalogModel(entry) {
  byId('status').textContent = `Loading ${entry.title}…`;
  requested = entry;
  try {
    const [fold, stickFigure] = await Promise.all([
      fetchJson(`models/${entry.slug}/shaped_3d.fold`),
      fetchJson(`models/${entry.slug}/stick_figure.json`),
    ]);
    if (requested !== entry) return;
    show(parseFold(fold), stickFigure, entry);
  } catch (error) {
    byId('status').textContent =
        `Could not load ${entry.title}: ${error.message}`;
  }
}

/**
 * Shows files chosen or dropped by the user: a `.fold` model, a stick figure
 * `.json` to go with it, or both.
 *
 * @param {!Array<!File>} files
 */
async function showLocalFiles(files) {
  let {model, stickFigure, entry} = shown;
  try {
    for (const file of files) {
      const json = JSON.parse(await file.text());
      if (file.name.endsWith('.fold')) {
        model = parseFold(json);
        stickFigure = null;
        entry = {title: model.title || file.name};
      } else if (Array.isArray(json.children)) {
        stickFigure = json;
      } else {
        throw new Error(`${file.name} is neither a .fold nor a stick figure.`);
      }
    }
    if (!model) throw new Error('Open a .fold file first.');
    requested = null;
    show(model, stickFigure, entry);
  } catch (error) {
    byId('status').textContent = `Could not open the file: ${error.message}`;
  }
}

/**
 * Resolves the requested model slug from `?model=` or `#slug`.
 *
 * @return {?string}
 */
function slugInUrl() {
  const fromQuery = new URLSearchParams(location.search).get('model');
  if (fromQuery) return fromQuery;
  const fromHash = location.hash.replace(/^#/, '');
  return fromHash || null;
}

/**
 * Initializes the 3D stage, loads the catalog, and wires up all controls.
 *
 * @return {!Promise<void>}
 */
async function main() {
  const [loadedStage, catalog] = await Promise.all([
    Stage.create(byId('canvas'), byId('pins')),
    fetchJson('models/catalog.json'),
  ]);
  stage = loadedStage;
  const showSlug = (slug) => showCatalogModel(
      catalog.find((entry) => entry.slug === slug) ?? catalog[0]);

  byId('gallery').replaceChildren(...catalog.map((entry) => {
    const card = html(
        'a', {href: `?model=${entry.slug}`}, html('img', {
          src: `models/${entry.slug}/preview.png`,
          alt: '',
          loading: 'lazy',
        }),
        html('span', {}, entry.title));
    card.dataset.slug = entry.slug;
    card.addEventListener('click', (event) => {
      event.preventDefault();
      history.pushState(null, '', card.href);
      showSlug(entry.slug);
    });
    return card;
  }));
  window.addEventListener('popstate', () => showSlug(slugInUrl()));
  window.addEventListener('hashchange', () => showSlug(slugInUrl()));

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
  byId('stage-caption').addEventListener('click', () => selectTab('schedule'));
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
    if (shown.tab === 'sticks') drawDiagram();
  });

  for (const tab of byId('tabs').children) {
    tab.addEventListener('click', () => selectTab(tab.dataset.tab));
  }
  byId('diagram').addEventListener('click', (event) => {
    const {region} = event.target.dataset;
    if (region !== undefined) toggleHighlight(Number(region));
  });
  byId('theme').addEventListener('click', () => {
    document.documentElement.dataset.theme = isDarkTheme() ? 'light' : 'dark';
  });

  byId('file').addEventListener('change', (event) => {
    showLocalFiles([...event.target.files]);
    event.target.value = '';
  });
  window.addEventListener('dragover', (event) => event.preventDefault());
  window.addEventListener('drop', (event) => {
    event.preventDefault();
    showLocalFiles([...event.dataTransfer.files]);
  });

  await showSlug(slugInUrl());
}

main().catch((error) => {
  byId('status').textContent = `Could not start: ${error.message}`;
});
