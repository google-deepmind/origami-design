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
 * A showcase model is a directory `models/<slug>/` holding `shaped_3d.fold`
 * (matching the FOLD layout written by `bp-pipeline -o`; note the ten bundled
 * models are precomputed snapshots without recorded sculpting tool histories
 * and cannot be regenerated from `bp-pipeline` alone), the `stick_figure.json`
 * it was designed from and a `preview.png`, plus an entry in
 * `models/catalog.json` with the descriptive text, colors and camera angle.
 * Every numeric metric on the page is computed from those files.
 */

import {creasePatternSvg, drawCreasePattern, drawPacking, drawStickFigure} from './diagram.js';
import {parseFold, regionTips, strainPercent} from './fold.js';
import {Stage} from './stage.js';

/** Seconds the folding animation takes between flat and folded. */
const FOLD_SECONDS = 6;

/** Pause at each end of a continuous back-and-forth fold loop, in seconds. */
const LOOP_DWELL_SECONDS = 0.45;

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

/** The model on screen and how it is shown. */
const shown = {
  /** @type {?Object} Parsed model from `fold.js`. */
  model: null,
  /** @type {?Object} Parsed stick figure JSON. */
  stickFigure: null,
  /** @type {?Object} Catalog entry; a local file has one without a `slug`. */
  entry: null,
  /** Folding progress from 0 (flat) to 1 (folded). */
  fold: 1,
  /** Highlighted region of the model, or null. */
  region: null,
  /** Selected diagram: 'pattern', 'base', 'packing' or 'sticks'. */
  tab: 'pattern',
  /** Camera rotation of the stage, shared with the stick figure drawing. */
  view: [1, 0, 0, 0, 1, 0, 0, 0, 1],
  /** `requestAnimationFrame` handle while the folding animation plays. */
  animation: 0,
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
 * Shows a model and everything derived from it.
 *
 * @param {!Object} model Parsed model from `fold.js`.
 * @param {?Object} stickFigure Parsed stick figure JSON, if known.
 * @param {!Object} entry Catalog entry, or its stand-in for a local file.
 */
function show(model, stickFigure, entry) {
  Object.assign(shown, {model, stickFigure, entry, region: null});
  stopAnimation();

  stage.setModel(model, entry.camera);
  // A region is named after the last stick of its chain, the one at the tip.
  const tips = regionTips(model);
  stage.setPins(model.regions.flatMap(
      (region, r) => region.kind === 'flap' && tips[r] >= 0 ?
          [{vertex: tips[r], label: words(region.sticks.at(-1))}] :
          []));
  byId('color-top').value = entry.colorTop ?? DEFAULT_COLORS.top;
  byId('color-bottom').value = entry.colorBottom ?? DEFAULT_COLORS.bottom;
  applyOptions();
  byId('staged').disabled = model.shaping === null;
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

  const strain = strainPercent(model);
  const creases = model.assignments.filter((a) => a === 'M' || a === 'V');
  byId('grid').textContent =
      model.gridSize ? `${model.gridSize}×${model.gridSize}` : '–';
  byId('stick-count').textContent = stickFigure?.children.length ?? '–';
  byId('crease-count').textContent = creases.length.toLocaleString('en');
  byId('strain').textContent = strain === null ? '–' : `${strain.toFixed(2)}%`;

  byId('regions').replaceChildren(...model.regions.map((region, r) => {
    const chip = html(
        'button', {
          type: 'button',
          className: region.kind,
          title: `${region.kind}: ${region.sticks.map(words).join(' + ')}`,
          ariaPressed: 'false',
        },
        words(region.sticks.at(-1)));
    chip.addEventListener('click', () => toggleHighlight(r));
    return chip;
  }));

  // A local file is already on the user's disk; only catalog models link to
  // their files.
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

  // Not every file can feed every diagram.
  const available = {
    pattern: true,
    base: model.shaping !== null,
    packing: model.regions.length > 0,
    sticks: stickFigure !== null,
  };
  for (const tab of byId('tabs').children) {
    tab.disabled = !available[tab.dataset.tab];
  }
  selectTab(available[shown.tab] ? shown.tab : 'pattern');
}

/** Switches the diagram panel to the tab called `name`. */
function selectTab(name) {
  shown.tab = name;
  for (const tab of byId('tabs').children) {
    const selected = tab.dataset.tab === name;
    tab.ariaSelected = String(selected);
    if (selected) byId('diagram').ariaLabel = tab.textContent;
  }
  for (const legend of document.querySelectorAll('[data-tabs]')) {
    legend.hidden = !legend.dataset.tabs.split(' ').includes(name);
  }
  drawDiagram();
}

/** Draws the diagram of the selected tab. */
function drawDiagram() {
  const svg = byId('diagram');
  const {model, stickFigure, region, view} = shown;
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
      drawCreasePattern(svg, model, {highlight: region});
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
  stage.setHighlight(shown.region);
  drawDiagram();
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
  shown.fold = progress;
  byId('fold').value = 100 * progress;
  byId('fold-value').textContent = `${Math.round(100 * progress)}%`;
  const direction = progress > 0.5 ? 'Unfold' : 'Fold';
  byId('play').textContent = shown.animation ? 'Pause' : direction;
  stage.setFold(progress, byId('staged').checked);
}

/**
 * Smoothly animates folding towards whichever end is farther away, reversing
 * continuously when `#loop` is checked.
 */
function playAnimation() {
  let target = shown.fold > 0.5 ? 0 : 1;
  const clamped = Math.min(1, Math.max(0, shown.fold));
  let linear = Math.acos(1 - 2 * clamped) / Math.PI;
  let dwellUntil = 0;
  let last = performance.now();
  const step = (now) => {
    const dt = (now - last) / 1000;
    last = now;
    if (now < dwellUntil) {
      shown.animation = requestAnimationFrame(step);
      return;
    }
    const delta = dt / FOLD_SECONDS;
    linear = target === 1 ? Math.min(1, linear + delta) :
                            Math.max(0, linear - delta);
    const eased = 0.5 - 0.5 * Math.cos(Math.PI * linear);
    if (linear === target) {
      if (byId('loop').checked) {
        target = 1 - target;
        dwellUntil = now + 1000 * LOOP_DWELL_SECONDS;
        shown.animation = requestAnimationFrame(step);
      } else {
        shown.animation = 0;
      }
    } else {
      shown.animation = requestAnimationFrame(step);
    }
    setFold(eased);
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
  // A stick figure on its own joins the model that is already shown.
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
  const slugInUrl = () => new URLSearchParams(location.search).get('model');

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
  const displayOptions =
      ['color-top', 'color-bottom', 'creases', 'xray', 'rotate', 'show-pins'];
  for (const id of displayOptions) {
    byId(id).addEventListener('input', applyOptions);
  }
  for (const button of document.querySelectorAll('[data-view]')) {
    button.addEventListener('click', () => stage.setView(button.dataset.view));
  }
  // Grabbing the camera takes over from the turntable.
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
  // The packing diagram marks the path of each region with its index.
  byId('diagram').addEventListener('click', (event) => {
    const {region} = event.target.dataset;
    if (region !== undefined) toggleHighlight(Number(region));
  });
  byId('theme').addEventListener('click', () => {
    const root = document.documentElement;
    const dark = root.dataset.theme ?
        root.dataset.theme === 'dark' :
        matchMedia('(prefers-color-scheme: dark)').matches;
    root.dataset.theme = dark ? 'light' : 'dark';
  });

  byId('file').addEventListener('change', (event) => {
    showLocalFiles([...event.target.files]);
    event.target.value = '';  // Allow choosing the same file again.
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
