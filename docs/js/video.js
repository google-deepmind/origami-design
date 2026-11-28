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
 * @fileoverview Client-side 1080p folding & 360-degree turntable video exporter.
 *
 * Captures real-time 1920x1080 camera and multi-stage folding choreography via
 * `requestAnimationFrame` and `MediaRecorder`, composites the theme's radial
 * studio backdrop and optional stage-caption HUD (since the WebGL canvas uses
 * `alpha: true`), and encodes an MP4 (H.264, when supported) or VP9 WebM video.
 */

import {DEFAULT_BASE_COMPLETE_T, DEFAULT_SHAPING_START_T, DEFAULT_STANDUP_START_T, smootherstep, unwarpFoldingTime, warpFoldingTime, warpScheduleProgress} from './schedule.js';

/** Output video dimensions (16:9 Full HD) and target frame rate. */
const VIDEO_WIDTH = 1920;
const VIDEO_HEIGHT = 1080;
const VIDEO_FPS = 60;
const VIDEO_BITS_PER_SECOND = 16_000_000;

/** Ordered MIME candidates preferring hardware-playable MP4 before VP9 WebM. */
const MIME_CANDIDATES = [
  {mimeType: 'video/mp4;codecs=avc1.42E01E', ext: 'mp4'},
  {mimeType: 'video/mp4;codecs=avc1', ext: 'mp4'},
  {mimeType: 'video/mp4', ext: 'mp4'},
  {mimeType: 'video/webm;codecs=vp9', ext: 'webm'},
  {mimeType: 'video/webm;codecs=vp8', ext: 'webm'},
  {mimeType: 'video/webm', ext: 'webm'},
];

/** Act badge colors matching `style.css`. */
const ACT_COLORS = {
  I: '#2563eb',
  II: '#0d9488',
  III: '#d97706',
  IV: '#9333ea',
  SPIN: '#b83a2d',
};

/** Available camera & folding presets for exported videos. */
export const VIDEO_PRESETS = [
  {
    id: 'guided',
    label: 'Guided fold + 360°',
    description:
        'Tabletop view for flat-base collapse, smooth glide into hero angle for 3D shaping, then one 360° showcase rotation.',
  },
  {
    id: 'hero',
    label: 'Hero fold + 360°',
    description:
        'One full fold in the fixed hero camera angle followed by one 360° showcase rotation.',
  },
  {
    id: 'fold',
    label: 'Fold only',
    description:
        'Multi-stage fold from flat crease pattern to finished 3D pose without turntable.',
  },
  {
    id: 'spin',
    label: '360° spin only',
    description: 'Seamless 360° showcase rotation at the current fold state.',
  },
];

/**
 * Picks the best `MediaRecorder` format supported by the current browser.
 *
 * @return {?{mimeType: string, ext: string}}
 */
export function pickSupportedVideoFormat() {
  if (typeof MediaRecorder === 'undefined') return null;
  for (const candidate of MIME_CANDIDATES) {
    if (MediaRecorder.isTypeSupported(candidate.mimeType)) {
      return candidate;
    }
  }
  return {mimeType: '', ext: 'webm'};
}

/**
 * Evaluates normalized fold progress, camera orbit, and HUD caption at
 * `elapsedSec` for the chosen preset.
 *
 * @param {string} presetId One of `'guided'`, `'hero'`, `'fold'`, `'spin'`.
 * @param {number} elapsedSec Elapsed video time in seconds.
 * @param {{
 *   staged: boolean,
 *   baseCompleteT: number,
 *   shapingStartT: number,
 *   standupStartT: number,
 *   s2TimeMap: (?Float64Array|?Array<number>|undefined),
 *   heroAzimuth: number,
 *   heroElevation: number,
 *   initialFold: number,
 * }} cfg Choreography parameters.
 * @return {{
 *   done: boolean,
 *   totalSec: number,
 *   fold: number,
 *   azimuth: number,
 *   elevation: number,
 *   isSpin: boolean,
 *   spinDeg: number,
 * }}
 */
export function evaluateVideoTimeline(presetId, elapsedSec, cfg) {
  const {
    staged,
    baseCompleteT,
    shapingStartT,
    standupStartT,
    s2TimeMap = null,
    heroAzimuth,
    heroElevation,
    initialFold,
  } = cfg;

  if (presetId === 'spin') {
    const totalSec = 6.0;
    const u = Math.max(0, Math.min(1, elapsedSec / totalSec));
    const eased = smootherstep(u);
    return {
      done: elapsedSec >= totalSec,
      totalSec,
      fold: initialFold,
      azimuth: heroAzimuth + 360 * eased,
      elevation: heroElevation + 4 * Math.sin(Math.PI * eased),
      isSpin: true,
      spinDeg: Math.round(360 * eased),
    };
  }

  const holdFlatSec = 0.6;
  const foldTotalSec = staged ? 12.6 : 6.4;
  const hasBaseDwell = staged && baseCompleteT > 0.05 && baseCompleteT < 0.92;
  const uAtBase = hasBaseDwell ?
      unwarpFoldingTime(baseCompleteT, baseCompleteT, shapingStartT) :
      0;
  const s1Sec = hasBaseDwell ? foldTotalSec * uAtBase : foldTotalSec;
  const holdBaseSec = hasBaseDwell ? 0.65 : 0;
  const s2Sec = hasBaseDwell ? foldTotalSec * (1 - uAtBase) : 0;
  const holdHeroSec = presetId === 'fold' ? 0.9 : 0.55;
  const spinSec = presetId === 'fold' ? 0 : 5.2;
  const holdEndSec = presetId === 'fold' ? 0 : 0.4;

  const t1 = holdFlatSec;
  const t2 = t1 + s1Sec;
  const t3 = t2 + holdBaseSec;
  const t4 = t3 + s2Sec;
  const t5 = t4 + holdHeroSec;
  const t6 = t5 + spinSec;
  const totalSec = t6 + holdEndSec;

  const t = Math.max(0, Math.min(totalSec, elapsedSec));
  let fold = 1;
  let isSpin = false;
  let spinDeg = 0;

  const mapClockToFold = (uClock) => staged ?
      warpFoldingTime(uClock, baseCompleteT, shapingStartT) :
      0.5 - 0.5 * Math.cos(Math.PI * Math.max(0, Math.min(1, uClock)));

  if (t < t1) {
    fold = 0;
  } else if (t < t2) {
    const frac = (t - t1) / Math.max(s1Sec, 1e-6);
    const uClock = hasBaseDwell ? frac * uAtBase : frac;
    fold = mapClockToFold(uClock);
  } else if (t < t3) {
    fold = baseCompleteT;
  } else if (t < t4) {
    const frac = (t - t3) / Math.max(s2Sec, 1e-6);
    const uClock = uAtBase + frac * (1 - uAtBase);
    fold = mapClockToFold(uClock);
  } else {
    fold = 1;
  }

  let azimuth = heroAzimuth;
  let elevation = heroElevation;

  if (presetId !== 'hero') {
    // Tabletop overview during flat-sheet & Act-I/II base collapse so creases
    // are clearly visible rather than viewed edge-on, gliding smoothly into the
    // exact hero camera as the model stands upright in 3D.
    const tableElev = Math.max(46, heroElevation + 24);
    const midBaseElev = Math.max(28, heroElevation + 10);
    const pivotT = Math.max(shapingStartT, standupStartT || shapingStartT);
    const warpedFold = (staged && s2TimeMap) ?
        warpScheduleProgress({baseCompleteT, shapingStartT, s2TimeMap}, fold) :
        fold;
    if (warpedFold <= pivotT) {
      const e1 = smootherstep(warpedFold / Math.max(pivotT, 1e-4));
      azimuth = (heroAzimuth - 18) * (1 - e1) + (heroAzimuth - 6) * e1;
      elevation = tableElev * (1 - e1) + midBaseElev * e1;
    } else if (t < t4) {
      const settleEnd = Math.min(0.96, pivotT + 0.65 * (1 - pivotT));
      const e2 = smootherstep(
          (warpedFold - pivotT) / Math.max(settleEnd - pivotT, 1e-4));
      azimuth = (heroAzimuth - 6) * (1 - e2) + heroAzimuth * e2;
      elevation = midBaseElev * (1 - e2) + heroElevation * e2;
    }
  }

  if (spinSec > 0 && t >= t5 && t < t6) {
    const uSpin = (t - t5) / spinSec;
    const easedSpin = smootherstep(uSpin);
    azimuth = heroAzimuth + 360 * easedSpin;
    elevation = heroElevation + 4 * Math.sin(Math.PI * easedSpin);
    isSpin = true;
    spinDeg = Math.round(360 * easedSpin);
  }

  return {
    done: elapsedSec >= totalSec,
    totalSec,
    fold,
    azimuth,
    elevation,
    isSpin,
    spinDeg,
  };
}

/**
 * Draws a rounded rectangle path on `ctx`.
 *
 * @param {!CanvasRenderingContext2D} ctx
 * @param {number} x
 * @param {number} y
 * @param {number} w
 * @param {number} h
 * @param {number} r
 */
function roundRectPath(ctx, x, y, w, h, r) {
  const radius = Math.min(r, w / 2, h / 2);
  ctx.beginPath();
  ctx.moveTo(x + radius, y);
  ctx.arcTo(x + w, y, x + w, y + h, radius);
  ctx.arcTo(x + w, y + h, x, y + h, radius);
  ctx.arcTo(x, y + h, x, y, radius);
  ctx.arcTo(x, y, x + w, y, radius);
  ctx.closePath();
}

/**
 * Composites the rendered WebGL frame onto a 1920x1080 studio canvas with the
 * radial parchment/dark backdrop and optional stage-caption HUD.
 *
 * @param {!CanvasRenderingContext2D} ctx
 * @param {!HTMLCanvasElement} webglCanvas
 * @param {{
 *   dark: boolean,
 *   includeCaptions: boolean,
 *   title: string,
 *   subtitle: string,
 *   schedule: ?Object,
 *   lastState: ?Object,
 *   step: !Object,
 * }} hud
 */
function compositeVideoFrame(ctx, webglCanvas, hud) {
  const w = VIDEO_WIDTH;
  const h = VIDEO_HEIGHT;
  const {dark, includeCaptions, title, subtitle, schedule, lastState, step} =
      hud;

  const grad =
      ctx.createRadialGradient(w * 0.5, h * 0.35, 40, w * 0.5, h * 0.5, w * 0.7);
  if (dark) {
    grad.addColorStop(0, '#262a32');
    grad.addColorStop(1, '#14161a');
  } else {
    grad.addColorStop(0, '#ffffff');
    grad.addColorStop(1, '#ebe6dd');
  }
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, w, h);
  ctx.drawImage(webglCanvas, 0, 0, w, h);

  if (!includeCaptions) return;

  const textMain = dark ? '#f2f4f7' : '#1f2328';
  const textMuted = dark ? '#98a2b3' : '#667085';
  const cardBg = dark ? 'rgba(20, 22, 26, 0.82)' : 'rgba(255, 255, 255, 0.86)';
  const cardLine = dark ? 'rgba(255, 255, 255, 0.12)' : 'rgba(31, 35, 40, 0.12)';

  ctx.save();
  ctx.font = '700 32px system-ui, -apple-system, sans-serif';
  ctx.fillStyle = textMain;
  ctx.fillText(title, 44, 58);
  if (subtitle) {
    ctx.font = '400 19px system-ui, -apple-system, sans-serif';
    ctx.fillStyle = textMuted;
    ctx.fillText(subtitle, 44, 86);
  }

  let badgeText = 'FOLD';
  let badgeKey = 'I';
  let stageLabel = lastState?.label ?? 'Sculpted 3D Origami';
  if (step.isSpin) {
    badgeText = '360°';
    badgeKey = 'SPIN';
    stageLabel = `Showcase Turntable · ${step.spinDeg}°`;
  } else if (schedule && lastState) {
    const idx = lastState.activePhaseIndex ?? -1;
    const ph = idx >= 0 ? schedule.phases[idx] : null;
    const warpedFold = lastState.warpedT ?? step.fold;
    if (ph?.act) {
      badgeKey = ph.act;
      badgeText = `Act ${ph.act}`;
    } else if (warpedFold <= 0.002) {
      badgeKey = 'I';
      badgeText = 'Flat';
    } else if (
        schedule.baseCompleteT < 0.95 &&
        Math.abs(warpedFold - schedule.baseCompleteT) <= 0.015) {
      badgeKey = 'I';
      badgeText = 'Base';
    } else if (warpedFold >= 0.998) {
      badgeKey = 'IV';
      badgeText = '3D';
    }
    stageLabel = stageLabel.replace(/^Act\s+[IVX]+(?:\.\d+)?\s*·\s*/i, '');
  }

  const pctText = step.isSpin ? '100%' : `${Math.round(100 * step.fold)}%`;
  const pillW = Math.min(860, w - 96);
  const pillH = 58;
  const pillX = (w - pillW) / 2;
  const pillY = h - pillH - 32;

  roundRectPath(ctx, pillX, pillY, pillW, pillH, 14);
  ctx.fillStyle = cardBg;
  ctx.fill();
  ctx.lineWidth = 1.5;
  ctx.strokeStyle = cardLine;
  ctx.stroke();

  // Act badge.
  ctx.font = '700 16px system-ui, -apple-system, sans-serif';
  const badgePadX = 12;
  const badgeW = Math.max(58, ctx.measureText(badgeText).width + 2 * badgePadX);
  const badgeH = 28;
  const badgeX = pillX + 16;
  const badgeY = pillY + 11;
  roundRectPath(ctx, badgeX, badgeY, badgeW, badgeH, 7);
  ctx.fillStyle = ACT_COLORS[badgeKey] ?? ACT_COLORS.I;
  ctx.fill();
  ctx.fillStyle = '#ffffff';
  ctx.textBaseline = 'middle';
  ctx.fillText(badgeText, badgeX + badgePadX, badgeY + badgeH / 2 + 0.5);

  // Stage label & right-aligned percentage.
  ctx.font = '600 19px system-ui, -apple-system, sans-serif';
  ctx.fillStyle = textMain;
  const labelX = badgeX + badgeW + 14;
  ctx.fillText(stageLabel, labelX, badgeY + badgeH / 2 + 0.5);

  ctx.font = '600 18px system-ui, -apple-system, sans-serif';
  ctx.fillStyle = textMuted;
  const pctW = ctx.measureText(pctText).width;
  ctx.fillText(pctText, pillX + pillW - 18 - pctW, badgeY + badgeH / 2 + 0.5);

  // Thin segmented progress bar along bottom inside pill.
  const barX = pillX + 16;
  const barY = pillY + pillH - 9;
  const barW = pillW - 32;
  const barH = 4;
  roundRectPath(ctx, barX, barY, barW, barH, 2);
  ctx.fillStyle = dark ? 'rgba(255, 255, 255, 0.12)' : 'rgba(0, 0, 0, 0.09)';
  ctx.fill();

  const fillFrac = step.isSpin ? 1 : Math.max(0, Math.min(1, step.fold));
  if (fillFrac > 0.002) {
    roundRectPath(ctx, barX, barY, Math.max(6, barW * fillFrac), barH, 2);
    ctx.fillStyle = ACT_COLORS[badgeKey] ?? '#b83a2d';
    ctx.fill();
  }
  ctx.restore();
}

/**
 * Records a 1080p video of the active model on `stage` and resolves to a
 * downloadable `{blob, ext, filename}` object (or `null` if cancelled).
 *
 * @param {{
 *   stage: !Object,
 *   presetId: string,
 *   staged: boolean,
 *   includeCaptions: boolean,
 *   dark: boolean,
 *   title: string,
 *   subtitle: string,
 *   slug: string,
 *   schedule: ?Object,
 *   initialFold: number,
 *   signal: !AbortSignal,
 *   onProgress: function(number, number, ?Object): void,
 * }} options
 * @return {!Promise<?{blob: !Blob, ext: string, filename: string}>}
 */
export async function recordFoldingVideo(options) {
  const format = pickSupportedVideoFormat();
  if (!format) {
    throw new Error('MediaRecorder video capture is not supported here.');
  }

  const {
    stage,
    presetId,
    staged,
    includeCaptions,
    dark,
    title,
    subtitle,
    slug,
    schedule,
    initialFold,
    signal,
    onProgress,
  } = options;

  const savedState = stage.captureState();
  const envelope = stage.computeVideoEnvelope(staged);
  const hero = stage.hero;

  const compCanvas = document.createElement('canvas');
  compCanvas.width = VIDEO_WIDTH;
  compCanvas.height = VIDEO_HEIGHT;
  const ctx = compCanvas.getContext('2d', {alpha: false});

  const manualFrames = typeof CanvasCaptureMediaStreamTrack !== 'undefined' &&
      typeof CanvasCaptureMediaStreamTrack.prototype?.requestFrame ===
          'function';
  const stream = compCanvas.captureStream(manualFrames ? 0 : VIDEO_FPS);
  const videoTrack = stream.getVideoTracks()[0];
  const recorderOptions = {videoBitsPerSecond: VIDEO_BITS_PER_SECOND};
  if (format.mimeType) recorderOptions.mimeType = format.mimeType;
  const recorder = new MediaRecorder(stream, recorderOptions);

  const chunks = [];
  recorder.addEventListener('dataavailable', (event) => {
    if (event.data && event.data.size > 0) chunks.push(event.data);
  });

  const stopped = new Promise((resolve, reject) => {
    recorder.addEventListener('stop', resolve, {once: true});
    recorder.addEventListener(
        'error', (e) => reject(e.error ?? new Error('MediaRecorder failed.')),
        {once: true});
  });

  const cfg = {
    staged,
    baseCompleteT: schedule?.baseCompleteT ?? DEFAULT_BASE_COMPLETE_T,
    shapingStartT: schedule?.shapingStartT ?? DEFAULT_SHAPING_START_T,
    standupStartT: schedule?.standupStartT ?? DEFAULT_STANDUP_START_T,
    s2TimeMap: schedule?.s2TimeMap ?? null,
    heroAzimuth: hero.azimuth,
    heroElevation: hero.elevation,
    initialFold,
  };

  let hiddenError = null;
  let finishLoop = null;
  let rafId = 0;
  const onVisibilityChange = () => {
    if (document.visibilityState === 'hidden' && !hiddenError) {
      hiddenError = new Error(
          'Video recording aborted because the tab became hidden.');
      cancelAnimationFrame(rafId);
      finishLoop?.();
    }
  };
  const onAbort = () => {
    cancelAnimationFrame(rafId);
    finishLoop?.();
  };
  document.addEventListener('visibilitychange', onVisibilityChange);
  signal.addEventListener('abort', onAbort, {once: true});
  stage.beginVideoSession(VIDEO_WIDTH, VIDEO_HEIGHT);

  try {
    // Render initial t=0 frame onto composite canvas before starting recorder.
    const firstStep = evaluateVideoTimeline(presetId, 0, cfg);
    const firstRadius = envelope.radiusAt(firstStep.fold);
    const firstRender = stage.renderVideoFrame({
      progress: firstStep.fold,
      staged,
      azimuthDeg: firstStep.azimuth,
      elevationDeg: firstStep.elevation,
      radius: firstRadius,
    });
    compositeVideoFrame(ctx, firstRender.canvas, {
      dark,
      includeCaptions,
      title,
      subtitle,
      schedule,
      lastState: firstRender.state,
      step: firstStep,
    });

    recorder.start(250);
    if (manualFrames && videoTrack) videoTrack.requestFrame();
    const startMs = performance.now();

    await new Promise((resolve) => {
      finishLoop = resolve;
      if (signal.aborted || hiddenError) {
        resolve();
        return;
      }
      const tick = (nowMs) => {
        if (signal.aborted || hiddenError) {
          resolve();
          return;
        }
        const elapsedSec = (nowMs - startMs) / 1000;
        const step = evaluateVideoTimeline(presetId, elapsedSec, cfg);
        const radius = envelope.radiusAt(step.fold);
        const rendered = stage.renderVideoFrame({
          progress: step.fold,
          staged,
          azimuthDeg: step.azimuth,
          elevationDeg: step.elevation,
          radius,
        });
        compositeVideoFrame(ctx, rendered.canvas, {
          dark,
          includeCaptions,
          title,
          subtitle,
          schedule,
          lastState: rendered.state,
          step,
        });
        if (manualFrames && videoTrack) {
          videoTrack.requestFrame();
        }
        onProgress(
            Math.min(1, elapsedSec / step.totalSec), step.fold, rendered.state);
        if (step.done) {
          resolve();
          return;
        }
        rafId = requestAnimationFrame(tick);
      };
      rafId = requestAnimationFrame(tick);
    });

    if (recorder.state !== 'inactive') recorder.stop();
    await stopped;
  } finally {
    cancelAnimationFrame(rafId);
    signal.removeEventListener('abort', onAbort);
    document.removeEventListener('visibilitychange', onVisibilityChange);
    for (const track of stream.getTracks()) track.stop();
    stage.endVideoSession(savedState);
  }

  if (hiddenError) throw hiddenError;
  if (signal.aborted) return null;
  const mimeType = recorder.mimeType || format.mimeType || 'video/webm';
  const ext = mimeType.includes('mp4') ? 'mp4' : format.ext;
  const safeSlug =
      (slug || title || 'origami').toLowerCase().replace(/[^a-z0-9]+/g, '-');
  return {
    blob: new Blob(chunks, {type: mimeType}),
    ext,
    filename: `${safeSlug}-${presetId}.${ext}`,
  };
}
