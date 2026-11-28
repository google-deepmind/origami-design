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
 * @fileoverview Background Web Worker running `solveLiveSignedXPBD` off the
 * main thread when Studio opens a `.fold` model without a precomputed
 * `schedule.json`.
 */

import {Folder, buildShortHopDijkstra} from './fold.js';
import {buildSignedXPBDSystem, solveLiveSignedXPBD} from './xpbd.js';

self.addEventListener('message', (event) => {
  const {id, model, rootFace, shapingHopCosts} = event.data;
  const guideFolder = new Folder(model);
  const s2Tree = buildShortHopDijkstra(model, rootFace, shapingHopCosts);
  guideFolder.adoptTree(s2Tree);
  const posBase =
      Float32Array.from(guideFolder.foldByHingesRaw(model.baseFoldAngles));
  const xpbdSys = buildSignedXPBDSystem(model);
  const act1Keyframes = solveLiveSignedXPBD(
      model, posBase,
      (curAngles) => {
        guideFolder.adoptTree(s2Tree);
        return guideFolder.foldByHingesRaw(curAngles);
      },
      xpbdSys);
  const transfer = act1Keyframes.map((kf) => kf.buffer);
  self.postMessage({id, act1Keyframes}, transfer);
});
