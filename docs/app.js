/**
 * Copyright 2026 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import earcut from 'earcut';

// ─── Curated Japanese Washi Duo-Tone Paper Swatches ──────────────────────────

const PAPER_SWATCHES = [
  { id: 'model', name: 'Model Signature Duo-Tone', top: null, bottom: null },
  { id: 'karakurenai', name: 'Karakurenai Crimson & Washi Cream', top: '#A62B2B', bottom: '#F5ECDA' },
  { id: 'ai', name: 'Ai Indigo & Ivory', top: '#23476E', bottom: '#F4EFE4' },
  { id: 'matcha', name: 'Matcha Moss & Warm Parchment', top: '#3B6B52', bottom: '#F2E9D5' },
  { id: 'kohaku', name: 'Kohaku Terracotta & Sand', top: '#B85A28', bottom: '#F6EEDB' },
  { id: 'sumi', name: 'Kurosumi Obsidian & Gold Leaf', top: '#252830', bottom: '#E5C687' },
  { id: 'shiro', name: 'Pure Museum Porcelain', top: '#EAE5DC', bottom: '#FFFFFF' },
];

// ─── Studio State ────────────────────────────────────────────────────────────

const state = {
  catalog: [],
  activeSlug: null,
  activeMeta: null,
  bundle: null,           // Active pre-indexed or locally parsed bundle
  svgCache: {},           // key: `${slug}:${tab}` -> svg text
  foldPct: 100.0,         // 0..100
  twoStage: true,
  showCreases: true,
  showPins: false,
  xray: false,
  turntable: true,
  isAnimatingFold: false,
  animDir: -1,
  animPauseUntil: 0,
  activeTab: 'shaped',
  activeCategory: 'All',
  highlightedRegion: -1,
  activeSwatch: 'model',
  colorTop: '#9E3329',
  colorBottom: '#F3E7CE',
  paperThickness: 0.0032,
};

// Three.js Main Origami Stage Objects
let renderer, scene, camera, controls;
let paperMesh, creaseLines, shadowPlane, keyLight;
let currentModelRadius = 0.5;
let currentModelCenterY = 0.2;
const paperUniforms = {
  uFrontColor: { value: new THREE.Color('#9E3329') },
  uBackColor: { value: new THREE.Color('#F3E7CE') },
};

// Three.js Secondary 3D Stick-Figure Skeleton Objects
let skelRenderer, skelScene, skelCamera, skelControls;

// Flat-on-table quaternion: maps XY paper (normal +Z) to XZ floor (normal +Y)
const Q_FLAT_SHEET = new THREE.Quaternion().setFromEuler(
  new THREE.Euler(-Math.PI / 2, 0, 0, 'XYZ')
);

// ─── Procedural Multi-Scale Washi Paper Fiber Texture ────────────────────────

function createWashiFiberTexture() {
  const size = 256;
  const canvas = document.createElement('canvas');
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext('2d');
  const img = ctx.createImageData(size, size);
  const data = img.data;

  const hash = (x, y) => {
    let h = (Math.imul(x, 374761393) + Math.imul(y, 668265263) + 1274126177) | 0;
    h = Math.imul((h ^ (h >>> 13)), 1274126177);
    return ((h ^ (h >>> 16)) & 0xff) / 255.0;
  };

  const smoothNoise = (x, y, scaleX, scaleY) => {
    const sx = x / scaleX;
    const sy = y / scaleY;
    const ix = Math.floor(sx);
    const iy = Math.floor(sy);
    const fx = sx - ix;
    const fy = sy - iy;
    const w = size / scaleX;
    const h = size / scaleY;
    const x0 = ((ix % w) + w) % w;
    const x1 = (x0 + 1) % w;
    const y0 = ((iy % h) + h) % h;
    const y1 = (y0 + 1) % h;
    const v00 = hash(x0, y0);
    const v10 = hash(x1, y0);
    const v01 = hash(x0, y1);
    const v11 = hash(x1, y1);
    const top = v00 + (v10 - v00) * fx;
    const bot = v01 + (v11 - v01) * fx;
    return top + (bot - top) * fy;
  };

  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const fine = smoothNoise(x, y, 2, 8) * 0.42;
      const cross = smoothNoise(x, y, 8, 2) * 0.25;
      const med = smoothNoise(x, y, 8, 8) * 0.22;
      const broad = smoothNoise(x, y, 32, 32) * 0.11;
      const v = Math.floor(110 + (fine + cross + med + broad) * 115);
      const i = (y * size + x) * 4;
      data[i] = v;
      data[i + 1] = v;
      data[i + 2] = v;
      data[i + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
  const tex = new THREE.CanvasTexture(canvas);
  tex.wrapS = THREE.RepeatWrapping;
  tex.wrapT = THREE.RepeatWrapping;
  tex.repeat.set(5, 5);
  return tex;
}

// ─── Initialize Main 3D Studio Stage ─────────────────────────────────────────

function initMainStage() {
  const wrap = document.getElementById('stage-canvas-wrap');
  const canvas = document.getElementById('origami-canvas');
  const w = wrap.clientWidth || 860;
  const h = wrap.clientHeight || 560;

  renderer = new THREE.WebGLRenderer({
    canvas,
    antialias: true,
    alpha: true,
    logarithmicDepthBuffer: true,
    powerPreference: 'high-performance',
    preserveDrawingBuffer: true,
  });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.setSize(w, h, false);
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 0.96;
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;

  scene = new THREE.Scene();

  // Studio softbox environment reflections via PMREM RoomEnvironment
  const pmrem = new THREE.PMREMGenerator(renderer);
  const envTex = pmrem.fromScene(new RoomEnvironment(renderer), 0.05).texture;
  scene.environment = envTex;
  scene.environmentIntensity = 0.18;
  pmrem.dispose();

  camera = new THREE.PerspectiveCamera(36, w / h, 0.02, 30);
  camera.position.set(0.85, 0.62, 1.15);

  controls = new OrbitControls(camera, canvas);
  controls.enableDamping = true;
  controls.dampingFactor = 0.07;
  controls.rotateSpeed = 0.85;
  controls.zoomSpeed = 0.95;
  controls.minDistance = 0.18;
  controls.maxDistance = 6.0;
  controls.target.set(0, 0.18, 0);

  // Pause gentle turntable immediately on user interaction
  controls.addEventListener('start', () => {
    if (state.turntable) {
      state.turntable = false;
      document.getElementById('turntable-btn')?.classList.remove('active');
    }
  });

  // Three-point museum lighting rig
  const hemi = new THREE.HemisphereLight(0xfffaf0, 0xc9bfb0, 0.36);
  scene.add(hemi);

  keyLight = new THREE.DirectionalLight(0xfffaf2, 1.28);
  keyLight.position.set(1.6, 2.7, 1.85);
  keyLight.castShadow = true;
  keyLight.shadow.mapSize.set(2048, 2048);
  keyLight.shadow.camera.near = 0.2;
  keyLight.shadow.camera.far = 8.0;
  keyLight.shadow.camera.left = -1.1;
  keyLight.shadow.camera.right = 1.1;
  keyLight.shadow.camera.top = 1.1;
  keyLight.shadow.camera.bottom = -1.1;
  keyLight.shadow.bias = -0.0004;
  scene.add(keyLight);

  const fillLight = new THREE.DirectionalLight(0xdce8f8, 0.38);
  fillLight.position.set(-1.9, 1.15, -1.2);
  scene.add(fillLight);

  const rimLight = new THREE.DirectionalLight(0xffebd2, 0.34);
  rimLight.position.set(-0.4, 1.6, -2.2);
  scene.add(rimLight);

  // Soft gallery pedestal shadow catcher at y = 0
  const floorGeo = new THREE.PlaneGeometry(8, 8);
  floorGeo.rotateX(-Math.PI / 2);
  const floorMat = new THREE.ShadowMaterial({ opacity: 0.16 });
  shadowPlane = new THREE.Mesh(floorGeo, floorMat);
  shadowPlane.position.y = -0.001;
  shadowPlane.receiveShadow = true;
  scene.add(shadowPlane);

  // Tactile Duo-Tone Washi Physical Material (gl_FrontFacing shader avoids
  // opposing slab collisions and shadow self-acne on thin 180-deg pleats)
  const fiberTex = createWashiFiberTexture();
  const paperMat = new THREE.MeshPhysicalMaterial({
    vertexColors: true,
    side: THREE.DoubleSide,
    roughness: 0.85,
    metalness: 0.0,
    sheen: 0.12,
    sheenRoughness: 0.78,
    sheenColor: new THREE.Color(0xfff8eb),
    bumpMap: fiberTex,
    bumpScale: 0.0028,
    transparent: false,
    opacity: 1.0,
    shadowSide: THREE.DoubleSide,
    polygonOffset: true,
    polygonOffsetFactor: 1.0,
    polygonOffsetUnits: 1.0,
  });
  paperMat.onBeforeCompile = (shader) => {
    shader.uniforms.uFrontColor = paperUniforms.uFrontColor;
    shader.uniforms.uBackColor = paperUniforms.uBackColor;
    shader.fragmentShader =
      'uniform vec3 uFrontColor;\nuniform vec3 uBackColor;\n' +
      shader.fragmentShader.replace(
        '#include <color_fragment>',
        `#include <color_fragment>
         diffuseColor.rgb *= gl_FrontFacing ? uFrontColor : uBackColor;`
      );
  };

  paperMesh = new THREE.Mesh(new THREE.BufferGeometry(), paperMat);
  paperMesh.castShadow = true;
  paperMesh.receiveShadow = false;
  scene.add(paperMesh);

  // Crisp 3D Crease & Cut-Border Line Overlay
  const lineMat = new THREE.LineBasicMaterial({
    vertexColors: true,
    transparent: true,
    opacity: 0.36,
    depthWrite: false,
  });
  creaseLines = new THREE.LineSegments(new THREE.BufferGeometry(), lineMat);
  scene.add(creaseLines);

  const resizeObserver = new ResizeObserver(() => {
    const nw = wrap.clientWidth;
    const nh = wrap.clientHeight;
    if (nw > 0 && nh > 0) {
      camera.aspect = nw / nh;
      camera.updateProjectionMatrix();
      renderer.setSize(nw, nh, false);
      if (scene && camera) {
        renderer.render(scene, camera);
      }
    }
  });
  resizeObserver.observe(wrap);
}

// ─── Deterministic Real-Time BFS Hinge Kinematics & Mesh Builder ─────────────

// Reusable scratch objects for zero-GC 60fps folding
const _tmpV1 = new THREE.Vector3();
const _tmpV2 = new THREE.Vector3();
const _tmpAxis = new THREE.Vector3();
const _tmpPivot = new THREE.Matrix4();
const _tmpPivotInv = new THREE.Matrix4();
const _tmpRot = new THREE.Matrix4();
const _tmpQuat = new THREE.Quaternion();
const _uQuat = new THREE.Quaternion();

function computeEffectiveAngles(bundle, pct, twoStage) {
  const n = bundle.edges.length;
  const out = new Float32Array(n);
  const shaping = bundle.shapingAngles;

  if (!twoStage || !bundle.edgeStage || !bundle.solvingAngles) {
    const f = pct / 100.0;
    for (let i = 0; i < n; i++) {
      out[i] = shaping[i] * f;
    }
    return out;
  }

  const solving = bundle.solvingAngles;
  const stage = bundle.edgeStage;
  if (pct <= 50.0) {
    // Stage 1 (0 -> 50%): uniform angular fold to uniaxial box-pleated base
    const t = Math.max(0.0, Math.min(1.0, pct / 50.0));
    for (let i = 0; i < n; i++) {
      const target = solving[i];
      if (t >= 1.0) {
        out[i] = target;
      } else {
        const s = Math.sign(target);
        const cur = 180.0 * s * t;
        out[i] = s > 0 ? Math.min(cur, target) : (s < 0 ? Math.max(cur, target) : 0.0);
      }
    }
  } else {
    // Stage 2 (50 -> 100%): sculpt base into 3D shaped model
    const t = Math.max(0.0, Math.min(1.0, (pct - 50.0) / 50.0));
    for (let i = 0; i < n; i++) {
      if (stage[i] === 1) {
        out[i] = solving[i];
      } else if (t >= 1.0) {
        out[i] = shaping[i];
      } else {
        const target = shaping[i];
        const s = Math.sign(target);
        const cur = 180.0 * s * t;
        out[i] = s > 0 ? Math.min(cur, target) : (s < 0 ? Math.max(cur, target) : 0.0);
      }
    }
  }
  return out;
}

function ensureTopologyAdjacency(bundle) {
  if (bundle._adjReady) return;
  const { faces, edges } = bundle;
  const edgeKeyToIdx = new Map();
  for (let i = 0; i < edges.length; i++) {
    const u = edges[i][0];
    const v = edges[i][1];
    edgeKeyToIdx.set(u < v ? `${u},${v}` : `${v},${u}`, i);
  }
  const edgeToFaces = new Map();
  for (let fIdx = 0; fIdx < faces.length; fIdx++) {
    const f = faces[fIdx];
    const len = f.length;
    for (let i = 0; i < len; i++) {
      const u = f[i];
      const v = f[(i + 1) % len];
      const key = u < v ? `${u},${v}` : `${v},${u}`;
      let arr = edgeToFaces.get(key);
      if (!arr) {
        arr = [];
        edgeToFaces.set(key, arr);
      }
      arr.push(fIdx);
    }
  }

  // Precompute BFS spanning tree transitions once per model -> O(F) 60fps solve!
  const startFace = bundle.centralFace || 0;
  const visited = new Uint8Array(faces.length);
  visited[startFace] = 1;
  const queue = [startFace];
  const bfsSteps = []; // [parentFace, childFace, v1Idx, v2Idx, edgeIdx]

  while (queue.length > 0) {
    const currF = queue.shift();
    const f = faces[currF];
    const len = f.length;
    for (let i = 0; i < len; i++) {
      const v1 = f[i];
      const v2 = f[(i + 1) % len];
      const key = v1 < v2 ? `${v1},${v2}` : `${v2},${v1}`;
      const nbs = edgeToFaces.get(key);
      if (!nbs) continue;
      for (let k = 0; k < nbs.length; k++) {
        const nextF = nbs[k];
        if (nextF !== currF && !visited[nextF]) {
          visited[nextF] = 1;
          const eIdx = edgeKeyToIdx.get(key) ?? -1;
          bfsSteps.push([currF, nextF, v1, v2, eIdx]);
          queue.push(nextF);
        }
      }
    }
  }

  bundle._bfsSteps = bfsSteps;
  bundle._adjReady = true;
}

function solveFoldFast(bundle, effectiveAngles) {
  ensureTopologyAdjacency(bundle);
  const { vertices2d, faces, _bfsSteps } = bundle;
  const numFaces = faces.length;
  const mats = new Array(numFaces);
  const startFace = bundle.centralFace || 0;
  mats[startFace] = new THREE.Matrix4();

  const deg2rad = Math.PI / 180.0;
  for (let s = 0; s < _bfsSteps.length; s++) {
    const [parentF, childF, v1, v2, eIdx] = _bfsSteps[s];
    const parentMat = mats[parentF];
    const deg = eIdx >= 0 ? effectiveAngles[eIdx] : 0.0;
    if (Math.abs(deg) < 1e-5) {
      mats[childF] = parentMat.clone();
      continue;
    }
    const p1 = vertices2d[v1];
    const p2 = vertices2d[v2];
    _tmpV1.set(p1[0], p1[1], 0).applyMatrix4(parentMat);
    _tmpV2.set(p2[0], p2[1], 0).applyMatrix4(parentMat);
    _tmpAxis.subVectors(_tmpV2, _tmpV1).normalize();

    _tmpPivot.makeTranslation(_tmpV1.x, _tmpV1.y, _tmpV1.z);
    _tmpPivotInv.makeTranslation(-_tmpV1.x, -_tmpV1.y, -_tmpV1.z);
    _tmpRot.makeRotationAxis(_tmpAxis, deg * deg2rad);

    mats[childF] = _tmpPivot.clone().multiply(_tmpRot).multiply(_tmpPivotInv).multiply(parentMat);
  }

  // Average vertex 3D positions across incident faces
  const nVerts = vertices2d.length;
  const vx = new Float64Array(nVerts);
  const vy = new Float64Array(nVerts);
  const vz = new Float64Array(nVerts);
  const vc = new Uint16Array(nVerts);

  for (let fIdx = 0; fIdx < numFaces; fIdx++) {
    const m = mats[fIdx];
    if (!m) continue;
    const e = m.elements;
    const f = faces[fIdx];
    for (let i = 0; i < f.length; i++) {
      const vid = f[i];
      const lx = vertices2d[vid][0];
      const ly = vertices2d[vid][1];
      vx[vid] += e[0] * lx + e[4] * ly + e[12];
      vy[vid] += e[1] * lx + e[5] * ly + e[13];
      vz[vid] += e[2] * lx + e[6] * ly + e[14];
      vc[vid] += 1;
    }
  }

  // Centroid stabilization + smooth SO(3) slerp from flat table -> upright hero pose
  let cx = 0, cy = 0, cz = 0;
  for (let i = 0; i < nVerts; i++) {
    if (vc[i] > 0) {
      vx[i] /= vc[i];
      vy[i] /= vc[i];
      vz[i] /= vc[i];
    } else {
      vx[i] = vertices2d[i][0];
      vy[i] = vertices2d[i][1];
      vz[i] = 0;
    }
    cx += vx[i];
    cy += vy[i];
    cz += vz[i];
  }
  cx /= nVerts || 1;
  cy /= nVerts || 1;
  cz /= nVerts || 1;

  const uq = bundle.uprightQuat || [0, 0, 0, 1];
  _uQuat.set(uq[0], uq[1], uq[2], uq[3]).normalize();
  const tRaw = Math.max(0.0, Math.min(1.0, state.foldPct / 100.0));
  const tSmooth = tRaw * tRaw * (3.0 - 2.0 * tRaw);
  _tmpQuat.slerpQuaternions(Q_FLAT_SHEET, _uQuat, tSmooth);

  let minY = Infinity;
  let maxY = -Infinity;
  let maxR2 = 0.04;
  const pts = new Array(nVerts);
  for (let i = 0; i < nVerts; i++) {
    const p = new THREE.Vector3(vx[i] - cx, vy[i] - cy, vz[i] - cz).applyQuaternion(_tmpQuat);
    if (p.y < minY) minY = p.y;
    pts[i] = p;
  }

  // Ground lowest point onto gallery pedestal y = 0.002
  const liftY = -minY + 0.002;
  for (let i = 0; i < nVerts; i++) {
    pts[i].y += liftY;
    if (pts[i].y > maxY) maxY = pts[i].y;
    const r2 = pts[i].x * pts[i].x + pts[i].z * pts[i].z;
    if (r2 > maxR2) maxR2 = r2;
  }

  currentModelCenterY = (maxY + 0.002) * 0.46;
  currentModelRadius = Math.max(Math.sqrt(maxR2), maxY * 0.62, 0.18);
  return { pts, centroid: [cx, cy, cz], liftY, rotQuat: _tmpQuat.clone() };
}

function updateOrigamiGeometry(refitCamera = false) {
  const bundle = state.bundle;
  if (!bundle || !paperMesh) return;

  const effectiveAngles = computeEffectiveAngles(bundle, state.foldPct, state.twoStage);
  const { pts, centroid, liftY, rotQuat } = solveFoldFast(bundle, effectiveAngles);
  bundle._lastTransform = { centroid, liftY, rotQuat };

  // Compute per-vertex crease cavity AO factor from incident fold angles
  const nVerts = pts.length;
  const vertShade = new Float32Array(nVerts);
  vertShade.fill(1.0);
  const { edges } = bundle;
  for (let e = 0; e < edges.length; e++) {
    const absAng = Math.abs(effectiveAngles[e]);
    if (absAng > 25.0) {
      const cavity = 1.0 - 0.14 * Math.min(1.0, absAng / 180.0);
      const u = edges[e][0];
      const v = edges[e][1];
      if (cavity < vertShade[u]) vertShade[u] = cavity;
      if (cavity < vertShade[v]) vertShade[v] = cavity;
    }
  }

  const frontBase = new THREE.Color(state.colorTop);
  const backRaw = new THREE.Color(state.colorBottom);
  // During flat/partial folding (0%..84%), display full contrasting washi
  // reverse side; as 180-deg box-pleated layers close flat (84%..100%), blend
  // inner pleat facets smoothly to a warm companion shade so untucked axial
  // river pleats read as cohesive sculpted paper rather than zebra patches.
  const closeBlend = Math.max(0.0, Math.min(1.0, (state.foldPct - 82.0) / 18.0));
  const tuckedCompanion = frontBase.clone().multiplyScalar(0.84).lerp(backRaw, 0.24);
  const effectiveBack = backRaw.clone().lerp(tuckedCompanion, closeBlend * 0.85);
  paperUniforms.uFrontColor.value.copy(frontBase);
  paperUniforms.uBackColor.value.copy(effectiveBack);

  const hiRegion = state.highlightedRegion;
  const { faces, triangulations, faceRegions } = bundle;
  const numFaces = faces.length || 1;

  // Pre-count single-sheet triangle vertices (3x leaner than double-slab + rim)
  let totalVerts = 0;
  for (let fIdx = 0; fIdx < faces.length; fIdx++) {
    const tri = triangulations[fIdx];
    if (!tri || tri.length < 3 || faces[fIdx].length < 3) continue;
    totalVerts += tri.length;
  }

  const posArr = new Float32Array(totalVerts * 3);
  const normArr = new Float32Array(totalVerts * 3);
  const colArr = new Float32Array(totalVerts * 3);
  let cursor = 0;

  const e1 = new THREE.Vector3();
  const e2 = new THREE.Vector3();
  const nrm = new THREE.Vector3();
  const canonNrm = new THREE.Vector3();

  for (let fIdx = 0; fIdx < faces.length; fIdx++) {
    const face = faces[fIdx];
    const tri = triangulations[fIdx];
    if (!tri || tri.length < 3 || face.length < 3) continue;

    // Robust Newell polygon normal across all vertices of the face
    nrm.set(0, 0, 0);
    for (let i = 0; i < face.length; i++) {
      const curr = pts[face[i]];
      const next = pts[face[(i + 1) % face.length]];
      nrm.x += (curr.y - next.y) * (curr.z + next.z);
      nrm.y += (curr.z - next.z) * (curr.x + next.x);
      nrm.z += (curr.x - next.x) * (curr.y + next.y);
    }
    if (nrm.lengthSq() < 1e-16) {
      e1.subVectors(pts[face[1]], pts[face[0]]);
      e2.subVectors(pts[face[2]], pts[face[0]]);
      nrm.crossVectors(e1, e2);
    }
    nrm.normalize();

    // Canonicalize normal sign so ALL coplanar 180-deg faces share the exact
    // same displacement axis — strictly separating stacked planes with zero Z-fighting!
    canonNrm.copy(nrm);
    const ax = Math.abs(canonNrm.x);
    const ay = Math.abs(canonNrm.y);
    const az = Math.abs(canonNrm.z);
    if (ax >= ay && ax >= az) {
      if (canonNrm.x < 0) canonNrm.negate();
    } else if (ay >= ax && ay >= az) {
      if (canonNrm.y < 0) canonNrm.negate();
    } else if (canonNrm.z < 0) {
      canonNrm.negate();
    }
    const layerSep = ((fIdx + 0.5) / numFaces - 0.5) * 0.00075;
    const ox = canonNrm.x * layerSep;
    const oy = canonNrm.y * layerSep;
    const oz = canonNrm.z * layerSep;

    const regionDim =
      hiRegion >= 0 && faceRegions && faceRegions[fIdx] !== hiRegion ? 0.38 : 1.0;
    const regionBoost =
      hiRegion >= 0 && faceRegions && faceRegions[fIdx] === hiRegion ? 1.14 : 1.0;

    for (let i = 0; i < tri.length; i++) {
      const vid = face[tri[i]];
      const p = pts[vid];
      const sh = Math.min(1.08, vertShade[vid] * regionDim * regionBoost);
      const o = cursor * 3;
      posArr[o] = p.x + ox;
      posArr[o + 1] = p.y + oy;
      posArr[o + 2] = p.z + oz;
      normArr[o] = nrm.x;
      normArr[o + 1] = nrm.y;
      normArr[o + 2] = nrm.z;
      colArr[o] = sh;
      colArr[o + 1] = sh;
      colArr[o + 2] = sh;
      cursor++;
    }
  }

  const geom = paperMesh.geometry;
  geom.setAttribute('position', new THREE.BufferAttribute(posArr, 3));
  geom.setAttribute('normal', new THREE.BufferAttribute(normArr, 3));
  geom.setAttribute('color', new THREE.BufferAttribute(colArr, 3));
  geom.computeBoundingSphere();

  paperMesh.material.transparent = state.xray;
  paperMesh.material.opacity = state.xray ? 0.58 : 1.0;
  paperMesh.material.depthWrite = !state.xray;
  paperMesh.material.needsUpdate = true;

  // Update 3D Crease Ridges Overlay
  creaseLines.visible = state.showCreases;
  if (state.showCreases) {
    const assigns = bundle.assignments;
    let activeLineCount = 0;
    for (let i = 0; i < edges.length; i++) {
      const a = assigns[i];
      if (a === 'B' || a === 'M' || a === 'V') activeLineCount++;
    }
    const lPos = new Float32Array(activeLineCount * 6);
    const lCol = new Float32Array(activeLineCount * 6);
    const borderCol = new THREE.Color(0x1c1917);
    const mtnCol = frontBase.clone().multiplyScalar(0.45);
    const valCol = frontBase.clone().multiplyScalar(0.32);

    let lc = 0;
    for (let i = 0; i < edges.length; i++) {
      const a = assigns[i];
      if (a !== 'B' && a !== 'M' && a !== 'V') continue;
      const p1 = pts[edges[i][0]];
      const p2 = pts[edges[i][1]];
      const o = lc * 6;
      lPos[o] = p1.x;
      lPos[o + 1] = p1.y;
      lPos[o + 2] = p1.z;
      lPos[o + 3] = p2.x;
      lPos[o + 4] = p2.y;
      lPos[o + 5] = p2.z;

      const c = a === 'B' ? borderCol : (a === 'M' ? mtnCol : valCol);
      lCol[o] = c.r; lCol[o + 1] = c.g; lCol[o + 2] = c.b;
      lCol[o + 3] = c.r; lCol[o + 4] = c.g; lCol[o + 5] = c.b;
      lc++;
    }
    creaseLines.geometry.setAttribute('position', new THREE.BufferAttribute(lPos, 3));
    creaseLines.geometry.setAttribute('color', new THREE.BufferAttribute(lCol, 3));
    creaseLines.material.opacity = state.foldPct < 10 ? 0.28 : 0.36;
  }

  if (refitCamera) {
    setCameraPreset('hero');
  } else if (camera && controls) {
    // Ensure unfolding to the larger flat sheet never clips the viewport
    const offset = camera.position.clone().sub(controls.target);
    const minSafeDist = Math.max(currentModelRadius, 0.24) * 2.85;
    controls.target.y = currentModelCenterY;
    if (offset.length() < minSafeDist) {
      offset.setLength(minSafeDist);
      camera.position.copy(controls.target).add(offset);
    }
    camera.lookAt(controls.target);
  }
  updateAnatomyPinPositions();
}

// ─── Camera Presets & 3D Anatomy Pin Projections ─────────────────────────────

function setCameraPreset(preset) {
  if (!camera || !controls) return;
  const r = Math.max(currentModelRadius, 0.23);
  const dist = r * 3.12;
  const targetY = currentModelCenterY;
  controls.target.set(0, targetY, 0);

  const azDefault = ((state.bundle?.cameraAzimuthDeg ?? 36) * Math.PI) / 180.0;
  const elDefault = ((state.bundle?.cameraElevationDeg ?? 20) * Math.PI) / 180.0;

  if (preset === 'hero') {
    camera.position.set(
      dist * Math.cos(elDefault) * Math.sin(azDefault),
      targetY + dist * Math.sin(elDefault),
      dist * Math.cos(elDefault) * Math.cos(azDefault)
    );
  } else if (preset === 'side') {
    camera.position.set(dist * 1.05, targetY + dist * 0.12, 0.001);
  } else if (preset === 'front') {
    camera.position.set(0.001, targetY + dist * 0.15, dist * 1.08);
  } else if (preset === 'top') {
    camera.position.set(0.001, targetY + dist * 1.35, dist * 0.08);
  }
  camera.lookAt(controls.target);
  controls.update();

  document.querySelectorAll('[data-cam]').forEach((btn) => {
    btn.classList.toggle('active', btn.dataset.cam === preset);
  });
}

function rebuildAnatomyPinsDOM() {
  const host = document.getElementById('anatomy-pins');
  host.innerHTML = '';
  if (!state.showPins || !state.bundle?.regions) return;

  const flaps = state.bundle.regions.filter((r) => r.kind === 'flap' && r.tip3d);
  flaps.slice(0, 14).forEach((reg) => {
    const el = document.createElement('div');
    el.className = 'anatomy-pin';
    el.dataset.regionId = String(reg.id);
    el.textContent = reg.label;
    el.addEventListener('mouseenter', () => setHighlightedRegion(reg.id));
    el.addEventListener('mouseleave', () => setHighlightedRegion(-1));
    host.appendChild(el);
  });
  updateAnatomyPinPositions();
}

function updateAnatomyPinPositions() {
  const host = document.getElementById('anatomy-pins');
  if (!state.showPins || !host.children.length || !state.bundle) return;
  const wrap = document.getElementById('stage-canvas-wrap');
  const w = wrap.clientWidth;
  const h = wrap.clientHeight;
  const regions = state.bundle.regions || [];

  for (const pin of host.children) {
    const rid = Number(pin.dataset.regionId);
    const reg = regions.find((r) => r.id === rid);
    if (!reg || !reg.tip3d || state.foldPct < 75) {
      pin.style.display = 'none';
      continue;
    }
    const v = new THREE.Vector3(reg.tip3d[0], reg.tip3d[1], reg.tip3d[2]);
    v.project(camera);
    if (v.z > 1.0 || v.z < -1.0) {
      pin.style.display = 'none';
      continue;
    }
    pin.style.display = 'block';
    pin.style.left = `${(v.x * 0.5 + 0.5) * w}px`;
    pin.style.top = `${(-v.y * 0.5 + 0.5) * h}px`;
    pin.classList.toggle('highlighted', state.highlightedRegion === rid);
  }
}

// ─── Secondary 3D Stick-Figure Skeleton Renderer ─────────────────────────────

function initSkeletonStage() {
  if (skelRenderer) return;
  const wrap = document.getElementById('skeleton-canvas-wrap');
  const canvas = document.getElementById('skeleton-canvas');
  const w = wrap.clientWidth || 340;
  const h = wrap.clientHeight || 280;

  skelRenderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
  skelRenderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  skelRenderer.setSize(w, h, false);

  skelScene = new THREE.Scene();
  skelCamera = new THREE.PerspectiveCamera(42, w / h, 0.05, 200);
  skelCamera.position.set(12, 9, 14);

  skelControls = new OrbitControls(skelCamera, canvas);
  skelControls.enableDamping = true;

  skelScene.add(new THREE.AmbientLight(0xffffff, 0.85));
  const dl = new THREE.DirectionalLight(0xffffff, 0.75);
  dl.position.set(10, 18, 12);
  skelScene.add(dl);
}

function renderStickFigure3D(sf) {
  initSkeletonStage();
  const toRemove = [];
  skelScene.traverse((o) => {
    if (o.userData?.isSkeleton) toRemove.push(o);
  });
  toRemove.forEach((o) => skelScene.remove(o));
  if (!sf || !Array.isArray(sf.children) || !sf.children.length) return;

  // Compute 3D joint positions in Three.js frame (X=left, Y=up, Z=forward)
  const sticks = sf.children;
  const root = sticks.find((s) => s.name === sf.root) || sticks[0];
  const adj = {};
  sticks.forEach((s) => {
    (adj[s.start] ||= []).push(s);
    (adj[s.end] ||= []).push(s);
  });

  const pos = { [root.start]: new THREE.Vector3(0, 0, 0) };
  const visitedSticks = new Set();
  const queue = [root.start];

  const vecOf = (s) => {
    const az = (s.azimuth_angle * Math.PI) / 180.0;
    const el = (s.elevation_angle * Math.PI) / 180.0;
    const xf = s.length * Math.cos(el) * Math.cos(az);
    const yl = s.length * Math.cos(el) * Math.sin(az);
    const zu = s.length * Math.sin(el);
    return new THREE.Vector3(yl, zu, xf);
  };

  while (queue.length > 0) {
    const j = queue.shift();
    for (const s of adj[j] || []) {
      if (visitedSticks.has(s.name)) continue;
      visitedSticks.add(s.name);
      const d = vecOf(s);
      if (s.start === j) {
        pos[s.end] = pos[j].clone().add(d);
        queue.push(s.end);
      } else {
        pos[s.start] = pos[j].clone().sub(d);
        queue.push(s.start);
      }
    }
  }

  const box = new THREE.Box3();
  Object.values(pos).forEach((p) => box.expandByPoint(p));
  const center = new THREE.Vector3();
  box.getCenter(center);
  const size = new THREE.Vector3();
  box.getSize(size);
  const maxDim = Math.max(size.x, size.y, size.z, 2.0);

  const stickMat = new THREE.MeshStandardMaterial({ color: 0x2563eb, roughness: 0.45 });
  const rootMat = new THREE.MeshStandardMaterial({ color: 0xb83a2d, roughness: 0.45 });
  const jointMat = new THREE.MeshStandardMaterial({ color: 0x64748b, roughness: 0.35 });

  for (const s of sticks) {
    const p0 = pos[s.start];
    const p1 = pos[s.end];
    if (!p0 || !p1) continue;
    const a = p0.clone().sub(center);
    const b = p1.clone().sub(center);
    const dist = a.distanceTo(b);
    if (dist < 1e-5) continue;
    const rad = Math.max(0.05, Math.min(0.28, (s.width || 1) * 0.055));
    const cyl = new THREE.Mesh(
      new THREE.CylinderGeometry(rad, rad, dist, 12),
      s.name === sf.root ? rootMat : stickMat
    );
    cyl.position.copy(a).lerp(b, 0.5);
    cyl.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), b.clone().sub(a).normalize());
    cyl.userData = { isSkeleton: true };
    skelScene.add(cyl);
  }

  const jGeo = new THREE.SphereGeometry(maxDim * 0.018, 12, 10);
  for (const p of Object.values(pos)) {
    const sp = new THREE.Mesh(jGeo, jointMat);
    sp.position.copy(p).sub(center);
    sp.userData = { isSkeleton: true };
    skelScene.add(sp);
  }

  const grid = new THREE.GridHelper(maxDim * 1.4, 10, 0x94a3b8, 0xcbd5e1);
  grid.position.y = box.min.y - center.y - 0.1;
  grid.userData = { isSkeleton: true };
  skelScene.add(grid);

  const wrap = document.getElementById('skeleton-canvas-wrap');
  const w = wrap.clientWidth || 340;
  const h = wrap.clientHeight || 280;
  skelCamera.aspect = w / h;
  skelCamera.updateProjectionMatrix();
  skelRenderer.setSize(w, h, false);
  skelControls.target.set(0, 0, 0);
  skelCamera.position.set(maxDim * 0.95, maxDim * 0.65, maxDim * 1.05);
  skelCamera.lookAt(0, 0, 0);
  skelControls.update();
}

// ─── Main Animation Loop ─────────────────────────────────────────────────────

function animateLoop(now) {
  requestAnimationFrame(animateLoop);

  if (state.isAnimatingFold && state.bundle) {
    if (now >= state.animPauseUntil) {
      state.foldPct += state.animDir * 0.55;
      if (state.foldPct <= 0.0) {
        state.foldPct = 0.0;
        state.animDir = 1;
        state.animPauseUntil = now + 650;
      } else if (state.foldPct >= 100.0) {
        state.foldPct = 100.0;
        state.animDir = -1;
        state.animPauseUntil = now + 1100;
      }
      syncFoldSliderUI();
      updateOrigamiGeometry(false);
    }
  }

  if (state.turntable && controls && camera) {
    const offset = camera.position.clone().sub(controls.target);
    offset.applyAxisAngle(new THREE.Vector3(0, 1, 0), 0.0032);
    camera.position.copy(controls.target).add(offset);
    camera.lookAt(controls.target);
    if (state.showPins) updateAnatomyPinPositions();
  }

  if (controls) controls.update();
  if (renderer && scene && camera) renderer.render(scene, camera);

  if (state.activeTab === 'skeleton' && skelRenderer && skelScene && skelCamera) {
    skelControls?.update();
    skelRenderer.render(skelScene, skelCamera);
  }
}

// ─── Inspector Diagram Rendering (SVG Crease Patterns + Region Highlight) ────

function formatSvgForDisplay(svgText) {
  const clean = svgText.trim().replace(/^<\?xml[^?]*\?>/, '').trim();
  const doc = new DOMParser().parseFromString(clean, 'image/svg+xml');
  const svg = doc.documentElement;
  if (doc.querySelector('parsererror')) return svgText;

  svg.setAttribute('width', '100%');
  svg.setAttribute('height', '100%');
  const vb = (svg.getAttribute('viewBox') || '0 0 1000 1000').trim().split(/\s+/).map(Number);
  if (vb.length === 4 && vb[2] > 0 && vb[3] > 0) {
    const [minX, minY, w, h] = vb;
    const scale = 1000 / Math.max(w, h);
    svg.setAttribute('viewBox', '-16 -16 1032 1032');
    const g = doc.createElementNS('http://www.w3.org/2000/svg', 'g');
    g.setAttribute('transform', `scale(${scale}) translate(${-minX}, ${-minY})`);
    g.setAttribute('stroke-width', String(1.65 / scale));
    g.setAttribute('stroke-linecap', 'round');
    while (svg.firstChild) g.appendChild(svg.firstChild);

    const bg = doc.createElementNS('http://www.w3.org/2000/svg', 'rect');
    bg.setAttribute('x', '0');
    bg.setAttribute('y', '0');
    bg.setAttribute('width', String(w * scale));
    bg.setAttribute('height', String(h * scale));
    bg.setAttribute('fill', '#ffffff');
    bg.setAttribute('stroke', '#111827');
    bg.setAttribute('stroke-width', '2.5');
    svg.appendChild(bg);
    svg.appendChild(g);
  }
  return new XMLSerializer().serializeToString(svg);
}

async function updateInspectorTab() {
  const cpHost = document.getElementById('cp-svg-host');
  const skelWrap = document.getElementById('skeleton-canvas-wrap');
  const legend = document.getElementById('cp-legend-pills');

  document.querySelectorAll('.tab-btn').forEach((b) => {
    b.classList.toggle('active', b.dataset.tab === state.activeTab);
  });

  if (state.activeTab === 'skeleton') {
    cpHost.style.display = 'none';
    skelWrap.style.display = 'block';
    legend.innerHTML = '<span>Drag to orbit 3D metric StickFigure tree</span>';
    renderStickFigure3D(state.bundle?.stickFigure);
    return;
  }

  skelWrap.style.display = 'none';
  cpHost.style.display = 'flex';
  legend.innerHTML =
    state.activeTab === 'packing'
      ? '<span><i class="legend-dot" style="background:#dc2626"></i>Flap Circles</span><span><i class="legend-dot" style="background:#16a34a"></i>Rivers</span>'
      : '<span><i class="legend-dot" style="background:#dc2626"></i>Mountain</span><span><i class="legend-dot" style="background:#2563eb"></i>Valley</span><span><i class="legend-dot" style="background:#111827"></i>Border</span>';

  const meta = state.activeMeta;
  if (!meta) return;

  // Check local uploaded SVG override first
  const localSvgMap = {
    shaped: state.bundle?._localShapedSvg,
    solving: state.bundle?._localSolvingSvg || state.bundle?._localShapedSvg,
    packing: state.bundle?._localPackingSvg || state.bundle?._localShapedSvg,
  };
  if (localSvgMap[state.activeTab]) {
    cpHost.innerHTML = formatSvgForDisplay(localSvgMap[state.activeTab]);
    return;
  }

  const urlMap = {
    shaped: meta.files?.shapedSvg,
    solving: meta.files?.solvingSvg,
    packing: meta.files?.packingSvg,
  };
  const url = urlMap[state.activeTab];
  if (!url) {
    cpHost.innerHTML = '<span style="color:var(--text-muted);font-size:0.82rem;">Diagram unavailable</span>';
    return;
  }
  const cacheKey = `${meta.slug}:${state.activeTab}`;
  if (!state.svgCache[cacheKey]) {
    try {
      const res = await fetch(url);
      state.svgCache[cacheKey] = res.ok ? await res.text() : '';
    } catch {
      state.svgCache[cacheKey] = '';
    }
  }
  cpHost.innerHTML = state.svgCache[cacheKey]
    ? formatSvgForDisplay(state.svgCache[cacheKey])
    : '<span style="color:var(--text-muted);font-size:0.82rem;">Diagram unavailable</span>';
}

// ─── Loading Curated Showcase Models ─────────────────────────────────────────

function setHighlightedRegion(regionId) {
  if (state.highlightedRegion === regionId) return;
  state.highlightedRegion = regionId;
  document.querySelectorAll('.region-pill').forEach((el) => {
    el.classList.toggle('active', Number(el.dataset.regionId) === regionId);
  });
  updateOrigamiGeometry(false);
}

function syncFoldSliderUI() {
  const slider = document.getElementById('fold-slider');
  const cap = document.getElementById('fold-stage-caption');
  slider.value = String(state.foldPct);
  const pctInt = Math.round(state.foldPct);
  let phase = '3D Sculpted Pose';
  if (pctInt === 0) phase = 'Flat Washi Square';
  else if (state.twoStage && pctInt < 50) phase = 'Stage 1: Box-Pleated Base';
  else if (state.twoStage && pctInt === 50) phase = 'Uniaxial Flat-Foldable Base';
  else if (pctInt < 100) phase = 'Stage 2: 3D Shaping';
  cap.innerHTML = `<strong>${pctInt}%</strong> &middot; ${phase}`;
}

function applyPaperColors(topHex, bottomHex) {
  state.colorTop = topHex;
  state.colorBottom = bottomHex;
  document.getElementById('front-color-picker').value = topHex;
  document.getElementById('back-color-picker').value = bottomHex;
  updateOrigamiGeometry(false);
}

async function selectShowcaseModel(slug, scrollToStage = false) {
  const meta = state.catalog.find((m) => m.slug === slug);
  if (!meta) return;

  const loader = document.getElementById('stage-loader');
  loader.style.display = 'flex';
  state.activeSlug = slug;
  state.activeMeta = meta;
  state.highlightedRegion = -1;
  state.isAnimatingFold = false;
  document.getElementById('fold-play-btn').textContent = '▶ Unfold / Fold';

  // Update URL hash cleanly without jumping
  if (history.replaceState) {
    history.replaceState(null, '', `#${slug}`);
  }

  // Highlight active pill & gallery card
  document.querySelectorAll('.model-pill').forEach((b) => {
    b.classList.toggle('active', b.dataset.slug === slug);
  });
  document.querySelectorAll('.gallery-card').forEach((c) => {
    c.classList.toggle('selected', c.dataset.slug === slug);
  });

  try {
    const res = await fetch(meta.bundleUrl);
    const bundle = await res.json();
    state.bundle = bundle;

    // Apply paper palette (model signature duo-tone unless user locked another preset)
    if (state.activeSwatch === 'model') {
      state.colorTop = meta.colorTop || '#9E3329';
      state.colorBottom = meta.colorBottom || '#F3E7CE';
      document.getElementById('front-color-picker').value = state.colorTop;
      document.getElementById('back-color-picker').value = state.colorBottom;
      updateSwatchButtons();
    }

    // Populate HUD & Inspector specs
    document.getElementById('stage-title').textContent = meta.title;
    document.getElementById('stage-subtitle').textContent = meta.subtitle || '';
    document.getElementById('spec-grid').textContent = meta.gridSize ? `${meta.gridSize}×${meta.gridSize}` : 'Custom';
    document.getElementById('spec-sticks').textContent = String(meta.sticks);
    document.getElementById('spec-creases').textContent = meta.creases.toLocaleString();
    document.getElementById('spec-strain').textContent = `${meta.vertexErrorPct.toFixed(2)}%`;
    document.getElementById('inspector-summary').textContent = meta.summary;

    const techWrap = document.getElementById('inspector-techniques');
    techWrap.innerHTML = '';
    (meta.techniques || []).forEach((t) => {
      const span = document.createElement('span');
      span.className = 'tech-tag';
      span.textContent = t;
      techWrap.appendChild(span);
    });

    // Populate interactive flap/river pills
    const regHost = document.getElementById('region-pills');
    const regExplorer = document.getElementById('region-explorer');
    regHost.innerHTML = '';
    const regions = bundle.regions || [];
    if (regions.length > 0) {
      regExplorer.style.display = 'flex';
      regions.forEach((r) => {
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = 'region-pill';
        btn.dataset.regionId = String(r.id);
        btn.textContent = r.label;
        btn.addEventListener('mouseenter', () => setHighlightedRegion(r.id));
        btn.addEventListener('mouseleave', () => setHighlightedRegion(-1));
        btn.addEventListener('click', () => {
          setHighlightedRegion(state.highlightedRegion === r.id ? -1 : r.id);
        });
        regHost.appendChild(btn);
      });
    } else {
      regExplorer.style.display = 'none';
    }

    // Wire download links
    document.getElementById('dl-fold').href = meta.files.fold;
    document.getElementById('dl-fold').download = `${meta.slug}.fold`;
    document.getElementById('dl-svg').href = meta.files.shapedSvg;
    document.getElementById('dl-svg').download = `${meta.slug}_shaped_cp.svg`;
    document.getElementById('dl-json').href = meta.files.stickFigure;
    document.getElementById('dl-json').download = `${meta.slug}_stick_figure.json`;

    state.foldPct = 100.0;
    syncFoldSliderUI();
    updateOrigamiGeometry(true);
    rebuildAnatomyPinsDOM();
    await updateInspectorTab();

    if (scrollToStage) {
      window.scrollTo({ top: 0, behavior: 'smooth' });
    }
  } finally {
    loader.style.display = 'none';
  }
}

// ─── Client-Side Local File Parser (.fold, .svg, stick_figure.json) ──────────

function parseBrowserSvgToBundle(svgText, title = 'Uploaded Crease Pattern') {
  const doc = new DOMParser().parseFromString(svgText, 'image/svg+xml');
  const vb = (doc.documentElement.getAttribute('viewBox') || '0 0 1000 1000').trim().split(/\s+/).map(Number);
  const [xMin, yMin, width, height] = vb.length === 4 ? vb : [0, 0, 1000, 1000];
  const scale = Math.max(width || 1000, height || 1000);
  const tol = 1e-4;
  const cellSize = tol * 2;
  const vertices = [];
  const vGrid = new Map();

  const getVid = (x, y) => {
    const gx = Math.floor(x / cellSize);
    const gy = Math.floor(y / cellSize);
    for (let dx = -1; dx <= 1; dx++) {
      for (let dy = -1; dy <= 1; dy++) {
        const arr = vGrid.get(`${gx + dx},${gy + dy}`);
        if (!arr) continue;
        for (const idx of arr) {
          const d2 = (vertices[idx][0] - x) ** 2 + (vertices[idx][1] - y) ** 2;
          if (d2 < tol * tol) return idx;
        }
      }
    }
    const idx = vertices.length;
    vertices.push([x, y]);
    const key = `${gx},${gy}`;
    if (!vGrid.has(key)) vGrid.set(key, []);
    vGrid.get(key).push(idx);
    return idx;
  };

  const prec = { M: 3, V: 3, F: 2, B: 1, U: 0 };
  const parseStroke = (s) => {
    const c = (s || '').trim().toLowerCase().replace(/\s+/g, '');
    if (c === 'red' || c === '#ff0000' || c === '#f00' || c.startsWith('rgb(255,0,0)')) return ['M', -180];
    if (c === 'blue' || c === '#0000ff' || c === '#00f' || c.startsWith('rgb(0,0,255)')) return ['V', 180];
    if (c === 'black' || c === '#000000' || c === '#000' || c.startsWith('rgb(0,0,0)')) return ['B', 0];
    if (c === 'yellow' || c === 'green' || c === 'gray' || c === 'grey' || c === '#808080') return ['F', 0];
    return ['U', 0];
  };

  const rawMap = new Map();
  const addSeg = (rx1, ry1, rx2, ry2, stroke, op) => {
    const x1 = (rx1 - xMin) / scale - 0.5;
    const y1 = (ry1 - yMin) / scale - 0.5;
    const x2 = (rx2 - xMin) / scale - 0.5;
    const y2 = (ry2 - yMin) / scale - 0.5;
    if (Math.hypot(x2 - x1, y2 - y1) < 1e-5) return;
    const [a, baseAng] = parseStroke(stroke);
    if (a === 'U') return;
    const v1 = getVid(x1, y1);
    const v2 = getVid(x2, y2);
    if (v1 === v2) return;
    const k = v1 < v2 ? `${v1},${v2}` : `${v2},${v1}`;
    const prev = rawMap.get(k);
    if (!prev || prec[a] > prec[prev.a]) {
      rawMap.set(k, { u: Math.min(v1, v2), v: Math.max(v1, v2), a, ang: baseAng * op });
    }
  };

  doc.querySelectorAll('line').forEach((ln) => {
    const stroke = ln.getAttribute('stroke') || ln.parentElement?.getAttribute('stroke') || '';
    const op = parseFloat(ln.getAttribute('stroke-opacity') || ln.getAttribute('opacity') || '1');
    addSeg(
      parseFloat(ln.getAttribute('x1') || '0'),
      parseFloat(ln.getAttribute('y1') || '0'),
      parseFloat(ln.getAttribute('x2') || '0'),
      parseFloat(ln.getAttribute('y2') || '0'),
      stroke,
      op
    );
  });
  doc.querySelectorAll('rect').forEach((rc) => {
    const rx = parseFloat(rc.getAttribute('x') || '0');
    const ry = parseFloat(rc.getAttribute('y') || '0');
    const rw = parseFloat(rc.getAttribute('width') || '0');
    const rh = parseFloat(rc.getAttribute('height') || '0');
    if (rw > 0 && rh > 0) {
      addSeg(rx, ry, rx + rw, ry, 'black', 1);
      addSeg(rx + rw, ry, rx + rw, ry + rh, 'black', 1);
      addSeg(rx + rw, ry + rh, rx, ry + rh, 'black', 1);
      addSeg(rx, ry + rh, rx, ry, 'black', 1);
    }
  });

  // Fast spatial-bucket collinear split
  const bucket = 0.03;
  const vBuckets = new Map();
  vertices.forEach(([vx, vy], idx) => {
    const k = `${Math.floor(vx / bucket)},${Math.floor(vy / bucket)}`;
    if (!vBuckets.has(k)) vBuckets.set(k, []);
    vBuckets.get(k).push(idx);
  });

  const splitMap = new Map();
  for (const seg of rawMap.values()) {
    const [ax, ay] = vertices[seg.u];
    const [bx, by] = vertices[seg.v];
    const abx = bx - ax, aby = by - ay;
    const l2 = abx * abx + aby * aby;
    const len = Math.sqrt(l2);
    const ix0 = Math.floor((Math.min(ax, bx) - tol) / bucket);
    const ix1 = Math.floor((Math.max(ax, bx) + tol) / bucket);
    const iy0 = Math.floor((Math.min(ay, by) - tol) / bucket);
    const iy1 = Math.floor((Math.max(ay, by) + tol) / bucket);
    const mids = [];
    for (let gx = ix0; gx <= ix1; gx++) {
      for (let gy = iy0; gy <= iy1; gy++) {
        for (const cand of vBuckets.get(`${gx},${gy}`) || []) {
          if (cand === seg.u || cand === seg.v) continue;
          const [px, py] = vertices[cand];
          const avx = px - ax, avy = py - ay;
          if (Math.abs(abx * avy - aby * avx) / len < tol) {
            const dot = abx * avx + aby * avy;
            if (dot > tol * len && dot < l2 - tol * len) mids.push([dot / l2, cand]);
          }
        }
      }
    }
    mids.sort((p, q) => p[0] - q[0]);
    const chain = [seg.u, ...mids.map((m) => m[1]), seg.v];
    for (let k = 0; k < chain.length - 1; k++) {
      const u = Math.min(chain[k], chain[k + 1]);
      const v = Math.max(chain[k], chain[k + 1]);
      if (u === v) continue;
      const ek = `${u},${v}`;
      const prev = splitMap.get(ek);
      if (!prev || prec[seg.a] > prec[prev.a]) splitMap.set(ek, { u, v, a: seg.a, ang: seg.ang });
    }
  }

  const adj = Array.from({ length: vertices.length }, () => []);
  for (const e of splitMap.values()) {
    adj[e.u].push(e.v);
    adj[e.v].push(e.u);
  }
  vertices.forEach(([x0, y0], v) => {
    adj[v].sort((a, b) => Math.atan2(vertices[a][1] - y0, vertices[a][0] - x0) - Math.atan2(vertices[b][1] - y0, vertices[b][0] - x0));
  });
  const nextMap = new Map();
  adj.forEach((nbs, v) => {
    const n = nbs.length;
    nbs.forEach((u, i) => nextMap.set(`${u},${v}`, nbs[(i - 1 + n) % n]));
  });

  const faces = [];
  const visited = new Set();
  for (const key of nextMap.keys()) {
    if (visited.has(key)) continue;
    let [u, v] = key.split(',').map(Number);
    const face = [u, v];
    visited.add(key);
    let w = nextMap.get(key);
    let guard = 0;
    while (w !== undefined && w !== face[0] && guard++ < 2000) {
      face.push(w);
      visited.add(`${v},${w}`);
      u = v; v = w;
      w = nextMap.get(`${u},${v}`);
    }
    if (w === face[0] && face.length >= 3) {
      visited.add(`${face[face.length - 1]},${face[0]}`);
      let area = 0;
      for (let i = 0; i < face.length; i++) {
        const p1 = vertices[face[i]], p2 = vertices[face[(i + 1) % face.length]];
        area += p1[0] * p2[1] - p2[0] * p1[1];
      }
      if (area > 1e-9) faces.push(face.reverse());
    }
  }

  const edges = [];
  const assigns = [];
  const angles = [];
  const seenE = new Map();
  faces.forEach((f) => {
    for (let i = 0; i < f.length; i++) {
      const u = Math.min(f[i], f[(i + 1) % f.length]);
      const v = Math.max(f[i], f[(i + 1) % f.length]);
      const ek = `${u},${v}`;
      if (!seenE.has(ek)) {
        seenE.set(ek, edges.length);
        edges.push([u, v]);
        const info = splitMap.get(ek) || { a: 'B', ang: 0 };
        assigns.push(info.a);
        angles.push(info.ang);
      }
    }
  });

  const triangulations = faces.map((f) => {
    const flat = [];
    f.forEach((vid) => flat.push(vertices[vid][0], vertices[vid][1]));
    return earcut(flat);
  });

  let centralFace = 0, bestD = Infinity;
  faces.forEach((f, idx) => {
    const fx = f.reduce((s, vid) => s + vertices[vid][0], 0) / f.length;
    const fy = f.reduce((s, vid) => s + vertices[vid][1], 0) / f.length;
    if (fx * fx + fy * fy < bestD) { bestD = fx * fx + fy * fy; centralFace = idx; }
  });

  return {
    slug: 'custom-svg',
    title,
    subtitle: 'Local SVG Crease Pattern',
    colorTop: state.colorTop,
    colorBottom: state.colorBottom,
    uprightQuat: [0, 0, 0, 1],
    centralFace,
    vertices2d: vertices,
    faces,
    triangulations,
    edges,
    assignments: assigns.join(''),
    solvingAngles: null,
    shapingAngles: angles,
    edgeStage: null,
    faceRegions: null,
    regions: [],
    _localShapedSvg: svgText,
  };
}

function parseFoldJsonToBundle(fold, title = 'Uploaded FOLD Model') {
  const rawVerts = fold.vertices_coords || [];
  const is2D = rawVerts.length > 0 && rawVerts[0].length === 2;
  const vertices2d = rawVerts.map((v) => [v[0] - 0.5, v[1] - 0.5]);
  const faces = fold.faces_vertices || [];
  const edges = fold.edges_vertices || [];
  const assigns = (fold.edges_assignment || edges.map(() => 'B')).join('');
  const angles = fold.edges_foldAngle || (fold.edges_assignment || []).map((a) => (a === 'M' ? -180 : a === 'V' ? 180 : 0));
  const triangulations = faces.map((f) => {
    const flat = [];
    f.forEach((vid) => flat.push(vertices2d[vid][0], vertices2d[vid][1]));
    return earcut(flat);
  });
  return {
    slug: 'custom-fold',
    title: fold.file_title || title,
    subtitle: is2D ? 'Local FOLD v1.2 Crease Pattern' : 'Local FOLD v1.2 Model',
    colorTop: state.colorTop,
    colorBottom: state.colorBottom,
    uprightQuat: [0, 0, 0, 1],
    centralFace: 0,
    vertices2d,
    faces,
    triangulations,
    edges,
    assignments: assigns,
    solvingAngles: null,
    shapingAngles: angles,
    edgeStage: null,
    faceRegions: null,
    regions: [],
  };
}

async function handleUploadedFiles(fileList) {
  const files = Array.from(fileList || []);
  if (!files.length) return;

  let foldJson = null;
  let shapedSvg = null;
  let solvingSvg = null;
  let stickJson = null;
  let baseName = files[0].name.replace(/\.(fold|svg|json)$/i, '');

  for (const file of files) {
    const txt = await file.text();
    const lower = file.name.toLowerCase();
    if (lower.endsWith('.fold')) {
      foldJson = JSON.parse(txt);
    } else if (lower.endsWith('.svg')) {
      if (lower.includes('solv')) solvingSvg = txt;
      else shapedSvg = txt;
    } else if (lower.endsWith('.json')) {
      const parsed = JSON.parse(txt);
      if (parsed.vertices_coords && parsed.faces_vertices) foldJson = parsed;
      else if (parsed.children && parsed.root) stickJson = parsed;
    }
  }

  let bundle = null;
  if (foldJson) {
    bundle = parseFoldJsonToBundle(foldJson, baseName);
  } else if (shapedSvg || solvingSvg) {
    bundle = parseBrowserSvgToBundle(shapedSvg || solvingSvg, baseName);
    bundle._localSolvingSvg = solvingSvg;
  } else if (stickJson) {
    // StickFigure-only upload: switch straight to 3D Skeleton tab
    state.bundle = {
      ...(state.bundle || {}),
      stickFigure: stickJson,
    };
    document.getElementById('stage-title').textContent = stickJson.name || baseName;
    document.getElementById('stage-subtitle').textContent = 'Uploaded StickFigure Skeleton';
    state.activeTab = 'skeleton';
    await updateInspectorTab();
    return;
  }

  if (bundle) {
    if (stickJson) bundle.stickFigure = stickJson;
    state.bundle = bundle;
    state.activeMeta = {
      slug: bundle.slug,
      title: bundle.title,
      subtitle: bundle.subtitle,
      files: {},
    };
    document.getElementById('stage-title').textContent = bundle.title;
    document.getElementById('stage-subtitle').textContent = bundle.subtitle;
    document.getElementById('spec-grid').textContent = 'Local';
    document.getElementById('spec-sticks').textContent = stickJson ? String(stickJson.children.length) : '—';
    document.getElementById('spec-creases').textContent = bundle.edges.length.toLocaleString();
    document.getElementById('spec-strain').textContent = '—';
    document.getElementById('inspector-summary').textContent =
      'Loaded directly in your browser. Drag the Fold slider below to inspect continuous hinge kinematics.';
    document.getElementById('inspector-techniques').innerHTML = '';
    document.getElementById('region-explorer').style.display = 'none';
    state.foldPct = 100.0;
    syncFoldSliderUI();
    updateOrigamiGeometry(true);
    await updateInspectorTab();
  }
}

// ─── UI Construction & Event Wiring ──────────────────────────────────────────

function updateSwatchButtons() {
  document.querySelectorAll('.paper-swatch').forEach((el) => {
    el.classList.toggle('active', el.dataset.swatch === state.activeSwatch);
  });
}

function buildPaletteSwatches() {
  const group = document.getElementById('paper-palette-group');
  const pickers = group.querySelector('.color-pair-pickers');

  PAPER_SWATCHES.forEach((sw) => {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'paper-swatch';
    btn.dataset.swatch = sw.id;
    btn.title = sw.name;
    const c1 = sw.top || '#9E3329';
    const c2 = sw.bottom || '#F3E7CE';
    btn.style.background = `linear-gradient(135deg, ${c1} 50%, ${c2} 50%)`;
    btn.addEventListener('click', () => {
      state.activeSwatch = sw.id;
      updateSwatchButtons();
      const top = sw.top || state.activeMeta?.colorTop || '#9E3329';
      const bottom = sw.bottom || state.activeMeta?.colorBottom || '#F3E7CE';
      applyPaperColors(top, bottom);
    });
    group.insertBefore(btn, pickers);
  });
  updateSwatchButtons();
}

function renderCatalogUI() {
  const strip = document.getElementById('model-selector-strip');
  const grid = document.getElementById('gallery-grid');
  const filtersHost = document.getElementById('category-filters');

  strip.innerHTML = '';
  state.catalog.forEach((m) => {
    const pill = document.createElement('button');
    pill.type = 'button';
    pill.className = 'model-pill';
    pill.dataset.slug = m.slug;
    pill.innerHTML = `<span class="swatch-dot" style="background:linear-gradient(135deg, ${m.colorTop} 55%, ${m.colorBottom} 55%)"></span>${m.title}`;
    pill.addEventListener('click', () => selectShowcaseModel(m.slug, false));
    strip.appendChild(pill);
  });

  const categories = ['All', ...new Set(state.catalog.map((m) => m.category))];
  filtersHost.innerHTML = '';
  categories.forEach((cat) => {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = `filter-chip ${cat === state.activeCategory ? 'active' : ''}`;
    btn.textContent = cat === 'All' ? `All Models (${state.catalog.length})` : cat;
    btn.addEventListener('click', () => {
      state.activeCategory = cat;
      filtersHost.querySelectorAll('.filter-chip').forEach((b) => b.classList.remove('active'));
      btn.classList.add('active');
      renderGalleryCards();
    });
    filtersHost.appendChild(btn);
  });

  renderGalleryCards();
}

function renderGalleryCards() {
  const grid = document.getElementById('gallery-grid');
  grid.innerHTML = '';
  const visible = state.catalog.filter(
    (m) => state.activeCategory === 'All' || m.category === state.activeCategory
  );

  visible.forEach((m) => {
    const card = document.createElement('article');
    card.className = `gallery-card ${m.slug === state.activeSlug ? 'selected' : ''}`;
    card.dataset.slug = m.slug;
    card.innerHTML = `
      <div class="card-thumb">
        <img src="${m.preview}" alt="${m.title} 3D Origami Render" loading="lazy">
        <span class="card-grid-badge">${m.gridSize}×${m.gridSize} grid</span>
      </div>
      <div class="card-content">
        <div class="card-title-row">
          <h3>${m.title}</h3>
          <span class="card-taxon">${m.subtitle}</span>
        </div>
        <p class="card-desc">${m.summary}</p>
        <div class="technique-tags">
          ${(m.techniques || []).slice(0, 3).map((t) => `<span class="tech-tag">${t}</span>`).join('')}
        </div>
        <div class="card-footer">
          <span>${m.creases.toLocaleString()} creases &middot; ${m.sticks} sticks</span>
          <span class="card-cta">Fold in 3D →</span>
        </div>
      </div>
    `;
    card.addEventListener('click', () => selectShowcaseModel(m.slug, true));
    grid.appendChild(card);
  });
}

function bindEvents() {
  // Fold Scrubber & Play/Pause
  const slider = document.getElementById('fold-slider');
  slider.addEventListener('input', () => {
    state.foldPct = parseFloat(slider.value);
    state.isAnimatingFold = false;
    document.getElementById('fold-play-btn').textContent = '▶ Unfold / Fold';
    syncFoldSliderUI();
    updateOrigamiGeometry(false);
  });

  document.getElementById('fold-play-btn').addEventListener('click', (e) => {
    state.isAnimatingFold = !state.isAnimatingFold;
    e.currentTarget.textContent = state.isAnimatingFold ? '⏸ Pause' : '▶ Unfold / Fold';
    if (state.isAnimatingFold) {
      state.animDir = state.foldPct >= 99 ? -1 : 1;
      state.animPauseUntil = 0;
    }
  });

  // Camera Presets & Turntable
  document.querySelectorAll('[data-cam]').forEach((btn) => {
    btn.addEventListener('click', () => setCameraPreset(btn.dataset.cam));
  });

  document.getElementById('turntable-btn').addEventListener('click', (e) => {
    state.turntable = !state.turntable;
    e.currentTarget.classList.toggle('active', state.turntable);
  });

  // Studio Material Toggles
  document.getElementById('front-color-picker').addEventListener('input', (e) => {
    state.activeSwatch = 'custom';
    updateSwatchButtons();
    applyPaperColors(e.target.value, state.colorBottom);
  });
  document.getElementById('back-color-picker').addEventListener('input', (e) => {
    state.activeSwatch = 'custom';
    updateSwatchButtons();
    applyPaperColors(state.colorTop, e.target.value);
  });

  document.getElementById('toggle-creases-btn').addEventListener('click', (e) => {
    state.showCreases = !state.showCreases;
    e.currentTarget.classList.toggle('active', state.showCreases);
    updateOrigamiGeometry(false);
  });

  document.getElementById('toggle-two-stage-btn').addEventListener('click', (e) => {
    state.twoStage = !state.twoStage;
    e.currentTarget.classList.toggle('active', state.twoStage);
    syncFoldSliderUI();
    updateOrigamiGeometry(false);
  });

  document.getElementById('toggle-pins-btn').addEventListener('click', (e) => {
    state.showPins = !state.showPins;
    e.currentTarget.classList.toggle('active', state.showPins);
    rebuildAnatomyPinsDOM();
  });

  document.getElementById('toggle-xray-btn').addEventListener('click', (e) => {
    state.xray = !state.xray;
    e.currentTarget.classList.toggle('active', state.xray);
    updateOrigamiGeometry(false);
  });

  // Inspector Pipeline Tabs
  document.querySelectorAll('.tab-btn').forEach((btn) => {
    btn.addEventListener('click', () => {
      state.activeTab = btn.dataset.tab;
      updateInspectorTab();
    });
  });

  // Theme Toggle
  document.getElementById('theme-toggle-btn').addEventListener('click', () => {
    const html = document.documentElement;
    const next = html.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
    html.setAttribute('data-theme', next);
    document.getElementById('theme-icon').textContent = next === 'dark' ? '☀' : '☾';
    if (shadowPlane) {
      shadowPlane.material.opacity = next === 'dark' ? 0.34 : 0.18;
    }
  });

  // Local File Input & Full-Window Drag-and-Drop
  document.getElementById('local-file-input').addEventListener('change', (e) => {
    handleUploadedFiles(e.target.files);
  });

  const dropOverlay = document.getElementById('drop-overlay');
  let dragDepth = 0;
  window.addEventListener('dragenter', (e) => {
    e.preventDefault();
    dragDepth++;
    dropOverlay.classList.add('visible');
  });
  window.addEventListener('dragover', (e) => e.preventDefault());
  window.addEventListener('dragleave', () => {
    dragDepth = Math.max(0, dragDepth - 1);
    if (dragDepth === 0) dropOverlay.classList.remove('visible');
  });
  window.addEventListener('drop', (e) => {
    e.preventDefault();
    dragDepth = 0;
    dropOverlay.classList.remove('visible');
    if (e.dataTransfer?.files?.length) {
      handleUploadedFiles(e.dataTransfer.files);
    }
  });

  // Contribute Modal
  const modal = document.getElementById('contribute-modal');
  document.getElementById('contribute-btn').addEventListener('click', () => modal.classList.add('open'));
  document.getElementById('close-modal-btn').addEventListener('click', () => modal.classList.remove('open'));
  modal.addEventListener('click', (e) => {
    if (e.target === modal) modal.classList.remove('open');
  });
}

// ─── Headless Studio Thumbnail Helper (for scripts/render_previews) ──────────

window.renderCleanThumbnailMode = async function (slug) {
  state.turntable = false;
  document.querySelector('.site-header').style.display = 'none';
  document.querySelector('.selector-strip').style.display = 'none';
  document.querySelector('.inspector-card').style.display = 'none';
  document.querySelector('.gallery-section').style.display = 'none';
  document.querySelector('.site-footer').style.display = 'none';
  document.querySelector('.stage-top-hud').style.display = 'none';
  document.querySelector('.stage-bottom-dock').style.display = 'none';
  document.querySelector('.main-wrap').style.padding = '0';
  document.querySelector('.main-wrap').style.maxWidth = 'none';
  document.querySelector('.studio-grid').style.gridTemplateColumns = '1fr';
  const sc = document.querySelector('.stage-card');
  sc.style.borderRadius = '0';
  sc.style.border = 'none';
  sc.style.minHeight = '100vh';
  const wrap = document.getElementById('stage-canvas-wrap');
  wrap.style.height = '100vh';

  await new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));
  const w = wrap.clientWidth || 800;
  const h = wrap.clientHeight || 600;
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  renderer.setSize(w, h, false);

  await selectShowcaseModel(slug, false);
  setCameraPreset('hero');
  renderer.render(scene, camera);
  await new Promise((r) => setTimeout(r, 80));
  renderer.render(scene, camera);
  document.title = `THUMB_READY:${slug}`;
};

// ─── Bootstrap ───────────────────────────────────────────────────────────────

async function init() {
  const params = new URLSearchParams(window.location.search);
  const isThumbMode = params.get('thumb') === '1';
  const isStaticShot = params.get('static') === '1';

  initMainStage();
  buildPaletteSwatches();
  bindEvents();
  if (!isThumbMode && !isStaticShot) {
    requestAnimationFrame(animateLoop);
  }

  const res = await fetch('models/catalog.json');
  state.catalog = await res.json();
  if (!isThumbMode) {
    renderCatalogUI();
  }

  const hashSlug = window.location.hash.replace(/^#/, '');
  const initialSlug =
    params.get('model') ||
    (state.catalog.some((m) => m.slug === hashSlug) ? hashSlug : state.catalog[0]?.slug);

  if (initialSlug) {
    if (isThumbMode) {
      await window.renderCleanThumbnailMode(initialSlug);
    } else {
      await selectShowcaseModel(initialSlug, false);
      if (params.get('fold')) {
        state.foldPct = parseFloat(params.get('fold'));
        syncFoldSliderUI();
        updateOrigamiGeometry(false);
      }
      if (params.get('theme') === 'dark') {
        document.getElementById('theme-toggle-btn').click();
      }
      if (params.get('tab')) {
        state.activeTab = params.get('tab');
        await updateInspectorTab();
      }
      if (params.get('pins') === '1') {
        document.getElementById('toggle-pins-btn').click();
      }
      if (isStaticShot) {
        renderer.render(scene, camera);
        if (state.activeTab === 'skeleton' && skelRenderer) {
          skelRenderer.render(skelScene, skelCamera);
        }
      }
    }
  }
}

init();
