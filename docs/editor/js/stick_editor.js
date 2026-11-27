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
 * @fileoverview Interactive WebGL 3D Stick Figure Editor using Three.js,
 * matching the COrigami stick-figure editor (`Stick.get_coords` + 3D cylinders,
 * joint spheres, OrbitControls, and camera-plane raycast joint dragging).
 *
 * Pure math helpers (`computeStickSkeleton`, `findMirroredStickName`,
 * `createStarterStickFigure`) stay top-level and free of DOM/WebGL imports so
 * Node unit tests run directly.
 */

/**
 * @param {number} deg
 * @return {number}
 */
function wrapDeg180(deg) {
  let a = deg % 360;
  if (a > 180) a -= 360;
  if (a <= -180) a += 360;
  return a === 0 ? 0 : +a.toFixed(1);
}

/**
 * Finds the bilateral L↔R partner stick name in `doc`, if present.
 * Supports `left`/`right`, `Left`/`Right`, `_l`/`_r`, `_L`/`_R`, `l_`/`r_`, `L_`/`R_`.
 *
 * @param {!Object} doc
 * @param {string} name
 * @return {?string}
 */
export function findMirroredStickName(doc, name) {
  if (!doc?.children) return null;
  const names = new Set(doc.children.map((s) => s.name));
  const rules = [
    [/left/g, 'right'],
    [/right/g, 'left'],
    [/Left/g, 'Right'],
    [/Right/g, 'Left'],
    [/_L(\d*)$/, '_R$1'],
    [/_R(\d*)$/, '_L$1'],
    [/_l(\d*)$/, '_r$1'],
    [/_r(\d*)$/, '_l$1'],
    [/_L(\d*)_/, '_R$1_'],
    [/_R(\d*)_/, '_L$1_'],
    [/_l(\d*)_/, '_r$1_'],
    [/_r(\d*)_/, '_l$1_'],
    [/^L_/, 'R_'],
    [/^R_/, 'L_'],
    [/^l_/, 'r_'],
    [/^r_/, 'l_'],
  ];
  for (const [pat, repl] of rules) {
    if (pat.test(name)) {
      const cand = name.replace(pat, repl);
      if (cand !== name && names.has(cand)) return cand;
    }
  }
  return null;
}

/**
 * Creates a clean starter stick figure for scratch designs.
 *
 * @return {!Object}
 */
export function createStarterStickFigure() {
  return {
    name: 'custom_creature',
    category: 'custom',
    complexity: 2,
    root: 'spine',
    children: [
      {name: 'spine', start: 'pelvis', end: 'chest', length: 4, width: 1, azimuth_angle: 0, elevation_angle: 10},
      {name: 'neck_head', start: 'chest', end: 'head', length: 3, width: 1, azimuth_angle: 0, elevation_angle: 55},
      {name: 'tail', start: 'pelvis', end: 'tail_tip', length: 4, width: 1, azimuth_angle: 180, elevation_angle: 15},
      {name: 'arm_left', start: 'chest', end: 'hand_left', length: 4, width: 1, azimuth_angle: 65, elevation_angle: -35},
      {name: 'arm_right', start: 'chest', end: 'hand_right', length: 4, width: 1, azimuth_angle: -65, elevation_angle: -35},
      {name: 'leg_left', start: 'pelvis', end: 'foot_left', length: 4, width: 1, azimuth_angle: 120, elevation_angle: -55},
      {name: 'leg_right', start: 'pelvis', end: 'foot_right', length: 4, width: 1, azimuth_angle: -120, elevation_angle: -55},
    ],
  };
}

/**
 * Computes 3D joint positions and parent-to-child directed traversal edges,
 * matching `Stick.get_coords` in `bp_pipeline.stick_figures.stick`:
 * `+X` = `-len * cos(el) * sin(az)`, `+Y` = `len * cos(el) * cos(az)` (forward),
 * `+Z` = `len * sin(el)` (up).
 *
 * @param {!Object} doc
 * @return {{
 *   joints: !Map<string, !Array<number>>,
 *   directed: !Map<string, {parent: string, child: string, forward: boolean}>
 * }}
 */
export function computeStickSkeleton(doc) {
  const joints = new Map();
  const directed = new Map();
  const sticks = doc?.children || [];
  if (!sticks.length) return {joints, directed};

  const root = sticks.find((s) => s.name === doc.root) || sticks[0];
  joints.set(root.start, [0, 0, 0]);
  const byJoint = new Map();
  for (const s of sticks) {
    if (!byJoint.has(s.start)) byJoint.set(s.start, []);
    if (!byJoint.has(s.end)) byJoint.set(s.end, []);
    byJoint.get(s.start).push(s);
    byJoint.get(s.end).push(s);
  }

  const queue = [root.start];
  while (queue.length > 0) {
    const cur = queue.shift();
    const p0 = joints.get(cur);
    for (const s of byJoint.get(cur) || []) {
      const forward = s.start === cur;
      const nextJ = forward ? s.end : s.start;
      if (joints.has(nextJ)) continue;
      const azDeg = forward ? Number(s.azimuth_angle || 0) :
                              Number(s.azimuth_angle || 0) + 180;
      const elDeg = forward ? Number(s.elevation_angle || 0) :
                              -Number(s.elevation_angle || 0);
      const az = (azDeg * Math.PI) / 180;
      const el = (elDeg * Math.PI) / 180;
      const len = Math.max(0.25, Number(s.length || 1));
      const xyProj = len * Math.cos(el);
      const dx = -xyProj * Math.sin(az);
      const dy = xyProj * Math.cos(az);
      const dz = len * Math.sin(el);
      joints.set(nextJ, [p0[0] + dx, p0[1] + dy, p0[2] + dz]);
      directed.set(s.name, {parent: cur, child: nextJ, forward});
      queue.push(nextJ);
    }
  }
  return {joints, directed};
}

let threePromise = null;
/** @return {!Promise<!Object>} */
async function ensureThree() {
  if (!threePromise) {
    threePromise = Promise.all([
      import('three'),
      import('three/addons/controls/OrbitControls.js'),
    ]).then(([THREE, ctrl]) => ({THREE, OrbitControls: ctrl.OrbitControls}));
  }
  return threePromise;
}

export class StickFigureEditor {
  /**
   * @param {!HTMLElement} container Dedicated `<div id="stick-stage">` element.
   * @param {!Object=} callbacks
   */
  constructor(container, callbacks = {}) {
    /** @private @const {!HTMLElement} */
    this._container = container;
    /** @private @const {!Object} */
    this._cb = callbacks;

    /** @type {boolean} */
    this.active = false;
    /** @type {boolean} */
    this.mirrorPose = true;
    /** @type {boolean} */
    this.showLabels = false;
    /** @type {boolean} */
    this.showAxes = true;

    /** @private {!Object} */
    this._doc = createStarterStickFigure();
    /** @private {?string} */
    this._selectedStick = 'spine';
    /** @private {boolean} */
    this._needsCameraFit = true;

    // Three.js objects (initialized lazily via `_initWebGL`).
    this._THREE = null;
    this._renderer = null;
    this._scene = null;
    this._camera = null;
    this._controls = null;
    this._raycaster = null;
    this._mouse = null;
    this._dragPlane = null;
    this._rootGroup = null;
    this._axesHelper = null;
    this._stickMeshes = new Map();
    this._jointMeshes = new Map();
    this._labelAnchors = [];
    this._tooltipEl = null;
    this._raf = 0;

    // Pointer / drag state.
    this._dragState = null;
    this._downClient = null;
    this._hoveredJoint = null;

    this._initPromise = this._initWebGL();
  }

  /** @private */
  async _initWebGL() {
    const {THREE, OrbitControls} = await ensureThree();
    this._THREE = THREE;

    const canvas = document.createElement('canvas');
    canvas.className = 'stick-canvas';
    this._container.append(canvas);

    const tooltip = document.createElement('div');
    tooltip.className = 'stick-tooltip';
    tooltip.hidden = true;
    this._container.append(tooltip);
    this._tooltipEl = tooltip;

    const renderer = new THREE.WebGLRenderer({
      canvas,
      antialias: true,
      alpha: true,
    });
    renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
    renderer.setClearColor(0x000000, 0);
    this._renderer = renderer;

    const scene = new THREE.Scene();
    scene.add(new THREE.AmbientLight(0xffffff, 0.85));
    const dir1 = new THREE.DirectionalLight(0xffffff, 0.65);
    dir1.position.set(12, -14, 18);
    scene.add(dir1);
    const dir2 = new THREE.DirectionalLight(0xffffff, 0.3);
    dir2.position.set(-10, 12, -8);
    scene.add(dir2);
    this._scene = scene;

    const camera = new THREE.PerspectiveCamera(44, 1, 0.05, 2000);
    camera.up.set(0, 0, 1);
    camera.position.set(14, -14, 10);
    this._camera = camera;

    const controls = new OrbitControls(camera, canvas);
    controls.enableDamping = true;
    controls.dampingFactor = 0.12;
    this._controls = controls;

    this._raycaster = new THREE.Raycaster();
    this._mouse = new THREE.Vector2();
    this._dragPlane = new THREE.Plane();
    this._rootGroup = new THREE.Group();
    scene.add(this._rootGroup);

    this._bindCanvasEvents(canvas);

    if (typeof ResizeObserver !== 'undefined') {
      const ro = new ResizeObserver(() => {
        if (this.active) this._resize();
      });
      ro.observe(this._container);
    }

    if (this.active) {
      this._resize();
      this._rebuildScene(/* fitCamera= */ true);
      this._startLoop();
    }
  }

  /** Enables or pauses the 3D Stick Figure viewport. */
  setActive(enabled) {
    const next = Boolean(enabled);
    if (this.active === next) return;
    this.active = next;
    this._container.hidden = !next;
    if (!next) {
      if (this._raf) {
        cancelAnimationFrame(this._raf);
        this._raf = 0;
      }
      if (this._tooltipEl) this._tooltipEl.hidden = true;
      return;
    }
    this._initPromise.then(() => {
      if (!this.active) return;
      const shouldFit = this._needsCameraFit;
      this._needsCameraFit = false;
      this._resize();
      this._rebuildScene(shouldFit);
      this._startLoop();
      requestAnimationFrame(() => {
        if (!this.active) return;
        this._resize();
        if (shouldFit) this._fitCamera();
      });
    });
  }

  /** Resets the 3D camera to frame the entire stick figure cleanly. */
  resetCamera() {
    if (!this._THREE) return;
    this._camera.position.set(14, -14, 10);
    this._controls.target.set(0, 0, 0);
    this._fitCamera();
  }

  /** Re-renders / refreshes scene geometry & labels. */
  render() {
    if (!this._THREE) return;
    this._rebuildScene(false);
  }

  /** @return {!Object} Deep copy of the edited StickFigure JSON. */
  get stickFigure() {
    return JSON.parse(JSON.stringify(this._doc));
  }

  /** @return {?Object} Currently selected stick record, or null. */
  get selectedStick() {
    return (this._doc.children || []).find((s) => s.name === this._selectedStick) || null;
  }

  /** Loads a StickFigure JSON into the editor. */
  setStickFigure(doc) {
    if (!doc?.children?.length) return;
    this._doc = JSON.parse(JSON.stringify(doc));
    if (!this._doc.children.some((s) => s.name === this._selectedStick)) {
      this._selectedStick = this._doc.root || this._doc.children[0].name;
    }
    this._needsCameraFit = true;
    if (this.active && this._THREE) {
      this._resize();
      this._rebuildScene(/* fitCamera= */ true);
      this._needsCameraFit = false;
    }
    this._emitSelect();
  }

  /** Updates numeric/boolean properties on the currently selected stick. */
  updateSelectedStick(patch) {
    const stick = this.selectedStick;
    if (!stick) return;
    Object.assign(stick, patch);
    if (this.mirrorPose) {
      const mirName = findMirroredStickName(this._doc, stick.name);
      const mir = mirName ? this._doc.children.find((s) => s.name === mirName) : null;
      if (mir) {
        if ('length' in patch) mir.length = stick.length;
        if ('width' in patch) mir.width = stick.width;
        if ('elevation_angle' in patch) mir.elevation_angle = stick.elevation_angle;
        if ('azimuth_angle' in patch) mir.azimuth_angle = wrapDeg180(-stick.azimuth_angle);
        if ('shaping_flap' in patch) mir.shaping_flap = stick.shaping_flap;
      }
    }
    if (this.active && this._THREE) this._rebuildScene(false);
    this._emitSelect();
    this._cb.onChange?.(this.stickFigure, true);
  }

  /** Appends a child stick branching from the selected stick's distal joint. */
  addChildStick(withMirror = false) {
    const parent = this.selectedStick || this._doc.children[0];
    if (!parent) return;
    const {directed} = computeStickSkeleton(this._doc);
    const dir = directed.get(parent.name);
    const attachJoint = dir ? dir.child : parent.end;
    const idx = this._doc.children.length + 1;

    if (withMirror) {
      const nameL = `branch_${idx}_left`;
      const nameR = `branch_${idx}_right`;
      this._doc.children.push(
          {
            name: nameL,
            start: attachJoint,
            end: `${nameL}_tip`,
            length: 3,
            width: 1,
            azimuth_angle: 55,
            elevation_angle: 15,
          },
          {
            name: nameR,
            start: attachJoint,
            end: `${nameR}_tip`,
            length: 3,
            width: 1,
            azimuth_angle: -55,
            elevation_angle: 15,
          });
      this._selectedStick = nameL;
    } else {
      const name = `branch_${idx}`;
      this._doc.children.push({
        name,
        start: attachJoint,
        end: `${name}_tip`,
        length: 3,
        width: 1,
        azimuth_angle: wrapDeg180((parent.azimuth_angle || 0) + 25),
        elevation_angle: parent.elevation_angle || 0,
      });
      this._selectedStick = name;
    }
    if (this.active && this._THREE) this._rebuildScene(true);
    this._emitSelect();
    this._cb.onChange?.(this.stickFigure, true);
  }

  /** Deletes the selected stick (and any disconnected subtree) if >1 stick remains. */
  deleteSelectedStick() {
    if (!this._selectedStick || this._doc.children.length <= 1) return;
    if (this._selectedStick === this._doc.root) return;
    const toRemove = new Set([this._selectedStick]);
    if (this.mirrorPose) {
      const mir = findMirroredStickName(this._doc, this._selectedStick);
      if (mir && mir !== this._doc.root) toRemove.add(mir);
    }
    const remaining = this._doc.children.filter((s) => !toRemove.has(s.name));
    const tempDoc = {...this._doc, children: remaining};
    const {joints} = computeStickSkeleton(tempDoc);
    this._doc.children = remaining.filter(
        (s) => joints.has(s.start) && joints.has(s.end));
    this._selectedStick = this._doc.root || this._doc.children[0]?.name || null;
    if (this.active && this._THREE) this._rebuildScene(false);
    this._emitSelect();
    this._cb.onChange?.(this.stickFigure, true);
  }

  /** @private */
  _emitSelect() {
    this._cb.onSelectStick?.(this.selectedStick);
  }

  /** @private */
  _resize() {
    if (!this._renderer || !this._camera) return;
    const w = Math.max(220, this._container.clientWidth);
    const h = Math.max(220, this._container.clientHeight);
    this._renderer.setSize(w, h, false);
    this._camera.aspect = w / h;
    this._camera.updateProjectionMatrix();
  }

  /** @private */
  _startLoop() {
    if (this._raf) return;
    const tick = () => {
      if (!this.active) {
        this._raf = 0;
        return;
      }
      this._raf = requestAnimationFrame(tick);
      this._controls.update();
      this._renderer.render(this._scene, this._camera);
      this._updateLabels();
    };
    this._raf = requestAnimationFrame(tick);
  }

  /** @private */
  _clearLabels() {
    for (const item of this._labelAnchors) {
      item.div?.remove();
    }
    this._labelAnchors = [];
  }

  /** @private */
  _rebuildScene(fitCamera = false) {
    const THREE = this._THREE;
    if (!THREE || !this._rootGroup) return;

    // Dispose old meshes cleanly.
    while (this._rootGroup.children.length > 0) {
      const child = this._rootGroup.children.pop();
      child.geometry?.dispose?.();
      child.material?.dispose?.();
    }
    if (this._axesHelper) {
      this._scene.remove(this._axesHelper);
      this._axesHelper.dispose?.();
      this._axesHelper = null;
    }
    this._clearLabels();
    this._stickMeshes.clear();
    this._jointMeshes.clear();

    const {joints, directed} = computeStickSkeleton(this._doc);
    this._lastSkeleton = {joints, directed};
    const pts = [...joints.values()];
    if (!pts.length) return;

    const box = new THREE.Box3();
    for (const [x, y, z] of pts) {
      box.expandByPoint(new THREE.Vector3(x, y, z));
    }
    const size = new THREE.Vector3();
    box.getSize(size);
    const maxDim = Math.max(size.x, size.y, size.z, 4);
    // Adaptive cylinder/sphere thickness so both 6-unit cats and 28-unit dragons look crisp!
    const unit = maxDim / 16;

    const degMap = new Map();
    for (const s of this._doc.children) {
      degMap.set(s.start, (degMap.get(s.start) || 0) + 1);
      degMap.set(s.end, (degMap.get(s.end) || 0) + 1);
    }

    const upVec = new THREE.Vector3(0, 1, 0);
    for (const stick of this._doc.children) {
      const p0Arr = joints.get(stick.start);
      const p1Arr = joints.get(stick.end);
      if (!p0Arr || !p1Arr) continue;
      const p0 = new THREE.Vector3(...p0Arr);
      const p1 = new THREE.Vector3(...p1Arr);
      const dist = p0.distanceTo(p1);
      if (dist <= 1e-6) continue;

      const isSelected = stick.name === this._selectedStick;
      const isRiver = (degMap.get(stick.start) || 0) > 1 &&
          (degMap.get(stick.end) || 0) > 1;
      // Softly scale width so 0.5-width claws stay clearly visible and 4-width wings stay sleek.
      const wFactor = 0.65 + 0.35 * Math.min(4, Math.max(0.4, Number(stick.width || 1)));
      const cylRadius = (isSelected ? 0.115 : 0.085) * unit * wFactor;

      const hex = isSelected ? 0xd97706 :
                  (stick.shaping_flap ? 0x9333ea :
                   (isRiver ? 0x0284c7 : 0x2563eb));
      const geo = new THREE.CylinderGeometry(cylRadius, cylRadius, dist, 12);
      const mat = new THREE.MeshStandardMaterial({
        color: hex,
        emissive: isSelected ? 0x451a03 : 0x000000,
        roughness: 0.45,
        metalness: 0.12,
      });
      const mesh = new THREE.Mesh(geo, mat);
      const mid = p0.clone().lerp(p1, 0.5);
      const dir = p1.clone().sub(p0).normalize();
      mesh.position.copy(mid);
      mesh.quaternion.setFromUnitVectors(upVec, dir);
      mesh.userData = {type: 'stick', stickName: stick.name};
      this._rootGroup.add(mesh);
      this._stickMeshes.set(stick.name, mesh);

      if (this.showLabels || isSelected) {
        const div = document.createElement('div');
        div.className = isSelected ? 'stick-3d-label selected' : 'stick-3d-label';
        div.textContent = `${stick.name.replaceAll('_', ' ')} (${stick.length})`;
        this._container.append(div);
        this._labelAnchors.push({pos: mid, div});
      }
    }

    const rootStick = this._doc.children.find((s) => s.name === this._doc.root) ||
        this._doc.children[0];
    const rootOrigin = rootStick?.start;
    const sphereRadius = 0.17 * unit;

    for (const [jName, posArr] of joints.entries()) {
      const isRoot = jName === rootOrigin;
      const isHover = jName === this._hoveredJoint;
      const geo = new THREE.SphereGeometry(
          isHover ? sphereRadius * 1.28 : sphereRadius, 16, 12);
      const mat = new THREE.MeshStandardMaterial({
        color: isHover ? 0xf59e0b : (isRoot ? 0xdc2626 : 0x38bdf8),
        roughness: 0.35,
        metalness: 0.15,
      });
      const sphere = new THREE.Mesh(geo, mat);
      sphere.position.set(posArr[0], posArr[1], posArr[2]);
      sphere.userData = {type: 'joint', jointName: jName, isRoot};
      this._rootGroup.add(sphere);
      this._jointMeshes.set(jName, sphere);
    }

    if (this.showAxes) {
      const center = new THREE.Vector3();
      box.getCenter(center);
      this._axesHelper = new THREE.AxesHelper(maxDim * 0.28);
      this._axesHelper.position.copy(center);
      this._scene.add(this._axesHelper);
    }

    if (fitCamera) {
      this._fitCamera();
    }
    this._updateLabels();
  }

  /** @private */
  _fitCamera() {
    const THREE = this._THREE;
    if (!THREE || !this._lastSkeleton) return;
    const pts = [...this._lastSkeleton.joints.values()];
    if (!pts.length) return;
    const box = new THREE.Box3();
    for (const [x, y, z] of pts) box.expandByPoint(new THREE.Vector3(x, y, z));
    const center = new THREE.Vector3();
    box.getCenter(center);
    const size = new THREE.Vector3();
    box.getSize(size);
    const maxDim = Math.max(size.x, size.y, size.z, 4);

    // Nice 3/4 isometric front-left angle (`+Y` is forward, `+Z` is up).
    const viewDir = new THREE.Vector3(-0.72, 0.72, 0.46).normalize();
    this._controls.target.copy(center);
    this._camera.position.copy(center).addScaledVector(viewDir, maxDim * 1.55);
    this._camera.lookAt(center);
    this._controls.update();
  }

  /** @private */
  _updateLabels() {
    if (!this._camera || !this._labelAnchors.length) return;
    const w = this._container.clientWidth;
    const h = this._container.clientHeight;
    if (w <= 0 || h <= 0) return;
    for (const {pos, div} of this._labelAnchors) {
      const proj = pos.clone().project(this._camera);
      if (proj.z > 1) {
        div.style.display = 'none';
        continue;
      }
      div.style.display = 'block';
      div.style.left = `${(proj.x * 0.5 + 0.5) * w}px`;
      div.style.top = `${(-proj.y * 0.5 + 0.5) * h}px`;
    }
  }

  /** @private */
  _setNDC(evt) {
    const rect = this._renderer.domElement.getBoundingClientRect();
    this._mouse.x = ((evt.clientX - rect.left) / Math.max(1, rect.width)) * 2 - 1;
    this._mouse.y = -((evt.clientY - rect.top) / Math.max(1, rect.height)) * 2 + 1;
  }

  /** @private */
  _bindCanvasEvents(canvas) {
    canvas.addEventListener('pointerdown', (evt) => {
      if (!this.active || evt.button !== 0 || !this._THREE) return;
      this._downClient = {x: evt.clientX, y: evt.clientY};
      this._setNDC(evt);
      this._raycaster.setFromCamera(this._mouse, this._camera);

      const spheres = [...this._jointMeshes.values()];
      const hits = this._raycaster.intersectObjects(spheres, false);
      if (hits.length > 0) {
        const sphere = hits[0].object;
        const jointName = sphere.userData.jointName;
        const {joints, directed} = this._lastSkeleton || computeStickSkeleton(this._doc);

        // Find the incoming stick whose distal `child` is `jointName`.
        let incoming = null;
        for (const s of this._doc.children) {
          const d = directed.get(s.name);
          if (d && d.child === jointName) {
            incoming = {stick: s, ...d};
            break;
          }
        }
        if (incoming) {
          this._selectedStick = incoming.stick.name;
          this._emitSelect();
          const camDir = new this._THREE.Vector3();
          this._camera.getWorldDirection(camDir);
          this._dragPlane.setFromNormalAndCoplanarPoint(camDir, sphere.position.clone());
          this._controls.enabled = false;
          this._dragState = {
            jointName,
            stickName: incoming.stick.name,
            forward: incoming.forward,
            parentPos: new this._THREE.Vector3(...joints.get(incoming.parent)),
          };
          canvas.setPointerCapture(evt.pointerId);
          this._rebuildScene(false);
          evt.preventDefault();
          evt.stopPropagation();
        }
      }
    });

    canvas.addEventListener('pointermove', (evt) => {
      if (!this.active || !this._THREE) return;
      this._setNDC(evt);

      if (this._dragState) {
        this._raycaster.setFromCamera(this._mouse, this._camera);
        const hitPos = new this._THREE.Vector3();
        if (!this._raycaster.ray.intersectPlane(this._dragPlane, hitPos)) return;
        const dir = hitPos.sub(this._dragState.parentPos);
        const len = dir.length();
        if (len <= 1e-4) return;
        dir.divideScalar(len);
        if (!this._dragState.forward) dir.negate();

        let el = (Math.asin(Math.max(-1, Math.min(1, dir.z))) * 180) / Math.PI;
        let az = (Math.atan2(-dir.x, dir.y) * 180) / Math.PI;
        if (evt.shiftKey) {
          az = Math.round(az / 15) * 15;
          el = Math.round(el / 15) * 15;
        }
        az = wrapDeg180(az);
        el = +Math.max(-89, Math.min(89, el)).toFixed(1);

        const stick = this._doc.children.find((s) => s.name === this._dragState.stickName);
        if (stick) {
          stick.azimuth_angle = az;
          stick.elevation_angle = el;
          if (this.mirrorPose) {
            const mirName = findMirroredStickName(this._doc, stick.name);
            const mir = mirName ? this._doc.children.find((s) => s.name === mirName) : null;
            if (mir) {
              mir.azimuth_angle = wrapDeg180(-az);
              mir.elevation_angle = el;
            }
          }
          this._rebuildScene(false);
          this._emitSelect();
          this._cb.onChange?.(this.stickFigure, false);
        }
        return;
      }

      // Hover detection on joints & sticks for live tooltip & cursor feedback.
      this._raycaster.setFromCamera(this._mouse, this._camera);
      const jointHits = this._raycaster.intersectObjects(
          [...this._jointMeshes.values()], false);
      if (jointHits.length > 0) {
        const obj = jointHits[0].object;
        const jName = obj.userData.jointName;
        if (this._hoveredJoint !== jName) {
          this._hoveredJoint = jName;
          this._rebuildScene(false);
        }
        canvas.style.cursor = obj.userData.isRoot ? 'default' : 'grab';
        const rect = this._container.getBoundingClientRect();
        this._tooltipEl.hidden = false;
        this._tooltipEl.style.left = `${evt.clientX - rect.left + 12}px`;
        this._tooltipEl.style.top = `${evt.clientY - rect.top - 10}px`;
        this._tooltipEl.textContent = obj.userData.isRoot ?
            `${jName} (root origin)` :
            `${jName} — drag to pose in 3D`;
        return;
      }

      if (this._hoveredJoint !== null) {
        this._hoveredJoint = null;
        this._rebuildScene(false);
      }

      const stickHits = this._raycaster.intersectObjects(
          [...this._stickMeshes.values()], false);
      if (stickHits.length > 0) {
        const sName = stickHits[0].object.userData.stickName;
        const s = this._doc.children.find((it) => it.name === sName);
        canvas.style.cursor = 'pointer';
        const rect = this._container.getBoundingClientRect();
        this._tooltipEl.hidden = false;
        this._tooltipEl.style.left = `${evt.clientX - rect.left + 12}px`;
        this._tooltipEl.style.top = `${evt.clientY - rect.top - 10}px`;
        this._tooltipEl.textContent = s ?
            `${s.name} · len ${s.length}, w ${s.width || 1} (click to select)` :
            sName;
      } else {
        canvas.style.cursor = 'grab';
        this._tooltipEl.hidden = true;
      }
    });

    canvas.addEventListener('pointerup', (evt) => {
      if (!this.active || !this._THREE) return;
      if (this._dragState) {
        this._dragState = null;
        this._controls.enabled = true;
        canvas.style.cursor = 'grab';
        this._cb.onChange?.(this.stickFigure, true);
        return;
      }
      // Click (not orbit drag) on a stick selects it!
      if (this._downClient &&
          Math.hypot(evt.clientX - this._downClient.x, evt.clientY - this._downClient.y) < 5) {
        this._setNDC(evt);
        this._raycaster.setFromCamera(this._mouse, this._camera);
        const hits = this._raycaster.intersectObjects(
            [...this._stickMeshes.values()], false);
        if (hits.length > 0) {
          this._selectedStick = hits[0].object.userData.stickName;
          this._emitSelect();
          this._rebuildScene(false);
        }
      }
      this._downClient = null;
    });
  }
}
