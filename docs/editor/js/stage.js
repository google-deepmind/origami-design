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
 * @fileoverview The 3D viewport: a folding sheet of paper on a ground plane.
 *
 * World axes follow `bp_pipeline`'s folded frame: +X is the model's left, +Y
 * is up and +Z is its front, with lengths in sheet widths.
 */

import {Folder} from '../../js/fold.js';

/** Camera presets as [azimuth, elevation] in degrees from the model's front. */
const VIEWS = {
  side: [90, 7],
  front: [0, 8],
  top: [0, 80]
};

/** The 'hero' view of a model that does not bring its own. */
const DEFAULT_HERO = {
  azimuth: 36,
  elevation: 20
};

/** Vertical field of view of the camera, in degrees. */
const FIELD_OF_VIEW = 36;

/** Opacity of the paper in x-ray mode. */
const XRAY_OPACITY = 0.55;

/** Color of the highlighted region of the paper. */
const HIGHLIGHT_COLOR = 0xf59e0b;

/** Soft hover tint when pointing at a selectable region on the 3D model. */
const HOVER_HIGHLIGHT_COLOR = 0xfbbf24;

/** Colors for creases actively folding in the current multi-stage step. */
const ACTIVE_MOUNTAIN_COLOR = 0xdc2626;
const ACTIVE_VALLEY_COLOR = 0x2563eb;

/**
 * Loads three.js, OrbitControls, and RoomEnvironment via the page's import map.
 *
 * @return {!Promise<{
 *   THREE: !Object,
 *   OrbitControls: !Function,
 *   RoomEnvironment: !Function
 * }>}
 */
async function loadThree() {
  const [THREE, controls, env] = await Promise.all([
    import('three'),
    import('three/addons/controls/OrbitControls.js'),
    import('three/addons/environments/RoomEnvironment.js'),
  ]);
  return {
    THREE,
    OrbitControls: controls.OrbitControls,
    RoomEnvironment: env.RoomEnvironment,
  };
}

/**
 * Generates a tileable procedural Japanese washi fiber bump texture.
 *
 * @param {!Object} THREE Loaded three.js module namespace.
 * @return {!Object}
 */
function buildWashiBumpTexture(THREE) {
  const size = 128;
  const canvas = document.createElement('canvas');
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = '#808080';
  ctx.fillRect(0, 0, size, size);

  let seed = 0x51f15eed;
  const rand = () => {
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };

  const img = ctx.getImageData(0, 0, size, size);
  const data = img.data;
  for (let i = 0; i < data.length; i += 4) {
    const grain = 120 + Math.floor(rand() * 16);
    data[i] = grain;
    data[i + 1] = grain;
    data[i + 2] = grain;
    data[i + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);

  ctx.lineWidth = 0.7;
  for (let i = 0; i < 220; i++) {
    const x = rand() * size;
    const y = rand() * size;
    const angle = rand() * Math.PI * 2;
    const len = 6 + rand() * 18;
    const dx = Math.cos(angle) * len;
    const dy = Math.sin(angle) * len;
    const shade = rand() > 0.45 ? 156 : 104;
    ctx.strokeStyle = `rgba(${shade}, ${shade}, ${shade}, 0.28)`;
    for (const ox of [-size, 0, size]) {
      for (const oy of [-size, 0, size]) {
        ctx.beginPath();
        ctx.moveTo(x + ox, y + oy);
        ctx.lineTo(x + dx + ox, y + dy + oy);
        ctx.stroke();
      }
    }
  }

  const texture = new THREE.CanvasTexture(canvas);
  texture.wrapS = THREE.RepeatWrapping;
  texture.wrapT = THREE.RepeatWrapping;
  texture.repeat.set(6, 6);
  return texture;
}

/**
 * Triangle vertex indices for the faces of a model, or for selected regions /
 * exact face subsets, together with the pick-group index of each triangle.
 *
 * @param {!Object} THREE Loaded three.js module namespace.
 * @param {!Object} model Parsed model from `fold.js`.
 * @param {?number|!Array<number>|!Set<number>|{faces: !Set<number>}=} filter
 *     Region/group indices, `{faces: Set<number>}` exact face subset, or null.
 * @param {?Int32Array=} facePickIds Optional per-face pick IDs overriding
 *     `model.faceRegions`.
 * @return {{index: !Array<number>, triangleRegions: !Int32Array}}
 */
function triangulate(THREE, model, filter = null, facePickIds = null) {
  const {flat, faces, faceRegions} = model;
  let groupSet = null;
  let exactFaces = null;
  if (filter !== null && filter !== undefined) {
    if (filter.faces instanceof Set) {
      exactFaces = filter.faces;
    } else if (typeof filter === 'number') {
      groupSet = new Set([filter]);
    } else if (filter instanceof Set) {
      groupSet = filter;
    } else if (Array.isArray(filter)) {
      groupSet = new Set(filter);
    }
  }
  const index = [];
  const triRegs = [];
  faces.forEach((face, f) => {
    const r = facePickIds ? facePickIds[f] : (faceRegions[f] ?? -1);
    if (exactFaces !== null && !exactFaces.has(f)) return;
    if (groupSet !== null && !groupSet.has(r)) return;
    // Faces can be concave, so fan triangulation is not enough.
    const outline =
        face.map((v) => new THREE.Vector2(flat[2 * v], flat[2 * v + 1]));
    for (const [a, b, c] of THREE.ShapeUtils.triangulateShape(outline, [])) {
      index.push(face[a], face[b], face[c]);
      triRegs.push(r);
    }
  });
  return {index, triangleRegions: Int32Array.from(triRegs)};
}

/** A three.js scene that shows one `Model` at any stage of folding. */
export class Stage {
  /**
   * Loads three.js asynchronously and creates a `Stage` attached to `canvas`.
   *
   * @param {!HTMLCanvasElement} canvas
   * @param {!HTMLElement} pinLayer Element laid over the canvas to hold labels.
   * @return {!Promise<!Stage>}
   */
  static async create(canvas, pinLayer) {
    const {THREE, OrbitControls, RoomEnvironment} = await loadThree();
    return new Stage(canvas, pinLayer, THREE, OrbitControls, RoomEnvironment);
  }

  /**
   * @param {!HTMLCanvasElement} canvas
   * @param {!HTMLElement} pinLayer Element laid over the canvas to hold labels.
   * @param {!Object} THREE Loaded three.js module namespace.
   * @param {!Function} OrbitControls OrbitControls constructor.
   * @param {!Function} RoomEnvironment Studio environment constructor.
   */
  constructor(canvas, pinLayer, THREE, OrbitControls, RoomEnvironment) {
    /** @private @const {!Object} */
    this._three = THREE;
    /**
     * Lays the flat frame on the ground with its front side up: x stays x, the
     * paper normal +z becomes up, and so +y points away from the default
     * camera.
     * @private @const {!Object}
     */
    this._flatOnGround = new THREE.Matrix4().set(
        1, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 1);
    /** @private @const {!Object} */
    this._up = new THREE.Vector3(0, 1, 0);

    /** @private @const {!HTMLElement} */
    this._pinLayer = pinLayer;
    /** @private @const {!Object} */
    this._scene = new THREE.Scene();
    /** @private @const {!Object} */
    this._camera = new THREE.PerspectiveCamera(FIELD_OF_VIEW, 1, 0.02, 40);
    /**
     * Everything that moves with the paper.
     * @private @const {!Object}
     */
    this._paper = new THREE.Group();
    /** @private {!Array<{vertex: number, element: !HTMLElement}>} */
    this._pins = [];
    /** @private {?Object} Parsed model from `fold.js`. */
    this._model = null;
    /** @private {?Folder} */
    this._folder = null;
    /** @private {?Object} */
    this._positions = null;
    /** @private {?Object} */
    this._uvs = null;
    /** @private @const {!Object} */
    this._flatPose = new THREE.Quaternion();
    /** @private @const {!Object} */
    this._foldedPose = new THREE.Quaternion();
    /** @private {{azimuth: number, elevation: number}} */
    this._hero = DEFAULT_HERO;
    /** @private {number} */
    this._radius = 0;
    /** @private {boolean} */
    this._xray = false;
    /** @private {boolean} */
    this._creasesVisible = false;
    /** @private {number} */
    this._progress = 1;
    /** @private {boolean} */
    this._videoMode = false;
    /**
     * Eye-space ray bias on the reverse side: shifts reverse-side fragments
     * back along the camera ray by up to 0.9% of camera distance so warped
     * interior layers do not bleed through exterior colored faces while screen
     * (x, y) stays fixed. Stopgap: this also suppresses genuine reverse-color
     * details that sit within 0.9% of camera distance of a colored face.
     * @private @const {{value: number}}
     */
    this._reverseBiasUniform = {value: 0};

    /** @private @const {!Object} */
    this._renderer =
        new THREE.WebGLRenderer({canvas, antialias: true, alpha: true});
    this._renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this._renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this._renderer.toneMappingExposure = 1.02;
    this._renderer.shadowMap.enabled = true;
    this._renderer.shadowMap.type = THREE.PCFSoftShadowMap;

    const pmrem = new THREE.PMREMGenerator(this._renderer);
    const roomEnv = new RoomEnvironment();
    this._scene.environment = pmrem.fromScene(roomEnv, 0.04).texture;
    roomEnv.dispose();
    pmrem.dispose();

    /** @private @const {!Object} */
    this._controls = new OrbitControls(this._camera, canvas);
    this._controls.enableDamping = true;
    this._controls.minDistance = 0.1;
    this._controls.maxDistance = 20;
    this._controls.autoRotateSpeed = 1.2;

    this._scene.add(new THREE.HemisphereLight(0xffffff, 0xcfc8bc, 2.2));
    const key = new THREE.DirectionalLight(0xfff8ee, 2.6);
    key.position.set(1.6, 2.7, 1.9);
    key.castShadow = true;
    key.shadow.mapSize.set(2048, 2048);
    Object.assign(
        key.shadow.camera,
        {left: -1.1, right: 1.1, top: 1.1, bottom: -1.1, near: 0.2, far: 8});
    const fill = new THREE.DirectionalLight(0xdce8f8, 0.9);
    fill.position.set(-1.9, 1.2, -1.2);
    this._scene.add(key, fill);

    const ground = new THREE.Mesh(
        new THREE.PlaneGeometry(8, 8).rotateX(-Math.PI / 2),
        new THREE.ShadowMaterial({opacity: 0.18}));
    ground.position.y = -0.002;
    ground.receiveShadow = true;
    this._scene.add(ground);

    const washiBump = buildWashiBumpTexture(THREE);
    const paperMaterial = (side, depthOffset, bumpScale) =>
        new THREE.MeshStandardMaterial({
          side,
          roughness: 0.86,
          metalness: 0.02,
          envMapIntensity: 0.3,
          bumpMap: bumpScale > 0 ? washiBump : null,
          bumpScale,
          flatShading: true,
          shadowSide: THREE.DoubleSide,
          polygonOffset: true,
          polygonOffsetFactor: depthOffset,
          polygonOffsetUnits: depthOffset,
        });
    /**
     * The side of the paper the crease pattern is drawn on (reverse).
     * @private @const {!Object}
     */
    this._front =
        new THREE.Mesh(undefined, paperMaterial(THREE.FrontSide, 3, 0.0035));
    this._front.material.onBeforeCompile = (shader) => {
      // Shifts reverse-side fragments away along the eye ray by uReverseBias
      // (up to 0.9% of camera distance). Stopgap: hides warped interior layer
      // interpenetration, at the cost of suppressing genuine reverse-color
      // details within 0.9% of camera distance of a colored face.
      shader.uniforms.uReverseBias = this._reverseBiasUniform;
      shader.vertexShader = 'uniform float uReverseBias;\n' +
          shader.vertexShader.replace(
              '#include <project_vertex>',
              `vec4 mvPosition = vec4( transformed, 1.0 );
               #ifdef USE_INSTANCING
                 mvPosition = instanceMatrix * mvPosition;
               #endif
               mvPosition = modelViewMatrix * mvPosition;
               mvPosition.xyz *= 1.0 + uReverseBias;
               gl_Position = projectionMatrix * mvPosition;`);
    };
    /** @private @const {!Object} */
    this._back =
        new THREE.Mesh(undefined, paperMaterial(THREE.BackSide, 2, 0.0035));
    this._front.castShadow = true;
    /** @private @const {!Object} */
    this._highlight =
        new THREE.Mesh(undefined, paperMaterial(THREE.DoubleSide, 1, 0));
    this._highlight.material.color.set(HIGHLIGHT_COLOR);
    this._highlight.visible = false;
    /** @private @const {!Object} */
    this._hoverHighlight = new THREE.Mesh(
        undefined,
        new THREE.MeshBasicMaterial({
          color: HOVER_HIGHLIGHT_COLOR,
          side: THREE.DoubleSide,
          transparent: true,
          opacity: 0.28,
          depthWrite: false,
          polygonOffset: true,
          polygonOffsetFactor: 0.5,
          polygonOffsetUnits: 0.5,
        }));
    this._hoverHighlight.visible = false;
    /** @private {?Int32Array} Pick group (sub-flap or region) per face. */
    this._facePickIds = null;
    /** @private {number} Number of valid pick groups. */
    this._pickGroupCount = 0;
    /** @private {?Int32Array} Region/sub-flap index per triangle in `_front`. */
    this._triangleRegions = null;
    /** @private @const {!Set<number>} Currently selected region indices. */
    this._highlightedSet = new Set();
    /** @private {?number} Region currently hovered under pointer. */
    this._hoveredRegion = null;
    /** @private @const {!Object} */
    this._raycaster = new THREE.Raycaster();
    /** @private @const {!Object} */
    this._pointerNdc = new THREE.Vector2();
    /** @private @const {!Array<function(?number)>} */
    this._regionClickListeners = [];
    /** @private @const {!Array<function(?number, ?PointerEvent)>} */
    this._regionHoverListeners = [];
    /** @private @const {!Object} */
    this._creases = new THREE.LineSegments(
        undefined,
        new THREE.LineBasicMaterial({transparent: true, opacity: 0.45}));
    this._creases.visible = false;
    /** @private @const {!Object} */
    this._activeMountains = new THREE.LineSegments(
        undefined,
        new THREE.LineBasicMaterial(
            {color: ACTIVE_MOUNTAIN_COLOR, transparent: true, opacity: 0.92}));
    /** @private @const {!Object} */
    this._activeValleys = new THREE.LineSegments(
        undefined,
        new THREE.LineBasicMaterial(
            {color: ACTIVE_VALLEY_COLOR, transparent: true, opacity: 0.92}));
    this._activeMountains.visible = false;
    this._activeValleys.visible = false;
    /** @private @const {!Object} */
    this._userMountains = new THREE.LineSegments(
        undefined,
        new THREE.LineBasicMaterial(
            {color: ACTIVE_MOUNTAIN_COLOR, transparent: true, opacity: 0.96}));
    /** @private @const {!Object} */
    this._userValleys = new THREE.LineSegments(
        undefined,
        new THREE.LineBasicMaterial(
            {color: ACTIVE_VALLEY_COLOR, transparent: true, opacity: 0.96}));
    this._userMountains.visible = false;
    this._userValleys.visible = false;

    this._paper.add(
        this._front, this._back, this._highlight, this._hoverHighlight,
        this._creases, this._activeMountains, this._activeValleys,
        this._userMountains, this._userValleys);
    for (const object of this._paper.children) {
      object.frustumCulled = false;
    }
    this._scene.add(this._paper);

    this._bindRegionPicking(canvas);

    new ResizeObserver(() => this._resize()).observe(canvas.parentElement);
    this._resize();
    this._renderer.setAnimationLoop(() => {
      if (this._videoMode) return;
      this._controls.update();
      this._renderer.render(this._scene, this._camera);
      this._placePins();
    });
  }

  /**
   * Shows a model, fully folded, from its hero view (or preserving current
   * camera and fold state when `preserveView` is true).
   *
   * @param {!Object} model Parsed model from `fold.js`.
   * @param {{azimuth: number, elevation: number}=} hero Camera angles of the
   *     'hero' view, in degrees.
   * @param {?Object=} schedule Optional multi-stage schedule.
   * @param {boolean=} preserveView Whether to keep the current camera pose and
   *     fold progress (for live interactive shaping).
   * @param {?Int32Array=} facePickIds Optional per-face sub-flap IDs for fine
   *     unmerged-stick picking and highlighting.
   * @param {number=} pickGroupCount Number of pick groups in `facePickIds`.
   * @param {?Array<boolean>=} userShaping Optional `edges_bp:userShaping` flags
   *     to highlight newly added Mountain/Valley shaping creases in 3D.
   */
  setModel(
      model, hero = DEFAULT_HERO, schedule = null, preserveView = false,
      facePickIds = null, pickGroupCount = 0, userShaping = null) {
    const THREE = this._three;
    const savedCameraPos = preserveView ? this._camera.position.clone() : null;
    const savedTarget = preserveView ? this._controls.target.clone() : null;
    const savedProgress = preserveView ? this._progress : 1;

    this._model = model;
    this._facePickIds = facePickIds;
    this._pickGroupCount = facePickIds ? pickGroupCount : model.regions.length;
    this._folder = new Folder(model);
    this._folder.setSchedule(schedule);
    if (!preserveView || !this._hero) this._hero = hero;
    this._positions = new THREE.BufferAttribute(this._folder.positions, 3);
    this._positions.setUsage(THREE.DynamicDrawUsage);
    this._uvs = new THREE.BufferAttribute(new Float32Array(model.flat), 2);

    const {index, triangleRegions} =
        triangulate(THREE, model, null, this._facePickIds);
    this._triangleRegions = triangleRegions;
    this._setGeometry(this._front, index);
    this._back.geometry = this._front.geometry;
    this._setGeometry(
        this._creases,
        model.edges.filter((_, e) => 'MVB'.includes(model.assignments[e]))
            .flat());
    this._setGeometry(this._activeMountains, []);
    this._setGeometry(this._activeValleys, []);
    const userM = [];
    const userV = [];
    if (Array.isArray(userShaping)) {
      model.edges.forEach(([u, v], e) => {
        if (!userShaping[e]) return;
        const a = model.assignments[e];
        if (a === 'M') userM.push(u, v);
        else if (a === 'V') userV.push(u, v);
      });
    }
    this._setGeometry(this._userMountains, userM);
    this._setGeometry(this._userValleys, userV);
    this._userMountains.visible = userM.length > 0;
    this._userValleys.visible = userV.length > 0;
    this._setHoveredRegion(null);
    if (!preserveView) {
      this.setHighlight(null);
    } else if (this._highlightedSet.size > 0) {
      this.setHighlight([...this._highlightedSet]);
    }

    this._flatPose.setFromRotationMatrix(this._flatOnGround);
    const pose = this._folder.pose;
    if (pose) {
      const rotation = new THREE.Matrix4().set(
          pose[0], pose[1], pose[2], 0, pose[3], pose[4], pose[5], 0, pose[6],
          pose[7], pose[8], 0, 0, 0, 0, 1);
      this._foldedPose.setFromRotationMatrix(rotation);
      const m = rotation.transpose().premultiply(this._flatOnGround).elements;
      const yaw = Math.atan2(m[2] - m[8], m[0] + m[10]);
      this._flatPose.premultiply(
          new THREE.Quaternion().setFromAxisAngle(this._up, yaw));
    } else {
      this._foldedPose.copy(this._flatPose);
    }

    if (preserveView && savedCameraPos && savedTarget) {
      this.setFold(savedProgress, Boolean(schedule));
      this._controls.target.copy(savedTarget);
      this._camera.position.copy(savedCameraPos);
      this._controls.update();
    } else {
      this._radius = 0;
      this.setFold(1, Boolean(schedule));
      this.setView('hero');
    }
  }

  /**
   * Folds the paper and returns the latest schedule evaluation state.
   *
   * @param {number} progress From 0 (flat) to 1 (folded).
   * @param {boolean} staged Whether to fold along the multi-stage schedule.
   * @return {?Object} `this._folder.lastState`.
   */
  setFold(progress, staged) {
    if (!this._folder) return null;
    const THREE = this._three;
    this._progress = progress;
    const positions = this._folder.positionsAt(progress, staged);
    this._positions.needsUpdate = true;
    this._updateReverseBias();
    this._updateActiveCreases(progress);

    // Keep the sheet flat on the table through Act I collapse and Act II
    // flat-clipping, then smoothly stand upright during 3D posing.
    const ease = this._folder.standupProgress(progress, staged);
    const pose = this._paper.quaternion.slerpQuaternions(
        this._flatPose, this._foldedPose, ease);
    const box = new THREE.Box3();
    const point = new THREE.Vector3();
    for (let i = 0; i < positions.length; i += 3) {
      box.expandByPoint(point.fromArray(positions, i).applyQuaternion(pose));
    }
    const center = box.getCenter(new THREE.Vector3());
    this._paper.position.set(-center.x, -box.min.y, -center.z);
    let radius = 0;
    for (let i = 0; i < positions.length; i += 3) {
      point.fromArray(positions, i).applyQuaternion(pose);
      radius = Math.max(radius, point.distanceTo(center));
    }

    const target = this._controls.target;
    const offset = this._camera.position.clone().sub(target);
    if (this._radius > 0 && radius > 0) {
      offset.multiplyScalar(radius / this._radius);
    }
    target.set(0, center.y - box.min.y, 0);
    this._camera.position.copy(target).add(offset);
    this._radius = radius;
    return this._folder.lastState;
  }

  /**
   * Moves the camera to a preset view.
   *
   * @param {string} name 'hero', 'side', 'front' or 'top'.
   */
  setView(name) {
    const THREE = this._three;
    const view = VIEWS[name] ?? [this._hero.azimuth, this._hero.elevation];
    const [azimuth, elevation] = view.map(THREE.MathUtils.degToRad);
    const halfAngle = THREE.MathUtils.degToRad(FIELD_OF_VIEW / 2);
    const narrowHalfAngle =
        Math.atan(Math.tan(halfAngle) * Math.min(1, this._camera.aspect));
    const distance = this._radius / Math.sin(narrowHalfAngle);
    this._camera.position
        .set(
            Math.cos(elevation) * Math.sin(azimuth), Math.sin(elevation),
            Math.cos(elevation) * Math.cos(azimuth))
        .multiplyScalar(distance)
        .add(this._controls.target);
  }

  /**
   * Sets the colors of the two sides of the paper.
   *
   * @param {string} top CSS color of the display side.
   * @param {string} bottom CSS color of the reverse, which carries the crease
   *     pattern.
   */
  setColors(top, bottom) {
    const THREE = this._three;
    this._back.material.color.set(top);
    this._front.material.color.set(bottom);
    const {h, s, l} =
        this._back.material.color.getHSL({}, THREE.SRGBColorSpace);
    this._creases.material.color.setHSL(
        h, s, l < 0.35 ? l + 0.35 : l * 0.45, THREE.SRGBColorSpace);
  }

  /** @param {boolean} visible Whether to draw creases and the paper's edge. */
  setCreasesVisible(visible) {
    this._creasesVisible = visible;
    this._creases.visible = visible;
    this._updateActiveCreases(this._progress);
  }

  /** @param {boolean} visible Alias for `setCreasesVisible`. */
  setShowCreases(visible) {
    this.setCreasesVisible(visible);
  }

  /** @param {boolean} xray Whether the paper is translucent. */
  setXray(xray) {
    this._xray = xray;
    for (const {material} of [this._front, this._back]) {
      material.transparent = xray;
      material.opacity = xray ? XRAY_OPACITY : 1;
      material.depthWrite = !xray;
      material.needsUpdate = true;
    }
    this._updateReverseBias();
  }

  /**
   * Highlights one or more regions of the model, or clears the highlight.
   *
   * @param {?number|!Array<number>|!Set<number>} region Region index/indices,
   *     or null.
   */
  setHighlight(region) {
    this._highlightedSet.clear();
    if (typeof region === 'number' && region >= 0) {
      this._highlightedSet.add(region);
    } else if (region instanceof Set || Array.isArray(region)) {
      for (const r of region) {
        if (typeof r === 'number' && r >= 0) this._highlightedSet.add(r);
      }
    }
    const visible = this._highlightedSet.size > 0 && this._model !== null;
    this._highlight.visible = visible;
    if (visible) {
      this._setGeometry(
          this._highlight,
          triangulate(
              this._three, this._model, this._highlightedSet, this._facePickIds)
              .index);
    }
    if (this._hoveredRegion !== null &&
        this._highlightedSet.has(this._hoveredRegion)) {
      this._hoverHighlight.visible = false;
    }
  }

  /**
   * Registers a callback invoked when the user clicks a flap/river region (or
   * empty space -> `null`) on the 3D canvas without orbiting.
   *
   * @param {function(?number)} listener
   */
  onRegionClick(listener) {
    this._regionClickListeners.push(listener);
  }

  /**
   * Registers a callback invoked when the pointer hovers over a region on the
   * 3D canvas.
   *
   * @param {function(?number, ?PointerEvent)} listener
   */
  onRegionHover(listener) {
    this._regionHoverListeners.push(listener);
  }

  /**
   * Sets the labels that follow vertices of the paper.
   *
   * @param {!Array<{vertex: number, label: string}>} pins
   */
  setPins(pins) {
    this._pins = pins.map(({vertex, label}) => {
      const element = document.createElement('span');
      element.className = 'pin';
      element.textContent = label;
      return {vertex, element};
    });
    this._pinLayer.replaceChildren(...this._pins.map((pin) => pin.element));
  }

  /** @param {boolean} rotate Whether the camera circles the model. */
  set autoRotate(rotate) {
    this._controls.autoRotate = rotate;
  }

  /** @param {boolean} rotate Sets camera auto-rotation. */
  setAutoRotate(rotate) {
    this._controls.autoRotate = rotate;
  }

  /** @return {{azimuth: number, elevation: number}} Hero camera angles. */
  get hero() {
    return {...this._hero};
  }

  /** @return {?Object} Latest schedule state evaluated by `Folder`. */
  get lastState() {
    return this._folder?.lastState ?? null;
  }

  /**
   * Captures camera and fold state so it can be restored after video export.
   *
   * @return {!Object}
   */
  captureState() {
    return {
      position: this._camera.position.clone(),
      target: this._controls.target.clone(),
      autoRotate: this._controls.autoRotate,
      progress: this._progress,
    };
  }

  /**
   * Computes a smooth bounding-radius envelope across the fold so video export
   * frames both the flat sheet and 360-degree 3D spin without camera pumping.
   *
   * @param {boolean} staged
   * @return {{
   *   flatRadius: number,
   *   foldedRadius: number,
   *   radiusAt: function(number): number,
   * }}
   */
  computeVideoEnvelope(staged) {
    if (!this._folder) {
      return {flatRadius: 0.75, foldedRadius: 0.5, radiusAt: () => 0.75};
    }
    const THREE = this._three;
    const savedProgress = this._progress;
    const box = new THREE.Box3();
    const point = new THREE.Vector3();
    const pose = new THREE.Quaternion();

    const measureAt = (p) => {
      const positions = this._folder.positionsAt(p, staged);
      const ease = this._folder.standupProgress(p, staged);
      pose.slerpQuaternions(this._flatPose, this._foldedPose, ease);
      box.makeEmpty();
      for (let i = 0; i < positions.length; i += 3) {
        box.expandByPoint(point.fromArray(positions, i).applyQuaternion(pose));
      }
      const center = box.getCenter(new THREE.Vector3());
      let r = 0;
      for (let i = 0; i < positions.length; i += 3) {
        point.fromArray(positions, i).applyQuaternion(pose);
        r = Math.max(r, point.distanceTo(center));
      }
      return r;
    };

    const flatRadius = Math.max(0.25, measureAt(0));
    let foldedRadius = Math.max(0.2, measureAt(1));
    for (let k = 6; k <= 15; k++) {
      foldedRadius = Math.max(foldedRadius, measureAt(k / 15));
    }
    this._folder.positionsAt(savedProgress, staged);

    return {
      flatRadius,
      foldedRadius,
      radiusAt(progress) {
        const u = Math.max(0, Math.min(1, progress));
        const s = u * u * u * (u * (u * 6 - 15) + 10);
        return (1 - s) * (flatRadius * 1.1) + s * (foldedRadius * 1.18);
      },
    };
  }

  /**
   * Switches the stage into fixed-resolution video render mode.
   *
   * @param {number} width
   * @param {number} height
   */
  beginVideoSession(width, height) {
    this._videoMode = true;
    this._controls.enabled = false;
    this._renderer.domElement.parentElement?.classList.add('recording');
    this._renderer.setPixelRatio(1);
    this._renderer.setSize(width, height, false);
    this._camera.aspect = width / height;
    this._camera.updateProjectionMatrix();
  }

  /**
   * Renders a single video frame at `(progress, azimuth, elevation, radius)`
   * and returns the canvas and schedule state.
   *
   * @param {{
   *   progress: number,
   *   staged: boolean,
   *   azimuthDeg: number,
   *   elevationDeg: number,
   *   radius: number,
   * }} params
   * @return {{canvas: !HTMLCanvasElement, state: ?Object}}
   */
  renderVideoFrame({progress, staged, azimuthDeg, elevationDeg, radius}) {
    const THREE = this._three;
    const state = this.setFold(progress, staged);
    const azimuth = THREE.MathUtils.degToRad(azimuthDeg);
    const elevation = THREE.MathUtils.degToRad(elevationDeg);
    const halfAngle = THREE.MathUtils.degToRad(FIELD_OF_VIEW / 2);
    const narrowHalfAngle =
        Math.atan(Math.tan(halfAngle) * Math.min(1, this._camera.aspect));
    const distance = Math.max(0.15, radius / Math.sin(narrowHalfAngle));
    this._camera.position
        .set(
            Math.cos(elevation) * Math.sin(azimuth),
            Math.sin(elevation),
            Math.cos(elevation) * Math.cos(azimuth))
        .multiplyScalar(distance)
        .add(this._controls.target);
    this._camera.lookAt(this._controls.target);
    this._renderer.render(this._scene, this._camera);
    this._placePins();
    return {canvas: this._renderer.domElement, state};
  }

  /**
   * Exits video render mode and restores the interactive viewport state.
   *
   * @param {!Object} savedState State captured by `captureState()`.
   */
  endVideoSession(savedState) {
    this._videoMode = false;
    this._controls.enabled = true;
    this._renderer.domElement.parentElement?.classList.remove('recording');
    this._renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this._resize();
    if (savedState) {
      this.setFold(savedState.progress, true);
      this._controls.target.copy(savedState.target);
      this._camera.position.copy(savedState.position);
      this._controls.autoRotate = savedState.autoRotate;
      this._controls.update();
    }
  }

  /** @param {function()} listener */
  onInteraction(listener) {
    this._controls.addEventListener('start', listener);
  }

  /** @param {function(!Array<number>)} listener */
  onViewChange(listener) {
    this._controls.addEventListener('change', () => {
      this._camera.updateMatrixWorld();
      const m = this._camera.matrixWorld.elements;
      listener([m[0], m[1], m[2], m[4], m[5], m[6], m[8], m[9], m[10]]);
    });
  }

  /** @private */
  _updateReverseBias() {
    if (this._xray || this._progress <= 0.05) {
      this._reverseBiasUniform.value = 0;
      return;
    }
    const u = Math.max(0, Math.min(1, (this._progress - 0.05) / 0.25));
    this._reverseBiasUniform.value = 0.009 * u * u * (3 - 2 * u);
  }

  /**
   * Highlights creases belonging to the active stage while folding.
   *
   * @private
   * @param {number} progress
   */
  _updateActiveCreases(progress) {
    const active = this._folder?.lastState?.activeEdges;
    if (!this._creasesVisible || !this._model || !active?.size ||
        progress <= 0.005 || progress >= 0.995) {
      this._activeMountains.visible = false;
      this._activeValleys.visible = false;
      return;
    }
    const mountains = [];
    const valleys = [];
    for (const e of active) {
      const [u, v] = this._model.edges[e];
      const assignment = this._model.assignments[e];
      if (assignment === 'M') mountains.push(u, v);
      else if (assignment === 'V') valleys.push(u, v);
    }
    this._activeMountains.visible = mountains.length > 0;
    this._activeValleys.visible = valleys.length > 0;
    if (mountains.length > 0) this._activeMountains.geometry.setIndex(mountains);
    if (valleys.length > 0) this._activeValleys.geometry.setIndex(valleys);
  }

  /**
   * Points an object at new index data over the shared vertex positions.
   *
   * @private
   * @param {!Object} object Mesh or line-segments object whose geometry
   *     updates.
   * @param {!Array<number>} index Vertex index list.
   */
  _setGeometry(object, index) {
    const THREE = this._three;
    object.geometry.dispose();
    const geometry = new THREE.BufferGeometry()
                         .setAttribute('position', this._positions)
                         .setIndex(index);
    if (this._uvs) geometry.setAttribute('uv', this._uvs);
    object.geometry = geometry;
  }

  /** @private */
  _resize() {
    if (this._videoMode) return;
    const {clientWidth, clientHeight} = this._renderer.domElement.parentElement;
    if (clientWidth === 0 || clientHeight === 0) return;
    this._renderer.setSize(clientWidth, clientHeight, false);
    this._camera.aspect = clientWidth / clientHeight;
    this._camera.updateProjectionMatrix();
  }

  /**
   * Binds click-vs-orbit detection and hover raycasting on the 3D canvas so
   * clicking any wing, horn, tail, or limb selects its region directly.
   *
   * @private
   * @param {!HTMLCanvasElement} canvas
   */
  _bindRegionPicking(canvas) {
    let downX = 0;
    let downY = 0;
    let downId = -1;
    let hoverRaf = 0;
    let lastMoveEvent = null;

    canvas.addEventListener('pointerdown', (e) => {
      if (e.button !== 0) return;
      downX = e.clientX;
      downY = e.clientY;
      downId = e.pointerId;
    });

    canvas.addEventListener('pointermove', (e) => {
      if (this._videoMode) return;
      if (e.buttons !== 0) {
        if (this._hoveredRegion !== null) {
          this._setHoveredRegion(null);
          for (const fn of this._regionHoverListeners) fn(null, e);
        }
        return;
      }
      lastMoveEvent = e;
      if (hoverRaf) return;
      hoverRaf = requestAnimationFrame(() => {
        hoverRaf = 0;
        if (!lastMoveEvent || this._videoMode) return;
        const region = this._pickRegionAt(lastMoveEvent.clientX, lastMoveEvent.clientY);
        canvas.style.cursor = region !== null ? 'pointer' : '';
        if (region !== this._hoveredRegion) {
          this._setHoveredRegion(region);
          for (const fn of this._regionHoverListeners) fn(region, lastMoveEvent);
        } else if (region !== null) {
          for (const fn of this._regionHoverListeners) fn(region, lastMoveEvent);
        }
      });
    });

    canvas.addEventListener('pointerleave', () => {
      lastMoveEvent = null;
      canvas.style.cursor = '';
      if (this._hoveredRegion !== null) {
        this._setHoveredRegion(null);
        for (const fn of this._regionHoverListeners) fn(null, null);
      }
    });

    canvas.addEventListener('pointerup', (e) => {
      if (this._videoMode || e.button !== 0 || e.pointerId !== downId) return;
      downId = -1;
      if (Math.hypot(e.clientX - downX, e.clientY - downY) >= 5) return;
      const region = this._pickRegionAt(e.clientX, e.clientY);
      for (const fn of this._regionClickListeners) fn(region);
    });
  }

  /**
   * Raycasts against both sides of the paper mesh at `(clientX, clientY)`.
   *
   * @private
   * @param {number} clientX
   * @param {number} clientY
   * @return {?number} Region index into `model.regions`, or null.
   */
  _pickRegionAt(clientX, clientY) {
    if (!this._model || !this._triangleRegions || !this._front.geometry) {
      return null;
    }
    const rect = this._renderer.domElement.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) return null;
    this._pointerNdc.set(
        ((clientX - rect.left) / rect.width) * 2 - 1,
        -((clientY - rect.top) / rect.height) * 2 + 1);
    this._camera.updateMatrixWorld();
    this._paper.updateMatrixWorld();
    if (!this._front.geometry.boundingSphere) {
      this._front.geometry.computeBoundingSphere();
    }
    // Recompute bounding sphere when vertices have folded dynamically.
    this._front.geometry.boundingSphere.set(
        new this._three.Vector3(0, 0, 0), 4);
    this._raycaster.setFromCamera(this._pointerNdc, this._camera);
    const prevSide = this._front.material.side;
    this._front.material.side = this._three.DoubleSide;
    const hits = this._raycaster.intersectObject(this._front, false);
    this._front.material.side = prevSide;
    if (!hits.length) return null;
    const triIdx = hits[0].faceIndex;
    if (triIdx == null || triIdx < 0 || triIdx >= this._triangleRegions.length) {
      return null;
    }
    const r = this._triangleRegions[triIdx];
    return (r >= 0 && r < this._pickGroupCount) ? r : null;
  }

  /**
   * Updates the translucent hover highlight mesh.
   *
   * @private
   * @param {?number} region
   */
  _setHoveredRegion(region) {
    this._hoveredRegion = region;
    const show = region !== null && this._model !== null &&
        !this._highlightedSet.has(region);
    this._hoverHighlight.visible = show;
    if (show) {
      this._setGeometry(
          this._hoverHighlight,
          triangulate(this._three, this._model, region, this._facePickIds)
              .index);
    }
  }

  /** @private */
  _placePins() {
    if (this._pinLayer.hidden || !this._folder) return;
    const THREE = this._three;
    const {clientWidth, clientHeight} = this._renderer.domElement;
    const point = new THREE.Vector3();
    for (const {vertex, element} of this._pins) {
      point.fromArray(this._folder.positions, 3 * vertex)
          .applyMatrix4(this._paper.matrixWorld)
          .project(this._camera);
      const x = (point.x + 1) / 2 * clientWidth;
      const y = (1 - point.y) / 2 * clientHeight;
      element.style.transform = `translate(${x}px, ${y}px)`;
      element.hidden = point.z > 1;
    }
  }
}
