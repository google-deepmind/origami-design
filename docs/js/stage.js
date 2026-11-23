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

import {Folder} from './fold.js';

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
 * Triangle vertex indices for the faces of a model, or of one of its regions.
 *
 * @param {!Object} THREE Loaded three.js module namespace.
 * @param {!Object} model Parsed model from `fold.js`.
 * @param {?number=} region Index into `model.regions`, or null for all faces.
 * @return {!Array<number>}
 */
function triangulate(THREE, model, region = null) {
  const {flat, faces, faceRegions} = model;
  const index = [];
  faces.forEach((face, f) => {
    if (region !== null && faceRegions[f] !== region) return;
    // Faces can be concave, so fan triangulation is not enough.
    const outline =
        face.map((v) => new THREE.Vector2(flat[2 * v], flat[2 * v + 1]));
    for (const [a, b, c] of THREE.ShapeUtils.triangulateShape(outline, [])) {
      index.push(face[a], face[b], face[c]);
    }
  });
  return index;
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
    /**
     * Vertex positions shared by all paper geometry; owned by the folder.
     * @private {?Object}
     */
    this._positions = null;
    /**
     * Flat-sheet UV coordinates shared by paper meshes so fiber grain folds
     * physically with each facet.
     * @private {?Object}
     */
    this._uvs = null;
    /** @private @const {!Object} */
    this._flatPose = new THREE.Quaternion();
    /** @private @const {!Object} */
    this._foldedPose = new THREE.Quaternion();
    /** @private {{azimuth: number, elevation: number}} */
    this._hero = DEFAULT_HERO;
    /**
     * Bounding radius the camera distance was last matched to.
     * @private {number}
     */
    this._radius = 0;

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
    // The flat sheet has a half-diagonal of 0.71.
    Object.assign(
        key.shadow.camera,
        {left: -1.1, right: 1.1, top: 1.1, bottom: -1.1, near: 0.2, far: 8});
    const fill = new THREE.DirectionalLight(0xdce8f8, 0.9);
    fill.position.set(-1.9, 1.2, -1.2);
    this._scene.add(key, fill);

    // Shows only the paper's shadow. It sits a hair below the paper so that
    // the unfolded sheet does not coincide with it.
    const ground = new THREE.Mesh(
        new THREE.PlaneGeometry(8, 8).rotateX(-Math.PI / 2),
        new THREE.ShadowMaterial({opacity: 0.18}));
    ground.position.y = -0.002;
    ground.receiveShadow = true;
    this._scene.add(ground);

    // The two sides of the paper are separate meshes over the same geometry,
    // so each can have its own color without a custom shader. Flat shading
    // derives normals per pixel, so the geometry needs none.
    //
    // Folded layers lie exactly on top of each other and the file does not
    // say in which order, so polygon offsets decide what shows where surfaces
    // coincide: crease lines (no offset) over the highlight, over the display
    // side of the paper, over its reverse.
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
     * The side of the paper the crease pattern is drawn on.
     * @private @const {!Object}
     */
    this._front =
        new THREE.Mesh(undefined, paperMaterial(THREE.FrontSide, 3, 0.0035));
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
    this._creases = new THREE.LineSegments(
        undefined,
        new THREE.LineBasicMaterial({transparent: true, opacity: 0.45}));
    this._paper.add(this._front, this._back, this._highlight, this._creases);
    for (const object of this._paper.children) {
      // Bounding spheres would go stale as the paper folds.
      object.frustumCulled = false;
    }
    this._scene.add(this._paper);

    new ResizeObserver(() => this._resize()).observe(canvas.parentElement);
    this._resize();
    this._renderer.setAnimationLoop(() => {
      this._controls.update();
      this._renderer.render(this._scene, this._camera);
      this._placePins();
    });
  }

  /**
   * Shows a model, fully folded, from its hero view.
   *
   * @param {!Object} model Parsed model from `fold.js`.
   * @param {{azimuth: number, elevation: number}=} hero Camera angles of the
   *     'hero' view, in degrees.
   */
  setModel(model, hero = DEFAULT_HERO) {
    const THREE = this._three;
    this._model = model;
    this._folder = new Folder(model);
    this._hero = hero;
    this._positions = new THREE.BufferAttribute(this._folder.positions, 3);
    this._positions.setUsage(THREE.DynamicDrawUsage);
    this._uvs = new THREE.BufferAttribute(new Float32Array(model.flat), 2);

    this._setGeometry(this._front, triangulate(THREE, model));
    this._back.geometry = this._front.geometry;
    this._setGeometry(
        this._creases,
        model.edges.filter((_, e) => 'MVB'.includes(model.assignments[e]))
            .flat());
    this.setHighlight(null);

    this._flatPose.setFromRotationMatrix(this._flatOnGround);
    const pose = this._folder.pose;
    if (pose) {
      const rotation = new THREE.Matrix4().set(
          pose[0], pose[1], pose[2], 0, pose[3], pose[4], pose[5], 0, pose[6],
          pose[7], pose[8], 0, 0, 0, 0, 1);
      this._foldedPose.setFromRotationMatrix(rotation);
      // The flat sheet may face any way round on the ground. Turn it about
      // the vertical to the yaw nearest the folded pose, which maximizes the
      // trace of yaw * FLAT_ON_GROUND * rotation^T, so that unfolding does not
      // spin the model needlessly.
      const m = rotation.transpose().premultiply(this._flatOnGround).elements;
      const yaw = Math.atan2(m[2] - m[8], m[0] + m[10]);
      this._flatPose.premultiply(
          new THREE.Quaternion().setFromAxisAngle(this._up, yaw));
    } else {
      // Without a folded frame there is no upright pose to turn to.
      this._foldedPose.copy(this._flatPose);
    }

    this._radius = 0;
    this.setFold(1, false);
    this.setView('hero');
  }

  /**
   * Folds the paper; see `Folder.positionsAt`.
   *
   * @param {number} progress From 0 (flat) to 1 (folded).
   * @param {boolean} staged Whether to fold the base before the shaping.
   */
  setFold(progress, staged) {
    const THREE = this._three;
    const positions = this._folder.positionsAt(progress, staged);
    this._positions.needsUpdate = true;

    // Turn the sheet from lying flat to the folded pose as it folds.
    const ease = progress * progress * (3 - 2 * progress);
    const pose = this._paper.quaternion.slerpQuaternions(
        this._flatPose, this._foldedPose, ease);
    const box = new THREE.Box3();
    const point = new THREE.Vector3();
    for (let i = 0; i < positions.length; i += 3) {
      box.expandByPoint(point.fromArray(positions, i).applyQuaternion(pose));
    }
    // Stand the paper on the ground, centered on the vertical axis.
    const center = box.getCenter(new THREE.Vector3());
    this._paper.position.set(-center.x, -box.min.y, -center.z);
    let radius = 0;
    for (let i = 0; i < positions.length; i += 3) {
      point.fromArray(positions, i).applyQuaternion(pose);
      radius = Math.max(radius, point.distanceTo(center));
    }

    // Follow the paper with the camera, keeping its apparent size: the flat
    // sheet is much larger than the folded model.
    const target = this._controls.target;
    const offset = this._camera.position.clone().sub(target);
    if (this._radius > 0) offset.multiplyScalar(radius / this._radius);
    target.set(0, center.y - box.min.y, 0);
    this._camera.position.copy(target).add(offset);
    this._radius = radius;
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
    // The distance at which the model's bounding sphere just fits the
    // narrower of the two fields of view.
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
   * Crease patterns are drawn on the reverse of the paper, the side that
   * mostly ends up inside the model.
   *
   * @param {string} top CSS color of the display side.
   * @param {string} bottom CSS color of the reverse, which carries the crease
   *     pattern.
   */
  setColors(top, bottom) {
    const THREE = this._three;
    this._back.material.color.set(top);
    this._front.material.color.set(bottom);
    // Crease lines contrast with the display side, whatever its lightness.
    const {h, s, l} =
        this._back.material.color.getHSL({}, THREE.SRGBColorSpace);
    this._creases.material.color.setHSL(
        h, s, l < 0.35 ? l + 0.35 : l * 0.45, THREE.SRGBColorSpace);
  }

  /** @param {boolean} visible Whether to draw creases and the paper's edge. */
  setCreasesVisible(visible) {
    this._creases.visible = visible;
  }

  /** @param {boolean} xray Whether the paper is translucent. */
  setXray(xray) {
    for (const {material} of [this._front, this._back]) {
      material.transparent = xray;
      material.opacity = xray ? XRAY_OPACITY : 1;
      material.depthWrite = !xray;
      material.needsUpdate = true;
    }
  }

  /** @param {?number} region Region of the model to pick out, or null. */
  setHighlight(region) {
    this._highlight.visible = region !== null;
    if (region !== null) {
      this._setGeometry(
          this._highlight, triangulate(this._three, this._model, region));
    }
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

  /**
   * Registers a listener for when the user starts moving the camera.
   *
   * @param {function()} listener
   */
  onInteraction(listener) {
    this._controls.addEventListener('start', listener);
  }

  /**
   * Registers a listener for camera movement of any origin.
   *
   * @param {function(!Array<number>)} listener Receives the row-major 3x3
   *     rotation from world to camera axes (x right, y up, z towards the
   *     viewer).
   */
  onViewChange(listener) {
    this._controls.addEventListener('change', () => {
      // The camera's axes in the world are the rows of the rotation from
      // world to camera, and the columns of its (column-major) world matrix.
      this._camera.updateMatrixWorld();
      const m = this._camera.matrixWorld.elements;
      listener([m[0], m[1], m[2], m[4], m[5], m[6], m[8], m[9], m[10]]);
    });
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

  /**
   * Matches renderer and camera aspect ratio to the canvas container.
   *
   * @private
   */
  _resize() {
    const {clientWidth, clientHeight} = this._renderer.domElement.parentElement;
    if (clientWidth === 0 || clientHeight === 0) return;
    this._renderer.setSize(clientWidth, clientHeight, false);
    this._camera.aspect = clientWidth / clientHeight;
    this._camera.updateProjectionMatrix();
  }

  /**
   * Moves each pin over the screen position of its vertex.
   *
   * @private
   */
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
      element.hidden = point.z > 1;  // Behind the camera.
    }
  }
}
