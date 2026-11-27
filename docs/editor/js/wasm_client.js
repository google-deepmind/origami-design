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
 * @fileoverview Client RPC bridge to the local Pyodide Web Worker (`py_worker.js`).
 *
 * Lazily boots Python WebAssembly only when needed (dropping non-precomputed
 * files or running packing/solving/shaping tools) and supports instant task
 * cancellation via `worker.terminate()` without requiring COOP/COEP headers.
 */

export class WasmPipelineClient {
  /**
   * @param {function({busy: boolean, ready: boolean, label: string})=} onStatus
   */
  constructor(onStatus = () => {}) {
    /** @private {?Worker} */
    this._worker = null;
    /** @private {number} */
    this._nextId = 1;
    /** @private @const {!Map<number, {resolve: !Function, reject: !Function}>} */
    this._pending = new Map();
    /** @private {boolean} */
    this._ready = false;
    /** @private {?Object} */
    this._initData = null;
    /** @private @const {function({busy: boolean, ready: boolean, label: string})} */
    this._onStatus = onStatus;
  }

  /** @return {boolean} Whether Pyodide has booted in the current worker. */
  get isReady() {
    return this._ready;
  }

  /** @return {boolean} Whether a WASM task is currently running. */
  get isBusy() {
    return this._pending.size > 0;
  }

  /** @return {?Object} Cached `{examples, tools, allToolNames}` metadata. */
  get initData() {
    return this._initData;
  }

  /**
   * @return {!Worker}
   * @private
   */
  _ensureWorker() {
    if (this._worker) return this._worker;
    const workerUrl = new URL('./py_worker.js', import.meta.url);
    const worker = new Worker(workerUrl);
    worker.addEventListener('message', (event) => {
      const msg = event.data || {};
      if (msg.type === 'progress') {
        if (msg.stage === 'Python WASM engine ready') this._ready = true;
        this._onStatus({
          busy: this._pending.size > 0,
          ready: this._ready,
          label: msg.stage,
        });
        return;
      }
      const entry = this._pending.get(msg.id);
      if (!entry) return;
      this._pending.delete(msg.id);
      if (msg.type === 'result') {
        this._ready = true;
        this._onStatus({
          busy: this._pending.size > 0,
          ready: true,
          label: this._pending.size > 0 ? 'Working…' : 'Local WASM ready',
        });
        entry.resolve(msg.result);
      } else {
        this._onStatus({
          busy: this._pending.size > 0,
          ready: this._ready,
          label: this._ready ? 'Local WASM ready' : 'WASM load error',
        });
        const err = new Error(msg.error || 'Python WASM error');
        err.detail = msg.detail;
        entry.reject(err);
      }
    });
    worker.addEventListener('error', (event) => {
      const err = new Error(event.message || 'Web Worker error');
      for (const {reject} of this._pending.values()) reject(err);
      this._pending.clear();
      this._worker = null;
      this._ready = false;
      this._onStatus({busy: false, ready: false, label: 'Worker error'});
    });
    this._worker = worker;
    return worker;
  }

  /**
   * Executes `action` inside the Pyodide worker.
   *
   * @param {string} action
   * @param {!Object=} payload
   * @return {!Promise<!Object>}
   */
  call(action, payload = {}) {
    const worker = this._ensureWorker();
    const id = this._nextId++;
    this._onStatus({
      busy: true,
      ready: this._ready,
      label: this._ready ? `Running ${action.replaceAll('_', ' ')}…` :
                           'Starting Python WASM…',
    });
    return new Promise((resolve, reject) => {
      this._pending.set(id, {resolve, reject});
      worker.postMessage({id, action, payload});
    });
  }

  /**
   * Boots Pyodide (if not already booted) and caches built-in example trees
   * and shaping tool schemas.
   *
   * @return {!Promise<!Object>}
   */
  async ensureInitialized() {
    if (this._initData && this._ready) return this._initData;
    this._initData = await this.call('init');
    return this._initData;
  }

  /**
   * Immediately terminates any running solver/packer call in the worker.
   */
  cancel() {
    if (!this._worker) return;
    this._worker.terminate();
    this._worker = null;
    this._ready = false;
    const err = new Error('Operation cancelled.');
    err.cancelled = true;
    for (const {reject} of this._pending.values()) reject(err);
    this._pending.clear();
    this._onStatus({busy: false, ready: false, label: 'Cancelled'});
  }
}
