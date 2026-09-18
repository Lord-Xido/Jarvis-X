(function (root) {
  'use strict';
  if (!root.AE3D || !root.AE3D.RecursiveByteANN3D) throw new Error('engine.js must be loaded before sdk.js');

  class AE3DSDK {
    constructor(options = {}) {
      this.engine = new root.AE3D.RecursiveByteANN3D(options);
    }
    run(input, options) { return this.engine.run(input, options); }
    step(input) { return this.engine.step(input); }
    train(input, options) { return this.engine.train(input, options); }
    configure(patch) { return this.engine.setConfig(patch); }
    snapshot() { return this.engine.snapshot(); }
    exportModel() { return this.engine.exportModel(); }
    importModel(model) { this.engine.importModel(model); return this; }
    encodeText(text) { return new TextEncoder().encode(text); }
    decodeText(bytes) { return new TextDecoder().decode(bytes); }
    hex(bytes) { return Array.from(bytes, b => b.toString(16).padStart(2, '0').toUpperCase()).join(' '); }
  }

  root.AE3D.AE3DSDK = AE3DSDK;
})(typeof globalThis !== 'undefined' ? globalThis : this);
