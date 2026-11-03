// Independent browser/Node verifier for the streamed 4D -> 3D torus geometry.
// This is a numerical consistency check, NOT an authentication mechanism.
'use strict';
globalThis.JarvisXFrameVerifier = (() => {
  const PI2 = 2 * Math.PI;
  const finite = (n) => typeof n === 'number' && Number.isFinite(n);

  function expectedVertex(frame, n) {
    const side = frame.grid;
    const i = Math.floor(n / side), j = n % side;
    const u = PI2 * i / side, v = PI2 * j / side;
    let x = frame.radii[0] * Math.cos(u);
    let y = frame.radii[0] * Math.sin(u);
    let z = frame.radii[1] * Math.cos(v);
    let w = frame.radii[1] * Math.sin(v);
    // Must match the C++ engine's XW -> YW -> ZW plane rotation sequence.
    const [ax, ay, az] = frame.angles;
    let tmp = x * Math.cos(ax) - w * Math.sin(ax);
    w = x * Math.sin(ax) + w * Math.cos(ax); x = tmp;
    tmp = y * Math.cos(ay) - w * Math.sin(ay);
    w = y * Math.sin(ay) + w * Math.cos(ay); y = tmp;
    tmp = z * Math.cos(az) - w * Math.sin(az);
    w = z * Math.sin(az) + w * Math.cos(az); z = tmp;
    const scale = frame.projection_radius / (frame.projection_radius - w);
    return [scale * x, scale * y, scale * z];
  }

  function verifyFrame(frame, sampleCount = 32) {
    const fail = (error) => ({ ok:false, error, checked:0, maxError:Infinity });
    if (!frame || frame.protocol !== 'jx-clifford-frame-v1' ||
        !Number.isInteger(frame.frame) || frame.frame < 0 ||
        !Number.isInteger(frame.grid) || frame.grid < 2 || frame.grid > 64 ||
        !Array.isArray(frame.radii) || frame.radii.length !== 2 ||
        !frame.radii.every((r) => finite(r) && r > 0) ||
        !Array.isArray(frame.angles) || frame.angles.length !== 3 ||
        !frame.angles.every(finite) ||
        !finite(frame.projection_radius) ||
        frame.projection_radius <= Math.hypot(...frame.radii) ||
        !finite(frame.dt) || frame.dt <= 0 ||
        !finite(frame.time_scale) || frame.time_scale <= 0 ||
        !finite(frame.simulated_s) || frame.simulated_s < 0 ||
        !finite(frame.compute_seconds) || frame.compute_seconds <= 0 ||
        !finite(frame.measured_vertices_per_s) || frame.measured_vertices_per_s <= 0 ||
        !finite(frame.max_norm2_error) || frame.max_norm2_error < 0 ||
        !finite(frame.orthogonality_error) || frame.orthogonality_error < 0 ||
        !finite(frame.min_projection_denominator) || frame.min_projection_denominator <= 0 ||
        !Array.isArray(frame.vertices) || frame.vertices.length !== 3 * frame.grid ** 2)
      return fail('Malformed or unsafe frame metadata');

    if (!frame.vertices.every(finite)) return fail('Nonfinite projected coordinate');
    if (frame.max_norm2_error > 1e-8 || frame.orthogonality_error > 1e-9)
      return fail('4D invariant error exceeds tolerance');
    const count = Math.min(frame.grid ** 2, Math.max(2, Math.floor(sampleCount)));
    let maxError = 0;
    for (let k = 0; k < count; k++) {
      const n = Math.floor(k * (frame.grid ** 2 - 1) / (count - 1));
      const expected = expectedVertex(frame, n);
      for (let d = 0; d < 3; d++) {
        const actual = frame.vertices[n * 3 + d];
        const error = Math.abs(actual - expected[d]);
        maxError = Math.max(maxError, error);
        if (!(error <= 5e-8 * Math.max(1, Math.abs(expected[d])))) {
          return { ok:false, error:'Projection mismatch at vertex ' + n,
            checked:k + 1, maxError };
        }
      }
    }
    return { ok:true, error:null, checked:count, maxError };
  }
  return Object.freeze({ verifyFrame, expectedVertex });
})();
