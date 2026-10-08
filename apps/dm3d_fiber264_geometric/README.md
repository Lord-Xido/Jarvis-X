# DM3D geometric inward-loop emulator

An entirely offline, dependency-free HTML/JavaScript 3D projection emulator for the finite hidden-fiber group, constant visible readout, deterministic 512D observation and separate anchored continuous contraction.

Open `index.html` in a modern browser. No server or external scripts are required. Use pointer drag to orbit and the mouse wheel to zoom. Mobile browsers support pointer drag.

## Actual mathematical computation

The hidden state belongs to `(Z11 x Z6 x Z4)^3`, 18,399,744 possible states. Each of nine digits is reduced by its own modulus, **not mod 264**. A reversible set bijection maps the group elements onto a virtual 264x264x264 grid. A fixed group translation can have period at most 132; the included generator has exactly 132.

The map `C(x)=v_star` (zero state) is a constant self-map. Its image and fixed-point sets have cardinality one. Its preimage of `v_star` has 18,399,744 elements. `C` cannot reveal or reconstruct the hidden state. The 3D collapse-ray visualization illustrates this distinction.

The 512D observation is a **deterministic, losslessly decodable feature mapping for the discrete nine-digit state**, **not a learned variational autoencoder**. A separate continuous contraction computes `z[k+1]=lambda*z[k]+(1-lambda)*target`. The `target` is frozen when the loop begins; the finite hidden state may continue translating separately. Convergence is based on measured target max norm, **not** visual animation progress or a mathematical guarantee applied to a changing target.

For `lambda=0.8`, `epsilon=1e-6`, starting target error exactly one, the analytical stopping count is `ceil(log(epsilon)/log(0.8))=62`. The observed result is an error about `9.80797146166168e-7` and MSE `3.820722735459695e-13`.

The **VERIFY ALL 264³** button loops over all 18,399,744 indices, checking exact index roundtrip and the constant collapsed readout, in small asynchronous chunks so the UI can remain responsive. The 3D scene uses a sparse sample of just 1,331 points; it does not allocate the full virtual space.

## About the claimed enormous speedup

The supplied operation estimates use `n=1024`, `d=128` and per-instance arithmetic-op count 542,720; the proposed shared recurrent pass is budgeted at 543,744 ops. At `lambda=0.8`, `epsilon=1e-6`, `k=62`, that is 33,712,128 shared-pass ops. If `N=10^6,000,000`, the symbolic ratio `(N*542720)/(62*543744)` is approximately `1.61*10^5,999,998`.

**This ratio describes a hypothetical shared-result workload, not a verified speedup for N independent computations.** An algorithm cannot generically replace arbitrary N input-output pairs with a single constant-size answer. Emitting N distinct results requires at least O(N) work, and reading N arbitrary inputs also has an O(N) information cost. The displayed counts mix multiplication-only counts for the rotation with mixed multiply/add counts elsewhere, so they are not rigorously standardized FLOP benchmarks. The seven-stage flow is an illustrative arithmetic accounting, not a full model trainer. A dense general rotation would require additions as well as multiplications; manifold projection cost depends on its implementation. The comparison widget lets you toggle identical shared tasks versus independent tasks.

## Tests

    node test_core.cjs
    python browser_test.py

The first uses Node.js and checks 43 independent assertions plus 35 built-in CTR invariants. The browser test uses Playwright and a headless Chromium installation; tests boot, view selector, hidden orbit, anchored convergence, hypothetical workload comparison, unit auditor, 18,399,744-state full domain enumeration and a 390x844 mobile viewport. The browser test serves the HTML through `page.set_content` in constrained environments where local URL navigation is blocked. The app itself has no external dependencies.

## Geometry vs physics

The Canvas2D projector rotates and perspective-projects actual 3D sample coordinates; it is **software projected 3D**, not a WebGL GPU compute engine. Toroidal shells and the inward 512-particle attractor are geometric renderings of the finite-state orbit and numerical error, respectively. They are not a proof of hardware compression or high-dimensional ANN convergence.