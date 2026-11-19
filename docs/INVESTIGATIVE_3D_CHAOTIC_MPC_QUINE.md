# Jarvis-X 3D Investigative Planner + Exact C++17 Quine

**Classification:** deterministic research simulation. This is a sampled model-predictive controller with fixed, hand-initialized recurrent weights. It is **not a trained neural network**, a learned probability model, real-world robot controller, or a bit-volume autoencoder.

## User-supplied design and corrections

The contributed design combines Euler-integrated Lorenz exploration, logistic modulation, feature-wise attention-like gating, a GRU, 3D dynamics and candidate trajectory search. The following correctness issues were addressed:

1. The original nested `R"QUINE(...R"QUINE(%s)QUINE"...)QUINE"` raw string closes at its inner delimiter, so it does not represent a valid self-embedded C++ string. The revised, conventional quine defers emitting the `"` tokens with two `%c` replacements and inserts its own body with a single `%s` replacement. The default executable output exactly matches its C++ source, byte-for-byte.
2. The original observation dimension D=8 omitted the Z-axis target displacement. D=9 includes position XYZ, velocity XYZ and target displacement XYZ.
3. The original planner advanced a shared chaotic generator across candidate rollouts, making candidate scores depend on candidate ordering. Each rollout now receives the *same copied* chaotic starting state; actual execution advances the live chaos once.
4. The original planner committed the hidden state at the **end of the best simulated horizon** before the real controller executed the **first** action. The new planner is const; only `apply_first_action` commits one hidden and chaotic state update.
5. A zero-action baseline is included among 256 sampled paths; the best sampled objective therefore cannot be larger than that baseline, in exact arithmetic.
6. Action coordinates are bounded to [-1,1], 3D finite-valued input is validated, and CLI steps are limited to 1..50.

## Mathematical operation

State observation:

    u_t = (p_t + eta_t, v_t, g - p_t) in R^9.

The feature-wise gate (not token self-attention) computes per-feature q,k,v projections and softmax(q_i*k_i/sqrt(9)), then adds a residual. The GRU updates hidden state `h_t` from the gated observation.

The simulated velocity increment uses fixed model weights:

    v_{t+1} = f_theta(h_{t+1}, a_t)
    p_{t+1} = p_t + v_{t+1}

For candidate action sequence A^(k) over 12 steps:

    J(A^(k)) = sum_{tau=0}^{11} 0.95^tau * [NLL_N(p_{t+tau+1}; target, 0.25 I) + 0.02 ||a_{t+tau}||^2].
    a_t* = first_action(argmin_{k=0..255} J(A^(k)))

The target-NLL is a **specified Gaussian surrogate cost**, not an empirically learned generative density. The search finds the best *sampled* sequence among 256 candidates; it does not solve a global argmin over all action sequences. The weights are initialized deterministically but **not trained** in this version. Model success on the synthetic target does not imply transfer to physical dynamics.

## Commands

When run without arguments, the executable prints **its entire own source**.

    ./jarvisx-investigative3d > reproduced.cpp
    cmp cpp_runtime/src/investigative3d_quine_main.cpp reproduced.cpp
    ./jarvisx-investigative3d --self-test
    ./jarvisx-investigative3d --run
    ./jarvisx-investigative3d --steps 10
    ./jarvisx-investigative3d --help

## CMake integration

    cmake -S cpp_runtime -B build/cpp-runtime -DJARVISX_BUILD_GL_VISUALIZER=OFF
    cmake --build build/cpp-runtime --target jarvisx-investigative3d --parallel
    ctest --test-dir build/cpp-runtime -R investigative3d --output-on-failure

A separate CTest script reproduces the complete source, hashes the original and reproduced bytes using SHA-256 and fails on any mismatch. This is a **source-equivalence** test, not a cryptographic identity/authentication system or a claim that the code learns its own implementation.

## Local validation

- GCC strict C++17: `-O2 -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror`: PASS.
- Clang strict C++17 with AddressSanitizer and UndefinedBehaviorSanitizer: PASS.
- Byte-for-byte source reproduction: PASS (33,702-byte source at time of local test).
- Built-in deterministic state/baseline/Z-observation/invalid input assertions: PASS.
- Synthetic target `(2.5, -1.5, 3.0)` from initial position `(0,0,0)`: first-step planned objective `150.980104337`, baseline zero-action objective `329.551017848`; after 10 replanning steps, remaining target distance approximately `0.3359806313` model spatial units.
- 3 isolated local CTests: PASS, 0 failures.

All numbers refer to one deterministic simulation environment, **not** experimental hardware, generalization, learned intelligence, globally optimal control or end-to-end system safety.

## Relationship to prior Jarvis-X 3D projects

This engine is *independent* of the logical `2^96`-voxel ANN bit codec and of the 3-bit two-plane VNAND reference. It can later be evaluated as an outer planner for bounded sparse tile operations, but integrating its actions with real encoding/decoding, losses and a learned model would be separate implementation and testing work.
