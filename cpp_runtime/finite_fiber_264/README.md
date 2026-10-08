# DM3D finite 264-cubed hidden-fiber operator

A verified discrete 3D group substrate for Jarvis-X. It separates **constant visible collapse**, **invertible hidden evolution**, **nonconstant observation**, and **continuous inward contraction**.

## Exact state algebra

Per-axis group: Λ₀ = Z₁₁ × Z₆ × Z₄ (264 elements).

Hidden group: X = Λ₀³ = (Z₁₁ × Z₆ × Z₄)³ (18,399,744 elements).

The nine digits have modular radices (11,6,4, 11,6,4, 11,6,4). A state can be indexed in a **264 × 264 × 264 virtual array**; this indexing is a bijection of sets, **not a group isomorphism** to (Z₂₆₄)³.

Define C:X→X by C(x)=v*=(0,...,0). Thus:

- |im C| = 1.
- |Fix C| = 1 (because v* belongs to X and C is a self-map).
- |C⁻¹(v*)| = 18,399,744.
- C(x+g)=C(x), so every group translation is invisible to C.
- Ψ(r,t)=v* is spatially and temporally constant; its derivatives are zero.

For a fixed group translation g, the cycle length is ord(g), which divides lcm(11,6,4)=132. Arbitrary permutations can have a cycle of 18,399,744 states, but such cycles **are not translations on X**. A separate one-cycle permutation on virtual indices demonstrates that distinction.

## Observable and inward channels

R(x) exposes nine normalized modular digits and is injective. A deterministic 512D feature adapter retains those digits in its first nine components. It is **not a learned encoder or compression model**.

The anchored contraction for a fixed observed target is:

    z[k+1] = lambda*z[k] + (1-lambda)*embed512(x)
    0 <= lambda < 1

It has Lipschitz constant lambda, converges toward the fixed target and measures convergence with the full-vector maximum step delta and MSE. This convergence does **not** imply the finite group translation converges.

The module exposes a CTR invariant check for translation invertibility, array index roundtrip, constant visible output, and optional exhaustive enumeration of all 18,399,744 states. No full hidden-state array is allocated.

## Build and test

    cmake -S cpp_runtime/finite_fiber_264 -B build/fiber264 -DCMAKE_BUILD_TYPE=Release
    cmake --build build/fiber264 --config Release --parallel 2
    ctest --test-dir build/fiber264 -C Release --output-on-failure

CLI examples on Linux/macOS:

    ./build/fiber264/dm3d-fiber264 --steps 16 --lambda 0.73
    ./build/fiber264/dm3d-fiber264 --steps 3 --exhaustive

On Windows, use the executable under build/fiber264/Release.

Unit tests check group arithmetic, inverse translations, exact orbit period 132, virtual coordinate roundtrip, the independent one-cycle index permutation, observable recovery, and 512D contraction. The third CTest case exhaustively verifies index conversion and constant collapse across the 18.4-million-state domain.

## Integration boundary

Include the header at cpp_runtime/finite_fiber_264/include/dm3d/fiber264.hpp from the existing SSA/compiler or scale-out runtime. Embed its observable 512D features into a learned model if needed. The constant collapse output contains zero information about the hidden state: it cannot reconstruct the hidden state or perform intelligence. This module is a finite algebraic engine plus a deterministic contraction test, **not** a trained 3D ANN or an allocation of 264-cubed neural weights.
