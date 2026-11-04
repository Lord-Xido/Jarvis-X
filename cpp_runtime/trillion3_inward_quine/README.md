# JARVIS-X Trillion³ Inward-Loop Quine (C++20)

Standalone reference engine demonstrating (1) exact source-code reproduction in `--quine` mode, (2) an enormous *virtual* cube domain `[0, 10^12]^3`, (3) a nonlinear, lossy coordinate-to-feature map, (4) deterministic hash-based routing to 1,000,000 virtual expert identifiers, and (5) mathematically bounded inward recurrent evolution.

**This is not a trained ANN, lossless neural codec, billion-node physical allocation, or multimedia generator.** No trainable weights are present. The physical memory occupied by the demonstration is small and does not scale with the virtual coordinate domain.

## Compile

```bash
g++ -std=c++20 -O2 -Wall -Wextra -Wpedantic -Werror \
  jarvisx_trillion3_inward_quine.cpp -o jx_inward
```

## Execute and validate

```bash
./jx_inward                 # 12 inward iterations
./jx_inward --run 30        # custom iteration count
./jx_inward --selftest      # 10,000 deterministic encoder/decoder tests
./jx_inward --quine > regenerated.cpp
cmp jarvisx_trillion3_inward_quine.cpp regenerated.cpp
```

`--quine` does *not* open a source file at runtime. It reconstructs its own complete source solely from an embedded, escaped source template. The resulting bytes were compared with the compiled file using `cmp`.

## Mathematical definitions

Given input `p = (x,y,z)` in cube `[0,T]^3` with `T=10^12`, feature encoding is

```text
lx = fract(sin((x/T)*12.9898 + (y/T)*78.233) * 43758.5453)
ly = fract(cos((y/T)*93.9898 + (z/T)*67.345) * 127.1)
lz = fract(sin((z/T)*53.739 + (x/T)*19.237) * 269.5)
```

`decode()` applies a bounded, heuristic projection of these three features back to cube coordinates. It is **not** an exact inverse. The correctly normalized reconstruction error is `||p-decode(encode(p))||_2/T` (one division by `T`, not two).

For inward propagation, let `c=(T/2,T/2,T/2)`, `r_t = ||p_t-c||_2`, `d_t = decoder(encoder(p_t),t) - c`, and `rho=0.85`. For `r_t > 0` and `||d_t|| > 0`:

```text
p_(t+1) = PROJECT_CUBE(c + rho * r_t * d_t / ||d_t||)
```

Otherwise the next state is the center. Since Euclidean projection onto a convex set containing `c` is non-expansive relative to `c`:

```text
||p_(t+1)-c|| <= rho * ||p_t-c||
||p_n-c|| <= rho^n * ||p_0-c||
```

This guarantees center-distance reduction by construction. It does NOT guarantee lower reconstruction loss, learning, or useful media generation. `energy=sqrt((x/T)^2+(y/T)^2+(z/T)^2)` measures distance from the origin, not the center, and need not decrease.

## Exact measured reference (GCC 14.2.0, x86-64)

- Source bytes: `16571`.
- Exact source-file reproduction: PASS.
- 10,000 numeric encoder/decoder tests: PASS.
- Initial normalized center radius: `0.0489278838171`.
- Radius after 12 steps: `0.0069595881671`.
- Average normalized reconstruction error: `0.379680089134`.
- Only **one active sample** is updated per step. A virtual `10^36` coordinate domain is not `10^36` stored parameters.

Because `sin` / `cos` use platform floating-point math, exact numerical results can differ across C++ runtime libraries or compilation targets. The quine source-byte check is independent of those math results.
