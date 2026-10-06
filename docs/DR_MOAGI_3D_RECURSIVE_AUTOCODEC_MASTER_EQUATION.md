# Dr Moagi 3D Recursive Auto-Codec Master Equation

## Canonical operator

The canonical total-collapse operator is

```text
Psi = D_Omega o C_{v*} o pi_0
```

with

```text
Lambda_0 = Z_11 x Z_6 x Z_4
|Lambda_0| = 264
v* = (1, 2, 1)
```

The operational chain is

```text
Lambda_infinity
    --pi_0-->
Lambda_0
    --C_{v*}-->
{v*}
    --D_Omega-->
constant R^4 spatial field
```

The runtime never attempts to instantiate a transfinite or hyper-exponential
domain. Instead, `pi_0` is implemented as a bounded coordinate projection:

```text
pi_0(f1, f2, f3)
  = (f1 mod 11, f2 mod 6, f3 mod 4).
```

This is the finite operational boundary of the abstraction.

## Constant-target kernel

For every `lambda in Lambda_0`,

```text
C_{v*}(lambda) = v* = (1, 2, 1).
```

As a function it is strictly idempotent:

```text
C_{v*}(C_{v*}(lambda)) = C_{v*}(lambda)
C_{v*}^n = C_{v*}, n >= 1.
```

The resulting invariants are:

| Property | Canonical value |
| --- | ---: |
| `|Lambda_0|` | 264 |
| `|im C|` | 1 |
| `|Fix C|` | 1 |
| `|C^{-1}(v*)|` | 264 |
| `|Lambda_0 / ~_C|` | 1 |

The fixed-point count is **one**, not 264. All 264 lattice states form the
one-step basin of attraction of the single fixed point `v*`.

## Quotient semantics

Because `v* != 0`, the non-zero constant map is not generally an additive
group endomorphism. It therefore must not be modeled using a group-theoretic
kernel quotient.

The canonical quotient is the equivalence quotient

```text
lambda ~_C mu  iff  C(lambda) = C(mu).
```

For the constant kernel all states are equivalent, hence

```text
Lambda_0 / ~_C ~= {*}.
```

This distinction is enforced by the C++ regression suite.

## Constant spatial decoder

In the total-collapse limit,

```text
D_Omega(v*, r) = (1, 2, 1, 1) in R^4
```

for every spatial coordinate `r in Omega`. The fourth component is the
homogeneous/field coordinate used by the runtime.

Therefore

```text
Psi(x, r) = (1, 2, 1, 1)
nabla_r Psi = 0
partial_t Psi = 0
```

for the constant-field decoder.

If a future decoder interprets `v*` as the coefficient of a spatial basis
function rather than as a constant field value, the zero-gradient invariant
must be removed.

## C++ implementation

The canonical implementation lives in:

- `cpp_runtime/include/jarvisx/master_equation.hpp`
- `cpp_runtime/src/master_equation_main.cpp`
- `cpp_runtime/tests/master_equation_tests.cpp`

Build and run:

```bash
cmake -S cpp_runtime -B build -DJARVISX_BUILD_GL_VISUALIZER=OFF
cmake --build build --target jarvisx-master-equation jarvisx-master-equation-tests -j
ctest --test-dir build -R dr-moagi-master-equation --output-on-failure
./build/jarvisx-master-equation
```

## Extension boundary

The total-collapse kernel is the limiting case. The information-bearing
generalization is

```text
C_{v*} -> C_theta
```

where the singleton attractor is replaced by a small learned invariant
manifold. Any such extension must preserve explicit verification of:

1. image cardinality / effective latent rank,
2. fixed-point structure,
3. measured idempotence error,
4. reconstruction error,
5. external CTR consistency.

No performance or SOTA claim follows from the algebraic collapse alone.
