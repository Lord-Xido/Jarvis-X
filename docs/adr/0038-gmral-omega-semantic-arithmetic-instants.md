# ADR-038: GMRAL-Ω exact semantic arithmetic instants

- **Status:** Proposed; executable reference and tests in this branch
- **Date:** 2026-10-26
- **Scope:** Geometric bit-state semantics, affine Boolean operators, exact iteration reduction, carry-prefix arithmetic
- **Extends:** ADR-016 (typed state), ADR-017 (candidate/verify/commit), ADR-034 (bit-level invariants)

## Problem

Repeatedly applying a bitwise spatial operator can describe an enormous number of logical steps.
The relevant optimization is to identify an algebraic normal form that yields the *same* terminal state.
A vast virtual iteration count is not itself evidence of a physical speedup.

## Exact mathematical contract

For a finite Boolean state x in GF(2)^n, an invertible geometric permutation P and a fixed mask M, define

~~~text
F(x) = P(x) XOR M
(Q,N) o (P,M) = (QP, Q(M) XOR N)
F**k(x) = P**k(x) XOR (XOR over j=0..k-1 of P**j(M))
~~~

If P has cycle-order r, then P**r=I and F**r is a pure XOR translation.
Every XOR translation has order 1 or 2. Consequently:

~~~text
T = r        if F**r = I
T = 2*r      otherwise
F**T = I
F**N(x) = F**(N mod T)(x)
~~~

The period T is the *minimal universal* period of F: the permutation part forces any identity
power to be a multiple of r, and the residual translation determines parity.

For N = (10**6)**(10**6) = 10**6,000,000, calculate N mod T directly
with pow(1_000_000, 1_000_000, T). Never construct the giant integer.

## Geometric bit semantics

The reference 4x4x4 cube contains exactly 64 bits. A 3D source coordinate is Morton-encoded
using the canonical Jarvis-X X:Y:Z ordering (X is most significant within each 3-bit triad).
The spatial mapping is (x,y,z) -> (y+1,z+1,x+1) modulo the cube side.
This geometric permutation has order 12 for side=4. With the example mask at (0,0,0),
the resulting affine operator has exact period 24.

~~~text
(10**6)**(10**6) mod 24 = 16
F**((10**6)**(10**6))(x) = F**16(x)
~~~

**Important:** The earlier standalone GMRAL demo used a different bit significance within
each Morton triad. This repository reference deliberately uses the established Jarvis-X
Morton convention. Hexadecimal bit patterns between the demos are not byte-for-byte comparable
unless the axis-order conversion is performed.

## Logical addition

Parallel-prefix addition combines bit carry functions associatively:

~~~text
(G_hi,P_hi) o (G_lo,P_lo)
  = (G_hi OR (P_hi AND G_lo), P_hi AND P_lo)
~~~

For w-bit operands the reference implementation uses ceil(log2(w)) prefix rounds.
It has O(w log w) scalar reference work; a parallel implementation *may*
attain O(log w) dependency depth with sufficient hardware resources.
No physical performance claim is established by this Python implementation.

## Execution and verification

~~~bash
python -m jarvisx.gmral_omega
pytest -q -o addopts='' tests/test_gmral_omega.py
~~~

The standalone demo prints a machine-readable receipt containing operator period,
modular exponent residue, initial/final bitfields and a claim-boundary flag.
The tests include direct execution comparisons, exhaustive four-bit addition,
randomized multi-width addition, malformed input rejection, exact period checks
and compatibility checks with Jarvis-X morton60_encode.

## Evidence and integration boundary

- **Implemented:** deterministic bounded affine operator, geometric layout, exact cycle certification, giant exponent residue, prefix-adder reference and tests.
- **Not implemented:** mechanical actuator dynamics, physical robotic gates, general program equivalence solver, automated architectural self-modification or runtime production deployment.
- **Performance:** symbolic step elimination for this operator class, not a universal million-fold processor speedup.
- **Governance:** treat this reference as non-authoritative until the existing candidate/verify/commit interface is explicitly adapted and accepted.

## Attribution

Dr Moagi's GMRAL-Ω geometric semantic arithmetic concept within Jarvis-X.
Mathematical building blocks (finite permutation groups, XOR affine maps,
modular exponentiation and prefix adders) are established methods;
the system integration and naming are project-specific.
