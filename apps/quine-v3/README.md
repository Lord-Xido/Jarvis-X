# QUINE V3 — verified source fixed point and bounded 3D marcher

A zero-dependency Node.js reference for **two separate claims**:

1. A **source-level quine**: evaluating the 55-byte JavaScript expression returns the same 55-byte expression, byte-for-byte. The harness verifies 1,000 executions and SHA-256 integrity.
2. A **bounded 3D implicit-field marcher** for a deformed sphere, with non-mutating ray coordinates, a conservative Lipschitz step divisor, maximum range, and termination limits.

This module is not a model trainer, GPU shader, 3D ANN, autonomous code rewriter, or a 1,000x throughput acceleration.

## Run

~~~sh
node apps/quine-v3/quine_v3_verified.js
node --test apps/quine-v3/quine_v3.test.js
~~~

Node.js 18+ is required. No third-party dependencies.

## Mechanism

The original proposed transform was:

~~~text
Q(s) = E(R(D(s))) + "\n// " + string(march)
~~~

For D(s) = s XOR 42, R(D(s)) = s, so the construction becomes E(s) plus an appended function. It is not self-equal, and whitespace compression changes JavaScript semantics. The original ro.add(...) also mutates the input ray origin when using a mutable vector API.

The replacement source-level quine uses exact self-description rather than whitespace reformatting:

~~~text
P = '(function quine(){return "("+quine.toString()+")()"})()'
F(P) = eval_in_bounded_vm(P) = P
F^1000(P) = P
~~~

The 55-byte string's SHA-256 is:

~~~text
d60ef0606a6e33dc604fd84fb8c73f79dfdd697ae62114ee57f047e8ce3d1619
~~~

verifyQuine executes the **constant bundled quine expression** using a 1-second-per-execution VM timeout; vm is **not** an isolation boundary for executing untrusted attacker-controlled JavaScript. Do not expose the verification entry point to untrusted source strings.

The independent geometric field is:

~~~text
f(x,y,z) = sqrt(x²+y²+z²) - 0.9 + 0.2 sin(3x) sin(3y)
~~~

It is **not an exact signed-distance function**. A global Lipschitz bound is L <= 1.6; for f(p) > epsilon, marching by f(p)/1.6 is a conservative bound from exterior starting points. The ray direction must have unit norm. An accepted hit means f(p) <= epsilon and is a **numerical surface tolerance**, not bitwise geometric closure. Default limits are maxSteps=128, maxDistance=10, and epsilon=0.001.

The CLI tests an axial hit from (0,0,3) along (0,0,-1), a miss along (0,0,1), and origin immutability.

## Relationship to Jarvis-X

- [ADR-034](../../docs/adr/0034-exact-bitwise-fixed-point-verification.md) defines bitwise state closure by zero Hamming residual. This module's quine asserts **exact UTF-8 source string equality**, a narrower independently verified state.
- [Fixed-Point 10e81](../fixed-point-10e81/) provides a separate sparse virtual-address/latent fixed-point emulator. This module **does not modify** that model's state transition or prove convergence of the Dr Moagi ANN equation.
- The canonical Dr Moagi equation remains unchanged. No claim of IP novelty, production deployment, training acceleration, or verified 3D ANN convergence is made here.
