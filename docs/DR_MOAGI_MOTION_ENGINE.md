# Dr Moagi Motion Engine

## Canonical state

The executable motion state is

\[
\mathcal M_t=(\mathbf x_t,\mathbf q_t,\mathbf v_t,\boldsymbol\omega_t,
\mathbf a_t,\boldsymbol\alpha_t,\mathbf F_t,\boldsymbol\tau_t,
\Omega_t,\Lambda_t).
\]

The implemented transition is

\[
\boxed{
\mathcal M_{t+\Delta t}=\Pi_{\Lambda_t}\left[
\mathcal K_{\Delta t}(\mathcal M_t)+
\mathcal D_{\Delta t}(\mathbf F_t,\boldsymbol\tau_t)+
K_t\mathbf E_t^{motion}+\Delta\Omega_t
\right]
}
\]

The engine is a deterministic software rigid-body kernel. It is not a claim of a
continuous physical universe, a full multibody contact solver, or bit-exact
cross-platform hardware execution.

## Operational mechanics

For translation, the runtime uses bounded semi-implicit Euler integration:

\[
\mathbf a_t=\operatorname{clip}_{a_{max}}\left(\frac{\mathbf F_t}{m}\right),
\qquad
\mathbf v_{t+1}=\operatorname{clip}_{v_{max}}(
\mathbf v_t+\mathbf a_t\Delta t),
\]

\[
\mathbf x_{t+1}=\mathbf x_t+\mathbf v_{t+1}\Delta t.
\]

For rotation with diagonal inertia \(\mathbf I\):

\[
\boldsymbol\alpha_t=\mathbf I^{-1}\left[
\boldsymbol\tau_t-\boldsymbol\omega_t\times(\mathbf I\boldsymbol\omega_t)
\right],
\]

\[
\boldsymbol\omega_{t+1}=\operatorname{clip}_{\omega_{max}}(
\boldsymbol\omega_t+\boldsymbol\alpha_t\Delta t),
\]

\[
\mathbf q_{t+1}=\operatorname{normalize}\left[
\mathbf q_t\otimes\operatorname{Exp}\left(
\tfrac12\boldsymbol\omega_{t+1/2}\Delta t
\right)
\right].
\]

## Prediction and observation correction

The propagated state is treated as the motion prediction. Optional position,
velocity, orientation, and angular-velocity observations generate residuals:

\[
\mathbf E_x=\mathbf z_x-\widehat{\mathbf x},
\qquad
\mathbf E_v=\mathbf z_v-\widehat{\mathbf v}.
\]

Confidence-weighted gains correct the prediction. Residuals also update retained
motion memory:

\[
\Omega_{t+1}=\rho\Omega_t+(1-\rho)c_t\mathbf E_t.
\]

## Constraint projection

The projector enforces:

- finite position, orientation, velocity, acceleration, force, torque and memory;
- positive mass and positive diagonal inertia;
- normalized quaternions;
- maximum linear and angular velocity;
- maximum linear and angular acceleration;
- world-space bounds;
- optional floor contact with configurable restitution;
- a bounded integration step \(0<\Delta t\le\Delta t_{max}\).

The state dataclass is immutable. A failed step raises before a replacement state
is returned, providing transactional state-transition semantics to the caller.

## Deterministic state seal

Each accepted state is serialized using hexadecimal floating-point strings,
canonical key ordering and SHA-256:

\[
H_t=SHA256(CanonicalHexFloatJSON(\mathcal M_t)).
\]

This gives deterministic state hashes for identical Python floating-point
states. It does not by itself guarantee bit-identical arithmetic across every
CPU, Python implementation or compiler.

## CLI

```bash
cat > state.json <<'JSON'
{
  "mass": 2.0,
  "position": [0, 0, 0],
  "orientation": [1, 0, 0, 0],
  "velocity": [0, 0, 0],
  "inertia": [1, 1, 1]
}
JSON

drmoagi-motion state.json \
  --dt 0.0166666667 \
  --steps 120 \
  --force '[4,0,-19.62]' \
  --torque '[0,0,0.5]' \
  --summary-only
```

Constraints and per-step observations may be supplied as JSON strings or files.

## Complexity

A single rigid-body transition performs a constant number of three-component
vector and four-component quaternion operations:

\[
T_{step}=O(1),\qquad M_{step}=O(1).
\]

A trajectory of \(N\) retained results costs:

\[
T_{trajectory}=O(N),\qquad M_{trajectory}=O(N).
\]

## Validation

The dedicated tests cover:

- constant-force translation arithmetic;
- quaternion normalization during torque-driven rotation;
- observation residual correction and memory update;
- speed limits and floor contact;
- deterministic state hashing;
- invalid mass, time step, confidence and non-finite input rejection;
- quaternion composition;
- command-line execution.
