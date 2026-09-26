# Operational 3D Toroidal Feedback Machine

This module implements the toroidal dynamical system as an executable state machine rather than as a rendering metaphor.

## State

Each particle is

[
X_t=(u_t,v_t,sigma_t)in T^2	imes(0,1].
]

Angles are always wrapped to ([0,2pi)). The default example is (u_0=1.4), (v_0=4.8), (sigma_0=1).

## 3D embedding

With major radius (R=18) and minor radius (r_0=6),

[
r_t=sigma_t r_0,qquad
ho_t=R+r_tcos v_t,
]

[
x_t=ho_tcos u_t,qquad
y_t=r_tsin v_t,qquad
z_t=ho_tsin u_t.
]

As (sigma	o0), the embedded point approaches the core circle
((18cos u,0,18sin u)).

## Circular memory

The engine stores angular state as

[
[cos u,sin u,cos v,sin v],
]

which is the real-channel representation of (e^{iu}) and (e^{iv}). A rolling window of length (W) is flattened to

[
Minmathbb R^{B	imes4W}.
]

For the default (B=64,W=32), this is (64	imes128).

## Spectral feedback

Each cycle computes

[
M=USV^	op.
]

The retained spectral energy is

[
E=rac{sum_{i=1}^{K}s_i^2}{sum_i s_i^2}.
]

The strongest right-singular mode is reshaped back to (W	imes4). FFT analysis of its complex (u)- and (v)-channels selects the strongest non-DC temporal frequency (q), together with phases (phi_u,phi_v).

The feedback field is

[
f=Ecos(qu+phi_u)cos v,
qquad
g=Esin(qv+phi_v)cos u.
]

Angular velocities are depth gated:

[
dot u=sigma f,qquad dot v=sigma g.
]

## Permeation and integration

Depth obeys

[
dotsigma=-sigmaln(1/lambda),
qquad
sigma(t)=sigma_0lambda^t.
]

The implementation evaluates this closed form exactly at the RK2 midpoint and endpoint. The angular coordinates use midpoint RK2:

[
k_1=F(X_t),qquad
X_{mid}=X_t+rac{Delta t}{2}k_1,
]

[
k_2=F(X_{mid}),qquad
X_{t+1}=X_t+Delta t,k_2.
]

The updated angles are wrapped, re-embedded in 3D, and appended to the rolling memory before the next cycle.

## Closed loop

[
X_t
ightarrow H_t
ightarrow M_t
ightarrow operatorname{SVD}(M_t)
ightarrow (E_t,q_t,phi_t)
ightarrow F_t
ightarrow X_{t+1}
ightarrow H_{t+1}.
]

The trajectory therefore couples toroidal circulation, poloidal circulation, and exponential inward contraction. Because the field is multiplied by (sigma), angular mobility also decays as the trajectory approaches the core ring.

## Run

Install the NumPy-enabled extra used by the 3D numerical modules:

```bash
pip install -e ".[graphics]"
python -m jarvisx.toroidal_feedback
```

Programmatic use:

```python
from jarvisx.toroidal_feedback import ToroidalFeedbackEngine, make_state

engine = ToroidalFeedbackEngine(make_state(u=1.4, v=4.8, sigma=1.0, batch=64))
result = engine.run(256)

print(result.trajectory.shape)   # (257, 64, 3)
print(result.final_state.sigma)
print(result.telemetry[-1].winding)
```

## Verification

`tests/test_toroidal_feedback.py` checks:

- the Euclidean torus embedding,
- (2pi)-periodic complex-angle encoding,
- the (B	imes4W) history layout,
- bounded SVD energy,
- FFT recovery of a synthetic (q=4) winding,
- exact (sigma_0lambda^t) permeation,
- monotone contraction toward the core ring,
- finite 3D trajectories,
- angular domain invariants after repeated feedback steps.
