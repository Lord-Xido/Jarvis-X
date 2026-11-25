# Stochastic Reality-Informed Reckoner (3D)
 
This additive reference extends the reality-constrained reckoning design with
a finite POMDP-style belief filter and bounded risk-constrained receding-horizon
planning. It is not a trained neural world model or physical controller.

## Contract

- State: a normalized probability field over a finite 3D grid (side 2–8).
- Transition: move along one axis or stay; uncertain move-or-stay dynamics,
  clamped boundaries.
- Observation: independent Gaussian coordinate likelihood, normalized in log space.
- Planner: exact open-loop sequence evaluation within an explicit transition
  budget, with discounted expected squared goal distance and action cost.
- Risk: union bound on probability of visiting forbidden cells across the
  horizon including the initial state. This is conservative under the supplied
  transition model, not a real-world safety guarantee.
- Result: proposal only. No actuator access, authorization, or real-world action.
  No complete feasible sequence means no proposal. Budget exhaustion is explicit.
  Re-observe and replan after an independently authorized action.

Run focused tests:
    
    python -m pip install -e '.[test]'
    pytest -o addopts='' -q tests/test_stochastic_reckoner_3d.py

## Integration boundary

The next stage is replacing the fixed transition model with a calibrated
action-conditioned neural world model, comparing against the finite reference
and classical baselines, and connecting a separately authorized transactional
execution adapter. Local attention, photonic arithmetic/backpropagation, HBM
and 3D packaging are research extensions, not implemented by this module.
The quintillion-agent and septillion^septillion spaces remain symbolic.
No exhaustive simulation, learned policy, hardware acceleration or formal
physical safety is claimed. This implementation uses no new dependencies.
