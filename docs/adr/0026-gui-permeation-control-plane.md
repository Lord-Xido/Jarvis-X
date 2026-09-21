# ADR-026: GUI permeation control and observation plane

- **Status:** Proposed
- **Date:** 2026-09-21
- **Decision scope:** browser, desktop, 3D and multi-panel GUI surfaces

## Context

Jarvis-X now exposes several visual surfaces, including the Dr Moagi OS browser
control plane, Total Permeation 3D visualization, Windows multimodal tooling and
multi-panel research interfaces. These surfaces need one shared rule for how a
GUI observes and influences a recursive runtime.

The GUI must not become an alternate authority path. A button, slider, shell
message, drag event or panel interaction is an **intent**, not a direct mutation
of authoritative runtime memory.

The canonical processing sequence is:

```text
user event
  -> GUI reducer / intent
  -> bounded command proposal
  -> runtime capability and bounds checks
  -> candidate state
  -> CTR / Pi_Lambda verification
  -> commit | rollback
  -> immutable snapshot / telemetry
  -> GUI projection
  -> render
```

## Decision

Add `jarvisx.gui_control_plane` as the shared deterministic GUI contract.

The module defines:

1. immutable engine snapshots;
2. epoch-targeted GUI commands;
3. a bounded command queue with duplicate command-ID rejection;
4. render-only versus runtime-mutating command classification;
5. explicit capability requirements for runtime proposals;
6. stale-snapshot rejection;
7. bounded batch execution before runtime dispatch;
8. sparse panel refresh scheduling;
9. hierarchical region/tile/voxel/bit level-of-detail selection;
10. pure snapshot-to-panel projection.

The module **does not execute authoritative runtime mutations**. It produces
validated proposals that still require the existing runtime verification and
commit path.

## State separation

The architecture explicitly separates:

```text
authoritative runtime state S_t
        |
        v
immutable GUI snapshot G_t
        |
        +--> 3D projection
        +--> telemetry panel
        +--> VM / memory panel
        +--> entropy / residual panel
        +--> shell / status surface
```

A GUI projection may discard detail for rendering. That loss does not alter the
authoritative runtime state.

Formally, each panel is a projection:

\[
V_i = P_i(S_t).
\]

The projection is one-way with respect to authority. User interaction produces a
new command proposal rather than mutating `S_t` in place.

## Command law

For GUI command `c_t` targeting snapshot epoch `e_t`:

\[
Q(c_t, S_t)=
\begin{cases}
\text{render-only}, & c_t \in C_{\rm visual},\\
\text{runtime proposal}, & c_t \in C_{\rm operational}.
\end{cases}
\]

Operational proposals remain subordinate to the canonical transaction:

\[
S_{t+1}
=
\begin{cases}
S'_t,&\Pi_\Lambda(S_t,S'_t)=1,\\
S_t,&\text{otherwise}.
\end{cases}
\]

If the command epoch does not match the current snapshot epoch, the GUI contract
fails closed as a stale command.

## Sparse panel scheduling

A logical GUI may expose thousands of panels without treating each panel as a
full physical runtime.

The reference scheduling classes are:

| Panel state | Target refresh interval |
| --- | ---: |
| foreground | 16 ms |
| visible background | 100 ms |
| inactive | 1000 ms |
| off-screen | not scheduled |

These values define deterministic default scheduling policy, not guaranteed
display refresh rates.

The invariant is:

\[
N_{\rm panels,logical} \gg N_{\rm panels,rendered\ now}
\]

when most panels are inactive or off-screen.

## Hierarchical 3D projection

The reference GUI projection uses level-of-detail rather than materializing every
logical bit:

```text
global region
  -> tile
  -> voxel
  -> bit
```

LOD selection depends on camera distance and active-state density. Exact
thresholds are policy and may evolve, but the authoritative/runtime separation
must not.

## Telemetry boundary

GUI fields must identify their source class:

- measured runtime telemetry;
- measured local render telemetry;
- externally injected measured telemetry;
- derived/model values;
- purely visual state.

Rendered motion, glow, particle count, apparent convergence or visual complexity
must not be promoted into unmeasured runtime-performance claims.

## Relationship to existing surfaces

- `src/jarvisx/dr_moagi_os_ui.py` remains a live GUI for the bounded Dr Moagi OS.
- `apps/total-permeation-3d/` remains visualization-only unless explicitly
  connected through this command/snapshot contract.
- native Windows or future desktop GUIs may adopt the same contract.
- large panel-count interfaces should share the same runtime snapshots and sparse
  scheduler rather than instantiate one full engine per panel.

## Validation

The reference tests verify:

- immutable snapshot constraints and sparse-state fractions;
- render-only commands do not request Pi_Lambda authority;
- runtime commands remain proposals;
- stale commands fail closed;
- batch execution is bounded before dispatch;
- command queues are bounded and command IDs are unique;
- foreground/background/off-screen scheduling is deterministic;
- LOD projection exposes bit/tile/region views without changing runtime state;
- telemetry epochs cannot silently move backward.

## Final invariant

The GUI processing invariant is:

```text
OBSERVE
 -> PROJECT
 -> INTERACT
 -> PROPOSE
 -> VERIFY
 -> COMMIT | ROLLBACK
 -> SNAPSHOT
 -> RENDER
 -> OBSERVE
```

The GUI permeates the runtime as an observation and command surface, while
authority remains in the verified state-transition system.
