# Global Cognitive–Action 3D Kinetics

This runtime converts the historical progression

```text
chatroom -> forum -> web -> social graph -> global multimedia
         -> AI mediator -> agent network -> ambient intelligence
```

into an executable bounded dynamical system.

## State geometry

The active semantic state is a point

```text
r(t) = (I(t), C(t), A(t)) in [0,1]^3
```

where:

- `I` is normalized information state;
- `C` is normalized cognition / synthesis state;
- `A` is normalized action state.

This is an abstract semantic manifold, not a claim that information or cognition are literal physical spatial coordinates.

Velocity and acceleration are explicit:

```text
v = dr/dt
a = dv/dt
```

## Kinetic law

Each agent state contracts toward a bounded target `q` using a damped second-order law:

```text
m a = k(q - r) - gamma v
```

Default parameters are:

```text
m     = 1.0
k     = 2.4
gamma = 1.8
dt    = 0.05
```

The runtime integrates the system with semi-implicit Euler:

```text
v[n+1] = v[n] + a[n] dt
r[n+1] = r[n] + v[n+1] dt
```

The bounded domain is enforced axis-wise on `[0,1]^3`.

## Closed operational loop

```text
world
  -> multimodal content
  -> social / agent graph
  -> recommendation
  -> AI mediator / situation model
  -> kinetic target q
  -> force / acceleration
  -> velocity / position
  -> bounded agent action
  -> world actuator
  -> updated world
  -> feedback
  -> recur
```

The executable also preserves content provenance with deterministic FNV-1a hashes and gives each event a sequence-dependent identifier.

## Verification invariants

A successful run requires all of the following:

1. every event point remains inside `[0,1]^3`;
2. all kinetic position, velocity and acceleration values remain finite;
3. provenance hashes reproduce the stored content hashes;
4. event identifiers are unique;
5. the discrete kinematic residual

   ```text
   ||(r[n+1] - r[n])/dt - v[n+1]||
   ```

   remains below `1e-12`;

6. the discrete dynamic residual

   ```text
   ||(v[n+1] - v[n])/dt - a[n]||
   ```

   remains below `1e-12`;

7. a damped-attractor self-test contracts its target error below `1e-3` while decreasing total kinetic-plus-potential energy.

Verification failure propagates to the executable process exit code.

## Build

```bash
cmake -S cpp_runtime -B build/cpp-runtime \
  -DCMAKE_BUILD_TYPE=Release \
  -DJARVISX_BUILD_GL_VISUALIZER=OFF

cmake --build build/cpp-runtime \
  --target jarvisx-global-cognitive-action-kinetics \
  --parallel
```

Run three closed-loop cycles:

```bash
./build/cpp-runtime/DrMoagi-Global-Cognitive-Action-3D-Kinetics 3
```

Stress the deterministic loop for 1,000 cycles:

```bash
./build/cpp-runtime/DrMoagi-Global-Cognitive-Action-3D-Kinetics 1000
```

Run the CTest gate:

```bash
ctest --test-dir build/cpp-runtime \
  -R global-cognitive-action-kinetics-smoke \
  --output-on-failure
```

## Relationship to the existing Kinetic 3D backend

This target complements, rather than replaces, `native/kinetic3d_backend.cpp`.

- The existing backend performs sparse voxel residual/coarse/fine correction for the Kinetic 3D API.
- This executable models the higher-level information -> cognition -> action network as an explicit second-order 3D dynamical system.

Together they separate physical/voxel correction mechanics from the semantic cognitive-action kinetics layer.
