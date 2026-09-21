# Jarvis-X

Jarvis-X is a deterministic, auditable virtual machine with a reflex
control layer and policy gate.

## Install

```bash
git clone https://github.com/Lord-Xido/Jarvis-X.git
cd Jarvis-X
pip install -r requirements.txt
pip install .
```

## Dr Moagi Operational Spatial Runtime

`jarvisx.spatial` is the first executable architectural-spatial intelligence
kernel. It provides:

- a canonical hierarchical world-state contract;
- typed architectural entities and relations;
- deterministic 3D predicates for support, containment, intersection, and
  relative height;
- a multi-term objective over geometry, semantics, relations, hierarchy,
  architecture, physics, uncertainty, and description length;
- a bounded propose-shadow-verify-commit echo controller;
- rollback, journaling, canonical serialization, and real SHA-256 world-state
  fingerprints.

```python
from jarvisx.spatial import (
    AABB,
    ArchitecturalWorldModel,
    EchoController,
    Entity,
    EntityKind,
    Relation,
    RelationKind,
    Vector3,
)

world = ArchitecturalWorldModel()
world.add_entity(
    Entity(
        "table",
        EntityKind.OBJECT,
        AABB(Vector3(-1, -0.5, 0), Vector3(1, 0.5, 1)),
        semantic_label="table",
    )
)
world.add_entity(
    Entity(
        "lamp",
        EntityKind.OBJECT,
        AABB(Vector3(-0.2, -0.2, 1.2), Vector3(0.2, 0.2, 2.2)),
        semantic_label="lamp",
    )
)
world.add_relation(Relation("table", "lamp", RelationKind.SUPPORTS))

controller = EchoController(world)
reports = controller.auto_repair_supports()

assert reports[0].accepted
assert controller.world.revision == 1
assert len(controller.world.fingerprint()) == 64
```

Run the complete example with:

```bash
python examples/dr_moagi_spatial_echo.py
```

The formal operational boundary and integration roadmap are documented in
[`docs/DR_MOAGI_SPATIAL_RUNTIME.md`](docs/DR_MOAGI_SPATIAL_RUNTIME.md).

## Sparse 3D Virtual Computer

`jarvisx.virtual3d` provides a sparse logical 3D memory volume with:

- exact coordinate-to-block and local-offset translation;
- clipped boundary blocks;
- lazy allocation at both block and cell level;
- deterministic per-block compression with checksum verification;
- exact global reblocking across candidate block shapes;
- online, offline, and adaptive modes;
- canonical ROM snapshots with SHA-256 state fingerprints.

```python
from jarvisx.virtual3d import OperationalMode, Virtual3DComputer, VolumeGeometry

geometry = VolumeGeometry(
    extent=(6400, 6400, 6400),
    block_shape=(1024, 1024, 1024),
    cell_bytes=1_000_000_000,
)
computer = Virtual3DComputer(geometry, mode=OperationalMode.ADAPTIVE)

computer.write((10, 20, 30), b"Jarvis-X")
assert computer.read((10, 20, 30)) == b"Jarvis-X"

report = computer.optimize_layout(
    [(256, 256, 256), (512, 512, 512), (1024, 1024, 1024)]
)
fingerprint = computer.save_rom("jarvisx-state.jxrom")
```

The capacity is explicit: `6400^3` one-byte cells represent 262.144 GB decimal;
`6400^3` one-gigabyte cells represent 262.144 EB decimal. The runtime never
allocates the logical capacity densely.

Run the complete ROM demonstration with:

```bash
python examples/virtual3d_rom_demo.py
```

The geometry, compression, layout, persistence, and current implementation
boundary are documented in
[`docs/DR_MOAGI_3D_VIRTUAL_COMPUTER.md`](docs/DR_MOAGI_3D_VIRTUAL_COMPUTER.md).

## Sparse Fractal Octree

The concrete recursive spatial substrate is available as
`jarvisx.fractal_octree`. It materializes an eight-location octree while
retaining four active children per active parent under the inward-folding
rule `dx + dy + dz < 2`.

```python
from jarvisx.fractal_octree import build_fractal_octree

root = build_fractal_octree(size=1.0, max_depth=3)
metrics = root.metrics()

assert metrics.active_nodes == 85
assert metrics.active_leaves == 64
assert metrics.retained_volume == 0.125
```

At depth `D`, the deterministic invariants are:

- active leaves: `4 ** D`
- active nodes: `(4 ** (D + 1) - 1) // 3`
- retained volume: `2 ** (-D)` for a unit cube
- similarity dimension: `2`
