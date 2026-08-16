# QSOL Kinetic 3D Processor

This browser app is the bounded research visualization surface for the QSOL kinetic assembly model.

It maps the equation-state registers into a live three-dimensional phase-space view:

- `X` -> 3D position/state trajectory;
- `V` -> velocity vector;
- `F` -> residual vector;
- `G` -> residual force / descent vector;
- `Omega` -> toroidal memory geometry;
- `R0` -> equation energy `H_eq`;
- `PROJ_LAMBDA` -> the bounded `[-10, 10]^3` projection cube.

The embedded presets cover dissipative kinetic flow, exact residual flow, graph-Laplacian synchronization, and the Dr Moagi bounded recurrence.

## Run

Open `index.html` in a modern browser. The app is self-contained and has no external JavaScript or CSS dependencies.

Controls:

- **Run** continuously executes the selected preset.
- **Step** executes one instruction.
- **Pause** stops continuous execution.
- **Reset** clears machine state.
- Drag the 3D canvas to rotate it.
- Use the mouse wheel or trackpad to zoom.

## Trust boundary

This app is a research and visualization surface. It does **not** replace `jarvisx.system_runtime`, does not mutate authoritative Jarvis-X state, and has no network, shell, market, medical, filesystem, or device authority.

The canonical production rule remains:

```text
prediction -> plan -> projection -> execution -> verification -> audit -> commit
```

The browser VM is therefore an observable kinetic model, not a privileged execution path.
