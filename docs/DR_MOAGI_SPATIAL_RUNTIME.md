# Dr Moagi Operational Spatial Runtime

## Status

This document defines the first executable boundary of the Dr Moagi
architectural-spatial intelligence framework inside Jarvis-X.

The implementation in `jarvisx.spatial` is a deterministic world-state,
geometry, verification, provenance, and rollback kernel. It does **not** claim
that image perception, neural 3D reconstruction, rendering, BIM export, or
language-conditioned planning are complete. Those systems must compile their
outputs into the world-state contract defined here.

## 1. System boundary

Jarvis-X is divided into two planes:

1. **Spatial intelligence plane** — perception, geometry, entities, relations,
   architectural constraints, uncertainty, and reasoning.
2. **Control plane** — policy, transactions, verification, journaling,
   deterministic execution, commit, and rollback.

The spatial plane produces candidate worlds. The control plane decides whether
a candidate is admissible.

```text
observation -> world hypothesis -> shadow correction -> verification
      ^                                                |
      |                                                v
      +------------- committed world <- commit/rollback
```

## 2. Canonical world state

The operational state is:

\[
\mathcal W_t=(\mathcal G_t,\mathcal F_t,\mathcal K_t,\mathcal U_t,\mathcal M_t),
\]

where:

- `G`: hierarchical entity and relation graph;
- `F`: future continuous geometric or renderable field;
- `K`: geometric, physical, architectural, and policy constraints;
- `U`: uncertainty and confidence;
- `M`: temporal identity and revision history.

The current executable milestone implements the discrete, auditable subset:

\[
\mathcal W_t^{(0)}=(V_t,E_t,K_t,U_t,J_t).
\]

## 3. Entity hierarchy

The implemented entity kinds are:

```text
surface -> part -> object -> opening/boundary -> room -> zone -> building
```

Every entity contains:

- stable identifier;
- entity kind;
- axis-aligned 3D bounds;
- semantic label;
- confidence;
- uncertainty;
- optional parent identifier;
- typed metadata.

Axis-aligned boxes are intentionally the first geometry contract. They make
spatial predicates deterministic and testable. Future mesh, Gaussian, SDF,
point-map, or `SE(3)` pose representations can be attached without changing
entity identity or relation semantics.

## 4. Relation ontology

The runtime declares geometric, constructive, spatial, and functional
relations, including:

- `above`, `below`, `inside`, `intersects`;
- `aligned`, `parallel`, `orthogonal`;
- `supports`, `attached_to`, `embedded_in`, `spans`, `encloses`;
- `adjacent`, `connected_through`, `visible_from`, `accessible_from`;
- `used_for`, `serves`, `controls`, `permits`, `obstructs`.

Relations are directed and confidence-bearing. The present verification kernel
has hard geometric validators for `supports` and `inside`. Other predicates are
part of the stable ontology and will acquire validators incrementally.

## 5. Geometric predicates

For a supporter box `A` and supported box `B`, support requires:

\[
|z_{B,\min}-z_{A,\max}|\leq\varepsilon
\]

and:

\[
\frac{\operatorname{area}(A_{xy}\cap B_{xy})}
{\operatorname{area}(B_{xy})}\geq\tau.
\]

The denominator is the supported object's footprint. A large floor can
therefore completely support a small object.

Containment requires both corners of an inner box to lie inside the outer box,
within the declared tolerance.

## 6. Operational objective

The executable objective is decomposed as:

\[
\mathcal F=
\lambda_gL_g+
\lambda_sL_s+
\lambda_rL_r+
\lambda_hL_h+
\lambda_aL_a+
\lambda_pL_p+
\lambda_uL_u+
\lambda_mL_{MDL}.
\]

The current terms measure:

- invalid or degenerate geometry;
- semantic confidence error;
- relation confidence error;
- parent-child containment error;
- architectural containment error;
- physical support error;
- entity uncertainty;
- description length relative to a declared budget.

Observation, rendering, and query terms are present in the objective contract
but remain zero until their corresponding subsystems are connected.

## 7. Verified echo loop

A correction is never committed directly.

```text
PROPOSE -> SHADOW -> SCORE -> VALIDATE -> COMMIT or REJECT
```

For candidate `W'` and committed world `W`, acceptance requires:

\[
\mathcal F(\mathcal W')
\leq
\mathcal F(\mathcal W)-\delta
\]

and:

\[
\operatorname{Violations}(\mathcal W')=\varnothing.
\]

A successful commit:

1. increments the revision;
2. records the previous fingerprint;
3. records the operation and rationale;
4. appends the score change to the journal;
5. stores a rollback snapshot.

A failed proposal cannot mutate the committed world.

## 8. Bounded auto-repair

`EchoController.auto_repair_supports()` is the first complete automatic
operation. It:

1. finds the first declared support relation that fails geometric validation;
2. computes the exact vertical correction required to close the support gap;
3. creates a shadow world;
4. evaluates the candidate;
5. validates all hard constraints;
6. commits only when the objective improves;
7. fingerprints and journals the revision;
8. stops at a declared maximum number of steps.

This is bounded self-correction, not unrestricted self-rewriting.

## 9. Cryptographic anchoring

Every world has a canonical JSON serialization. Entity and relation ordering is
normalized before serialization. The operational fingerprint is:

\[
h_t=\operatorname{SHA256}(\operatorname{CanonicalJSON}(\mathcal W_t)).
\]

The digest is computed by the runtime; it is not a conceptual placeholder.

## 10. Running the milestone

```bash
python examples/dr_moagi_spatial_echo.py
pytest -q tests/test_spatial_runtime.py
```

The demonstration begins with a lamp floating above a table while a `supports`
relation is declared. The echo controller closes the vertical gap, verifies the
new state, commits revision 1, and emits a SHA-256 fingerprint and journal.

## 11. Integration contract for perception models

A future image or multi-view encoder must produce:

```python
ArchitecturalWorldModel(
    entities={...},
    relations=[...],
    metadata={
        "observation_ids": [...],
        "camera_model": "...",
        "model_version": "...",
        "calibration": {...},
    },
)
```

It must not bypass verification by directly mutating committed state.

## 12. Next operational increments

1. Add oriented boxes and `SE(3)` poses.
2. Add room, opening, adjacency, and circulation validators.
3. Add probabilistic multi-hypothesis worlds.
4. Add observation and differentiable-rendering residuals.
5. Add a geometry compiler for images, video, depth, or point clouds.
6. Add open-vocabulary object and relation compilation.
7. Add BIM/IFC and glTF projections.
8. Bind spatial commits into the existing persistent Jarvis-X ledger.
9. Add benchmark gates for geometry, relation recall, architectural reasoning,
   uncertainty calibration, and rollback integrity.

## 13. Non-claims

This milestone is not a trained architectural vision model and is not evidence
of general intelligence. It is the deterministic operational substrate needed
for such models to produce auditable, geometrically testable, reversible world
states.
