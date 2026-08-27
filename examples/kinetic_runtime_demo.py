"""Minimal executable demonstration of the canonical Jarvis-X kinetic cycle."""

from dataclasses import dataclass, replace

from jarvisx.kinetic_runtime import KineticTransactionEngine, ValidatorResult


@dataclass(frozen=True)
class RuntimeState:
    energy: float
    generation: int = 0


@dataclass(frozen=True)
class Candidate:
    energy: float


def main() -> None:
    engine = KineticTransactionEngine[
        RuntimeState, dict[str, float], tuple[float, float], Candidate
    ](
        snapshot=lambda state: replace(state),
        observe=lambda state: {"energy": state.energy},
        encode=lambda state, obs: (state.energy, obs["energy"]),
        propose=lambda state, obs, encoded: Candidate(energy=max(0.0, encoded[0] * 0.8)),
        shadow=lambda state, candidate: {
            "baseline_energy": state.energy,
            "candidate_energy": candidate.energy,
        },
        validators=(
            lambda state, candidate: ValidatorResult(
                name="energy_improvement",
                passed=candidate.energy <= state.energy,
                metrics={"delta": candidate.energy - state.energy},
            ),
        ),
        commit=lambda state, candidate: RuntimeState(
            energy=candidate.energy,
            generation=state.generation + 1,
        ),
        rollback=lambda state: state,
    )

    state = RuntimeState(energy=1.0)
    for _ in range(4):
        result = engine.step(state)
        state = result.state
        print(result.receipt.to_dict())


if __name__ == "__main__":
    main()
