from __future__ import annotations

from dataclasses import dataclass, replace

from jarvisx.kinetic_runtime import (
    KineticTransactionEngine,
    ValidatorResult,
    canonical_hash,
)


@dataclass(frozen=True)
class State:
    value: int
    generation: int = 0


@dataclass(frozen=True)
class Candidate:
    value: int
    predicted_cost: int


def build_engine(*, proposed_value: int, pass_gate: bool):
    def snapshot(state: State) -> State:
        return replace(state)

    def observe(state: State) -> dict[str, int]:
        return {"value": state.value}

    def encode(state: State, observation: dict[str, int]) -> tuple[int, int]:
        return (state.value, observation["value"])

    def propose(
        state: State,
        observation: dict[str, int],
        encoded: tuple[int, int],
    ) -> Candidate:
        del state, observation, encoded
        return Candidate(value=proposed_value, predicted_cost=abs(proposed_value))

    def shadow(state: State, candidate: Candidate) -> dict[str, object]:
        return {
            "baseline_cost": abs(state.value),
            "candidate_cost": candidate.predicted_cost,
        }

    def gate(state: State, candidate: Candidate) -> ValidatorResult:
        del state
        return ValidatorResult(
            name="lambda_test_gate",
            passed=pass_gate,
            metrics={"candidate_value": candidate.value},
            reason="fixture gate",
        )

    def commit(state: State, candidate: Candidate) -> State:
        return State(candidate.value, state.generation + 1)

    def rollback(state: State) -> State:
        return state

    return KineticTransactionEngine(
        snapshot=snapshot,
        observe=observe,
        encode=encode,
        propose=propose,
        shadow=shadow,
        validators=(gate,),
        commit=commit,
        rollback=rollback,
    )


def test_commit_follows_full_kinetic_cycle_and_hash_links_receipts():
    engine = build_engine(proposed_value=3, pass_gate=True)
    first = engine.step(State(5))

    assert first.committed is True
    assert first.state == State(3, 1)
    assert first.receipt.decision == "commit"
    assert first.receipt.stages == (
        "snapshot",
        "observe",
        "encode",
        "propose",
        "shadow",
        "verify",
        "commit",
        "journal",
        "reenter",
    )
    assert first.receipt.parent_state_hash == canonical_hash(State(5))
    assert first.receipt.resulting_state_hash == canonical_hash(State(3, 1))
    assert first.receipt.previous_receipt_hash == "0" * 64

    second = engine.step(first.state)
    assert second.receipt.previous_receipt_hash == first.receipt.receipt_hash
    assert second.receipt.transaction_id != first.receipt.transaction_id


def test_failed_lambda_gate_rolls_back_authoritative_state():
    engine = build_engine(proposed_value=999, pass_gate=False)
    start = State(7, 4)
    result = engine.step(start)

    assert result.committed is False
    assert result.state == start
    assert result.receipt.decision == "rollback"
    assert "rollback" in result.receipt.stages
    assert "commit" not in result.receipt.stages
    assert result.receipt.parent_state_hash == result.receipt.resulting_state_hash
    assert result.receipt.validators[0].passed is False
