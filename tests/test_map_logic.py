from __future__ import annotations

import pytest

from jarvisx.map_logic import (
    MapLogicConfig,
    MapLogicEngine,
    MapLogicError,
    build_toroidal_field,
    canonical_encode,
    map_coordinate,
    toroidal_step,
)
from jarvisx.permeation import PermeationConstitution


def test_coordinate_mapping_is_deterministic_and_bounded() -> None:
    first = map_coordinate("encode", 8)
    second = map_coordinate("encode", 8)

    assert first == second
    assert all(0 <= axis < 8 for axis in first)


def test_canonical_encode_rejects_inconsistent_widths() -> None:
    with pytest.raises(MapLogicError, match="same width"):
        canonical_encode({"a": (1.0, 2.0), "b": (3.0,)})


def test_build_toroidal_field_materializes_full_periodic_cube() -> None:
    encoded = canonical_encode({"a": (1.0,), "b": (-1.0,)})
    field, mapped, core = build_toroidal_field(encoded, side=4)

    assert len(field) == 4**3
    assert len(mapped) == 2
    assert len(core) == 1
    assert all(coordinate in field for _, coordinate in mapped)


def test_toroidal_step_propagates_across_periodic_boundary() -> None:
    side = 3
    field = {
        (x, y, z): (0.0,)
        for x in range(side)
        for y in range(side)
        for z in range(side)
    }
    field[(0, 0, 0)] = (1.0,)

    updated = toroidal_step(
        field,
        (0.0,),
        side=side,
        constitution=PermeationConstitution(),
    )

    assert updated[(side - 1, 0, 0)][0] > 0.0
    assert updated[(0, side - 1, 0)][0] > 0.0
    assert updated[(0, 0, side - 1)][0] > 0.0


def test_engine_executes_monotone_closed_loop_and_emits_receipt() -> None:
    engine = MapLogicEngine(
        MapLogicConfig(
            side=4,
            fold_steps=5,
            memory_decay=0.88,
            residual_gain=0.35,
            saturation_tolerance=0.1,
        )
    )

    receipt = engine.execute(
        {
            "encode": (1.0, 0.0, 0.25),
            "transform": (0.25, 1.0, -0.25),
            "verify": (-0.5, 0.25, 1.0),
        }
    )

    assert receipt.verified is True
    assert receipt.accepted_steps > 0
    assert all(
        later <= earlier
        for earlier, later in zip(
            receipt.objective_history,
            receipt.objective_history[1:],
        )
    )
    assert 0.0 <= receipt.saturation_fraction <= 1.0
    assert receipt.memory_rms >= 0.0
    assert receipt.theoretical_spectral_ceiling < 1.0
    assert receipt.semantic_hbar > 0.0
    assert len(receipt.field_digest) == 64


def test_engine_is_deterministic_for_identical_inputs() -> None:
    samples = {
        "alpha": (0.5, -0.25),
        "beta": (-1.0, 1.0),
    }

    first = MapLogicEngine(MapLogicConfig(side=4, fold_steps=3)).execute(samples)
    second = MapLogicEngine(MapLogicConfig(side=4, fold_steps=3)).execute(samples)

    assert first.field_digest == second.field_digest
    assert first.mapped_coordinates == second.mapped_coordinates
    assert first.objective_history == pytest.approx(second.objective_history)
