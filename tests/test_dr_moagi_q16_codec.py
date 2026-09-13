from __future__ import annotations

import copy
import random
from itertools import product

import pytest

from jarvisx.dr_moagi_q16_field import (
    INT32_MAX,
    INT32_MIN,
    MAX_CODEC_TAPS,
    Q_SCALE,
    UINT32_MASK,
    DrMoagiQ16Config,
    DrMoagiQ16FieldRuntime,
    Q16Interval,
    Sha3Ledger,
    binary_intent_mask,
    q_from_float,
    q_mul,
    q_shift_left,
)


def cell(**changes):
    kwargs = dict(
        coordinate=(0, 0, 0),
        values=[Q_SCALE // 4] * 64,
        psi_masks=[UINT32_MASK] * 64,
        phi_weights=[Q_SCALE] * 64,
        theta_weights=[Q_SCALE // 4],
        constraint=Q16Interval(0, INT32_MAX),
        ledger=Sha3Ledger(),
        encoder_shift=6,
    )
    kwargs.update(changes)
    return DrMoagiQ16FieldRuntime.encode_decode_cell(**kwargs)


def snapshot(runtime):
    return copy.deepcopy(
        (
            runtime.state,
            runtime.previous_state,
            runtime.tick,
            runtime.rng.getstate(),
            runtime.ledger.entries,
            runtime.ledger.head,
        )
    )


def test_shift_six_preserves_constant_64_tap_mean_and_decoder_gain():
    trace = cell()
    assert trace.convolution_raw == Q_SCALE // 4
    assert trace.upshift_raw == Q_SCALE
    assert trace.output_raw == Q_SCALE // 4


def test_wide_sum_saturates_after_normalization_and_cancellation():
    assert cell(values=[INT32_MAX] * 64).convolution_raw == INT32_MAX
    trace = cell(
        values=[INT32_MAX, INT32_MAX, -INT32_MAX],
        psi_masks=[UINT32_MASK] * 3,
        phi_weights=[Q_SCALE] * 3,
        encoder_shift=0,
    )
    assert trace.convolution_raw == INT32_MAX


def test_signed_product_matches_integer_floor_oracle():
    rng = random.Random(29)
    cases = [(INT32_MIN, INT32_MIN), (-1, 1), (-3, Q_SCALE // 2)]
    cases += [
        (rng.randint(INT32_MIN, INT32_MAX), rng.randint(INT32_MIN, INT32_MAX)) for _ in range(500)
    ]
    for a, b in cases:
        expected = max(INT32_MIN, min(INT32_MAX, (a * b) // Q_SCALE))
        assert q_mul(a, b) == expected
    assert q_mul(-1, 1) == -1
    assert q_from_float(1e308) == INT32_MAX
    assert q_shift_left(-1, 10**9) == INT32_MIN


def test_binary_enable_preserves_all_bits():
    assert binary_intent_mask(True) == UINT32_MASK
    assert binary_intent_mask(False) == 0
    assert cell(psi_masks=[binary_intent_mask(False)] * 64).output_raw == 0
    with pytest.raises(TypeError):
        binary_intent_mask(1)


@pytest.mark.parametrize(
    "changes",
    [
        {"theta_weights": [0.5]},
        {"encoder_shift": -1},
        {"encoder_shift": True},
        {"coordinate": (0, 0, -1)},
        {"tick": True},
        {"constraint": None},
        {"theta_weights": []},
        {"theta_weights": [1] * (MAX_CODEC_TAPS + 1)},
    ],
)
def test_invalid_cell_does_not_bind_ledger(changes):
    ledger = Sha3Ledger()
    with pytest.raises((ValueError, TypeError)):
        cell(ledger=ledger, **changes)
    assert ledger.entries == []
    assert ledger.head == Sha3Ledger.GENESIS


def test_sparse_codec_matches_independent_dense_average_with_zero_boundaries():
    side = 5
    domain = list(product(range(side), repeat=3))
    source = {coord: ((coord[0] + 2 * coord[1] + coord[2]) % 4) * 8192 for coord in domain}
    source = {c: v for c, v in source.items() if v}
    offsets = list(product(range(-1, 3), repeat=3))
    runtime = DrMoagiQ16FieldRuntime(DrMoagiQ16Config(side=side))
    runtime.load(source)
    original = dict(source)
    for _ in range(3):
        # For unit encoder weights and quarter decoder, the law reduces to floor(mean).
        dense = {}
        for x, y, z in domain:
            mean = sum(source.get((x + dx, y + dy, z + dz), 0) for dx, dy, dz in offsets) // 64
            if mean:
                dense[(x, y, z)] = mean
        report = runtime.step_codec(
            phi_kernel={o: Q_SCALE for o in reversed(offsets)},
            theta_weights=[Q_SCALE // 4],
        )
        assert report.output == dense
        assert report.squared_error_raw == sum(
            (dense.get(c, 0) - source.get(c, 0)) ** 2 for c in domain
        )
        assert runtime.previous_state == source
        source = dense
    assert source != original  # The lossy filter does not establish zero distortion.
    assert runtime.ledger.verify()


def test_virtual_exacell_identity_loop_is_sparse_and_replayable():
    config = DrMoagiQ16Config(side=1_000_000, max_active_cells=8)
    data = {(999999, 999999, 999999): 16000, (0, 0, 0): 8000}
    a, b = DrMoagiQ16FieldRuntime(config), DrMoagiQ16FieldRuntime(config)
    a.load(data)
    b.load(dict(reversed(list(data.items()))))
    for _ in range(3):
        for runtime in (a, b):
            report = runtime.step_codec(
                phi_kernel={(0, 0, 0): 64 * Q_SCALE}, theta_weights=[Q_SCALE // 4]
            )
            assert report.output == data
            assert report.squared_error_raw == 0
            assert report.processed_cells == 2
        assert a.ledger.head == b.ledger.head
    assert a.logical_layout()["logical_cells"] == 10**18
    assert a.logical_layout()["dense_q16_bytes"] == 4 * 10**18


def test_halo_mask_and_positive_constraint_on_absent_cell():
    runtime = DrMoagiQ16FieldRuntime(DrMoagiQ16Config(side=8))
    runtime.load({(3, 3, 3): 16000})
    report = runtime.step_codec(
        phi_kernel={(1, 0, 0): 64 * Q_SCALE},
        theta_weights=[Q_SCALE // 4],
        constraints={(7, 7, 7): Q16Interval(500, 500)},
    )
    assert report.output == {(2, 3, 3): 16000, (7, 7, 7): 500}
    runtime.load({(3, 3, 3): 16000})
    assert (
        runtime.step_codec(
            phi_kernel={(1, 0, 0): 64 * Q_SCALE},
            theta_weights=[Q_SCALE // 4],
            psi_masks={(3, 3, 3): 0},
        ).output
        == {}
    )


@pytest.mark.parametrize("failure", ["budget", "coordinate", "decoder", "ledger"])
def test_failed_cycle_rolls_back_all_state(failure):
    runtime = DrMoagiQ16FieldRuntime(
        DrMoagiQ16Config(side=8, max_active_cells=1, max_ledger_entries=1)
    )
    runtime.load({(2, 2, 2): 16000})
    kwargs = dict(phi_kernel={(0, 0, 0): 64 * Q_SCALE}, theta_weights=[Q_SCALE // 4])
    if failure == "budget":
        kwargs["phi_kernel"] = {(1, 0, 0): Q_SCALE}
    elif failure == "coordinate":
        kwargs["psi_masks"] = {(8, 0, 0): UINT32_MASK}
    elif failure == "decoder":
        kwargs["theta_weights"] = [0.5]
    else:
        runtime.step_codec(**kwargs)
    before = snapshot(runtime)
    with pytest.raises((ValueError, TypeError)):
        runtime.step_codec(**kwargs)
    assert snapshot(runtime) == before


def test_ledger_detaches_inputs_and_requires_external_anchor_for_rewritten_history():
    ledger = Sha3Ledger()
    data = {"nested": [1, 2]}
    head = ledger.bind(data)
    data["nested"][0] = 99
    assert ledger.verify(head)
    replacement = Sha3Ledger()
    replacement.bind({"nested": [99, 2]})
    assert replacement.verify()
    assert not replacement.verify(head)
    ledger.entries[0]["record"]["nested"][0] = 99
    assert not ledger.verify(head)
    with pytest.raises(ValueError):
        ledger.bind({"new": 1})


def test_empty_field_still_validates_and_commits_exactly_one_receipt():
    runtime = DrMoagiQ16FieldRuntime()
    with pytest.raises(TypeError):
        runtime.step_codec(phi_kernel={(0, 0, 0): Q_SCALE}, theta_weights=[None])
    report = runtime.step_codec(phi_kernel={(0, 0, 0): Q_SCALE}, theta_weights=[Q_SCALE])
    assert report.processed_cells == 0
    assert len(runtime.ledger.entries) == 1
    assert set(runtime.ledger.entries[0]["record"]) == {
        "structural",
        "semantic",
        "behavioral",
        "temporal",
        "metadata",
    }


def test_master_field_support_rng_rollback_and_reload():
    config = DrMoagiQ16Config(side=8, eta_amplitude_raw=100, seed=37, max_ledger_entries=1)
    runtime = DrMoagiQ16FieldRuntime(config)
    source = {(2, 2, 2): 16000}
    runtime.load(source)
    first = runtime.step_field(phi_kernel={(1, 0, 0): Q_SCALE})
    assert (1, 2, 2) in first.output
    before = snapshot(runtime)
    with pytest.raises(ValueError, match="ledger entry budget"):
        runtime.step_field()
    assert snapshot(runtime) == before
    runtime.load(source)
    assert runtime.step_field(phi_kernel={(1, 0, 0): Q_SCALE}) == first


@pytest.mark.parametrize(
    "kwargs",
    [
        {"side": 0},
        {"max_active_cells": 0},
        {"max_ledger_entries": True},
        {"eta_amplitude_raw": -1},
        {"gamma_gain_raw": 0.5},
    ],
)
def test_config_rejects_invalid_numeric_contract(kwargs):
    with pytest.raises((ValueError, TypeError)):
        DrMoagiQ16Config(**kwargs)
