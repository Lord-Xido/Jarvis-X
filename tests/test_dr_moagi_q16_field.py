from __future__ import annotations

import pytest

from jarvisx.dr_moagi_q16_field import (
    INT32_MAX,
    INT32_MIN,
    Q_OUTPUT_MAX_RAW,
    DrMoagiQ16Config,
    DrMoagiQ16FieldRuntime,
    Q16Interval,
    Sha3Ledger,
    bitwise_gate,
    q_add,
    q_from_float,
    q_mul,
    q_to_float,
    temporal_compression_law,
)


def test_q16_quantization_multiply_and_saturation():
    a = q_from_float(1.5)
    b = q_from_float(0.5)
    assert q_to_float(q_mul(a, b)) == pytest.approx(0.75, abs=1 / 65536)
    assert q_add(INT32_MAX, 1) == INT32_MAX
    assert q_add(INT32_MIN, -1) == INT32_MIN


def test_bitwise_intent_gate_operates_on_32_bit_payload():
    raw = 0x0001FFFF
    assert bitwise_gate(raw, 0xFFFF0000) == 0x00010000
    assert bitwise_gate(raw, 0x0000FFFF) == 0x0000FFFF


def test_discrete_codec_bus_binds_sha3_and_projects_output():
    ledger = Sha3Ledger()
    trace = DrMoagiQ16FieldRuntime.encode_decode_cell(
        coordinate=(1, 2, 3),
        values=[q_from_float(0.25), q_from_float(0.50)],
        psi_masks=[0xFFFFFFFF, 0xFFFFFFFF],
        phi_weights=[q_from_float(1.0), q_from_float(1.0)],
        theta_weights=[q_from_float(0.50)],
        constraint=Q16Interval(0, q_from_float(1.0)),
        ledger=ledger,
        tick=7,
    )
    assert trace.activation_raw == q_from_float(0.75)
    assert trace.safe_activation_raw == q_from_float(0.75)
    # U = 3.0, D = 1.5, then V_out clips to raw 65535 (~1.0 raw-output ceiling)
    assert trace.output_raw == Q_OUTPUT_MAX_RAW
    assert len(trace.ledger_hash) == 64
    assert ledger.verify()


def test_sparse_master_step_uses_typed_terms_and_constraint_projection():
    runtime = DrMoagiQ16FieldRuntime(
        DrMoagiQ16Config(
            side=8,
            lambda_inverse_raw=q_from_float(1.0),
            gamma_gain_raw=q_from_float(0.0),
            eta_amplitude_raw=q_from_float(0.0),
            seed=11,
        )
    )
    runtime.load_float({(2, 2, 2): 0.25})
    report = runtime.step_field(
        psi_raw={(2, 2, 2): q_from_float(0.10)},
        phi_kernel={(0, 0, 0): q_from_float(0.50)},
        adaptive_gradient_raw={(2, 2, 2): q_from_float(0.05)},
        constraints={(2, 2, 2): Q16Interval(0, q_from_float(0.50))},
    )
    # current .25 + intent .10 + phi .125 - adaptive .05 + previous .25 = .675 -> .50
    assert q_to_float(report.output[(2, 2, 2)]) == pytest.approx(0.50, abs=2 / 65536)
    assert runtime.ledger.verify()
    assert report.tick == 1


def test_seeded_eta_is_reproducible():
    config = DrMoagiQ16Config(side=4, eta_amplitude_raw=q_from_float(0.1), seed=99)
    a = DrMoagiQ16FieldRuntime(config)
    b = DrMoagiQ16FieldRuntime(config)
    a.load_float({(1, 1, 1): 0.1})
    b.load_float({(1, 1, 1): 0.1})
    assert a.step_field().output == b.step_field().output


def test_temporal_compression_is_preserved_symbolically_not_materialized():
    positive = temporal_compression_law(0.45)
    zero = temporal_compression_law(0.0)
    negative = temporal_compression_law(-0.1)
    assert positive["extended_real_limit"] == "+infinity"
    assert zero["extended_real_limit"] == "1"
    assert negative["extended_real_limit"] == "0"
    assert positive["materialized"] is False
