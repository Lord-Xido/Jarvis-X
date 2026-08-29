import math

import pytest

from jarvisx.quantum_text import (
    BasisEncoding,
    SparseQuantumState,
    bits_to_utf8,
    index_to_bits,
    text_basis_state,
    utf8_to_bits,
)


def test_ascii_g_has_expected_bits() -> None:
    encoding = BasisEncoding.from_text("G")
    assert encoding.bits == tuple(int(bit) for bit in "01000111")
    assert encoding.basis_index == 0x47
    assert encoding.ket == "|01000111>"
    assert encoding.decode() == "G"


def test_utf8_round_trip_preserves_multibyte_text() -> None:
    text = "Λ Logos — Ἐν ἀρχῇ"
    bits = utf8_to_bits(text)
    assert len(bits) % 8 == 0
    assert bits_to_utf8(bits) == text


def test_index_round_trip_preserves_leading_zeroes() -> None:
    bits = tuple(int(bit) for bit in "00000001")
    assert index_to_bits(1, 8) == bits


def test_text_basis_state_is_deterministic_and_normalized() -> None:
    encoding, state = text_basis_state("In the beginning")
    assert state.qubits == encoding.qubit_count
    assert state.amplitudes == ((encoding.basis_index, 1.0 + 0.0j),)
    assert state.probabilities() == ((encoding.basis_index, 1.0),)


def test_hadamard_creates_balanced_superposition() -> None:
    state = SparseQuantumState.basis((0,)).hadamard(0)
    probabilities = dict(state.probabilities())
    assert probabilities[0] == pytest.approx(0.5)
    assert probabilities[1] == pytest.approx(0.5)
    assert sum(probabilities.values()) == pytest.approx(1.0)


def test_hadamard_is_self_inverse() -> None:
    state = SparseQuantumState.basis((1,))
    restored = state.hadamard(0).hadamard(0)
    assert restored.amplitude(1) == pytest.approx(1.0 + 0.0j)
    assert restored.amplitude(0) == pytest.approx(0.0 + 0.0j)


def test_phase_changes_amplitude_not_measurement_probability() -> None:
    state = SparseQuantumState.basis((0,)).hadamard(0)
    shifted = state.phase(1, math.pi / 2.0)
    assert dict(shifted.probabilities()) == pytest.approx(dict(state.probabilities()))
    assert shifted.amplitude(1) != state.amplitude(1)


def test_orthogonal_basis_states_have_zero_inner_product() -> None:
    zero = SparseQuantumState.basis((0,))
    one = SparseQuantumState.basis((1,))
    assert zero.inner_product(one) == pytest.approx(0.0 + 0.0j)


def test_invalid_state_normalization_is_rejected() -> None:
    with pytest.raises(ValueError, match="normalized"):
        SparseQuantumState(1, ((0, 0.5 + 0.0j),))


def test_empty_text_is_rejected() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        utf8_to_bits("")
