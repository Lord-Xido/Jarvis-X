from __future__ import annotations

import pytest

from jarvisx.omega3_electronic_logic import (
    Q15_MAX,
    Omega3ElectronicCore,
    approve_lambda,
    hamming_distance,
    linear_address,
    pack_xyz,
    select_word,
    unpack_xyz,
)


def test_xyz_round_trip_and_linear_address():
    packed = pack_xyz(999, 500, 1)
    assert unpack_xyz(packed) == (999, 500, 1)
    assert linear_address(999, 500, 1) == 999 + 1000 * (500 + 1000)


def test_out_of_domain_coordinate_is_rejected():
    with pytest.raises(ValueError):
        pack_xyz(1000, 0, 0)


def test_lambda_gate_is_fail_closed():
    assert approve_lambda(0xFF)
    assert not approve_lambda(0xFE)
    assert select_word(0xAA, 0x55, 0xFE) == 0xAA
    assert select_word(0xAA, 0x55, 0xFF) == 0x55


def test_hamming_distance_uses_xor_population_count():
    assert hamming_distance(0b1010, 0b0011) == 2
    assert hamming_distance(0xFFFFFFFFFFFFFFFF, 0) == 64


def test_rejected_candidate_rolls_back_word_and_omega():
    core = Omega3ElectronicCore(initial_word=0x10, omega_q15=1024)
    report = core.step(
        candidate_word=0x20,
        lambda_mask=0x7F,
        error_q15=4096,
        rho_q15=Q15_MAX,
        gain_q15=8192,
    )

    assert not report.committed
    assert core.word == 0x10
    assert core.omega_q15 == 1024
    assert core.cycle == 1


def test_approved_candidate_commits_word_and_omega():
    core = Omega3ElectronicCore(initial_word=0, omega_q15=0)
    report = core.step(
        candidate_word=0xF0,
        lambda_mask=0xFF,
        error_q15=8192,
        rho_q15=Q15_MAX,
        gain_q15=16384,
    )

    assert report.committed
    assert core.word == 0xF0
    assert core.omega_q15 > 0


def test_bit_identical_candidate_is_converged_at_zero_threshold():
    core = Omega3ElectronicCore(initial_word=0x1234)
    report = core.step(
        candidate_word=0x1234,
        lambda_mask=0xFF,
        convergence_threshold=0,
    )

    assert report.hamming_delta == 0
    assert report.converged
