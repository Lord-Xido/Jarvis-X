import pytest

from jarvisx.verification import check_equivalence_bounded


def test_equivalent_simple_programs():
    p1 = """
    SET A 1
    SET B 2
    ADD C A B
    HALT
    """
    p2 = """
    SET A 1
    SET B 2
    ADD C A B
    HALT
    """
    eq, cex = check_equivalence_bounded(p1, p2, bound=8, timeout_ms=2000)
    assert eq is True
    assert cex is None


def test_equivalent_programs_share_symbolic_initial_state():
    p1 = """
    ADD C A B
    HALT
    """
    p2 = """
    NOP
    ADD C A B
    HALT
    """
    eq, cex = check_equivalence_bounded(p1, p2, bound=8, timeout_ms=2000)
    assert eq is True
    assert cex is None


def test_non_equivalent_programs_return_shared_initial_counterexample():
    p1 = """
    ADD C A B
    HALT
    """
    p2 = """
    SUB C A B
    HALT
    """
    eq, cex = check_equivalence_bounded(p1, p2, bound=8, timeout_ms=2000)
    assert eq is False
    assert isinstance(cex, dict)
    assert set(cex) == {"A", "B", "C"}


def test_non_equivalent_constants_are_rejected():
    p1 = """
    SET A 1
    SET B 2
    ADD C A B
    HALT
    """
    p2 = """
    SET A 1
    SET B 3
    ADD C A B
    HALT
    """
    eq, cex = check_equivalence_bounded(p1, p2, bound=8, timeout_ms=2000)
    assert eq is False
    assert isinstance(cex, dict)


def test_unsupported_instruction_fails_closed():
    with pytest.raises(ValueError, match="unsupported instruction"):
        check_equivalence_bounded("MUL A B C", "NOP", bound=4)


def test_bound_must_be_positive():
    with pytest.raises(ValueError, match="bound must be positive"):
        check_equivalence_bounded("NOP", "NOP", bound=0)
