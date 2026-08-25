import pytest

z3 = pytest.importorskip("z3")

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


def test_non_equivalent_programs():
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
