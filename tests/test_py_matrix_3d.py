import importlib.util
import math
import sys
from pathlib import Path

MODULE = Path(__file__).resolve().parents[1] / "src" / "jarvisx" / "py_matrix_3d.py"
spec = importlib.util.spec_from_file_location("py_matrix_3d", MODULE)
assert spec and spec.loader
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)


def test_scale_and_cluster_factorization() -> None:
    assert m.TOTAL_LOC == 1_000_000
    assert m.CLUSTER_COUNT == 250
    assert m.LOC_PER_CLUSTER == 4_000
    assert m.GRID_X * m.GRID_Y * m.GRID_Z == 4_000
    assert len(m.cluster_transforms()) == 250


def test_line_mapping_is_arithmetic_and_exact() -> None:
    first = m.locate_line(1)
    assert (first.cluster, first.local_index, first.cell_x, first.cell_y, first.cell_z) == (0, 0, 0, 0, 0)

    sample = m.locate_line(48_201)
    assert (sample.cluster, sample.local_index) == (12, 200)
    assert (sample.cell_x, sample.cell_y, sample.cell_z) == (0, 10, 0)

    last = m.locate_line(1_000_000)
    assert (last.cluster, last.local_index) == (249, 3999)
    assert (last.cell_x, last.cell_y, last.cell_z) == (19, 19, 9)


def test_pulse_is_one_hertz_and_travels() -> None:
    for t in (0.0, 0.125, 0.37, 1.9):
        assert math.isclose(m.pulse(t), m.pulse(t + 1.0), abs_tol=1e-12)
    assert not math.isclose(m.travelling_pulse(0.0, 0), m.travelling_pulse(0.0, 8))


def test_coherence_and_frame_budget_are_explicit_gates() -> None:
    assert m.coherence_pass(0.9998)
    assert not m.coherence_pass(0.99979)
    assert m.frame_budget_pass(16.0)
    assert not m.frame_budget_pass(17.0)


ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "apps" / "py-matrix-3d" / "index.html"
README = APP.parent / "README.md"


def test_py_matrix_surface_is_self_contained_instanced_and_bounded() -> None:
    html = APP.read_text(encoding="utf-8")
    readme = README.read_text(encoding="utf-8")

    for marker in (
        "PY-MATRIX 3D :: 1M LOC MEGA-CODE ENGINE",
        "TOTAL_LOC=1000000",
        "CLUSTER_COUNT=250",
        "LOC_PER_CLUSTER=4000",
        "PULSE_HZ=1.0",
        "COHERENCE_TARGET=0.9998",
        "getContext('webgl2'",
        "drawArraysInstanced",
        "20×20×10",
        "cluster=(LOC−1)//4000",
    ):
        assert marker in html

    assert "https://" not in html
    assert "http://" not in html
    assert "60 FPS" in readme and "not" in readme
    assert "does not mutate authoritative Jarvis-X state" in readme
    assert "jarvisx.system_runtime" in readme
