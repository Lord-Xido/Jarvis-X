from __future__ import annotations

import os
from pathlib import Path

import pytest

from jarvisx.kinetic3d import Kinetic3DRuntime
from jarvisx.kinetic3d_backend import (
    BackendUnavailable,
    NativeBackend,
    ReferenceBackend,
    available_backends,
    resolve_backend,
)


def _native_from_env() -> NativeBackend:
    path = os.environ.get("JARVISX_KINETIC3D_NATIVE_LIB")
    if not path or not Path(path).is_file():
        pytest.skip("native kinetic backend is not compiled in this test environment")
    return NativeBackend(path)


def test_auto_falls_back_to_reference_without_native_library(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("JARVISX_KINETIC3D_NATIVE_LIB", raising=False)
    assert resolve_backend("auto").name == "cpu-reference"
    assert available_backends() == ("cpu-reference",)


def test_explicit_missing_native_backend_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("JARVISX_KINETIC3D_NATIVE_LIB", raising=False)
    with pytest.raises(BackendUnavailable, match="native-cpu backend requested"):
        resolve_backend("native-cpu")


def test_reference_backend_semantics() -> None:
    backend = ReferenceBackend()
    result = backend.step(
        [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        [0.0] * 8,
        (2, 2, 2),
        active_threshold=0.0,
        coarse_factor=2,
        refine_threshold=10.0,
    )
    assert result.active_indices == tuple(range(1, 8))
    assert result.coarse_values == (((0, 0, 0), 4.0),)
    assert result.reconstructed == pytest.approx((0.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0))


def test_native_backend_matches_reference_for_sparse_change() -> None:
    native = _native_from_env()
    reference = ReferenceBackend()
    current = [1.0] * 64
    prediction = current.copy()
    current[3] = 8.0
    current[42] = -5.0
    kwargs = {
        "active_threshold": 0.1,
        "coarse_factor": 2,
        "refine_threshold": 0.0,
    }

    expected = reference.step(current, prediction, (4, 4, 4), **kwargs)
    actual = native.step(current, prediction, (4, 4, 4), **kwargs)

    assert actual.active_indices == expected.active_indices
    assert [block for block, _ in actual.coarse_values] == [
        block for block, _ in expected.coarse_values
    ]
    assert [value for _, value in actual.coarse_values] == pytest.approx(
        [value for _, value in expected.coarse_values]
    )
    assert [index for index, _ in actual.fine_corrections] == [
        index for index, _ in expected.fine_corrections
    ]
    assert [value for _, value in actual.fine_corrections] == pytest.approx(
        [value for _, value in expected.fine_corrections]
    )
    assert actual.residual == pytest.approx(expected.residual)
    assert actual.reconstructed == pytest.approx(expected.reconstructed)


def test_runtime_reports_native_backend_when_available() -> None:
    _native_from_env()
    runtime = Kinetic3DRuntime(backend="native-cpu")
    result = runtime.execute(
        [2.0] * 8,
        (2, 2, 2),
        active_threshold=0.0,
        coarse_factor=2,
        refine_threshold=0.0,
        tolerance=0.0,
    )

    assert result.verification.passed
    assert result.telemetry.backend == "native-cpu"
    assert result.schedule[0].resource == "native-cpu:0"
