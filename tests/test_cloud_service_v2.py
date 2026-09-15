from jarvisx.cloud_os import DrMoagiCloudOS
from jarvisx.cloud_service import (
    QVectorConvolutionRequest,
    QVectorFieldOpRequest,
    QVectorOptimizeRequest,
    QVectorPayload,
    QVectorRoundTripRequest,
    create_app,
)


def _endpoint(app, path: str, method: str):
    for route in app.routes:
        if getattr(route, "path", None) == path and method in getattr(route, "methods", set()):
            return route.endpoint
    raise AssertionError(f"missing {method} {path}")


def _vector_payload() -> QVectorPayload:
    return QVectorPayload(
        shape=(2, 2, 2),
        vectors=[
            (0, 0, 0),
            (1, 2, 3),
            (2, 4, 6),
            (3, 6, 9),
            (4, 8, 12),
            (5, 10, 15),
            (6, 12, 18),
            (7, 14, 21),
        ],
    )


def test_service_exposes_qvector_roundtrip_and_exact_optimizer() -> None:
    runtime = DrMoagiCloudOS()
    runtime.register_node("vector-api", max_cells=128)
    app = create_app(runtime)

    roundtrip = _endpoint(app, "/v2/qvector/roundtrip", "POST")
    result = roundtrip(
        QVectorRoundTripRequest(
            request_id="v2-roundtrip",
            field=_vector_payload(),
            latent_shape=(1, 1, 1),
        )
    )
    assert result["status"] == "succeeded"

    optimize = _endpoint(app, "/v2/qvector/auto-optimize", "POST")
    optimized = optimize(
        QVectorOptimizeRequest(
            request_id="v2-optimize",
            field=_vector_payload(),
            complexity_weight=0.01,
            candidates=[(1, 1, 1), (2, 2, 2)],
        )
    )
    assert optimized["status"] == "succeeded"
    assert optimized["result"]["objective_q"]["denominator"] > 0

    get_job = _endpoint(app, "/v2/qvector/jobs/{job_id}", "GET")
    assert get_job(optimized["job_id"])["job_id"] == optimized["job_id"]


def test_service_exposes_deterministic_curl_field_operation() -> None:
    runtime = DrMoagiCloudOS()
    runtime.register_node("vector-api", max_cells=256)
    app = create_app(runtime)
    vectors = []
    for z in range(1):
        for y in range(3):
            for x in range(3):
                vectors.append((-y, x, 0))

    field_op = _endpoint(app, "/v2/qvector/field-op", "POST")
    result = field_op(
        QVectorFieldOpRequest(
            field=QVectorPayload(shape=(3, 3, 1), vectors=vectors),
            operation="curl",
            spacing=1.0,
        )
    )
    center = result["vectors_q16"][4]
    assert center == [0, 0, 2 * 65536]
    assert result["numeric_status"]["divide_by_zero"] is False


def test_service_exposes_fixed_point_convolution() -> None:
    runtime = DrMoagiCloudOS()
    runtime.register_node("vector-api", max_cells=128)
    app = create_app(runtime)
    convolve = _endpoint(app, "/v2/qvector/convolve", "POST")
    result = convolve(
        QVectorConvolutionRequest(
            field=_vector_payload(),
            kernel_shape=(1, 1, 1),
            kernel_weights=[1.0],
        )
    )
    assert result["vectors_q16"][1] == [65536, 2 * 65536, 3 * 65536]
    assert result["numeric_status"]["saturated"] is False
