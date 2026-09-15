from jarvisx.qvector3d import QVectorField3D
from jarvisx.qvector_cloud import DrMoagiQVectorCloudEngine3D


def test_optimizer_exposes_exact_common_denominator_objective() -> None:
    field = QVectorField3D.from_vectors(
        [
            (0, 0, 0),
            (1, 2, 3),
            (2, 4, 6),
            (3, 6, 9),
            (4, 8, 12),
            (5, 10, 15),
            (6, 12, 18),
            (7, 14, 21),
        ],
        (2, 2, 2),
    )
    engine = DrMoagiQVectorCloudEngine3D()
    job = engine.auto_optimize(
        field,
        request_id="exact-objective",
        complexity_weight=0.01,
        candidates=[(1, 1, 1), (2, 2, 2)],
    )
    assert job.result is not None
    objective_q = job.result["objective_q"]
    assert isinstance(objective_q, dict)
    assert isinstance(objective_q["numerator"], int)
    assert isinstance(objective_q["denominator"], int)
    assert objective_q["denominator"] > 0
    candidates = job.result["candidates"]
    assert isinstance(candidates, list)
    ranked = [candidate["objective_q"]["numerator"] for candidate in candidates]
    selected = objective_q["numerator"]
    assert selected == min(ranked)


def test_optimizer_is_deterministic_for_same_field_and_quantized_weight() -> None:
    field = QVectorField3D.from_vectors([(1, 2, 3)] * 64, (4, 4, 4))
    numerators = []
    shapes = []
    for index in range(2):
        engine = DrMoagiQVectorCloudEngine3D()
        job = engine.auto_optimize(
            field,
            request_id=f"deterministic-{index}",
            complexity_weight=0.012345,
        )
        assert job.result is not None
        objective_q = job.result["objective_q"]
        assert isinstance(objective_q, dict)
        numerators.append(objective_q["numerator"])
        shapes.append(job.result["selected_latent_shape"])
    assert numerators[0] == numerators[1]
    assert shapes[0] == shapes[1]
