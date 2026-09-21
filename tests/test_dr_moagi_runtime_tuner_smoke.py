from jarvisx.dr_moagi_field_runtime import DrMoagiFieldConfig, DrMoagiFieldRuntime, IdentityFieldCodec
from jarvisx.dr_moagi_runtime_tuner import DrMoagiInward3DTuner, InwardTuningPolicy


def test_runtime_tuner_constructs_without_advancing_field_state():
    runtime = DrMoagiFieldRuntime(
        IdentityFieldCodec(),
        DrMoagiFieldConfig(
            side=5,
            alpha=0.0,
            lambda_residual=0.0,
            eta=0.0,
            dt=0.1,
            expand_halo=False,
        ),
    )
    runtime.load({(2, 2, 2): 0.5})
    tuner = DrMoagiInward3DTuner(
        runtime,
        InwardTuningPolicy(
            dt_factors=(0.8,),
            alpha_factors=(),
            lambda_factors=(),
            eta_factors=(),
            prune_increments=(),
        ),
    )

    report = tuner.optimize_once()

    assert runtime.snapshot() == {(2, 2, 2): 0.5}
    assert runtime.cycle == 0
    assert report.baseline_metrics.committed
