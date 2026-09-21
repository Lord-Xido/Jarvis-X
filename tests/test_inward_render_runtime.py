import math

from jarvisx.inward_render_runtime import (
    BytecodeProgram,
    EngineState,
    Instruction,
    InwardOptimizer,
    OP_HALT,
    OP_SCALE,
    OP_TRANSLATE,
    RendererConfig,
    demo_program,
    execute_program,
    forward,
    write_ppm,
)


def tiny_config() -> RendererConfig:
    return RendererConfig(
        grid_size=8,
        latent_grid=2,
        image_width=6,
        image_height=4,
        max_ray_steps=24,
        ray_target_steps=10.0,
    )


def test_bytecode_round_trip_and_vm_transform() -> None:
    program = BytecodeProgram(
        (
            Instruction(OP_TRANSLATE, (0.5, -0.25, 0.125)),
            Instruction(OP_SCALE, (1.25,)),
            Instruction(OP_HALT),
        )
    )
    decoded = BytecodeProgram.from_bytes(program.to_bytes())
    assert decoded.to_bytes() == program.to_bytes()
    vm = execute_program(decoded)
    assert math.isclose(vm.uniform_scale, 1.25, rel_tol=1e-6)
    assert math.isclose(vm.matrix[0][3], 0.5, rel_tol=1e-6)


def test_forward_pipeline_produces_finite_loss_and_frame() -> None:
    config = tiny_config()
    result = forward(EngineState(program=demo_program()), config)
    assert len(result.raw_volume.values) == config.grid_size**3
    assert len(result.latent) == config.latent_grid**3
    assert len(result.image) == config.image_height
    assert len(result.image[0]) == config.image_width
    assert math.isfinite(result.loss.total)
    assert result.loss.total >= 0.0
    assert result.telemetry.rays == config.image_width * config.image_height


def test_sdf_has_inside_and_outside_samples() -> None:
    result = forward(EngineState(program=demo_program()), tiny_config())
    assert min(result.raw_volume.values) < 0.0
    assert max(result.raw_volume.values) > 0.0


def test_inward_optimizer_never_regresses_authoritative_loss() -> None:
    config = tiny_config()
    state = EngineState(program=demo_program())
    before = forward(state, config).loss.total
    outcome = InwardOptimizer(config).step(state)
    after = forward(outcome.state, config).loss.total
    assert after <= before + config.non_regression_tolerance
    assert outcome.receipt.decision in {"commit", "rollback"}


def test_ppm_writer_emits_portable_frame(tmp_path) -> None:
    result = forward(EngineState(program=demo_program()), tiny_config())
    target = tmp_path / "frame.ppm"
    write_ppm(result.image, target)
    text = target.read_text(encoding="ascii")
    assert text.startswith("P3\n6 4\n255\n")
