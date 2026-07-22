from jarvisx.inward_optimizer import InwardOptimizer, MechanicsState, Telemetry


def evaluate(state: MechanicsState) -> Telemetry:
    """Synthetic Level-0 execution measurement for a deterministic demo."""

    target_learning_rate = 0.0015
    target_layers = 3
    target_momentum = 0.85

    task_loss = (
        (state.hyper.learning_rate - target_learning_rate) ** 2 * 1_000_000.0
        + (state.architecture.layers - target_layers) ** 2 * 0.05
        + (state.rule.momentum - target_momentum) ** 2
        + 0.1
    )
    latency_ms = 5.0 + state.architecture.layers * 1.5 + state.architecture.experts * 0.4
    memory_mb = 128.0 + state.architecture.latent_dim * state.architecture.experts * 0.5

    return Telemetry(
        task_loss=task_loss,
        latency_ms=latency_ms,
        memory_mb=memory_mb,
        semantic_distance=0.0,
        gradient_norm=1.0,
    )


def main() -> None:
    optimizer = InwardOptimizer()

    for cycle in range(4):
        result = optimizer.optimize_once(evaluate)
        state = result.active_state
        telemetry = evaluate(state)
        print(
            "cycle={cycle} committed={committed} transform={transform} "
            "version={version} score={score:.6f}".format(
                cycle=cycle,
                committed=result.committed,
                transform=result.transformation,
                version=state.version,
                score=telemetry.score(optimizer.weights),
            )
        )
        print(
            "  lr={lr:.8f} layers={layers} experts={experts} "
            "momentum={momentum:.4f}".format(
                lr=state.hyper.learning_rate,
                layers=state.architecture.layers,
                experts=state.architecture.experts,
                momentum=state.rule.momentum,
            )
        )

    print("journal_entries={}".format(len(optimizer.journal)))


if __name__ == "__main__":
    main()
