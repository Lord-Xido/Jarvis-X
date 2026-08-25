"""Small deterministic KTME dual-synchronous processor demonstration."""

from jarvisx.equation_kinetics import (
    EquationKineticConfig,
    EquationKineticState,
    dual_synchronous_step,
)


def residual(x):
    return x


def jacobian(x):
    del x
    return ((1.0,),)


def main() -> None:
    state_a = EquationKineticState(position=(-2.0,), velocity=(0.0,))
    state_b = EquationKineticState(position=(3.0,), velocity=(0.0,))
    config = EquationKineticConfig(
        dt=0.05,
        mass=1.0,
        damping=0.35,
        coupling=0.20,
        memory_retention=0.80,
        memory_gain=0.03,
        max_speed=10.0,
    )

    print("step        x_A        x_B     disagreement      energy")
    for _ in range(40):
        result = dual_synchronous_step(
            state_a,
            state_b,
            residual,
            jacobian,
            residual,
            jacobian,
            config,
            validator=lambda candidate: abs(candidate.position[0]) <= 10.0,
        )
        if not result.committed:
            raise RuntimeError("joint validator rejected the KTME candidate")
        state_a, state_b = result.state_a, result.state_b
        print(
            f"{state_a.step:4d}  "
            f"{state_a.position[0]:10.6f}  "
            f"{state_b.position[0]:10.6f}  "
            f"{result.disagreement_after:14.8f}  "
            f"{result.total_energy_after:10.6f}"
        )


if __name__ == "__main__":
    main()
