"""Benchmark a promoted DMSO operator as primitive versus fused bytecode execution."""

from dataclasses import asdict
import json

from jarvisx.dmso_bytecode import CellExecutionContext, benchmark_operator
from jarvisx.dmso_runtime import DMSOConfig, DMSOParameters, DMSORuntime


def main() -> None:
    runtime = DMSORuntime(DMSOConfig(side=4, promotion_repeats=1))
    runtime.seed({(1, 1, 1): 0.1})
    runtime.step()
    operator = runtime.operators[0]

    context = CellExecutionContext(
        current=(0.25, -0.5),
        neighbour_mean=(0.1, 0.2),
        projected=(0.25, -0.5),
        stimulus=(0.7, -0.1),
        alpha=0.25,
    )
    result = benchmark_operator(
        operator,
        context,
        DMSOParameters(),
        repetitions=10_000,
        samples=7,
    )
    print(json.dumps(asdict(result), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
