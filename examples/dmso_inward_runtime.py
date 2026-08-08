"""Minimal deterministic DMSO inward-runtime demonstration."""

from jarvisx.dmso_runtime import DMSOConfig, DMSORuntime


def main() -> None:
    runtime = DMSORuntime(
        DMSOConfig(
            side=8,
            channels=1,
            alpha=0.35,
            promotion_repeats=4,
            max_settle_steps=64,
        )
    )
    runtime.seed(
        {
            (3, 3, 3): 0.25,
            (4, 3, 3): -0.10,
            (3, 4, 3): 0.15,
            (3, 3, 4): 0.05,
        }
    )
    metrics = runtime.step(
        external={(3, 3, 3): 0.5},
        targets={(3, 3, 3): 0.4},
        learn=True,
    )
    print(metrics)
    print("verified:", runtime.verify())
    print("operators:", runtime.operators)


if __name__ == "__main__":
    main()
