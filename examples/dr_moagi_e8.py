from __future__ import annotations

import random

from jarvisx.dr_moagi_e8 import DrMoagiE8System, random_genome


def main() -> None:
    rng = random.Random(42)
    system = DrMoagiE8System(k=48, resolution=18)
    population = tuple(random_genome(rng) for _ in range(24))

    champion = max(
        population,
        key=lambda genome: system.forward(genome, coherence=0.90, rng=random.Random(7)).fitness,
    )
    result = system.forward(champion, coherence=0.90, rng=random.Random(11))

    print("Dr. Moagi Equation System E8")
    print(f"points={len(result.points)}")
    print(f"active_latent_nodes={result.latent.active_nodes}/{len(system.nodes)}")
    print(f"entropy={result.latent.entropy:.6f}")
    print(f"loss={result.loss:.6f}")
    print(f"fitness={result.fitness:.6f}")
    print(f"epsilon={result.epsilon:.6f}")
    print(f"lambda={result.governor.lambda_value:.6f}")
    print(f"v_clock={result.governor.clock_velocity:.6f}")

    next_population = system.evolve_generation(population, rng=rng)
    print(f"next_generation={len(next_population)} genomes")


if __name__ == "__main__":
    main()
