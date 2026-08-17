"""Minimal executable demonstration of Xi^recur_Phi_3D.

Run from a source checkout with:
    python apps/dr_moagi_codex_demo.py
"""

from __future__ import annotations

import json

from jarvisx.dr_moagi_codex import DrMoagiCodex, DrMoagiCodexConfig, l2_norm


COORDS = (
    (0.0, 0.0, 0.0),
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
)


def encoder(scene):
    return tuple(float(scene.get(coordinate, 0.0)) for coordinate in COORDS)


def inward(latent, time_index, condition):
    # Deterministic contractive reference map. Replace with a learned operator
    # only after measuring/validating its contraction and resource behavior.
    return tuple(0.5 * value for value in latent)


def decoder(latent):
    return {coordinate: float(value) for coordinate, value in zip(COORDS, latent)}


def source_mapper(latent):
    return {coordinate: float(value) for coordinate, value in zip(COORDS, latent)}


def main() -> None:
    scene = {
        (0.0, 0.0, 0.0): 0.8,
        (1.0, 0.0, 0.0): -0.4,
        (0.0, 1.0, 0.0): 0.2,
        (0.0, 0.0, 1.0): 0.1,
    }
    codex = DrMoagiCodex(
        encoder=encoder,
        decoder=decoder,
        inward_operator=inward,
        source_mapper=source_mapper,
        config=DrMoagiCodexConfig(
            lambda_max=1.0,
            fixed_point_tolerance=1.0e-6,
            max_fixed_point_iterations=64,
            claimed_contraction=0.5,
            gamma=1.0,
            beta=0.0,
            eta_theta=0.1,
            wave_number=0.0,
            green_softening=0.25,
            virtual_depth_label="1000000^1000000",
        ),
    )
    result = codex.execute(
        scene,
        theta=(1.0, 1.0),
        theta_gradient=(0.1, -0.1),
    )
    payload = {
        "virtual_depth": result.virtual_depth_label,
        "actual_fixed_point_iterations": result.fixed_point.iterations,
        "converged": result.fixed_point.converged,
        "final_delta": result.fixed_point.final_delta,
        "projected_norm": l2_norm(result.projected_latent),
        "theta_after": result.theta_after,
        "permeation": {
            str(coordinate): {"real": value.real, "imag": value.imag}
            for coordinate, value in result.permeation_field.items()
        },
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
