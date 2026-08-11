"""Render a small deterministic electromagnetic-photonic reference frame."""

from jarvisx.photonic_rendering import (
    Camera,
    Material,
    PhotonicRenderer,
    PhotonicScene,
    PointEmitter,
    RenderConfig,
    Sphere,
    Spectrum,
    mean_irradiance,
)


def main() -> None:
    scene = PhotonicScene(
        spheres=(
            Sphere(
                center=(0.0, 0.0, -3.5),
                radius=1.2,
                material=Material(reflectance=(0.8, 0.32, 0.14), roughness=0.35),
            ),
        ),
        emitters=(
            PointEmitter(
                position=(-2.5, 3.0, 0.0),
                spectrum=Spectrum.white_reference(),
                intensity=120.0,
            ),
        ),
    )
    camera = Camera(width=32, height=18)
    config = RenderConfig(samples_per_axis=2, tile_edge=8)
    renderer = PhotonicRenderer()

    frame, state, receipt = renderer.cycle(scene, camera, config)
    if frame is None:
        raise SystemExit(receipt.reason)

    print("committed:", receipt.committed)
    print("cycle:", state.cycle)
    print("frame digest:", frame.digest)
    print("mean irradiance:", mean_irradiance(frame))
    print("centre pixel:", frame.at(camera.width // 2, camera.height // 2))


if __name__ == "__main__":
    main()
