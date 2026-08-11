import math

import pytest

from jarvisx.photonic_rendering import (
    Camera,
    Material,
    PhotonicRenderer,
    PhotonicScene,
    PointEmitter,
    RenderConfig,
    Sphere,
    Spectrum,
    frame_rgb_bytes,
    iter_lattice_samples,
    mean_irradiance,
    partition_tiles,
    pixel_to_lattice_coordinate,
    sensor_response,
    wavelength_to_linear_rgb,
)


def reference_scene(light_z: float = 0.0, intensity: float = 120.0) -> PhotonicScene:
    return PhotonicScene(
        spheres=(
            Sphere(
                center=(0.0, 0.0, -3.5),
                radius=1.2,
                material=Material(reflectance=(0.8, 0.32, 0.14), roughness=0.35),
            ),
        ),
        emitters=(
            PointEmitter(
                position=(-2.5, 3.0, light_z),
                spectrum=Spectrum.white_reference(),
                intensity=intensity,
            ),
        ),
    )


def test_wavelength_and_sensor_response_are_bounded() -> None:
    for wavelength in (380.0, 455.0, 540.0, 610.0, 700.0):
        colour = wavelength_to_linear_rgb(wavelength)
        response = sensor_response(wavelength)
        assert all(0.0 <= value <= 1.0 for value in colour)
        assert all(0.0 <= value <= 1.0 for value in response)
        assert max(colour) == pytest.approx(1.0)


@pytest.mark.parametrize("wavelength", [379.99, 700.01, float("nan")])
def test_wavelength_validation_fails_closed(wavelength: float) -> None:
    with pytest.raises(ValueError):
        wavelength_to_linear_rgb(wavelength)


def test_tile_partition_is_complete_and_deterministic() -> None:
    first = partition_tiles(17, 9, 8)
    second = partition_tiles(17, 9, 8)
    assert first == second
    assert sum(tile.pixel_count for tile in first) == 17 * 9
    assert [tile.tile_id for tile in first] == list(range(len(first)))


def test_frame_render_is_deterministic_and_quantized() -> None:
    camera = Camera(width=12, height=8)
    config = RenderConfig(samples_per_axis=2, tile_edge=5, max_pixels=200)
    renderer = PhotonicRenderer()

    first = renderer.render(reference_scene(), camera, config)
    second = renderer.render(reference_scene(), camera, config)

    assert first.digest == second.digest
    assert frame_rgb_bytes(first) == frame_rgb_bytes(second)
    assert len(first.pixels) == 96
    assert len(frame_rgb_bytes(first)) == 96 * 3
    assert all(
        0 <= channel <= 255 for pixel in first.pixels for channel in pixel.quantized_rgb
    )
    assert mean_irradiance(first) > 0.0


def test_inverse_square_attenuation_reduces_mean_irradiance() -> None:
    camera = Camera(width=9, height=7)
    config = RenderConfig(samples_per_axis=1, max_pixels=100)
    renderer = PhotonicRenderer()

    near = renderer.render(reference_scene(light_z=0.0), camera, config)
    far = renderer.render(reference_scene(light_z=8.0), camera, config)

    assert mean_irradiance(near) > mean_irradiance(far)


def test_transaction_commit_and_replay() -> None:
    camera = Camera(width=8, height=6)
    config = RenderConfig(samples_per_axis=1, tile_edge=4, max_pixels=100)
    first_renderer = PhotonicRenderer()
    second_renderer = PhotonicRenderer()

    first_frame, first_state, first_receipt = first_renderer.cycle(
        reference_scene(), camera, config
    )
    second_frame, second_state, second_receipt = second_renderer.cycle(
        reference_scene(), camera, config
    )

    assert first_receipt.committed
    assert second_receipt.committed
    assert first_frame is not None and second_frame is not None
    assert first_frame.digest == second_frame.digest
    assert first_state == second_state
    assert first_receipt.state_digest == second_receipt.state_digest
    assert first_state.cycle == 1


def test_transaction_rolls_back_on_resource_bound() -> None:
    camera = Camera(width=11, height=10)
    renderer = PhotonicRenderer()
    previous = renderer.state

    frame, state, receipt = renderer.cycle(
        reference_scene(),
        camera,
        RenderConfig(max_pixels=100),
    )

    assert frame is None
    assert not receipt.committed
    assert receipt.reason == "frame exceeds max_pixels"
    assert state == previous
    assert renderer.state == previous


def test_pixel_to_lattice_coordinate_is_bounded() -> None:
    assert pixel_to_lattice_coordinate(0, 0, 0.0, 100, 50) == (5, 10, 0)
    x, y, z = pixel_to_lattice_coordinate(99, 49, 10.0, 100, 50)
    assert 0 <= x < 1000
    assert 0 <= y < 1000
    assert 0 <= z < 1000
    assert x == 995
    assert y == 990
    assert z > 900


def test_lattice_projection_covers_every_pixel_once() -> None:
    renderer = PhotonicRenderer()
    frame = renderer.render(
        reference_scene(),
        Camera(width=7, height=5),
        RenderConfig(max_pixels=100),
    )
    samples = tuple(iter_lattice_samples(frame))
    assert len(samples) == 35
    assert len({coordinate for coordinate, _ in samples}) == 35


def test_camera_and_config_validation() -> None:
    with pytest.raises(ValueError, match="parallel"):
        Camera(
            width=2,
            height=2,
            forward=(0.0, 1.0, 0.0),
            up=(0.0, 1.0, 0.0),
        )
    with pytest.raises(ValueError, match="samples_per_axis"):
        RenderConfig(samples_per_axis=9)
    with pytest.raises(ValueError, match="positive"):
        Sphere(center=(0.0, 0.0, 0.0), radius=0.0)


def test_background_only_pixels_remain_finite() -> None:
    scene = PhotonicScene(
        spheres=(),
        emitters=(
            PointEmitter(
                position=(0.0, 2.0, 0.0),
                spectrum=Spectrum.monochromatic(540.0),
                intensity=1.0,
            ),
        ),
        background=(0.02, 0.03, 0.04),
    )
    frame = PhotonicRenderer().render(
        scene,
        Camera(width=3, height=2),
        RenderConfig(),
    )
    assert all(math.isfinite(value) for pixel in frame.pixels for value in pixel.linear_rgb)
    assert len({pixel.quantized_rgb for pixel in frame.pixels}) == 1
