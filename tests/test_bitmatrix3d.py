import pytest

from jarvisx.bitmatrix3d import (
    BitMatrix3D,
    Coordinate3D,
    Dimensions3D,
    coordinate_to_index,
    decode_bytes,
    decode_text,
    encode_bytes,
    encode_text,
    from_active_coordinates,
    index_to_coordinate,
    normalized_scene_coordinate,
    verify_round_trip,
)


def test_dimensions_capacity_and_validation() -> None:
    dimensions = Dimensions3D(8, 8, 2)
    assert dimensions.cells == 128
    assert dimensions.payload_capacity_bytes == 12

    with pytest.raises(ValueError):
        Dimensions3D(0, 2, 2)
    with pytest.raises(TypeError):
        Dimensions3D(True, 2, 2)


def test_index_coordinate_bijection_x_fastest() -> None:
    dimensions = Dimensions3D(4, 3, 2)
    expected = {
        0: Coordinate3D(0, 0, 0),
        1: Coordinate3D(1, 0, 0),
        3: Coordinate3D(3, 0, 0),
        4: Coordinate3D(0, 1, 0),
        11: Coordinate3D(3, 2, 0),
        12: Coordinate3D(0, 0, 1),
        23: Coordinate3D(3, 2, 1),
    }
    for index, coordinate in expected.items():
        assert index_to_coordinate(index, dimensions) == coordinate
        assert coordinate_to_index(coordinate, dimensions) == index

    for index in range(dimensions.cells):
        coordinate = index_to_coordinate(index, dimensions)
        assert coordinate_to_index(coordinate, dimensions) == index


def test_coordinate_bounds_are_fail_closed() -> None:
    dimensions = Dimensions3D(2, 2, 2)
    with pytest.raises(IndexError):
        coordinate_to_index(Coordinate3D(2, 0, 0), dimensions)
    with pytest.raises(IndexError):
        index_to_coordinate(8, dimensions)
    with pytest.raises(TypeError):
        index_to_coordinate(True, dimensions)


def test_ascii_round_trip_and_header_layout() -> None:
    dimensions = Dimensions3D(8, 8, 2)
    matrix = encode_text("JARVIS", dimensions)

    # Six bytes encoded as uint32 big-endian: 00000000 00000000 00000000 00000110
    assert matrix.bits[:24] == (False,) * 24
    assert matrix.bits[24:32] == (
        False,
        False,
        False,
        False,
        False,
        True,
        True,
        False,
    )
    assert decode_text(matrix) == "JARVIS"


def test_utf8_round_trip() -> None:
    dimensions = Dimensions3D(16, 16, 4)
    text = "Dr Moagi — 3D Δ Ξ 🚀"
    matrix = encode_text(text, dimensions)
    assert decode_text(matrix) == text


def test_arbitrary_bytes_round_trip() -> None:
    dimensions = Dimensions3D(16, 16, 2)
    payload = bytes(range(32)) + b"\x00\xff\x80"
    matrix = encode_bytes(payload, dimensions)
    decoded = decode_bytes(matrix)
    assert decoded.payload == payload
    assert decoded.declared_length == len(payload)
    assert decoded.payload_capacity_bytes == dimensions.payload_capacity_bytes
    assert decoded.padding_bits == dimensions.cells - 32 - len(payload) * 8


def test_capacity_is_enforced() -> None:
    dimensions = Dimensions3D(8, 8, 1)  # 64 bits => 4 payload bytes
    assert dimensions.payload_capacity_bytes == 4
    encode_bytes(b"1234", dimensions)
    with pytest.raises(ValueError, match="exceeds lattice capacity"):
        encode_bytes(b"12345", dimensions)

    with pytest.raises(ValueError, match="at least 32 cells"):
        encode_bytes(b"", Dimensions3D(3, 3, 3))


def test_sparse_active_transport_is_lossless() -> None:
    dimensions = Dimensions3D(16, 8, 2)
    payload = b"Bit Matrix 3D"
    matrix = encode_bytes(payload, dimensions)
    active = tuple(matrix.active_coordinates())
    rebuilt = from_active_coordinates(dimensions, active)

    assert rebuilt == matrix
    assert decode_bytes(rebuilt).payload == payload
    assert verify_round_trip(payload, dimensions)


def test_duplicate_sparse_coordinates_are_idempotent() -> None:
    dimensions = Dimensions3D(4, 4, 4)
    coordinate = Coordinate3D(1, 2, 3)
    matrix = from_active_coordinates(dimensions, [coordinate, coordinate])
    assert matrix.active_count == 1
    assert matrix.bit_at(coordinate)


def test_declared_length_larger_than_capacity_is_rejected() -> None:
    dimensions = Dimensions3D(8, 8, 1)
    # uint32 header declares 5 bytes, but this lattice can hold only 4.
    bits = [False] * dimensions.cells
    bits[29] = True
    bits[31] = True
    matrix = BitMatrix3D(dimensions, tuple(bits))
    with pytest.raises(ValueError, match="declared payload length exceeds"):
        decode_bytes(matrix)


def test_non_zero_padding_is_rejected_by_default() -> None:
    dimensions = Dimensions3D(8, 8, 2)
    matrix = encode_bytes(b"A", dimensions)
    bits = list(matrix.bits)
    bits[-1] = True
    noncanonical = BitMatrix3D(dimensions, tuple(bits))

    with pytest.raises(ValueError, match="non-zero bits"):
        decode_bytes(noncanonical)
    assert decode_bytes(noncanonical, require_zero_padding=False).payload == b"A"


def test_bitmatrix_shape_and_type_validation() -> None:
    dimensions = Dimensions3D(2, 2, 8)
    with pytest.raises(ValueError, match="bit count"):
        BitMatrix3D(dimensions, (False,))
    with pytest.raises(TypeError, match="bool"):
        BitMatrix3D(dimensions, tuple([False] * 31 + [1]))  # type: ignore[list-item]


def test_normalized_scene_coordinate_centres_geometry() -> None:
    dimensions = Dimensions3D(3, 3, 3)
    assert normalized_scene_coordinate(Coordinate3D(1, 1, 1), dimensions) == (0.0, 0.0, 0.0)
    assert normalized_scene_coordinate(Coordinate3D(2, 1, 0), dimensions, scale=2.0) == (
        2.0,
        0.0,
        -2.0,
    )
