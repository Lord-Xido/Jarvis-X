from jarvisx.exynos3d_bit_fabric import (
    DOMAIN_ANCHORS,
    TILE_SIDE,
    UINT32_MASK,
    VIRTUAL_CELLS,
    VIRTUAL_TILES,
    ProcessingDomain,
    bit_residual,
    branchless_commit,
    coordinate_from_address,
    coordinate_from_tile,
    decode_transport_word,
    encode_transport_word,
    linear_address,
    procedural_word,
    route_domain,
    spatial_address,
    transition,
)


def test_billion_coordinate_address_roundtrip() -> None:
    assert VIRTUAL_CELLS == 1_000_000_000
    cases = (
        (0, 0, 0),
        (999, 999, 999),
        (1, 2, 3),
        (500, 500, 500),
        (123, 456, 789),
    )
    for coordinate in cases:
        address = linear_address(coordinate)
        assert 0 <= address < VIRTUAL_CELLS
        assert coordinate_from_address(address) == coordinate


def test_ten_cubed_tile_mapping_is_exact() -> None:
    assert TILE_SIDE == 10
    assert VIRTUAL_TILES == 1_000_000

    for coordinate in ((0, 0, 0), (9, 9, 9), (10, 10, 10), (999, 999, 999), (321, 654, 987)):
        mapping = spatial_address(coordinate)
        assert 0 <= mapping.tile_id < VIRTUAL_TILES
        assert 0 <= mapping.local_id < 1000
        assert coordinate_from_tile(mapping.tile_id, mapping.local_id) == coordinate


def test_domain_anchors_route_to_their_own_domains() -> None:
    for domain, coordinate in DOMAIN_ANCHORS.items():
        assert route_domain(coordinate) == domain

    assert route_domain((500, 500, 500)) == ProcessingDomain.NPU


def test_procedural_words_are_deterministic_and_coordinate_sensitive() -> None:
    a = procedural_word((12, 34, 56))
    b = procedural_word((12, 34, 56))
    c = procedural_word((12, 34, 57))

    assert a == b
    assert a != c
    assert 0 <= a <= UINT32_MASK


def test_transport_codec_roundtrips_without_claiming_compression() -> None:
    coordinates = (
        (120, 500, 500),
        (300, 500, 500),
        (500, 500, 500),
        (680, 350, 500),
        (680, 650, 500),
        (500, 250, 500),
        (500, 750, 500),
        (880, 500, 500),
    )
    words = (0, 1, 0x12345678, 0xA5A5A5A5, UINT32_MASK)

    for coordinate in coordinates:
        for word in words:
            encoded = encode_transport_word(coordinate, word)
            assert decode_transport_word(coordinate, encoded) == word


def test_bit_residual_keeps_hamming_and_numeric_error_separate() -> None:
    residual = bit_residual(0x000000B6, 0x000000A6)

    assert residual.xor_mask == 0x10
    assert residual.hamming_distance == 1
    assert residual.numeric_delta == -16


def test_branchless_commit_preserves_baseline_on_rejection() -> None:
    before = 0x11223344
    candidate = 0xDEADBEEF

    assert branchless_commit(before, candidate, False) == before
    assert branchless_commit(before, candidate, True) == candidate


def test_transition_receipt_binds_mapping_routing_and_authority() -> None:
    coordinate = (500, 500, 500)
    receipt = transition(
        coordinate,
        0xAAAAAAAA,
        0xAAAAAAAB,
        accepted=False,
    )

    assert receipt.spatial.coordinate == coordinate
    assert receipt.spatial.linear == linear_address(coordinate)
    assert receipt.domain == ProcessingDomain.NPU
    assert receipt.residual.hamming_distance == 1
    assert receipt.committed == receipt.before
    assert not receipt.accepted
