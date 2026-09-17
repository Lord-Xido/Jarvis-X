import hashlib

from jarvisx.dm3d_rom import LATENT_BYTES, TILE_BYTES, deterministic_tile
from jarvisx.dr_moagi_1b_intelligence import (
    BLOCK_WEIGHTS,
    LAYOUT,
    ONE_BILLION,
    WEIGHT_PARTITIONS,
    DrMoagi1B3DIntelligenceEngine,
    EngineConfig,
    VirtualInt8Weights,
    smoke,
)


def test_exact_one_billion_weight_layout_is_contiguous() -> None:
    assert sum(WEIGHT_PARTITIONS.values()) == ONE_BILLION
    offset = 0
    for name in WEIGHT_PARTITIONS:
        layout = LAYOUT[name]
        assert layout.offset == offset
        assert layout.count == WEIGHT_PARTITIONS[name]
        assert layout.end == layout.offset + layout.count
        offset = layout.end
    assert offset == 1_000_000_000


def test_virtual_weight_pages_are_deterministic_without_dense_allocation() -> None:
    a = VirtualInt8Weights(seed=17)
    b = VirtualInt8Weights(seed=17)
    block_a = a.read_block("shared_fusion_core", 123)
    block_b = b.read_block("shared_fusion_core", 123)

    assert block_a == block_b
    assert len(block_a) == BLOCK_WEIGHTS
    assert all(-16 <= value <= 15 for value in block_a)
    assert a.manifest()["logical_weights"] == ONE_BILLION


def test_multimodal_forward_and_decode_are_deterministic() -> None:
    config = EngineConfig(latent_edge=4, active_experts=2, refine_steps=2, seed=23)
    inputs = {
        "text": "residual autoencoding",
        "image": bytes(range(128)),
        "audio": bytes((i * 9) & 255 for i in range(512)),
        "code": b"r = x - decode(encode(x))",
    }
    first = DrMoagi1B3DIntelligenceEngine(config)
    second = DrMoagi1B3DIntelligenceEngine(config)

    s1 = first.forward(inputs)
    s2 = second.forward(inputs)
    o1 = first.decode_bytes(s1, "text", 128)
    o2 = second.decode_bytes(s2, "text", 128)

    assert s1.refined == s2.refined
    assert o1 == o2
    assert len(o1) == 128
    assert s1.active_weight_blocks == (len(inputs) + 2 + config.refine_steps) * 2
    assert hashlib.sha256(o1).digest() == hashlib.sha256(o2).digest()


def test_dm3d_tile_path_remains_byte_exact_after_residual_correction() -> None:
    engine = DrMoagi1B3DIntelligenceEngine(
        EngineConfig(latent_edge=4, active_experts=1, refine_steps=1, seed=7)
    )
    source = deterministic_tile(0, 0, 0, 1)
    result = engine.process_tile(source, quant_step=8)

    assert len(source) == TILE_BYTES
    assert result.latent_bytes == LATENT_BYTES
    assert result.pre_correction_mse > 0.0
    assert result.exact
    assert result.source_sha256 == result.corrected_sha256


def test_evolution_is_validation_gated_and_cannot_worsen_committed_loss() -> None:
    engine = DrMoagi1B3DIntelligenceEngine(
        EngineConfig(latent_edge=4, active_experts=1, refine_steps=1, seed=11)
    )
    result = engine.evolve_adapter("text", b"abc", b"abd", max_trials=8)

    assert result.after_loss <= result.before_loss
    assert result.bank == "text_adapter"


def test_reference_smoke_path_passes() -> None:
    report = smoke()
    assert report["logical_weights"] == ONE_BILLION
    assert report["status"] == "PASS"
    assert report["active_weight_values_upper_bound"] < ONE_BILLION
