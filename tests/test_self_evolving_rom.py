from jarvisx.self_evolving_rom import (
    Instruction,
    Opcode,
    SelfEvolvingROM,
    SpatialCore,
    run_self_evolving_rom,
)


def test_instruction_round_trip_is_exact():
    instruction = Instruction(Opcode.LDC, rs=1, rt=2, ru=3, imm=0x1234, addr=0xBEEF)
    assert Instruction.unpack(instruction.pack()) == instruction


def test_load_encode_fusion_preserves_machine_state():
    rom = (
        Instruction(Opcode.LOAD3D, rs=0, addr=0x1000),
        Instruction(Opcode.ENC, rs=0, rt=2),
        Instruction(Opcode.HALT),
    )
    engine = SelfEvolvingROM(rom)
    baseline = SpatialCore().run(rom)
    report = engine.inward_turn()
    candidate = SpatialCore().run(engine.rom)

    assert report.accepted
    assert report.rule == "FUSE_LOAD3D_ENC_TO_LDC"
    assert report.semantic_equal
    assert candidate.snapshot == baseline.snapshot
    assert len(engine.rom) == 2
    assert Opcode(engine.rom[0].opcode) == Opcode.LDC


def test_decode_store_fusion_preserves_machine_state():
    rom = (
        Instruction(Opcode.LOAD3D, rs=0, addr=0x1000),
        Instruction(Opcode.ENC, rs=0, rt=2),
        Instruction(Opcode.DEC, rs=2, rt=3),
        Instruction(Opcode.STORE3D, rt=3, addr=0x4000),
        Instruction(Opcode.HALT),
    )
    engine = SelfEvolvingROM(rom)
    engine.inward_turn()  # First remove LOAD3D + ENC.
    baseline = SpatialCore().run(engine.rom)
    report = engine.inward_turn()
    candidate = SpatialCore().run(engine.rom)

    assert report.accepted
    assert report.rule == "FUSE_DEC_STORE3D_TO_DSM"
    assert candidate.snapshot == baseline.snapshot
    assert any(Opcode(i.opcode) == Opcode.DSM for i in engine.rom)


def test_non_adjacent_operations_are_not_illegally_fused():
    rom = (
        Instruction(Opcode.LOAD3D, rs=0, addr=0x1000),
        Instruction(Opcode.LOAD3D, rs=1, addr=0x2000),
        Instruction(Opcode.ENC, rs=0, rt=2),
        Instruction(Opcode.HALT),
    )
    engine = SelfEvolvingROM(rom)
    report = engine.inward_turn()

    assert not report.accepted
    assert report.rule == "FIXED_POINT"
    assert len(engine.rom) == 4


def test_demo_rom_reaches_rule_set_fixed_point_at_six_instructions():
    result = run_self_evolving_rom(max_epochs=8)

    assert result["locked"] is True
    assert result["rom_length"] == 6
    assert [epoch["rule"] for epoch in result["epochs"]] == [
        "FUSE_LOAD3D_ENC_TO_LDC",
        "FUSE_DEC_STORE3D_TO_DSM",
        "FIXED_POINT",
    ]
    assert len(result["versions"]) == 3
    assert all(epoch["semantic_equal"] for epoch in result["epochs"])


def test_version_chain_is_parent_linked_and_manifested():
    result = run_self_evolving_rom(max_epochs=8)
    versions = result["versions"]

    assert versions[0]["parent_hash"] == "0" * 64
    assert versions[1]["parent_hash"] == versions[0]["manifest_hash"]
    assert versions[2]["parent_hash"] == versions[1]["manifest_hash"]
    assert len({version["manifest_hash"] for version in versions}) == len(versions)


def test_analysis_budget_never_exceeds_half_of_measured_cycles():
    result = run_self_evolving_rom(max_epochs=8)
    accepted = [epoch for epoch in result["epochs"] if epoch["accepted"]]

    assert accepted
    assert all(epoch["analysis_share"] <= 0.5 for epoch in accepted)
