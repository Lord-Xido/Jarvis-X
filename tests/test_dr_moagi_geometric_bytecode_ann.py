from jarvisx.dr_moagi_geometric_bytecode_ann import ANNConfig, GeometricBytecodeANN, Instruction, Op, synthetic_field


def test_instruction_round_trip():
    ins = Instruction(Op.CORRECT, a=1, b=2, c=3, immediate=123)
    assert Instruction.unpack(ins.pack()) == ins


def test_bounded_cycle():
    cfg = ANNConfig(side=3, channels=1, latent_dim=3, max_refine_steps=8)
    vm = GeometricBytecodeANN(cfg)
    vm.load(synthetic_field(cfg, 0.25))
    receipt = vm.run_cycle()
    assert 1 <= receipt.physical_refine_steps <= cfg.max_refine_steps
    assert len(vm.geometry) == cfg.side ** 3
