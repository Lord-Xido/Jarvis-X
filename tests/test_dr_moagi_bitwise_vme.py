from jarvisx.dr_moagi_bitwise_vme import (
    BitwiseVME3D,
    BitwiseVMEConfig,
    Instruction64,
    LOGICAL_CELLS,
    Modality,
    BitOpcode,
    VoxelWord,
    dense_capacity_for_one_gib,
    morton3d,
    unmorton3d,
)


def test_voxel_word_round_trip():
    word = VoxelWord(
        x=3999,
        y=17,
        z=2048,
        value=15,
        modality=Modality.VIDEO,
        residual=200,
        flags=1234,
    )
    assert VoxelWord.unpack(word.pack()) == word


def test_instruction_round_trip():
    ins = Instruction64(BitOpcode.RECURRENT, dst=1, src_a=2, src_b=3, immediate=99)
    assert Instruction64.unpack(ins.pack()) == ins


def test_morton_round_trip():
    coord = (3999, 1024, 7)
    assert unmorton3d(morton3d(*coord)) == coord


def test_one_gib_dense_capacity():
    assert dense_capacity_for_one_gib(1) == 1 << 33
    assert dense_capacity_for_one_gib(4) == 1 << 31
    assert dense_capacity_for_one_gib(8) == 1 << 30


def test_sparse_cycle_is_bounded():
    vm = BitwiseVME3D(
        BitwiseVMEConfig(
            latent_dim=8,
            max_refine_steps=16,
            fixed_point_l1_tolerance=2,
        )
    )
    vm.load({
        (0, 0, 0): 15,
        (1, 0, 0): 12,
        (3999, 0, 0): 8,
        (10, 10, 10): 5,
    })
    receipt = vm.run_cycle()

    assert receipt.logical_cells == LOGICAL_CELLS
    assert receipt.physical_refine_steps <= 16
    assert receipt.physical_bytes == receipt.active_cells * 8
    assert all(-128 <= value <= 127 for value in receipt.latent)
