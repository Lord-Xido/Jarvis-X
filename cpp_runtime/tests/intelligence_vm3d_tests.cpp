#include "jarvisx/intelligence_vm3d.hpp"

#include <cassert>
#include <filesystem>
#include <iostream>
#include <string>

using namespace jarvisx::intelligence3d;

namespace {
std::filesystem::path fresh_dir(const std::string& name) {
    auto path = std::filesystem::temp_directory_path() / ("jarvisx-" + name);
    std::error_code ec;
    std::filesystem::remove_all(path, ec);
    std::filesystem::create_directories(path);
    return path;
}

void test_sparse_virtual_capacity() {
    const auto dir = fresh_dir("virtual-capacity");
    {
        VirtualVolume3D volume({kDefaultAxisExtent, 8U, 4096U, dir});
        const Coord3 far{kDefaultAxisExtent - 1U, kDefaultAxisExtent - 2U, kDefaultAxisExtent - 3U};
        assert(volume.read(far) == 0U);
        volume.write(far, 231U);
        assert(volume.read(far) == 231U);
        assert(volume.resident_bytes() <= volume.resident_limit_bytes());
        volume.flush();
    }
    std::filesystem::remove_all(dir);
}

void test_eviction_round_trip() {
    const auto dir = fresh_dir("eviction");
    {
        VirtualVolume3D volume({1024U, 4U, 64U, dir});
        const Coord3 a{1U, 1U, 1U};
        const Coord3 b{9U, 1U, 1U};
        volume.write(a, 17U);
        volume.write(b, 33U);
        assert(volume.stats().evictions >= 1U);
        assert(volume.read(a) == 17U);
        assert(volume.stats().disk_loads >= 1U);
        volume.flush();
    }
    std::filesystem::remove_all(dir);
}

void test_octree_mask() {
    OctreeMask3D mask(1024U, 1U);
    assert(mask.evaluate({1U, 1U, 1U}) == 1.0);
    assert(mask.evaluate({900U, 900U, 1U}) == 0.0);
    assert(mask.evaluate({900U, 1U, 900U}) == 0.0);
}

void test_bytecode_vm_store_load() {
    const auto dir = fresh_dir("vm");
    {
        VirtualVolume3D volume({4096U, 8U, 4096U, dir});
        PsiIntelligenceCore psi(volume, {8U, 2U, 0.01, 0.9});
        IntelligenceVm3D vm(volume, psi);
        BytecodeProgram program;
        program.instructions = {
            {Opcode::MovImm, 0U, 0U, 0U, 7U},
            {Opcode::MovImm, 1U, 0U, 0U, 8U},
            {Opcode::MovImm, 2U, 0U, 0U, 9U},
            {Opcode::MovImm, 3U, 0U, 0U, 201U},
            {Opcode::StoreVoxel, 3U, 0U, 0U, 0U},
            {Opcode::LoadVoxel, 4U, 0U, 0U, 0U},
            {Opcode::Halt, 0U, 0U, 0U, 0U},
        };
        vm.run(program, 1U, 100U);
        assert(vm.registers()[4] == 201U);
        assert(vm.stats().steps == 7U);
        volume.flush();
    }
    std::filesystem::remove_all(dir);
}

void test_bytecode_round_trip_and_psi() {
    const auto dir = fresh_dir("bytecode");
    const auto file = dir / "demo.jxb3";
    auto program = make_demo_program(1U, 1U, 1U, 99U);
    program.save(file);
    auto loaded = BytecodeProgram::load(file);
    assert(loaded.instructions.size() == program.instructions.size());

    {
        VirtualVolume3D volume({4096U, 8U, 4096U, dir / "pages"});
        PsiIntelligenceCore psi(volume, {8U, 2U, 0.02, 0.8});
        IntelligenceVm3D vm(volume, psi);
        vm.run(loaded, 3U, 100U);
        assert(vm.stats().cycles == 3U);
        assert(vm.stats().psi_learning_steps == 3U);
        assert(vm.stats().psi_inferences == 6U);
        assert(psi.refinement_steps() == 3U);
        assert(std::isfinite(psi.best_abs_error()));
        volume.flush();
    }
    std::filesystem::remove_all(dir);
}
}  // namespace

int main() {
    test_sparse_virtual_capacity();
    test_eviction_round_trip();
    test_octree_mask();
    test_bytecode_vm_store_load();
    test_bytecode_round_trip_and_psi();
    std::cout << "intelligence_vm3d_tests: OK\n";
    return 0;
}
