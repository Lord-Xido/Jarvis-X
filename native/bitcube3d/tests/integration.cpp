#include "bitcube3d/program.hpp"
#include <iostream>
#include <sstream>
#include <stdexcept>
using namespace bitcube3d;
static void require(bool yes, const char* why) {
    if (!yes) throw std::runtime_error(why);
}
static bool rejects(Program& p, const std::string& command) {
    try { p.run_line(command); } catch (const std::exception&) { return true; }
    return false;
}
int main() {
    try {
        Engine cube;
        cube.set(0, 0, 0, true);
        cube.set(15, 15, 15, true);
        cube.set(16, 16, 16, true);
        cube.set(999999999, 999999999, 999999999, true);
        const Engine original = cube;
        cube.forward(1000);
        cube.inverse(1000);
        require(cube == original, "1000-round identity failed");
        std::ostringstream log;
        Program prog(log, 16);
        prog.run_line("SET 0 0 0 1");
        prog.run_line("SET 999999999 999999999 999999999 1");
        prog.run_line("VERIFY 40");
        prog.run_line("ENCODE 8");
        require(rejects(prog,"SET 3 3 3 1"), "encoded-state mutation accepted");
        require(rejects(prog,"DECODE 9"), "excessive decoding accepted");
        prog.run_line("DECODE");
        prog.run_line("ASSERT 0 0 0 1");
        require(rejects(prog,"SET 1000000000 0 0 1"), "out-of-range accepted");
        require(rejects(prog,"SET -1 0 0 1"), "negative coordinate accepted");
        require(rejects(prog,"SPHERE 0 0 0 33"), "huge sphere accepted");
        std::istringstream in("SET 0 0 0 1\nVERIFY 3\n");
        std::ostringstream generated;
        generate_cpp(in, generated);
        require(generated.str().find("vm.run_line") != std::string::npos, "codegen failed");
        std::cout << "BitCube3D integration: PASS\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "BitCube3D integration: FAIL: " << e.what() << '\n';
        return 1;
    }
}
